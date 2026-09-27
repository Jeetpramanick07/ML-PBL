"""Phase 3: evaluate the generic (pooled) LoRA fine-tuned Whisper model.

Same metrics/breakdown as Phase 2's baseline eval (scripts/run_baseline_eval.py),
run on both pooled_test.csv and speaker_holdout_test.csv, but with the Phase 3
LoRA adapter (checkpoints/generic/best by default) applied on top of the base
checkpoint. Also renders a baseline-vs-generic comparison table by reading
reports/baseline_eval.json — the first "does fine-tuning help" result.

Usage:
    python scripts/eval_generic.py
    python scripts/eval_generic.py --adapter-dir checkpoints/generic/last
    python scripts/eval_generic.py --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.report_utils import (
    evaluate_split,
    fmt_metrics_table,
    per_speaker_table,
    render_markdown,
    summarize_split,
)
from src.inference.whisper_infer import WhisperRunner
from src.utils.config import load_config
from src.utils.seed import set_seed


def render_comparison(baseline_results: dict, generic_results: dict) -> str:
    lines = ["# Baseline vs. generic fine-tuned model\n"]
    for split_name in generic_results:
        if split_name not in baseline_results:
            continue
        lines.append(f"## {split_name}\n")
        lines.append("| arm | group | n | WER | CER |")
        lines.append("|---|---|---|---|---|")
        for label, results in (("baseline (zero-shot)", baseline_results), ("generic LoRA", generic_results)):
            overall = results[split_name]["overall"]
            lines.append(f"| {label} | overall | {overall['n_samples']} | {overall['wer']:.4f} | {overall['cer']:.4f} |")
            dys = results[split_name]["by_dysarthria_status"].get("dysarthric")
            if dys:
                lines.append(f"| {label} | dysarthric-only | {dys['n_samples']} | {dys['wer']:.4f} | {dys['cer']:.4f} |")
        lines.append("")

        base_dys = baseline_results[split_name]["by_dysarthria_status"].get("dysarthric")
        gen_dys = generic_results[split_name]["by_dysarthria_status"].get("dysarthric")
        if base_dys and gen_dys and base_dys["wer"] > 0:
            rel_change = (gen_dys["wer"] - base_dys["wer"]) / base_dys["wer"] * 100
            direction = "reduction" if rel_change < 0 else "increase"
            lines.append(f"Dysarthric-only WER {direction}: {abs(rel_change):.1f}% relative "
                         f"({base_dys['wer']:.4f} -> {gen_dys['wer']:.4f}).\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--adapter-dir", default="checkpoints/generic/best")
    parser.add_argument("--tag", default="generic", help="Prefix for output files (e.g. generic_unseen), so runs do not overwrite each other")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows per split (smoke test)")
    parser.add_argument("--force", action="store_true", help="Ignore cached hypotheses and re-run inference")
    parser.add_argument("--model-name", default=None, help="Override configs/base.yaml model.name")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)

    model_name = args.model_name or cfg.model.name
    batch_size = args.batch_size or cfg.training.batch_size
    out_dir = Path(args.out_dir)
    splits_dir = Path(args.splits_dir)
    adapter_dir = Path(args.adapter_dir)

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter dir {adapter_dir} not found — run src.training.train_generic first, "
            f"or pass --adapter-dir pointing at a saved checkpoint."
        )

    runner = WhisperRunner(
        model_name=model_name, language=cfg.model.language, task=cfg.model.task,
        max_new_tokens=args.max_new_tokens, adapter_dir=str(adapter_dir),
    )

    split_files = {
        "pooled_test": splits_dir / "pooled_test.csv",
        "speaker_holdout_test": splits_dir / "speaker_holdout_test.csv",
    }

    results = {}
    per_speaker: dict[str, pd.DataFrame] = {}

    for split_name, path in split_files.items():
        df = pd.read_csv(path)
        if args.limit:
            df = df.head(args.limit)

        hyps_path = out_dir / f"{args.tag}_hyps_{split_name}.csv"
        scored = evaluate_split(runner, df, hyps_path, batch_size, args.force, args.checkpoint_every, label=split_name)

        summary = summarize_split(scored)
        results[split_name] = summary

        if split_name == "speaker_holdout_test":
            per_speaker[split_name] = per_speaker_table(scored)

    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{args.tag}_eval.json"
    json_payload = {
        "model_name": model_name,
        "adapter_dir": str(adapter_dir),
        "results": results,
        "per_speaker": {k: v.to_dict(orient="records") for k, v in per_speaker.items()},
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    title = f"Generic pooled LoRA fine-tune evaluation — `{model_name}` + `{adapter_dir}`"
    md_path = out_dir / f"{args.tag}_eval.md"
    md_content = render_markdown(title, results, per_speaker)

    baseline_json_path = out_dir / "baseline_eval.json"
    comparison_md = ""
    if baseline_json_path.exists():
        baseline_payload = json.loads(baseline_json_path.read_text(encoding="utf-8"))
        comparison_md = render_comparison(baseline_payload["results"], results)
        md_content = md_content + "\n\n" + comparison_md
        (out_dir / f"{args.tag}_vs_baseline_comparison.md").write_text(comparison_md, encoding="utf-8")
    else:
        print(f"NOTE: {baseline_json_path} not found — skipping baseline comparison table.")

    md_path.write_text(md_content, encoding="utf-8")

    print(f"\nSaved {json_path} and {md_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
