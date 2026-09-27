"""Phase 2: zero-shot (no fine-tuning) Whisper baseline evaluation on TORGO.

Runs the pretrained Whisper checkpoint from configs/base.yaml over
pooled_test.csv and speaker_holdout_test.csv separately, breaks results down
by overall / healthy-only / dysarthric-only / per-severity, and (for the
speaker-holdout split) per individual held-out speaker. This is the "before"
number every later phase's personalization result is measured against.

Usage:
    python scripts/run_baseline_eval.py
    python scripts/run_baseline_eval.py --limit 20          # smoke test
    python scripts/run_baseline_eval.py --batch-size 16
    python scripts/run_baseline_eval.py --force              # ignore cached hypotheses
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
    per_speaker_table,
    render_markdown,
    summarize_split,
)
from src.inference.whisper_infer import WhisperRunner
from src.utils.config import load_config
from src.utils.seed import set_seed

# Dysarthric-speech WER on generic (non-adapted) ASR is reported in the
# literature (e.g. Rudzicz et al. 2012 on TORGO itself, and follow-on work
# with modern end-to-end models) as roughly 50-80%. Numbers far outside this
# band more likely indicate a normalization/data bug than a genuine result.
EXPECTED_DYSARTHRIC_WER_RANGE = (0.50, 0.80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configs/base.yaml training.batch_size")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows per split (smoke test)")
    parser.add_argument("--force", action="store_true", help="Ignore cached hypotheses and re-run inference")
    parser.add_argument("--model-name", default=None, help="Override configs/base.yaml model.name")
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Cap on generated tokens per utterance. TORGO prompts are short (words/short phrases); "
             "keeping this low avoids a rare non-terminating greedy decode dragging out a whole batch.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Flush transcribed hypotheses to reports/baseline_hyps_{split}.csv every N utterances, "
             "so a crash/OOM-kill mid-split loses at most this many utterances of progress and a "
             "re-run resumes rather than starting over.",
    )
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)

    model_name = args.model_name or cfg.model.name
    batch_size = args.batch_size or cfg.training.batch_size
    out_dir = Path(args.out_dir)
    splits_dir = Path(args.splits_dir)

    runner = WhisperRunner(
        model_name=model_name, language=cfg.model.language, task=cfg.model.task,
        max_new_tokens=args.max_new_tokens,
    )

    split_files = {
        "pooled_test": splits_dir / "pooled_test.csv",
        "speaker_holdout_test": splits_dir / "speaker_holdout_test.csv",
    }

    results = {}
    per_speaker: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    for split_name, path in split_files.items():
        df = pd.read_csv(path)
        if args.limit:
            df = df.head(args.limit)

        hyps_path = out_dir / f"baseline_hyps_{split_name}.csv"
        scored = evaluate_split(runner, df, hyps_path, batch_size, args.force, args.checkpoint_every, label=split_name)

        summary = summarize_split(scored)
        results[split_name] = summary

        if split_name == "speaker_holdout_test":
            per_speaker[split_name] = per_speaker_table(scored)

        dys_wer = summary["by_dysarthria_status"].get("dysarthric", {}).get("wer")
        if dys_wer is not None:
            lo, hi = EXPECTED_DYSARTHRIC_WER_RANGE
            if not (lo <= dys_wer <= hi):
                warnings.append(
                    f"{split_name}: dysarthric WER={dys_wer:.4f} is outside the expected "
                    f"literature range [{lo:.2f}, {hi:.2f}] — check normalization/data before trusting this result."
                )

    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "baseline_eval.json"
    json_payload = {
        "model_name": model_name,
        "results": results,
        "per_speaker": {k: v.to_dict(orient="records") for k, v in per_speaker.items()},
        "warnings": warnings,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    md_path = out_dir / "baseline_eval.md"
    title = f"Baseline (zero-shot) Whisper evaluation — `{model_name}`"
    md_path.write_text(render_markdown(title, results, per_speaker, warnings), encoding="utf-8")

    print(f"\nSaved {json_path} and {md_path}")
    print("\n" + render_markdown(title, results, per_speaker, warnings))

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
