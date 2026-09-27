"""Phase 4: run the full few-shot personalization experiment grid.

For every speaker_holdout TEST speaker (configs/lora_personalize.yaml:
test_speakers) and every enrollment size, trains a per-speaker LoRA adapter
on top of the Phase 3 generic model under both initialization strategies
("naive" and "meta"), evaluates WER/CER on that speaker's fixed held-out
audio, and appends one row per (speaker, size, strategy) combo to
reports/personalization_results.csv.

Resumable at the row level: before training a combo, the script checks
whether a matching row already exists in the results CSV and skips it if so
— a crash or interruption only loses the combo that was in flight. Per-
combo adapters are also saved to
checkpoints/personalized/<speaker>/<strategy>/<minutes>min/ for audit/reuse.

Usage:
    python scripts/run_personalization_experiment.py
    python scripts/run_personalization_experiment.py --smoke-test   # tiny/fast sanity run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from transformers import WhisperProcessor

from src.data.enrollment import load_or_build_enrollment
from src.data.preprocessing import normalize_transcription
from src.evaluation.metrics import compute_wer_cer
from src.models.whisper_lora import WhisperLoRAConfig
from src.training.meta_init import load_meta_state, run_reptile_meta_init, save_meta_state
from src.training.personalize import (
    TrainingConfig,
    build_personalized_base,
    evaluate_adapter,
    new_adapter,
    train_adapter,
)
from src.utils.config import load_experiment_config
from src.utils.seed import set_seed

CONFIG_PATH = PROJECT_ROOT / "configs" / "lora_personalize.yaml"
DEFAULT_STRATEGIES = ["naive", "meta"]


def already_done(results_path: Path, speaker_id: str, minutes: float, strategy: str) -> bool:
    if not results_path.exists():
        return False
    df = pd.read_csv(results_path)
    if df.empty:
        return False
    match = (
        (df.speaker_id == speaker_id) & (df.enrollment_minutes == minutes) & (df.strategy == strategy)
    )
    return bool(match.any())


def append_result(results_path: Path, row: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(results_path, mode="a", header=not results_path.exists(), index=False)


def unseen_text_metrics(metrics: dict, enroll_df: pd.DataFrame) -> dict:
    """WER restricted to eval utterances whose (normalized) transcript does NOT
    occur in the enrollment set. TORGO repeats prompts within a speaker, so
    plain WER can partly reflect memorized prompt text; this isolates
    generalization to text the adapter never trained on."""
    seen = {normalize_transcription(str(t)) for t in enroll_df["transcription"]}
    keep = [i for i, r in enumerate(metrics["references"]) if normalize_transcription(str(r)) not in seen]
    if not keep:
        return {"n_unseen_text": 0, "wer_unseen_text": float("nan")}
    m = compute_wer_cer([metrics["references"][i] for i in keep], [metrics["hypotheses"][i] for i in keep])
    return {"n_unseen_text": len(keep), "wer_unseen_text": m.wer}


def save_hypotheses(hyps_dir: Path, speaker_id: str, strategy: str, minutes: float, metrics: dict) -> None:
    hyps_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"reference": metrics["references"], "hypothesis": metrics["hypotheses"]}).to_csv(
        hyps_dir / f"{speaker_id}_{strategy}_{minutes}min.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Tiny/fast sanity run: 1 speaker, 1 enrollment size, 1 epoch, few meta iterations")
    parser.add_argument("--speakers", nargs="*", default=None, help="Override test_speakers from config")
    parser.add_argument("--minutes", nargs="*", type=float, default=None, help="Override enrollment.minutes")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override personalize_training.num_epochs")
    parser.add_argument("--outer-iterations", type=int, default=None, help="Override meta_init.outer_iterations")
    parser.add_argument("--force-meta-init", action="store_true", help="Recompute meta init even if cached")
    parser.add_argument("--limit-eval", type=int, default=None, help="Cap eval_holdout rows per speaker (smoke test)")
    parser.add_argument("--strategies", nargs="*", default=None, choices=["naive", "meta"],
                         help="Subset of DEFAULT_STRATEGIES to run. Passing just 'naive' skips the Reptile "
                              "meta-init step entirely (it's only needed for the 'meta' arm), saving real time.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Experiment config (merged onto configs/base.yaml)")
    parser.add_argument("--generic-adapter-dir", default=None, help="Override generic_adapter_dir from config")
    args = parser.parse_args()

    strategies = args.strategies or DEFAULT_STRATEGIES
    cfg = load_experiment_config(args.config)
    set_seed(cfg.seed)

    device = "cpu"
    import torch
    if torch.cuda.is_available():
        device = "cuda"

    splits_dir = Path(args.splits_dir)
    test_df = pd.read_csv(splits_dir / "speaker_holdout_test.csv")
    meta_train_df = pd.read_csv(splits_dir / "speaker_holdout_train.csv")

    test_speakers = args.speakers or list(cfg.test_speakers)
    minutes = args.minutes or list(cfg.enrollment.minutes)
    num_epochs = args.num_epochs or cfg.personalize_training.num_epochs
    outer_iterations = args.outer_iterations if args.outer_iterations is not None else cfg.meta_init.outer_iterations

    if args.smoke_test:
        test_speakers = test_speakers[:1]
        minutes = minutes[:1]
        num_epochs = 1
        outer_iterations = 4

    enrollment_out_dir = Path(cfg.enrollment.out_dir)
    results_path = Path(cfg.results_path)
    checkpoints_dir = Path(cfg.checkpoints_dir)
    if args.smoke_test:
        # Keep smoke runs away from real results/caches: otherwise their rows
        # would make the real run skip combos, and their 1-size enrollment
        # cache would be reused by it.
        enrollment_out_dir = Path("data/processed/enrollment_smoke")
        results_path = Path("reports/smoke/personalization_results.csv")
        checkpoints_dir = Path("checkpoints/personalized_smoke")
    # Configurable so a new run (e.g. lora_personalize_final.yaml) doesn't
    # overwrite an earlier run's same-named per-speaker hypothesis files.
    hyps_dir = Path(cfg.get("hyps_dir") or results_path.parent / "personalization_hyps")
    if args.smoke_test:
        hyps_dir = results_path.parent / "personalization_hyps"

    print(f"device={device} test_speakers={test_speakers} minutes={minutes} "
          f"num_epochs={num_epochs} outer_iterations={outer_iterations}")

    generic_adapter_dir = args.generic_adapter_dir or cfg.generic_adapter_dir
    processor = WhisperProcessor.from_pretrained(cfg.model.name, language=cfg.model.language, task=cfg.model.task)
    print(f"Building personalized base (Whisper + generic adapter from {generic_adapter_dir} merged) ...")
    merged_base = build_personalized_base(cfg.model.name, generic_adapter_dir, device)

    lora_cfg = WhisperLoRAConfig(
        r=cfg.lora.r, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, target_modules=list(cfg.lora.target_modules)
    )
    training_cfg = TrainingConfig(
        batch_size=cfg.personalize_training.batch_size,
        num_epochs=num_epochs,
        learning_rate=cfg.personalize_training.learning_rate,
        warmup_steps=cfg.personalize_training.warmup_steps,
        mixed_precision=(device == "cuda"),
    )

    # Meta-init is only needed for the "meta" strategy arm — skip the whole
    # (non-trivial) Reptile computation entirely when it's not in scope, e.g.
    # a naive-only run under a tight time budget.
    meta_state = None
    if "meta" in strategies:
        meta_ckpt_path = Path(cfg.meta_init.checkpoint_path)
        meta_state = None if args.smoke_test else load_meta_state(meta_ckpt_path)
        if meta_state is None or args.force_meta_init:
            print(f"Running Reptile meta-init ({outer_iterations} outer iterations over "
                  f"{list(cfg.meta_init.speakers)}) ...")
            meta_state = run_reptile_meta_init(
                merged_base, processor, lora_cfg, meta_train_df,
                speakers=list(cfg.meta_init.speakers),
                enrollment_out_dir=enrollment_out_dir,
                enrollment_minutes=minutes,
                outer_iterations=outer_iterations,
                inner_steps=cfg.meta_init.inner_steps,
                inner_lr=cfg.meta_init.inner_lr,
                meta_lr=cfg.meta_init.meta_lr,
                seed=cfg.meta_init.seed,
                device=device,
            )
            if not args.smoke_test:
                save_meta_state(meta_state, meta_ckpt_path)
                print(f"Saved meta-init weights to {meta_ckpt_path}")
        else:
            print(f"Loaded cached meta-init weights from {meta_ckpt_path}")

    for speaker_id in test_speakers:
        print(f"\n=== speaker {speaker_id} ===")
        splits = load_or_build_enrollment(
            test_df, speaker_id, enrollment_out_dir, minutes, cfg.enrollment.seed,
            eval_max_samples=cfg.enrollment.get("eval_max_samples"),
        )
        eval_df = splits["eval_holdout"]
        if args.limit_eval:
            eval_df = eval_df.head(args.limit_eval)
        if len(eval_df) == 0:
            print(f"  WARNING: no eval_holdout audio for {speaker_id}; skipping")
            continue

        # 0-enrollment reference: the frozen Phase 3 generic-unseen model on this
        # exact eval set, so the adaptation curve compares like with like.
        if not already_done(results_path, speaker_id, 0, "generic"):
            g = evaluate_adapter(merged_base, processor, eval_df, device, batch_size=training_cfg.batch_size)
            print(f"  [{speaker_id} generic 0min] n={g['n_samples']} WER={g['wer']:.4f} CER={g['cer']:.4f}")
            save_hypotheses(hyps_dir, speaker_id, "generic", 0, g)
            append_result(results_path, {
                "speaker_id": speaker_id, "enrollment_minutes": 0, "strategy": "generic",
                "n_samples": g["n_samples"], "wer": g["wer"], "cer": g["cer"],
                "n_unseen_text": "", "wer_unseen_text": "",
            })

        for m in minutes:
            enroll_key = f"enroll_{m}min"
            if enroll_key not in splits:
                print(f"  skipping {m}min: not enough audio for this speaker")
                continue
            enroll_df = splits[enroll_key]

            for strategy in strategies:
                if already_done(results_path, speaker_id, m, strategy):
                    print(f"  [{speaker_id} {m}min {strategy}] already in {results_path}, skipping")
                    continue

                print(f"  [{speaker_id} {m}min {strategy}] training on {len(enroll_df)} utterances ...")
                set_seed(cfg.seed)  # same random init + data order for a given combo, resumed or not
                model = new_adapter(merged_base, lora_cfg, device)
                init_state = meta_state if strategy == "meta" else None
                train_adapter(model, processor, enroll_df, training_cfg, device, init_state_dict=init_state)

                metrics = evaluate_adapter(model, processor, eval_df, device, batch_size=training_cfg.batch_size)
                print(f"  [{speaker_id} {m}min {strategy}] n={metrics['n_samples']} "
                      f"WER={metrics['wer']:.4f} CER={metrics['cer']:.4f}")

                adapter_dir = checkpoints_dir / speaker_id / strategy / f"{m}min"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(adapter_dir)

                unseen = unseen_text_metrics(metrics, enroll_df)
                print(f"  [{speaker_id} {m}min {strategy}] unseen-text subset n={unseen['n_unseen_text']} "
                      f"WER={unseen['wer_unseen_text']:.4f}")
                save_hypotheses(hyps_dir, speaker_id, strategy, m, metrics)
                append_result(results_path, {
                    "speaker_id": speaker_id,
                    "enrollment_minutes": m,
                    "strategy": strategy,
                    "n_samples": metrics["n_samples"],
                    "wer": metrics["wer"],
                    "cer": metrics["cer"],
                    **unseen,
                })
                del model

    print(f"\nDone. Results in {results_path}")


if __name__ == "__main__":
    main()
