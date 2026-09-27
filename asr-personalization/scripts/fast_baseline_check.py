"""Fast, standalone zero-shot Whisper baseline check on a random TORGO subset.

This is NOT the full evaluation pipeline (scripts/run_baseline_eval.py) — it's
a quick, real-data sanity check: sample up to --n utterances at random from
pooled_test.csv (whatever audio actually exists on disk right now), run
zero-shot Whisper over them, score with the project's existing normalization
(src.data.preprocessing.normalize_transcription) + jiwer, and report S/D/I
counts alongside WER/CER. No fine-tuning, no caching/resuming, no per-severity
breakdown — just one pass, one CSV, real numbers.

Usage:
    python scripts/fast_baseline_check.py --n 100 --model openai/whisper-small
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jiwer
import pandas as pd

from src.data.preprocessing import normalize_transcription
from src.inference.whisper_infer import WhisperRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/splits/pooled_test.csv")
    parser.add_argument("--n", type=int, default=100, help="Max utterances to sample")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default="reports/fast_baseline_check.csv")
    args = parser.parse_args()

    random.seed(args.seed)

    manifest_path = Path(args.manifest)
    df = pd.read_csv(manifest_path)

    exists_mask = df["audio_path"].apply(os.path.exists)
    available = df[exists_mask].reset_index(drop=True)
    print(f"Manifest {manifest_path}: {len(df)} rows total, {len(available)} with audio present on disk.")

    n = min(args.n, len(available))
    sample = available.sample(n=n, random_state=args.seed).reset_index(drop=True)

    status_counts = sample["dysarthria_status"].value_counts().to_dict()
    print(f"Sampled N={n} utterances. Status breakdown in sample: {status_counts}")

    print(f"Loading {args.model} ...")
    t_load0 = time.time()
    runner = WhisperRunner(model_name=args.model, language="en", task="transcribe", max_new_tokens=64)
    print(f"Model loaded on {runner.device} in {time.time() - t_load0:.1f}s")

    t0 = time.time()
    hyps: list[str] = []
    for start in range(0, n, args.batch_size):
        batch_paths = sample["audio_path"].iloc[start : start + args.batch_size].tolist()
        batch_hyps = runner.transcribe_batch(batch_paths)
        hyps.extend(batch_hyps)
        print(f"  transcribed {min(start + args.batch_size, n)}/{n}", flush=True)
    elapsed = time.time() - t0

    sample["hypothesis"] = hyps
    sample["reference_norm"] = sample["transcription"].astype(str).apply(normalize_transcription)
    sample["hypothesis_norm"] = sample["hypothesis"].astype(str).apply(normalize_transcription)
    # jiwer chokes on empty strings on both sides of a pair (same guard as
    # src/evaluation/metrics.py) — substitute a placeholder so it counts as a
    # full miss rather than crashing the whole batch.
    refs_scored = sample["reference_norm"].apply(lambda s: s if s else "<empty>").tolist()
    hyps_scored = sample["hypothesis_norm"].apply(lambda s: s if s else "<empty>").tolist()

    def wer_cer_sdi(refs: list[str], hyps: list[str]) -> dict:
        if not refs:
            return {"n": 0, "wer": float("nan"), "cer": float("nan"), "s": 0, "d": 0, "i": 0}
        out = jiwer.process_words(refs, hyps)
        cer = jiwer.cer(refs, hyps)
        return {
            "n": len(refs),
            "wer": round(out.wer, 4),
            "cer": round(cer, 4),
            "s": out.substitutions,
            "d": out.deletions,
            "i": out.insertions,
        }

    overall = wer_cer_sdi(refs_scored, hyps_scored)

    by_status = {}
    for status in sample["dysarthria_status"].unique():
        idx = sample["dysarthria_status"] == status
        r = [refs_scored[i] for i in range(n) if idx.iloc[i]]
        h = [hyps_scored[i] for i in range(n) if idx.iloc[i]]
        by_status[status] = wer_cer_sdi(r, h)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sample[[
        "speaker_id", "audio_path", "dysarthria_status", "severity",
        "transcription", "hypothesis", "reference_norm", "hypothesis_norm",
    ]].to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print(f"N evaluated:        {n}")
    print(f"Whisper checkpoint: {args.model}")
    print(f"Device:             {runner.device}")
    print(f"Inference time:     {elapsed:.1f}s ({elapsed / n:.2f}s/utterance)")
    print("-" * 60)
    print(f"OVERALL   n={overall['n']:>3}  WER={overall['wer']*100:6.2f}%  CER={overall['cer']*100:6.2f}%  "
          f"S={overall['s']} D={overall['d']} I={overall['i']}")
    for status, m in by_status.items():
        print(f"{status.upper():9} n={m['n']:>3}  WER={m['wer']*100:6.2f}%  CER={m['cer']*100:6.2f}%  "
              f"S={m['s']} D={m['d']} I={m['i']}")
    print("=" * 60)
    print(f"\nSaved per-utterance ref/hyp CSV to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
