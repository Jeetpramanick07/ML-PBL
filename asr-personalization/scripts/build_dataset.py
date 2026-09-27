"""End-to-end TORGO data pipeline: load -> clean -> split -> summarize.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --raw-dir data/raw/torgo --skip-duration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.preprocessing import DurationFilter, is_valid_clip, load_and_clean_audio, normalize_transcription
from src.data.splits import SpeakerHoldoutConfig, make_pooled_split, make_speaker_holdout_split, save_splits
from src.data.torgo_loader import load_torgo
from src.utils.seed import set_seed


def clean_manifest(df: pd.DataFrame, filt: DurationFilter, check_audio: bool) -> pd.DataFrame:
    """Drop rows with unusable transcriptions and (optionally) verify/measure audio."""
    before = len(df)
    df = df[df.transcription.notna()].copy()
    df = df[df.transcription.str.strip() != ""].copy()
    df["transcription"] = df.transcription.map(normalize_transcription)
    df = df[df.transcription.str.strip() != ""].copy()  # drop prompts that were pure punctuation
    dropped_text = before - len(df)

    if check_audio:
        durations = []
        keep_mask = []
        for path in df.audio_path:
            wav, sr = load_and_clean_audio(path, trim_silence=True)
            ok = is_valid_clip(wav, sr, filt)
            keep_mask.append(ok)
            durations.append(len(wav) / sr if (ok and wav is not None) else None)
        df["duration"] = durations
        before_audio = len(df)
        df = df[keep_mask].copy()
        dropped_audio = before_audio - len(df)
    else:
        dropped_audio = 0

    print(f"  dropped {dropped_text} rows (unusable transcription), "
          f"{dropped_audio} rows (corrupt/too-short/too-long audio)")
    return df.reset_index(drop=True)


def summarize(splits: dict[str, pd.DataFrame], label: str) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        n = len(df)
        n_healthy = int((df.dysarthria_status == "healthy").sum())
        n_dysarthric = int((df.dysarthria_status == "dysarthric").sum())
        avg_dur = df.duration.dropna().mean() if "duration" in df and n else float("nan")
        rows.append({
            "split_set": label, "split": name, "n_samples": n,
            "n_healthy": n_healthy, "n_dysarthric": n_dysarthric,
            "n_speakers": df.speaker_id.nunique() if n else 0,
            "avg_duration_sec": round(avg_dur, 2) if pd.notna(avg_dur) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/torgo")
    parser.add_argument("--out-dir", default="data/processed/splits")
    parser.add_argument("--skip-duration", action="store_true",
                         help="Skip loading every wav file to verify/measure duration (faster, less thorough).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading TORGO manifest from {args.raw_dir} ...")
    df = load_torgo(args.raw_dir, compute_duration=False)
    print(f"  found {len(df)} raw (audio, prompt) rows across {df.speaker_id.nunique() if len(df) else 0} speakers")

    if df.empty:
        print("No data found. Run scripts/download_torgo.py first, or check --raw-dir.", file=sys.stderr)
        sys.exit(1)

    print("Cleaning manifest (transcription normalization, audio validity) ...")
    df = clean_manifest(df, DurationFilter(), check_audio=not args.skip_duration)
    print(f"  {len(df)} rows remain after cleaning")

    print("Building pooled split (stratified by dysarthria_status) ...")
    pooled = make_pooled_split(df, seed=args.seed)
    save_splits(pooled, args.out_dir, "pooled")

    print("Building speaker-holdout split (disjoint speakers per split) ...")
    holdout = make_speaker_holdout_split(df, SpeakerHoldoutConfig(seed=args.seed))
    save_splits(holdout, args.out_dir, "speaker_holdout")

    summary = pd.concat([summarize(pooled, "pooled"), summarize(holdout, "speaker_holdout")], ignore_index=True)

    print("\n=== Dataset summary ===")
    print(f"Total cleaned samples: {len(df)}  "
          f"(healthy={int((df.dysarthria_status == 'healthy').sum())}, "
          f"dysarthric={int((df.dysarthria_status == 'dysarthric').sum())})")
    print(summary.to_string(index=False))

    holdout_test_speakers = sorted(holdout["test"].speaker_id.unique())
    print(f"\nSpeaker-holdout TEST speakers ({len(holdout_test_speakers)}): {holdout_test_speakers}")

    summary_path = Path(args.out_dir).parent / "dataset_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved manifests to {args.out_dir}/")


if __name__ == "__main__":
    main()
