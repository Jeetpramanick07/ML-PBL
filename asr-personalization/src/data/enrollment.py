"""Phase 4: deterministic enrollment-subset sampling for few-shot personalization.

Given a held-out speaker's full manifest (all of their TORGO utterances),
sample nested enrollment subsets of increasing target duration (e.g. 1, 2, 5,
10 minutes), and hold out everything beyond the largest requested size as
that speaker's fixed evaluation set. Subsets are nested — the 1-minute set is
a strict prefix of the 2-minute set, and so on — and sampled with a fixed
seed, so:

  - both initialization strategies (naive vs. meta) train on the exact same
    enrollment audio at each size, making the comparison fair;
  - the evaluation set is identical across enrollment sizes for one speaker,
    so WER at different enrollment sizes is directly comparable;
  - re-running the pipeline reproduces byte-identical enrollment/eval splits.

The exact sampled file lists are cached to disk under
`data/processed/enrollment/<speaker_id>/` so results are auditable and
re-running an experiment doesn't reshuffle a speaker's enrollment audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ENROLLMENT_MINUTES = [1, 2, 5, 10]


def _shuffled_speaker_df(df: pd.DataFrame, speaker_id: str, seed: int) -> pd.DataFrame:
    speaker_df = df[df.speaker_id == speaker_id].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(speaker_df))
    return speaker_df.iloc[order].reset_index(drop=True)


def build_enrollment_splits(
    df: pd.DataFrame,
    speaker_id: str,
    minutes: list[float] = DEFAULT_ENROLLMENT_MINUTES,
    seed: int = 42,
    eval_max_samples: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Returns `{"enroll_{m}min": subset, ..., "eval_holdout": remainder}`.

    Enrollment subsets are nested prefixes of one fixed per-speaker shuffle.
    `eval_holdout` is everything not in the *largest* enrollment subset, so
    it stays fixed across enrollment sizes for a given speaker.

    If a speaker doesn't have roughly enough total audio to approach a
    requested size, that size is skipped (with a warning) rather than
    silently padded or duplicated.

    `eval_max_samples` caps the size of `eval_holdout` (still a deterministic
    prefix of the same fixed shuffle). This matters because a speaker's
    remainder can be hundreds of utterances — evaluating that many at every
    one of several enrollment sizes/strategies multiplies inference cost
    considerably on top of training cost, so capping it keeps a still-solid
    WER estimate without that blowup.
    """
    shuffled = _shuffled_speaker_df(df, speaker_id, seed)
    cum_duration = shuffled["duration"].cumsum()
    total_minutes = cum_duration.iloc[-1] / 60 if len(shuffled) else 0.0

    out: dict[str, pd.DataFrame] = {}
    max_reached_idx = 0
    for m in sorted(minutes):
        target_seconds = m * 60
        idx = int((cum_duration <= target_seconds).sum())
        idx = max(idx, 1) if len(shuffled) else 0
        if idx == 0 or cum_duration.iloc[idx - 1] < target_seconds * 0.5:
            print(f"WARNING: speaker {speaker_id} has only {total_minutes:.1f} min total audio; "
                  f"skipping {m}min enrollment size")
            continue
        out[f"enroll_{m}min"] = shuffled.iloc[:idx].reset_index(drop=True)
        max_reached_idx = max(max_reached_idx, idx)

    eval_holdout = shuffled.iloc[max_reached_idx:].reset_index(drop=True)
    if eval_max_samples is not None:
        eval_holdout = eval_holdout.iloc[:eval_max_samples].reset_index(drop=True)
    out["eval_holdout"] = eval_holdout
    return out


def save_enrollment_splits(splits: dict[str, pd.DataFrame], speaker_id: str, out_dir: Path) -> dict[str, str]:
    speaker_dir = out_dir / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, sub_df in splits.items():
        path = speaker_dir / f"{name}.csv"
        sub_df.to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def load_or_build_enrollment(
    df: pd.DataFrame,
    speaker_id: str,
    out_dir: Path,
    minutes: list[float] = DEFAULT_ENROLLMENT_MINUTES,
    seed: int = 42,
    force: bool = False,
    eval_max_samples: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Reuse cached enrollment CSVs for `speaker_id` if present, otherwise
    build and cache them. Cached splits are keyed only by speaker_id (not by
    the `minutes`/`seed`/`eval_max_samples` args), so changing those args for
    an already-cached speaker requires `force=True` to regenerate."""
    speaker_dir = out_dir / speaker_id
    expected_names = [f"enroll_{m}min" for m in minutes] + ["eval_holdout"]

    if not force and speaker_dir.exists():
        cached = {
            name: pd.read_csv(speaker_dir / f"{name}.csv")
            for name in expected_names
            if (speaker_dir / f"{name}.csv").exists()
        }
        # Only trust a COMPLETE cache. A partial one (e.g. from a run with a
        # different `minutes` list) would silently drop enrollment sizes and
        # carry an eval_holdout sized for a different largest enrollment set.
        if len(cached) == len(expected_names):
            return cached

    splits = build_enrollment_splits(df, speaker_id, minutes, seed, eval_max_samples)
    save_enrollment_splits(splits, speaker_id, out_dir)
    return splits
