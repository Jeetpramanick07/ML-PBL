"""Two independent split strategies over a cleaned TORGO manifest.

1. Pooled split (`make_pooled_split`): random utterance-level train/val/test,
   stratified by dysarthria_status so each split has a similar
   healthy:dysarthric ratio. This is what Phase 3's "generic model" trains
   and is evaluated on — it says nothing about generalizing to a new speaker.

2. Speaker-holdout split (`make_speaker_holdout_split`): whole speakers are
   assigned to train/val/test, never split across them. Test speakers
   simulate brand-new users the system has never heard, which is the only
   way to honestly measure few-shot personalization (Phase 4) — a model
   can't get credit for "adapting" to a speaker it already saw pooled
   utterances from during training.

Both are written to CSV manifests under data/processed/splits/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test


def make_pooled_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Random utterance-level split, stratified by dysarthria_status.

    Stratifying per-group (rather than a single global shuffle) keeps the
    healthy:dysarthric ratio close to constant across train/val/test even
    though the corpus itself is imbalanced (~2:1 healthy:dysarthric).
    """
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "ratios must sum to 1"

    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}

    for _, group in df.groupby("dysarthria_status", group_keys=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        parts["train"].append(group.loc[idx[:n_train]])
        parts["val"].append(group.loc[idx[n_train:n_train + n_val]])
        parts["test"].append(group.loc[idx[n_train + n_val:]])

    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=df.columns)
            for k, v in parts.items()}


@dataclass
class SpeakerHoldoutConfig:
    """How many *speakers* (not utterances) go into each split.

    Dysarthric speakers are assigned severity-first so test coverage spans
    the severity range when severity metadata is available; speakers with
    unknown severity are treated as their own bucket.
    """

    test_dysarthric_per_severity: int = 1
    test_healthy_speakers: int = 2
    val_dysarthric_speakers: int = 1
    val_healthy_speakers: int = 1
    seed: int = 42


def make_speaker_holdout_split(
    df: pd.DataFrame,
    config: SpeakerHoldoutConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Whole-speaker split: disjoint speaker sets for train/val/test.

    Test speakers are the "new user" simulation set for Phase 4 few-shot
    personalization; they must never appear in train or val.
    """
    if config is None:
        config = SpeakerHoldoutConfig()

    rng = np.random.default_rng(config.seed)

    speaker_meta = (
        df[["speaker_id", "dysarthria_status", "severity"]]
        .drop_duplicates("speaker_id")
        .set_index("speaker_id")
    )

    dysarthric_speakers = speaker_meta[speaker_meta.dysarthria_status == "dysarthric"].index.tolist()
    healthy_speakers = speaker_meta[speaker_meta.dysarthria_status == "healthy"].index.tolist()

    # Group dysarthric speakers by severity bucket (unknown severity -> "unknown").
    by_severity: dict[str, list[str]] = {}
    for spk in dysarthric_speakers:
        sev = speaker_meta.loc[spk, "severity"] or "unknown"
        by_severity.setdefault(sev, []).append(spk)
    for sev in by_severity:
        rng.shuffle(by_severity[sev])

    shuffled_healthy = healthy_speakers.copy()
    rng.shuffle(shuffled_healthy)

    test_speakers: set[str] = set()
    val_speakers: set[str] = set()

    # --- test: cover as many severity buckets as possible ---
    for sev, speakers in by_severity.items():
        take = speakers[: config.test_dysarthric_per_severity]
        test_speakers.update(take)
        by_severity[sev] = speakers[config.test_dysarthric_per_severity:]

    test_speakers.update(shuffled_healthy[: config.test_healthy_speakers])
    remaining_healthy = shuffled_healthy[config.test_healthy_speakers:]

    # --- val: from what's left after test, disjoint from test ---
    remaining_dysarthric = [s for sevlist in by_severity.values() for s in sevlist]
    rng.shuffle(remaining_dysarthric)
    val_speakers.update(remaining_dysarthric[: config.val_dysarthric_speakers])
    remaining_dysarthric = remaining_dysarthric[config.val_dysarthric_speakers:]

    val_speakers.update(remaining_healthy[: config.val_healthy_speakers])
    remaining_healthy = remaining_healthy[config.val_healthy_speakers:]

    # --- train: everyone left ---
    train_speakers = set(remaining_dysarthric) | set(remaining_healthy)

    assert not (test_speakers & train_speakers), "leakage: test/train speaker overlap"
    assert not (test_speakers & val_speakers), "leakage: test/val speaker overlap"
    assert not (val_speakers & train_speakers), "leakage: val/train speaker overlap"

    return {
        "train": df[df.speaker_id.isin(train_speakers)].reset_index(drop=True),
        "val": df[df.speaker_id.isin(val_speakers)].reset_index(drop=True),
        "test": df[df.speaker_id.isin(test_speakers)].reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path, prefix: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(out_dir / f"{prefix}_{name}.csv", index=False)
