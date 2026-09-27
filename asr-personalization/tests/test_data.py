"""Data pipeline tests.

Split-integrity tests (speaker leakage, duplicate paths, manifest/disk
consistency) run against whatever manifests currently exist under
data/processed/splits/ and are skipped if that directory is empty — they're
meant to be run after scripts/build_dataset.py, not as a substitute for it.
Loader/preprocessing unit tests use small in-memory/synthetic fixtures and
always run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import DurationFilter, is_valid_clip, normalize_transcription
from src.data.splits import SpeakerHoldoutConfig, make_pooled_split, make_speaker_holdout_split

SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits"


def _existing_split_manifests() -> dict[str, pd.DataFrame]:
    if not SPLITS_DIR.exists():
        return {}
    return {p.stem: pd.read_csv(p) for p in SPLITS_DIR.glob("*.csv")}


# --- normalize_transcription ---

def test_normalize_transcription_lowercases_and_strips_punctuation():
    assert normalize_transcription("Hello, World!") == "hello world"


def test_normalize_transcription_collapses_whitespace():
    assert normalize_transcription("  too   many\tspaces  ") == "too many spaces"


# --- is_valid_clip ---

def test_is_valid_clip_rejects_none():
    assert not is_valid_clip(None, 16000, DurationFilter())


def test_is_valid_clip_rejects_too_short():
    wav = np.zeros(100, dtype=np.float32)  # ~6ms at 16kHz
    assert not is_valid_clip(wav, 16000, DurationFilter(min_seconds=0.3))


def test_is_valid_clip_accepts_in_range():
    wav = np.zeros(16000, dtype=np.float32)  # 1s at 16kHz
    assert is_valid_clip(wav, 16000, DurationFilter(min_seconds=0.3, max_seconds=20))


# --- splits, on synthetic data (always runs, doesn't need TORGO downloaded) ---

def _synthetic_manifest(n_dysarthric_speakers=6, n_healthy_speakers=6, per_speaker=20) -> pd.DataFrame:
    rows = []
    severities = ["mild", "moderate", "severe"]
    for i in range(n_dysarthric_speakers):
        spk = f"F{i:02d}" if i % 2 == 0 else f"M{i:02d}"
        for j in range(per_speaker):
            rows.append({
                "speaker_id": spk, "audio_path": f"/fake/{spk}/{j}.wav",
                "transcription": f"utterance {j}", "dysarthria_status": "dysarthric",
                "severity": severities[i % 3], "gender": "female" if spk.startswith("F") else "male",
                "mic_source": "head", "duration": 1.5,
            })
    for i in range(n_healthy_speakers):
        spk = f"FC{i:02d}" if i % 2 == 0 else f"MC{i:02d}"
        for j in range(per_speaker):
            rows.append({
                "speaker_id": spk, "audio_path": f"/fake/{spk}/{j}.wav",
                "transcription": f"utterance {j}", "dysarthria_status": "healthy",
                "severity": None, "gender": "female" if spk.startswith("FC") else "male",
                "mic_source": "head", "duration": 1.5,
            })
    return pd.DataFrame(rows)


def test_speaker_holdout_split_no_speaker_overlap_synthetic():
    df = _synthetic_manifest()
    splits = make_speaker_holdout_split(df, SpeakerHoldoutConfig(seed=42))
    train_spk = set(splits["train"].speaker_id)
    val_spk = set(splits["val"].speaker_id)
    test_spk = set(splits["test"].speaker_id)
    assert train_spk & test_spk == set()
    assert train_spk & val_spk == set()
    assert val_spk & test_spk == set()
    assert len(test_spk) > 0


def test_pooled_split_stratified_ratio_synthetic():
    df = _synthetic_manifest()
    splits = make_pooled_split(df, ratios=(0.8, 0.1, 0.1), seed=42)
    total = sum(len(s) for s in splits.values())
    assert total == len(df)
    for name, split_df in splits.items():
        if len(split_df) == 0:
            continue
        ratio = (split_df.dysarthria_status == "dysarthric").mean()
        overall_ratio = (df.dysarthria_status == "dysarthric").mean()
        assert abs(ratio - overall_ratio) < 0.1, f"{name} split stratification drifted too far"


# --- integrity checks against real manifests, if they exist on disk ---

@pytest.mark.skipif(not _existing_split_manifests(), reason="run scripts/build_dataset.py first")
def test_speaker_holdout_no_speaker_leakage_on_disk():
    manifests = _existing_split_manifests()
    train = manifests.get("speaker_holdout_train")
    val = manifests.get("speaker_holdout_val")
    test = manifests.get("speaker_holdout_test")
    if train is None or val is None or test is None:
        pytest.skip("speaker_holdout manifests not found")
    train_spk, val_spk, test_spk = set(train.speaker_id), set(val.speaker_id), set(test.speaker_id)
    assert train_spk & test_spk == set(), "speaker leaked between speaker_holdout train and test"
    assert val_spk & test_spk == set(), "speaker leaked between speaker_holdout val and test"
    assert train_spk & val_spk == set(), "speaker leaked between speaker_holdout train and val"


@pytest.mark.skipif(not _existing_split_manifests(), reason="run scripts/build_dataset.py first")
def test_no_duplicate_audio_paths_across_splits_on_disk():
    manifests = _existing_split_manifests()
    for prefix in ("pooled", "speaker_holdout"):
        # *_subsample manifests (e.g. pooled_train_subsample) are subsets of a
        # split by design, not splits of their own.
        relevant = {k: v for k, v in manifests.items() if k.startswith(prefix) and not k.endswith("_subsample")}
        if len(relevant) < 2:
            continue
        seen = set()
        for name, df in relevant.items():
            paths = set(df.audio_path)
            overlap = seen & paths
            assert not overlap, f"duplicate audio_path(s) across {prefix} splits: {list(overlap)[:5]}"
            seen |= paths


@pytest.mark.skipif(not _existing_split_manifests(), reason="run scripts/build_dataset.py first")
def test_all_manifest_audio_files_exist_on_disk():
    manifests = _existing_split_manifests()
    missing = []
    for name, df in manifests.items():
        for path in df.audio_path:
            if not Path(path).exists():
                missing.append((name, path))
    assert not missing, f"{len(missing)} manifest audio paths do not exist, e.g. {missing[:5]}"
