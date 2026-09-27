"""Audio and transcript preprocessing for TORGO.

Two independent concerns live here:
  - Audio: resample to a target rate (16kHz, what Whisper expects), force
    mono, trim leading/trailing silence, and filter out corrupt or
    implausibly short/long clips.
  - Text: normalize transcriptions consistently so WER/CER comparisons later
    aren't polluted by casing/punctuation noise that has nothing to do with
    recognition accuracy.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000


@dataclass
class DurationFilter:
    """Clip-length bounds, in seconds. Defaults are generous for TORGO,
    which contains both single-word prompts and full-sentence readings."""

    min_seconds: float = 0.3
    max_seconds: float = 20.0


def load_and_clean_audio(
    path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    trim_silence: bool = True,
    top_db: float = 30.0,
) -> tuple[np.ndarray | None, int]:
    """Load an audio file, force mono + target_sr, optionally trim silence.

    Returns (waveform, sample_rate). waveform is None if the file is corrupt
    or unreadable, so callers can filter it out rather than crash.
    """
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None, target_sr

    if wav is None or len(wav) == 0:
        return None, target_sr

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if trim_silence:
        wav, _ = librosa.effects.trim(wav, top_db=top_db)

    return wav, sr


def is_valid_clip(waveform: np.ndarray | None, sample_rate: int, filt: DurationFilter) -> bool:
    """True if a loaded waveform passes the corrupt/too-short/too-long checks."""
    if waveform is None or len(waveform) == 0:
        return False
    if not np.isfinite(waveform).all():
        return False
    duration = len(waveform) / sample_rate
    return filt.min_seconds <= duration <= filt.max_seconds


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_transcription(text: str) -> str:
    """Normalize a transcript the same way for both training targets and
    WER/CER scoring: lowercase, strip punctuation, collapse whitespace.

    This mirrors the normalization applied on the hypothesis side in
    src/evaluation/metrics.py — keep the two in sync, since a mismatch there
    silently inflates WER on punctuation/casing differences that aren't real
    recognition errors.
    """
    text = text.lower().strip()
    text = text.translate(_PUNCT_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
