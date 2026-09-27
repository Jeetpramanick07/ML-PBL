"""Turn an arbitrary uploaded audio file into a waveform Whisper can consume.

Phone mic recordings won't always be clean 16kHz WAV — Android's default
recorder formats (e.g. m4a/AAC) aren't decodable by libsndfile (what
`src.data.preprocessing.load_and_clean_audio` uses under the hood). Rather
than requiring the client to get this exactly right, we try the fast path
first and fall back to an `ffmpeg` conversion for anything libsndfile
chokes on. This is what makes /transcribe robust to live mic input instead
of only TORGO's pre-cleaned wav files (see Phase A's "test on your own
recorded clip" requirement).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np

from src.data.preprocessing import TARGET_SAMPLE_RATE, load_and_clean_audio

FFMPEG_BIN = shutil.which("ffmpeg")

UPLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "asr_demo_uploads"
UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)


class UnreadableAudioError(RuntimeError):
    """Raised when neither libsndfile nor an ffmpeg fallback can decode the
    uploaded file. The caller (an API route) turns this into a 400."""


def save_upload_to_tmp(raw_bytes: bytes, filename_hint: str) -> Path:
    suffix = Path(filename_hint).suffix or ".bin"
    out_path = UPLOAD_TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    out_path.write_bytes(raw_bytes)
    return out_path


def _ffmpeg_to_wav(src_path: Path) -> Path | None:
    if FFMPEG_BIN is None:
        return None
    wav_path = src_path.with_suffix(".converted.wav")
    try:
        subprocess.run(
            [
                FFMPEG_BIN, "-y", "-loglevel", "error",
                "-i", str(src_path),
                "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
                str(wav_path),
            ],
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return wav_path if wav_path.exists() else None


def load_uploaded_audio(raw_bytes: bytes, filename_hint: str) -> tuple[np.ndarray, Path]:
    """Save an uploaded file to a temp path and return a clean 16kHz mono
    waveform, converting via ffmpeg if the raw format isn't directly
    readable. Raises UnreadableAudioError if both paths fail (corrupt file,
    zero-length recording, unsupported codec with no ffmpeg on PATH).

    Returns (waveform, temp_path_used) — the caller is responsible for
    cleaning up temp_path_used (and any sibling .converted.wav) once done.
    """
    tmp_path = save_upload_to_tmp(raw_bytes, filename_hint)

    wav, sr = load_and_clean_audio(str(tmp_path), target_sr=TARGET_SAMPLE_RATE, trim_silence=False)
    if wav is not None and len(wav) > 0:
        return wav, tmp_path

    converted = _ffmpeg_to_wav(tmp_path)
    if converted is not None:
        wav, sr = load_and_clean_audio(str(converted), target_sr=TARGET_SAMPLE_RATE, trim_silence=False)
        if wav is not None and len(wav) > 0:
            return wav, tmp_path

    raise UnreadableAudioError(
        f"Could not decode uploaded audio '{filename_hint}' (tried libsndfile"
        + (" and an ffmpeg fallback" if FFMPEG_BIN else "; ffmpeg not found on PATH for a fallback")
        + "). Have the client record as 16kHz mono WAV if this keeps happening."
    )


def cleanup_tmp(path: Path) -> None:
    for p in (path, path.with_suffix(".converted.wav")):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
