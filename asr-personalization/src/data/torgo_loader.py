"""Parse the raw TORGO corpus directory tree into a flat pandas DataFrame.

TORGO ships as per-speaker folders (e.g. F01, FC01, M03, MC04 — the prefix
letter(s) encode gender and control/dysarthric status, see `_speaker_gender`
and `_speaker_dysarthria_status` below), each containing one or more session
folders. A session folder holds a `wav_headMic/` and/or `wav_arrayMic/`
directory of audio files, plus a `prompts/` directory of same-stem `.txt`
files holding what the speaker was asked to say. We don't assume a fixed
session-folder depth — TORGO releases have varied on this — so we search
recursively for any `wav_headMic`/`wav_arrayMic` directory under a speaker
folder and look for a sibling `prompts/` directory next to it.

Some TORGO prompts point to an image the speaker described rather than
containing literal text (e.g. a `.jpg`/`.bmp` filename as the "transcript").
Those aren't usable ground truth for ASR and are kept in the output with
`transcription=None` so downstream code can filter them explicitly rather
than silently training on a filename.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

WAV_DIR_NAMES = {"wav_headmic": "head", "wav_arraymic": "array"}
IMAGE_PROMPT_RE = re.compile(r"^\S+\.(jpe?g|bmp|png|gif)$", re.IGNORECASE)

# TORGO prompt .txt files mix three kinds of content, discovered by inspecting
# the actual corpus (not documented anywhere machine-readable):
#   1. Plain literal text the speaker was asked to say -> use as-is.
#   2. Pure bracketed task instructions, e.g. "[relax your mouth in its
#      normal position]" or "[say Ah-P-Eee repeatedly]" -> nothing was
#      actually *spoken* as words, so these become transcription=None.
#   3. A word with a bracketed disambiguation note, e.g.
#      "lead [as in I will lead you]" or 'tear [as in "tear up that paper"]'
#      -> only "lead"/"tear" is spoken; the bracket is a note for whoever ran
#      the session, not part of the utterance, and must be stripped rather
#      than kept as ground truth text.
# A handful of files (rare, ~1 in 1000) also contain leaked terminal ANSI
# cursor-movement escape codes (\x1b[D, \x1b[C, ...) from however the prompt
# files were originally authored; those are stripped as control characters.
ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")  # e.g. leaked cursor-movement codes like \x1b[D
BRACKET_RE = re.compile(r"\[[^\]]*\]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_COLLAPSE_WS_RE = re.compile(r"\s+")

# Best-effort severity labels for dysarthric speakers, as commonly reported in
# TORGO literature (Rudzicz et al., 2012 and follow-on work). These are NOT
# shipped inside the raw TORGO download itself — there is no machine-readable
# severity field in the corpus — so treat this mapping as approximate. Verify
# against any documentation bundled with your TORGO copy before relying on it
# for anything beyond coarse stratification. Unlisted / control speakers get
# severity=None.
KNOWN_SEVERITY = {
    "F01": "severe",
    "F03": "mild",
    "F04": "moderate",
    "M01": "severe",
    "M02": "severe",
    "M03": "mild",
    "M04": "severe",
    "M05": "moderate",
}


@dataclass
class TorgoRow:
    speaker_id: str
    audio_path: str
    transcription: str | None
    dysarthria_status: str
    severity: str | None
    gender: str
    mic_source: str
    duration: float | None


def _speaker_gender(speaker_id: str) -> str:
    return "female" if speaker_id.upper().startswith("F") else "male"


def _speaker_dysarthria_status(speaker_id: str) -> str:
    # Control speakers are named FC../MC.. ; dysarthric speakers F../M..
    return "healthy" if re.match(r"^(F|M)C", speaker_id.upper()) else "dysarthric"


def _find_speaker_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and re.match(r"^(F|M)C?\d+$", p.name.upper()))


def _read_prompt(prompt_path: Path) -> str | None:
    try:
        text = prompt_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text:
        return None

    text = ANSI_CSI_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub("", text)
    text = BRACKET_RE.sub("", text)  # drop instructions / disambiguation notes, see module docstring
    text = _COLLAPSE_WS_RE.sub(" ", text).strip()

    if not text:
        return None  # was a pure bracketed instruction, e.g. "[relax your mouth ...]"
    if IMAGE_PROMPT_RE.match(text):
        return None  # references an image the speaker described; no fixed transcript
    return text


def _audio_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return None


def load_torgo(raw_dir: str | Path, compute_duration: bool = True) -> pd.DataFrame:
    """Walk `raw_dir` (e.g. data/raw/torgo) and return one row per audio file.

    Columns: speaker_id, audio_path, transcription, dysarthria_status,
    severity, gender, mic_source, duration.

    `transcription` is None for prompts that reference an image file instead
    of literal text (see module docstring) — callers should drop those rows
    before training/eval, not treat None as an empty string.
    """
    root = Path(raw_dir)
    rows: list[TorgoRow] = []

    for speaker_dir in _find_speaker_dirs(root):
        speaker_id = speaker_dir.name.upper()
        gender = _speaker_gender(speaker_id)
        status = _speaker_dysarthria_status(speaker_id)
        severity = KNOWN_SEVERITY.get(speaker_id)

        for wav_dir in speaker_dir.rglob("*"):
            if not wav_dir.is_dir() or wav_dir.name.lower() not in WAV_DIR_NAMES:
                continue
            mic_source = WAV_DIR_NAMES[wav_dir.name.lower()]
            prompts_dir = wav_dir.parent / "prompts"

            for wav_path in sorted(wav_dir.glob("*.wav")):
                prompt_path = prompts_dir / f"{wav_path.stem}.txt"
                transcription = _read_prompt(prompt_path) if prompt_path.exists() else None
                duration = _audio_duration(wav_path) if compute_duration else None

                rows.append(
                    TorgoRow(
                        speaker_id=speaker_id,
                        audio_path=str(wav_path),
                        transcription=transcription,
                        dysarthria_status=status,
                        severity=severity,
                        gender=gender,
                        mic_source=mic_source,
                        duration=duration,
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "speaker_id", "audio_path", "transcription", "dysarthria_status",
                "severity", "gender", "mic_source", "duration",
            ]
        )
    return df


if __name__ == "__main__":
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/torgo"
    df = load_torgo(raw, compute_duration=False)
    print(f"Loaded {len(df)} rows from {raw}")
    print(df.groupby(["speaker_id", "dysarthria_status", "mic_source"]).size())
