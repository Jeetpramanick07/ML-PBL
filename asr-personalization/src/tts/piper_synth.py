"""Piper text-to-speech for the demo backend's POST /speak.

Integration choice: the `piper-tts` Python package (pip), not the standalone
Piper binary. It ships a prebuilt Windows wheel with espeak-ng bundled, adds
only onnxruntime + two small pure-Python deps to the existing venv (no
conflicts with torch/transformers), and runs in-process — so the voice is
loaded once and kept warm, exactly like the Whisper/LoRA models in
model_manager.py. The binary would instead mean either spawning a subprocess
per request (reloading the ~63 MB voice every time) or managing a long-lived
child process over stdin/stdout, plus a separate platform-specific download.

The voice model is NOT committed to git (see README): download it with
    python -m piper.download_voices --download-dir models/piper en_US-lessac-medium
If it's missing or fails to load, the server still starts — /health reports
tts_available: false and /speak returns 503, so a TTS problem can never take
down transcription.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import wave
from pathlib import Path

logger = logging.getLogger("asr_demo.tts")

DEFAULT_VOICE_PATH = Path("models/piper/en_US-lessac-medium.onnx")
MAX_TEXT_CHARS = 1000  # a transcription is a sentence or two; this just bounds request cost


class InvalidTextError(ValueError):
    """Text that can't be synthesized (empty, too long, or nothing speakable).
    Its message is safe to return to the client as a 400."""


class PiperSynthesizer:
    def __init__(self, voice_path: Path = DEFAULT_VOICE_PATH):
        from piper import PiperVoice  # imported here so a broken install only disables TTS

        self.voice_path = Path(voice_path)
        t0 = time.time()
        self.voice = PiperVoice.load(self.voice_path)  # CPU onnxruntime; fast enough for short phrases
        self.sample_rate = self.voice.config.sample_rate
        self._lock = threading.Lock()  # serialize synthesis, same single-laptop-demo tradeoff as ModelManager
        logger.info("Loaded Piper voice %s (%d Hz) in %.1fs", self.voice_path.name, self.sample_rate, time.time() - t0)

    @classmethod
    def try_load(cls, voice_path: Path = DEFAULT_VOICE_PATH) -> tuple[PiperSynthesizer | None, str | None]:
        """Returns (synthesizer, None) on success or (None, reason) on failure,
        so startup can continue without TTS instead of crashing."""
        if not Path(voice_path).exists():
            reason = (f"Piper voice model not found at {voice_path} — download it with "
                      f"`python -m piper.download_voices --download-dir {Path(voice_path).parent.as_posix()} "
                      f"{Path(voice_path).stem}`")
            logger.warning(reason)
            return None, reason
        try:
            return cls(voice_path), None
        except Exception as exc:  # broad on purpose: any load failure just disables TTS
            logger.exception("Failed to load Piper voice from %s", voice_path)
            return None, f"Piper failed to load: {exc}"

    def synthesize(self, text: str) -> bytes:
        """Return a complete 16-bit mono WAV file (bytes) speaking `text`."""
        text = (text or "").strip()
        if not text:
            raise InvalidTextError("Text is empty.")
        if len(text) > MAX_TEXT_CHARS:
            raise InvalidTextError(f"Text is too long ({len(text)} chars; max {MAX_TEXT_CHARS}).")

        # Collect chunks first rather than using voice.synthesize_wav(): when the
        # text has nothing speakable (e.g. only punctuation) Piper yields no
        # chunks, never sets the WAV format, and closing the wave writer then
        # raises "# channels not specified" — a 500 instead of a clean 400.
        with self._lock:
            chunks = list(self.voice.synthesize(text))
        if not chunks:
            raise InvalidTextError("Text contains nothing speakable.")

        first = chunks[0]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setframerate(first.sample_rate)
            wav_file.setsampwidth(first.sample_width)
            wav_file.setnchannels(first.sample_channels)
            for chunk in chunks:
                wav_file.writeframes(chunk.audio_int16_bytes)
        return buf.getvalue()
