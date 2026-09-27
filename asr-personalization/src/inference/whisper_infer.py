"""Batched Whisper inference over a manifest CSV.

Used as-is by Phase 2's zero-shot baseline eval, and with `adapter_dir` set
by Phase 3+ eval scripts to run a LoRA-adapted model instead of the raw
pretrained checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.data.preprocessing import TARGET_SAMPLE_RATE, load_and_clean_audio


@dataclass
class WhisperRunner:
    """Loads a Whisper checkpoint once and runs batched transcription."""

    model_name: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"
    device: str | None = None
    max_new_tokens: int = 128
    adapter_dir: str | None = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        if self.adapter_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)
        self.model.to(self.device)
        self.model.eval()

    def transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        """Transcribe a batch of audio files. Unreadable files return "" as
        the hypothesis rather than raising, so one corrupt file in a manifest
        doesn't kill a long evaluation run."""
        waveforms: list[np.ndarray] = []
        valid_idx: list[int] = []
        for i, path in enumerate(audio_paths):
            wav, sr = load_and_clean_audio(path, target_sr=TARGET_SAMPLE_RATE, trim_silence=False)
            if wav is not None and len(wav) > 0:
                waveforms.append(wav)
                valid_idx.append(i)

        hyps = [""] * len(audio_paths)
        if not waveforms:
            return hyps

        inputs = self.processor(
            waveforms, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                language=self.language,
                task=self.task,
                max_new_tokens=self.max_new_tokens,
            )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        for i, text in zip(valid_idx, texts):
            hyps[i] = text.strip()
        return hyps

    def transcribe_manifest(
        self, df: pd.DataFrame, batch_size: int = 8, progress_every: int = 50
    ) -> list[str]:
        """Run inference over every row of a manifest DataFrame (must have an
        `audio_path` column), preserving row order."""
        hyps: list[str] = []
        n = len(df)
        paths = df["audio_path"].tolist()
        for start in range(0, n, batch_size):
            batch_paths = paths[start : start + batch_size]
            hyps.extend(self.transcribe_batch(batch_paths))
            done = min(start + batch_size, n)
            if progress_every and (done % progress_every < batch_size or done == n):
                print(f"  transcribed {done}/{n}", flush=True)
        return hyps
