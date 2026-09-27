"""Owns every model the demo backend can serve, loaded once at process
startup (never inside a request handler — model loading is slow and must
not block /transcribe).

Model resolution order for the "no personalized adapter" case:
  1. checkpoints/generic_full/best    (the final master-run pooled LoRA:
     stratified 40% subsample, epoch 6, val WER 16.30% — see
     reports/master_results_final.md)
  2. checkpoints/generic_unseen/best  (early 1,200-row / 2-epoch test
     checkpoint — fallback only)
  3. checkpoints/generic/best          (same, without speaker exclusion —
     fallback only)
  4. zero-shot Whisper                 (no fine-tune found at all)

Whichever is found first is logged and reported via /health, per Phase A's
"log which one was used" requirement.

Personalized profiles come from two places:
  - Pre-trained master-run adapters, checkpoints/personalized_final/
    <speaker_id>/naive/5min/ (F03/M04/M05, 5 min of TORGO enrollment audio
    each). These were trained on top of generic_full/best specifically, so
    they are only registered when that is the loaded pooled base — stacking
    them on any other base would be a mismatched, never-evaluated model.
  - Live-enrolled adapters (see enrollment.py), kept under
    checkpoints/demo_personalized/<speaker_id>/ — a directory separate from
    the research checkpoints so a live demo enrollment can never overwrite
    them. A live enrollment under the same speaker_id replaces the
    pre-trained profile for the rest of the session.
Both kinds are deep-copies of the pooled base with a per-speaker LoRA
adapter applied on top.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperProcessor

from src.models.whisper_lora import load_base_model
from src.training.personalize import build_personalized_base

logger = logging.getLogger("asr_demo.model_manager")

DEMO_PERSONALIZED_DIR = Path("checkpoints/demo_personalized")
PRETRAINED_PERSONALIZED_DIR = Path("checkpoints/personalized_final")
# The pooled checkpoint the pre-trained personalized adapters were fit on top of.
PRETRAINED_PERSONALIZED_BASE = "checkpoints/generic_full/best"

_POOLED_CANDIDATES = [
    "checkpoints/generic_full/best",
    "checkpoints/generic_unseen/best",
    "checkpoints/generic/best",
]


@dataclass
class LoadedProfile:
    speaker_id: str
    adapter_dir: str
    source: str = "live-enrolled"  # "live-enrolled" | "pretrained"
    model: PeftModel | None = None  # lazily populated on first use


@dataclass
class ModelManager:
    model_name: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"
    device: str | None = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.pooled_label, self.pooled_source_dir, self.pooled_base = self._load_pooled_base()
        self._profiles: dict[str, LoadedProfile] = {}
        self._gen_lock = threading.Lock()  # serialize GPU generate() calls; this is a
        # single-laptop demo, not a throughput service, so simplicity beats concurrency here.
        self._scan_pretrained_checkpoints()
        self._scan_demo_checkpoints()
        logger.info(
            "ModelManager ready: pooled=%s (%s), device=%s, %d pre-existing personalized profile(s)",
            self.pooled_label, self.pooled_source_dir or "none", self.device, len(self._profiles),
        )

    def _load_pooled_base(self) -> tuple[str, str | None, torch.nn.Module]:
        for candidate_dir in _POOLED_CANDIDATES:
            if (Path(candidate_dir) / "adapter_config.json").exists():
                merged = build_personalized_base(self.model_name, candidate_dir, device=self.device)
                merged.eval()
                logger.info("Loaded pooled LoRA model from %s", candidate_dir)
                # e.g. "pooled-lora:generic_full", so model_used names the exact checkpoint.
                return f"pooled-lora:{Path(candidate_dir).parent.name}", candidate_dir, merged
        logger.warning("No pooled LoRA adapter found under %s — falling back to zero-shot Whisper", _POOLED_CANDIDATES)
        base = load_base_model(self.model_name)
        base.to(self.device)
        base.eval()
        return "zero-shot", None, base

    def _scan_pretrained_checkpoints(self) -> None:
        if not PRETRAINED_PERSONALIZED_DIR.exists():
            return
        if Path(self.pooled_source_dir or "") != Path(PRETRAINED_PERSONALIZED_BASE):
            logger.warning(
                "Not registering pre-trained personalized adapters from %s: they were trained on %s, "
                "but the loaded pooled base is %s", PRETRAINED_PERSONALIZED_DIR, PRETRAINED_PERSONALIZED_BASE,
                self.pooled_source_dir or "zero-shot",
            )
            return
        for speaker_dir in sorted(PRETRAINED_PERSONALIZED_DIR.iterdir()):
            adapter_dir = speaker_dir / "naive" / "5min"
            if (adapter_dir / "adapter_config.json").exists():
                self._profiles[speaker_dir.name] = LoadedProfile(
                    speaker_id=speaker_dir.name, adapter_dir=str(adapter_dir), source="pretrained",
                )
                logger.info("Registered pre-trained personalized profile %s from %s", speaker_dir.name, adapter_dir)

    def _scan_demo_checkpoints(self) -> None:
        if not DEMO_PERSONALIZED_DIR.exists():
            return
        for speaker_dir in DEMO_PERSONALIZED_DIR.iterdir():
            if (speaker_dir / "adapter_config.json").exists():
                self._profiles[speaker_dir.name] = LoadedProfile(speaker_id=speaker_dir.name, adapter_dir=str(speaker_dir))
                logger.info("Found pre-existing demo personalized profile on disk: %s", speaker_dir.name)

    # -- personalization registry -------------------------------------------------

    def register_personalized_adapter(self, speaker_id: str, adapter_dir: str) -> None:
        """Called by enrollment.py once a background fine-tune finishes and
        saves its adapter. Registers it as ready but does not eagerly load
        the model onto the GPU — the next /transcribe for that speaker does."""
        self._profiles[speaker_id] = LoadedProfile(speaker_id=speaker_id, adapter_dir=adapter_dir)

    def has_profile(self, speaker_id: str) -> bool:
        return speaker_id in self._profiles

    def list_profiles(self) -> list[str]:
        return sorted(self._profiles.keys())

    def new_personalization_base(self):
        """A fresh deep-copy of the frozen pooled base for enrollment.py to
        attach a new LoRA adapter to. Deep-copying (rather than re-loading
        from disk) keeps enrollment fast — this is the same pattern
        src.training.personalize.new_adapter uses for TORGO experiments."""
        return copy.deepcopy(self.pooled_base)

    # -- inference -----------------------------------------------------------

    def _resolve_model(self, speaker_id: str | None) -> tuple[torch.nn.Module, str]:
        if speaker_id and speaker_id in self._profiles:
            profile = self._profiles[speaker_id]
            if profile.model is None:
                logger.info("Lazily loading personalized adapter for speaker_id=%s from %s", speaker_id, profile.adapter_dir)
                model = PeftModel.from_pretrained(self.new_personalization_base(), profile.adapter_dir)
                model.to(self.device)
                model.eval()
                profile.model = model
            return profile.model, f"personalized:{speaker_id}"
        return self.pooled_base, self.pooled_label

    def transcribe(self, waveform, speaker_id: str | None = None, max_new_tokens: int = 64) -> tuple[str, str, float]:
        """Returns (transcription_text, model_used, latency_ms). latency_ms
        covers only the forward/generate call, not model resolution/loading
        (the one-time lazy load of a personalized adapter is logged
        separately and excluded, matching how a "warm" profile behaves for
        every request after the first)."""
        model, model_used = self._resolve_model(speaker_id)

        inputs = self.processor([waveform], sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with self._gen_lock:
            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features, language=self.language, task=self.task, max_new_tokens=max_new_tokens,
                )
            latency_ms = (time.time() - t0) * 1000

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text, model_used, latency_ms

    # -- health ----------------------------------------------------------------

    def health(self) -> dict:
        return {
            "device": self.device,
            "pooled_model": {
                "model_used_label": self.pooled_label,
                "source_checkpoint": self.pooled_source_dir,
            },
            "personalized_profiles": [
                {"speaker_id": sid, "adapter_dir": p.adapter_dir, "source": p.source,
                 "loaded_in_memory": p.model is not None}
                for sid, p in sorted(self._profiles.items())
            ],
        }  # TTS status is added by main.py's /health (Piper is owned by the API layer, not this class)
