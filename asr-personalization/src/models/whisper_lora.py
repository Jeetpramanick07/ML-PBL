"""Whisper wrapped with a PEFT LoRA adapter.

LoRA targets the attention projection layers (`q_proj`, `v_proj`) across both
the encoder and decoder — the standard, minimal target set for adapting
Whisper reported in the LoRA/Whisper fine-tuning literature. Rank, alpha, and
dropout are configurable so different experiments (generic pooled fine-tune
in Phase 3, per-speaker adapters in Phase 4) can use different capacities
without touching this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


@dataclass
class WhisperLoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))


def load_base_model(model_name: str) -> WhisperForConditionalGeneration:
    return WhisperForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True)


def apply_lora(model: WhisperForConditionalGeneration, lora_cfg: WhisperLoRAConfig) -> PeftModel:
    """Freeze the base model and attach a fresh LoRA adapter."""
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        bias="none",
    )
    return get_peft_model(model, peft_config)


def load_lora_for_inference(
    base_model_name: str, adapter_dir: str, device: str | None = None
) -> tuple[PeftModel, WhisperProcessor]:
    """Load a base Whisper checkpoint with a saved LoRA adapter applied,
    ready for `.generate()`. Used by eval scripts once training has produced
    an adapter directory (`save_pretrained` output)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained(base_model_name)
    base = load_base_model(base_model_name)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device)
    model.eval()
    return model, processor
