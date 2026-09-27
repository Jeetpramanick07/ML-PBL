"""Phase 4: few-shot per-speaker LoRA personalization.

Given a held-out speaker and a small enrollment subset of their audio, fits
a per-speaker LoRA adapter on top of the *Phase 3 generic model* (base
Whisper with the pooled LoRA fine-tune merged in — see `build_personalized_base`)
using one of two initialization strategies:

  - "naive": the adapter starts from PEFT's default random init (LoRA A
    kaiming-uniform, B zeros) and is fit purely on that speaker's enrollment
    data.
  - "meta": the adapter starts from a shared initialization produced by
    `src.training.meta_init.run_reptile_meta_init` (Reptile meta-learning
    across other speakers' enrollment-sized subsets), then fine-tuned the
    same way. See src/training/meta_init.py for how that initialization is
    produced.

This module does not touch the Phase 3 generic model or its checkpoint — it
only reads `checkpoints/generic/best` and treats it as frozen, pretrained
weights to personalize on top of.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor, get_linear_schedule_with_warmup

from src.data.preprocessing import TARGET_SAMPLE_RATE, load_and_clean_audio, normalize_transcription
from src.evaluation.metrics import compute_wer_cer
from src.models.whisper_lora import WhisperLoRAConfig, apply_lora, load_base_model


class SpeakerASRDataset(Dataset):
    """Loads audio + normalized transcription for one speaker's enrollment
    (or eval) manifest. Mirrors src.training.train_generic's dataset, kept
    separate so Phase 4 doesn't depend on / modify Phase 3's training module."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav, sr = load_and_clean_audio(row.audio_path, target_sr=TARGET_SAMPLE_RATE, trim_silence=False)
        if wav is None or len(wav) == 0:
            wav = np.zeros(TARGET_SAMPLE_RATE // 10, dtype=np.float32)
        return {"waveform": wav, "text": normalize_transcription(str(row.transcription))}


class Collator:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        waveforms = [b["waveform"] for b in batch]
        texts = [b["text"] for b in batch]
        feats = self.processor.feature_extractor(waveforms, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        labels = self.processor.tokenizer(texts, return_tensors="pt", padding=True)
        label_ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return {"input_features": feats.input_features, "labels": label_ids}


def build_personalized_base(model_name: str, generic_adapter_dir: str, device: str = "cpu"):
    """Load pretrained Whisper, apply the Phase 3 generic LoRA adapter, and
    merge it into the base weights. The returned model is a plain
    (non-PEFT) WhisperForConditionalGeneration with the generic fine-tune
    baked in — the frozen starting point every per-speaker adapter builds on.
    """
    base = load_base_model(model_name)
    generic = PeftModel.from_pretrained(base, generic_adapter_dir)
    merged = generic.merge_and_unload()
    merged.to(device)
    return merged


def new_adapter(merged_base, lora_cfg: WhisperLoRAConfig, device: str = "cpu") -> PeftModel:
    """Wrap a fresh copy of the (frozen) merged base with a new, randomly
    initialized LoRA adapter. Deep-copies the base so many adapters can be
    trained in the same process without PEFT layers stacking on each other."""
    model = apply_lora(copy.deepcopy(merged_base), lora_cfg)
    model.to(device)
    return model


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 6
    learning_rate: float = 5.0e-4
    warmup_steps: int = 5
    # fp16 autocast + GradScaler — see src/training/train_generic.py's
    # identical mechanism for the measurements justifying this (~8x speedup
    # on this project's GPU vs. plain fp32). Off by default so every prior
    # caller's exact behavior (including CPU runs) is unchanged; the master
    # personalization run turns it on explicitly to fit its time budget.
    mixed_precision: bool = False


def train_adapter(
    model: PeftModel,
    processor: WhisperProcessor,
    train_df: pd.DataFrame,
    cfg: TrainingConfig,
    device: str = "cpu",
    init_state_dict: dict | None = None,
    verbose: bool = False,
) -> PeftModel:
    """Fine-tune `model`'s LoRA parameters on `train_df` in place, optionally
    starting from `init_state_dict` (the meta-initialized weights) instead of
    PEFT's default random init."""
    if init_state_dict is not None:
        set_peft_model_state_dict(model, init_state_dict)

    loader = DataLoader(
        SpeakerASRDataset(train_df), batch_size=cfg.batch_size, shuffle=True,
        collate_fn=Collator(processor),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    total_steps = max(len(loader) * cfg.num_epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(cfg.warmup_steps, total_steps), num_training_steps=total_steps
    )
    mixed_precision = cfg.mixed_precision and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    model.train()
    for epoch in range(cfg.num_epochs):
        for batch in loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                out = model(input_features=input_features, labels=labels)
            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        if verbose:
            print(f"    epoch {epoch} loss {out.loss.item():.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_adapter(
    model: PeftModel,
    processor: WhisperProcessor,
    eval_df: pd.DataFrame,
    device: str = "cpu",
    batch_size: int = 4,
    max_new_tokens: int = 64,
    language: str = "en",
    task: str = "transcribe",
) -> dict:
    """Transcribe `eval_df` with `model` and return {n_samples, wer, cer,
    hypotheses}. Standalone rather than reusing src.inference.whisper_infer
    because that module loads a fresh model from a single adapter_dir path —
    here the model is already an in-memory PeftModel wrapping a merged
    (generic + personal) base, so we just call .generate() directly."""
    model.eval()
    refs, hyps = [], []
    paths = eval_df["audio_path"].tolist()
    texts = eval_df["transcription"].astype(str).tolist()

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_refs = texts[start : start + batch_size]
        waveforms = []
        valid_refs = []
        for p, r in zip(batch_paths, batch_refs):
            wav, _ = load_and_clean_audio(p, target_sr=TARGET_SAMPLE_RATE, trim_silence=False)
            if wav is not None and len(wav) > 0:
                waveforms.append(wav)
                valid_refs.append(r)
        if not waveforms:
            continue
        inputs = processor(waveforms, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        generated_ids = model.generate(
            inputs.input_features.to(device), language=language, task=task, max_new_tokens=max_new_tokens
        )
        batch_hyps = [t.strip() for t in processor.batch_decode(generated_ids, skip_special_tokens=True)]
        hyps.extend(batch_hyps)
        refs.extend(valid_refs)

    m = compute_wer_cer(refs, hyps)
    return {"n_samples": m.n_samples, "wer": m.wer, "cer": m.cer, "references": refs, "hypotheses": hyps}
