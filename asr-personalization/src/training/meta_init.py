"""Phase 4: Reptile-style first-order meta-initialization for per-speaker LoRA.

Idea: instead of starting every speaker's personalization adapter from PEFT's
default random init ("naive" strategy), meta-train a *shared* LoRA
initialization across several other speakers' enrollment-sized data subsets
first, so per-speaker adaptation starts closer to "a good adapter for a
dysarthric speaker" and needs less enrollment data to converge.

This deliberately uses the simplest first-order meta-learning update
(Reptile, Nichol et al. 2018) rather than a full second-order MAML: for each
outer iteration, clone the current shared weights, take a few gradient steps
on one sampled speaker's enrollment-sized subset (the "inner loop"), then
nudge the shared weights toward the adapted weights:

    meta_weights <- meta_weights + meta_lr * (adapted_weights - meta_weights)

The speakers used here (`meta_init.speakers` in configs/lora_personalize.yaml)
are speaker_holdout TRAIN speakers — never the speaker_holdout TEST speakers
Phase 4's actual few-shot comparison evaluates on. This keeps the comparison
fair: the meta-init strategy has "seen" other dysarthric speakers' speech
patterns in aggregate, but never the specific test speaker or their data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from src.data.enrollment import load_or_build_enrollment
from src.models.whisper_lora import WhisperLoRAConfig
from src.training.personalize import Collator, SpeakerASRDataset, new_adapter
from src.utils.seed import set_seed


def _inner_loop_update(model, processor, train_df: pd.DataFrame, lr: float, steps: int, device: str) -> None:
    """A handful of plain gradient steps on one speaker's enrollment subset,
    cycling through the data if `steps` exceeds one epoch."""
    loader = DataLoader(
        SpeakerASRDataset(train_df), batch_size=min(4, len(train_df)), shuffle=True, collate_fn=Collator(processor)
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    done = 0
    while done < steps:
        for batch in loader:
            if done >= steps:
                break
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_features=input_features, labels=labels)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            done += 1
    model.eval()


def run_reptile_meta_init(
    merged_base,
    processor: WhisperProcessor,
    lora_cfg: WhisperLoRAConfig,
    meta_speakers_df: pd.DataFrame,
    speakers: list[str],
    enrollment_out_dir: Path,
    enrollment_minutes: list[float],
    outer_iterations: int,
    inner_steps: int,
    inner_lr: float,
    meta_lr: float,
    seed: int,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Returns a LoRA state dict usable as `init_state_dict` for
    `src.training.personalize.train_adapter`."""
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # One reference adapter to establish the parameter shapes / initial
    # (random) meta state before any meta-training happens.
    ref_model = new_adapter(merged_base, lora_cfg, device)
    meta_state = {k: v.clone() for k, v in get_peft_model_state_dict(ref_model).items()}
    del ref_model

    speaker_enrollments = {
        spk: load_or_build_enrollment(meta_speakers_df, spk, enrollment_out_dir, enrollment_minutes, seed)
        for spk in speakers
    }

    for it in range(outer_iterations):
        speaker = speakers[rng.integers(0, len(speakers))]
        splits = speaker_enrollments[speaker]
        enroll_keys = [k for k in splits if k.startswith("enroll_")]
        chosen_key = enroll_keys[rng.integers(0, len(enroll_keys))]
        task_df = splits[chosen_key]
        if len(task_df) == 0:
            continue

        task_model = new_adapter(merged_base, lora_cfg, device)
        set_peft_model_state_dict(task_model, meta_state)
        _inner_loop_update(task_model, processor, task_df, inner_lr, inner_steps, device)
        adapted_state = get_peft_model_state_dict(task_model)

        meta_state = {
            k: meta_state[k] + meta_lr * (adapted_state[k].to(meta_state[k].dtype) - meta_state[k])
            for k in meta_state
        }
        del task_model

        if verbose and (it + 1) % 5 == 0:
            print(f"  [meta-init] outer iter {it + 1}/{outer_iterations} (last task: {speaker}/{chosen_key})",
                  flush=True)

    return meta_state


def save_meta_state(meta_state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta_state, path)


def load_meta_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")
