"""Phase 3: generic (pooled) LoRA fine-tune of Whisper.

Fine-tunes a fresh LoRA adapter on top of the pretrained Whisper checkpoint
using pooled_train.csv / pooled_val.csv, with standard Whisper seq2seq
cross-entropy loss. This is the "one model for everyone" baseline that
Phase 4's per-speaker personalization needs to beat, especially on
speaker_holdout_test.

Checkpointing/resume: the adapter, optimizer state, and trainer state
(epoch, global_step, best_val_loss) are saved to `<checkpoints_dir>/last`
every `save_every_steps` steps and at the end of every epoch, and to
`<checkpoints_dir>/best` whenever val loss improves. Re-running with
--resume continues from `<checkpoints_dir>/last`. This matters on CPU, where
a full run can take hours and an interrupted run shouldn't lose all progress.

Usage:
    python -m src.training.train_generic
    python -m src.training.train_generic --max-train-samples 32 --max-steps 4  # smoke test
    python -m src.training.train_generic --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor, get_linear_schedule_with_warmup

from src.data.preprocessing import TARGET_SAMPLE_RATE, load_and_clean_audio, normalize_transcription
from src.evaluation.metrics import compute_wer_cer
from src.models.whisper_lora import WhisperLoRAConfig, apply_lora, load_base_model
from src.utils.config import load_experiment_config
from src.utils.seed import set_seed

CONFIG_PATH = PROJECT_ROOT / "configs" / "lora_generic.yaml"


def subsample_stratified(df: pd.DataFrame, n: int, seed: int, group_col: str = "dysarthria_status") -> pd.DataFrame:
    """Sample n rows from df, stratified by group_col so the sampled subset
    keeps roughly the same healthy:dysarthric ratio as the full split."""
    if n >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    parts = []
    for _, group in df.groupby(group_col, group_keys=False):
        take = max(1, round(len(group) * frac))
        idx = rng.choice(group.index.to_numpy(), size=min(take, len(group)), replace=False)
        parts.append(group.loc[idx])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class TorgoASRDataset(Dataset):
    """Loads audio + normalized transcription for one manifest CSV, lazily
    (audio is read from disk on __getitem__, not pre-loaded into memory)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav, sr = load_and_clean_audio(row.audio_path, target_sr=TARGET_SAMPLE_RATE, trim_silence=False)
        if wav is None or len(wav) == 0:
            # Corrupt file slipped through Phase 1 filtering; return silence
            # rather than crashing a multi-hour training run over one bad row.
            wav = np.zeros(TARGET_SAMPLE_RATE // 10, dtype=np.float32)
        return {"waveform": wav, "text": normalize_transcription(str(row.transcription))}


class Collator:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        waveforms = [b["waveform"] for b in batch]
        texts = [b["text"] for b in batch]

        feats = self.processor.feature_extractor(
            waveforms, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        labels = self.processor.tokenizer(
            texts, return_tensors="pt", padding=True
        )
        label_ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)

        return {"input_features": feats.input_features, "labels": label_ids}


def build_optimizer_and_schedule(model, lr: float, warmup_steps: int, total_steps: int):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_steps, 1)
    )
    return optimizer, scheduler


def evaluate_loss(model, loader: DataLoader, device: str, max_batches: int | None = None) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_features=input_features, labels=labels)
            total_loss += out.loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_wer(
    model, processor: WhisperProcessor, df: pd.DataFrame, device: str,
    batch_size: int, max_new_tokens: int = 64, language: str = "en", task: str = "transcribe",
) -> float:
    """Greedy-decode the whole of `df` and return corpus WER (normalized,
    same normalize_transcription used everywhere else). This is what actually
    drives early stopping/best-checkpoint selection when
    training.checkpoint_metric == "wer" — val LOSS (teacher-forced
    next-token loss) can keep improving well past the point where greedy
    decoding quality plateaus or regresses, so it's a weaker proxy for the
    metric this project is actually evaluated on."""
    model.eval()
    refs, hyps = [], []
    paths = df["audio_path"].tolist()
    texts = df["transcription"].astype(str).tolist()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_refs = texts[start : start + batch_size]
        waveforms, valid_refs = [], []
        for p, r in zip(batch_paths, batch_refs):
            wav, _ = load_and_clean_audio(p, target_sr=TARGET_SAMPLE_RATE, trim_silence=False)
            if wav is not None and len(wav) > 0:
                waveforms.append(wav)
                valid_refs.append(r)
        if not waveforms:
            continue
        inputs = processor(waveforms, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        generated_ids = model.generate(
            inputs.input_features.to(device), language=language, task=task, max_new_tokens=max_new_tokens,
        )
        batch_hyps = [t.strip() for t in processor.batch_decode(generated_ids, skip_special_tokens=True)]
        hyps.extend(batch_hyps)
        refs.extend(valid_refs)
    model.train()
    return compute_wer_cer(refs, hyps).wer


def save_checkpoint(path: Path, model, optimizer, scheduler, state: dict, scaler=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "state": state,
        },
        path / "trainer_state.pt",
    )


def load_trainer_state(path: Path) -> dict | None:
    state_path = path / "trainer_state.pt"
    if not state_path.exists():
        return None
    return torch.load(state_path, map_location="cpu")


def append_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Smoke test: cap train set size")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Smoke test: cap val set size")
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke test: stop after N optimizer steps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from <checkpoints_dir>/last")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Experiment config (merged onto configs/base.yaml)")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    set_seed(cfg.seed)

    batch_size = args.batch_size or cfg.training.batch_size
    num_epochs = args.num_epochs or cfg.training.num_epochs
    lr = args.learning_rate or cfg.training.learning_rate
    save_every_steps = cfg.training.save_every_steps
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fp16 autocast + GradScaler. Measured on this project's GPU (RTX 4050,
    # 6GB): plain fp32 ran ~15-26s/batch-of-8 (a full pooled_train epoch
    # would take on the order of 5+ hours), fp16 autocast cut that to
    # ~2-3s/batch — an ~8x speedup from a standard, widely-used optimization,
    # not a change to the model or a result-affecting hyperparameter. Default
    # off so every existing config's exact prior behavior (and CPU runs,
    # where autocast('cuda') isn't applicable) is unchanged.
    mixed_precision = bool(cfg.training.get("mixed_precision", False)) and device == "cuda"
    if cfg.training.get("mixed_precision", False) and device != "cuda":
        print("training.mixed_precision=true but device is cpu — ignoring (autocast('cuda') needs a GPU).")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    splits_dir = Path(args.splits_dir)
    # training.train_file lets a config point at a subsample (e.g.
    # pooled_train_subsample.csv) instead of the full pooled_train.csv, for a
    # documented compute-budget trade-off — validation always still reads
    # pooled_val.csv in full; only training data volume is ever reduced here.
    train_file = cfg.training.get("train_file", "pooled_train.csv")
    train_df = pd.read_csv(splits_dir / train_file)
    val_df = pd.read_csv(splits_dir / "pooled_val.csv")

    # Optionally drop every utterance from the speakers of another split (e.g.
    # speaker_holdout_test) so those speakers are genuinely unseen. pooled_train
    # is a random utterance-level split over ALL speakers, so without this the
    # "held-out" test speakers leak into generic training.
    exclude_from = cfg.training.get("exclude_speakers_of")
    if exclude_from:
        excluded = set(pd.read_csv(splits_dir / f"{exclude_from}.csv")["speaker_id"].unique())
        n_train_before, n_val_before = len(train_df), len(val_df)
        train_df = train_df[~train_df["speaker_id"].isin(excluded)].reset_index(drop=True)
        val_df = val_df[~val_df["speaker_id"].isin(excluded)].reset_index(drop=True)
        print(
            f"Excluding speakers of {exclude_from}: {sorted(excluded)}; "
            f"train {n_train_before}->{len(train_df)}, val {n_val_before}->{len(val_df)}; "
            f"remaining train speakers: {sorted(train_df['speaker_id'].unique())}"
        )

    train_sample_size = cfg.training.get("train_sample_size")
    val_sample_size = cfg.training.get("val_sample_size")
    if train_sample_size:
        train_df = subsample_stratified(train_df, train_sample_size, cfg.seed)
    if val_sample_size:
        val_df = subsample_stratified(val_df, val_sample_size, cfg.seed)

    if args.max_train_samples:
        train_df = train_df.head(args.max_train_samples)
    if args.max_val_samples:
        val_df = val_df.head(args.max_val_samples)

    print(f"train={len(train_df)} val={len(val_df)} device={device} batch_size={batch_size} num_epochs={num_epochs}")

    processor = WhisperProcessor.from_pretrained(cfg.model.name, language=cfg.model.language, task=cfg.model.task)
    base_model = load_base_model(cfg.model.name)
    lora_cfg = WhisperLoRAConfig(
        r=cfg.lora.r, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
    )
    model = apply_lora(base_model, lora_cfg)
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    collator = Collator(processor)
    train_loader = DataLoader(
        TorgoASRDataset(train_df), batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        TorgoASRDataset(val_df), batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
    )

    total_steps = len(train_loader) * num_epochs
    optimizer, scheduler = build_optimizer_and_schedule(model, lr, cfg.training.warmup_steps, total_steps)

    checkpoints_dir = Path(cfg.checkpoints_dir)
    last_dir = checkpoints_dir / "last"
    best_dir = checkpoints_dir / "best"
    log_path = Path(cfg.get("train_log_path") or Path(cfg.reports_dir) / "generic_train_log.csv")
    summary_path = Path(cfg.get("train_summary_path") or Path(cfg.reports_dir) / f"{checkpoints_dir.name}_train_summary.json")

    # "loss" (default) matches every existing config's prior behavior
    # unchanged. "wer" checkpoints/early-stops on greedy-decoded validation
    # WER instead — what a new full-data run with early stopping should
    # actually optimize for, since that's the metric this whole project is
    # evaluated on.
    checkpoint_metric = cfg.training.get("checkpoint_metric", "loss")
    early_stopping_patience = cfg.training.get("early_stopping_patience")
    max_wall_clock_seconds = cfg.training.get("max_wall_clock_seconds")
    assert checkpoint_metric in ("loss", "wer"), f"unknown checkpoint_metric {checkpoint_metric!r}"

    start_epoch, global_step = 0, 0
    best_val_loss, best_val_wer = float("inf"), float("inf")
    best_epoch = 0
    epochs_since_improvement = 0
    prior_wall_clock = 0.0
    run_t0 = time.time()
    if args.resume:
        ckpt = load_trainer_state(last_dir)
        if ckpt is not None:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            adapter_weights = load_file(last_dir / "adapter_model.safetensors")
            set_peft_model_state_dict(model, adapter_weights)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["state"]["epoch"]
            global_step = ckpt["state"]["global_step"]
            best_val_loss = ckpt["state"]["best_val_loss"]
            best_val_wer = ckpt["state"].get("best_val_wer", float("inf"))
            best_epoch = ckpt["state"].get("best_epoch", 0)
            epochs_since_improvement = ckpt["state"].get("epochs_since_improvement", 0)
            prior_wall_clock = ckpt["state"].get("cumulative_wall_clock_seconds", 0.0)
            print(f"Resumed from {last_dir}: epoch={start_epoch} global_step={global_step} "
                  f"best_val_loss={best_val_loss:.4f} best_val_wer={best_val_wer:.4f} "
                  f"best_epoch={best_epoch} epochs_since_improvement={epochs_since_improvement} "
                  f"prior_wall_clock={prior_wall_clock:.1f}s")
        else:
            print(f"--resume given but no checkpoint found at {last_dir}; starting fresh")

    stop_early = False
    stopped_reason = None
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                out = model(input_features=input_features, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 10 == 0 or n_batches == 1:
                print(f"  epoch {epoch} step {global_step} loss {loss.item():.4f}", flush=True)

            if save_every_steps and global_step % save_every_steps == 0:
                save_checkpoint(
                    last_dir, model, optimizer, scheduler,
                    {
                        "epoch": epoch, "global_step": global_step,
                        "best_val_loss": best_val_loss, "best_val_wer": best_val_wer,
                        "best_epoch": best_epoch, "epochs_since_improvement": epochs_since_improvement,
                    },
                    scaler=scaler,
                )
                print(f"  checkpointed to {last_dir} at step {global_step}")

            if args.max_steps and global_step >= args.max_steps:
                stop_early = True
                stopped_reason = "max_steps"
                break

        train_loss = running_loss / max(n_batches, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        # Always computed (not just when it drives checkpointing) so the log
        # shows both metrics regardless of which one this run is optimizing.
        val_wer = evaluate_wer(
            model, processor, val_df, device, batch_size=batch_size,
            language=cfg.model.language, task=cfg.model.task,
        )
        elapsed = time.time() - t0

        append_log(log_path, {
            "epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
            "val_wer": round(val_wer, 4), "elapsed_seconds": round(elapsed, 1), "global_step": global_step,
        })
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_wer={val_wer:.4f} ({elapsed:.1f}s)")

        improved = (val_wer < best_val_wer) if checkpoint_metric == "wer" else (val_loss < best_val_loss)
        best_val_loss = min(best_val_loss, val_loss)
        best_val_wer = min(best_val_wer, val_wer)
        # Update best_epoch/patience BEFORE saving `last`, so a resume sees
        # this epoch's outcome (previously `last` lagged by one epoch, which
        # would silently grant early stopping an extra epoch of patience).
        if improved:
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        state = {
            "epoch": epoch + 1, "global_step": global_step,
            "best_val_loss": best_val_loss, "best_val_wer": best_val_wer,
            "best_epoch": best_epoch, "epochs_since_improvement": epochs_since_improvement,
            "cumulative_wall_clock_seconds": prior_wall_clock + (time.time() - run_t0),
        }
        save_checkpoint(last_dir, model, optimizer, scheduler, state, scaler=scaler)

        if improved:
            save_checkpoint(best_dir, model, optimizer, scheduler, state, scaler=scaler)
            print(f"  new best {checkpoint_metric} "
                  f"(val_loss={val_loss:.4f}, val_wer={val_wer:.4f}), saved to {best_dir}")
        else:
            print(f"  no improvement in {checkpoint_metric} for {epochs_since_improvement} epoch(s)")

        if stop_early:
            print(f"Stopping early: {stopped_reason}")
            break

        if early_stopping_patience and epochs_since_improvement >= early_stopping_patience:
            stop_early = True
            stopped_reason = (
                f"early stopping: no {checkpoint_metric} improvement for "
                f"{epochs_since_improvement} consecutive epochs (patience={early_stopping_patience})"
            )
            print(f"Stopping early: {stopped_reason}")
            break

        if max_wall_clock_seconds and prior_wall_clock + (time.time() - run_t0) >= max_wall_clock_seconds:
            stop_early = True
            stopped_reason = (
                f"TIME-CAPPED, NOT CONVERGED: hit the {max_wall_clock_seconds/3600:.1f}h wall-clock cap "
                f"after epoch {epoch} without early stopping having triggered "
                f"(best {checkpoint_metric} was at epoch {best_epoch}, "
                f"{epochs_since_improvement} epoch(s) since improvement)"
            )
            print(f"Stopping: {stopped_reason}")
            break

    total_epochs_run = epoch - start_epoch + 1
    # Cumulative across resumes, so a paused-and-resumed run is held to the
    # same wall-clock cap as an uninterrupted one.
    wall_clock = prior_wall_clock + (time.time() - run_t0)
    time_capped = bool(max_wall_clock_seconds and wall_clock >= max_wall_clock_seconds and "TIME-CAPPED" in (stopped_reason or ""))
    print(
        f"Training complete. total_epochs_run={total_epochs_run} best_epoch={best_epoch} "
        f"best_val_loss={best_val_loss:.4f} best_val_wer={best_val_wer:.4f} "
        f"stopped_early={stop_early} ({stopped_reason}) wall_clock={wall_clock:.1f}s. "
        f"Adapters at {checkpoints_dir}"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "checkpoints_dir": str(checkpoints_dir),
        "train_file": train_file,
        "checkpoint_metric": checkpoint_metric,
        "early_stopping_patience": early_stopping_patience,
        "max_wall_clock_seconds": max_wall_clock_seconds,
        "num_epochs_budget": num_epochs,
        "total_epochs_run": total_epochs_run,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_wer": best_val_wer,
        "stopped_early": bool(stop_early and stopped_reason != "max_steps"),
        "time_capped_not_converged": time_capped,
        "stopped_reason": stopped_reason,
        "wall_clock_seconds": wall_clock,
        "train_n": len(train_df),
        "val_n": len(val_df),
        "batch_size": batch_size,
    }, indent=2), encoding="utf-8")
    print(f"Wrote training summary to {summary_path}")


if __name__ == "__main__":
    main()
