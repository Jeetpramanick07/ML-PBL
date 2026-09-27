"""Background per-speaker LoRA fine-tuning for the live enrollment demo.

This is deliberately a *simplified, time-boxed* variant of
src.training.personalize's "naive" strategy (see that module's docstring),
not a call into it directly — Phase 4's real pipeline trains from CSV
manifests with a DataLoader, scheduler, and held-out eval split, sized for
1-10 minutes of enrollment audio. A live demo enrollment is 1-3 short clips
(a few seconds total), so this module trains directly on in-memory waveforms
with a single full-batch step per "epoch" and a hard wall-clock cap, and
skips warmup/scheduling and eval entirely — there usually isn't enough
enrollment audio left over to hold anything out.

Per Phase A's scope constraint, this always uses the "naive" (random LoRA
init) strategy — never the meta-learned initialization from
src.training.meta_init, which is explicitly Future Scope for this app.

If the capped budget doesn't produce a useful fine-tune (very likely for a
single 2-second clip), that's an honest, expected demo limitation, not a
bug — the job result reports final training loss and epochs actually run so
that's visible rather than hidden.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.api.model_manager import DEMO_PERSONALIZED_DIR, ModelManager
from src.data.preprocessing import normalize_transcription
from src.models.whisper_lora import WhisperLoRAConfig, apply_lora

logger = logging.getLogger("asr_demo.enrollment")

# Per Phase A: "hard time budget (e.g. 60-90 seconds of training)". Also cap
# epochs so a fast GPU doesn't just keep hammering 1-3 clips indefinitely
# within the time budget and overfit harder than the demo needs.
MAX_TRAIN_SECONDS = 75.0
MAX_EPOCHS = 30
LORA_CFG = WhisperLoRAConfig(r=4, alpha=8, dropout=0.05)  # matches configs/lora_personalize.yaml's per-speaker rank
LEARNING_RATE = 5.0e-4


@dataclass
class EnrollmentJob:
    job_id: str
    speaker_id: str
    status: str = "running"  # "running" | "done" | "failed"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    elapsed_sec: float | None = None
    epochs_run: int | None = None
    final_loss: float | None = None
    checkpoint_path: str | None = None
    error: str | None = None
    note: str | None = None


class EnrollmentManager:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self._jobs: dict[str, EnrollmentJob] = {}
        # Enrollment fine-tunes run one at a time — this is a single-laptop
        # demo backend, and serializing keeps GPU memory/time predictable
        # instead of racing multiple background trains against live /transcribe
        # requests and each other.
        self._train_lock = threading.Lock()

    def start_job(self, speaker_id: str, waveforms: list[np.ndarray], texts: list[str]) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = EnrollmentJob(job_id=job_id, speaker_id=speaker_id)
        self._jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job, waveforms, texts), daemon=True)
        thread.start()
        return job_id

    def get(self, job_id: str) -> EnrollmentJob | None:
        return self._jobs.get(job_id)

    def _run(self, job: EnrollmentJob, waveforms: list[np.ndarray], texts: list[str]) -> None:
        with self._train_lock:
            try:
                model, stats = self._train(waveforms, texts)
                out_dir = DEMO_PERSONALIZED_DIR / job.speaker_id
                out_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(out_dir))

                self.model_manager.register_personalized_adapter(job.speaker_id, str(out_dir))

                job.status = "done"
                job.epochs_run = stats["epochs_run"]
                job.final_loss = stats["final_loss"]
                job.checkpoint_path = str(out_dir)
                if stats["epochs_run"] < 3:
                    job.note = (
                        "Time budget was hit almost immediately (very short enrollment audio) — "
                        "this adapter is unlikely to have converged to anything useful. Expected "
                        "for a single very short clip; try 2-3 slightly longer clips."
                    )
                logger.info(
                    "Enrollment job %s (speaker=%s) done: %d epochs, final_loss=%.4f, saved to %s",
                    job.job_id, job.speaker_id, stats["epochs_run"], stats["final_loss"] or float("nan"), out_dir,
                )
            except Exception as exc:  # noqa: BLE001 - report to the polling client, don't crash the server
                job.status = "failed"
                job.error = str(exc)
                logger.exception("Enrollment job %s (speaker=%s) failed", job.job_id, job.speaker_id)
            finally:
                job.finished_at = time.time()
                job.elapsed_sec = job.finished_at - job.started_at

    def _train(self, waveforms: list[np.ndarray], texts: list[str]) -> tuple[torch.nn.Module, dict]:
        mm = self.model_manager
        device = mm.device

        model = apply_lora(mm.new_personalization_base(), LORA_CFG)
        model.to(device)
        model.train()

        norm_texts = [normalize_transcription(t) for t in texts]
        feats = mm.processor.feature_extractor(waveforms, sampling_rate=16_000, return_tensors="pt")
        labels = mm.processor.tokenizer(norm_texts, return_tensors="pt", padding=True)
        label_ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)

        input_features = feats.input_features.to(device)
        label_ids = label_ids.to(device)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

        t0 = time.time()
        epochs_run = 0
        final_loss = None
        for _ in range(MAX_EPOCHS):
            if time.time() - t0 > MAX_TRAIN_SECONDS:
                break
            out = model(input_features=input_features, labels=label_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            final_loss = out.loss.item()
            epochs_run += 1

        model.eval()
        return model, {"epochs_run": epochs_run, "elapsed_sec": time.time() - t0, "final_loss": final_loss}
