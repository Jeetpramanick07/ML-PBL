"""FastAPI demo backend for the asr-personalization pipeline.

SCOPE: this exists to power a live PBL review/expo demo over the local
network (phone <-> laptop on the same WiFi, or USB-tethered as a fallback).
It is explicitly NOT a production service:
  - localhost-only, CORS wide open, no auth
  - one enrollment fine-tune at a time, no request queueing beyond that
  - in-memory job state (an /enroll/status lookup does not survive a
    server restart)

Consistency constraint (per the demo build plan): this API only ever uses
zero-shot Whisper, pooled LoRA, and naive per-speaker personalization — the
same capabilities the project report claims are done, not Future Scope — plus
Piper TTS with a stock voice (src/tts/piper_synth.py) for /speak. It never
does confidence-based fallback/candidate ranking, meta-learned adapter init,
or voice cloning.

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Then open http://<this-laptop's-LAN-IP>:8000/docs from any device on the
same network, or http://127.0.0.1:8000/docs locally.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.audio_io import UnreadableAudioError, cleanup_tmp, load_uploaded_audio
from src.api.enrollment import EnrollmentManager
from src.api.model_manager import ModelManager
from src.tts.piper_synth import InvalidTextError, PiperSynthesizer
from src.utils.config import load_config

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_LOG_PATH = REPORTS_DIR / "api_requests.log"
SERVER_LOG_PATH = REPORTS_DIR / "api_server.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(SERVER_LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("asr_demo.api")

_request_log_lock = threading.Lock()


def log_request(endpoint: str, **fields) -> None:
    """Append one JSON line per request to reports/api_requests.log for
    post-demo review, per Phase A's logging requirement."""
    record = {"ts": time.time(), "endpoint": endpoint, **fields}
    with _request_log_lock:
        with REQUEST_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# -- app state (populated at startup, never inside a request handler) -----------

class AppState:
    model_manager: ModelManager
    enrollment_manager: EnrollmentManager
    tts: PiperSynthesizer | None = None
    tts_unavailable_reason: str | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    logger.info("Loading models (this happens once at startup, not per-request)...")
    state.model_manager = ModelManager(model_name=cfg.model.name, language=cfg.model.language, task=cfg.model.task)
    state.enrollment_manager = EnrollmentManager(state.model_manager)
    # Never fatal: if the voice is missing/broken, /speak returns 503 and the
    # transcription flow is unaffected.
    state.tts, state.tts_unavailable_reason = PiperSynthesizer.try_load()
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        local_ip = "unknown"
    logger.info("Ready. LAN IP appears to be %s — enter this in the app's Settings screen.", local_ip)
    yield


app = FastAPI(title="ASR Personalization Demo API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- schemas -----------------------------------------------------------------

class TranscribeResponse(BaseModel):
    transcription: str
    model_used: str
    latency_ms: float


class EnrollStartResponse(BaseModel):
    job_id: str
    status: str


class EnrollStatusResponse(BaseModel):
    job_id: str
    speaker_id: str
    status: str
    elapsed_sec: float | None = None
    epochs_run: int | None = None
    final_loss: float | None = None
    checkpoint_path: str | None = None
    error: str | None = None
    note: str | None = None


# -- endpoints -----------------------------------------------------------------

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...), speaker_id: str | None = Form(None)):
    raw = await file.read()
    if not raw:
        log_request("/transcribe", speaker_id=speaker_id, error="empty upload")
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        waveform, tmp_path = load_uploaded_audio(raw, file.filename or "upload")
    except UnreadableAudioError as exc:
        log_request("/transcribe", speaker_id=speaker_id, error=str(exc))
        raise HTTPException(400, str(exc)) from exc

    try:
        if len(waveform) == 0:
            log_request("/transcribe", speaker_id=speaker_id, error="zero-length decoded audio")
            raise HTTPException(400, "Decoded audio has zero length.")
        text, model_used, latency_ms = state.model_manager.transcribe(waveform, speaker_id=speaker_id)
    finally:
        cleanup_tmp(tmp_path)

    log_request("/transcribe", speaker_id=speaker_id, model_used=model_used, latency_ms=latency_ms)
    return TranscribeResponse(transcription=text, model_used=model_used, latency_ms=latency_ms)


@app.post("/enroll/start", response_model=EnrollStartResponse)
async def enroll_start(
    speaker_id: str = Form(...),
    prompts: list[str] = Form(...),
    files: list[UploadFile] = File(...),
):
    """`prompts[i]` is the ground-truth text the speaker was asked to read
    for `files[i]` — required because fine-tuning needs a target transcript
    for every clip, not just the audio (the Flutter enrollment screen shows
    these prompts on-screen before recording each clip)."""
    if not (1 <= len(files) <= 3):
        raise HTTPException(400, f"Expected 1-3 enrollment clips, got {len(files)}.")
    if len(files) != len(prompts):
        raise HTTPException(400, f"Got {len(files)} audio files but {len(prompts)} prompts — these must pair up 1:1.")
    if not speaker_id.strip():
        raise HTTPException(400, "speaker_id must be non-empty.")

    waveforms = []
    tmp_paths = []
    try:
        for f in files:
            raw = await f.read()
            if not raw:
                raise HTTPException(400, f"Enrollment clip '{f.filename}' is empty.")
            try:
                wav, tmp_path = load_uploaded_audio(raw, f.filename or "clip")
            except UnreadableAudioError as exc:
                raise HTTPException(400, str(exc)) from exc
            waveforms.append(wav)
            tmp_paths.append(tmp_path)
    finally:
        for p in tmp_paths:
            cleanup_tmp(p)

    job_id = state.enrollment_manager.start_job(speaker_id.strip(), waveforms, prompts)
    log_request("/enroll/start", speaker_id=speaker_id, job_id=job_id, n_clips=len(files))
    return EnrollStartResponse(job_id=job_id, status="running")


@app.get("/enroll/status/{job_id}", response_model=EnrollStatusResponse)
async def enroll_status(job_id: str):
    job = state.enrollment_manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"No enrollment job with id '{job_id}' (server may have restarted since it was started).")
    return EnrollStatusResponse(
        job_id=job.job_id, speaker_id=job.speaker_id, status=job.status,
        elapsed_sec=job.elapsed_sec, epochs_run=job.epochs_run, final_loss=job.final_loss,
        checkpoint_path=job.checkpoint_path, error=job.error, note=job.note,
    )


@app.post("/speak", response_class=Response, responses={200: {"content": {"audio/wav": {}}}})
def speak(text: str = Form("")):
    """Synthesize `text` with Piper and return a WAV file. Plain `def` (not
    async) so FastAPI runs the CPU-bound synthesis in its threadpool instead
    of blocking the event loop."""
    if state.tts is None:
        log_request("/speak", error="tts unavailable")
        raise HTTPException(503, f"TTS is not available on this server: {state.tts_unavailable_reason}")
    t0 = time.time()
    try:
        wav_bytes = state.tts.synthesize(text)
    except InvalidTextError as exc:
        log_request("/speak", n_chars=len(text or ""), error=str(exc))
        raise HTTPException(400, str(exc)) from exc
    latency_ms = (time.time() - t0) * 1000
    log_request("/speak", n_chars=len(text), n_bytes=len(wav_bytes), latency_ms=latency_ms)
    return Response(content=wav_bytes, media_type="audio/wav", headers={"X-Synthesis-Ms": f"{latency_ms:.0f}"})


@app.get("/health")
async def health():
    return {
        **state.model_manager.health(),
        "tts_available": state.tts is not None,
        "tts_voice": state.tts.voice_path.name if state.tts else None,
        "tts_unavailable_reason": state.tts_unavailable_reason,
    }
