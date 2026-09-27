# Personalized ASR for Dysarthric Speech

Speech recognition for people with dysarthria (motor-speech disorders that make speech slurred, slow or effortful), built by adapting OpenAI Whisper with lightweight **LoRA adapters**:

1. **Pooled adaptation.** One small adapter (0.36% of Whisper-small's parameters) fine-tuned on pooled dysarthric and healthy speech from the TORGO corpus.
2. **Per-speaker personalization.** An extra adapter per speaker, trained on only **5 minutes** of that speaker's audio.
3. **Live demo.** A FastAPI backend serves both, plus Piper text-to-speech. A Flutter Android app records speech on a phone and shows the transcription and which model produced it.

## Final results

Dysarthric speech, measured on the **clean subset of the held-out test set** (utterances never seen in training or validation):

| model | dysarthric WER | healthy WER |
|---|---|---|
| Whisper-small, zero-shot | **57.52%** | 10.90% |
| + pooled LoRA (final model) | **31.55%** | 4.70% |
| relative reduction | **−45.1%** | −56.9% |

**Per-speaker personalization** (5 min of enrollment audio, clean rows, about 170 test utterances per speaker, single seed):

| speaker | pooled only | + personalization | change |
|---|---|---|---|
| M05 (moderate) | 29.1% | 23.5% | improved |
| M04 (severe) | 56.1% | 52.8% | improved (CER got worse) |
| F03 (mild) | 24.2% | 23.5% | within noise |

WER improved for 2 of 3 speakers. These are small-sample, indicative results, not definitive ones.

**Read before quoting these numbers:**
- **Not speaker-independent.** The splits are utterance-level. Test utterances were never trained on, but the model has heard these speakers' voices on *other* utterances.
- **Test/train overlap was found and removed.** 32% of the nominal speaker-held-out set overlapped with training audio. The table above uses only the clean subset. The full-set figure (23.3%) is inflated and shouldn't be quoted.
- **Compute-capped training.** The pooled model was trained on a stratified 40% subsample (3,213 of 8,032 utterances) and stopped by a 4.5 h compute cap after 7 epochs, while still improving.

Full methodology, per-speaker tables and every caveat: [`asr-personalization/reports/master_results_final.md`](asr-personalization/reports/master_results_final.md).

## Repository layout

```
asr-personalization/          Python: data pipeline, training, evaluation, demo API
  src/
    data/                     TORGO loading, preprocessing, pooled + speaker-holdout splits
    models/  training/        Whisper + LoRA, pooled training, per-speaker personalization
    evaluation/  inference/   WER/CER metrics, report helpers, batched inference
    api/                      FastAPI demo backend (/health, /transcribe, /enroll, /speak)
    tts/                      Piper text-to-speech (piper_synth.py)
  scripts/                    download_torgo, build_dataset, eval + personalization runners,
                              leakage_breakdown (train/test overlap audit)
  configs/                    YAML experiment configs (lora_generic_full.yaml = final model)
  checkpoints/
    generic_full/best/        final pooled LoRA adapter (epoch 6, val WER 16.30%)
    personalized_final/       per-speaker adapters for F03, M04, M05
  reports/                    results: master_results_final.md, eval reports, training logs
  docs/DEMO_RUNBOOK.md        live demo script and talking points
  tests/
asr_demo_app/                 Flutter Android client (records audio, calls the backend)
docs/BUILD_PLAN.md            original project build plan
```

## Dataset: TORGO (not included)

This repo contains **no audio, transcripts or dataset-derived manifests**. TORGO is a licensed research corpus with its own terms of use. Download it yourself from the official source:

**http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html**

Review its usage terms first. The four archives (F, FC, M, MC) total about 9.5 GB. Then, from `asr-personalization/`:

```bash
python scripts/download_torgo.py        # downloads + extracts to data/raw/torgo/
                                        # (or extract the archives there manually)
python scripts/build_dataset.py         # builds data/processed/splits/*.csv (seed 42)
```

The final model was trained on a stratified 40% subsample. This command regenerates it exactly:

```bash
python -c "import pandas as pd; from src.training.train_generic import subsample_stratified; df = pd.read_csv('data/processed/splits/pooled_train.csv'); subsample_stratified(df, 3213, 42).to_csv('data/processed/splits/pooled_train_subsample.csv', index=False)"
```

**You don't need TORGO to run the demo.** The trained adapters are committed, so the backend and app work without the dataset.

## Running the backend

Requires Python ≥ 3.10. A CUDA GPU is recommended; it was developed on a 6 GB RTX 4050. CPU works but is slow.

```bash
cd asr-personalization
python -m venv .venv
.venv\Scripts\activate                  # macOS/Linux: source .venv/bin/activate
# For GPU, install the CUDA build of torch first: https://pytorch.org/get-started/locally/
pip install -e .
# Piper TTS voice (~63 MB, not committed) — needed only for /speak:
python -m piper.download_voices --download-dir models/piper en_US-lessac-medium
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

If the voice model is missing, the server still starts. `/health` shows `"tts_available": false`, and `/speak` returns 503. Transcription is unaffected.

On first start, `openai/whisper-small` is downloaded from Hugging Face. Then check `http://127.0.0.1:8000/health`. It should report:
- `"model_used_label": "pooled-lora:generic_full"`
- personalized profiles **F03, M04, M05**, each with `"source": "pretrained"`

API docs are at `http://127.0.0.1:8000/docs`. Example comparison (needs the TORGO audio locally):

```bash
curl -F "file=@data/raw/torgo/M04/Session2/wav_headMic/0092.wav" http://127.0.0.1:8000/transcribe
curl -F "file=@data/raw/torgo/M04/Session2/wav_headMic/0092.wav" -F "speaker_id=M04" http://127.0.0.1:8000/transcribe
# reference "gadget": pooled -> "gatish", personalized:M04 -> "gadget"

# Text-to-speech (Piper): returns a 22.05 kHz mono WAV
curl -F "text=I would like a glass of water." http://127.0.0.1:8000/speak -o speech.wav
```

The backend is a **local-network demo server**: plain HTTP, no auth, CORS open. Don't expose it to the internet.

## Building the Android app

Requires the Flutter SDK (tested with 3.47), Android SDK platform 36, NDK 28.2 and **JDK 17**. Newer JDKs such as 25 are too new for the Android Gradle toolchain.

```bash
cd asr_demo_app
flutter config --jdk-dir <path-to-jdk17>     # if JDK 17 isn't your default
flutter pub get
flutter analyze
flutter build apk --release                  # -> build/app/outputs/flutter-apk/app-release.apk
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

A prebuilt APK is attached to this repo's **GitHub Releases** page, not committed to git.

**Signing:** the APK is signed with the Flutter **debug key**. That's fine for installing directly on a demo phone. It's **not suitable for Google Play or any store distribution**, which needs a proper release keystore and signing config.

**Using the app:**
1. Start the backend on a laptop and find that laptop's LAN IP with `ipconfig` (use the Wi-Fi adapter's IPv4 address).
2. On the phone, on the same Wi-Fi, open **ASR Demo** → Settings and enter `<laptop-ip>:8000`.
3. Record speech. The `model:` line shows `pooled-lora:generic_full` or `personalized:<speaker>`.

See [`asr_demo_app/README.md`](asr_demo_app/README.md) for details.

## Demo

[`asr-personalization/docs/DEMO_RUNBOOK.md`](asr-personalization/docs/DEMO_RUNBOOK.md) has the live-demo script, the real-number talking points, prepared answers to methodology questions, and the pre-demo checklist.

## Reproducing the results

| step | command (from `asr-personalization/`) |
|---|---|
| zero-shot baseline | `python scripts/run_baseline_eval.py` |
| pooled LoRA training | `python -m src.training.train_generic --config configs/lora_generic_full.yaml` |
| pooled evaluation | `python scripts/eval_generic.py --adapter-dir checkpoints/generic_full/best --tag generic_full` |
| personalization | `python scripts/run_personalization_experiment.py --config configs/lora_personalize_final.yaml --strategies naive` |
| train/test overlap audit | `python scripts/leakage_breakdown.py` |

Per-utterance hypothesis files are git-ignored because they contain TORGO transcripts. The scripts above regenerate them locally.
