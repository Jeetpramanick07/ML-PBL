# ASR Personalization

> **Superseded, kept as a development log.** For the final results, setup and demo instructions, see the [top-level README](../README.md) and [`reports/master_results_final.md`](reports/master_results_final.md).
>
> - The final pooled model is `checkpoints/generic_full`, not `generic_unseen` as stated in the phase log below.
> - Phases 5–8 (confidence fallback, TTS, etc.) are planned scope. They are not implemented results.

Few-shot, per-speaker personalization for dysarthric speech recognition. The
core contribution is a data-efficient personalization layer on top of a
dysarthria-aware Whisper base model — not a single pooled fine-tune shared
across every speaker.

The system targets the TORGO dysarthric speech corpus: it fine-tunes Whisper
generically on pooled data, then adapts a lightweight per-speaker LoRA adapter
from only a few minutes of a new speaker's enrollment audio, with a
confidence-aware fallback for low-confidence utterances and Piper TTS to close
the loop.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -e .
```

## Verify the install

```bash
pytest
```

## Phases

This project is built in phases, each adding one layer of the system on top
of the last. Details for each phase are filled in as the corresponding work
lands.

- **Phase 0 — Project Scaffolding & Environment.** Repo skeleton, dependency
  management (pyproject.toml), config system (OmegaConf), global seeding
  utility, tests. *(done — this phase)*
- **Phase 1 — Data Pipeline (TORGO).** Loading, preprocessing, and two split
  strategies: pooled (random) and speaker-holdout (unseen new users).
  *(done)*
- **Phase 2 — Baseline Evaluation (Zero-Shot Whisper).** WER/CER of
  pretrained Whisper on TORGO before any fine-tuning. *(done — see
  `reports/baseline_eval.md`: dysarthric WER 55.4% pooled / 57.5%
  speaker-holdout vs. healthy 10.7% / 10.8%, both within the expected
  literature range)*
- **Phase 3 — Generic LoRA Fine-Tuning.** A single pooled fine-tuned model,
  the "one model for everyone" baseline personalization needs to beat.
  *(done — headline model is the leakage-free `checkpoints/generic_unseen`,
  trained on the 6 speakers that are NOT in speaker_holdout_test
  (`configs/lora_generic_unseen.yaml`); see `reports/generic_unseen_eval.md`
  and `reports/generic_unseen_vs_baseline_comparison.md`. Dysarthric-only WER
  vs. the zero-shot baseline: 55.4% -> 40.1% on pooled_test (-27.6%
  relative) and **57.5% -> 48.8% on speaker_holdout_test (-15.1% relative)**
  for speakers the model has never heard. Per held-out dysarthric speaker
  (baseline -> generic): F03 mild 38.2% -> 33.1%, M05 moderate 55.3% ->
  43.2%, M04 severe 92.6% -> 80.9%; the severe speaker is still at 81% WER,
  which is the gap Phase 4 personalization targets. Best val loss 0.3850.
  Trained on a stratified 1,200-utterance subsample (150 val, 2 epochs) — a
  compute-budget scope-down; set `train_sample_size`/`val_sample_size: null`
  for the full-data result. The earlier `checkpoints/generic` model
  (`reports/generic_eval.md`) is kept only as a "speaker-seen" reference: it
  trained on pooled_train, which contains all 11 speakers including the five
  held-out test speakers (51% of its training utterances were theirs), so
  its speaker_holdout numbers (dysarthric WER 45.3%, -21.2% relative) are
  optimistic — leakage inflated the apparent gain by ~6 points of relative
  WER. Caveats: ~99% of test transcripts also occur in the training text
  (TORGO reuses a small prompt set), so part of the gain is vocabulary/
  domain adaptation rather than acoustic adaptation; on pooled_test the
  leakage-free model is no worse than the speaker-seen one (dysarthric WER
  40.1% vs 40.6%), which suggests the same. Training first ran on CPU and
  was finished on the RTX 4050 in a project-local `.venv` with CUDA torch;
  the system Python's torch is a CPU-only build.)*
- **Phase 4 — Few-Shot Personalization Framework.** Per-speaker LoRA
  adapters, naive vs. meta-initialized, evaluated across enrollment sizes.
- **Phase 5 — Confidence-Aware Decoding & Fallback.** Confidence scoring and
  candidate fallback for low-confidence predictions.
- **Phase 6 — TTS Integration & Voice Banking.** Piper TTS output and
  voice-identity preservation.
- **Phase 7 — Full Evaluation Suite & Ablations.** Master comparison table,
  significance testing, results summary.
- **Phase 8 — Real-Time Inference Pipeline.** Latency budget, benchmarking,
  streaming inference with VAD.
- **Phase 9 — Demo UI.** Local Streamlit/Gradio app for live demonstration.
- **Phase 10 — Documentation & Final Report Packaging.** Final README,
  architecture docs, limitations, locked environment.

## Configuration

Experiment configuration lives under `configs/`, with `configs/base.yaml` as
the shared default (data paths, model checkpoint, sample rate, batch size,
seed). Later phases add per-experiment configs (e.g.
`configs/lora_generic.yaml`) that override or extend the base config.
