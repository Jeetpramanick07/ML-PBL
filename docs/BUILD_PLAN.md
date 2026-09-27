# Build Plan: Few-Shot Personalized ASR for Dysarthric Speech

**Project:** Adaptive Speech Recognition System for Speech-Impaired Users
**Core contribution:** data-efficient per-speaker personalization on top of a dysarthria-aware Whisper base, not just a single pooled fine-tune.

## How to use this document

Each phase below is a **separate Claude Code session prompt**. Paste the whole "PROMPT" block verbatim as your first message in a fresh Claude Code session (or a fresh conversation turn if continuing in the same repo). Do not skip ahead — later phases assume earlier ones produced the exact file structure and artifacts listed under "Deliverables." After each phase, verify the "Definition of Done" checklist yourself before starting the next one. Don't let Claude Code silently expand scope into a later phase — the "Do NOT" list in each prompt exists to keep sessions bounded and reviewable.

Recommended repo layout (Phase 0 creates this; every later phase writes into it):

```
asr-personalization/
├── data/
│   ├── raw/                 # untouched TORGO download
│   └── processed/           # cleaned, resampled, split
├── src/
│   ├── data/                # loaders, preprocessing, splitting
│   ├── models/              # Whisper wrapper, LoRA adapter logic
│   ├── training/            # base fine-tune, personalization loop
│   ├── evaluation/          # WER/CER, adaptation-curve, ablations
│   ├── inference/           # decoding, confidence fallback, real-time pipeline
│   ├── tts/                 # Piper integration, voice banking
│   └── ui/                  # demo app
├── configs/                 # yaml configs per experiment
├── scripts/                 # thin CLI entry points calling src/
├── reports/                 # generated metrics, plots, tables
└── tests/
```

---

## Phase 0 — Project Scaffolding & Environment

**Goal:** reproducible repo skeleton, dependency management, config system — nothing ML-specific yet.

**PROMPT:**
```
Set up a new Python project called asr-personalization for a dysarthric-speech ASR
personalization research project. Requirements:

1. Create the directory structure: data/{raw,processed}, src/{data,models,training,
   evaluation,inference,tts,ui}, configs/, scripts/, reports/, tests/. Add __init__.py
   where needed for src to be an installable package.
2. Use pyproject.toml (not requirements.txt) with dependencies: torch, transformers,
   peft (for LoRA), datasets, librosa, soundfile, jiwer (WER/CER), hydra-core or
   omegaconf for config management, pytest, matplotlib, pandas, numpy.
3. Set up a config system (Hydra or plain OmegaConf YAML) with a base config at
   configs/base.yaml covering: data paths, model name (openai/whisper-small as
   default, medium as an override), sample rate, batch size, seed.
4. Add a src/utils/seed.py that sets seed everywhere (torch, numpy, python random)
   for reproducibility — every later phase's experiments must be seeded.
5. Add a README.md with project description, setup instructions (venv/conda + pip
   install -e .), and a placeholder "Phases" section listing Phase 0 through Phase 10
   (I will fill in details as we go).
6. Add a .gitignore appropriate for Python + data science (data/, *.ckpt, __pycache__,
   .venv, wandb/, reports/*.png).
7. Write one trivial pytest (e.g. test that config loads) so `pytest` passes from a
   clean checkout.

Do NOT write any data loading, model, or training code yet — that's later phases.
Do NOT pick a specific Whisper checkpoint size beyond making it configurable.
When done, run pytest and show me it passes, and show me the full directory tree.
```

**Deliverables:** working repo skeleton, `pip install -e .` succeeds, `pytest` passes, config loads.

**Definition of Done:** you can `cd asr-personalization && pip install -e . && pytest` on a clean machine with no errors.

---

## Phase 1 — Data Pipeline (TORGO)

**Goal:** deterministic, leakage-free data pipeline with the splits your whole thesis depends on.

**Critical design decision to lock in here:** you need **two different split strategies**, not one:
- A **pooled split** (random utterance-level train/val/test) — used for Phase 3's baseline pooled fine-tune, matching what you already did in the earlier version of this project.
- A **speaker-held-out split** — entire speakers reserved as "unseen new users" — this is what makes Phase 4's personalization claim testable at all. If you skip this, you cannot demonstrate few-shot adaptation to a truly new speaker.

**PROMPT:**
```
Build the TORGO data pipeline in src/data/. TORGO is already downloaded at
data/raw/torgo/ (assume standard TORGO directory structure: per-speaker folders,
each with wav_arrayMic or wav_headMic audio and a prompts/transcription folder).
If the raw data isn't there yet, write a scripts/download_torgo.py that fetches it
and document the manual steps in README if the source requires manual download.

Implement:
1. src/data/torgo_loader.py — parses TORGO into a pandas DataFrame / HuggingFace
   Dataset with columns: speaker_id, audio_path, transcription, dysarthria_status
   (healthy/dysarthric), severity (if available in TORGO metadata), gender, duration.
2. src/data/preprocessing.py — resample to 16kHz mono, trim silence, filter out
   corrupt/too-short/too-long clips (make thresholds configurable), normalize
   transcriptions (case, punctuation) consistently with what Whisper expects.
3. src/data/splits.py implementing TWO split strategies, both saved as CSV manifests
   under data/processed/splits/:
   a) pooled_{train,val,test}.csv — random utterance-level split, stratified by
      dysarthria_status so train/val/test have similar healthy:dysarthric ratios.
   b) speaker_holdout_{train,val,test}.csv — entire speakers assigned to test
      (the "new user" simulation set), disjoint speakers in train/val. Pick enough
      held-out dysarthric speakers to cover a range of severities if severity data
      exists.
4. A script scripts/build_dataset.py that runs the full pipeline end-to-end and
   prints a summary table: total samples, per-split counts, per-split healthy/
   dysarthric counts, per-split average duration — so I can sanity-check against
   the known totals (~16,552 total, ~5,574 dysarthric, ~10,978 healthy from the
   original TORGO stats).
5. Unit tests in tests/test_data.py checking: no speaker appears in both train and
   test of the speaker_holdout split, no duplicate audio paths across splits, all
   manifest audio files actually exist on disk.

Do NOT touch any model or training code. Do NOT decide the personalization
methodology yet (that's Phase 4) — just make sure speaker_holdout test speakers
are cleanly reserved for it.
Run the pipeline and show me the summary table and passing tests.
```

**Deliverables:** `data/processed/splits/*.csv`, passing `tests/test_data.py`, a printed summary table.

**Definition of Done:** summary table roughly matches TORGO's known composition; zero speaker overlap between speaker-holdout train and test; tests pass.

---

## Phase 2 — Baseline Evaluation (Zero-Shot Whisper)

**Goal:** the "before" numbers that justify the entire project. Do this before any fine-tuning.

**PROMPT:**
```
Implement baseline (zero-shot, no fine-tuning) evaluation of pretrained Whisper on
TORGO in src/evaluation/.

1. src/evaluation/metrics.py — WER and CER computation using jiwer, with a
   normalization function applied consistently to both reference and hypothesis
   text before scoring (lowercase, strip punctuation) — document exactly what
   normalization is applied since this affects reported numbers.
2. src/inference/whisper_infer.py — loads a pretrained Whisper checkpoint (size
   from config, default whisper-small) via transformers and runs batched inference
   over a manifest CSV, returning hypotheses.
3. scripts/run_baseline_eval.py — runs zero-shot Whisper over the pooled_test.csv
   and speaker_holdout_test.csv splits separately, and breaks results down by:
   overall, healthy-only, dysarthric-only, and (if severity metadata exists)
   per-severity-level. Save results to reports/baseline_eval.json and a
   human-readable reports/baseline_eval.md table.
4. Also break down results per individual held-out speaker in the speaker_holdout
   test set (a small table: speaker_id, n_samples, WER, CER) — this is the
   per-speaker baseline you'll compare every later personalization result against.

Do NOT fine-tune anything in this phase. Do NOT write LoRA or training code.
Run it and show me the resulting tables. I expect dysarthric WER to be
substantially worse than healthy WER (the literature range is roughly 50-80% WER
for dysarthric speech on generic models) — flag it clearly if your numbers land
far outside that range, since it may indicate a normalization or data bug rather
than a real result.
```

**Deliverables:** `reports/baseline_eval.md`, `reports/baseline_eval.json`, per-speaker baseline table.

**Definition of Done:** you have a citable baseline WER/CER split by healthy vs. dysarthric vs. per-held-out-speaker, and the numbers are sanity-checked against known literature ranges.

---

## Phase 3 — Generic LoRA Fine-Tuning (Pooled Baseline Model)

**Goal:** reproduce and formalize the pooled fine-tune from the earlier version of the project — this becomes the "generic model" arm of your final ablation, not the end product.

**PROMPT:**
```
Implement LoRA fine-tuning of Whisper on the pooled TORGO split in src/training/
and src/models/.

1. src/models/whisper_lora.py — wraps a pretrained Whisper model with a PEFT LoRA
   adapter (target the attention projection layers; make rank, alpha, dropout
   configurable via configs/lora_generic.yaml).
2. src/training/train_generic.py — fine-tunes the LoRA-wrapped model on
   pooled_train.csv, validates on pooled_val.csv, using Whisper's standard seq2seq
   cross-entropy loss. Log training/val loss per epoch to reports/generic_train_log.csv.
   Save the resulting adapter weights to a checkpoints/ dir (add checkpoints/ to
   .gitignore).
3. scripts/eval_generic.py — evaluate the fine-tuned generic model with the same
   metrics/breakdown as Phase 2's baseline script, on BOTH pooled_test.csv and
   speaker_holdout_test.csv. Save to reports/generic_eval.md /.json.
4. Produce a comparison table: baseline (Phase 2) vs. generic fine-tuned model,
   for both test sets, overall and dysarthric-only WER/CER — this is the first
   real "our fine-tuning helps" result.

Do NOT implement per-speaker personalization yet — that's Phase 4. This model is
intentionally the "one model for everyone" baseline your personalization approach
needs to beat, especially on speaker_holdout_test (unseen speakers), where you
expect generic fine-tuning to help less than it does on pooled_test.
Run training, run eval, show me the comparison table.
```

**Deliverables:** trained generic LoRA adapter, `reports/generic_eval.md`, baseline-vs-generic comparison table.

**Definition of Done:** generic model clearly beats zero-shot baseline on pooled_test; you have (and have looked at) its performance specifically on speaker_holdout_test, since that gap is the motivation for Phase 4.

---

## Phase 4 — Few-Shot Personalization Framework (Core Contribution)

**Goal:** this is the actual research contribution. Given a new held-out speaker and only N minutes of their speech, adapt faster/better than fine-tuning from scratch would.

**PROMPT:**
```
Implement few-shot per-speaker personalization in src/training/personalize.py and
src/models/.

Approach: per-speaker LoRA adapters on top of the Phase 3 generic model (not on top
of vanilla Whisper), with two initialization strategies to compare:
  (a) naive: initialize a fresh small-rank LoRA adapter randomly per speaker,
      fine-tune on their enrollment data only.
  (b) meta-initialized: before per-speaker adaptation, run a lightweight
      meta-learning-style step (e.g. Reptile-style first-order meta-learning, or
      simply pretraining a shared LoRA initialization across several
      speaker_holdout TRAIN speakers' enrollment-sized subsets) so per-speaker
      adaptation starts from a better initialization and converges faster with
      less data. Keep this implementation simple and clearly documented — the
      point is a defensible comparison, not a novel meta-learning algorithm.

1. src/data/enrollment.py — given a speaker_holdout test speaker, deterministically
   sample enrollment subsets of increasing size (e.g. 1, 2, 5, 10 minutes) from
   their available audio, holding out the rest for evaluation. Same enrollment
   subsets must be reused across both initialization strategies for a fair
   comparison — save the exact sampled file lists so this is reproducible.
2. src/training/personalize.py — given a speaker, an enrollment subset, and an
   initialization strategy, fine-tunes a per-speaker LoRA adapter and returns it.
3. scripts/run_personalization_experiment.py — for every held-out speaker and every
   enrollment size, run both initialization strategies, evaluate WER/CER on that
   speaker's held-out (non-enrollment) audio, and save results to
   reports/personalization_results.csv with columns: speaker_id, enrollment_minutes,
   strategy, WER, CER.
4. scripts/plot_adaptation_curve.py — produce the single most important chart in
   the whole project: WER vs. enrollment minutes, one line per strategy, with the
   Phase 3 generic-model WER on that speaker plotted as a flat reference line.
   Save to reports/adaptation_curve.png.

Do NOT touch decoding/confidence logic (Phase 5) or TTS (Phase 6). Do NOT change
the Phase 3 generic model — this phase builds on top of it, doesn't retrain it.
Run the full experiment (it may take a while — checkpoint per speaker/size/strategy
combo so it's resumable) and show me the results table and the adaptation curve.
```

**Deliverables:** `reports/personalization_results.csv`, `reports/adaptation_curve.png`, per-speaker per-speaker adapters.

**Definition of Done:** you can state, with numbers, "X minutes of enrollment audio reduces WER by Y% relative to the generic model," and you can show whether meta-initialization beats naive initialization at low enrollment sizes (it should, if it's working — that gap is the strongest publishable claim in the project).

---

## Phase 5 — Confidence-Aware Decoding & Fallback

**Goal:** turn "short single-word recordings are hard to recognize" from an unsolved challenge into a designed feature.

**PROMPT:**
```
Implement confidence-aware decoding in src/inference/.

1. src/inference/confidence.py — extract a per-utterance confidence score from
   Whisper decoding (e.g. average token log-probability, or use
   compute_dtw/logprob outputs already available from the generate() call with
   output_scores=True). Document exactly how the score is computed.
2. Determine a confidence threshold empirically: on speaker_holdout_test (or a
   held-out val subset), plot confidence score vs. whether the transcription was
   correct/incorrect, and pick a threshold that reasonably separates likely-correct
   from likely-wrong predictions. Save this analysis to reports/confidence_analysis.png
   and document the chosen threshold and why.
3. src/inference/fallback.py — when confidence falls below threshold, instead of
   returning the raw (likely garbled) transcription, return a small ranked list of
   candidate words/phrases from Whisper's beam search alternatives, constrained to
   a vocabulary/prompt list relevant to the use case (start with a simple common-
   phrases list for an AAC-style use case; this doesn't need to be sophisticated).
4. scripts/eval_with_fallback.py — re-run evaluation on speaker_holdout_test with
   fallback enabled and report: WER on high-confidence predictions only, and
   separately, how often fallback triggers and whether the top-3 fallback
   candidates contain the correct answer (top-3 accuracy) when it does trigger.

Do NOT implement the real-time audio pipeline yet (Phase 8) — this operates on
already-recorded utterances from the existing pipeline for now.
Show me the confidence analysis plot, chosen threshold, and fallback evaluation
results.
```

**Deliverables:** `reports/confidence_analysis.png`, fallback module, fallback evaluation results.

**Definition of Done:** you have a justified (not arbitrary) confidence threshold and a measurable top-3 fallback accuracy for low-confidence cases.

---

## Phase 6 — TTS Integration & Voice Banking

**Goal:** close the loop with Piper, and add voice-identity preservation for progressive-condition users.

**PROMPT:**
```
Implement TTS integration in src/tts/.

1. src/tts/piper_wrapper.py — wraps Piper TTS to synthesize speech from the final
   (post-fallback) transcription text, using a default voice.
2. src/tts/voice_bank.py — voice banking feature: given a small set of clean audio
   samples from a user (recorded while they still have clear/higher-functioning
   speech, or their pre-impairment recordings if available), fine-tune or select a
   Piper voice model to match, OR (simpler, if Piper fine-tuning isn't practical in
   scope) use a voice-cloning-lite approach via speaker embeddings if Piper/a
   secondary lightweight TTS library supports it. If full voice cloning is out of
   scope for the timeline, implement it as voice SELECTION from a small bank of
   Piper voices by closest match (pitch/gender/age heuristics) and document this as
   a scoped-down version with cloning as documented future work — don't overclaim
   what's implemented.
3. scripts/demo_pipeline.py — chains: audio in -> Phase 4 personalized model ->
   Phase 5 confidence/fallback -> Phase 6 TTS out, on a handful of sample
   utterances, saving output audio to reports/demo_outputs/.

Do NOT build the UI yet (Phase 9). Do NOT overclaim voice cloning quality — be
explicit in code comments and README about what's a real clone vs. a nearest-voice
selection, since this will come up in your final review.
Show me the demo pipeline running end-to-end on a few examples.
```

**Deliverables:** working TTS module, `reports/demo_outputs/`, end-to-end demo script.

**Definition of Done:** full audio-in to audio-out pipeline runs on real examples; scope of voice banking is honestly documented.

---

## Phase 7 — Full Evaluation Suite & Ablations

**Goal:** the results section of your report/paper. This is what a reviewer actually grades.

**PROMPT:**
```
Build the consolidated evaluation suite in src/evaluation/ and reports/.

1. scripts/run_full_ablation.py — produces one master comparison table across ALL
   arms tested so far, on speaker_holdout_test:
   - Zero-shot Whisper (Phase 2)
   - Generic pooled LoRA fine-tune (Phase 3)
   - Per-speaker naive LoRA, at each enrollment size (Phase 4)
   - Per-speaker meta-initialized LoRA, at each enrollment size (Phase 4)
   - (optional if time allows) generic model + confidence fallback (Phase 5) as an
     additional row
   Metrics: overall WER/CER, dysarthric-only WER/CER, and per-severity breakdown if
   available. Save as reports/master_ablation.csv and a formatted
   reports/master_ablation.md table suitable for pasting into a report.
2. scripts/statistical_significance.py — run a paired significance test (e.g.
   bootstrap resampling over utterances, or a paired t-test over per-speaker WER)
   comparing generic vs. meta-initialized personalization, so you can state
   whether the improvement is statistically meaningful, not just numerically
   larger.
3. Regenerate reports/adaptation_curve.png as a final polished version (clear
   labels, legend, title) suitable for direct inclusion in slides/report.
4. Write reports/RESULTS_SUMMARY.md — a plain-English summary of what the results
   show, written the way you'd present it in a project review: what improved, by
   how much, under what conditions, and what didn't work or didn't help as much as
   expected (a reviewer will trust the project more if some negative/mixed results
   are reported honestly rather than everything looking perfect).

Do NOT modify any training code in this phase — this is evaluation and reporting
only, using the checkpoints already produced by Phases 2-4.
Show me master_ablation.md and RESULTS_SUMMARY.md.
```

**Deliverables:** `reports/master_ablation.md`, `reports/RESULTS_SUMMARY.md`, significance test results, final adaptation curve.

**Definition of Done:** you have one master table and one plain-English summary you could present to a mentor cold, and you know whether your core claim (meta-init beats naive at low enrollment) is statistically supported or not.

---

## Phase 8 — Real-Time Inference Pipeline

**Goal:** define and hit a latency budget, addressing the "increased inference time" challenge by design rather than leaving it open.

**PROMPT:**
```
Implement a real-time-capable inference path in src/inference/realtime.py.

1. Define an explicit latency budget (propose: end-to-end under 2 seconds from
   end-of-utterance to transcription for short commands; document the reasoning).
2. Benchmark current pipeline latency (Whisper-small vs Whisper-medium, LoRA-
   adapted, with and without confidence fallback) on CPU and GPU if both available,
   using scripts/benchmark_latency.py. Save results to reports/latency_benchmark.md.
3. If the default Whisper-medium base doesn't meet budget, implement a fallback to
   a smaller checkpoint (whisper-small or distil-whisper) for the real-time
   deployed path specifically, keeping the larger model for offline/batch use
   (document this as an explicit design tradeoff, not a compromise you're hiding).
4. src/inference/streaming.py — implement chunked/streaming-style inference over a
   microphone input stream (use sounddevice or similar for capture), with voice
   activity detection (VAD) to segment utterances (webrtcvad or silero-vad are
   reasonable lightweight choices).
5. scripts/run_realtime_demo.py — live mic-in, transcription-out demo, printing
   latency per utterance to the console.

Do NOT build the graphical UI yet (Phase 9) — console output is fine here.
Show me the latency benchmark table and a description of what happened when you
ran the real-time demo (since I can't watch it live, report the per-utterance
latencies it printed).
```

**Deliverables:** `reports/latency_benchmark.md`, working streaming inference, documented latency-vs-model-size tradeoff.

**Definition of Done:** you have a stated latency budget, a benchmark showing whether you meet it, and a concrete decision (with reasoning) about which model size ships in the real-time path.

---

## Phase 9 — Demo UI

**Goal:** a presentable interface for your final review/demo day — this is presentation surface, not new ML work.

**PROMPT:**
```
Build a simple demo UI in src/ui/ for the personalized ASR + TTS pipeline. Use
Streamlit or Gradio (pick whichever integrates more simply with the existing
src/inference and src/tts modules — justify the choice in one line).

Features:
1. Upload or record audio -> show transcription, confidence score, and (if
   fallback triggered) the candidate list.
2. A "speaker profile" selector: choose an existing personalized adapter (from
   Phase 4 checkpoints) or fall back to the generic model, so the demo can show
   the before/after personalization difference live.
3. Play back the TTS output audio.
4. A results/metrics tab showing the master_ablation table and adaptation curve
   from Phase 7, for use during the review presentation.

Do NOT add user accounts, persistence, or deployment/hosting concerns — this is a
local demo app for a project review, not a production system. Keep it to a single
runnable script (scripts/run_ui.py).
Show me a description of the resulting UI layout and confirm it runs locally.
```

**Deliverables:** working local demo app.

**Definition of Done:** `python scripts/run_ui.py` launches a working demo covering transcription, personalization comparison, and TTS playback.

---

## Phase 10 — Documentation & Final Report Packaging

**Goal:** package everything into what you actually hand in / present.

**PROMPT:**
```
Finalize documentation and packaging for the project.

1. Rewrite README.md top-to-bottom: project description reframed around few-shot
   personalization (not just "we fine-tuned Whisper"), setup instructions, how to
   reproduce each phase's results from scratch (list the exact scripts in order),
   and a results summary linking to reports/RESULTS_SUMMARY.md and
   reports/master_ablation.md.
2. Write docs/ARCHITECTURE.md describing the full system: base model -> generic
   fine-tune -> per-speaker personalization -> confidence fallback -> TTS ->
   real-time pipeline, with a text description of the data flow suitable for
   redrawing as an architecture diagram slide.
3. Write docs/LIMITATIONS.md honestly covering: TORGO's speaker count and
   diversity limits, what voice banking does and doesn't do (per Phase 6), latency
   tradeoffs made (per Phase 8), and any statistically non-significant results
   from Phase 7 — reviewers respect honest limitations sections.
4. Generate a requirements-locked environment file (pip freeze or poetry.lock
   equivalent) so the whole thing is reproducible from a clean machine.
5. Run the full test suite (tests/) one final time and fix any that broke across
   phases.

Do NOT write any new ML code in this phase. Show me the final README.md and
LIMITATIONS.md content.
```

**Deliverables:** finalized README, ARCHITECTURE.md, LIMITATIONS.md, locked environment, passing full test suite.

**Definition of Done:** a stranger could clone the repo, follow the README, and reproduce your master ablation table.

---

## Notes on sequencing risk

- **Phase 4 is the load-bearing phase.** If meta-initialization doesn't clearly beat naive per-speaker fine-tuning, that's still a valid (if less exciting) result — report it honestly in Phase 7 rather than tuning until you get the answer you want. A well-documented negative result is more defensible in review than a suspiciously clean positive one.
- Don't start Phase 8 (real-time) or Phase 9 (UI) before Phase 7's numbers exist — reviewers will ask for results before they ask to see a demo, and having the ablation table first also tells you which model size Phase 8 actually needs to hit the latency budget.
- If time runs short before a review deadline, Phases 0–5 and 7 are the non-negotiable core (data, baseline, generic model, personalization, confidence, results). Phases 6, 8, 9 are strengthening/demo-polish and can be compressed or partially scoped down without weakening the core research claim.
