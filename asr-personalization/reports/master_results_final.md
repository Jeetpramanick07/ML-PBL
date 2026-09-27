# Master results (final): pooled LoRA + few-shot personalization

Model: `openai/whisper-small` + LoRA (r=8, alpha=16, dropout=0.05, target_modules=[q_proj, v_proj]).
Final pooled adapter: `checkpoints/generic_full/best` = **epoch 6, val WER 16.30%** (val loss 0.1957).
All WERs are corpus-level after `normalize_transcription`. Single training seed (42) throughout, and no confidence intervals were computed.

---

## 1. Methodology note (read this first)

### 1.1 Compute-constrained training
- **Training data: stratified 40% subsample.** The pooled LoRA was trained on a stratified random subsample of the pooled training split, `pooled_train_subsample.csv`: **N = 3,213 of 8,032 rows (40.0%)**, `subsample_stratified(seed=42)`. The dysarthric/healthy ratio is preserved exactly (55.9% / 44.1% in both the full split and the subsample). The reason was compute and time: the full split at about 42 min/epoch made a 40–50 epoch budget a 10–34 hour job.
- **Validation: full split.** Validation used the **full** `pooled_val.csv` (N = 1,004) every epoch. Checkpoint selection and early stopping used greedy-decoded validation WER.
- **Evaluation: full test sets.** Evaluation used the **full** speaker-held-out test set (`speaker_holdout_test.csv`, N = 4,853) and the full `pooled_test.csv` (N = 1,005). The eval protocol was not changed.
- **Unchanged settings.** LoRA rank, alpha, target modules, learning rate (1e-4), warmup (200), batch size (8, fp16) and the eval protocol are all unchanged from the plan.

### 1.2 How training ended: time-capped, not converged
- **Stop point.** Training ran **epochs 0–6 (7 epochs, 2,814 optimizer steps)** and was **stopped by the strict 4.5 h wall-clock cap after epoch 6**. Epoch 7 (~34 min) would have ended at about 4.73 h, so the process was stopped immediately after epoch 6's checkpoints were written, and epoch 7 never started.
- **Early stopping did NOT trigger.** Patience was 5, and epochs_since_improvement was 0 at the stop, because epoch 6 was itself a new best.
- **Training was not converged.** Val WER was still improving: 18.07% at epoch 4, 19.79% at epoch 5 (one non-improving epoch), then 16.30% at epoch 6. Longer training would likely have lowered it further. The results below are therefore a lower bound on what this setup can reach.
- **Total training wall-clock: 15,003 s = 4.17 h.** That is the cumulative training-loop time: epochs 0–2 took 7,056.8 s before an overnight pause, and epochs 3–6 took 7,946 s after resuming from `last/` at epoch 3 / step 1206. The pause and model-load time are excluded.
- **Resume caveats.**
  - The DataLoader shuffle order for epochs 3–6 differs from what an uninterrupted run would have used, because RNG state is not checkpointed.
  - About 1 minute of discarded epoch-3 steps from before the pause is not counted.
  - A save-order bug that would have made a resumed run under-count early-stopping patience was fixed before resuming. `last/` now records the post-epoch `best_epoch` and `epochs_since_improvement`. At the time of the pause, this bug had only mislabelled `best_epoch` (1 instead of 2) in the saved state; that was corrected from the training log.

| epoch | train_loss | val_loss | val_WER | epoch time |
|---|---|---|---|---|
| 0 | 2.9950 | 0.3455 | 25.45% | 38.9 min |
| 1 | 0.2759 | 0.2659 | 21.29% | 39.0 min |
| 2 | 0.1976 | 0.2443 | 19.72% | 39.8 min |
| 3 | 0.1477 | 0.2169 | 18.37% | 31.0 min |
| 4 | 0.1094 | 0.2041 | 18.07% | 33.1 min |
| 5 | 0.0809 | 0.2064 | 19.79% | 34.2 min |
| **6** | **0.0608** | **0.1957** | **16.30%** | 34.1 min |

(Epochs 0–2 were slower because the baseline eval shared the GPU at the time.)

### 1.3 Train/test audio overlap: this changes how the results must be read
The pooled splits are a random **utterance-level** split over all 11 speakers. `speaker_holdout_test` contains **every** utterance of F03, FC01, FC02, M04 and M05. As a result, the exact same audio files appear in both:

| speaker_holdout_test rows (N = 4,853) | n | share |
|---|---|---|
| audio **trained on** (in `pooled_train_subsample.csv`) | 1,574 | 32% |
| audio in `pooled_val.csv` (used for checkpoint **selection**) | 475 | 10% |
| **clean**: never seen in training or validation | 2,804 | 58% |

The model has **memorized** the trained-on rows: WER on them is **2.96%**, compared with **17.71%** on clean rows. The full-set speaker-held-out WER (13.15%) is therefore **inflated and must not be read as a held-out result**.

This report uses the **clean subset** as the honest number. It is still "utterance-unseen, **speaker-seen**": the model has heard these speakers' voices on other utterances. So it is **not** a speaker-independent result. A sanity check confirms the clean subset is not easier or harder than the full set: the zero-shot baseline scores 33.49% on the clean subset and 33.44% on the full set.

The only true speaker-unseen fine-tuned result in this project is `generic_unseen`, which excluded these speakers from training. It was trained on 1,200 rows for 2 epochs, so it is **not** compute-comparable (see §2.3). `pooled_test` has **zero** audio overlap with training.

The breakdown was computed post hoc from saved hypotheses by `scripts/leakage_breakdown.py`. Full tables are in `reports/leakage_breakdown.md` and `reports/leakage_breakdown.json`.

---

## 2. Pooled LoRA vs. zero-shot baseline

### 2.1 Speaker-held-out test set: clean subset (headline)

| group | n | baseline WER | pooled LoRA WER | relative change |
|---|---|---|---|---|
| **all (clean)** | 2,804 | 33.49% | **17.71%** | −47.1% |
| **dysarthric (clean)** | 1,363 | 57.52% | **31.55%** | **−45.1%** |
| healthy (clean) | 1,441 | 10.90% | 4.70% | −56.9% |
| F03 (mild) | 623 | 37.79% | 23.44% | −38.0% |
| M05 (moderate) | 354 | 53.50% | 29.14% | −45.5% |
| M04 (severe) | 386 | 95.99% | 48.10% | −49.9% |
| FC01 (healthy) | 170 | 12.32% | 3.20% | −74.0% |
| FC02 (healthy) | 1,271 | 10.73% | 4.89% | −54.4% |

### 2.2 Full sets as run by the eval protocol (for completeness)

| split | group | n | baseline WER | pooled LoRA WER | note |
|---|---|---|---|---|---|
| pooled_test | all | 1,005 | 35.16% | 16.82% | no audio overlap with training |
| pooled_test | dysarthric | 562 | 55.37% | 26.36% | −52.4% relative |
| pooled_test | healthy | 443 | 10.74% | 5.28% | |
| speaker_holdout_test | all | 4,853 | 33.44% | 13.15% | **inflated by overlap (§1.3)** |
| speaker_holdout_test | dysarthric | 2,363 | 57.48% | 23.28% | **inflated**; −59.5% relative as run, vs −45.1% on clean |
| speaker_holdout_test | healthy | 2,490 | 10.81% | 3.61% | **inflated** |

The pooled-val WER (16.30%) agrees with pooled_test (16.82%), which is consistent with pooled_test being a fair utterance-level test.

### 2.3 Context: earlier pooled adapters (same LoRA config, not compute-comparable)

| model | training data | speaker_holdout_test dysarthric WER | pooled_test dysarthric WER |
|---|---|---|---|
| `generic` (Sept 20) | 1,200-row subsample, 2 epochs | 45.31% (also overlap-affected) | 40.62% |
| `generic_unseen` (Sept 20) | 1,200 rows, **held-out speakers excluded**, 2 epochs | **48.82%** (true speaker-unseen) | 40.11% |
| **`generic_full`** (this run) | 3,213-row subsample, 7 epochs (time-capped) | 31.55% clean / 23.28% full | **26.36%** |

---

## 3. Few-shot personalization (3 held-out dysarthric speakers)

### 3.1 Setup
- **Scope.** Reduced from 5 to 3 speakers per the time budget: F03 (mild), M05 (moderate) and M04 (severe). The healthy held-out speakers FC01 and FC02 are omitted.
- **Adapter.** Naive per-speaker LoRA (r=4, alpha=8, q/v, 3 epochs, lr 5e-4, batch size 4), stacked on the merged `generic_full/best` (epoch 6).
- **Enrollment and eval data.**
  - 5 minutes of enrollment: F03 135, M04 and M05 93 utterances.
  - A fixed 300-utterance `eval_holdout` per speaker (seed 42), the same F03 sets as the Sept-20 run.
- **Not run.** The meta-init arm and the 1/2/10-minute sizes were not run (budget).
- **Leakage in these eval sets as well.**
  - About 1/3 of each 300-row eval set was in pooled training (F03 98, M04 96, M05 103).
  - Personalization is also **not** a "never heard this speaker" test, because the base model trained on these speakers' audio.
- **Timing.** Ran 11:51–12:29 (about 38 min). It overlapped with the held-out eval for its first ~26 min, until GPU-memory spill forced the two to be serialized.

### 3.2 Results

| speaker | subset | n | pooled only (0 min) WER / CER | + 5 min personalization WER / CER | ΔWER (points) |
|---|---|---|---|---|---|
| F03 (mild) | all 300 | 300 | 16.94% / 9.37% | 17.49% / 9.92% | +0.55 |
| | **clean** | 174 | 24.16% / 13.56% | **23.49%** / 13.38% | **−0.67** |
| M04 (severe) | all 300 | 300 | 39.41% / 25.22% | 39.41% / 29.97% | 0.00 |
| | **clean** | 172 | 56.08% / 36.36% | **52.82%** / 42.28% | **−3.26** (CER +5.92) |
| M05 (moderate) | all 300 | 300 | 23.35% / 14.44% | 20.56% / 12.47% | −2.79 |
| | **clean** | 169 | 29.14% / 18.39% | **23.46%** / 13.95% | **−5.68** |

Unseen-*text* subset (eval utterances whose prompt is not in the enrollment set; all 300 rows, not leakage-filtered), 5-minute adapter: F03 17.13% (n=230), M04 44.56% (n=201), M05 19.33% (n=258).

### 3.3 Reading
- **On clean rows, 5-minute personalization lowers WER for all three speakers.**
  - M05: −5.7 points, −19.5% relative.
  - M04: −3.3 points.
  - F03: −0.7 points.
- **On the full 300-row sets the gain is masked, or looks like a regression.** The personalized adapter partly "unlearns" the memorized trained-on rows: F03 trained-on WER went from 1.9% to 6.5%. That is a leakage artifact, not a real degradation.
- **M04's result is mixed.** WER improved on clean rows but CER worsened (36.4% → 42.3%), and its unseen-text WER (44.6%) is worse than its all-row WER. The adapter may be fitting the enrollment prompts rather than the speaker. This is not a clear win.
- **Caveats.**
  - n ≈ 170 clean rows per speaker, one seed, no significance testing. Differences under ~2 points (F03) are within plausible noise.
  - F03 is mild and already well served by the pooled model, so its small gain is expected.

---

## 4. Bottom line
1. The time-capped, 40%-subsample pooled LoRA (epoch 6) roughly **halves dysarthric WER** relative to zero-shot Whisper-small:
   - 57.5% → 31.6% on the clean speaker-held-out subset (−45%).
   - 55.4% → 26.4% on pooled_test (−52%).
2. The headline **"speaker-held-out" full-set figure (23.3% dysarthric, −59.5%) is inflated** by train/test audio overlap and should not be quoted. These results are utterance-unseen, **speaker-seen**. They are not speaker-independent.
3. Training was **time-capped at 4.17 h (7 epochs), not converged**. Early stopping never triggered, and val WER was still falling.
4. Naive 5-minute personalization on top of this model gives **modest WER gains on clean rows**: M05 −5.7, M04 −3.3, F03 −0.7 points. M04's CER got worse, and the evidence is small-n and single-seed.
5. A clean, speaker-independent version of this study needs pooled training that **excludes the held-out speakers** (as `generic_unseen` did), at a comparable compute budget.

## Artifacts
- Training: `reports/generic_full_train_log.csv`, `reports/generic_full_train_summary.json`, `reports/generic_full_train_stdout*.log`
- Pooled eval: `reports/generic_full_eval.{md,json}`, `reports/generic_full_hyps_{pooled_test,speaker_holdout_test}.csv`
- Personalization: `reports/personalization_results_final.csv`, `reports/personalization_hyps_final/`, `checkpoints/personalized_final/`
- Leakage breakdown: `reports/leakage_breakdown.{md,json}` (`scripts/leakage_breakdown.py`)
- Checkpoint backup from before the resume: `checkpoints/generic_full_backup_epoch2/`
