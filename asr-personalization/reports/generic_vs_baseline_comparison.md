# Baseline vs. generic fine-tuned model

## pooled_test

| arm | group | n | WER | CER |
|---|---|---|---|---|
| baseline (zero-shot) | overall | 1005 | 0.3516 | 0.2083 |
| baseline (zero-shot) | dysarthric-only | 562 | 0.5537 | 0.3411 |
| generic LoRA | overall | 1005 | 0.2595 | 0.1541 |
| generic LoRA | dysarthric-only | 562 | 0.4062 | 0.2553 |

Dysarthric-only WER reduction: 26.6% relative (0.5537 -> 0.4062).

## speaker_holdout_test

| arm | group | n | WER | CER |
|---|---|---|---|---|
| baseline (zero-shot) | overall | 4853 | 0.3344 | 0.2116 |
| baseline (zero-shot) | dysarthric-only | 2363 | 0.5748 | 0.3911 |
| generic LoRA | overall | 4853 | 0.2624 | 0.1551 |
| generic LoRA | dysarthric-only | 2363 | 0.4531 | 0.2879 |

Dysarthric-only WER reduction: 21.2% relative (0.5748 -> 0.4531).
