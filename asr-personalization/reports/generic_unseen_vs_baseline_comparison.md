# Baseline vs. generic fine-tuned model

## pooled_test

| arm | group | n | WER | CER |
|---|---|---|---|---|
| baseline (zero-shot) | overall | 1005 | 0.3516 | 0.2083 |
| baseline (zero-shot) | dysarthric-only | 562 | 0.5537 | 0.3411 |
| generic LoRA | overall | 1005 | 0.2560 | 0.1518 |
| generic LoRA | dysarthric-only | 562 | 0.4011 | 0.2521 |

Dysarthric-only WER reduction: 27.6% relative (0.5537 -> 0.4011).

## speaker_holdout_test

| arm | group | n | WER | CER |
|---|---|---|---|---|
| baseline (zero-shot) | overall | 4853 | 0.3344 | 0.2116 |
| baseline (zero-shot) | dysarthric-only | 2363 | 0.5748 | 0.3911 |
| generic LoRA | overall | 4853 | 0.2828 | 0.1663 |
| generic LoRA | dysarthric-only | 2363 | 0.4882 | 0.3075 |

Dysarthric-only WER reduction: 15.1% relative (0.5748 -> 0.4882).
