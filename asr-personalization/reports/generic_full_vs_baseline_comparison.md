# Baseline vs. generic fine-tuned model

## pooled_test

| arm | group | n | WER | CER |
|---|---|---|---|---|
| baseline (zero-shot) | overall | 1005 | 0.3516 | 0.2083 |
| baseline (zero-shot) | dysarthric-only | 562 | 0.5537 | 0.3411 |
| generic LoRA | overall | 1005 | 0.1682 | 0.0983 |
| generic LoRA | dysarthric-only | 562 | 0.2636 | 0.1617 |

Dysarthric-only WER reduction: 52.4% relative (0.5537 -> 0.2636).

## speaker_holdout_test

| arm | group | n | WER | CER |
|---|---|---|---|---|
| baseline (zero-shot) | overall | 4853 | 0.3344 | 0.2116 |
| baseline (zero-shot) | dysarthric-only | 2363 | 0.5748 | 0.3911 |
| generic LoRA | overall | 4853 | 0.1315 | 0.0760 |
| generic LoRA | dysarthric-only | 2363 | 0.2328 | 0.1440 |

Dysarthric-only WER reduction: 59.5% relative (0.5748 -> 0.2328).
