# Generic pooled LoRA fine-tune evaluation — `openai/whisper-small` + `checkpoints\generic_full\best`

## pooled_test

### Overall

| group | n | WER | CER |
|---|---|---|---|
| all | 1005 | 0.1682 | 0.0983 |

### By dysarthria status

| group | n | WER | CER |
|---|---|---|---|
| dysarthric | 562 | 0.2636 | 0.1617 |
| healthy | 443 | 0.0528 | 0.0248 |

### By severity (dysarthric speakers only)

| group | n | WER | CER |
|---|---|---|---|
| mild | 179 | 0.2166 | 0.1281 |
| moderate | 134 | 0.1372 | 0.0893 |
| severe | 249 | 0.3919 | 0.2419 |

## speaker_holdout_test

### Overall

| group | n | WER | CER |
|---|---|---|---|
| all | 4853 | 0.1315 | 0.0760 |

### By dysarthria status

| group | n | WER | CER |
|---|---|---|---|
| dysarthric | 2363 | 0.2328 | 0.1440 |
| healthy | 2490 | 0.0361 | 0.0133 |

### By severity (dysarthric speakers only)

| group | n | WER | CER |
|---|---|---|---|
| mild | 1103 | 0.1596 | 0.0895 |
| moderate | 591 | 0.2340 | 0.1466 |
| severe | 669 | 0.3580 | 0.2400 |

### Per held-out speaker

| speaker_id | status | severity | n | WER | CER |
|---|---|---|---|---|---|
| F03 | dysarthric | mild | 1103 | 0.1596 | 0.0895 |
| FC01 | healthy |  | 304 | 0.0235 | 0.0141 |
| FC02 | healthy |  | 2186 | 0.0376 | 0.0132 |
| M04 | dysarthric | severe | 669 | 0.3580 | 0.2400 |
| M05 | dysarthric | moderate | 591 | 0.2340 | 0.1466 |


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
