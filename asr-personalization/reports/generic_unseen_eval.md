# Generic pooled LoRA fine-tune evaluation — `openai/whisper-small` + `checkpoints\generic_unseen\best`

## pooled_test

### Overall

| group | n | WER | CER |
|---|---|---|---|
| all | 1005 | 0.2560 | 0.1518 |

### By dysarthria status

| group | n | WER | CER |
|---|---|---|---|
| dysarthric | 562 | 0.4011 | 0.2521 |
| healthy | 443 | 0.0805 | 0.0355 |

### By severity (dysarthric speakers only)

| group | n | WER | CER |
|---|---|---|---|
| mild | 179 | 0.2951 | 0.1697 |
| moderate | 134 | 0.2243 | 0.1321 |
| severe | 249 | 0.6154 | 0.4089 |

## speaker_holdout_test

### Overall

| group | n | WER | CER |
|---|---|---|---|
| all | 4853 | 0.2828 | 0.1663 |

### By dysarthria status

| group | n | WER | CER |
|---|---|---|---|
| dysarthric | 2363 | 0.4882 | 0.3075 |
| healthy | 2490 | 0.0894 | 0.0361 |

### By severity (dysarthric speakers only)

| group | n | WER | CER |
|---|---|---|---|
| mild | 1103 | 0.3311 | 0.1878 |
| moderate | 591 | 0.4315 | 0.2611 |
| severe | 669 | 0.8090 | 0.5650 |

### Per held-out speaker

| speaker_id | status | severity | n | WER | CER |
|---|---|---|---|---|---|
| F03 | dysarthric | mild | 1103 | 0.3311 | 0.1878 |
| FC01 | healthy |  | 304 | 0.0924 | 0.0475 |
| FC02 | healthy |  | 2186 | 0.0890 | 0.0348 |
| M04 | dysarthric | severe | 669 | 0.8090 | 0.5650 |
| M05 | dysarthric | moderate | 591 | 0.4315 | 0.2611 |


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
