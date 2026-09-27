# Generic pooled LoRA fine-tune evaluation — `openai/whisper-small` + `checkpoints\generic\best`

## pooled_test

### Overall

| group | n | WER | CER |
|---|---|---|---|
| all | 1005 | 0.2595 | 0.1541 |

### By dysarthria status

| group | n | WER | CER |
|---|---|---|---|
| dysarthric | 562 | 0.4062 | 0.2553 |
| healthy | 443 | 0.0823 | 0.0368 |

### By severity (dysarthric speakers only)

| group | n | WER | CER |
|---|---|---|---|
| mild | 179 | 0.2972 | 0.1827 |
| moderate | 134 | 0.2243 | 0.1390 |
| severe | 249 | 0.6264 | 0.4007 |

## speaker_holdout_test

### Overall

| group | n | WER | CER |
|---|---|---|---|
| all | 4853 | 0.2624 | 0.1551 |

### By dysarthria status

| group | n | WER | CER |
|---|---|---|---|
| dysarthric | 2363 | 0.4531 | 0.2879 |
| healthy | 2490 | 0.0828 | 0.0328 |

### By severity (dysarthric speakers only)

| group | n | WER | CER |
|---|---|---|---|
| mild | 1103 | 0.3146 | 0.1786 |
| moderate | 591 | 0.4230 | 0.2544 |
| severe | 669 | 0.7184 | 0.5150 |

### Per held-out speaker

| speaker_id | status | severity | n | WER | CER |
|---|---|---|---|---|---|
| F03 | dysarthric | mild | 1103 | 0.3146 | 0.1786 |
| FC01 | healthy |  | 304 | 0.0865 | 0.0370 |
| FC02 | healthy |  | 2186 | 0.0823 | 0.0323 |
| M04 | dysarthric | severe | 669 | 0.7184 | 0.5150 |
| M05 | dysarthric | moderate | 591 | 0.4230 | 0.2544 |


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
