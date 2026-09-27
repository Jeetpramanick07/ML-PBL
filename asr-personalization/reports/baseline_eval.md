# Baseline (zero-shot) Whisper evaluation — `openai/whisper-small`

## pooled_test

### Overall

| group | n | WER | CER |
|---|---|---|---|
| all | 1005 | 0.3516 | 0.2083 |

### By dysarthria status

| group | n | WER | CER |
|---|---|---|---|
| dysarthric | 562 | 0.5537 | 0.3411 |
| healthy | 443 | 0.1074 | 0.0544 |

### By severity (dysarthric speakers only)

| group | n | WER | CER |
|---|---|---|---|
| mild | 179 | 0.3333 | 0.1928 |
| moderate | 134 | 0.3113 | 0.1738 |
| severe | 249 | 0.9121 | 0.5893 |

## speaker_holdout_test

### Overall

| group | n | WER | CER |
|---|---|---|---|
| all | 4853 | 0.3344 | 0.2116 |

### By dysarthria status

| group | n | WER | CER |
|---|---|---|---|
| dysarthric | 2363 | 0.5748 | 0.3911 |
| healthy | 2490 | 0.1081 | 0.0463 |

### By severity (dysarthric speakers only)

| group | n | WER | CER |
|---|---|---|---|
| mild | 1103 | 0.3821 | 0.2244 |
| moderate | 591 | 0.5531 | 0.4162 |
| severe | 669 | 0.9261 | 0.6689 |

### Per held-out speaker

| speaker_id | status | severity | n | WER | CER |
|---|---|---|---|---|---|
| F03 | dysarthric | mild | 1103 | 0.3821 | 0.2244 |
| FC01 | healthy |  | 304 | 0.1320 | 0.0707 |
| FC02 | healthy |  | 2186 | 0.1051 | 0.0435 |
| M04 | dysarthric | severe | 669 | 0.9261 | 0.6689 |
| M05 | dysarthric | moderate | 591 | 0.5531 | 0.4162 |
