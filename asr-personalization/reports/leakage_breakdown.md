# Train/test audio-overlap breakdown (generic_full)

`clean` = audio never used in training or validation; `val_only` = in pooled_val (used for best-checkpoint selection, not gradient updates); `trained_on` = in pooled_train_subsample.csv.

speaker_holdout_test overlap counts: {'clean': 2804, 'trained_on': 1574, 'val_only': 475}

## speaker_holdout_test — baseline

| group_by | group | subset | n | WER | CER |
|---|---|---|---|---|---|
| all | all | all | 4853 | 0.3344 | 0.2116 |
| all | all | clean | 2804 | 0.3349 | 0.2076 |
| all | all | val_only | 475 | 0.3223 | 0.2523 |
| all | all | trained_on | 1574 | 0.3375 | 0.2059 |
| dysarthria_status | dysarthric | all | 2363 | 0.5748 | 0.3911 |
| dysarthria_status | dysarthric | clean | 1363 | 0.5752 | 0.3830 |
| dysarthria_status | dysarthric | val_only | 227 | 0.5690 | 0.4900 |
| dysarthria_status | dysarthric | trained_on | 773 | 0.5758 | 0.3746 |
| dysarthria_status | healthy | all | 2490 | 0.1081 | 0.0463 |
| dysarthria_status | healthy | clean | 1441 | 0.1090 | 0.0475 |
| dysarthria_status | healthy | val_only | 248 | 0.0936 | 0.0404 |
| dysarthria_status | healthy | trained_on | 801 | 0.1111 | 0.0460 |
| speaker_id | F03 | all | 1103 | 0.3821 | 0.2244 |
| speaker_id | F03 | clean | 623 | 0.3779 | 0.2188 |
| speaker_id | F03 | val_only | 112 | 0.3345 | 0.1915 |
| speaker_id | F03 | trained_on | 368 | 0.4042 | 0.2437 |
| speaker_id | FC01 | all | 304 | 0.1320 | 0.0707 |
| speaker_id | FC01 | clean | 170 | 0.1232 | 0.0667 |
| speaker_id | FC01 | val_only | 21 | 0.1000 | 0.0591 |
| speaker_id | FC01 | trained_on | 113 | 0.1525 | 0.0796 |
| speaker_id | FC02 | all | 2186 | 0.1051 | 0.0435 |
| speaker_id | FC02 | clean | 1271 | 0.1073 | 0.0452 |
| speaker_id | FC02 | val_only | 227 | 0.0932 | 0.0393 |
| speaker_id | FC02 | trained_on | 688 | 0.1052 | 0.0415 |
| speaker_id | M04 | all | 669 | 0.9261 | 0.6689 |
| speaker_id | M04 | clean | 386 | 0.9599 | 0.6893 |
| speaker_id | M04 | val_only | 65 | 0.9940 | 0.7782 |
| speaker_id | M04 | trained_on | 218 | 0.8462 | 0.5999 |
| speaker_id | M05 | all | 591 | 0.5531 | 0.4162 |
| speaker_id | M05 | clean | 354 | 0.5350 | 0.3785 |
| speaker_id | M05 | val_only | 50 | 0.5263 | 0.7195 |
| speaker_id | M05 | trained_on | 187 | 0.6075 | 0.3860 |

## speaker_holdout_test — generic_full

| group_by | group | subset | n | WER | CER |
|---|---|---|---|---|---|
| all | all | all | 4853 | 0.1315 | 0.0760 |
| all | all | clean | 2804 | 0.1771 | 0.1025 |
| all | all | val_only | 475 | 0.1765 | 0.1012 |
| all | all | trained_on | 1574 | 0.0296 | 0.0171 |
| dysarthria_status | dysarthric | all | 2363 | 0.2328 | 0.1440 |
| dysarthria_status | dysarthric | clean | 1363 | 0.3155 | 0.1957 |
| dysarthria_status | dysarthric | val_only | 227 | 0.3131 | 0.1957 |
| dysarthria_status | dysarthric | trained_on | 773 | 0.0496 | 0.0308 |
| dysarthria_status | healthy | all | 2490 | 0.0361 | 0.0133 |
| dysarthria_status | healthy | clean | 1441 | 0.0470 | 0.0174 |
| dysarthria_status | healthy | val_only | 248 | 0.0499 | 0.0170 |
| dysarthria_status | healthy | trained_on | 801 | 0.0105 | 0.0041 |
| speaker_id | F03 | all | 1103 | 0.1596 | 0.0895 |
| speaker_id | F03 | clean | 623 | 0.2344 | 0.1322 |
| speaker_id | F03 | val_only | 112 | 0.1418 | 0.0780 |
| speaker_id | F03 | trained_on | 368 | 0.0319 | 0.0177 |
| speaker_id | FC01 | all | 304 | 0.0235 | 0.0141 |
| speaker_id | FC01 | clean | 170 | 0.0320 | 0.0195 |
| speaker_id | FC01 | val_only | 21 | 0.0250 | 0.0215 |
| speaker_id | FC01 | trained_on | 113 | 0.0085 | 0.0035 |
| speaker_id | FC02 | all | 2186 | 0.0376 | 0.0132 |
| speaker_id | FC02 | clean | 1271 | 0.0489 | 0.0172 |
| speaker_id | FC02 | val_only | 227 | 0.0516 | 0.0167 |
| speaker_id | FC02 | trained_on | 688 | 0.0108 | 0.0041 |
| speaker_id | M04 | all | 669 | 0.3580 | 0.2400 |
| speaker_id | M04 | clean | 386 | 0.4810 | 0.3237 |
| speaker_id | M04 | val_only | 65 | 0.5808 | 0.3891 |
| speaker_id | M04 | trained_on | 218 | 0.0750 | 0.0504 |
| speaker_id | M05 | all | 591 | 0.2340 | 0.1466 |
| speaker_id | M05 | clean | 354 | 0.2914 | 0.1845 |
| speaker_id | M05 | val_only | 50 | 0.3289 | 0.1992 |
| speaker_id | M05 | trained_on | 187 | 0.0565 | 0.0358 |

## Personalization eval_holdout sets

| speaker | strategy | min | subset | n | WER | CER |
|---|---|---|---|---|---|---|
| F03 | generic | 0min | all | 300 | 0.1694 | 0.0937 |
| F03 | generic | 0min | clean | 174 | 0.2416 | 0.1356 |
| F03 | generic | 0min | val_only | 28 | 0.1692 | 0.0967 |
| F03 | generic | 0min | trained_on | 98 | 0.0187 | 0.0065 |
| F03 | naive | 5min | all | 300 | 0.1749 | 0.0992 |
| F03 | naive | 5min | clean | 174 | 0.2349 | 0.1338 |
| F03 | naive | 5min | val_only | 28 | 0.1231 | 0.0785 |
| F03 | naive | 5min | trained_on | 98 | 0.0654 | 0.0345 |
| M04 | generic | 0min | all | 300 | 0.3941 | 0.2522 |
| M04 | generic | 0min | clean | 172 | 0.5608 | 0.3636 |
| M04 | generic | 0min | val_only | 32 | 0.5595 | 0.3511 |
| M04 | generic | 0min | trained_on | 96 | 0.0841 | 0.0527 |
| M04 | naive | 5min | all | 300 | 0.3941 | 0.2997 |
| M04 | naive | 5min | clean | 172 | 0.5282 | 0.4228 |
| M04 | naive | 5min | val_only | 32 | 0.6071 | 0.4140 |
| M04 | naive | 5min | trained_on | 96 | 0.1150 | 0.0772 |
| M05 | generic | 0min | all | 300 | 0.2335 | 0.1444 |
| M05 | generic | 0min | clean | 169 | 0.2914 | 0.1839 |
| M05 | generic | 0min | val_only | 28 | 0.3478 | 0.1952 |
| M05 | generic | 0min | trained_on | 103 | 0.0489 | 0.0329 |
| M05 | naive | 5min | all | 300 | 0.2056 | 0.1247 |
| M05 | naive | 5min | clean | 169 | 0.2346 | 0.1395 |
| M05 | naive | 5min | val_only | 28 | 0.3587 | 0.2343 |
| M05 | naive | 5min | trained_on | 103 | 0.0652 | 0.0373 |
