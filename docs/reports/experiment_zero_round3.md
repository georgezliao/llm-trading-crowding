# Experiment Zero, round 3

**Synthetic-simulator disclaimer:** Every result in this report comes from the uncalibrated synthetic HerdLab simulator. It is not evidence about real markets, real funds, or live trading behavior.

## Configuration and runtime

- Fixed selected setting: `configs/experiment_zero.yaml` (linear impact sensitivity 0.03, q_max 200, order limit 40, drawdown limit 0.20, and the Amendment 1 cooldown rule).
- V2 cohort: `configs/heldout_families.yaml` and `configs/heldout_impact.yaml`; 200 training episodes and 50 episodes in each of in-distribution, held-out-family, and held-out-impact.
- Architecture cohort: `configs/heldout_architecture.yaml`; train population momentum 6/reversion 6/linear 9/MLP 9; held-out population momentum 6/reversion 6/GBDT 18; 200 training episodes and 50 episodes in each evaluation split.
- Episode seeds: v2 training `1000-1199`, in-distribution `2000-2049`, held-out-family `3000-3049`, held-out-impact `4000-4049`; architecture training `5000-5199`, in-distribution `5500-5549`, held-out-architecture `6500-6549`.
- Labels: default, onset, and forced-flow; horizon 10; each cohort/label threshold is its own training-only 0.95 score quantile.
- Observers: trivial reference, logistic regression, LightGBM, price-only strategy replicator, GRU seeds `6000-6004`, and their probability ensemble. Public-feature return lookback is 10.
- Metrics: PR-AUC, Brier score, and recall at FAR <= 0.05; 95% percentile CIs use 1000 whole-episode resamples. Paired PR-AUC differences use the same resampled episodes for both observers.
- Parallelism: 10 simulator workers and 5 concurrent GRU seed workers.
- Measured wall runtime: 20.80 minutes.

## Training-only thresholds

| Cohort | Label | Threshold |
| --- | --- | ---: |
| v2 | default | 0.1591791367 |
| v2 | onset | 0.1591791367 |
| v2 | forced_flow | 0.0008695983778 |
| architecture | default | 0.1285898016 |
| architecture | onset | 0.1285898016 |
| architecture | forced_flow | 0.0003238555982 |

## Base rates and event counts

Counts and observed base rates are descriptive; inferential CIs are reported in the observer and paired-difference tables below.

| Cohort | Label | Split | Events / labels | Base rate |
| --- | --- | --- | ---: | ---: |
| v2 | default | training | 4900 / 98000 | 0.050000 |
| v2 | default | in_distribution | 1441 / 24500 | 0.058816 |
| v2 | default | heldout_families | 0 / 24500 | 0.000000 |
| v2 | default | heldout_impact | 2632 / 24500 | 0.107429 |
| v2 | onset | training | 875 / 98000 | 0.008929 |
| v2 | onset | in_distribution | 207 / 24500 | 0.008449 |
| v2 | onset | heldout_families | 0 / 24500 | 0.000000 |
| v2 | onset | heldout_impact | 295 / 24500 | 0.012041 |
| v2 | forced_flow | training | 4900 / 98000 | 0.050000 |
| v2 | forced_flow | in_distribution | 1111 / 24500 | 0.045347 |
| v2 | forced_flow | heldout_families | 12 / 24500 | 0.000490 |
| v2 | forced_flow | heldout_impact | 1881 / 24500 | 0.076776 |
| architecture | default | training | 4900 / 98000 | 0.050000 |
| architecture | default | in_distribution | 846 / 24500 | 0.034531 |
| architecture | default | heldout_architecture | 378 / 24500 | 0.015429 |
| architecture | onset | training | 490 / 98000 | 0.005000 |
| architecture | onset | in_distribution | 81 / 24500 | 0.003306 |
| architecture | onset | heldout_architecture | 98 / 24500 | 0.004000 |
| architecture | forced_flow | training | 4900 / 98000 | 0.050000 |
| architecture | forced_flow | in_distribution | 1246 / 24500 | 0.050857 |
| architecture | forced_flow | heldout_architecture | 1150 / 24500 | 0.046939 |

## Observer metrics

Every entry is point estimate [episode-bootstrap 95% CI]. Higher is better for PR-AUC and recall; lower is better for Brier score.

### v2 / default / in_distribution

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1094 [0.0959, 0.1228] | 0.0545 [0.0466, 0.0626] | 0.1513 [0.1284, 0.1757] |
| logistic_regression | 0.4512 [0.4096, 0.4867] | 0.0410 [0.0357, 0.0465] | 0.5607 [0.5306, 0.5958] |
| lightgbm | 0.7138 [0.6692, 0.7535] | 0.0282 [0.0247, 0.0317] | 0.8473 [0.8116, 0.8749] |
| replicator | 0.2372 [0.2040, 0.2743] | 0.0498 [0.0434, 0.0572] | 0.2887 [0.2544, 0.3265] |
| gru_6000 | 0.7964 [0.7566, 0.8280] | 0.0229 [0.0203, 0.0252] | 0.9077 [0.8837, 0.9296] |
| gru_6001 | 0.7904 [0.7538, 0.8206] | 0.0233 [0.0210, 0.0257] | 0.9160 [0.8937, 0.9410] |
| gru_6002 | 0.7779 [0.7399, 0.8129] | 0.0239 [0.0210, 0.0264] | 0.9153 [0.8901, 0.9342] |
| gru_6003 | 0.7944 [0.7523, 0.8295] | 0.0232 [0.0208, 0.0257] | 0.9056 [0.8806, 0.9305] |
| gru_6004 | 0.7971 [0.7584, 0.8326] | 0.0235 [0.0210, 0.0260] | 0.9112 [0.8852, 0.9312] |
| gru_ensemble | 0.8048 [0.7700, 0.8345] | 0.0226 [0.0201, 0.0251] | 0.9160 [0.8915, 0.9377] |

### v2 / default / heldout_families

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | N/A | 0.0030 [0.0029, 0.0031] | N/A |
| logistic_regression | N/A | 0.0010 [0.0007, 0.0014] | N/A |
| lightgbm | N/A | 0.0007 [0.0006, 0.0009] | N/A |
| replicator | N/A | 0.0038 [0.0037, 0.0038] | N/A |
| gru_6000 | N/A | 0.0010 [0.0002, 0.0021] | N/A |
| gru_6001 | N/A | 0.0001 [0.0000, 0.0003] | N/A |
| gru_6002 | N/A | 0.0000 [0.0000, 0.0001] | N/A |
| gru_6003 | N/A | 0.0002 [0.0000, 0.0005] | N/A |
| gru_6004 | N/A | 0.0003 [0.0000, 0.0007] | N/A |
| gru_ensemble | N/A | 0.0001 [0.0000, 0.0002] | N/A |

### v2 / default / heldout_impact

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1830 [0.1599, 0.2066] | 0.0994 [0.0897, 0.1081] | 0.1326 [0.1138, 0.1530] |
| logistic_regression | 0.4455 [0.4156, 0.4769] | 0.0756 [0.0694, 0.0817] | 0.4141 [0.3845, 0.4469] |
| lightgbm | 0.6119 [0.5813, 0.6437] | 0.0580 [0.0531, 0.0629] | 0.6353 [0.5996, 0.6710] |
| replicator | 0.3258 [0.2901, 0.3615] | 0.0870 [0.0799, 0.0939] | 0.2078 [0.1851, 0.2307] |
| gru_6000 | 0.7598 [0.7368, 0.7821] | 0.0480 [0.0438, 0.0521] | 0.7633 [0.7302, 0.7999] |
| gru_6001 | 0.7248 [0.7021, 0.7497] | 0.0497 [0.0452, 0.0542] | 0.7629 [0.7258, 0.7973] |
| gru_6002 | 0.6672 [0.6451, 0.6887] | 0.0559 [0.0512, 0.0605] | 0.6752 [0.6394, 0.7223] |
| gru_6003 | 0.7347 [0.7146, 0.7537] | 0.0487 [0.0449, 0.0525] | 0.7652 [0.7324, 0.8008] |
| gru_6004 | 0.7501 [0.7284, 0.7711] | 0.0469 [0.0426, 0.0510] | 0.7694 [0.7351, 0.8007] |
| gru_ensemble | 0.7433 [0.7212, 0.7641] | 0.0480 [0.0438, 0.0518] | 0.7629 [0.7285, 0.7982] |

### v2 / onset / in_distribution

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0519 [0.0396, 0.0777] | 0.0082 [0.0074, 0.0091] | 0.2657 [0.2158, 0.3194] |
| logistic_regression | 0.0908 [0.0658, 0.1346] | 0.0080 [0.0072, 0.0089] | 0.4010 [0.3431, 0.4656] |
| lightgbm | 0.1863 [0.1517, 0.2308] | 0.0075 [0.0067, 0.0084] | 0.7246 [0.6792, 0.7970] |
| replicator | 0.0792 [0.0626, 0.1059] | 0.0082 [0.0073, 0.0090] | 0.3816 [0.3333, 0.4416] |
| gru_6000 | 0.2504 [0.2114, 0.3071] | 0.0071 [0.0063, 0.0078] | 0.7633 [0.6985, 0.8182] |
| gru_6001 | 0.2487 [0.2116, 0.2991] | 0.0071 [0.0063, 0.0080] | 0.7971 [0.7416, 0.8525] |
| gru_6002 | 0.2432 [0.2040, 0.2945] | 0.0071 [0.0064, 0.0079] | 0.8019 [0.7357, 0.8598] |
| gru_6003 | 0.2527 [0.2123, 0.3076] | 0.0071 [0.0063, 0.0080] | 0.7488 [0.6958, 0.8083] |
| gru_6004 | 0.2239 [0.1856, 0.2711] | 0.0073 [0.0065, 0.0082] | 0.7053 [0.6346, 0.7730] |
| gru_ensemble | 0.2620 [0.2270, 0.3134] | 0.0070 [0.0062, 0.0078] | 0.7971 [0.7378, 0.8474] |

### v2 / onset / heldout_families

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | N/A | 0.0002 [0.0002, 0.0002] | N/A |
| logistic_regression | N/A | 0.0001 [0.0001, 0.0002] | N/A |
| lightgbm | N/A | 0.0001 [0.0000, 0.0002] | N/A |
| replicator | N/A | 0.0001 [0.0001, 0.0001] | N/A |
| gru_6000 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6001 | N/A | 0.0000 [0.0000, 0.0001] | N/A |
| gru_6002 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6003 | N/A | 0.0003 [0.0000, 0.0008] | N/A |
| gru_6004 | N/A | 0.0039 [0.0017, 0.0065] | N/A |
| gru_ensemble | N/A | 0.0002 [0.0001, 0.0003] | N/A |

### v2 / onset / heldout_impact

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0359 [0.0296, 0.0495] | 0.0118 [0.0108, 0.0129] | 0.1424 [0.1038, 0.1793] |
| logistic_regression | 0.0515 [0.0439, 0.0644] | 0.0128 [0.0117, 0.0138] | 0.1966 [0.1508, 0.2305] |
| lightgbm | 0.0865 [0.0718, 0.1075] | 0.0119 [0.0109, 0.0131] | 0.3763 [0.3255, 0.4346] |
| replicator | 0.0596 [0.0507, 0.0732] | 0.0118 [0.0108, 0.0128] | 0.3288 [0.2736, 0.3901] |
| gru_6000 | 0.1194 [0.1048, 0.1399] | 0.0120 [0.0110, 0.0130] | 0.5627 [0.5068, 0.6204] |
| gru_6001 | 0.1214 [0.1034, 0.1444] | 0.0118 [0.0108, 0.0129] | 0.5458 [0.4830, 0.6141] |
| gru_6002 | 0.1200 [0.1029, 0.1458] | 0.0117 [0.0107, 0.0127] | 0.5695 [0.5000, 0.6289] |
| gru_6003 | 0.1167 [0.1001, 0.1426] | 0.0123 [0.0112, 0.0133] | 0.5017 [0.4444, 0.5734] |
| gru_6004 | 0.1133 [0.1001, 0.1351] | 0.0128 [0.0117, 0.0139] | 0.5831 [0.5050, 0.6423] |
| gru_ensemble | 0.1264 [0.1106, 0.1490] | 0.0118 [0.0107, 0.0129] | 0.5864 [0.5241, 0.6556] |

### v2 / forced_flow / in_distribution

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0579 [0.0449, 0.0784] | 0.0431 [0.0362, 0.0513] | 0.0738 [0.0430, 0.1056] |
| logistic_regression | 0.0787 [0.0597, 0.1003] | 0.0429 [0.0362, 0.0500] | 0.1035 [0.0741, 0.1410] |
| lightgbm | 0.0884 [0.0687, 0.1133] | 0.0424 [0.0356, 0.0501] | 0.1296 [0.0919, 0.1696] |
| replicator | 0.0839 [0.0661, 0.1050] | 0.0425 [0.0357, 0.0495] | 0.1251 [0.0940, 0.1583] |
| gru_6000 | 0.1422 [0.1080, 0.1896] | 0.0412 [0.0353, 0.0484] | 0.2070 [0.1547, 0.2716] |
| gru_6001 | 0.1513 [0.1144, 0.1987] | 0.0410 [0.0347, 0.0472] | 0.2223 [0.1671, 0.2839] |
| gru_6002 | 0.1305 [0.1008, 0.1705] | 0.0414 [0.0352, 0.0483] | 0.1917 [0.1393, 0.2486] |
| gru_6003 | 0.1430 [0.1085, 0.1889] | 0.0410 [0.0349, 0.0484] | 0.2160 [0.1595, 0.2874] |
| gru_6004 | 0.1324 [0.1000, 0.1743] | 0.0418 [0.0353, 0.0492] | 0.2043 [0.1477, 0.2680] |
| gru_ensemble | 0.1437 [0.1084, 0.1907] | 0.0411 [0.0347, 0.0483] | 0.2124 [0.1638, 0.2800] |

### v2 / forced_flow / heldout_families

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0004 [0.0004, 0.0018] | 0.0025 [0.0019, 0.0034] | 0.0000 [0.0000, 0.0000] |
| logistic_regression | 0.0003 [0.0003, 0.0010] | 0.0023 [0.0017, 0.0033] | 0.0000 [0.0000, 0.0000] |
| lightgbm | 0.0005 [0.0005, 0.0019] | 0.0011 [0.0006, 0.0021] | 0.0000 [0.0000, 0.0000] |
| replicator | 0.0003 [0.0003, 0.0008] | 0.0033 [0.0028, 0.0044] | 0.0000 [0.0000, 0.0000] |
| gru_6000 | 0.0003 [0.0003, 0.0013] | 0.0005 [0.0000, 0.0015] | 0.0000 [0.0000, 0.0000] |
| gru_6001 | 0.0003 [0.0003, 0.0013] | 0.0005 [0.0000, 0.0016] | 0.0000 [0.0000, 0.0000] |
| gru_6002 | 0.0003 [0.0003, 0.0011] | 0.0005 [0.0000, 0.0015] | 0.0000 [0.0000, 0.0000] |
| gru_6003 | 0.0003 [0.0003, 0.0013] | 0.0005 [0.0000, 0.0015] | 0.0000 [0.0000, 0.0000] |
| gru_6004 | 0.0003 [0.0003, 0.0010] | 0.0013 [0.0006, 0.0025] | 0.0000 [0.0000, 0.0000] |
| gru_ensemble | 0.0003 [0.0003, 0.0010] | 0.0006 [0.0000, 0.0015] | 0.0000 [0.0000, 0.0000] |

### v2 / forced_flow / heldout_impact

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0881 [0.0719, 0.1082] | 0.0714 [0.0624, 0.0811] | 0.0595 [0.0316, 0.0868] |
| logistic_regression | 0.1188 [0.0999, 0.1415] | 0.0711 [0.0623, 0.0801] | 0.1079 [0.0858, 0.1356] |
| lightgbm | 0.1326 [0.1090, 0.1607] | 0.0708 [0.0617, 0.0814] | 0.1287 [0.1002, 0.1572] |
| replicator | 0.1545 [0.1278, 0.1835] | 0.0699 [0.0607, 0.0795] | 0.1361 [0.1081, 0.1663] |
| gru_6000 | 0.1689 [0.1375, 0.2095] | 0.0706 [0.0613, 0.0801] | 0.1781 [0.1449, 0.2203] |
| gru_6001 | 0.1639 [0.1332, 0.2007] | 0.0701 [0.0604, 0.0798] | 0.1722 [0.1382, 0.2087] |
| gru_6002 | 0.1496 [0.1218, 0.1818] | 0.0716 [0.0615, 0.0814] | 0.1489 [0.1126, 0.1856] |
| gru_6003 | 0.1530 [0.1251, 0.1867] | 0.0713 [0.0617, 0.0813] | 0.1669 [0.1303, 0.2006] |
| gru_6004 | 0.1051 [0.0867, 0.1329] | 0.0728 [0.0636, 0.0817] | 0.0898 [0.0576, 0.1257] |
| gru_ensemble | 0.1508 [0.1237, 0.1822] | 0.0710 [0.0621, 0.0810] | 0.1568 [0.1240, 0.1928] |

### architecture / default / in_distribution

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0756 [0.0361, 0.1182] | 0.0330 [0.0204, 0.0495] | 0.1466 [0.0937, 0.2009] |
| logistic_regression | 0.8826 [0.7848, 0.9320] | 0.0095 [0.0065, 0.0123] | 0.9303 [0.8543, 0.9691] |
| lightgbm | 0.9052 [0.8052, 0.9475] | 0.0081 [0.0053, 0.0110] | 0.9409 [0.8681, 0.9767] |
| replicator | 0.0587 [0.0304, 0.0893] | 0.0332 [0.0192, 0.0485] | 0.0910 [0.0673, 0.1292] |
| gru_6000 | 0.9089 [0.8214, 0.9487] | 0.0081 [0.0054, 0.0110] | 0.9515 [0.8898, 0.9831] |
| gru_6001 | 0.9126 [0.8150, 0.9520] | 0.0080 [0.0053, 0.0107] | 0.9574 [0.9034, 0.9875] |
| gru_6002 | 0.9072 [0.8100, 0.9508] | 0.0081 [0.0055, 0.0109] | 0.9527 [0.8838, 0.9823] |
| gru_6003 | 0.9055 [0.7954, 0.9489] | 0.0081 [0.0056, 0.0108] | 0.9480 [0.8761, 0.9849] |
| gru_6004 | 0.9105 [0.8162, 0.9519] | 0.0079 [0.0054, 0.0111] | 0.9527 [0.8884, 0.9846] |
| gru_ensemble | 0.9115 [0.8118, 0.9522] | 0.0079 [0.0051, 0.0106] | 0.9551 [0.8922, 0.9862] |

### architecture / default / heldout_architecture

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0396 [0.0140, 0.0772] | 0.0163 [0.0094, 0.0244] | 0.1772 [0.1069, 0.2664] |
| logistic_regression | 0.5674 [0.2535, 0.7257] | 0.0099 [0.0054, 0.0146] | 0.8492 [0.6837, 0.9460] |
| lightgbm | 0.6032 [0.3195, 0.7591] | 0.0090 [0.0053, 0.0134] | 0.9339 [0.8531, 0.9767] |
| replicator | 0.0223 [0.0076, 0.0437] | 0.0167 [0.0098, 0.0250] | 0.0926 [0.0487, 0.1317] |
| gru_6000 | 0.6409 [0.3754, 0.7816] | 0.0103 [0.0066, 0.0147] | 0.8942 [0.7908, 0.9586] |
| gru_6001 | 0.6153 [0.3608, 0.7673] | 0.0119 [0.0072, 0.0164] | 0.8942 [0.7674, 0.9573] |
| gru_6002 | 0.6399 [0.3424, 0.7802] | 0.0091 [0.0054, 0.0133] | 0.9127 [0.8063, 0.9688] |
| gru_6003 | 0.6727 [0.4273, 0.8003] | 0.0082 [0.0048, 0.0123] | 0.9233 [0.8401, 0.9727] |
| gru_6004 | 0.6091 [0.3104, 0.7556] | 0.0124 [0.0074, 0.0184] | 0.8968 [0.7447, 0.9607] |
| gru_ensemble | 0.6571 [0.4096, 0.7874] | 0.0093 [0.0056, 0.0133] | 0.9153 [0.8196, 0.9672] |

### architecture / onset / in_distribution

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0081 [0.0032, 0.0274] | 0.0033 [0.0022, 0.0046] | 0.1111 [0.0308, 0.1852] |
| logistic_regression | 0.0445 [0.0267, 0.0777] | 0.0032 [0.0021, 0.0043] | 0.6049 [0.4682, 0.7408] |
| lightgbm | 0.1197 [0.0908, 0.1732] | 0.0031 [0.0021, 0.0043] | 0.7901 [0.6795, 0.8831] |
| replicator | 0.0078 [0.0042, 0.0138] | 0.0033 [0.0022, 0.0045] | 0.1481 [0.0550, 0.2468] |
| gru_6000 | 0.1341 [0.0863, 0.1970] | 0.0031 [0.0021, 0.0042] | 0.7778 [0.6407, 0.8983] |
| gru_6001 | 0.1297 [0.0802, 0.1952] | 0.0031 [0.0021, 0.0043] | 0.8025 [0.6614, 0.9057] |
| gru_6002 | 0.1288 [0.0821, 0.1944] | 0.0031 [0.0021, 0.0042] | 0.7531 [0.6216, 0.8730] |
| gru_6003 | 0.1342 [0.0795, 0.1900] | 0.0031 [0.0020, 0.0042] | 0.8025 [0.6508, 0.8990] |
| gru_6004 | 0.1277 [0.0789, 0.1930] | 0.0031 [0.0021, 0.0041] | 0.7778 [0.6351, 0.9033] |
| gru_ensemble | 0.1405 [0.0896, 0.2026] | 0.0031 [0.0020, 0.0041] | 0.8025 [0.6713, 0.9216] |

### architecture / onset / heldout_architecture

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0099 [0.0059, 0.0218] | 0.0040 [0.0023, 0.0058] | 0.1735 [0.0926, 0.2676] |
| logistic_regression | 0.0672 [0.0466, 0.0886] | 0.0039 [0.0022, 0.0056] | 0.6837 [0.5384, 0.8250] |
| lightgbm | 0.0750 [0.0500, 0.1180] | 0.0045 [0.0028, 0.0065] | 0.7347 [0.5999, 0.8556] |
| replicator | 0.0073 [0.0038, 0.0138] | 0.0040 [0.0024, 0.0057] | 0.1735 [0.0574, 0.3054] |
| gru_6000 | 0.0924 [0.0535, 0.1548] | 0.0041 [0.0024, 0.0059] | 0.7959 [0.6407, 0.9222] |
| gru_6001 | 0.1184 [0.0833, 0.1759] | 0.0039 [0.0023, 0.0057] | 0.8367 [0.7294, 0.9250] |
| gru_6002 | 0.1089 [0.0738, 0.1625] | 0.0040 [0.0024, 0.0057] | 0.8469 [0.7071, 0.9467] |
| gru_6003 | 0.1287 [0.0833, 0.1938] | 0.0039 [0.0021, 0.0054] | 0.8367 [0.7216, 0.9375] |
| gru_6004 | 0.0948 [0.0556, 0.1566] | 0.0039 [0.0023, 0.0055] | 0.7755 [0.6465, 0.9181] |
| gru_ensemble | 0.1242 [0.0823, 0.1887] | 0.0038 [0.0023, 0.0055] | 0.8571 [0.7442, 0.9508] |

### architecture / forced_flow / in_distribution

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0835 [0.0569, 0.1220] | 0.0477 [0.0392, 0.0579] | 0.1188 [0.0595, 0.1852] |
| logistic_regression | 0.0924 [0.0669, 0.1230] | 0.0474 [0.0385, 0.0573] | 0.1268 [0.0794, 0.1936] |
| lightgbm | 0.0950 [0.0662, 0.1279] | 0.0474 [0.0390, 0.0563] | 0.1356 [0.0984, 0.1837] |
| replicator | 0.0903 [0.0680, 0.1137] | 0.0474 [0.0379, 0.0568] | 0.1164 [0.0904, 0.1431] |
| gru_6000 | 0.1147 [0.0806, 0.1575] | 0.0473 [0.0391, 0.0558] | 0.1677 [0.1119, 0.2227] |
| gru_6001 | 0.1139 [0.0804, 0.1634] | 0.0470 [0.0383, 0.0562] | 0.1910 [0.1208, 0.2642] |
| gru_6002 | 0.1161 [0.0774, 0.1621] | 0.0469 [0.0386, 0.0558] | 0.1870 [0.1344, 0.2399] |
| gru_6003 | 0.1332 [0.0878, 0.1937] | 0.0467 [0.0382, 0.0557] | 0.1894 [0.1315, 0.2516] |
| gru_6004 | 0.1160 [0.0817, 0.1567] | 0.0473 [0.0388, 0.0560] | 0.1750 [0.1267, 0.2226] |
| gru_ensemble | 0.1226 [0.0845, 0.1705] | 0.0468 [0.0388, 0.0550] | 0.1894 [0.1323, 0.2477] |

### architecture / forced_flow / heldout_architecture

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1192 [0.0796, 0.1721] | 0.0432 [0.0351, 0.0522] | 0.1870 [0.1147, 0.2526] |
| logistic_regression | 0.1168 [0.0820, 0.1589] | 0.0432 [0.0346, 0.0518] | 0.1670 [0.1040, 0.2305] |
| lightgbm | 0.1280 [0.0882, 0.1786] | 0.0430 [0.0348, 0.0519] | 0.1974 [0.1403, 0.2537] |
| replicator | 0.1046 [0.0788, 0.1393] | 0.0435 [0.0352, 0.0530] | 0.1557 [0.1194, 0.1958] |
| gru_6000 | 0.1528 [0.1071, 0.1984] | 0.0422 [0.0347, 0.0510] | 0.2452 [0.1845, 0.3033] |
| gru_6001 | 0.1637 [0.1165, 0.2064] | 0.0419 [0.0344, 0.0500] | 0.2591 [0.1876, 0.3182] |
| gru_6002 | 0.1588 [0.1109, 0.2046] | 0.0420 [0.0347, 0.0506] | 0.2678 [0.1876, 0.3274] |
| gru_6003 | 0.1642 [0.1141, 0.2082] | 0.0418 [0.0334, 0.0497] | 0.2643 [0.1836, 0.3296] |
| gru_6004 | 0.1558 [0.1040, 0.2062] | 0.0421 [0.0346, 0.0501] | 0.2487 [0.1786, 0.3058] |
| gru_ensemble | 0.1646 [0.1134, 0.2089] | 0.0419 [0.0341, 0.0500] | 0.2548 [0.1866, 0.3194] |

## Per-seed GRU PR-AUC ranges

The range is the minimum–maximum of the five per-seed point estimates; each contributing per-seed estimate and CI appears in the observer tables above.

| Cohort | Label | Split | GRU seed min–max PR-AUC |
| --- | --- | --- | ---: |
| v2 | default | in_distribution | 0.7779–0.7971 |
| v2 | default | heldout_families | N/A |
| v2 | default | heldout_impact | 0.6672–0.7598 |
| v2 | onset | in_distribution | 0.2239–0.2527 |
| v2 | onset | heldout_families | N/A |
| v2 | onset | heldout_impact | 0.1133–0.1214 |
| v2 | forced_flow | in_distribution | 0.1305–0.1513 |
| v2 | forced_flow | heldout_families | 0.0003–0.0003 |
| v2 | forced_flow | heldout_impact | 0.1051–0.1689 |
| architecture | default | in_distribution | 0.9055–0.9126 |
| architecture | default | heldout_architecture | 0.6091–0.6727 |
| architecture | onset | in_distribution | 0.1277–0.1342 |
| architecture | onset | heldout_architecture | 0.0924–0.1287 |
| architecture | forced_flow | in_distribution | 0.1139–0.1332 |
| architecture | forced_flow | heldout_architecture | 0.1528–0.1642 |

## Paired PR-AUC differences

Entries are GRU ensemble minus comparator, with paired whole-episode-bootstrap 95% CIs.

| Cohort | Label | Split | Minus replicator (95% CI) | Minus LightGBM (95% CI) | Minus trivial (95% CI) |
| --- | --- | --- | ---: | ---: | ---: |
| v2 | default | in_distribution | 0.5676 [0.5344, 0.5977] | 0.0909 [0.0724, 0.1098] | 0.6954 [0.6610, 0.7243] |
| v2 | default | heldout_families | N/A | N/A | N/A |
| v2 | default | heldout_impact | 0.4175 [0.3823, 0.4554] | 0.1315 [0.1077, 0.1564] | 0.5604 [0.5266, 0.5902] |
| v2 | onset | in_distribution | 0.1827 [0.1421, 0.2309] | 0.0756 [0.0381, 0.1222] | 0.2101 [0.1715, 0.2569] |
| v2 | onset | heldout_families | N/A | N/A | N/A |
| v2 | onset | heldout_impact | 0.0668 [0.0490, 0.0869] | 0.0399 [0.0208, 0.0578] | 0.0905 [0.0757, 0.1069] |
| v2 | forced_flow | in_distribution | 0.0597 [0.0358, 0.0987] | 0.0552 [0.0293, 0.0907] | 0.0858 [0.0526, 0.1313] |
| v2 | forced_flow | heldout_families | 0.0000 [0.0000, 0.0002] | -0.0002 [-0.0006, -0.0001] | -0.0001 [-0.0005, -0.0001] |
| v2 | forced_flow | heldout_impact | -0.0037 [-0.0232, 0.0177] | 0.0182 [0.0030, 0.0371] | 0.0627 [0.0335, 0.0979] |
| architecture | default | in_distribution | 0.8529 [0.7712, 0.8842] | 0.0063 [0.0012, 0.0146] | 0.8359 [0.7645, 0.8655] |
| architecture | default | heldout_architecture | 0.6348 [0.4141, 0.7551] | 0.0539 [0.0272, 0.0995] | 0.6175 [0.3823, 0.7314] |
| architecture | onset | in_distribution | 0.1327 [0.0853, 0.1934] | 0.0208 [-0.0375, 0.0719] | 0.1325 [0.0814, 0.1843] |
| architecture | onset | heldout_architecture | 0.1170 [0.0770, 0.1741] | 0.0492 [0.0115, 0.1013] | 0.1143 [0.0716, 0.1754] |
| architecture | forced_flow | in_distribution | 0.0324 [0.0064, 0.0716] | 0.0277 [0.0020, 0.0613] | 0.0391 [0.0122, 0.0754] |
| architecture | forced_flow | heldout_architecture | 0.0599 [0.0260, 0.0927] | 0.0366 [0.0131, 0.0602] | 0.0454 [0.0054, 0.0743] |

## GRU training detail

- v2/default: seed 6000: epochs=40, best_validation_loss=0.076012301; seed 6001: epochs=40, best_validation_loss=0.073237447; seed 6002: epochs=40, best_validation_loss=0.08339832; seed 6003: epochs=40, best_validation_loss=0.073703222; seed 6004: epochs=40, best_validation_loss=0.073618495.
- v2/onset: seed 6000: epochs=40, best_validation_loss=0.033287956; seed 6001: epochs=40, best_validation_loss=0.028064075; seed 6002: epochs=40, best_validation_loss=0.032165728; seed 6003: epochs=40, best_validation_loss=0.031811254; seed 6004: epochs=40, best_validation_loss=0.032155045.
- v2/forced_flow: seed 6000: epochs=40, best_validation_loss=0.16356805; seed 6001: epochs=40, best_validation_loss=0.1559384; seed 6002: epochs=40, best_validation_loss=0.18637539; seed 6003: epochs=40, best_validation_loss=0.17053133; seed 6004: epochs=16, best_validation_loss=0.17352578.
- architecture/default: seed 6000: epochs=40, best_validation_loss=0.043296005; seed 6001: epochs=40, best_validation_loss=0.028970291; seed 6002: epochs=40, best_validation_loss=0.052249542; seed 6003: epochs=40, best_validation_loss=0.040415245; seed 6004: epochs=39, best_validation_loss=0.048254145.
- architecture/onset: seed 6000: epochs=40, best_validation_loss=0.018823618; seed 6001: epochs=40, best_validation_loss=0.012384614; seed 6002: epochs=40, best_validation_loss=0.022319196; seed 6003: epochs=40, best_validation_loss=0.019127707; seed 6004: epochs=40, best_validation_loss=0.020221928.
- architecture/forced_flow: seed 6000: epochs=40, best_validation_loss=0.16396214; seed 6001: epochs=38, best_validation_loss=0.19750617; seed 6002: epochs=21, best_validation_loss=0.17341491; seed 6003: epochs=40, best_validation_loss=0.16679435; seed 6004: epochs=40, best_validation_loss=0.20215697.

## Pre-registered rule outcomes (verbatim)

- Strategy-replication rule — v2/in_distribution/default: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — v2/heldout_families/default: the paired GRU-ensemble-minus-replicator PR-AUC interval is undefined, so the rule cannot establish an observer advantage.
- Strategy-replication rule — v2/heldout_impact/default: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — v2/in_distribution/onset: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — v2/heldout_families/onset: the paired GRU-ensemble-minus-replicator PR-AUC interval is undefined, so the rule cannot establish an observer advantage.
- Strategy-replication rule — v2/heldout_impact/onset: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — v2/in_distribution/forced_flow: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — v2/heldout_families/forced_flow: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — v2/heldout_impact/forced_flow: the paired GRU-ensemble-minus-replicator PR-AUC interval includes or lies below 0; the observer's advantage is reported as consistent with strategy replication.
- Strategy-replication rule — architecture/in_distribution/default: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — architecture/heldout_architecture/default: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — architecture/in_distribution/onset: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — architecture/heldout_architecture/onset: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — architecture/in_distribution/forced_flow: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — architecture/heldout_architecture/forced_flow: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- H2-style generalization rule — held-out architecture/default: the base rate is 0.015429, and the GRU ensemble beat both the trivial reference and replicator with paired intervals excluding 0 above; the pre-registered rule finds evidence for H2-style generalization in this simulator.
- Forced-flow power rule — v2/in_distribution: 1111 test events meets the 30-event minimum; the result is not flagged as underpowered by this rule.
- Forced-flow power rule — v2/heldout_families: 12 test events is fewer than 30; the result is reported as underpowered.
- Forced-flow power rule — v2/heldout_impact: 1881 test events meets the 30-event minimum; the result is not flagged as underpowered by this rule.
- Forced-flow power rule — architecture/in_distribution: 1246 test events meets the 30-event minimum; the result is not flagged as underpowered by this rule.
- Forced-flow power rule — architecture/heldout_architecture: 1150 test events meets the 30-event minimum; the result is not flagged as underpowered by this rule.

## Caveats, deviations, and open questions

- Observer inputs contain only public records. Agent-only flow and forced flags are reduced to labels before observer construction.
- Thresholds use training episodes only. Confidence intervals resample whole evaluation episodes, never overlapping time steps independently.
- No simulator setting, label rule, observer hyperparameter, seed range, or episode count was changed after viewing round-3 metrics.
- No deviations from the pre-registration and no open questions.
