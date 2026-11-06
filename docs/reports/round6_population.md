# Round 6: population decomposition

**Synthetic-simulator disclaimer:** Uncalibrated synthetic HerdLab results only; no evidence about real markets, funds, or live trading.

## Configuration, seeds, and runtime

Pre-run estimate: 45 minutes for R6-2, plus a 60-minute planning reserve for R6-5; combined 105 minutes versus the approximately 130-minute limit. R6-5 is not run here.

Measured invocation through analysis: 25.80 minutes. Retained condition compute: 25.80 minutes. Workers: 10 simulation, 3 GRU.

Each population uses 200 train seeds 8000–8199 and 50 test seeds 8500–8549, 500 steps each. Signals off; horizon 10; separate training-only 0.95 quantile thresholds for default (primary) and forced_flow (secondary). Return lookback 10; GRU seeds 6000–6002; LightGBM and replicator seeds 5000 + label index. Observer hyperparameters are unchanged from round 5.

All intervals are 95% percentile CIs with 1,000 whole-test-episode resamples. Shared test seeds are resampled jointly for gain contrasts. Gain = GRU ensemble PR-AUC minus trivial PR-AUC. Metric seed: 9000 + 100000 × population index + round-4 helper offsets; base-rate seed: 2000000 + 10 × population index + label index; contrast seed: 6000000 + 100 × label index + contrast index. Indices start at zero in displayed order.

The unchanged round-5 runner also computes ancillary logistic and individual-GRU metrics; the four registered observers are reported below. Unused forecast-scale diagnostics are not observer inputs. Atomic condition caches live in runs/round6_population_cache; fingerprints cover resolved configs, seed sets and Python source. The baseline is freshly fitted unless a matching round-6 cache exists.

## Baseline reproducibility

LightGBM default PR-AUC: 0.9510 [0.9281, 0.9653]; reference 0.9510; PASS (four decimal places).

## Base rates and thresholds

| Population | Label | Threshold | Train positives / labels | Train rate | Test positives / labels | Test rate (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | default | 0.122034186585 | 4900/98000 | 0.050000 | 1224/24500 | 0.0500 [0.0302, 0.0705] |
| baseline | forced_flow | 0.000282089357967 | 4900/98000 | 0.050000 | 1519/24500 | 0.0620 [0.0493, 0.0744] |
| without_momentum | default | 0.0155562160002 | 4900/98000 | 0.050000 | 1523/24500 | 0.0622 [0.0347, 0.0961] |
| without_momentum | forced_flow | 0 | 532/98000 | 0.005429 | 142/24500 | 0.0058 [0.0018, 0.0109] |
| without_reversion | default | 0.160283299129 | 4900/98000 | 0.050000 | 1192/24500 | 0.0487 [0.0405, 0.0577] |
| without_reversion | forced_flow | 0.000643408242852 | 4900/98000 | 0.050000 | 1311/24500 | 0.0535 [0.0451, 0.0630] |
| without_linear | default | 0.0750875715091 | 4900/98000 | 0.050000 | 1000/24500 | 0.0408 [0.0231, 0.0596] |
| without_linear | forced_flow | 0 | 4361/98000 | 0.044500 | 1347/24500 | 0.0550 [0.0409, 0.0689] |
| without_gbdt | default | 0.105845243002 | 4900/98000 | 0.050000 | 1207/24500 | 0.0493 [0.0304, 0.0694] |
| without_gbdt | forced_flow | 0.000158454421353 | 4900/98000 | 0.050000 | 1457/24500 | 0.0595 [0.0467, 0.0734] |
| without_mlp | default | 0.110209340623 | 4900/98000 | 0.050000 | 1148/24500 | 0.0469 [0.0318, 0.0631] |
| without_mlp | forced_flow | 0.000117167541744 | 4900/98000 | 0.050000 | 1644/24500 | 0.0671 [0.0513, 0.0827] |
| round3_v2 | default | 0.157103643788 | 4900/98000 | 0.050000 | 1334/24500 | 0.0544 [0.0472, 0.0619] |
| round3_v2 | forced_flow | 0.000865837829986 | 4900/98000 | 0.050000 | 1425/24500 | 0.0582 [0.0488, 0.0673] |

## Observer metrics

PR-AUC and recall are higher-is-better; Brier is lower-is-better. Recall uses FAR <= 0.05.

### baseline / default

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0897 [0.0528, 0.1324] | 0.0469 [0.0296, 0.0628] | 0.0923 [0.0520, 0.1365] |
| lightgbm | 0.9510 [0.9281, 0.9653] | 0.0082 [0.0055, 0.0115] | 0.9926 [0.9806, 0.9985] |
| replicator | 0.0831 [0.0494, 0.1203] | 0.0466 [0.0308, 0.0644] | 0.0891 [0.0639, 0.1186] |
| gru_ensemble | 0.9564 [0.9355, 0.9694] | 0.0077 [0.0049, 0.0108] | 0.9943 [0.9846, 0.9992] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.8733 [0.8344, 0.9049] |
| lightgbm | 0.0054 [0.0010, 0.0111] |
| trivial_reference | 0.8666 [0.8266, 0.8968] |

GRU seed PR-AUC range: (0.9541536246793603, 0.9562191439170422).

Training: 6000: epochs=40, best_validation_loss=0.033522398; 6001: epochs=40, best_validation_loss=0.037296954; 6002: epochs=40, best_validation_loss=0.030605735.

### baseline / forced_flow

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1079 [0.0804, 0.1405] | 0.0571 [0.0457, 0.0693] | 0.1099 [0.0655, 0.1600] |
| lightgbm | 0.1100 [0.0830, 0.1404] | 0.0568 [0.0464, 0.0684] | 0.1284 [0.0923, 0.1602] |
| replicator | 0.1186 [0.0857, 0.1584] | 0.0567 [0.0458, 0.0675] | 0.1442 [0.0717, 0.1881] |
| gru_ensemble | 0.1573 [0.1167, 0.2053] | 0.0554 [0.0463, 0.0647] | 0.2021 [0.1446, 0.2610] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0387 [0.0127, 0.0735] |
| lightgbm | 0.0472 [0.0249, 0.0761] |
| trivial_reference | 0.0494 [0.0178, 0.0849] |

GRU seed PR-AUC range: (0.11802803300915274, 0.1596684543289827).

Training: 6000: epochs=40, best_validation_loss=0.1673529; 6001: epochs=9, best_validation_loss=0.19388598; 6002: epochs=26, best_validation_loss=0.19049957.

### without_momentum / default

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1713 [0.1525, 0.1977] | 0.0557 [0.0300, 0.0833] | 0.1950 [0.1264, 0.3717] |
| lightgbm | 0.6264 [0.5588, 0.6766] | 0.0342 [0.0201, 0.0494] | 0.7525 [0.6789, 0.8166] |
| replicator | 0.2953 [0.2067, 0.3717] | 0.0495 [0.0283, 0.0733] | 0.3972 [0.3009, 0.5347] |
| gru_ensemble | 0.7291 [0.6294, 0.7955] | 0.0279 [0.0184, 0.0394] | 0.8575 [0.7880, 0.9237] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.4338 [0.3408, 0.5593] |
| lightgbm | 0.1026 [0.0609, 0.1354] |
| trivial_reference | 0.5578 [0.4582, 0.6324] |

GRU seed PR-AUC range: (0.6773901808508737, 0.7304958649227438).

Training: 6000: epochs=40, best_validation_loss=0.066054154; 6001: epochs=40, best_validation_loss=0.11488289; 6002: epochs=24, best_validation_loss=0.1012809.

### without_momentum / forced_flow

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0059 [0.0020, 0.0117] | 0.0058 [0.0017, 0.0102] | 0.0000 [0.0000, 0.0000] |
| lightgbm | 0.0116 [0.0039, 0.0229] | 0.0058 [0.0018, 0.0107] | 0.0986 [0.0178, 0.1620] |
| replicator | 0.0128 [0.0042, 0.0268] | 0.0058 [0.0017, 0.0103] | 0.1620 [0.0595, 0.2627] |
| gru_ensemble | 0.0113 [0.0026, 0.0234] | 0.0058 [0.0018, 0.0108] | 0.1197 [0.0143, 0.2286] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | -0.0014 [-0.0064, 0.0029] |
| lightgbm | -0.0003 [-0.0034, 0.0047] |
| trivial_reference | 0.0054 [0.0008, 0.0140] |

GRU seed PR-AUC range: (0.005048329405949766, 0.019358103601541412).

Training: 6000: epochs=9, best_validation_loss=0.040885875; 6001: epochs=23, best_validation_loss=0.023848521; 6002: epochs=40, best_validation_loss=0.030342426.

### without_reversion / default

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0880 [0.0687, 0.1103] | 0.0456 [0.0390, 0.0529] | 0.1216 [0.0985, 0.1482] |
| lightgbm | 0.7528 [0.6716, 0.8163] | 0.0216 [0.0185, 0.0255] | 0.8666 [0.8039, 0.9121] |
| replicator | 0.1224 [0.0963, 0.1538] | 0.0444 [0.0384, 0.0511] | 0.1795 [0.1405, 0.2129] |
| gru_ensemble | 0.8042 [0.7370, 0.8551] | 0.0189 [0.0162, 0.0219] | 0.9195 [0.8574, 0.9575] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.6818 [0.6276, 0.7255] |
| lightgbm | 0.0513 [0.0332, 0.0775] |
| trivial_reference | 0.7161 [0.6547, 0.7597] |

GRU seed PR-AUC range: (0.7925309286864965, 0.801654811385359).

Training: 6000: epochs=40, best_validation_loss=0.062601345; 6001: epochs=40, best_validation_loss=0.062830755; 6002: epochs=40, best_validation_loss=0.06078188.

### without_reversion / forced_flow

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0974 [0.0674, 0.1344] | 0.0498 [0.0413, 0.0577] | 0.1159 [0.0627, 0.1735] |
| lightgbm | 0.1176 [0.0963, 0.1428] | 0.0492 [0.0412, 0.0576] | 0.1564 [0.1245, 0.1924] |
| replicator | 0.0922 [0.0706, 0.1217] | 0.0498 [0.0416, 0.0584] | 0.1030 [0.0674, 0.1401] |
| gru_ensemble | 0.1225 [0.0963, 0.1860] | 0.0526 [0.0429, 0.0651] | 0.1632 [0.0856, 0.2508] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0303 [-0.0004, 0.0844] |
| lightgbm | 0.0048 [-0.0210, 0.0568] |
| trivial_reference | 0.0251 [-0.0171, 0.0884] |

GRU seed PR-AUC range: (0.11685505238140467, 0.12846794251065102).

Training: 6000: epochs=40, best_validation_loss=0.17962256; 6001: epochs=40, best_validation_loss=0.17798681; 6002: epochs=40, best_validation_loss=0.18434502.

### without_linear / default

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0713 [0.0405, 0.1074] | 0.0388 [0.0230, 0.0549] | 0.1080 [0.0726, 0.1541] |
| lightgbm | 0.8868 [0.7752, 0.9399] | 0.0103 [0.0070, 0.0139] | 0.9070 [0.8298, 0.9499] |
| replicator | 0.0573 [0.0336, 0.0880] | 0.0390 [0.0242, 0.0583] | 0.0650 [0.0418, 0.1016] |
| gru_ensemble | 0.9103 [0.8089, 0.9502] | 0.0092 [0.0066, 0.0123] | 0.9380 [0.8810, 0.9708] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.8530 [0.7788, 0.8822] |
| lightgbm | 0.0235 [0.0098, 0.0481] |
| trivial_reference | 0.8390 [0.7626, 0.8738] |

GRU seed PR-AUC range: (0.9047568776226531, 0.9073452404276732).

Training: 6000: epochs=40, best_validation_loss=0.046726987; 6001: epochs=40, best_validation_loss=0.055365929; 6002: epochs=40, best_validation_loss=0.04815784.

### without_linear / forced_flow

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1211 [0.0896, 0.1663] | 0.0504 [0.0394, 0.0618] | 0.1529 [0.0849, 0.2358] |
| lightgbm | 0.1332 [0.1040, 0.1634] | 0.0499 [0.0388, 0.0623] | 0.1752 [0.1255, 0.2264] |
| replicator | 0.1297 [0.1001, 0.1589] | 0.0500 [0.0383, 0.0611] | 0.1188 [0.0842, 0.1872] |
| gru_ensemble | 0.1748 [0.1341, 0.2178] | 0.0482 [0.0377, 0.0602] | 0.2591 [0.1781, 0.3491] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0450 [0.0143, 0.0808] |
| lightgbm | 0.0416 [0.0071, 0.0826] |
| trivial_reference | 0.0536 [0.0157, 0.0884] |

GRU seed PR-AUC range: (0.16830939929442085, 0.1744048101597456).

Training: 6000: epochs=17, best_validation_loss=0.16362481; 6001: epochs=27, best_validation_loss=0.15978374; 6002: epochs=14, best_validation_loss=0.1764232.

### without_gbdt / default

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0734 [0.0447, 0.1059] | 0.0466 [0.0315, 0.0649] | 0.0713 [0.0436, 0.1078] |
| lightgbm | 0.9283 [0.8849, 0.9526] | 0.0100 [0.0070, 0.0128] | 0.9743 [0.9436, 0.9936] |
| replicator | 0.0794 [0.0463, 0.1129] | 0.0461 [0.0302, 0.0642] | 0.0862 [0.0653, 0.1183] |
| gru_ensemble | 0.9307 [0.8886, 0.9556] | 0.0094 [0.0066, 0.0126] | 0.9685 [0.9284, 0.9924] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.8513 [0.8126, 0.8791] |
| lightgbm | 0.0024 [-0.0045, 0.0092] |
| trivial_reference | 0.8573 [0.8190, 0.8825] |

GRU seed PR-AUC range: (0.9283301706360803, 0.9307978041795051).

Training: 6000: epochs=40, best_validation_loss=0.033329486; 6001: epochs=40, best_validation_loss=0.047630712; 6002: epochs=40, best_validation_loss=0.039628682.

### without_gbdt / forced_flow

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1076 [0.0700, 0.1670] | 0.0551 [0.0436, 0.0671] | 0.1434 [0.0793, 0.2111] |
| lightgbm | 0.1069 [0.0793, 0.1365] | 0.0549 [0.0439, 0.0663] | 0.1359 [0.0941, 0.1753] |
| replicator | 0.1022 [0.0746, 0.1331] | 0.0548 [0.0435, 0.0659] | 0.1208 [0.0843, 0.1570] |
| gru_ensemble | 0.1185 [0.0883, 0.1539] | 0.0543 [0.0442, 0.0651] | 0.1428 [0.0864, 0.1941] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0162 [-0.0103, 0.0478] |
| lightgbm | 0.0116 [-0.0070, 0.0324] |
| trivial_reference | 0.0109 [-0.0317, 0.0416] |

GRU seed PR-AUC range: (0.11341482355934472, 0.11889155741103892).

Training: 6000: epochs=22, best_validation_loss=0.19737004; 6001: epochs=10, best_validation_loss=0.18922643; 6002: epochs=13, best_validation_loss=0.18471569.

### without_mlp / default

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0967 [0.0604, 0.1386] | 0.0441 [0.0322, 0.0576] | 0.1315 [0.0798, 0.1864] |
| lightgbm | 0.8899 [0.8534, 0.9156] | 0.0125 [0.0089, 0.0164] | 0.9817 [0.9658, 0.9957] |
| replicator | 0.1094 [0.0687, 0.1560] | 0.0434 [0.0301, 0.0567] | 0.1551 [0.0900, 0.2510] |
| gru_ensemble | 0.9010 [0.8700, 0.9248] | 0.0119 [0.0082, 0.0154] | 0.9878 [0.9741, 0.9974] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.7916 [0.7392, 0.8348] |
| lightgbm | 0.0111 [0.0052, 0.0186] |
| trivial_reference | 0.8043 [0.7603, 0.8404] |

GRU seed PR-AUC range: (0.8971403805469711, 0.8996376074918718).

Training: 6000: epochs=40, best_validation_loss=0.041974311; 6001: epochs=40, best_validation_loss=0.041248541; 6002: epochs=40, best_validation_loss=0.052680938.

### without_mlp / forced_flow

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1127 [0.0822, 0.1545] | 0.0616 [0.0489, 0.0758] | 0.1247 [0.0716, 0.1834] |
| lightgbm | 0.1105 [0.0809, 0.1425] | 0.0613 [0.0487, 0.0755] | 0.1034 [0.0763, 0.1405] |
| replicator | 0.1098 [0.0822, 0.1398] | 0.0615 [0.0489, 0.0744] | 0.0998 [0.0690, 0.1321] |
| gru_ensemble | 0.1379 [0.0972, 0.1840] | 0.0603 [0.0479, 0.0723] | 0.1490 [0.0830, 0.2004] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0281 [0.0008, 0.0563] |
| lightgbm | 0.0274 [0.0026, 0.0571] |
| trivial_reference | 0.0252 [-0.0106, 0.0594] |

GRU seed PR-AUC range: (0.1353890711584908, 0.13850291558865802).

Training: 6000: epochs=22, best_validation_loss=0.18552051; 6001: epochs=30, best_validation_loss=0.19457001; 6002: epochs=17, best_validation_loss=0.19281449.

### round3_v2 / default

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1031 [0.0923, 0.1166] | 0.0506 [0.0444, 0.0573] | 0.1717 [0.1442, 0.1972] |
| lightgbm | 0.6935 [0.6595, 0.7270] | 0.0276 [0.0244, 0.0306] | 0.8373 [0.8038, 0.8635] |
| replicator | 0.2123 [0.1830, 0.2443] | 0.0467 [0.0411, 0.0527] | 0.2616 [0.2269, 0.2957] |
| gru_ensemble | 0.7647 [0.7278, 0.7979] | 0.0229 [0.0207, 0.0249] | 0.8981 [0.8714, 0.9194] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.5525 [0.5175, 0.5860] |
| lightgbm | 0.0712 [0.0497, 0.0943] |
| trivial_reference | 0.6617 [0.6238, 0.6930] |

GRU seed PR-AUC range: (0.74126184250835, 0.7618660158225415).

Training: 6000: epochs=40, best_validation_loss=0.071211539; 6001: epochs=40, best_validation_loss=0.074983936; 6002: epochs=40, best_validation_loss=0.07020587.

### round3_v2 / forced_flow

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0877 [0.0650, 0.1136] | 0.0543 [0.0465, 0.0622] | 0.0933 [0.0482, 0.1402] |
| lightgbm | 0.1318 [0.1086, 0.1601] | 0.0530 [0.0454, 0.0612] | 0.1558 [0.1234, 0.2006] |
| replicator | 0.1110 [0.0926, 0.1336] | 0.0537 [0.0452, 0.0616] | 0.1354 [0.1050, 0.1638] |
| gru_ensemble | 0.1657 [0.1322, 0.2099] | 0.0518 [0.0450, 0.0586] | 0.2084 [0.1599, 0.2606] |

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0547 [0.0284, 0.0852] |
| lightgbm | 0.0339 [0.0137, 0.0637] |
| trivial_reference | 0.0780 [0.0448, 0.1157] |

GRU seed PR-AUC range: (0.15740144510806592, 0.16566340805549667).

Training: 6000: epochs=40, best_validation_loss=0.17130895; 6001: epochs=23, best_validation_loss=0.1813; 6002: epochs=40, best_validation_loss=0.18228164.

## Joint gain contrasts

Every contrast is gain(baseline) minus gain(other). Gap explanation requires the same point-estimate sign and at least half the absolute v2 contrast; it does not require a significant family CI. Interaction is the registered qualitative rule, not a factorial interaction estimate.

| Label | Other population | Gain difference (95% CI) | Family rule | Gap rule |
| --- | --- | ---: | --- | --- |
| default | without_momentum | 0.3088 [0.2264, 0.4176] | contributes to visibility | explains the population gap |
| default | without_reversion | 0.1505 [0.0880, 0.2212] | contributes to visibility | explains the population gap |
| default | without_linear | 0.0277 [-0.0218, 0.1181] | no directional family effect established | does not explain the population gap |
| default | without_gbdt | 0.0093 [-0.0238, 0.0478] | no directional family effect established | does not explain the population gap |
| default | without_mlp | 0.0623 [0.0338, 0.0932] | contributes to visibility | does not explain the population gap |
| default | round3_v2 | 0.2050 [0.1644, 0.2442] | population gap | — |
| forced_flow | without_momentum | 0.0440 [0.0083, 0.0801] | uninformative | uninformative |
| forced_flow | without_reversion | 0.0243 [-0.0485, 0.0787] | no directional family effect established | does not explain the population gap |
| forced_flow | without_linear | -0.0042 [-0.0451, 0.0464] | no directional family effect established | does not explain the population gap |
| forced_flow | without_gbdt | 0.0385 [-0.0017, 0.0917] | no directional family effect established | does not explain the population gap |
| forced_flow | without_mlp | 0.0242 [-0.0298, 0.0765] | no directional family effect established | does not explain the population gap |
| forced_flow | round3_v2 | -0.0286 [-0.0751, 0.0173] | population gap | — |

## Pre-registered rule outcomes (verbatim)

- baseline/default: not consistent with strategy replication by the registered rule
- baseline/forced_flow: not consistent with strategy replication by the registered rule
- without_momentum/default: not consistent with strategy replication by the registered rule
- without_momentum/forced_flow: uninformative
- without_reversion/default: not consistent with strategy replication by the registered rule
- without_reversion/forced_flow: consistent with strategy replication
- without_linear/default: not consistent with strategy replication by the registered rule
- without_linear/forced_flow: not consistent with strategy replication by the registered rule
- without_gbdt/default: not consistent with strategy replication by the registered rule
- without_gbdt/forced_flow: consistent with strategy replication
- without_mlp/default: not consistent with strategy replication by the registered rule
- without_mlp/forced_flow: not consistent with strategy replication by the registered rule
- round3_v2/default: not consistent with strategy replication by the registered rule
- round3_v2/forced_flow: not consistent with strategy replication by the registered rule
- default/momentum: contributes to visibility
- default/momentum: explains the population gap
- default/reversion: contributes to visibility
- default/reversion: explains the population gap
- default/linear: no directional family effect established
- default/linear: does not explain the population gap
- default/gbdt: no directional family effect established
- default/gbdt: does not explain the population gap
- default/mlp: contributes to visibility
- default/mlp: does not explain the population gap
- default/interaction: interaction not established by the registered rule
- forced_flow/momentum: uninformative
- forced_flow/momentum: uninformative
- forced_flow/reversion: no directional family effect established
- forced_flow/reversion: does not explain the population gap
- forced_flow/linear: no directional family effect established
- forced_flow/linear: does not explain the population gap
- forced_flow/gbdt: no directional family effect established
- forced_flow/gbdt: does not explain the population gap
- forced_flow/mlp: no directional family effect established
- forced_flow/mlp: does not explain the population gap
- forced_flow/interaction: interaction not established by the registered rule

## Resolved configurations

Omissions scale configured q_max, order_limit and initial_cash by 30/24. Risk/leverage limits are unchanged. Configured aggregate capital and capacity match; realized dispersed capacities need not match exactly.

### baseline

Condition compute minutes: 3.76.

```yaml
simulation:
  n_steps: 500
  initial_price: 100.0
  realized_vol_lookback: 20
exogenous:
  initial_regime: 0
  transition_matrix:
  - - 0.98
    - 0.02
  - - 0.1
    - 0.9
  regime_volatility:
  - 0.001
  - 0.004
  jump_probability: 0.005
  jump_scale: 0.01
  liquidity:
    mean: 1000.0
    persistence: 0.95
    innovation_scale: 0.05
    minimum: 100.0
  background_flow:
    scale: 50.0
    persistence: 0.2
agents:
  population_size: 30
  momentum_fraction: 0.5
  momentum:
    count: 6
  reversion:
    count: 6
  lookback: 20
  q_max: 200.0
  risk_aversion_k: 2.0
  order_limit: 40.0
  epsilon: 1.0e-08
  parameter_dispersion: 0.1
  feature_overlap: 0.5
  learned_overlap_mode: none
  private_signal:
    strength: 0.0
    beta: 1.0
    rng_stream: 2
    mode: 'off'
  linear:
    count: 6
    lookback: 5
    ridge_alpha: 0.0001
    refit_interval: 10
    training_window: 100
    min_samples: 20
  gbdt:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    n_estimators: 12
    learning_rate: 0.05
    num_leaves: 7
  mlp:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    hidden_size: 8
    epochs: 20
    learning_rate: 0.03
    l2: 0.0001
  initial_cash: 10000.0
  transaction_cost_rate: 0.0001
  risk:
    enabled: true
    max_drawdown: 0.2
    max_leverage: 2.0
    liquidation_steps: 5
    cooldown_steps: 20
market:
  impact_model: linear
  impact_sensitivity: 0.03
targets:
  unwind_horizon: 10
```

### without_momentum

Condition compute minutes: 3.65.

```yaml
simulation:
  n_steps: 500
  initial_price: 100.0
  realized_vol_lookback: 20
exogenous:
  initial_regime: 0
  transition_matrix:
  - - 0.98
    - 0.02
  - - 0.1
    - 0.9
  regime_volatility:
  - 0.001
  - 0.004
  jump_probability: 0.005
  jump_scale: 0.01
  liquidity:
    mean: 1000.0
    persistence: 0.95
    innovation_scale: 0.05
    minimum: 100.0
  background_flow:
    scale: 50.0
    persistence: 0.2
agents:
  population_size: 24
  momentum_fraction: 0.5
  momentum:
    count: 0
  reversion:
    count: 6
  lookback: 20
  q_max: 250.0
  risk_aversion_k: 2.0
  order_limit: 50.0
  epsilon: 1.0e-08
  parameter_dispersion: 0.1
  feature_overlap: 0.5
  learned_overlap_mode: none
  private_signal:
    strength: 0.0
    beta: 1.0
    rng_stream: 2
    mode: 'off'
  linear:
    count: 6
    lookback: 5
    ridge_alpha: 0.0001
    refit_interval: 10
    training_window: 100
    min_samples: 20
  gbdt:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    n_estimators: 12
    learning_rate: 0.05
    num_leaves: 7
  mlp:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    hidden_size: 8
    epochs: 20
    learning_rate: 0.03
    l2: 0.0001
  initial_cash: 12500.0
  transaction_cost_rate: 0.0001
  risk:
    enabled: true
    max_drawdown: 0.2
    max_leverage: 2.0
    liquidation_steps: 5
    cooldown_steps: 20
market:
  impact_model: linear
  impact_sensitivity: 0.03
targets:
  unwind_horizon: 10
```

### without_reversion

Condition compute minutes: 3.70.

```yaml
simulation:
  n_steps: 500
  initial_price: 100.0
  realized_vol_lookback: 20
exogenous:
  initial_regime: 0
  transition_matrix:
  - - 0.98
    - 0.02
  - - 0.1
    - 0.9
  regime_volatility:
  - 0.001
  - 0.004
  jump_probability: 0.005
  jump_scale: 0.01
  liquidity:
    mean: 1000.0
    persistence: 0.95
    innovation_scale: 0.05
    minimum: 100.0
  background_flow:
    scale: 50.0
    persistence: 0.2
agents:
  population_size: 24
  momentum_fraction: 0.5
  momentum:
    count: 6
  reversion:
    count: 0
  lookback: 20
  q_max: 250.0
  risk_aversion_k: 2.0
  order_limit: 50.0
  epsilon: 1.0e-08
  parameter_dispersion: 0.1
  feature_overlap: 0.5
  learned_overlap_mode: none
  private_signal:
    strength: 0.0
    beta: 1.0
    rng_stream: 2
    mode: 'off'
  linear:
    count: 6
    lookback: 5
    ridge_alpha: 0.0001
    refit_interval: 10
    training_window: 100
    min_samples: 20
  gbdt:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    n_estimators: 12
    learning_rate: 0.05
    num_leaves: 7
  mlp:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    hidden_size: 8
    epochs: 20
    learning_rate: 0.03
    l2: 0.0001
  initial_cash: 12500.0
  transaction_cost_rate: 0.0001
  risk:
    enabled: true
    max_drawdown: 0.2
    max_leverage: 2.0
    liquidation_steps: 5
    cooldown_steps: 20
market:
  impact_model: linear
  impact_sensitivity: 0.03
targets:
  unwind_horizon: 10
```

### without_linear

Condition compute minutes: 3.26.

```yaml
simulation:
  n_steps: 500
  initial_price: 100.0
  realized_vol_lookback: 20
exogenous:
  initial_regime: 0
  transition_matrix:
  - - 0.98
    - 0.02
  - - 0.1
    - 0.9
  regime_volatility:
  - 0.001
  - 0.004
  jump_probability: 0.005
  jump_scale: 0.01
  liquidity:
    mean: 1000.0
    persistence: 0.95
    innovation_scale: 0.05
    minimum: 100.0
  background_flow:
    scale: 50.0
    persistence: 0.2
agents:
  population_size: 24
  momentum_fraction: 0.5
  momentum:
    count: 6
  reversion:
    count: 6
  lookback: 20
  q_max: 250.0
  risk_aversion_k: 2.0
  order_limit: 50.0
  epsilon: 1.0e-08
  parameter_dispersion: 0.1
  feature_overlap: 0.5
  learned_overlap_mode: none
  private_signal:
    strength: 0.0
    beta: 1.0
    rng_stream: 2
    mode: 'off'
  linear:
    count: 0
    lookback: 5
    ridge_alpha: 0.0001
    refit_interval: 10
    training_window: 100
    min_samples: 20
  gbdt:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    n_estimators: 12
    learning_rate: 0.05
    num_leaves: 7
  mlp:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    hidden_size: 8
    epochs: 20
    learning_rate: 0.03
    l2: 0.0001
  initial_cash: 12500.0
  transaction_cost_rate: 0.0001
  risk:
    enabled: true
    max_drawdown: 0.2
    max_leverage: 2.0
    liquidation_steps: 5
    cooldown_steps: 20
market:
  impact_model: linear
  impact_sensitivity: 0.03
targets:
  unwind_horizon: 10
```

### without_gbdt

Condition compute minutes: 3.36.

```yaml
simulation:
  n_steps: 500
  initial_price: 100.0
  realized_vol_lookback: 20
exogenous:
  initial_regime: 0
  transition_matrix:
  - - 0.98
    - 0.02
  - - 0.1
    - 0.9
  regime_volatility:
  - 0.001
  - 0.004
  jump_probability: 0.005
  jump_scale: 0.01
  liquidity:
    mean: 1000.0
    persistence: 0.95
    innovation_scale: 0.05
    minimum: 100.0
  background_flow:
    scale: 50.0
    persistence: 0.2
agents:
  population_size: 24
  momentum_fraction: 0.5
  momentum:
    count: 6
  reversion:
    count: 6
  lookback: 20
  q_max: 250.0
  risk_aversion_k: 2.0
  order_limit: 50.0
  epsilon: 1.0e-08
  parameter_dispersion: 0.1
  feature_overlap: 0.5
  learned_overlap_mode: none
  private_signal:
    strength: 0.0
    beta: 1.0
    rng_stream: 2
    mode: 'off'
  linear:
    count: 6
    lookback: 5
    ridge_alpha: 0.0001
    refit_interval: 10
    training_window: 100
    min_samples: 20
  gbdt:
    count: 0
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    n_estimators: 12
    learning_rate: 0.05
    num_leaves: 7
  mlp:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    hidden_size: 8
    epochs: 20
    learning_rate: 0.03
    l2: 0.0001
  initial_cash: 12500.0
  transaction_cost_rate: 0.0001
  risk:
    enabled: true
    max_drawdown: 0.2
    max_leverage: 2.0
    liquidation_steps: 5
    cooldown_steps: 20
market:
  impact_model: linear
  impact_sensitivity: 0.03
targets:
  unwind_horizon: 10
```

### without_mlp

Condition compute minutes: 4.37.

```yaml
simulation:
  n_steps: 500
  initial_price: 100.0
  realized_vol_lookback: 20
exogenous:
  initial_regime: 0
  transition_matrix:
  - - 0.98
    - 0.02
  - - 0.1
    - 0.9
  regime_volatility:
  - 0.001
  - 0.004
  jump_probability: 0.005
  jump_scale: 0.01
  liquidity:
    mean: 1000.0
    persistence: 0.95
    innovation_scale: 0.05
    minimum: 100.0
  background_flow:
    scale: 50.0
    persistence: 0.2
agents:
  population_size: 24
  momentum_fraction: 0.5
  momentum:
    count: 6
  reversion:
    count: 6
  lookback: 20
  q_max: 250.0
  risk_aversion_k: 2.0
  order_limit: 50.0
  epsilon: 1.0e-08
  parameter_dispersion: 0.1
  feature_overlap: 0.5
  learned_overlap_mode: none
  private_signal:
    strength: 0.0
    beta: 1.0
    rng_stream: 2
    mode: 'off'
  linear:
    count: 6
    lookback: 5
    ridge_alpha: 0.0001
    refit_interval: 10
    training_window: 100
    min_samples: 20
  gbdt:
    count: 6
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    n_estimators: 12
    learning_rate: 0.05
    num_leaves: 7
  mlp:
    count: 0
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    hidden_size: 8
    epochs: 20
    learning_rate: 0.03
    l2: 0.0001
  initial_cash: 12500.0
  transaction_cost_rate: 0.0001
  risk:
    enabled: true
    max_drawdown: 0.2
    max_leverage: 2.0
    liquidation_steps: 5
    cooldown_steps: 20
market:
  impact_model: linear
  impact_sensitivity: 0.03
targets:
  unwind_horizon: 10
```

### round3_v2

Condition compute minutes: 3.70.

```yaml
simulation:
  n_steps: 500
  initial_price: 100.0
  realized_vol_lookback: 20
exogenous:
  initial_regime: 0
  transition_matrix:
  - - 0.98
    - 0.02
  - - 0.1
    - 0.9
  regime_volatility:
  - 0.001
  - 0.004
  jump_probability: 0.005
  jump_scale: 0.01
  liquidity:
    mean: 1000.0
    persistence: 0.95
    innovation_scale: 0.05
    minimum: 100.0
  background_flow:
    scale: 50.0
    persistence: 0.2
agents:
  population_size: 30
  momentum_fraction: 0.5
  momentum:
    count: 10
  reversion:
    count: 0
  lookback: 20
  q_max: 200.0
  risk_aversion_k: 2.0
  order_limit: 40.0
  epsilon: 1.0e-08
  parameter_dispersion: 0.1
  feature_overlap: 0.5
  learned_overlap_mode: none
  private_signal:
    strength: 0.0
    beta: 1.0
    rng_stream: 2
    mode: 'off'
  linear:
    count: 10
    lookback: 5
    ridge_alpha: 0.0001
    refit_interval: 10
    training_window: 100
    min_samples: 20
  gbdt:
    count: 0
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    n_estimators: 12
    learning_rate: 0.05
    num_leaves: 7
  mlp:
    count: 10
    lookback: 5
    refit_interval: 50
    training_window: 100
    min_samples: 20
    hidden_size: 8
    epochs: 20
    learning_rate: 0.03
    l2: 0.0001
  initial_cash: 10000.0
  transaction_cost_rate: 0.0001
  risk:
    enabled: true
    max_drawdown: 0.2
    max_leverage: 2.0
    liquidation_steps: 5
    cooldown_steps: 20
market:
  impact_model: linear
  impact_sensitivity: 0.03
targets:
  unwind_horizon: 10
```

## Scope, deviations, and open questions

No simulator equation, label, threshold, population, seed range or observer hyperparameter was changed. Private state is reduced to targets before observer fitting. Positive counts are overlapping future windows, not distinct liquidation episodes; bootstrap units are whole episodes. Rules are withheld for any required population below 1% base rate or, for forced flow, below 30 positives. No multiplicity adjustment was registered. Family removal also redistributes capacity, and the v2 mix changes remaining family counts; these results do not uniquely identify a causal family mechanism or hidden real-world positions.

Implementation detail: the reused round-5 runner fits an ancillary logistic observer; it is excluded from the registered tables and decisions. No research-design deviations or open questions.

Completed invocation including report analysis: 28.80 minutes.

## Completion validation

Recovered the already completed 28.80-minute run; no duplicate experiment was started. Verified all seven cache fingerprints against the current Python source, resolved configurations and registered seed sets. Baseline LightGBM PR-AUC is 0.9510024422779156, reproducing 0.9510 to four decimals. All 168 reported observer metric CIs and all 14 base-rate/threshold rows match the caches.

`.venv/bin/pytest -q`: **177 passed in 17.67s**. Toy tests cover contributes, suppresses, explains-gap (including negative gaps and the half-gap boundary), interaction, uninformative, underpowered, zero CI boundaries and undefined cases.
