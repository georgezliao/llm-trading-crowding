# Experiment Zero, round 4: signal sweep and ablations

**Synthetic-simulator disclaimer:** Every result in this report comes from the uncalibrated synthetic HerdLab simulator. It is not evidence about real markets, real funds, or live trading behavior.

## Runtime estimate, configuration, and measured runtime

- Pre-run combined R4-4 plus R4-5 runtime estimate: 48 minutes, below the pre-registered ~60-minute stop threshold. The estimate used the measured 20.80-minute round-3 run and the registered episode/model workloads; no episode, seed, or resample count was cut.
- Measured R4-4 wall runtime: 28.61 minutes.
- Fixed selected setting: `configs/experiment_zero.yaml`; signal beta 1; learned overlap mode `none`; 500 steps per episode.
- Signal strengths: 0, 0.1, 0.25, 0.5; training seeds `8000-8199` and test seeds `8500-8549` at every strength.
- Labels: default (primary) and forced-flow (secondary); horizon 10; a separate training-only 0.95 threshold is fitted within every condition.
- Observers: trivial, logistic, LightGBM, replicator, GRU seeds `6000-6002`, and the seed-averaged GRU ensemble; return lookback 10.
- Feature ablations on rho=0: order flow, volatility, liquidity, and volume removed one group at a time; LightGBM and GRU seed 6000 refitted.
- Simulator ablations on rho=0: impact sensitivity set to 0, or risk rules disabled; the same training and test seeds and full observer suite were used.
- Metrics: PR-AUC, Brier, and recall at FAR <= 0.05; 95% percentile CIs use 1000 whole-episode resamples. Paired and cross-rho comparisons keep episode pairing.
- Parallelism: 10 simulator workers and 3 concurrent GRU workers.

## Thresholds and base rates

Base rates are descriptive. Observer and comparison uncertainty is reported with whole-episode-bootstrap CIs below.

| Condition | Label | Training threshold | Training events / labels | Training base rate | Test events / labels | Test base rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rho=0 | default | 0.1220341866 | 4900 / 98000 | 0.050000 | 1224 / 24500 | 0.049959 |
| rho=0 | forced_flow | 0.000282089358 | 4900 / 98000 | 0.050000 | 1519 / 24500 | 0.062000 |
| rho=0.1 | default | 0.06050191722 | 4900 / 98000 | 0.050000 | 1300 / 24500 | 0.053061 |
| rho=0.1 | forced_flow | 0 | 68 / 98000 | 0.000694 | 14 / 24500 | 0.000571 |
| rho=0.25 | default | 0.07638609236 | 4900 / 98000 | 0.050000 | 1342 / 24500 | 0.054776 |
| rho=0.25 | forced_flow | 0 | 28 / 98000 | 0.000286 | 0 / 24500 | 0.000000 |
| rho=0.5 | default | 0.1265885901 | 4900 / 98000 | 0.050000 | 1345 / 24500 | 0.054898 |
| rho=0.5 | forced_flow | 0 | 14 / 98000 | 0.000143 | 14 / 24500 | 0.000571 |
| impact_off | default | 0.0541573464 | 4900 / 98000 | 0.050000 | 1176 / 24500 | 0.048000 |
| impact_off | forced_flow | 0 | 63 / 98000 | 0.000643 | 24 / 24500 | 0.000980 |
| risk_off | default | 0.1218949716 | 4900 / 98000 | 0.050000 | 1223 / 24500 | 0.049918 |
| risk_off | forced_flow | 0 | 0 / 98000 | 0.000000 | 0 / 24500 | 0.000000 |

## Observer metrics

Entries are point estimate [episode-bootstrap 95% CI]. Higher is better for PR-AUC and recall; lower is better for Brier.

### rho=0 / default

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0897 [0.0528, 0.1324] | 0.0469 [0.0296, 0.0628] | 0.0923 [0.0520, 0.1365] |
| logistic_regression | 0.9230 [0.8843, 0.9473] | 0.0102 [0.0070, 0.0139] | 0.9894 [0.9753, 0.9964] |
| lightgbm | 0.9510 [0.9281, 0.9653] | 0.0082 [0.0055, 0.0115] | 0.9926 [0.9806, 0.9985] |
| replicator | 0.0831 [0.0494, 0.1203] | 0.0466 [0.0308, 0.0644] | 0.0891 [0.0639, 0.1186] |
| gru_6000 | 0.9548 [0.9331, 0.9697] | 0.0078 [0.0050, 0.0107] | 0.9935 [0.9816, 0.9991] |
| gru_6001 | 0.9562 [0.9352, 0.9709] | 0.0079 [0.0051, 0.0109] | 0.9935 [0.9841, 0.9991] |
| gru_6002 | 0.9542 [0.9296, 0.9681] | 0.0079 [0.0051, 0.0111] | 0.9951 [0.9860, 0.9991] |
| gru_ensemble | 0.9564 [0.9355, 0.9694] | 0.0077 [0.0049, 0.0108] | 0.9943 [0.9846, 0.9992] |

### rho=0 / forced_flow

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1079 [0.0804, 0.1405] | 0.0571 [0.0457, 0.0693] | 0.1099 [0.0655, 0.1600] |
| logistic_regression | 0.1168 [0.0910, 0.1512] | 0.0567 [0.0455, 0.0676] | 0.1290 [0.0871, 0.1734] |
| lightgbm | 0.1100 [0.0830, 0.1404] | 0.0568 [0.0464, 0.0684] | 0.1284 [0.0923, 0.1602] |
| replicator | 0.1186 [0.0857, 0.1584] | 0.0567 [0.0458, 0.0675] | 0.1442 [0.0717, 0.1881] |
| gru_6000 | 0.1575 [0.1177, 0.2086] | 0.0566 [0.0472, 0.0662] | 0.1916 [0.1301, 0.2500] |
| gru_6001 | 0.1180 [0.0892, 0.1523] | 0.0566 [0.0473, 0.0664] | 0.1402 [0.0968, 0.1846] |
| gru_6002 | 0.1597 [0.1175, 0.2084] | 0.0556 [0.0461, 0.0655] | 0.2100 [0.1499, 0.2787] |
| gru_ensemble | 0.1573 [0.1167, 0.2053] | 0.0554 [0.0463, 0.0647] | 0.2021 [0.1446, 0.2610] |

### rho=0.1 / default

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0773 [0.0691, 0.0890] | 0.0500 [0.0458, 0.0547] | 0.1192 [0.0999, 0.1401] |
| logistic_regression | 0.2002 [0.1729, 0.2312] | 0.0464 [0.0429, 0.0500] | 0.2892 [0.2564, 0.3227] |
| lightgbm | 0.1922 [0.1670, 0.2216] | 0.0465 [0.0427, 0.0503] | 0.2815 [0.2443, 0.3118] |
| replicator | 0.0671 [0.0590, 0.0774] | 0.0501 [0.0460, 0.0547] | 0.0846 [0.0661, 0.1005] |
| gru_6000 | 0.2141 [0.1845, 0.2449] | 0.0459 [0.0424, 0.0493] | 0.2946 [0.2608, 0.3293] |
| gru_6001 | 0.2174 [0.1823, 0.2525] | 0.0459 [0.0421, 0.0496] | 0.2808 [0.2479, 0.3190] |
| gru_6002 | 0.2137 [0.1831, 0.2473] | 0.0459 [0.0422, 0.0495] | 0.2938 [0.2627, 0.3289] |
| gru_ensemble | 0.2180 [0.1875, 0.2515] | 0.0458 [0.0424, 0.0495] | 0.2962 [0.2605, 0.3309] |

### rho=0.1 / forced_flow

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0004 [0.0004, 0.0015] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| logistic_regression | 0.0005 [0.0004, 0.0018] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| lightgbm | 0.0005 [0.0005, 0.0022] | 0.0008 [0.0002, 0.0020] | 0.0000 [0.0000, 0.0000] |
| replicator | 0.0007 [0.0007, 0.0021] | 0.0007 [0.0001, 0.0018] | 0.0000 [0.0000, 0.0000] |
| gru_6000 | 0.0004 [0.0004, 0.0013] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| gru_6001 | 0.0003 [0.0003, 0.0014] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| gru_6002 | 0.0004 [0.0003, 0.0011] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| gru_ensemble | 0.0003 [0.0003, 0.0014] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |

### rho=0.25 / default

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0858 [0.0747, 0.1004] | 0.0514 [0.0468, 0.0557] | 0.1066 [0.0903, 0.1247] |
| logistic_regression | 0.2608 [0.2346, 0.2874] | 0.0459 [0.0421, 0.0498] | 0.3495 [0.3207, 0.3774] |
| lightgbm | 0.2251 [0.2005, 0.2515] | 0.0470 [0.0433, 0.0508] | 0.3070 [0.2788, 0.3343] |
| replicator | 0.0815 [0.0728, 0.0914] | 0.0514 [0.0470, 0.0560] | 0.1021 [0.0847, 0.1178] |
| gru_6000 | 0.2632 [0.2339, 0.2941] | 0.0456 [0.0421, 0.0493] | 0.3539 [0.3232, 0.3838] |
| gru_6001 | 0.2664 [0.2393, 0.2989] | 0.0455 [0.0418, 0.0491] | 0.3569 [0.3241, 0.3865] |
| gru_6002 | 0.2642 [0.2344, 0.2941] | 0.0456 [0.0417, 0.0493] | 0.3614 [0.3279, 0.3877] |
| gru_ensemble | 0.2665 [0.2394, 0.2964] | 0.0455 [0.0419, 0.0493] | 0.3592 [0.3297, 0.3854] |

### rho=0.25 / forced_flow

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| logistic_regression | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| lightgbm | N/A | 0.0033 [0.0026, 0.0041] | N/A |
| replicator | N/A | 0.0002 [0.0000, 0.0003] | N/A |
| gru_6000 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6001 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6002 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_ensemble | N/A | 0.0000 [0.0000, 0.0000] | N/A |

### rho=0.5 / default

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0738 [0.0654, 0.0843] | 0.0516 [0.0475, 0.0564] | 0.0870 [0.0725, 0.1052] |
| logistic_regression | 0.2819 [0.2542, 0.3110] | 0.0450 [0.0412, 0.0489] | 0.3844 [0.3482, 0.4155] |
| lightgbm | 0.2337 [0.2110, 0.2560] | 0.0468 [0.0426, 0.0508] | 0.3264 [0.2972, 0.3560] |
| replicator | 0.1083 [0.0950, 0.1238] | 0.0509 [0.0467, 0.0552] | 0.1502 [0.1249, 0.1749] |
| gru_6000 | 0.3111 [0.2774, 0.3486] | 0.0439 [0.0406, 0.0474] | 0.4126 [0.3729, 0.4515] |
| gru_6001 | 0.3123 [0.2795, 0.3444] | 0.0440 [0.0407, 0.0473] | 0.4059 [0.3711, 0.4452] |
| gru_6002 | 0.3120 [0.2792, 0.3482] | 0.0438 [0.0401, 0.0472] | 0.4186 [0.3736, 0.4577] |
| gru_ensemble | 0.3158 [0.2801, 0.3511] | 0.0437 [0.0403, 0.0473] | 0.4171 [0.3818, 0.4535] |

### rho=0.5 / forced_flow

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0007 [0.0006, 0.0021] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| logistic_regression | 0.0004 [0.0004, 0.0012] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| lightgbm | 0.0006 [0.0005, 0.0022] | 0.0028 [0.0018, 0.0043] | 0.0000 [0.0000, 0.0000] |
| replicator | 0.0006 [0.0006, 0.0025] | 0.0015 [0.0007, 0.0028] | 0.0714 [0.0714, 0.0714] |
| gru_6000 | 0.0006 [0.0006, 0.0024] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| gru_6001 | 0.0008 [0.0007, 0.0030] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| gru_6002 | 0.0006 [0.0006, 0.0024] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| gru_ensemble | 0.0007 [0.0006, 0.0026] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |

### impact_off / default

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0871 [0.0727, 0.1055] | 0.0451 [0.0404, 0.0502] | 0.1429 [0.1157, 0.1675] |
| logistic_regression | 0.1373 [0.1157, 0.1604] | 0.0438 [0.0394, 0.0482] | 0.2245 [0.1913, 0.2595] |
| lightgbm | 0.1427 [0.1215, 0.1662] | 0.0435 [0.0392, 0.0482] | 0.2296 [0.2041, 0.2660] |
| replicator | 0.0633 [0.0537, 0.0758] | 0.0455 [0.0408, 0.0504] | 0.0740 [0.0560, 0.0900] |
| gru_6000 | 0.2434 [0.2086, 0.2825] | 0.0406 [0.0361, 0.0450] | 0.3827 [0.3349, 0.4324] |
| gru_6001 | 0.2643 [0.2270, 0.3038] | 0.0400 [0.0363, 0.0439] | 0.3912 [0.3471, 0.4359] |
| gru_6002 | 0.2485 [0.2138, 0.2864] | 0.0404 [0.0361, 0.0452] | 0.3895 [0.3465, 0.4347] |
| gru_ensemble | 0.2594 [0.2198, 0.2982] | 0.0401 [0.0359, 0.0445] | 0.4031 [0.3553, 0.4492] |

### impact_off / forced_flow

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0203 [0.0007, 0.1273] | 0.0010 [0.0000, 0.0026] | 0.4167 [0.0000, 1.0000] |
| logistic_regression | 0.0018 [0.0009, 0.0049] | 0.0010 [0.0000, 0.0024] | 0.0833 [0.0000, 0.2000] |
| lightgbm | 0.0012 [0.0005, 0.0032] | 0.0012 [0.0002, 0.0028] | 0.0417 [0.0000, 0.0714] |
| replicator | 0.0007 [0.0002, 0.0020] | 0.0010 [0.0000, 0.0026] | 0.0000 [0.0000, 0.0000] |
| gru_6000 | 0.0009 [0.0003, 0.0027] | 0.0010 [0.0000, 0.0025] | 0.0000 [0.0000, 0.0000] |
| gru_6001 | 0.0009 [0.0004, 0.0024] | 0.0010 [0.0000, 0.0025] | 0.0000 [0.0000, 0.0000] |
| gru_6002 | 0.0012 [0.0006, 0.0031] | 0.0010 [0.0000, 0.0026] | 0.0000 [0.0000, 0.0000] |
| gru_ensemble | 0.0009 [0.0003, 0.0029] | 0.0010 [0.0000, 0.0025] | 0.0000 [0.0000, 0.0000] |

### risk_off / default

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0852 [0.0545, 0.1284] | 0.0470 [0.0316, 0.0644] | 0.0842 [0.0468, 0.1285] |
| logistic_regression | 0.9177 [0.8773, 0.9424] | 0.0105 [0.0069, 0.0142] | 0.9853 [0.9690, 0.9950] |
| lightgbm | 0.9455 [0.9214, 0.9611] | 0.0086 [0.0056, 0.0117] | 0.9902 [0.9769, 0.9973] |
| replicator | 0.0830 [0.0509, 0.1192] | 0.0465 [0.0308, 0.0647] | 0.1030 [0.0695, 0.1334] |
| gru_6000 | 0.9503 [0.9268, 0.9643] | 0.0082 [0.0054, 0.0113] | 0.9902 [0.9778, 0.9979] |
| gru_6001 | 0.9520 [0.9286, 0.9658] | 0.0082 [0.0054, 0.0111] | 0.9886 [0.9770, 0.9972] |
| gru_6002 | 0.9497 [0.9266, 0.9644] | 0.0082 [0.0055, 0.0111] | 0.9894 [0.9774, 0.9964] |
| gru_ensemble | 0.9516 [0.9280, 0.9651] | 0.0080 [0.0053, 0.0108] | 0.9902 [0.9795, 0.9970] |

### risk_off / forced_flow

| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |
| --- | ---: | ---: | ---: |
| trivial_reference | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| logistic_regression | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| lightgbm | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| replicator | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6000 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6001 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6002 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_ensemble | N/A | 0.0000 [0.0000, 0.0000] | N/A |

## Per-seed GRU PR-AUC ranges

Each range is the minimum–maximum of the three per-seed point estimates; the contributing estimates and CIs appear above.

| Condition | Label | GRU seed min–max PR-AUC |
| --- | --- | ---: |
| rho=0 | default | 0.9542–0.9562 |
| rho=0 | forced_flow | 0.1180–0.1597 |
| rho=0.1 | default | 0.2137–0.2174 |
| rho=0.1 | forced_flow | 0.0003–0.0004 |
| rho=0.25 | default | 0.2632–0.2664 |
| rho=0.25 | forced_flow | N/A |
| rho=0.5 | default | 0.3111–0.3123 |
| rho=0.5 | forced_flow | 0.0006–0.0008 |
| impact_off | default | 0.2434–0.2643 |
| impact_off | forced_flow | 0.0009–0.0012 |
| risk_off | default | 0.9497–0.9520 |
| risk_off | forced_flow | N/A |

## Main paired PR-AUC differences

Entries are GRU ensemble minus comparator with paired whole-episode-bootstrap 95% CIs.

| Condition | Label | Minus replicator (95% CI) | Minus LightGBM (95% CI) | Minus trivial (95% CI) |
| --- | --- | ---: | ---: | ---: |
| rho=0 | default | 0.8733 [0.8344, 0.9049] | 0.0054 [0.0010, 0.0111] | 0.8666 [0.8266, 0.8968] |
| rho=0 | forced_flow | 0.0387 [0.0127, 0.0735] | 0.0472 [0.0249, 0.0761] | 0.0494 [0.0178, 0.0849] |
| rho=0.1 | default | 0.1509 [0.1241, 0.1803] | 0.0258 [0.0146, 0.0368] | 0.1406 [0.1136, 0.1677] |
| rho=0.1 | forced_flow | -0.0003 [-0.0013, -0.0003] | -0.0002 [-0.0008, -0.0002] | -0.0000 [-0.0001, -0.0000] |
| rho=0.25 | default | 0.1850 [0.1600, 0.2084] | 0.0414 [0.0258, 0.0573] | 0.1807 [0.1576, 0.2036] |
| rho=0.25 | forced_flow | N/A | N/A | N/A |
| rho=0.5 | default | 0.2076 [0.1815, 0.2342] | 0.0821 [0.0581, 0.1057] | 0.2420 [0.2100, 0.2725] |
| rho=0.5 | forced_flow | 0.0000 [-0.0000, 0.0001] | 0.0001 [0.0001, 0.0004] | 0.0000 [-0.0000, 0.0001] |
| impact_off | default | 0.1961 [0.1623, 0.2299] | 0.1167 [0.0906, 0.1475] | 0.1723 [0.1390, 0.2083] |
| impact_off | forced_flow | 0.0002 [0.0001, 0.0007] | -0.0003 [-0.0007, 0.0001] | -0.0194 [-0.1239, 0.0003] |
| risk_off | default | 0.8685 [0.8341, 0.8977] | 0.0060 [0.0015, 0.0121] | 0.8663 [0.8296, 0.8925] |
| risk_off | forced_flow | N/A | N/A | N/A |

## Private-information comparison

The registered statistic is [(GRU ensemble minus trivial) at rho=0] minus [(GRU ensemble minus trivial) at rho=0.5]. The bootstrap selects the same test-seed episodes in both conditions on every replicate.

| Statistic | Estimate and jointly resampled 95% CI |
| --- | ---: |
| Difference of differences | 0.6246 [0.5803, 0.6632] |

## Feature ablations at rho=0

Full and ablated entries are PR-AUC point estimate [episode-bootstrap 95% CI]. Differences use paired whole-episode resamples.

| Label | Removed group | Observer | Full PR-AUC (95% CI) | Ablated PR-AUC (95% CI) | Full minus ablated (95% CI) | Test base rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| default | order_flow | lightgbm | 0.9510 [0.9281, 0.9653] | 0.9495 [0.9253, 0.9654] | 0.0015 [-0.0003, 0.0030] | 0.049959 |
| default | order_flow | gru_6000 | 0.9548 [0.9331, 0.9697] | 0.9549 [0.9336, 0.9692] | -0.0001 [-0.0030, 0.0025] | 0.049959 |
| default | volatility | lightgbm | 0.9510 [0.9281, 0.9653] | 0.9509 [0.9266, 0.9655] | 0.0001 [-0.0010, 0.0009] | 0.049959 |
| default | volatility | gru_6000 | 0.9548 [0.9331, 0.9697] | 0.9547 [0.9334, 0.9694] | 0.0001 [-0.0021, 0.0019] | 0.049959 |
| default | liquidity | lightgbm | 0.9510 [0.9281, 0.9653] | 0.9438 [0.9168, 0.9604] | 0.0072 [0.0040, 0.0106] | 0.049959 |
| default | liquidity | gru_6000 | 0.9548 [0.9331, 0.9697] | 0.9464 [0.9188, 0.9624] | 0.0084 [0.0041, 0.0137] | 0.049959 |
| default | volume | lightgbm | 0.9510 [0.9281, 0.9653] | 0.9506 [0.9268, 0.9654] | 0.0004 [-0.0003, 0.0011] | 0.049959 |
| default | volume | gru_6000 | 0.9548 [0.9331, 0.9697] | 0.9553 [0.9333, 0.9689] | -0.0005 [-0.0037, 0.0021] | 0.049959 |
| forced_flow | order_flow | lightgbm | 0.1100 [0.0830, 0.1404] | 0.1100 [0.0825, 0.1419] | 0.0001 [-0.0019, 0.0014] | 0.062000 |
| forced_flow | order_flow | gru_6000 | 0.1575 [0.1177, 0.2086] | 0.1515 [0.1057, 0.1957] | 0.0061 [-0.0010, 0.0125] | 0.062000 |
| forced_flow | volatility | lightgbm | 0.1100 [0.0830, 0.1404] | 0.1099 [0.0837, 0.1385] | 0.0001 [-0.0062, 0.0060] | 0.062000 |
| forced_flow | volatility | gru_6000 | 0.1575 [0.1177, 0.2086] | 0.1553 [0.1111, 0.2033] | 0.0023 [-0.0060, 0.0120] | 0.062000 |
| forced_flow | liquidity | lightgbm | 0.1100 [0.0830, 0.1404] | 0.1094 [0.0843, 0.1392] | 0.0006 [-0.0039, 0.0043] | 0.062000 |
| forced_flow | liquidity | gru_6000 | 0.1575 [0.1177, 0.2086] | 0.1598 [0.1163, 0.2136] | -0.0022 [-0.0168, 0.0121] | 0.062000 |
| forced_flow | volume | lightgbm | 0.1100 [0.0830, 0.1404] | 0.1097 [0.0845, 0.1376] | 0.0003 [-0.0071, 0.0076] | 0.062000 |
| forced_flow | volume | gru_6000 | 0.1575 [0.1177, 0.2086] | 0.1567 [0.1149, 0.2055] | 0.0008 [-0.0074, 0.0078] | 0.062000 |

## GRU training detail

- rho=0/default: seed 6000: epochs=40, best_validation_loss=0.033522398; seed 6001: epochs=40, best_validation_loss=0.037296954; seed 6002: epochs=40, best_validation_loss=0.030605735.
- rho=0/forced_flow: seed 6000: epochs=40, best_validation_loss=0.1673529; seed 6001: epochs=9, best_validation_loss=0.19388598; seed 6002: epochs=26, best_validation_loss=0.19049957.
- rho=0.1/default: seed 6000: epochs=24, best_validation_loss=0.17065497; seed 6001: epochs=24, best_validation_loss=0.16503722; seed 6002: epochs=31, best_validation_loss=0.16407879.
- rho=0.1/forced_flow: seed 6000: epochs=14, best_validation_loss=0.010097709; seed 6001: epochs=13, best_validation_loss=0.010101302; seed 6002: epochs=24, best_validation_loss=0.0059087941.
- rho=0.25/default: seed 6000: epochs=17, best_validation_loss=0.16242913; seed 6001: epochs=14, best_validation_loss=0.15247041; seed 6002: epochs=19, best_validation_loss=0.15707097.
- rho=0.25/forced_flow: seed 6000: epochs=17, best_validation_loss=0.0058927202; seed 6001: epochs=16, best_validation_loss=0.0058912992; seed 6002: epochs=18, best_validation_loss=0.0059000215.
- rho=0.5/default: seed 6000: epochs=22, best_validation_loss=0.163361; seed 6001: epochs=24, best_validation_loss=0.1474866; seed 6002: epochs=25, best_validation_loss=0.15362556.
- rho=0.5/forced_flow: seed 6000: epochs=40, best_validation_loss=0.00019079646; seed 6001: epochs=39, best_validation_loss=0.00019277023; seed 6002: epochs=40, best_validation_loss=0.00019037359.
- impact_off/default: seed 6000: epochs=20, best_validation_loss=0.16517378; seed 6001: epochs=30, best_validation_loss=0.16958437; seed 6002: epochs=33, best_validation_loss=0.15387125.
- impact_off/forced_flow: seed 6000: epochs=19, best_validation_loss=0.0062717369; seed 6001: epochs=31, best_validation_loss=0.00079116916; seed 6002: epochs=33, best_validation_loss=0.00078938818.
- risk_off/default: seed 6000: epochs=40, best_validation_loss=0.033966822; seed 6001: epochs=40, best_validation_loss=0.036229322; seed 6002: epochs=40, best_validation_loss=0.02807751.
- risk_off/forced_flow: seed 6000: epochs=40, best_validation_loss=4.3511634e-05; seed 6001: epochs=40, best_validation_loss=4.2366252e-05; seed 6002: epochs=40, best_validation_loss=5.7926956e-05.
- feature ablations/default/GRU seed 6000: without order_flow: epochs=40, best_validation_loss=0.033315326; without volatility: epochs=40, best_validation_loss=0.034350283; without liquidity: epochs=40, best_validation_loss=0.035610348; without volume: epochs=40, best_validation_loss=0.033898275.
- feature ablations/forced_flow/GRU seed 6000: without order_flow: epochs=40, best_validation_loss=0.16581863; without volatility: epochs=40, best_validation_loss=0.16625744; without liquidity: epochs=40, best_validation_loss=0.16532563; without volume: epochs=40, best_validation_loss=0.16513832.

## Pre-registered rule outcomes (verbatim)

- Private-information degradation rule — default label: the jointly resampled CI of [(GRU minus trivial) at rho=0] minus [(GRU minus trivial) at rho=0.5] lies above 0; private information degrades inference by the pre-registered rule.
- Strategy-replication rule — rho=0/default: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — rho=0.1/default: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — rho=0.25/default: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Strategy-replication rule — rho=0.5/default: the paired GRU-ensemble-minus-replicator PR-AUC interval excludes 0 above; the observer's advantage is not reported as consistent with strategy replication by this rule.
- Feature-use rule — default/lightgbm/order_flow: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — default/gru_6000/order_flow: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — default/lightgbm/volatility: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — default/gru_6000/volatility: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — default/lightgbm/liquidity: the paired full-minus-ablated PR-AUC interval excludes 0 above; the feature group is reported as used.
- Feature-use rule — default/gru_6000/liquidity: the paired full-minus-ablated PR-AUC interval excludes 0 above; the feature group is reported as used.
- Feature-use rule — default/lightgbm/volume: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — default/gru_6000/volume: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — forced_flow/lightgbm/order_flow: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — forced_flow/gru_6000/order_flow: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — forced_flow/lightgbm/volatility: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — forced_flow/gru_6000/volatility: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — forced_flow/lightgbm/liquidity: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — forced_flow/gru_6000/liquidity: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — forced_flow/lightgbm/volume: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Feature-use rule — forced_flow/gru_6000/volume: the paired full-minus-ablated PR-AUC interval does not exclude 0 above; the feature group is not reported as used.
- Forced-flow power rule — rho=0/forced_flow: 1519 test events meets the 30-event minimum; the result is not flagged as underpowered by this rule.
- Forced-flow power rule — rho=0.1/forced_flow: 14 test events is fewer than 30; the result is reported as underpowered.
- Forced-flow power rule — rho=0.25/forced_flow: 0 test events is fewer than 30; the result is reported as underpowered.
- Forced-flow power rule — rho=0.5/forced_flow: 14 test events is fewer than 30; the result is reported as underpowered.
- Forced-flow power rule — impact_off/forced_flow: 24 test events is fewer than 30; the result is reported as underpowered.
- Forced-flow power rule — risk_off/forced_flow: 0 test events is fewer than 30; the result is reported as underpowered.

## Caveats, deviations, and open questions

- Observer inputs contain only public records. Agent-only flow and forced flags are reduced to targets before observer construction.
- Thresholds use training episodes only. CIs resample whole episodes, never overlapping time steps independently.
- The rho=0 condition uses the registered private-signal bypass and therefore preserves the round-3 simulator path.
- No rho, beta, seed range, threshold, observer hyperparameter, episode count, or resample count was changed after viewing metrics.
- No deviations from the pre-registration and no open questions.
