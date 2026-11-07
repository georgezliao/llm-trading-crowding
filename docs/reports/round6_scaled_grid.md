# Round 6: per-family signal scaling

**Synthetic-simulator disclaimer:** All results are from the uncalibrated synthetic HerdLab simulator, not real markets, funds, or live trading.

## Configuration, seeds, and runtime

Runtime estimate before running: 57 minutes, inside the ~130-minute limit once added to the 25.80-minute population run. Measured invocation wall time: 117.85 minutes. Sum of condition compute times: 113.97 minutes. Workers: 10 simulator, 3 GRU.

This step was implemented by the research lead after the implementation agent hit a usage limit. The design is unchanged from the pre-registration.

Each condition uses 200 training episodes (8000–8199) and 50 test episodes (8500–8549), 500 steps each. Horizon 10, separate training-only 0.95 quantile per condition, GRU seeds 6000–6002. CIs are 95% percentile intervals from 1,000 whole-episode resamples, and cross-condition contrasts resample the same test seeds jointly.

### Family constants

Pooled standard deviation of each family's base forecast in the off condition, on training seeds only, including warm-up decisions. The signal term is β × constant × s.

| Family | Constant |
| --- | ---: |
| momentum | 0.00114766 |
| reversion | 0.00114612 |
| linear | 0.0189814 |
| gbdt | 0.00820489 |
| mlp | 0.0159013 |

### Base configuration

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

## Thresholds and base rates

| Condition | Label | Threshold | Test positives / labels | Test rate (95% CI) |
| --- | --- | ---: | ---: | ---: |
| off | default | 0.122034186585 | 1224/24500 | 0.0500 [0.0302, 0.0705] |
| off | forced_flow | 0.000282089357967 | 1519/24500 | 0.0620 [0.0493, 0.0744] |
| beta=0.25/noise | default | 0.0762986075157 | 1075/24500 | 0.0439 [0.0293, 0.0620] |
| beta=0.25/noise | forced_flow | 0.000336493893802 | 1237/24500 | 0.0505 [0.0374, 0.0627] |
| beta=0.25/rho=0.25 | default | 0.0649804319699 | 1255/24500 | 0.0512 [0.0321, 0.0710] |
| beta=0.25/rho=0.25 | forced_flow | 0.000314221732171 | 1365/24500 | 0.0557 [0.0411, 0.0704] |
| beta=0.25/rho=0.5 | default | 0.0618310048921 | 1111/24500 | 0.0453 [0.0297, 0.0630] |
| beta=0.25/rho=0.5 | forced_flow | 0.000285756179733 | 1473/24500 | 0.0601 [0.0456, 0.0733] |
| beta=0.5/noise | default | 0.0611405352692 | 1174/24500 | 0.0479 [0.0365, 0.0612] |
| beta=0.5/noise | forced_flow | 0 | 1173/24500 | 0.0479 [0.0357, 0.0591] |
| beta=0.5/rho=0.25 | default | 0.0586756352451 | 1030/24500 | 0.0420 [0.0309, 0.0550] |
| beta=0.5/rho=0.25 | forced_flow | 0 | 1434/24500 | 0.0585 [0.0435, 0.0742] |
| beta=0.5/rho=0.5 | default | 0.0625919911827 | 1127/24500 | 0.0460 [0.0364, 0.0565] |
| beta=0.5/rho=0.5 | forced_flow | 0 | 1260/24500 | 0.0514 [0.0379, 0.0669] |
| beta=1/noise | default | 0.055304785386 | 1244/24500 | 0.0508 [0.0465, 0.0551] |
| beta=1/noise | forced_flow | 0.013636397456 | 1134/24500 | 0.0463 [0.0352, 0.0576] |
| beta=1/rho=0.25 | default | 0.0635029104759 | 1357/24500 | 0.0554 [0.0510, 0.0601] |
| beta=1/rho=0.25 | forced_flow | 0 | 226/24500 | 0.0092 [0.0046, 0.0152] |
| beta=1/rho=0.5 | default | 0.0841813240386 | 1311/24500 | 0.0535 [0.0491, 0.0581] |
| beta=1/rho=0.5 | forced_flow | 0 | 28/24500 | 0.0011 [0.0000, 0.0029] |

## Forecast scale ratios

std(signal term) / std(base forecast), pooled over the 250 episodes of each condition. Base forecasts respond to the signal through prices, so ratios need not equal β exactly.

| Condition | momentum | reversion | linear | gbdt | mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| beta=0.25/noise | 0.2510 [0.2460, 0.2566] | 0.2492 [0.2442, 0.2548] | 0.2548 [0.2514, 0.2582] | 0.2541 [0.2506, 0.2576] | 0.2472 [0.2420, 0.2526] |
| beta=0.25/rho=0.25 | 0.2363 [0.2300, 0.2429] | 0.2364 [0.2302, 0.2423] | 0.2542 [0.2505, 0.2581] | 0.2518 [0.2480, 0.2557] | 0.2417 [0.2363, 0.2474] |
| beta=0.25/rho=0.5 | 0.2121 [0.2058, 0.2188] | 0.2127 [0.2065, 0.2196] | 0.2634 [0.2589, 0.2680] | 0.2596 [0.2552, 0.2641] | 0.2480 [0.2421, 0.2542] |
| beta=0.5/noise | 0.5451 [0.5340, 0.5557] | 0.5462 [0.5346, 0.5567] | 0.6185 [0.6027, 0.6331] | 0.6215 [0.6045, 0.6372] | 0.6053 [0.5859, 0.6252] |
| beta=0.5/rho=0.25 | 0.4894 [0.4757, 0.5017] | 0.4911 [0.4780, 0.5036] | 0.6074 [0.5942, 0.6220] | 0.6020 [0.5883, 0.6175] | 0.5806 [0.5638, 0.5994] |
| beta=0.5/rho=0.5 | 0.4251 [0.4129, 0.4384] | 0.4248 [0.4131, 0.4377] | 0.6178 [0.6031, 0.6331] | 0.6048 [0.5904, 0.6206] | 0.5868 [0.5688, 0.6064] |
| beta=1/noise | 1.2407 [1.2181, 1.2651] | 1.2388 [1.2181, 1.2616] | 5.7902 [5.2614, 6.2545] | 4.7645 [4.4480, 5.0393] | 5.1929 [4.7280, 5.5689] |
| beta=1/rho=0.25 | 1.0090 [0.9774, 1.0385] | 1.0079 [0.9763, 1.0381] | 3.5115 [3.4272, 3.5975] | 3.1521 [3.0873, 3.2179] | 3.5386 [3.4665, 3.6118] |
| beta=1/rho=0.5 | 0.8041 [0.7819, 0.8276] | 0.8029 [0.7810, 0.8265] | 2.4682 [2.4090, 2.5269] | 2.2507 [2.2010, 2.3042] | 2.5859 [2.5205, 2.6522] |

## Observer metrics

### off / default

Test base rate: 0.049959; positive labels: 1224.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0897 [0.0528, 0.1324] | 0.0469 [0.0296, 0.0628] | 0.0923 [0.0520, 0.1365] |
| logistic_regression | 0.9230 [0.8843, 0.9473] | 0.0102 [0.0070, 0.0139] | 0.9894 [0.9753, 0.9964] |
| lightgbm | 0.9510 [0.9281, 0.9653] | 0.0082 [0.0055, 0.0115] | 0.9926 [0.9806, 0.9985] |
| replicator | 0.0831 [0.0494, 0.1203] | 0.0466 [0.0308, 0.0644] | 0.0891 [0.0639, 0.1186] |
| gru_6000 | 0.9548 [0.9331, 0.9697] | 0.0078 [0.0050, 0.0107] | 0.9935 [0.9816, 0.9991] |
| gru_6001 | 0.9562 [0.9352, 0.9709] | 0.0079 [0.0051, 0.0109] | 0.9935 [0.9841, 0.9991] |
| gru_6002 | 0.9542 [0.9296, 0.9681] | 0.0079 [0.0051, 0.0111] | 0.9951 [0.9860, 0.9991] |
| gru_ensemble | 0.9564 [0.9355, 0.9694] | 0.0077 [0.0049, 0.0108] | 0.9943 [0.9846, 0.9992] |

GRU seed min–max PR-AUC: (0.9541536246793603, 0.9562191439170422).

GRU ensemble minus replicator: 0.8733 [0.8344, 0.9049].
GRU ensemble minus lightgbm: 0.0054 [0.0010, 0.0111].
GRU ensemble minus trivial_reference: 0.8666 [0.8266, 0.8968].

### off / forced_flow

Test base rate: 0.062000; positive labels: 1519.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1079 [0.0804, 0.1405] | 0.0571 [0.0457, 0.0693] | 0.1099 [0.0655, 0.1600] |
| logistic_regression | 0.1168 [0.0910, 0.1512] | 0.0567 [0.0455, 0.0676] | 0.1290 [0.0871, 0.1734] |
| lightgbm | 0.1100 [0.0830, 0.1404] | 0.0568 [0.0464, 0.0684] | 0.1284 [0.0923, 0.1602] |
| replicator | 0.1186 [0.0857, 0.1584] | 0.0567 [0.0458, 0.0675] | 0.1442 [0.0717, 0.1881] |
| gru_6000 | 0.1575 [0.1177, 0.2086] | 0.0566 [0.0472, 0.0662] | 0.1916 [0.1301, 0.2500] |
| gru_6001 | 0.1180 [0.0892, 0.1523] | 0.0566 [0.0473, 0.0664] | 0.1402 [0.0968, 0.1846] |
| gru_6002 | 0.1597 [0.1175, 0.2084] | 0.0556 [0.0461, 0.0655] | 0.2100 [0.1499, 0.2787] |
| gru_ensemble | 0.1573 [0.1167, 0.2053] | 0.0554 [0.0463, 0.0647] | 0.2021 [0.1446, 0.2610] |

GRU seed min–max PR-AUC: (0.11802803300915274, 0.1596684543289827).

GRU ensemble minus replicator: 0.0387 [0.0127, 0.0735].
GRU ensemble minus lightgbm: 0.0472 [0.0249, 0.0761].
GRU ensemble minus trivial_reference: 0.0494 [0.0178, 0.0849].

### beta=0.25/noise / default

Test base rate: 0.043878; positive labels: 1075.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0715 [0.0499, 0.0959] | 0.0418 [0.0288, 0.0577] | 0.1256 [0.0817, 0.1716] |
| logistic_regression | 0.4199 [0.1976, 0.6297] | 0.0325 [0.0221, 0.0433] | 0.5349 [0.3720, 0.6965] |
| lightgbm | 0.6383 [0.4361, 0.7731] | 0.0252 [0.0170, 0.0336] | 0.7572 [0.6403, 0.8416] |
| replicator | 0.1037 [0.0772, 0.1327] | 0.0410 [0.0278, 0.0558] | 0.1981 [0.1414, 0.2777] |
| gru_6000 | 0.6971 [0.5180, 0.8116] | 0.0217 [0.0150, 0.0287] | 0.7981 [0.6802, 0.8873] |
| gru_6001 | 0.7029 [0.5291, 0.8217] | 0.0216 [0.0142, 0.0284] | 0.7888 [0.6785, 0.8770] |
| gru_6002 | 0.6952 [0.5237, 0.8145] | 0.0219 [0.0153, 0.0292] | 0.7879 [0.6911, 0.8744] |
| gru_ensemble | 0.7125 [0.5543, 0.8357] | 0.0211 [0.0148, 0.0285] | 0.8112 [0.6938, 0.8892] |

GRU seed min–max PR-AUC: (0.695164275917456, 0.7028557703296713).

GRU ensemble minus replicator: 0.6088 [0.4373, 0.7329].
GRU ensemble minus lightgbm: 0.0741 [0.0399, 0.1219].
GRU ensemble minus trivial_reference: 0.6410 [0.4892, 0.7427].

### beta=0.25/noise / forced_flow

Test base rate: 0.050490; positive labels: 1237.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1028 [0.0705, 0.1381] | 0.0468 [0.0363, 0.0575] | 0.1520 [0.0854, 0.2047] |
| logistic_regression | 0.1255 [0.0910, 0.1655] | 0.0461 [0.0356, 0.0559] | 0.1795 [0.1307, 0.2396] |
| lightgbm | 0.1218 [0.0845, 0.1675] | 0.0463 [0.0360, 0.0579] | 0.1778 [0.1283, 0.2395] |
| replicator | 0.1113 [0.0775, 0.1478] | 0.0466 [0.0354, 0.0570] | 0.1681 [0.1384, 0.2239] |
| gru_6000 | 0.1814 [0.1247, 0.2492] | 0.0445 [0.0351, 0.0543] | 0.2829 [0.2015, 0.3683] |
| gru_6001 | 0.1775 [0.1145, 0.2546] | 0.0447 [0.0352, 0.0546] | 0.2749 [0.2013, 0.3659] |
| gru_6002 | 0.1825 [0.1160, 0.2496] | 0.0445 [0.0353, 0.0554] | 0.2789 [0.2063, 0.3592] |
| gru_ensemble | 0.1844 [0.1252, 0.2553] | 0.0445 [0.0359, 0.0549] | 0.2910 [0.2038, 0.3741] |

GRU seed min–max PR-AUC: (0.17745941979395952, 0.18248257153035546).

GRU ensemble minus replicator: 0.0731 [0.0318, 0.1235].
GRU ensemble minus lightgbm: 0.0626 [0.0257, 0.1096].
GRU ensemble minus trivial_reference: 0.0815 [0.0260, 0.1475].

### beta=0.25/rho=0.25 / default

Test base rate: 0.051224; positive labels: 1255.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0659 [0.0439, 0.0942] | 0.0486 [0.0319, 0.0674] | 0.0972 [0.0549, 0.1559] |
| logistic_regression | 0.6288 [0.4166, 0.7531] | 0.0289 [0.0210, 0.0377] | 0.6622 [0.4983, 0.8009] |
| lightgbm | 0.7496 [0.5941, 0.8470] | 0.0227 [0.0159, 0.0294] | 0.7697 [0.6520, 0.8554] |
| replicator | 0.0836 [0.0619, 0.1092] | 0.0484 [0.0308, 0.0666] | 0.1331 [0.0967, 0.1958] |
| gru_6000 | 0.7899 [0.6477, 0.8682] | 0.0203 [0.0148, 0.0265] | 0.8263 [0.7478, 0.8818] |
| gru_6001 | 0.7936 [0.6302, 0.8709] | 0.0201 [0.0145, 0.0261] | 0.8183 [0.7310, 0.8827] |
| gru_6002 | 0.7953 [0.6446, 0.8735] | 0.0203 [0.0141, 0.0259] | 0.8303 [0.7534, 0.8883] |
| gru_ensemble | 0.8022 [0.6498, 0.8775] | 0.0197 [0.0141, 0.0253] | 0.8382 [0.7520, 0.8941] |

GRU seed min–max PR-AUC: (0.7898891894142541, 0.795332236677753).

GRU ensemble minus replicator: 0.7186 [0.5745, 0.7906].
GRU ensemble minus lightgbm: 0.0526 [0.0297, 0.0824].
GRU ensemble minus trivial_reference: 0.7364 [0.6132, 0.8119].

### beta=0.25/rho=0.25 / forced_flow

Test base rate: 0.055714; positive labels: 1365.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1166 [0.0818, 0.1581] | 0.0512 [0.0392, 0.0640] | 0.1451 [0.0886, 0.2002] |
| logistic_regression | 0.1294 [0.0923, 0.1730] | 0.0507 [0.0383, 0.0633] | 0.1495 [0.0995, 0.2055] |
| lightgbm | 0.1276 [0.0948, 0.1641] | 0.0508 [0.0382, 0.0632] | 0.1502 [0.1140, 0.1922] |
| replicator | 0.1221 [0.0908, 0.1548] | 0.0509 [0.0376, 0.0631] | 0.1619 [0.1079, 0.2108] |
| gru_6000 | 0.1600 [0.1161, 0.2195] | 0.0496 [0.0379, 0.0610] | 0.2103 [0.1391, 0.2961] |
| gru_6001 | 0.1708 [0.1207, 0.2288] | 0.0490 [0.0378, 0.0602] | 0.2344 [0.1585, 0.3130] |
| gru_6002 | 0.1638 [0.1157, 0.2287] | 0.0494 [0.0379, 0.0618] | 0.2278 [0.1522, 0.3077] |
| gru_ensemble | 0.1673 [0.1145, 0.2263] | 0.0492 [0.0379, 0.0608] | 0.2205 [0.1453, 0.2963] |

GRU seed min–max PR-AUC: (0.15997156668980886, 0.17075255872133238).

GRU ensemble minus replicator: 0.0452 [0.0093, 0.0941].
GRU ensemble minus lightgbm: 0.0397 [0.0082, 0.0806].
GRU ensemble minus trivial_reference: 0.0508 [0.0125, 0.0925].

### beta=0.25/rho=0.5 / default

Test base rate: 0.045347; positive labels: 1111.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0662 [0.0550, 0.0829] | 0.0431 [0.0291, 0.0583] | 0.0864 [0.0628, 0.1355] |
| logistic_regression | 0.4790 [0.2993, 0.6248] | 0.0313 [0.0226, 0.0418] | 0.5716 [0.4585, 0.6979] |
| lightgbm | 0.5654 [0.3968, 0.7078] | 0.0282 [0.0197, 0.0382] | 0.6616 [0.5676, 0.7502] |
| replicator | 0.1092 [0.0859, 0.1354] | 0.0419 [0.0279, 0.0568] | 0.1782 [0.1190, 0.2486] |
| gru_6000 | 0.6146 [0.4374, 0.7392] | 0.0267 [0.0191, 0.0354] | 0.7156 [0.6371, 0.7885] |
| gru_6001 | 0.6271 [0.4577, 0.7514] | 0.0263 [0.0189, 0.0342] | 0.7309 [0.6550, 0.8043] |
| gru_6002 | 0.6251 [0.4564, 0.7394] | 0.0260 [0.0184, 0.0335] | 0.7237 [0.6417, 0.7959] |
| gru_ensemble | 0.6344 [0.4625, 0.7439] | 0.0257 [0.0183, 0.0340] | 0.7363 [0.6555, 0.8085] |

GRU seed min–max PR-AUC: (0.6146340597034754, 0.627096361398138).

GRU ensemble minus replicator: 0.5252 [0.3571, 0.6442].
GRU ensemble minus lightgbm: 0.0690 [0.0438, 0.0948].
GRU ensemble minus trivial_reference: 0.5682 [0.4077, 0.6811].

### beta=0.25/rho=0.5 / forced_flow

Test base rate: 0.060122; positive labels: 1473.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1138 [0.0851, 0.1502] | 0.0553 [0.0439, 0.0676] | 0.1331 [0.0925, 0.1842] |
| logistic_regression | 0.1267 [0.0918, 0.1690] | 0.0548 [0.0435, 0.0666] | 0.1514 [0.1128, 0.2037] |
| lightgbm | 0.1148 [0.0873, 0.1427] | 0.0550 [0.0432, 0.0673] | 0.1276 [0.0978, 0.1682] |
| replicator | 0.1126 [0.0868, 0.1417] | 0.0551 [0.0437, 0.0660] | 0.1242 [0.0960, 0.1546] |
| gru_6000 | 0.1758 [0.1303, 0.2212] | 0.0533 [0.0426, 0.0654] | 0.2383 [0.1887, 0.2786] |
| gru_6001 | 0.1712 [0.1278, 0.2203] | 0.0532 [0.0418, 0.0650] | 0.1976 [0.1597, 0.2481] |
| gru_6002 | 0.1733 [0.1312, 0.2104] | 0.0540 [0.0422, 0.0658] | 0.2179 [0.1760, 0.2642] |
| gru_ensemble | 0.1753 [0.1303, 0.2186] | 0.0534 [0.0414, 0.0653] | 0.2145 [0.1730, 0.2596] |

GRU seed min–max PR-AUC: (0.17123134608220045, 0.17582730659843762).

GRU ensemble minus replicator: 0.0627 [0.0366, 0.0929].
GRU ensemble minus lightgbm: 0.0606 [0.0373, 0.0893].
GRU ensemble minus trivial_reference: 0.0616 [0.0204, 0.1028].

### beta=0.5/noise / default

Test base rate: 0.047918; positive labels: 1174.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0896 [0.0641, 0.1172] | 0.0452 [0.0350, 0.0563] | 0.1627 [0.1236, 0.2095] |
| logistic_regression | 0.3572 [0.2150, 0.4803] | 0.0373 [0.0294, 0.0448] | 0.4566 [0.3558, 0.5467] |
| lightgbm | 0.5028 [0.3498, 0.6205] | 0.0323 [0.0256, 0.0386] | 0.6014 [0.5154, 0.6777] |
| replicator | 0.0969 [0.0753, 0.1202] | 0.0447 [0.0348, 0.0559] | 0.1576 [0.1135, 0.2019] |
| gru_6000 | 0.5951 [0.4814, 0.6878] | 0.0284 [0.0228, 0.0342] | 0.6968 [0.6326, 0.7574] |
| gru_6001 | 0.6000 [0.4624, 0.6957] | 0.0283 [0.0230, 0.0344] | 0.6925 [0.6292, 0.7505] |
| gru_6002 | 0.5837 [0.4451, 0.6846] | 0.0289 [0.0231, 0.0348] | 0.6874 [0.6194, 0.7523] |
| gru_ensemble | 0.6055 [0.4913, 0.6975] | 0.0280 [0.0225, 0.0334] | 0.7027 [0.6419, 0.7596] |

GRU seed min–max PR-AUC: (0.5837198769894383, 0.6000045286924006).

GRU ensemble minus replicator: 0.5086 [0.3858, 0.5968].
GRU ensemble minus lightgbm: 0.1028 [0.0776, 0.1330].
GRU ensemble minus trivial_reference: 0.5159 [0.3918, 0.6021].

### beta=0.5/noise / forced_flow

Test base rate: 0.047878; positive labels: 1173.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1134 [0.0834, 0.1485] | 0.0440 [0.0335, 0.0548] | 0.1765 [0.1139, 0.2538] |
| logistic_regression | 0.1248 [0.0937, 0.1587] | 0.0436 [0.0335, 0.0536] | 0.1765 [0.1212, 0.2405] |
| lightgbm | 0.1277 [0.0927, 0.1614] | 0.0435 [0.0342, 0.0540] | 0.1918 [0.1315, 0.2570] |
| replicator | 0.1032 [0.0702, 0.1327] | 0.0442 [0.0351, 0.0551] | 0.1415 [0.1015, 0.1996] |
| gru_6000 | 0.1379 [0.0990, 0.1762] | 0.0430 [0.0336, 0.0535] | 0.1884 [0.1353, 0.2596] |
| gru_6001 | 0.1484 [0.1068, 0.1939] | 0.0428 [0.0331, 0.0525] | 0.2276 [0.1485, 0.3101] |
| gru_6002 | 0.1576 [0.1086, 0.2106] | 0.0423 [0.0335, 0.0519] | 0.2455 [0.1498, 0.3327] |
| gru_ensemble | 0.1580 [0.1152, 0.2045] | 0.0424 [0.0324, 0.0524] | 0.2532 [0.1750, 0.3334] |

GRU seed min–max PR-AUC: (0.13791466476178282, 0.15756116276385762).

GRU ensemble minus replicator: 0.0549 [0.0286, 0.0917].
GRU ensemble minus lightgbm: 0.0304 [0.0072, 0.0607].
GRU ensemble minus trivial_reference: 0.0446 [0.0174, 0.0754].

### beta=0.5/rho=0.25 / default

Test base rate: 0.042041; positive labels: 1030.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0684 [0.0544, 0.0844] | 0.0400 [0.0297, 0.0512] | 0.1010 [0.0811, 0.1286] |
| logistic_regression | 0.2377 [0.1760, 0.3036] | 0.0361 [0.0283, 0.0446] | 0.3650 [0.2978, 0.4393] |
| lightgbm | 0.3575 [0.2712, 0.4310] | 0.0327 [0.0255, 0.0399] | 0.4816 [0.4206, 0.5414] |
| replicator | 0.0934 [0.0737, 0.1157] | 0.0392 [0.0293, 0.0488] | 0.1534 [0.1108, 0.2032] |
| gru_6000 | 0.4322 [0.3441, 0.5020] | 0.0304 [0.0238, 0.0375] | 0.5777 [0.5191, 0.6232] |
| gru_6001 | 0.4393 [0.3456, 0.5064] | 0.0301 [0.0237, 0.0367] | 0.5845 [0.5249, 0.6329] |
| gru_6002 | 0.4249 [0.3355, 0.4971] | 0.0305 [0.0250, 0.0376] | 0.5699 [0.5078, 0.6225] |
| gru_ensemble | 0.4465 [0.3617, 0.5187] | 0.0299 [0.0234, 0.0371] | 0.5951 [0.5291, 0.6433] |

GRU seed min–max PR-AUC: (0.4248904091035228, 0.4392552720827821).

GRU ensemble minus replicator: 0.3531 [0.2725, 0.4100].
GRU ensemble minus lightgbm: 0.0890 [0.0557, 0.1215].
GRU ensemble minus trivial_reference: 0.3781 [0.2933, 0.4384].

### beta=0.5/rho=0.25 / forced_flow

Test base rate: 0.058531; positive labels: 1434.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1563 [0.1049, 0.2171] | 0.0525 [0.0407, 0.0654] | 0.1750 [0.1075, 0.2569] |
| logistic_regression | 0.1552 [0.1003, 0.2193] | 0.0523 [0.0392, 0.0658] | 0.1946 [0.1271, 0.2781] |
| lightgbm | 0.1537 [0.1100, 0.1992] | 0.0524 [0.0403, 0.0655] | 0.1869 [0.1402, 0.2484] |
| replicator | 0.1225 [0.0840, 0.1646] | 0.0533 [0.0408, 0.0667] | 0.1402 [0.0946, 0.1776] |
| gru_6000 | 0.1607 [0.1108, 0.2192] | 0.0517 [0.0400, 0.0641] | 0.2148 [0.1462, 0.3130] |
| gru_6001 | 0.1583 [0.1087, 0.2094] | 0.0517 [0.0401, 0.0644] | 0.2085 [0.1450, 0.2907] |
| gru_6002 | 0.1763 [0.1230, 0.2431] | 0.0514 [0.0401, 0.0638] | 0.2434 [0.1795, 0.3426] |
| gru_ensemble | 0.1738 [0.1195, 0.2317] | 0.0513 [0.0399, 0.0630] | 0.2580 [0.1744, 0.3428] |

GRU seed min–max PR-AUC: (0.15827494418157206, 0.1763468708192961).

GRU ensemble minus replicator: 0.0513 [0.0232, 0.0879].
GRU ensemble minus lightgbm: 0.0201 [-0.0100, 0.0579].
GRU ensemble minus trivial_reference: 0.0175 [-0.0282, 0.0543].

### beta=0.5/rho=0.5 / default

Test base rate: 0.046000; positive labels: 1127.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0849 [0.0748, 0.0975] | 0.0432 [0.0346, 0.0524] | 0.1224 [0.0995, 0.1552] |
| logistic_regression | 0.2775 [0.2343, 0.3252] | 0.0379 [0.0312, 0.0460] | 0.4073 [0.3537, 0.4745] |
| lightgbm | 0.3010 [0.2442, 0.3551] | 0.0373 [0.0309, 0.0446] | 0.4241 [0.3692, 0.4775] |
| replicator | 0.0928 [0.0789, 0.1099] | 0.0429 [0.0347, 0.0522] | 0.1287 [0.0987, 0.1640] |
| gru_6000 | 0.3395 [0.2892, 0.3906] | 0.0360 [0.0295, 0.0432] | 0.4667 [0.4180, 0.5311] |
| gru_6001 | 0.3460 [0.2877, 0.3962] | 0.0358 [0.0290, 0.0431] | 0.4800 [0.4304, 0.5293] |
| gru_6002 | 0.3282 [0.2747, 0.3778] | 0.0364 [0.0297, 0.0433] | 0.4623 [0.4170, 0.5163] |
| gru_ensemble | 0.3454 [0.2876, 0.3939] | 0.0358 [0.0289, 0.0421] | 0.4800 [0.4241, 0.5303] |

GRU seed min–max PR-AUC: (0.32818303312246294, 0.3460475660720492).

GRU ensemble minus replicator: 0.2526 [0.2049, 0.2937].
GRU ensemble minus lightgbm: 0.0444 [0.0202, 0.0688].
GRU ensemble minus trivial_reference: 0.2605 [0.2116, 0.3033].

### beta=0.5/rho=0.5 / forced_flow

Test base rate: 0.051429; positive labels: 1260.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1801 [0.1228, 0.2489] | 0.0454 [0.0332, 0.0573] | 0.2627 [0.1705, 0.3550] |
| logistic_regression | 0.1735 [0.1070, 0.2565] | 0.0455 [0.0350, 0.0567] | 0.2405 [0.1528, 0.3277] |
| lightgbm | 0.1480 [0.1063, 0.1908] | 0.0456 [0.0348, 0.0580] | 0.1881 [0.1291, 0.2690] |
| replicator | 0.1317 [0.0987, 0.1673] | 0.0465 [0.0348, 0.0587] | 0.1548 [0.1183, 0.2070] |
| gru_6000 | 0.1930 [0.1411, 0.2447] | 0.0447 [0.0338, 0.0549] | 0.2841 [0.2021, 0.3874] |
| gru_6001 | 0.1891 [0.1357, 0.2489] | 0.0443 [0.0336, 0.0558] | 0.2817 [0.2080, 0.3794] |
| gru_6002 | 0.1908 [0.1383, 0.2501] | 0.0442 [0.0339, 0.0554] | 0.2865 [0.1995, 0.3835] |
| gru_ensemble | 0.1932 [0.1390, 0.2486] | 0.0442 [0.0339, 0.0549] | 0.2897 [0.2042, 0.3942] |

GRU seed min–max PR-AUC: (0.1890735813751729, 0.19299624406730514).

GRU ensemble minus replicator: 0.0615 [0.0298, 0.0979].
GRU ensemble minus lightgbm: 0.0452 [0.0167, 0.0771].
GRU ensemble minus trivial_reference: 0.0130 [-0.0301, 0.0575].

### beta=1/noise / default

Test base rate: 0.050776; positive labels: 1244.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0612 [0.0562, 0.0675] | 0.0481 [0.0444, 0.0518] | 0.0788 [0.0650, 0.0950] |
| logistic_regression | 0.1288 [0.1149, 0.1438] | 0.0465 [0.0428, 0.0500] | 0.1897 [0.1675, 0.2140] |
| lightgbm | 0.1198 [0.1072, 0.1347] | 0.0467 [0.0431, 0.0502] | 0.1768 [0.1586, 0.2043] |
| replicator | 0.0740 [0.0629, 0.0893] | 0.0480 [0.0441, 0.0515] | 0.1158 [0.0890, 0.1422] |
| gru_6000 | 0.1475 [0.1263, 0.1724] | 0.0459 [0.0425, 0.0494] | 0.2211 [0.1859, 0.2544] |
| gru_6001 | 0.1473 [0.1260, 0.1726] | 0.0460 [0.0427, 0.0494] | 0.2235 [0.1909, 0.2555] |
| gru_6002 | 0.1466 [0.1270, 0.1699] | 0.0461 [0.0426, 0.0497] | 0.2178 [0.1835, 0.2538] |
| gru_ensemble | 0.1505 [0.1309, 0.1748] | 0.0459 [0.0424, 0.0494] | 0.2235 [0.1900, 0.2578] |

GRU seed min–max PR-AUC: (0.14663529586597362, 0.14746782515708146).

GRU ensemble minus replicator: 0.0765 [0.0632, 0.0923].
GRU ensemble minus lightgbm: 0.0306 [0.0161, 0.0471].
GRU ensemble minus trivial_reference: 0.0892 [0.0703, 0.1124].

### beta=1/noise / forced_flow

Test base rate: 0.046286; positive labels: 1134.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0743 [0.0575, 0.0944] | 0.0441 [0.0353, 0.0533] | 0.1014 [0.0590, 0.1374] |
| logistic_regression | 0.0758 [0.0567, 0.1051] | 0.0439 [0.0352, 0.0536] | 0.1005 [0.0631, 0.1439] |
| lightgbm | 0.0830 [0.0600, 0.1143] | 0.0436 [0.0345, 0.0532] | 0.1190 [0.0812, 0.1572] |
| replicator | 0.0686 [0.0502, 0.0907] | 0.0440 [0.0344, 0.0539] | 0.1076 [0.0688, 0.1530] |
| gru_6000 | 0.0866 [0.0657, 0.1155] | 0.0437 [0.0347, 0.0532] | 0.1499 [0.1002, 0.2122] |
| gru_6001 | 0.0866 [0.0665, 0.1118] | 0.0441 [0.0346, 0.0526] | 0.1429 [0.0960, 0.2047] |
| gru_6002 | 0.0857 [0.0651, 0.1188] | 0.0439 [0.0349, 0.0539] | 0.1429 [0.0904, 0.2058] |
| gru_ensemble | 0.0871 [0.0651, 0.1173] | 0.0438 [0.0351, 0.0533] | 0.1517 [0.0961, 0.2128] |

GRU seed min–max PR-AUC: (0.08573727691901672, 0.08659807963478042).

GRU ensemble minus replicator: 0.0185 [0.0045, 0.0356].
GRU ensemble minus lightgbm: 0.0040 [-0.0083, 0.0148].
GRU ensemble minus trivial_reference: 0.0128 [0.0016, 0.0276].

### beta=1/rho=0.25 / default

Test base rate: 0.055388; positive labels: 1357.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0877 [0.0776, 0.1013] | 0.0520 [0.0477, 0.0563] | 0.1164 [0.1018, 0.1299] |
| logistic_regression | 0.2228 [0.2000, 0.2483] | 0.0478 [0.0445, 0.0514] | 0.2815 [0.2566, 0.3070] |
| lightgbm | 0.2045 [0.1859, 0.2276] | 0.0484 [0.0447, 0.0523] | 0.2498 [0.2271, 0.2743] |
| replicator | 0.0899 [0.0762, 0.1097] | 0.0518 [0.0478, 0.0558] | 0.1223 [0.0939, 0.1552] |
| gru_6000 | 0.2305 [0.2058, 0.2600] | 0.0475 [0.0442, 0.0510] | 0.2778 [0.2529, 0.3125] |
| gru_6001 | 0.2329 [0.2055, 0.2660] | 0.0475 [0.0443, 0.0513] | 0.2859 [0.2556, 0.3150] |
| gru_6002 | 0.2339 [0.2088, 0.2623] | 0.0475 [0.0443, 0.0508] | 0.2903 [0.2646, 0.3176] |
| gru_ensemble | 0.2362 [0.2101, 0.2666] | 0.0474 [0.0440, 0.0505] | 0.2867 [0.2589, 0.3183] |

GRU seed min–max PR-AUC: (0.2304819927147472, 0.23391050255424653).

GRU ensemble minus replicator: 0.1463 [0.1211, 0.1687].
GRU ensemble minus lightgbm: 0.0317 [0.0179, 0.0450].
GRU ensemble minus trivial_reference: 0.1485 [0.1256, 0.1746].

### beta=1/rho=0.25 / forced_flow

Test base rate: 0.009224; positive labels: 226.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0347 [0.0129, 0.0702] | 0.0091 [0.0043, 0.0146] | 0.2566 [0.1172, 0.3952] |
| logistic_regression | 0.0450 [0.0157, 0.0923] | 0.0090 [0.0042, 0.0145] | 0.2965 [0.1347, 0.4422] |
| lightgbm | 0.0471 [0.0148, 0.1012] | 0.0090 [0.0047, 0.0141] | 0.2920 [0.1099, 0.4358] |
| replicator | 0.0366 [0.0090, 0.1036] | 0.0090 [0.0043, 0.0144] | 0.2743 [0.0866, 0.4176] |
| gru_6000 | 0.1522 [0.0207, 0.2831] | 0.0086 [0.0043, 0.0136] | 0.3142 [0.0875, 0.5833] |
| gru_6001 | 0.1209 [0.0145, 0.2518] | 0.0087 [0.0041, 0.0139] | 0.3097 [0.0735, 0.5333] |
| gru_6002 | 0.1296 [0.0163, 0.2472] | 0.0086 [0.0042, 0.0134] | 0.3230 [0.0909, 0.5741] |
| gru_ensemble | 0.1468 [0.0143, 0.2737] | 0.0086 [0.0042, 0.0135] | 0.3097 [0.0735, 0.5335] |

GRU seed min–max PR-AUC: (0.12092812643141436, 0.15219630458360522).

GRU ensemble minus replicator: 0.1102 [-0.0000, 0.2226].
GRU ensemble minus lightgbm: 0.0996 [-0.0005, 0.2019].
GRU ensemble minus trivial_reference: 0.1121 [0.0021, 0.2184].

### beta=1/rho=0.5 / default

Test base rate: 0.053510; positive labels: 1311.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0834 [0.0755, 0.0929] | 0.0503 [0.0461, 0.0543] | 0.1159 [0.1007, 0.1347] |
| logistic_regression | 0.2557 [0.2317, 0.2818] | 0.0450 [0.0412, 0.0487] | 0.3318 [0.3025, 0.3611] |
| lightgbm | 0.2291 [0.2093, 0.2525] | 0.0459 [0.0425, 0.0496] | 0.3013 [0.2769, 0.3308] |
| replicator | 0.0832 [0.0716, 0.1011] | 0.0502 [0.0463, 0.0543] | 0.1076 [0.0851, 0.1338] |
| gru_6000 | 0.2634 [0.2376, 0.2915] | 0.0446 [0.0408, 0.0480] | 0.3425 [0.3079, 0.3749] |
| gru_6001 | 0.2654 [0.2403, 0.2905] | 0.0446 [0.0408, 0.0480] | 0.3402 [0.3097, 0.3747] |
| gru_6002 | 0.2652 [0.2387, 0.2947] | 0.0446 [0.0410, 0.0483] | 0.3509 [0.3153, 0.3835] |
| gru_ensemble | 0.2672 [0.2428, 0.2938] | 0.0445 [0.0410, 0.0485] | 0.3455 [0.3101, 0.3819] |

GRU seed min–max PR-AUC: (0.26340092480083693, 0.2654196007894988).

GRU ensemble minus replicator: 0.1840 [0.1600, 0.2080].
GRU ensemble minus lightgbm: 0.0381 [0.0231, 0.0529].
GRU ensemble minus trivial_reference: 0.1838 [0.1624, 0.2057].

### beta=1/rho=0.5 / forced_flow

Test base rate: 0.001143; positive labels: 28.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0195 [0.0019, 0.0926] | 0.0011 [0.0000, 0.0028] | 0.4643 [0.0000, 0.9286] |
| logistic_regression | 0.0219 [0.0017, 0.1096] | 0.0011 [0.0000, 0.0028] | 0.4643 [0.0000, 0.9286] |
| lightgbm | 0.0584 [0.0032, 0.2243] | 0.0012 [0.0001, 0.0029] | 0.5357 [0.2143, 0.8571] |
| replicator | 0.0145 [0.0020, 0.0641] | 0.0012 [0.0001, 0.0029] | 0.3929 [0.1429, 0.7857] |
| gru_6000 | 0.1159 [0.0033, 0.7699] | 0.0011 [0.0000, 0.0028] | 0.6071 [0.0714, 1.0000] |
| gru_6001 | 0.0349 [0.0027, 0.2256] | 0.0011 [0.0000, 0.0028] | 0.5714 [0.1429, 1.0000] |
| gru_6002 | 0.0457 [0.0015, 0.2385] | 0.0011 [0.0000, 0.0028] | 0.5000 [0.0000, 1.0000] |
| gru_ensemble | 0.0557 [0.0029, 0.4209] | 0.0011 [0.0000, 0.0028] | 0.5714 [0.0714, 1.0000] |

GRU seed min–max PR-AUC: (0.03494190760550041, 0.11585221567513618).

GRU ensemble minus replicator: 0.0413 [0.0009, 0.3061].
GRU ensemble minus lightgbm: -0.0027 [-0.0649, 0.2246].
GRU ensemble minus trivial_reference: 0.0362 [0.0009, 0.2895].

## Joint gain contrasts

Gain = GRU ensemble PR-AUC minus trivial PR-AUC. Noise effect = gain(off) − gain(noise). Information effect = gain(signal) − gain(noise). Compared with round 5 is descriptive, at the same numeric β, and that β meant a different size of signal in round 5.

| Label | Contrast | Gain difference (95% CI) | Outcome | Compared with round 5 |
| --- | --- | ---: | --- | --- |
| default | noise beta=0.25 | 0.2256 [0.1049, 0.3876] | noise degrades inference | — |
| default | information beta=0.25, rho=0.25 | 0.0954 [-0.0420, 0.2485] | no directional information effect established | same direction |
| default | information beta=0.25, rho=0.5 | -0.0728 [-0.2205, 0.0741] | no directional information effect established | direction changed |
| default | noise beta=0.5 | 0.3507 [0.2460, 0.4700] | noise degrades inference | — |
| default | information beta=0.5, rho=0.25 | -0.1378 [-0.2594, -0.0039] | private information makes inference harder | same direction |
| default | information beta=0.5, rho=0.5 | -0.2554 [-0.3465, -0.1332] | private information makes inference harder | same direction |
| default | noise beta=1 | 0.7774 [0.7326, 0.8120] | noise degrades inference | — |
| default | information beta=1, rho=0.25 | 0.0593 [0.0327, 0.0805] | private information makes inference easier | same direction |
| default | information beta=1, rho=0.5 | 0.0945 [0.0651, 0.1224] | private information makes inference easier | same direction |
| forced_flow | noise beta=0.25 | -0.0322 [-0.1024, 0.0334] | noise degradation not established | — |
| forced_flow | information beta=0.25, rho=0.25 | -0.0308 [-0.1084, 0.0389] | no directional information effect established | — |
| forced_flow | information beta=0.25, rho=0.5 | -0.0200 [-0.0904, 0.0486] | no directional information effect established | — |
| forced_flow | noise beta=0.5 | 0.0048 [-0.0392, 0.0486] | noise degradation not established | — |
| forced_flow | information beta=0.5, rho=0.25 | -0.0270 [-0.0653, 0.0095] | no directional information effect established | — |
| forced_flow | information beta=0.5, rho=0.5 | -0.0315 [-0.0908, 0.0205] | no directional information effect established | — |
| forced_flow | noise beta=1 | 0.0366 [0.0033, 0.0756] | noise degrades inference | — |
| forced_flow | information beta=1, rho=0.25 | 0.0993 [-0.0128, 0.2029] | uninformative | — |
| forced_flow | information beta=1, rho=0.5 | 0.0234 [-0.0231, 0.2877] | underpowered; uninformative | — |

## Pre-registered rule outcomes (verbatim)

- Replication / informativeness / power — off/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — off/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/noise/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/noise/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/rho=0.25/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/rho=0.25/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/rho=0.5/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/rho=0.5/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.5/noise/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.5/noise/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.5/rho=0.25/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.5/rho=0.25/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.5/rho=0.5/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.5/rho=0.5/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=1/noise/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=1/noise/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=1/rho=0.25/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=1/rho=0.25/forced_flow: uninformative.
- Replication / informativeness / power — beta=1/rho=0.5/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=1/rho=0.5/forced_flow: uninformative; underpowered.
- default / noise beta=0.25: noise degrades inference.
- default / information beta=0.25, rho=0.25: no directional information effect established.
- default / information beta=0.25, rho=0.5: no directional information effect established.
- default / noise beta=0.5: noise degrades inference.
- default / information beta=0.5, rho=0.25: private information makes inference harder.
- default / information beta=0.5, rho=0.5: private information makes inference harder.
- default / noise beta=1: noise degrades inference.
- default / information beta=1, rho=0.25: private information makes inference easier.
- default / information beta=1, rho=0.5: private information makes inference easier.
- Swamping — none.
- forced_flow / noise beta=0.25: noise degradation not established.
- forced_flow / information beta=0.25, rho=0.25: no directional information effect established.
- forced_flow / information beta=0.25, rho=0.5: no directional information effect established.
- forced_flow / noise beta=0.5: noise degradation not established.
- forced_flow / information beta=0.5, rho=0.25: no directional information effect established.
- forced_flow / information beta=0.5, rho=0.5: no directional information effect established.
- forced_flow / noise beta=1: noise degrades inference.
- forced_flow / information beta=1, rho=0.25: uninformative.
- forced_flow / information beta=1, rho=0.5: underpowered; uninformative.

## Research-lead notes (added by hand after the run)

**Runtime limit exceeded.** The 57-minute estimate was about half the real cost. This grid took 125.17 minutes end to end. The wall time quoted at the top stops before the contrast bootstraps, so it reads lower. With the 25.80-minute population run, the total was about 151 minutes against the ~130-minute limit, and the pre-registration says to stop and report at that point. Nothing in the runner enforces the limit, so all 10 conditions ran to completion. Conditions 9 and 10 finished past it. No setting was changed, and the results are kept with this deviation recorded.

**Population runtime corrected.** The runtime line originally said 28.80 minutes for the population run. That was a typing error; its report says 25.80. Both this report and the runner are fixed. Editing the runner changes the source fingerprint, so a rerun recomputes rather than reusing the cache.

**"Direction changed" overstates one row.** At β = 0.25, ρ = 0.5 the comparison column reports a change because the rule outcome went from *harder* in round 5 to *no directional effect*. The point estimate is still negative (−0.0728 [−0.2205, 0.0741]), so the sign did not flip.

**β is not the signal-to-forecast ratio at β = 1.** The signal term has standard deviation β × constant, so base-forecast spread relative to the constant is β / ratio. For linear agents that is about 1.0 at β = 0.25 and 0.81 at β = 0.5, then 0.17 at β = 1 with noise only. Their base forecasts collapse there, so the fixed constants stop equalizing families: ratios run from 1.24 for momentum to 5.79 for linear. The cause was not tested. The earlier unit test caps the ratio spread at 3 on a short run, which didn't reveal this. The β = 1 rows therefore aren't the fair comparison the design intended.

**Forced-flow thresholds.** The training 0.95 quantile is exactly 0 in five conditions (all three β = 0.5 cells, and β = 1 with ρ = 0.25 and 0.5), so any nonzero forced flow counts as an event there. The zero-threshold question from the mechanism step stays open.
