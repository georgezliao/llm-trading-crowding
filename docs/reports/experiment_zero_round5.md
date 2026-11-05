# Experiment Zero, round 5: noise and information

**Synthetic-simulator disclaimer:** All results are from the uncalibrated synthetic HerdLab simulator, not real markets, funds, or live trading.

## Configuration, seeds, and runtime

This tests a post-hoc noise hypothesis formed after round 4. Amendment 2 approved the 81.06-minute estimate and raises the combined R5-3/R5-4 allowance to approximately 120 minutes. R5-4 is not run here.

Measured resumed invocation time before report generation: 41.73 minutes; its completion log records 47.74 minutes including report generation. Sum of cached condition compute times: 97.57 minutes. Parallelism: 10 simulator workers, 3 GRU workers.

Each of 17 conditions uses 200 training episodes (8000–8199) and 50 test episodes (8500–8549), 500 steps each. Horizon 10; separate training-only 0.95 thresholds for default (primary) and forced-flow (secondary). Return lookback 10; unchanged round-4 observer defaults; GRU seeds 6000–6002; LightGBM/replicator RNG seeds 5000 + label index.

All CIs are 95% percentile intervals from 1,000 whole-episode resamples. Cross-condition gain differences resample the same ordered test seeds jointly. Metric RNG seeds: 9000 + 100000 × condition index + round-4 helper offsets; contrasts: 4000000 + 100 × label index + contrast index; scale: 3000000 + condition index; test base rate: 2000000 + 10 × condition index + label index. Indices start at zero in displayed order.

Finished conditions are atomically cached under `runs/round5_cache/`, including predictions needed for joint contrasts, diagnostic CIs, and compute times. Cache fingerprints cover resolved configuration, seeds, and Python source; mismatches are recomputed.

Execution provenance: `runs/round5.log` records a completed full invocation (104.19 minutes); `runs/round5_resume.log` records a completed resumed invocation (47.74 minutes), reusing the first 11 conditions and recomputing the last six. `runs/round5_session.log` also records an interrupted attempt. These are pre-existing execution logs verified during finalization; the 97.57-minute sum describes only the retained condition computations, not all attempted or duplicated work. The logs do not establish that total execution across attempts stayed within the approximately 120-minute allowance. No additional simulation was needed for final verification.

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

Every condition forces learned_overlap_mode=none. Overrides below are the complete changes to the base configuration.

| Condition | Signal mode | Beta | Rho | Max drawdown | Compute minutes |
| --- | --- | ---: | ---: | ---: | ---: |
| off | off | 1 | 0 | 0.2 | 3.37 |
| beta=0.1/noise | noise_only | 0.1 | 0 | 0.2 | 3.52 |
| beta=0.1/rho=0.25 | signal | 0.1 | 0.25 | 0.2 | 3.45 |
| beta=0.1/rho=0.5 | signal | 0.1 | 0.5 | 0.2 | 3.27 |
| beta=0.25/noise | noise_only | 0.25 | 0 | 0.2 | 3.46 |
| beta=0.25/rho=0.25 | signal | 0.25 | 0.25 | 0.2 | 3.58 |
| beta=0.25/rho=0.5 | signal | 0.25 | 0.5 | 0.2 | 3.48 |
| beta=0.5/noise | noise_only | 0.5 | 0 | 0.2 | 5.38 |
| beta=0.5/rho=0.25 | signal | 0.5 | 0.25 | 0.2 | 7.04 |
| beta=0.5/rho=0.5 | signal | 0.5 | 0.5 | 0.2 | 8.61 |
| beta=1/noise | noise_only | 1 | 0 | 0.2 | 10.70 |
| beta=1/rho=0.25 | signal | 1 | 0.25 | 0.2 | 7.65 |
| beta=1/rho=0.5 | signal | 1 | 0.5 | 0.2 | 8.63 |
| stress/rho=0.25/drawdown=0.05 | signal | 0.25 | 0.25 | 0.05 | 5.11 |
| stress/rho=0.25/drawdown=0.1 | signal | 0.25 | 0.25 | 0.1 | 4.94 |
| stress/rho=0.5/drawdown=0.05 | signal | 0.25 | 0.5 | 0.05 | 5.01 |
| stress/rho=0.5/drawdown=0.1 | signal | 0.25 | 0.5 | 0.1 | 10.38 |

## Thresholds and base rates

| Condition | Label | Threshold | Train positives / labels | Train rate | Test positives / labels | Test rate (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| off | default | 0.122034186585 | 4900/98000 | 0.050000 | 1224/24500 | 0.0500 [0.0302, 0.0705] |
| off | forced_flow | 0.000282089357967 | 4900/98000 | 0.050000 | 1519/24500 | 0.0620 [0.0493, 0.0744] |
| beta=0.1/noise | default | 0.0662937124006 | 4900/98000 | 0.050000 | 941/24500 | 0.0384 [0.0248, 0.0564] |
| beta=0.1/noise | forced_flow | 0.000633043721224 | 4900/98000 | 0.050000 | 1291/24500 | 0.0527 [0.0407, 0.0649] |
| beta=0.1/rho=0.25 | default | 0.0510066326198 | 4900/98000 | 0.050000 | 1171/24500 | 0.0478 [0.0306, 0.0669] |
| beta=0.1/rho=0.25 | forced_flow | 0.000383715925922 | 4900/98000 | 0.050000 | 1582/24500 | 0.0646 [0.0502, 0.0787] |
| beta=0.1/rho=0.5 | default | 0.052477918853 | 4900/98000 | 0.050000 | 1025/24500 | 0.0418 [0.0282, 0.0582] |
| beta=0.1/rho=0.5 | forced_flow | 0 | 4310/98000 | 0.043980 | 1298/24500 | 0.0530 [0.0397, 0.0671] |
| beta=0.25/noise | default | 0.059024725044 | 4900/98000 | 0.050000 | 1202/24500 | 0.0491 [0.0319, 0.0669] |
| beta=0.25/noise | forced_flow | 0 | 3334/98000 | 0.034020 | 1097/24500 | 0.0448 [0.0346, 0.0547] |
| beta=0.25/rho=0.25 | default | 0.0562578036611 | 4900/98000 | 0.050000 | 1254/24500 | 0.0512 [0.0338, 0.0716] |
| beta=0.25/rho=0.25 | forced_flow | 0 | 1615/98000 | 0.016480 | 471/24500 | 0.0192 [0.0123, 0.0275] |
| beta=0.25/rho=0.5 | default | 0.0660065605352 | 4900/98000 | 0.050000 | 1043/24500 | 0.0426 [0.0345, 0.0515] |
| beta=0.25/rho=0.5 | forced_flow | 0 | 220/98000 | 0.002245 | 67/24500 | 0.0027 [0.0006, 0.0050] |
| beta=0.5/noise | default | 0.0587394797832 | 4900/98000 | 0.050000 | 1285/24500 | 0.0524 [0.0403, 0.0678] |
| beta=0.5/noise | forced_flow | 0 | 439/98000 | 0.004480 | 86/24500 | 0.0035 [0.0007, 0.0069] |
| beta=0.5/rho=0.25 | default | 0.0648000402627 | 4900/98000 | 0.050000 | 1256/24500 | 0.0513 [0.0442, 0.0581] |
| beta=0.5/rho=0.25 | forced_flow | 0 | 28/98000 | 0.000286 | 0/24500 | 0.0000 [0.0000, 0.0000] |
| beta=0.5/rho=0.5 | default | 0.0908527806498 | 4900/98000 | 0.050000 | 1259/24500 | 0.0514 [0.0475, 0.0556] |
| beta=0.5/rho=0.5 | forced_flow | 0 | 0/98000 | 0.000000 | 0/24500 | 0.0000 [0.0000, 0.0000] |
| beta=1/noise | default | 0.0586071148439 | 4900/98000 | 0.050000 | 1225/24500 | 0.0500 [0.0449, 0.0548] |
| beta=1/noise | forced_flow | 0 | 28/98000 | 0.000286 | 28/24500 | 0.0011 [0.0000, 0.0029] |
| beta=1/rho=0.25 | default | 0.0763860923627 | 4900/98000 | 0.050000 | 1342/24500 | 0.0548 [0.0495, 0.0599] |
| beta=1/rho=0.25 | forced_flow | 0 | 28/98000 | 0.000286 | 0/24500 | 0.0000 [0.0000, 0.0000] |
| beta=1/rho=0.5 | default | 0.126588590119 | 4900/98000 | 0.050000 | 1345/24500 | 0.0549 [0.0503, 0.0600] |
| beta=1/rho=0.5 | forced_flow | 0 | 14/98000 | 0.000143 | 14/24500 | 0.0006 [0.0000, 0.0023] |
| stress/rho=0.25/drawdown=0.05 | default | 0.0560718396861 | 4900/98000 | 0.050000 | 1270/24500 | 0.0518 [0.0347, 0.0686] |
| stress/rho=0.25/drawdown=0.05 | forced_flow | 0.00311398237467 | 4900/98000 | 0.050000 | 1524/24500 | 0.0622 [0.0511, 0.0744] |
| stress/rho=0.25/drawdown=0.1 | default | 0.0564517803492 | 4900/98000 | 0.050000 | 1263/24500 | 0.0516 [0.0342, 0.0711] |
| stress/rho=0.25/drawdown=0.1 | forced_flow | 0.00087228914715 | 4900/98000 | 0.050000 | 1101/24500 | 0.0449 [0.0341, 0.0575] |
| stress/rho=0.5/drawdown=0.05 | default | 0.0663581112505 | 4900/98000 | 0.050000 | 1007/24500 | 0.0411 [0.0329, 0.0499] |
| stress/rho=0.5/drawdown=0.05 | forced_flow | 0.00109233435651 | 4900/98000 | 0.050000 | 1449/24500 | 0.0591 [0.0439, 0.0742] |
| stress/rho=0.5/drawdown=0.1 | default | 0.0659781726114 | 4900/98000 | 0.050000 | 1034/24500 | 0.0422 [0.0339, 0.0511] |
| stress/rho=0.5/drawdown=0.1 | forced_flow | 0 | 1554/98000 | 0.015857 | 398/24500 | 0.0162 [0.0097, 0.0234] |

## Forecast scale ratios

std(signal term) / std(base forecast), pooled across all decision times and agents within each family over the 250 training and test episodes, including warm-up and risk-overridden decisions. CIs resample whole episodes and recompute the pooled standard deviations. These private diagnostics never enter an observer. Undefined ratios are N/A.

| Condition | momentum | reversion | linear | gbdt | mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| beta=0.1/noise | 1.8660 [1.8259, 1.9147] | 1.8775 [1.8401, 1.9202] | 0.1050 [0.1046, 0.1054] | 0.2421 [0.2412, 0.2429] | 0.1194 [0.1180, 0.1207] |
| beta=0.1/rho=0.25 | 1.9373 [1.8937, 1.9831] | 1.9543 [1.9117, 1.9927] | 0.1072 [0.1067, 0.1077] | 0.2458 [0.2446, 0.2470] | 0.1194 [0.1182, 0.1208] |
| beta=0.1/rho=0.5 | 2.0254 [1.9855, 2.0689] | 2.0324 [1.9944, 2.0731] | 0.1133 [0.1122, 0.1144] | 0.2582 [0.2560, 0.2608] | 0.1257 [0.1240, 0.1277] |
| beta=0.25/noise | 4.6147 [4.5205, 4.7185] | 4.6339 [4.5392, 4.7284] | 0.2666 [0.2656, 0.2677] | 0.6133 [0.6106, 0.6164] | 0.3021 [0.2987, 0.3060] |
| beta=0.25/rho=0.25 | 4.6404 [4.5610, 4.7253] | 4.6864 [4.6013, 4.7710] | 0.2786 [0.2770, 0.2802] | 0.6360 [0.6322, 0.6403] | 0.3126 [0.3085, 0.3173] |
| beta=0.25/rho=0.5 | 4.7531 [4.6753, 4.8301] | 4.7582 [4.6840, 4.8311] | 0.3058 [0.3024, 0.3093] | 0.6934 [0.6849, 0.7018] | 0.3469 [0.3405, 0.3537] |
| beta=0.5/noise | 8.2992 [8.1491, 8.4781] | 8.3275 [8.1784, 8.4804] | 0.5532 [0.5505, 0.5560] | 1.2688 [1.2616, 1.2764] | 0.6348 [0.6266, 0.6438] |
| beta=0.5/rho=0.25 | 7.8987 [7.7868, 8.0029] | 7.9137 [7.8078, 8.0140] | 0.6182 [0.6128, 0.6237] | 1.4146 [1.4010, 1.4299] | 0.7257 [0.7128, 0.7397] |
| beta=0.5/rho=0.5 | 7.7017 [7.6107, 7.7924] | 7.6964 [7.6087, 7.7881] | 0.7631 [0.7534, 0.7735] | 1.7065 [1.6835, 1.7320] | 0.9715 [0.9521, 0.9928] |
| beta=1/noise | 11.8795 [11.6851, 12.0834] | 11.8755 [11.6903, 12.0658] | 1.3138 [1.3027, 1.3257] | 2.9775 [2.9503, 3.0072] | 1.5889 [1.5609, 1.6173] |
| beta=1/rho=0.25 | 11.1309 [10.9913, 11.2670] | 11.1366 [11.0046, 11.2687] | 1.8190 [1.7930, 1.8448] | 3.8864 [3.8373, 3.9383] | 2.2353 [2.2060, 2.2642] |
| beta=1/rho=0.5 | 11.4447 [11.3057, 11.5965] | 11.4412 [11.3105, 11.5887] | 2.2683 [2.2367, 2.3000] | 4.5266 [4.4677, 4.5854] | 2.5506 [2.5239, 2.5779] |
| stress/rho=0.25/drawdown=0.05 | 4.6611 [4.5843, 4.7471] | 4.7102 [4.6219, 4.7931] | 0.2779 [0.2763, 0.2797] | 0.6338 [0.6298, 0.6381] | 0.3115 [0.3075, 0.3161] |
| stress/rho=0.25/drawdown=0.1 | 4.6385 [4.5540, 4.7265] | 4.6828 [4.5999, 4.7673] | 0.2784 [0.2768, 0.2802] | 0.6355 [0.6317, 0.6398] | 0.3124 [0.3083, 0.3169] |
| stress/rho=0.5/drawdown=0.05 | 4.7682 [4.6832, 4.8522] | 4.7719 [4.6986, 4.8453] | 0.3053 [0.3021, 0.3087] | 0.6926 [0.6848, 0.7008] | 0.3460 [0.3403, 0.3524] |
| stress/rho=0.5/drawdown=0.1 | 4.7525 [4.6713, 4.8372] | 4.7602 [4.6842, 4.8342] | 0.3057 [0.3024, 0.3090] | 0.6931 [0.6849, 0.7013] | 0.3466 [0.3404, 0.3530] |

## Observer metrics

Entries are estimate [95% CI]. PR-AUC and recall are higher-is-better; Brier is lower-is-better. Base rates and power flags apply to every observer in a cell.

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

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.8733 [0.8344, 0.9049] |
| lightgbm | 0.0054 [0.0010, 0.0111] |
| trivial_reference | 0.8666 [0.8266, 0.8968] |

Training: 6000: epochs=40, best_validation_loss=0.033522398; 6001: epochs=40, best_validation_loss=0.037296954; 6002: epochs=40, best_validation_loss=0.030605735.

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

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0387 [0.0127, 0.0735] |
| lightgbm | 0.0472 [0.0249, 0.0761] |
| trivial_reference | 0.0494 [0.0178, 0.0849] |

Training: 6000: epochs=40, best_validation_loss=0.1673529; 6001: epochs=9, best_validation_loss=0.19388598; 6002: epochs=26, best_validation_loss=0.19049957.

### beta=0.1/noise / default

Test base rate: 0.038408; positive labels: 941.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0800 [0.0523, 0.1116] | 0.0367 [0.0230, 0.0505] | 0.1892 [0.1293, 0.2667] |
| logistic_regression | 0.5248 [0.2698, 0.7235] | 0.0251 [0.0164, 0.0352] | 0.6589 [0.5221, 0.8057] |
| lightgbm | 0.6729 [0.4678, 0.8129] | 0.0205 [0.0134, 0.0277] | 0.7588 [0.6593, 0.8686] |
| replicator | 0.1220 [0.0859, 0.1641] | 0.0356 [0.0224, 0.0501] | 0.2264 [0.1646, 0.3148] |
| gru_6000 | 0.7186 [0.5302, 0.8326] | 0.0187 [0.0124, 0.0252] | 0.8395 [0.7519, 0.9042] |
| gru_6001 | 0.7365 [0.5740, 0.8468] | 0.0186 [0.0120, 0.0256] | 0.8587 [0.7897, 0.9179] |
| gru_6002 | 0.7251 [0.5549, 0.8371] | 0.0185 [0.0124, 0.0251] | 0.8374 [0.7580, 0.9121] |
| gru_ensemble | 0.7402 [0.5717, 0.8460] | 0.0182 [0.0117, 0.0251] | 0.8587 [0.7788, 0.9210] |

GRU seed min–max PR-AUC: (0.7185697771913846, 0.7364553861127602).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.6182 [0.4458, 0.7451] |
| lightgbm | 0.0673 [0.0353, 0.1122] |
| trivial_reference | 0.6602 [0.5058, 0.7721] |

Training: 6000: epochs=40, best_validation_loss=0.10029913; 6001: epochs=40, best_validation_loss=0.049245762; 6002: epochs=40, best_validation_loss=0.082739702.

### beta=0.1/noise / forced_flow

Test base rate: 0.052694; positive labels: 1291.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.1187 [0.0729, 0.1821] | 0.0486 [0.0385, 0.0590] | 0.1580 [0.0924, 0.2246] |
| logistic_regression | 0.1030 [0.0702, 0.1522] | 0.0487 [0.0387, 0.0586] | 0.1278 [0.0716, 0.1988] |
| lightgbm | 0.1104 [0.0740, 0.1630] | 0.0485 [0.0382, 0.0589] | 0.1541 [0.1005, 0.2122] |
| replicator | 0.0785 [0.0579, 0.1028] | 0.0492 [0.0400, 0.0603] | 0.0775 [0.0355, 0.1081] |
| gru_6000 | 0.1170 [0.0808, 0.1599] | 0.0481 [0.0390, 0.0588] | 0.1743 [0.1091, 0.2335] |
| gru_6001 | 0.1196 [0.0842, 0.1627] | 0.0481 [0.0375, 0.0584] | 0.1890 [0.1224, 0.2627] |
| gru_6002 | 0.1221 [0.0844, 0.1665] | 0.0480 [0.0381, 0.0587] | 0.1851 [0.1242, 0.2467] |
| gru_ensemble | 0.1218 [0.0847, 0.1689] | 0.0480 [0.0388, 0.0590] | 0.1836 [0.1255, 0.2548] |

GRU seed min–max PR-AUC: (0.11699790048250314, 0.12214774161498479).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0433 [0.0200, 0.0819] |
| lightgbm | 0.0114 [-0.0054, 0.0283] |
| trivial_reference | 0.0031 [-0.0226, 0.0253] |

Training: 6000: epochs=32, best_validation_loss=0.17931824; 6001: epochs=17, best_validation_loss=0.1904371; 6002: epochs=40, best_validation_loss=0.17398987.

### beta=0.1/rho=0.25 / default

Test base rate: 0.047796; positive labels: 1171.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0818 [0.0600, 0.1064] | 0.0451 [0.0296, 0.0616] | 0.1375 [0.0975, 0.1911] |
| logistic_regression | 0.4461 [0.2490, 0.6202] | 0.0339 [0.0211, 0.0464] | 0.5978 [0.4595, 0.7334] |
| lightgbm | 0.6002 [0.3927, 0.7511] | 0.0285 [0.0186, 0.0391] | 0.7105 [0.5703, 0.8274] |
| replicator | 0.1537 [0.1153, 0.1889] | 0.0429 [0.0282, 0.0604] | 0.2425 [0.1771, 0.3296] |
| gru_6000 | 0.6588 [0.4890, 0.7788] | 0.0258 [0.0166, 0.0362] | 0.7728 [0.6889, 0.8612] |
| gru_6001 | 0.6631 [0.4692, 0.7955] | 0.0257 [0.0164, 0.0358] | 0.7746 [0.6643, 0.8800] |
| gru_6002 | 0.6542 [0.4937, 0.7829] | 0.0261 [0.0172, 0.0359] | 0.7660 [0.6563, 0.8610] |
| gru_ensemble | 0.6770 [0.5060, 0.7990] | 0.0250 [0.0159, 0.0346] | 0.7942 [0.6900, 0.8762] |

GRU seed min–max PR-AUC: (0.6541904368154726, 0.6630690141396917).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.5232 [0.3441, 0.6587] |
| lightgbm | 0.0768 [0.0456, 0.1218] |
| trivial_reference | 0.5952 [0.4271, 0.7244] |

Training: 6000: epochs=40, best_validation_loss=0.10141181; 6001: epochs=40, best_validation_loss=0.080985527; 6002: epochs=40, best_validation_loss=0.10568629.

### beta=0.1/rho=0.25 / forced_flow

Test base rate: 0.064571; positive labels: 1582.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0924 [0.0721, 0.1156] | 0.0599 [0.0475, 0.0720] | 0.0714 [0.0400, 0.1164] |
| logistic_regression | 0.0974 [0.0752, 0.1191] | 0.0596 [0.0470, 0.0721] | 0.0961 [0.0642, 0.1296] |
| lightgbm | 0.0990 [0.0799, 0.1202] | 0.0593 [0.0473, 0.0716] | 0.0815 [0.0550, 0.1089] |
| replicator | 0.0916 [0.0710, 0.1145] | 0.0596 [0.0467, 0.0717] | 0.0759 [0.0494, 0.0967] |
| gru_6000 | 0.1358 [0.1032, 0.1720] | 0.0584 [0.0468, 0.0711] | 0.1422 [0.0914, 0.2030] |
| gru_6001 | 0.1438 [0.1072, 0.1853] | 0.0580 [0.0462, 0.0704] | 0.1479 [0.0927, 0.2148] |
| gru_6002 | 0.1339 [0.1006, 0.1719] | 0.0583 [0.0466, 0.0704] | 0.1359 [0.0834, 0.2124] |
| gru_ensemble | 0.1394 [0.1052, 0.1808] | 0.0582 [0.0464, 0.0690] | 0.1448 [0.0881, 0.2093] |

GRU seed min–max PR-AUC: (0.13385874833311237, 0.14381402364471899).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0478 [0.0227, 0.0824] |
| lightgbm | 0.0405 [0.0106, 0.0826] |
| trivial_reference | 0.0471 [0.0128, 0.0881] |

Training: 6000: epochs=29, best_validation_loss=0.18819694; 6001: epochs=40, best_validation_loss=0.19004491; 6002: epochs=40, best_validation_loss=0.18817335.

### beta=0.1/rho=0.5 / default

Test base rate: 0.041837; positive labels: 1025.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0727 [0.0492, 0.0994] | 0.0398 [0.0279, 0.0529] | 0.1415 [0.1003, 0.1913] |
| logistic_regression | 0.3372 [0.2081, 0.5003] | 0.0328 [0.0227, 0.0444] | 0.5122 [0.4140, 0.6395] |
| lightgbm | 0.4641 [0.2971, 0.6303] | 0.0294 [0.0200, 0.0397] | 0.6000 [0.4911, 0.7164] |
| replicator | 0.1189 [0.0885, 0.1480] | 0.0384 [0.0261, 0.0526] | 0.2195 [0.1632, 0.2905] |
| gru_6000 | 0.5292 [0.3690, 0.6636] | 0.0273 [0.0189, 0.0361] | 0.6780 [0.5756, 0.7646] |
| gru_6001 | 0.5328 [0.3665, 0.6670] | 0.0271 [0.0197, 0.0356] | 0.6859 [0.5885, 0.7705] |
| gru_6002 | 0.5151 [0.3376, 0.6520] | 0.0281 [0.0201, 0.0365] | 0.6537 [0.5566, 0.7488] |
| gru_ensemble | 0.5399 [0.3731, 0.6684] | 0.0269 [0.0187, 0.0352] | 0.6820 [0.5832, 0.7721] |

GRU seed min–max PR-AUC: (0.5151357628166112, 0.5327784679767423).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.4210 [0.2645, 0.5521] |
| lightgbm | 0.0758 [0.0453, 0.1050] |
| trivial_reference | 0.4672 [0.3208, 0.5979] |

Training: 6000: epochs=40, best_validation_loss=0.10982806; 6001: epochs=40, best_validation_loss=0.10459817; 6002: epochs=40, best_validation_loss=0.11248435.

### beta=0.1/rho=0.5 / forced_flow

Test base rate: 0.052980; positive labels: 1298.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0881 [0.0632, 0.1201] | 0.0495 [0.0374, 0.0638] | 0.1279 [0.0677, 0.1916] |
| logistic_regression | 0.0893 [0.0643, 0.1222] | 0.0493 [0.0368, 0.0616] | 0.1240 [0.0707, 0.1665] |
| lightgbm | 0.1036 [0.0746, 0.1376] | 0.0489 [0.0377, 0.0642] | 0.1294 [0.0843, 0.1881] |
| replicator | 0.0863 [0.0654, 0.1102] | 0.0493 [0.0363, 0.0618] | 0.0894 [0.0741, 0.1247] |
| gru_6000 | 0.1335 [0.0890, 0.1844] | 0.0479 [0.0363, 0.0610] | 0.1780 [0.1157, 0.2433] |
| gru_6001 | 0.1264 [0.0870, 0.1762] | 0.0481 [0.0362, 0.0605] | 0.1680 [0.1124, 0.2251] |
| gru_6002 | 0.1286 [0.0873, 0.1801] | 0.0482 [0.0358, 0.0614] | 0.1733 [0.1130, 0.2359] |
| gru_ensemble | 0.1325 [0.0909, 0.1856] | 0.0480 [0.0366, 0.0621] | 0.1695 [0.1162, 0.2380] |

GRU seed min–max PR-AUC: (0.1264256381440713, 0.1335191721714326).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0462 [0.0164, 0.0886] |
| lightgbm | 0.0289 [0.0020, 0.0620] |
| trivial_reference | 0.0444 [0.0136, 0.0797] |

Training: 6000: epochs=30, best_validation_loss=0.17639316; 6001: epochs=15, best_validation_loss=0.18794833; 6002: epochs=24, best_validation_loss=0.16691393.

### beta=0.25/noise / default

Test base rate: 0.049061; positive labels: 1202.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0739 [0.0545, 0.0959] | 0.0464 [0.0307, 0.0626] | 0.1073 [0.0759, 0.1531] |
| logistic_regression | 0.4706 [0.2450, 0.6299] | 0.0340 [0.0243, 0.0444] | 0.5516 [0.3953, 0.6981] |
| lightgbm | 0.5987 [0.3922, 0.7366] | 0.0296 [0.0206, 0.0392] | 0.6913 [0.5898, 0.7863] |
| replicator | 0.1116 [0.0832, 0.1458] | 0.0451 [0.0294, 0.0613] | 0.1581 [0.1033, 0.2388] |
| gru_6000 | 0.6638 [0.5056, 0.7782] | 0.0263 [0.0182, 0.0342] | 0.7704 [0.6923, 0.8405] |
| gru_6001 | 0.6779 [0.5320, 0.7874] | 0.0255 [0.0181, 0.0333] | 0.8012 [0.7324, 0.8618] |
| gru_6002 | 0.6616 [0.5022, 0.7791] | 0.0263 [0.0181, 0.0346] | 0.7870 [0.6931, 0.8498] |
| gru_ensemble | 0.6811 [0.5027, 0.7833] | 0.0255 [0.0182, 0.0331] | 0.8037 [0.7176, 0.8678] |

GRU seed min–max PR-AUC: (0.6616290446800425, 0.6778812394355959).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.5695 [0.4121, 0.6867] |
| lightgbm | 0.0824 [0.0522, 0.1334] |
| trivial_reference | 0.6072 [0.4529, 0.7158] |

Training: 6000: epochs=40, best_validation_loss=0.086895579; 6001: epochs=40, best_validation_loss=0.07427969; 6002: epochs=40, best_validation_loss=0.083880484.

### beta=0.25/noise / forced_flow

Test base rate: 0.044776; positive labels: 1097.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0792 [0.0545, 0.1135] | 0.0423 [0.0333, 0.0515] | 0.1067 [0.0439, 0.1846] |
| logistic_regression | 0.0734 [0.0525, 0.1006] | 0.0423 [0.0332, 0.0510] | 0.0957 [0.0469, 0.1562] |
| lightgbm | 0.0715 [0.0526, 0.0960] | 0.0423 [0.0319, 0.0514] | 0.0912 [0.0569, 0.1380] |
| replicator | 0.0648 [0.0452, 0.0868] | 0.0426 [0.0337, 0.0522] | 0.0921 [0.0539, 0.1304] |
| gru_6000 | 0.0874 [0.0574, 0.1202] | 0.0421 [0.0334, 0.0508] | 0.1440 [0.0872, 0.1945] |
| gru_6001 | 0.0922 [0.0637, 0.1227] | 0.0420 [0.0340, 0.0513] | 0.1477 [0.1032, 0.2120] |
| gru_6002 | 0.0781 [0.0535, 0.1064] | 0.0422 [0.0337, 0.0511] | 0.1121 [0.0625, 0.1677] |
| gru_ensemble | 0.0884 [0.0635, 0.1182] | 0.0421 [0.0332, 0.0514] | 0.1395 [0.0859, 0.1899] |

GRU seed min–max PR-AUC: (0.0781414668963976, 0.09224999785485).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0236 [0.0071, 0.0432] |
| lightgbm | 0.0169 [0.0029, 0.0355] |
| trivial_reference | 0.0092 [-0.0199, 0.0280] |

Training: 6000: epochs=19, best_validation_loss=0.13932888; 6001: epochs=25, best_validation_loss=0.12800245; 6002: epochs=26, best_validation_loss=0.14880825.

### beta=0.25/rho=0.25 / default

Test base rate: 0.051184; positive labels: 1254.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0919 [0.0701, 0.1144] | 0.0480 [0.0339, 0.0651] | 0.1356 [0.1073, 0.1747] |
| logistic_regression | 0.3747 [0.2845, 0.4651] | 0.0391 [0.0276, 0.0511] | 0.5136 [0.4190, 0.6041] |
| lightgbm | 0.4947 [0.3888, 0.5807] | 0.0352 [0.0250, 0.0462] | 0.5933 [0.5208, 0.6775] |
| replicator | 0.1299 [0.1005, 0.1581] | 0.0465 [0.0319, 0.0621] | 0.1930 [0.1485, 0.2278] |
| gru_6000 | 0.5607 [0.4625, 0.6340] | 0.0318 [0.0233, 0.0415] | 0.6683 [0.5978, 0.7386] |
| gru_6001 | 0.5640 [0.4594, 0.6418] | 0.0316 [0.0225, 0.0413] | 0.6667 [0.6034, 0.7393] |
| gru_6002 | 0.5650 [0.4773, 0.6316] | 0.0315 [0.0236, 0.0413] | 0.6802 [0.6136, 0.7418] |
| gru_ensemble | 0.5768 [0.4788, 0.6488] | 0.0311 [0.0223, 0.0404] | 0.6778 [0.6140, 0.7442] |

GRU seed min–max PR-AUC: (0.560687236096969, 0.5650472340551475).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.4469 [0.3611, 0.5176] |
| lightgbm | 0.0820 [0.0568, 0.1110] |
| trivial_reference | 0.4849 [0.4006, 0.5485] |

Training: 6000: epochs=40, best_validation_loss=0.12905601; 6001: epochs=40, best_validation_loss=0.11290847; 6002: epochs=40, best_validation_loss=0.1101999.

### beta=0.25/rho=0.25 / forced_flow

Test base rate: 0.019224; positive labels: 471.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0395 [0.0215, 0.0638] | 0.0187 [0.0118, 0.0261] | 0.1274 [0.0517, 0.2171] |
| logistic_regression | 0.0400 [0.0236, 0.0598] | 0.0187 [0.0113, 0.0264] | 0.1168 [0.0588, 0.1994] |
| lightgbm | 0.0431 [0.0244, 0.0643] | 0.0186 [0.0117, 0.0255] | 0.1253 [0.0778, 0.1798] |
| replicator | 0.0304 [0.0202, 0.0408] | 0.0187 [0.0119, 0.0268] | 0.0637 [0.0394, 0.0932] |
| gru_6000 | 0.0739 [0.0408, 0.1155] | 0.0185 [0.0114, 0.0260] | 0.2442 [0.1672, 0.3309] |
| gru_6001 | 0.0638 [0.0361, 0.1039] | 0.0184 [0.0122, 0.0255] | 0.2399 [0.1462, 0.3576] |
| gru_6002 | 0.0711 [0.0378, 0.1215] | 0.0185 [0.0115, 0.0252] | 0.2420 [0.1469, 0.3412] |
| gru_ensemble | 0.0732 [0.0418, 0.1136] | 0.0184 [0.0117, 0.0256] | 0.2739 [0.1802, 0.3815] |

GRU seed min–max PR-AUC: (0.06379275163610207, 0.07393576209235073).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0428 [0.0130, 0.0818] |
| lightgbm | 0.0301 [0.0093, 0.0612] |
| trivial_reference | 0.0337 [0.0096, 0.0657] |

Training: 6000: epochs=22, best_validation_loss=0.091916422; 6001: epochs=38, best_validation_loss=0.091949246; 6002: epochs=23, best_validation_loss=0.098020528.

### beta=0.25/rho=0.5 / default

Test base rate: 0.042571; positive labels: 1043.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0751 [0.0606, 0.0904] | 0.0405 [0.0329, 0.0487] | 0.1505 [0.1216, 0.1718] |
| logistic_regression | 0.2724 [0.2337, 0.3077] | 0.0355 [0.0287, 0.0420] | 0.4171 [0.3716, 0.4626] |
| lightgbm | 0.2755 [0.2312, 0.3141] | 0.0352 [0.0290, 0.0416] | 0.4123 [0.3697, 0.4483] |
| replicator | 0.0717 [0.0585, 0.0857] | 0.0404 [0.0329, 0.0484] | 0.1141 [0.0921, 0.1333] |
| gru_6000 | 0.3003 [0.2539, 0.3413] | 0.0345 [0.0288, 0.0408] | 0.4535 [0.4026, 0.4991] |
| gru_6001 | 0.3017 [0.2518, 0.3460] | 0.0344 [0.0287, 0.0401] | 0.4573 [0.4030, 0.5137] |
| gru_6002 | 0.2966 [0.2486, 0.3352] | 0.0345 [0.0284, 0.0410] | 0.4506 [0.3992, 0.5005] |
| gru_ensemble | 0.3045 [0.2527, 0.3502] | 0.0343 [0.0284, 0.0405] | 0.4669 [0.4095, 0.5152] |

GRU seed min–max PR-AUC: (0.2966480037216143, 0.3016810623933121).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.2328 [0.1953, 0.2642] |
| lightgbm | 0.0290 [0.0135, 0.0446] |
| trivial_reference | 0.2294 [0.1932, 0.2612] |

Training: 6000: epochs=40, best_validation_loss=0.14900098; 6001: epochs=37, best_validation_loss=0.1348035; 6002: epochs=40, best_validation_loss=0.13164226.

### beta=0.25/rho=0.5 / forced_flow

Test base rate: 0.002735; positive labels: 67.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0057 [0.0010, 0.0157] | 0.0027 [0.0006, 0.0052] | 0.1791 [0.0000, 0.4286] |
| logistic_regression | 0.0067 [0.0014, 0.0193] | 0.0027 [0.0006, 0.0051] | 0.1642 [0.0000, 0.3580] |
| lightgbm | 0.0143 [0.0019, 0.0307] | 0.0028 [0.0006, 0.0051] | 0.1493 [0.0000, 0.3136] |
| replicator | 0.0049 [0.0016, 0.0096] | 0.0027 [0.0009, 0.0052] | 0.0746 [0.0000, 0.1600] |
| gru_6000 | 0.0422 [0.0017, 0.1907] | 0.0027 [0.0006, 0.0051] | 0.2836 [0.0000, 0.6873] |
| gru_6001 | 0.0076 [0.0018, 0.0176] | 0.0027 [0.0006, 0.0050] | 0.1493 [0.0000, 0.3037] |
| gru_6002 | 0.0021 [0.0006, 0.0046] | 0.0027 [0.0006, 0.0050] | 0.0149 [0.0000, 0.0714] |
| gru_ensemble | 0.0428 [0.0016, 0.1779] | 0.0027 [0.0006, 0.0054] | 0.2985 [0.0000, 0.7179] |

GRU seed min–max PR-AUC: (0.0021283765266125257, 0.04219632660450235).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0379 [-0.0017, 0.1763] |
| lightgbm | 0.0285 [-0.0098, 0.1675] |
| trivial_reference | 0.0372 [0.0005, 0.1732] |

Training: 6000: epochs=40, best_validation_loss=0.013766646; 6001: epochs=14, best_validation_loss=0.0071924883; 6002: epochs=11, best_validation_loss=0.024557283.

### beta=0.5/noise / default

Test base rate: 0.052449; positive labels: 1285.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0742 [0.0571, 0.0949] | 0.0494 [0.0374, 0.0637] | 0.0895 [0.0650, 0.1194] |
| logistic_regression | 0.2857 [0.2027, 0.3634] | 0.0428 [0.0337, 0.0532] | 0.4062 [0.3272, 0.4826] |
| lightgbm | 0.4221 [0.3127, 0.5061] | 0.0388 [0.0308, 0.0472] | 0.5331 [0.4462, 0.6024] |
| replicator | 0.0973 [0.0745, 0.1198] | 0.0488 [0.0371, 0.0617] | 0.1518 [0.1187, 0.1825] |
| gru_6000 | 0.4886 [0.3911, 0.5676] | 0.0353 [0.0279, 0.0426] | 0.6125 [0.5446, 0.6667] |
| gru_6001 | 0.4827 [0.3700, 0.5692] | 0.0357 [0.0283, 0.0435] | 0.6039 [0.5213, 0.6640] |
| gru_6002 | 0.4887 [0.3832, 0.5740] | 0.0352 [0.0280, 0.0423] | 0.5992 [0.5342, 0.6644] |
| gru_ensemble | 0.4982 [0.3924, 0.5775] | 0.0350 [0.0280, 0.0426] | 0.6117 [0.5487, 0.6722] |

GRU seed min–max PR-AUC: (0.4826702189631331, 0.4886946383675115).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.4009 [0.3037, 0.4666] |
| lightgbm | 0.0761 [0.0503, 0.1023] |
| trivial_reference | 0.4240 [0.3239, 0.4980] |

Training: 6000: epochs=40, best_validation_loss=0.1351817; 6001: epochs=40, best_validation_loss=0.11685095; 6002: epochs=40, best_validation_loss=0.10603104.

### beta=0.5/noise / forced_flow

Test base rate: 0.003510; positive labels: 86.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0069 [0.0016, 0.0137] | 0.0035 [0.0012, 0.0071] | 0.0465 [0.0000, 0.1071] |
| logistic_regression | 0.0059 [0.0012, 0.0142] | 0.0035 [0.0012, 0.0070] | 0.0116 [0.0000, 0.1229] |
| lightgbm | 0.0090 [0.0022, 0.0204] | 0.0035 [0.0012, 0.0071] | 0.0465 [0.0000, 0.1906] |
| replicator | 0.0061 [0.0020, 0.0116] | 0.0035 [0.0006, 0.0065] | 0.0698 [0.0199, 0.1087] |
| gru_6000 | 0.0071 [0.0020, 0.0145] | 0.0035 [0.0011, 0.0063] | 0.0233 [0.0000, 0.1364] |
| gru_6001 | 0.0069 [0.0013, 0.0183] | 0.0035 [0.0006, 0.0065] | 0.0930 [0.0000, 0.2729] |
| gru_6002 | 0.0127 [0.0026, 0.0340] | 0.0035 [0.0011, 0.0068] | 0.1977 [0.0000, 0.3929] |
| gru_ensemble | 0.0090 [0.0022, 0.0220] | 0.0035 [0.0007, 0.0065] | 0.1279 [0.0000, 0.2857] |

GRU seed min–max PR-AUC: (0.006941448694396045, 0.012749281923890753).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0029 [-0.0007, 0.0128] |
| lightgbm | 0.0000 [-0.0023, 0.0045] |
| trivial_reference | 0.0021 [-0.0012, 0.0114] |

Training: 6000: epochs=33, best_validation_loss=0.024081284; 6001: epochs=12, best_validation_loss=0.021069163; 6002: epochs=40, best_validation_loss=0.023108064.

### beta=0.5/rho=0.25 / default

Test base rate: 0.051265; positive labels: 1256.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0793 [0.0691, 0.0915] | 0.0484 [0.0428, 0.0546] | 0.1146 [0.0964, 0.1346] |
| logistic_regression | 0.2605 [0.2260, 0.2948] | 0.0429 [0.0385, 0.0482] | 0.3662 [0.3301, 0.3915] |
| lightgbm | 0.2526 [0.2141, 0.2974] | 0.0432 [0.0380, 0.0481] | 0.3439 [0.3152, 0.3807] |
| replicator | 0.0640 [0.0537, 0.0750] | 0.0485 [0.0421, 0.0548] | 0.0852 [0.0668, 0.1006] |
| gru_6000 | 0.2876 [0.2434, 0.3324] | 0.0419 [0.0377, 0.0471] | 0.3901 [0.3498, 0.4233] |
| gru_6001 | 0.2826 [0.2414, 0.3265] | 0.0421 [0.0376, 0.0472] | 0.3798 [0.3439, 0.4144] |
| gru_6002 | 0.2822 [0.2403, 0.3299] | 0.0420 [0.0379, 0.0467] | 0.3973 [0.3558, 0.4239] |
| gru_ensemble | 0.2886 [0.2476, 0.3386] | 0.0419 [0.0371, 0.0466] | 0.3933 [0.3565, 0.4266] |

GRU seed min–max PR-AUC: (0.28215783115760745, 0.2876274166707641).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.2246 [0.1873, 0.2655] |
| lightgbm | 0.0360 [0.0232, 0.0537] |
| trivial_reference | 0.2093 [0.1701, 0.2502] |

Training: 6000: epochs=24, best_validation_loss=0.15338427; 6001: epochs=30, best_validation_loss=0.13239369; 6002: epochs=40, best_validation_loss=0.13263409.

### beta=0.5/rho=0.25 / forced_flow

Test base rate: 0.000000; positive labels: 0.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| logistic_regression | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| lightgbm | N/A | 0.0011 [0.0007, 0.0014] | N/A |
| replicator | N/A | 0.0000 [0.0000, 0.0001] | N/A |
| gru_6000 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6001 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6002 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_ensemble | N/A | 0.0000 [0.0000, 0.0000] | N/A |

GRU seed min–max PR-AUC: (nan, nan).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | N/A |
| lightgbm | N/A |
| trivial_reference | N/A |

Training: 6000: epochs=31, best_validation_loss=0.00041322017; 6001: epochs=24, best_validation_loss=0.00050531188; 6002: epochs=28, best_validation_loss=0.00050994185.

### beta=0.5/rho=0.5 / default

Test base rate: 0.051388; positive labels: 1259.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0873 [0.0794, 0.0971] | 0.0482 [0.0444, 0.0522] | 0.1263 [0.1082, 0.1473] |
| logistic_regression | 0.3058 [0.2819, 0.3351] | 0.0414 [0.0388, 0.0442] | 0.4122 [0.3851, 0.4342] |
| lightgbm | 0.2642 [0.2404, 0.2883] | 0.0428 [0.0398, 0.0456] | 0.3757 [0.3503, 0.4010] |
| replicator | 0.0635 [0.0575, 0.0718] | 0.0486 [0.0451, 0.0524] | 0.0834 [0.0705, 0.0995] |
| gru_6000 | 0.3051 [0.2796, 0.3299] | 0.0413 [0.0386, 0.0441] | 0.4138 [0.3875, 0.4396] |
| gru_6001 | 0.3118 [0.2837, 0.3384] | 0.0411 [0.0385, 0.0438] | 0.4178 [0.3926, 0.4427] |
| gru_6002 | 0.3067 [0.2808, 0.3341] | 0.0412 [0.0387, 0.0440] | 0.4178 [0.3852, 0.4423] |
| gru_ensemble | 0.3105 [0.2841, 0.3383] | 0.0411 [0.0384, 0.0442] | 0.4194 [0.3910, 0.4434] |

GRU seed min–max PR-AUC: (0.3051401154355016, 0.31178576041100015).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.2471 [0.2236, 0.2701] |
| lightgbm | 0.0464 [0.0338, 0.0590] |
| trivial_reference | 0.2232 [0.2026, 0.2441] |

Training: 6000: epochs=26, best_validation_loss=0.15799524; 6001: epochs=25, best_validation_loss=0.14355573; 6002: epochs=31, best_validation_loss=0.1476181.

### beta=0.5/rho=0.5 / forced_flow

Test base rate: 0.000000; positive labels: 0.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| logistic_regression | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| lightgbm | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| replicator | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6000 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6001 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6002 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_ensemble | N/A | 0.0000 [0.0000, 0.0000] | N/A |

GRU seed min–max PR-AUC: (nan, nan).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | N/A |
| lightgbm | N/A |
| trivial_reference | N/A |

Training: 6000: epochs=40, best_validation_loss=4.1759695e-05; 6001: epochs=40, best_validation_loss=3.7092189e-05; 6002: epochs=40, best_validation_loss=5.7848668e-05.

### beta=1/noise / default

Test base rate: 0.050000; positive labels: 1225.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0726 [0.0640, 0.0826] | 0.0472 [0.0425, 0.0525] | 0.1127 [0.0947, 0.1318] |
| logistic_regression | 0.1728 [0.1512, 0.1962] | 0.0445 [0.0404, 0.0489] | 0.2604 [0.2295, 0.3006] |
| lightgbm | 0.1642 [0.1421, 0.1877] | 0.0446 [0.0405, 0.0487] | 0.2588 [0.2298, 0.2845] |
| replicator | 0.0558 [0.0470, 0.0674] | 0.0475 [0.0432, 0.0520] | 0.0710 [0.0523, 0.0905] |
| gru_6000 | 0.1947 [0.1672, 0.2242] | 0.0439 [0.0400, 0.0479] | 0.2922 [0.2558, 0.3304] |
| gru_6001 | 0.1877 [0.1591, 0.2145] | 0.0440 [0.0401, 0.0484] | 0.2882 [0.2529, 0.3237] |
| gru_6002 | 0.1944 [0.1672, 0.2231] | 0.0438 [0.0399, 0.0481] | 0.2980 [0.2598, 0.3339] |
| gru_ensemble | 0.1976 [0.1684, 0.2272] | 0.0437 [0.0397, 0.0477] | 0.3012 [0.2620, 0.3366] |

GRU seed min–max PR-AUC: (0.18767110141687318, 0.1946652058561175).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.1418 [0.1192, 0.1663] |
| lightgbm | 0.0334 [0.0212, 0.0465] |
| trivial_reference | 0.1250 [0.1033, 0.1490] |

Training: 6000: epochs=38, best_validation_loss=0.17420675; 6001: epochs=21, best_validation_loss=0.1665782; 6002: epochs=36, best_validation_loss=0.15724458.

### beta=1/noise / forced_flow

Test base rate: 0.001143; positive labels: 28.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0027 [0.0006, 0.0123] | 0.0011 [0.0000, 0.0029] | 0.2143 [0.0000, 0.4286] |
| logistic_regression | 0.0015 [0.0006, 0.0042] | 0.0011 [0.0000, 0.0029] | 0.0357 [0.0000, 0.0714] |
| lightgbm | 0.0020 [0.0004, 0.0086] | 0.0015 [0.0003, 0.0033] | 0.0714 [0.0000, 0.1429] |
| replicator | 0.0008 [0.0004, 0.0021] | 0.0012 [0.0000, 0.0029] | 0.0000 [0.0000, 0.0000] |
| gru_6000 | 0.0023 [0.0006, 0.0085] | 0.0011 [0.0000, 0.0029] | 0.1071 [0.0000, 0.3571] |
| gru_6001 | 0.0027 [0.0006, 0.0112] | 0.0011 [0.0000, 0.0029] | 0.1429 [0.0000, 0.2857] |
| gru_6002 | 0.0017 [0.0004, 0.0067] | 0.0011 [0.0000, 0.0029] | 0.1429 [0.0000, 0.2857] |
| gru_ensemble | 0.0021 [0.0005, 0.0079] | 0.0011 [0.0000, 0.0029] | 0.1071 [0.0000, 0.3571] |

GRU seed min–max PR-AUC: (0.0017363254778680287, 0.0026605873993214616).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0013 [0.0001, 0.0066] |
| lightgbm | 0.0001 [-0.0019, 0.0017] |
| trivial_reference | -0.0006 [-0.0043, -0.0000] |

Training: 6000: epochs=31, best_validation_loss=0.00039781619; 6001: epochs=29, best_validation_loss=0.00042109093; 6002: epochs=40, best_validation_loss=0.0003816538.

### beta=1/rho=0.25 / default

Test base rate: 0.054776; positive labels: 1342.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0858 [0.0754, 0.0989] | 0.0514 [0.0470, 0.0561] | 0.1066 [0.0900, 0.1239] |
| logistic_regression | 0.2608 [0.2332, 0.2884] | 0.0459 [0.0424, 0.0496] | 0.3495 [0.3208, 0.3778] |
| lightgbm | 0.2251 [0.2002, 0.2514] | 0.0470 [0.0433, 0.0507] | 0.3070 [0.2772, 0.3376] |
| replicator | 0.0815 [0.0726, 0.0917] | 0.0514 [0.0466, 0.0562] | 0.1021 [0.0850, 0.1195] |
| gru_6000 | 0.2632 [0.2333, 0.2935] | 0.0456 [0.0418, 0.0496] | 0.3539 [0.3219, 0.3844] |
| gru_6001 | 0.2664 [0.2388, 0.2958] | 0.0455 [0.0419, 0.0494] | 0.3569 [0.3260, 0.3865] |
| gru_6002 | 0.2642 [0.2360, 0.2919] | 0.0456 [0.0421, 0.0494] | 0.3614 [0.3271, 0.3915] |
| gru_ensemble | 0.2665 [0.2386, 0.2953] | 0.0455 [0.0420, 0.0494] | 0.3592 [0.3316, 0.3858] |

GRU seed min–max PR-AUC: (0.2632384525155201, 0.2664335011095742).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.1850 [0.1593, 0.2076] |
| lightgbm | 0.0414 [0.0256, 0.0569] |
| trivial_reference | 0.1807 [0.1569, 0.2038] |

Training: 6000: epochs=17, best_validation_loss=0.16242913; 6001: epochs=14, best_validation_loss=0.15247041; 6002: epochs=19, best_validation_loss=0.15707097.

### beta=1/rho=0.25 / forced_flow

Test base rate: 0.000000; positive labels: 0.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| logistic_regression | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| lightgbm | N/A | 0.0033 [0.0026, 0.0041] | N/A |
| replicator | N/A | 0.0002 [0.0000, 0.0003] | N/A |
| gru_6000 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6001 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_6002 | N/A | 0.0000 [0.0000, 0.0000] | N/A |
| gru_ensemble | N/A | 0.0000 [0.0000, 0.0000] | N/A |

GRU seed min–max PR-AUC: (nan, nan).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | N/A |
| lightgbm | N/A |
| trivial_reference | N/A |

Training: 6000: epochs=17, best_validation_loss=0.0058927202; 6001: epochs=16, best_validation_loss=0.0058912992; 6002: epochs=18, best_validation_loss=0.0059000215.

### beta=1/rho=0.5 / default

Test base rate: 0.054898; positive labels: 1345.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0738 [0.0655, 0.0844] | 0.0516 [0.0474, 0.0563] | 0.0870 [0.0715, 0.1049] |
| logistic_regression | 0.2819 [0.2559, 0.3105] | 0.0450 [0.0414, 0.0490] | 0.3844 [0.3495, 0.4154] |
| lightgbm | 0.2337 [0.2130, 0.2555] | 0.0468 [0.0430, 0.0507] | 0.3264 [0.2955, 0.3544] |
| replicator | 0.1083 [0.0946, 0.1234] | 0.0509 [0.0469, 0.0553] | 0.1502 [0.1268, 0.1749] |
| gru_6000 | 0.3111 [0.2788, 0.3435] | 0.0439 [0.0403, 0.0475] | 0.4126 [0.3744, 0.4527] |
| gru_6001 | 0.3123 [0.2797, 0.3460] | 0.0440 [0.0405, 0.0474] | 0.4059 [0.3698, 0.4433] |
| gru_6002 | 0.3120 [0.2801, 0.3476] | 0.0438 [0.0401, 0.0472] | 0.4186 [0.3767, 0.4575] |
| gru_ensemble | 0.3158 [0.2830, 0.3518] | 0.0437 [0.0406, 0.0473] | 0.4171 [0.3780, 0.4532] |

GRU seed min–max PR-AUC: (0.31112189070032115, 0.31231534228082725).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.2076 [0.1790, 0.2354] |
| lightgbm | 0.0821 [0.0596, 0.1070] |
| trivial_reference | 0.2420 [0.2126, 0.2756] |

Training: 6000: epochs=22, best_validation_loss=0.163361; 6001: epochs=24, best_validation_loss=0.1474866; 6002: epochs=25, best_validation_loss=0.15362556.

### beta=1/rho=0.5 / forced_flow

Test base rate: 0.000571; positive labels: 14.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0007 [0.0006, 0.0021] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| logistic_regression | 0.0004 [0.0004, 0.0012] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| lightgbm | 0.0006 [0.0005, 0.0022] | 0.0028 [0.0018, 0.0042] | 0.0000 [0.0000, 0.0000] |
| replicator | 0.0006 [0.0006, 0.0019] | 0.0015 [0.0007, 0.0029] | 0.0714 [0.0714, 0.0714] |
| gru_6000 | 0.0006 [0.0006, 0.0024] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| gru_6001 | 0.0008 [0.0007, 0.0029] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| gru_6002 | 0.0006 [0.0006, 0.0023] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |
| gru_ensemble | 0.0007 [0.0006, 0.0026] | 0.0006 [0.0000, 0.0017] | 0.0000 [0.0000, 0.0000] |

GRU seed min–max PR-AUC: (0.0006053983217498367, 0.0007915944824740115).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0000 [-0.0000, 0.0002] |
| lightgbm | 0.0001 [0.0001, 0.0004] |
| trivial_reference | 0.0000 [-0.0000, 0.0001] |

Training: 6000: epochs=40, best_validation_loss=0.00019079646; 6001: epochs=39, best_validation_loss=0.00019277023; 6002: epochs=40, best_validation_loss=0.00019037359.

### stress/rho=0.25/drawdown=0.05 / default

Test base rate: 0.051837; positive labels: 1270.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0913 [0.0684, 0.1151] | 0.0486 [0.0335, 0.0654] | 0.1283 [0.1015, 0.1648] |
| logistic_regression | 0.3671 [0.2691, 0.4648] | 0.0397 [0.0279, 0.0528] | 0.5142 [0.4141, 0.6120] |
| lightgbm | 0.5218 [0.4175, 0.6117] | 0.0348 [0.0248, 0.0452] | 0.6268 [0.5363, 0.7169] |
| replicator | 0.1319 [0.0988, 0.1748] | 0.0471 [0.0338, 0.0627] | 0.1772 [0.1326, 0.2360] |
| gru_6000 | 0.5539 [0.4606, 0.6381] | 0.0322 [0.0233, 0.0416] | 0.7008 [0.6286, 0.7626] |
| gru_6001 | 0.5795 [0.4830, 0.6594] | 0.0315 [0.0221, 0.0406] | 0.6913 [0.6316, 0.7515] |
| gru_6002 | 0.5627 [0.4706, 0.6360] | 0.0321 [0.0231, 0.0412] | 0.6858 [0.6281, 0.7439] |
| gru_ensemble | 0.5791 [0.4848, 0.6635] | 0.0313 [0.0227, 0.0399] | 0.7016 [0.6464, 0.7618] |

GRU seed min–max PR-AUC: (0.5538787666037479, 0.5794788145410931).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.4472 [0.3564, 0.5293] |
| lightgbm | 0.0573 [0.0335, 0.0844] |
| trivial_reference | 0.4877 [0.3947, 0.5682] |

Training: 6000: epochs=40, best_validation_loss=0.12111501; 6001: epochs=40, best_validation_loss=0.10944204; 6002: epochs=40, best_validation_loss=0.10993145.

### stress/rho=0.25/drawdown=0.05 / forced_flow

Test base rate: 0.062204; positive labels: 1524.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0951 [0.0733, 0.1213] | 0.0578 [0.0483, 0.0682] | 0.0958 [0.0627, 0.1292] |
| logistic_regression | 0.1089 [0.0848, 0.1374] | 0.0574 [0.0478, 0.0670] | 0.1227 [0.0771, 0.1666] |
| lightgbm | 0.1156 [0.0917, 0.1464] | 0.0570 [0.0474, 0.0671] | 0.1247 [0.0909, 0.1558] |
| replicator | 0.0793 [0.0636, 0.0981] | 0.0581 [0.0486, 0.0683] | 0.0682 [0.0501, 0.0867] |
| gru_6000 | 0.1460 [0.1114, 0.1854] | 0.0563 [0.0478, 0.0654] | 0.1844 [0.1374, 0.2308] |
| gru_6001 | 0.1389 [0.1070, 0.1737] | 0.0564 [0.0478, 0.0660] | 0.1667 [0.1228, 0.2093] |
| gru_6002 | 0.1426 [0.1072, 0.1804] | 0.0562 [0.0478, 0.0660] | 0.1778 [0.1313, 0.2211] |
| gru_ensemble | 0.1444 [0.1114, 0.1842] | 0.0562 [0.0471, 0.0659] | 0.1765 [0.1301, 0.2214] |

GRU seed min–max PR-AUC: (0.1388816511142375, 0.14597558945452593).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0651 [0.0409, 0.0976] |
| lightgbm | 0.0288 [0.0174, 0.0455] |
| trivial_reference | 0.0493 [0.0257, 0.0793] |

Training: 6000: epochs=36, best_validation_loss=0.17853544; 6001: epochs=40, best_validation_loss=0.19013198; 6002: epochs=40, best_validation_loss=0.1788632.

### stress/rho=0.25/drawdown=0.1 / default

Test base rate: 0.051551; positive labels: 1263.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0884 [0.0667, 0.1091] | 0.0484 [0.0321, 0.0645] | 0.1322 [0.1028, 0.1647] |
| logistic_regression | 0.3875 [0.2923, 0.4797] | 0.0391 [0.0272, 0.0517] | 0.5123 [0.4342, 0.6107] |
| lightgbm | 0.5127 [0.4103, 0.5945] | 0.0351 [0.0252, 0.0459] | 0.6152 [0.5428, 0.6942] |
| replicator | 0.1326 [0.1023, 0.1662] | 0.0467 [0.0316, 0.0614] | 0.1892 [0.1443, 0.2356] |
| gru_6000 | 0.5476 [0.4558, 0.6152] | 0.0324 [0.0239, 0.0429] | 0.6587 [0.5917, 0.7279] |
| gru_6001 | 0.5586 [0.4603, 0.6327] | 0.0322 [0.0230, 0.0417] | 0.6778 [0.5987, 0.7334] |
| gru_6002 | 0.5500 [0.4660, 0.6166] | 0.0325 [0.0236, 0.0425] | 0.6516 [0.5925, 0.7259] |
| gru_ensemble | 0.5651 [0.4731, 0.6369] | 0.0318 [0.0229, 0.0406] | 0.6778 [0.6052, 0.7380] |

GRU seed min–max PR-AUC: (0.5476403583603282, 0.5586469878605027).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.4325 [0.3604, 0.5032] |
| lightgbm | 0.0524 [0.0236, 0.0852] |
| trivial_reference | 0.4767 [0.3979, 0.5476] |

Training: 6000: epochs=40, best_validation_loss=0.13099486; 6001: epochs=40, best_validation_loss=0.1095543; 6002: epochs=40, best_validation_loss=0.11119787.

### stress/rho=0.25/drawdown=0.1 / forced_flow

Test base rate: 0.044939; positive labels: 1101.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0755 [0.0534, 0.1031] | 0.0425 [0.0321, 0.0529] | 0.0899 [0.0414, 0.1501] |
| logistic_regression | 0.0806 [0.0571, 0.1152] | 0.0423 [0.0328, 0.0537] | 0.1099 [0.0575, 0.1713] |
| lightgbm | 0.0836 [0.0588, 0.1134] | 0.0421 [0.0323, 0.0521] | 0.1262 [0.0734, 0.1887] |
| replicator | 0.0702 [0.0513, 0.0888] | 0.0425 [0.0331, 0.0534] | 0.0899 [0.0729, 0.1174] |
| gru_6000 | 0.1198 [0.0847, 0.1621] | 0.0414 [0.0317, 0.0523] | 0.2189 [0.1504, 0.2939] |
| gru_6001 | 0.1047 [0.0759, 0.1461] | 0.0416 [0.0324, 0.0522] | 0.1771 [0.1088, 0.2608] |
| gru_6002 | 0.1160 [0.0815, 0.1570] | 0.0414 [0.0320, 0.0521] | 0.1944 [0.1270, 0.2738] |
| gru_ensemble | 0.1154 [0.0821, 0.1556] | 0.0414 [0.0308, 0.0519] | 0.1989 [0.1347, 0.2923] |

GRU seed min–max PR-AUC: (0.10467101727439769, 0.11976258611228044).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0452 [0.0233, 0.0774] |
| lightgbm | 0.0319 [0.0105, 0.0602] |
| trivial_reference | 0.0399 [0.0178, 0.0688] |

Training: 6000: epochs=34, best_validation_loss=0.17020257; 6001: epochs=32, best_validation_loss=0.18618262; 6002: epochs=30, best_validation_loss=0.18390888.

### stress/rho=0.5/drawdown=0.05 / default

Test base rate: 0.041102; positive labels: 1007.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0705 [0.0574, 0.0845] | 0.0392 [0.0319, 0.0474] | 0.1370 [0.1139, 0.1649] |
| logistic_regression | 0.2611 [0.2243, 0.2947] | 0.0346 [0.0287, 0.0414] | 0.4101 [0.3607, 0.4495] |
| lightgbm | 0.2661 [0.2268, 0.3038] | 0.0342 [0.0287, 0.0413] | 0.4111 [0.3617, 0.4521] |
| replicator | 0.0702 [0.0573, 0.0844] | 0.0390 [0.0317, 0.0471] | 0.1122 [0.0879, 0.1325] |
| gru_6000 | 0.2942 [0.2520, 0.3314] | 0.0335 [0.0275, 0.0398] | 0.4638 [0.4107, 0.5148] |
| gru_6001 | 0.2898 [0.2517, 0.3325] | 0.0336 [0.0281, 0.0398] | 0.4568 [0.4110, 0.5104] |
| gru_6002 | 0.2945 [0.2526, 0.3336] | 0.0333 [0.0270, 0.0398] | 0.4558 [0.4115, 0.5109] |
| gru_ensemble | 0.2980 [0.2566, 0.3388] | 0.0333 [0.0273, 0.0393] | 0.4767 [0.4243, 0.5237] |

GRU seed min–max PR-AUC: (0.28984247862146134, 0.2945270807104472).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.2279 [0.1938, 0.2585] |
| lightgbm | 0.0319 [0.0128, 0.0512] |
| trivial_reference | 0.2275 [0.1945, 0.2583] |

Training: 6000: epochs=40, best_validation_loss=0.14417958; 6001: epochs=40, best_validation_loss=0.13296827; 6002: epochs=40, best_validation_loss=0.13079238.

### stress/rho=0.5/drawdown=0.05 / forced_flow

Test base rate: 0.059143; positive labels: 1449.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0818 [0.0615, 0.1052] | 0.0553 [0.0430, 0.0682] | 0.0814 [0.0506, 0.1143] |
| logistic_regression | 0.0881 [0.0662, 0.1133] | 0.0551 [0.0436, 0.0672] | 0.0897 [0.0605, 0.1321] |
| lightgbm | 0.1218 [0.0899, 0.1551] | 0.0540 [0.0425, 0.0664] | 0.1587 [0.1198, 0.2071] |
| replicator | 0.0897 [0.0686, 0.1101] | 0.0549 [0.0422, 0.0675] | 0.0870 [0.0651, 0.1106] |
| gru_6000 | 0.1690 [0.1161, 0.2320] | 0.0528 [0.0421, 0.0653] | 0.2146 [0.1551, 0.2956] |
| gru_6001 | 0.1800 [0.1241, 0.2477] | 0.0528 [0.0419, 0.0649] | 0.2409 [0.1671, 0.3111] |
| gru_6002 | 0.1791 [0.1221, 0.2411] | 0.0525 [0.0414, 0.0628] | 0.2215 [0.1558, 0.3015] |
| gru_ensemble | 0.1809 [0.1252, 0.2471] | 0.0525 [0.0410, 0.0639] | 0.2284 [0.1618, 0.3069] |

GRU seed min–max PR-AUC: (0.1690409401072409, 0.18002328991447566).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0912 [0.0434, 0.1525] |
| lightgbm | 0.0591 [0.0235, 0.1074] |
| trivial_reference | 0.0991 [0.0546, 0.1577] |

Training: 6000: epochs=25, best_validation_loss=0.17932282; 6001: epochs=35, best_validation_loss=0.18550942; 6002: epochs=37, best_validation_loss=0.16122905.

### stress/rho=0.5/drawdown=0.1 / default

Test base rate: 0.042204; positive labels: 1034.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0748 [0.0623, 0.0888] | 0.0401 [0.0331, 0.0482] | 0.1480 [0.1225, 0.1717] |
| logistic_regression | 0.2712 [0.2308, 0.3036] | 0.0352 [0.0290, 0.0421] | 0.4197 [0.3748, 0.4616] |
| lightgbm | 0.2748 [0.2299, 0.3153] | 0.0350 [0.0283, 0.0414] | 0.4255 [0.3757, 0.4652] |
| replicator | 0.0715 [0.0585, 0.0853] | 0.0400 [0.0328, 0.0481] | 0.1112 [0.0951, 0.1306] |
| gru_6000 | 0.2972 [0.2469, 0.3377] | 0.0343 [0.0283, 0.0409] | 0.4516 [0.3976, 0.4995] |
| gru_6001 | 0.2987 [0.2508, 0.3404] | 0.0342 [0.0285, 0.0399] | 0.4516 [0.4032, 0.5094] |
| gru_6002 | 0.2947 [0.2489, 0.3373] | 0.0343 [0.0284, 0.0406] | 0.4584 [0.4021, 0.5059] |
| gru_ensemble | 0.3017 [0.2514, 0.3453] | 0.0341 [0.0282, 0.0405] | 0.4555 [0.4118, 0.5153] |

GRU seed min–max PR-AUC: (0.29467296516043384, 0.2987444663929893).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.2302 [0.1913, 0.2647] |
| lightgbm | 0.0269 [0.0142, 0.0406] |
| trivial_reference | 0.2269 [0.1883, 0.2601] |

Training: 6000: epochs=40, best_validation_loss=0.14919473; 6001: epochs=34, best_validation_loss=0.13542278; 6002: epochs=40, best_validation_loss=0.13089663.

### stress/rho=0.5/drawdown=0.1 / forced_flow

Test base rate: 0.016245; positive labels: 398.

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| trivial_reference | 0.0197 [0.0127, 0.0283] | 0.0160 [0.0097, 0.0230] | 0.0352 [0.0000, 0.0866] |
| logistic_regression | 0.0237 [0.0145, 0.0357] | 0.0159 [0.0102, 0.0226] | 0.0729 [0.0287, 0.1345] |
| lightgbm | 0.0315 [0.0187, 0.0468] | 0.0158 [0.0093, 0.0226] | 0.0930 [0.0603, 0.1488] |
| replicator | 0.0281 [0.0168, 0.0415] | 0.0159 [0.0093, 0.0226] | 0.1030 [0.0677, 0.1328] |
| gru_6000 | 0.0921 [0.0316, 0.1941] | 0.0156 [0.0097, 0.0222] | 0.1910 [0.1015, 0.3247] |
| gru_6001 | 0.0485 [0.0265, 0.0879] | 0.0158 [0.0100, 0.0220] | 0.1357 [0.0528, 0.2566] |
| gru_6002 | 0.0844 [0.0321, 0.1736] | 0.0158 [0.0097, 0.0223] | 0.1809 [0.0892, 0.3070] |
| gru_ensemble | 0.0887 [0.0314, 0.1843] | 0.0156 [0.0097, 0.0223] | 0.1809 [0.0800, 0.3045] |

GRU seed min–max PR-AUC: (0.048454440307025747, 0.09213543748592404).

| GRU ensemble minus | Paired PR-AUC difference (95% CI) |
| --- | ---: |
| replicator | 0.0606 [0.0059, 0.1608] |
| lightgbm | 0.0572 [0.0027, 0.1688] |
| trivial_reference | 0.0690 [0.0127, 0.1710] |

Training: 6000: epochs=27, best_validation_loss=0.064464612; 6001: epochs=15, best_validation_loss=0.083777626; 6002: epochs=35, best_validation_loss=0.081095735.

## Joint gain contrasts

Gain = GRU ensemble PR-AUC minus trivial PR-AUC. Noise effect = gain(off) − gain(noise). Information effect = gain(signal) − gain(noise). The primary interpretation uses default labels; secondary forced-flow results retain power warnings.

| Label | Contrast | Gain difference (95% CI) | Test base rates (first / second) | Outcome |
| --- | --- | ---: | ---: | --- |
| default | noise beta=0.1 | 0.2064 [0.0849, 0.3691] | 0.049959 / 0.038408 | noise degrades inference |
| default | information beta=0.1, rho=0.25 | -0.0650 [-0.2108, 0.0825] | 0.047796 / 0.038408 | no directional information effect established |
| default | information beta=0.1, rho=0.5 | -0.1930 [-0.3233, -0.0798] | 0.041837 / 0.038408 | private information makes inference harder |
| default | noise beta=0.25 | 0.2595 [0.1560, 0.4105] | 0.049959 / 0.049061 | noise degrades inference |
| default | information beta=0.25, rho=0.25 | -0.1223 [-0.2427, 0.0186] | 0.051184 / 0.049061 | no directional information effect established |
| default | information beta=0.25, rho=0.5 | -0.3778 [-0.4856, -0.2160] | 0.042571 / 0.049061 | private information makes inference harder |
| default | noise beta=0.5 | 0.4426 [0.3680, 0.5419] | 0.049959 / 0.052449 | noise degrades inference |
| default | information beta=0.5, rho=0.25 | -0.2147 [-0.3091, -0.1133] | 0.051265 / 0.052449 | private information makes inference harder |
| default | information beta=0.5, rho=0.5 | -0.2008 [-0.2769, -0.1004] | 0.051388 / 0.052449 | private information makes inference harder |
| default | noise beta=1 | 0.7416 [0.6927, 0.7770] | 0.049959 / 0.050000 | noise degrades inference |
| default | information beta=1, rho=0.25 | 0.0557 [0.0254, 0.0874] | 0.054776 / 0.050000 | private information makes inference easier |
| default | information beta=1, rho=0.5 | 0.1170 [0.0797, 0.1547] | 0.054898 / 0.050000 | private information makes inference easier |
| forced_flow | noise beta=0.1 | 0.0463 [0.0051, 0.0874] | 0.062000 / 0.052694 | noise degrades inference |
| forced_flow | information beta=0.1, rho=0.25 | 0.0440 [0.0052, 0.0904] | 0.064571 / 0.052694 | private information makes inference easier |
| forced_flow | information beta=0.1, rho=0.5 | 0.0413 [0.0051, 0.0857] | 0.052980 / 0.052694 | private information makes inference easier |
| forced_flow | noise beta=0.25 | 0.0402 [0.0015, 0.0840] | 0.062000 / 0.044776 | noise degrades inference |
| forced_flow | information beta=0.25, rho=0.25 | 0.0245 [-0.0059, 0.0644] | 0.019224 / 0.044776 | no directional information effect established |
| forced_flow | information beta=0.25, rho=0.5 | 0.0279 [-0.0217, 0.1509] | 0.002735 / 0.044776 | uninformative |
| forced_flow | noise beta=0.5 | 0.0472 [0.0130, 0.0814] | 0.062000 / 0.003510 | uninformative |
| forced_flow | information beta=0.5, rho=0.25 | N/A | 0.000000 / 0.003510 | underpowered; uninformative |
| forced_flow | information beta=0.5, rho=0.5 | N/A | 0.000000 / 0.003510 | underpowered; uninformative |
| forced_flow | noise beta=1 | 0.0500 [0.0184, 0.0855] | 0.062000 / 0.001143 | underpowered; uninformative |
| forced_flow | information beta=1, rho=0.25 | N/A | 0.000000 / 0.001143 | underpowered; uninformative |
| forced_flow | information beta=1, rho=0.5 | 0.0006 [0.0000, 0.0036] | 0.000571 / 0.001143 | underpowered; uninformative |

## Pre-registered rule outcomes (verbatim)

- Replication / informativeness / power — off/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — off/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.1/noise/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.1/noise/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.1/rho=0.25/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.1/rho=0.25/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.1/rho=0.5/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.1/rho=0.5/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/noise/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/noise/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/rho=0.25/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/rho=0.25/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/rho=0.5/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.25/rho=0.5/forced_flow: uninformative.
- Replication / informativeness / power — beta=0.5/noise/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.5/noise/forced_flow: uninformative.
- Replication / informativeness / power — beta=0.5/rho=0.25/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.5/rho=0.25/forced_flow: uninformative; underpowered.
- Replication / informativeness / power — beta=0.5/rho=0.5/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=0.5/rho=0.5/forced_flow: uninformative; underpowered.
- Replication / informativeness / power — beta=1/noise/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=1/noise/forced_flow: uninformative; underpowered.
- Replication / informativeness / power — beta=1/rho=0.25/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=1/rho=0.25/forced_flow: uninformative; underpowered.
- Replication / informativeness / power — beta=1/rho=0.5/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — beta=1/rho=0.5/forced_flow: uninformative; underpowered.
- Replication / informativeness / power — stress/rho=0.25/drawdown=0.05/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — stress/rho=0.25/drawdown=0.05/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — stress/rho=0.25/drawdown=0.1/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — stress/rho=0.25/drawdown=0.1/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — stress/rho=0.5/drawdown=0.05/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — stress/rho=0.5/drawdown=0.05/forced_flow: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — stress/rho=0.5/drawdown=0.1/default: not consistent with strategy replication by the registered rule.
- Replication / informativeness / power — stress/rho=0.5/drawdown=0.1/forced_flow: not consistent with strategy replication by the registered rule.
- default / noise beta=0.1: noise degrades inference.
- default / information beta=0.1, rho=0.25: no directional information effect established.
- default / information beta=0.1, rho=0.5: private information makes inference harder.
- default / noise beta=0.25: noise degrades inference.
- default / information beta=0.25, rho=0.25: no directional information effect established.
- default / information beta=0.25, rho=0.5: private information makes inference harder.
- default / noise beta=0.5: noise degrades inference.
- default / information beta=0.5, rho=0.25: private information makes inference harder.
- default / information beta=0.5, rho=0.5: private information makes inference harder.
- default / noise beta=1: noise degrades inference.
- default / information beta=1, rho=0.25: private information makes inference easier.
- default / information beta=1, rho=0.5: private information makes inference easier.
- Round-4 private information degrades inference reading: not supported.
- Swamping — none.
- forced_flow / noise beta=0.1: noise degrades inference.
- forced_flow / information beta=0.1, rho=0.25: private information makes inference easier.
- forced_flow / information beta=0.1, rho=0.5: private information makes inference easier.
- forced_flow / noise beta=0.25: noise degrades inference.
- forced_flow / information beta=0.25, rho=0.25: no directional information effect established.
- forced_flow / information beta=0.25, rho=0.5: uninformative.
- forced_flow / noise beta=0.5: uninformative.
- forced_flow / information beta=0.5, rho=0.25: underpowered; uninformative.
- forced_flow / information beta=0.5, rho=0.5: underpowered; uninformative.
- forced_flow / noise beta=1: underpowered; uninformative.
- forced_flow / information beta=1, rho=0.25: underpowered; uninformative.
- forced_flow / information beta=1, rho=0.5: underpowered; uninformative.

## Scope and open questions

No simulator equation, label definition, threshold rule, scientific setting, observer hyperparameter, or registered seed count was changed. Observers receive public records and targets only. Positive-label counts include overlapping future windows; uncertainty resamples episodes, not windows. Undefined PR-AUC is N/A, not zero. Informativeness (<1%) and forced-flow power (<30 positives) flags limit interpretation. The forced-selling H1 rule belongs to R5-4 and is not evaluated here.

Execution deviation: interrupted and duplicate attempts are documented above; no research-design changes were made. Open question for the lead: how should duplicate/interrupted execution be charged against the combined runtime allowance? This experiment cannot establish real-market validity or identify hidden real fund positions.

## Validation

`.venv/bin/pytest -q`: `148 passed in 14.50s`.

Final cache-resume verification: all 17 source/configuration fingerprints matched; regenerated configuration tables, metrics, CIs, scale ratios, and verbatim rule outcomes matched exactly (`runs/round5_verify.log`, `runs/round5_verified_report.md`).
