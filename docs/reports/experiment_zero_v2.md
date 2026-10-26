# Experiment Zero v2

This report contains synthetic output from the uncalibrated HerdLab simulator. It is not evidence about real market behavior.

## Configuration

- Selected config: `configs/experiment_zero.yaml`; impact sensitivity `0.03`, q_max `200`, order limit `40`.
- Step-7 splits: `configs/heldout_families.yaml` and `configs/heldout_impact.yaml`. Training and in-distribution use the family split's familiar population (momentum/linear/MLP, 10 each); held-out family uses reversion/GBDT (15 each); held-out impact applies only the concave-impact override to the familiar population.
- The selected impact sensitivity, q_max, and order limit are identical in every split.
- Episodes: 200 training, 50 in-distribution, 50 held-out family, 50 held-out impact; 500 steps each; 10 generation workers.
- Seeds: training `1000-1199`; in-distribution `2000-2049`; held-out family `3000-3049`; held-out impact `4000-4049`.
- Labels: default future-window and onset; horizon 10. Both use the training-only score quantile 0.95, fitted threshold `0.1591791367`.
- Observers: trivial standardized logistic on 10 causal lags of absolute public OFI plus realized volatility; full-feature logistic; LightGBM (seed 5000); small GRU with training-episode validation (seeds 6000 and 6001).
- Metrics: PR-AUC with event base rate, Brier score, recall at FAR <= 0.05, and fraction of labeled events whose future window contains nonzero forced flow.
- Uncertainty: 95% percentile intervals from 1000 whole-episode bootstrap resamples; seed family begins at 7000.
- Measured wall runtime: 6.55 minutes.

## Event rates and forced-flow overlap

Forced-flow overlap uses private ground truth for diagnostics only and is not an observer input. A positive event overlaps forced flow when its strictly future horizon contains a nonzero forced order.

| Label | Split | Events / labels | Base rate | Events with forced flow | Fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | Training | 4900 / 98000 | 0.050000 | 485 / 4900 | 0.098980 |
| Default | In-distribution | 1441 / 24500 | 0.058816 | 167 / 1441 | 0.115892 |
| Default | Held-out family | 0 / 24500 | 0.000000 | 0 / 0 | 0.000000 |
| Default | Held-out impact | 2632 / 24500 | 0.107429 | 813 / 2632 | 0.308891 |
| Onset | Training | 875 / 98000 | 0.008929 | 101 / 875 | 0.115429 |
| Onset | In-distribution | 207 / 24500 | 0.008449 | 24 / 207 | 0.115942 |
| Onset | Held-out family | 0 / 24500 | 0.000000 | 0 / 0 | 0.000000 |
| Onset | Held-out impact | 295 / 24500 | 0.012041 | 81 / 295 | 0.274576 |

## Observer metrics

Values in brackets are episode-bootstrap 95% CIs. Higher is better for PR-AUC and recall; lower is better for Brier score.

### Default label

#### In-distribution

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| Trivial Reference | 0.1094 [0.0959, 0.1228] | 0.0545 [0.0466, 0.0626] | 0.1513 [0.1284, 0.1757] |
| Logistic Regression | 0.4512 [0.4069, 0.4908] | 0.0410 [0.0358, 0.0462] | 0.5607 [0.5272, 0.5944] |
| Lightgbm | 0.7138 [0.6659, 0.7510] | 0.0282 [0.0249, 0.0315] | 0.8473 [0.8096, 0.8770] |
| Gru | 0.7964 [0.7571, 0.8297] | 0.0229 [0.0204, 0.0253] | 0.9077 [0.8859, 0.9327] |

#### Held-out family

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| Trivial Reference | N/A | 0.0030 [0.0029, 0.0031] | N/A |
| Logistic Regression | N/A | 0.0010 [0.0007, 0.0014] | N/A |
| Lightgbm | N/A | 0.0007 [0.0006, 0.0009] | N/A |
| Gru | N/A | 0.0010 [0.0002, 0.0021] | N/A |

#### Held-out impact

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| Trivial Reference | 0.1830 [0.1598, 0.2093] | 0.0994 [0.0898, 0.1087] | 0.1326 [0.1135, 0.1526] |
| Logistic Regression | 0.4455 [0.4145, 0.4779] | 0.0756 [0.0690, 0.0818] | 0.4141 [0.3861, 0.4476] |
| Lightgbm | 0.6119 [0.5813, 0.6437] | 0.0580 [0.0531, 0.0629] | 0.6353 [0.5996, 0.6710] |
| Gru | 0.7598 [0.7380, 0.7798] | 0.0480 [0.0438, 0.0523] | 0.7633 [0.7309, 0.7955] |

Training detail: GRU epochs=40, best_validation_loss=0.076012301.

### Onset label

#### In-distribution

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| Trivial Reference | 0.0519 [0.0398, 0.0782] | 0.0082 [0.0074, 0.0090] | 0.2657 [0.2166, 0.3179] |
| Logistic Regression | 0.0908 [0.0653, 0.1309] | 0.0080 [0.0072, 0.0089] | 0.4010 [0.3418, 0.4646] |
| Lightgbm | 0.1863 [0.1493, 0.2352] | 0.0075 [0.0068, 0.0083] | 0.7246 [0.6804, 0.7952] |
| Gru | 0.2487 [0.2137, 0.2973] | 0.0071 [0.0064, 0.0080] | 0.7971 [0.7427, 0.8537] |

#### Held-out family

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| Trivial Reference | N/A | 0.0002 [0.0002, 0.0002] | N/A |
| Logistic Regression | N/A | 0.0001 [0.0001, 0.0002] | N/A |
| Lightgbm | N/A | 0.0001 [0.0000, 0.0002] | N/A |
| Gru | N/A | 0.0000 [0.0000, 0.0001] | N/A |

#### Held-out impact

| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |
| --- | ---: | ---: | ---: |
| Trivial Reference | 0.0359 [0.0297, 0.0497] | 0.0118 [0.0107, 0.0130] | 0.1424 [0.1064, 0.1753] |
| Logistic Regression | 0.0515 [0.0428, 0.0636] | 0.0128 [0.0117, 0.0140] | 0.1966 [0.1536, 0.2369] |
| Lightgbm | 0.0865 [0.0712, 0.1087] | 0.0119 [0.0108, 0.0130] | 0.3763 [0.3232, 0.4379] |
| Gru | 0.1214 [0.1043, 0.1456] | 0.0118 [0.0107, 0.0128] | 0.5458 [0.4832, 0.6073] |

Training detail: GRU epochs=40, best_validation_loss=0.028064075.

## Caveats

- Every number above comes from this simulator run; no public historical market data were used.
- The 95th-percentile threshold mechanically sets the pooled default-label training base rate near 5%; onset suppresses nearby qualifying windows and therefore has a lower base rate.
- Confidence intervals resample 50 complete evaluation episodes. Overlapping prediction windows are never resampled as independent observations.
- PR-AUC and recall are reported as N/A when a label/split has no positive events (or, for recall, lacks either class); Brier score remains defined and bootstrapped.
- The impact holdout changes the impact function from linear to concave while retaining the familiar agent population. The family holdout changes agent families while retaining population size and configured aggregate capacity.
- Forced-flow overlap is descriptive private ground truth, not a causal attribution and not an observer feature.
- No simulator parameter, threshold rule, or observer hyperparameter was changed after viewing these metrics.

## Deviations and open questions

No episode-count reduction or other run deviation. No open questions.
