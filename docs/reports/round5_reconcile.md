# Round 5: population and seed reconciliation

**Synthetic-simulator disclaimer:** These are results from the uncalibrated synthetic HerdLab simulator, not evidence about real markets, funds, or live trading.

## Design and runtime

Pre-run estimate for R5-2: 10 minutes, below the registered ~75-minute stop threshold. Measured runtime: 8.79 minutes. Later round-5 experiments are not included in this estimate or run.

Four independently fitted cells cross two populations and two seed sets. Each uses 200 training and 50 test episodes of 500 steps. Default label only: horizon 10, training-only 0.95 score quantile fitted separately per cell. Signals are explicitly off; learned overlap mode is none.

Observers retain historical defaults: public return lookback 10, LightGBM RNG seed 5000, GRU training seed 6000 (single model, not ensemble). PR-AUC uses 1,000 whole-test-episode percentile bootstrap resamples, 95% CIs; bootstrap seeds are 9000 + 100 × cell index + observer index in the order below. Base rates are descriptive. No overlapping time steps are independently resampled.

## 2×2 results

| Population | Round-3 seeds | Round-4 seeds |
| --- | --- | --- |
| round3_v2 | Train rate 0.050000; test rate 0.058816; LightGBM 0.7138 [0.6713, 0.7509]; GRU 6000 0.7964 [0.7581, 0.8272] | Train rate 0.050000; test rate 0.054449; LightGBM 0.6935 [0.6603, 0.7246]; GRU 6000 0.7619 [0.7186, 0.7961] |
| experiment_zero | Train rate 0.050000; test rate 0.055184; LightGBM 0.9515 [0.9124, 0.9728]; GRU 6000 0.9608 [0.9327, 0.9766] | Train rate 0.050000; test rate 0.049959; LightGBM 0.9510 [0.9273, 0.9655]; GRU 6000 0.9548 [0.9342, 0.9695] |

## Counts, thresholds, seeds, and training details

| Population / seeds | Train positives / labels | Test positives / labels | Threshold |
| --- | ---: | ---: | ---: |
| round3_v2 / round3 | 4900 / 98000 | 1441 / 24500 | 0.159179136679 |
| round3_v2 / round4 | 4900 / 98000 | 1334 / 24500 | 0.157103643788 |
| experiment_zero / round3 | 4900 / 98000 | 1352 / 24500 | 0.120784735193 |
| experiment_zero / round4 | 4900 / 98000 | 1224 / 24500 | 0.122034186585 |

- round3_v2/round3: train `1000-1199`; test `2000-2049`; GRU epochs=40, best_validation_loss=0.076012299.

- round3_v2/round4: train `8000-8199`; test `8500-8549`; GRU epochs=40, best_validation_loss=0.071211539.

- experiment_zero/round3: train `1000-1199`; test `2000-2049`; GRU epochs=40, best_validation_loss=0.028709297.

- experiment_zero/round4: train `8000-8199`; test `8500-8549`; GRU epochs=40, best_validation_loss=0.033522398.

## Historical LightGBM reproducibility checks

Agreement is required to four decimal places, regardless of CI overlap. References are the default-label in-distribution row in `experiment_zero_round3.md` and the rho=0/default row in `experiment_zero_round4.md`.

| Matching cell | Published PR-AUC | Rerun PR-AUC | Outcome |
| --- | ---: | ---: | --- |
| round3_v2/round3 | 0.7138 | 0.7138 | PASS |
| experiment_zero/round4 | 0.9510 | 0.9510 | PASS |

## Exact resolved configurations

Round-3 v2 is composed by `load_v2_configs()` from the selected experiment config and the training population override in `heldout_families.yaml`; the held-out impact override does not apply to this in-distribution cohort. Counts are momentum/reversion/linear/GBDT/MLP = 10/0/10/0/10. The experiment-zero population is 6/6/6/6/6. Both have 30 agents and aggregate configured q_max 6000. Full resolved configs below include every inherited population, risk, and market setting, with the explicit no-signal override.

### round3_v2: applies to both seed-set cells

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

### experiment_zero: applies to both seed-set cells

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

## Scope, deviations, and open questions

Only round-5 item 1 was run. No simulator equation, label, threshold rule, observer hyperparameter, or registered seed range was changed. Private agent flows are reduced to labels before observer construction; observers receive only public records and targets. The historical check concerns LightGBM point estimates, not bootstrap endpoints or GRU ensemble scores. This 2×2 table separates population and seed changes descriptively; it does not establish real-market generalization or a causal crowding mechanism.

No research-design deviations or open questions. Any failed check above is a reproducibility failure and requires investigation; it is not waived by uncertainty intervals.
