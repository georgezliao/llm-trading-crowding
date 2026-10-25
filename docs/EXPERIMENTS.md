# HerdLab: experiments

Status as of 2026-10-25: no experiments have been run. Everything here is planned design.

The interactive "synchronized unwinding" chart in the original spec was a hand-built illustration, not simulator output. Do not cite it as a result.

## Experiment Zero: can the observer predict a hidden liquidation cascade?

- Setup: one simulated asset, 20–30 heterogeneous strategies, stochastic shocks, variable liquidity, independently specified risk thresholds.
- Hidden state: each agent's exposure, leverage and risk limits.
- Public observations: past returns, volatility, volume, aggregate executed trade imbalance.
- Prediction: does a large episode of agent-originated net selling begin within the next 10 steps?
- Comparison: does a temporal observer beat logistic regression and LightGBM, especially when the test population uses unseen strategies?
- Success: measurable, statistically supported improvement over simple baselines on held-out episodes, persisting under at least some changes in agent composition or market mechanics.

## Generalization matrix

| Experiment | Changes at test time |
| --- | --- |
| In-distribution | New trajectories, familiar agent families |
| Unseen architectures | Whole ML model families held out |
| Unseen strategies | Different signals and risk objectives |
| New market regimes | Volatility, liquidity, shock distributions |
| New market mechanics | Alternative impact and execution models |
| Combined OOD | Unseen agents under unseen regimes |

Example split: train on momentum + linear + MLP populations; test on GRU + GBDT with unseen portfolio constraints.

## Metrics

- Flow forecasting: sign accuracy, out-of-sample R² vs zero-flow and persistence forecasts.
- Unwind warnings: PR-AUC (rare events), Brier score and calibration, recall at fixed false-alarm rate, lead time relative to unwind start.
- Market outcomes: drawdown, downside tail risk, excess volatility, persistence of impact.
- Many independent episodes, confidence intervals, and no treating overlapping step-level predictions as independent (split and bootstrap by episode).

## Ablations

Observer with and without: (1) order-flow info, (2) volatility info, (3) liquidity info, (4) impact feedback, (5) heterogeneous architectures, (6) forced-deleveraging rules.

Strong ablation: two populations with identical aggregate capital and comparable predictive skill but different degrees of shared positioning.

Explicit check: does predicted crowding survive a change of impact model? If not, the observer learned simulator artifacts.

## Planned milestones (organizational, not a time estimate)

1. Historical data and market spec: ingestion, exogenous processes, causal features, time-step convention, accounting. Deliverable: reproducible dataset and tested market-state generator.
2. Heterogeneous agents: strategies, positions, execution, capital constraints, synchronization measures. Deliverable: similarity varied independently of shocks.
3. Endogenous feedback: impact, liquidity variation, margin deleveraging, counterfactual comparisons. Deliverable: crowding vs instability plots under matched shocks.
4. Inverse observer: episodes, hidden state, logistic/LightGBM/temporal models. Deliverable: first out-of-sample crowding and unwind forecasts.
5. Generalization: unseen families, regimes, impact models, constraints; ablations and uncertainty. Deliverable: generalization matrix.
6. External validation and write-up on real data. Deliverable: report, figures, scripts, demo.

## Results log

(empty; append dated entries with config, seed range, metric, baseline and caveats)
