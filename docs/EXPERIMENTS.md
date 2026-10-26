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


### 2026-10-26: diagnostics on defaults (simulator only)

Baseline cascade diagnostics (20 episodes), matched low/high-overlap pairs (20) and a 3×3×3 overlap × liquidity × drawdown-limit grid (135 episodes) recorded zero forced orders. Excess realized volatility from agent impact was about 1e-7, and high overlap moved mean C^pos and C^flow by less than 0.003. At the defaults the feedback loop behind H3 never starts, so Experiment Zero v2 was paused. Likely cause: `impact_sensitivity: 0.001` and q_max 10 at price ~100 against 10,000 cash, a gross leverage cap of about 0.1 that never approaches the risk limits. Details in `docs/reports/cascade_diagnostics.md`, `matched_crowding.md`, `regime_grid.md`.

### Pre-registration (written 2026-10-26, before running): impact × leverage × drawdown grid

- Grid: `impact_sensitivity` ∈ {0.001, 0.003, 0.01, 0.03, 0.1} × leverage multiplier m ∈ {1, 5, 20} (q_max and order_limit both scaled by m, giving gross leverage caps of about 0.1, 0.5, 2.0) × `risk.max_drawdown` ∈ {0.05, 0.10, 0.20}. 45 cells, 10 episodes each, seeds 0–9 shared across cells. Everything else at baseline.
- Eligible cell: `max_drawdown` = 0.20 (the baseline value, fixed in advance); nonzero forced flow in at least 5 of 10 episodes; mean forced-step fraction ≤ 0.10; median price max drawdown ≤ 0.30; no non-finite or negative agent equity.
- Selection: the eligible cell with the lowest `impact_sensitivity`, ties broken by the lowest m. The other drawdown limits are reported as a map only.
- If no cell is eligible, the result is reported as a null and Experiment Zero v2 does not run until the research lead revisits the model.
- The selected setting is fixed before any observer is trained and is not revised based on observer metrics.

### 2026-10-26: pre-registered impact × leverage grid, result: null

Run exactly as pre-registered (450 episodes, `docs/reports/impact_leverage_grid.md`). No cell was eligible, so no setting was selected and Experiment Zero v2 did not run. At the 0.20 drawdown limit, forced flow appeared only at leverage ×20: in 2/10 episodes at impact 0.001 and 0.01, and in 10/10 at 0.03 and 0.1. The 10/10 cells failed the forced-step criterion (fractions 0.36 and 0.74 against a 0.10 cap), with median price drawdowns of 10.3% and 25.2%.

Post-hoc diagnosis by the research lead: in those cells forced orders were only about 0.02–0.04% of absolute agent flow. The risk rule measures drawdown from an all-time peak that never resets, so after an agent liquidates to flat it stays breached for the rest of the episode and logs a zero-size forced order every step. The forced-step fraction therefore mostly counts locked-out agents, not forced selling. This is a model-design flaw found after seeing the results; the null above stands as recorded.

### Amendment 1 (written 2026-10-26, after the null above, before rerunning)

Model change, motivated by the lockout described above, not by observer metrics:

- When a forced liquidation schedule completes, the agent enters a cooldown of `risk.cooldown_steps` = 20 steps with target exposure 0. At the end of the cooldown its peak equity resets to current equity and it resumes strategy trading.
- An order is labelled forced only if its quantity is nonzero. Cooldown steps with zero orders are not forced steps.

Rerun: the identical grid, seeds, episode count, eligibility rule and selection rule from the original pre-registration, under the amended model. The result is reported alongside the original null, not in place of it. If no cell is eligible, that null is final for this model version and Experiment Zero v2 does not run.
