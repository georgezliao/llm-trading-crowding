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

### 2026-10-26: round-2 evidence summary (simulator evidence only)

- **Simulator evidence only:** all results below are from the uncalibrated synthetic HerdLab model, not real-market evidence.
- Step 2 used 20 baseline episodes (10,000 steps): forced flow occurred in 0/20 episodes and 0/10,000 steps; its absolute-flow share was 0.
- Those baseline episodes had mean maximum drawdown 0.055845 and mean agent share of total absolute flow 0.340333.
- Step 4 used 20 matched low/high-overlap episode pairs; neither condition produced forced flow.
- High-minus-low overlap maximum drawdown was -0.00000014 (95% episode-bootstrap CI [-0.00000077, 0.00000035]).
- Excess realized-volatility difference was -0.00000001 (95% CI [-0.00000003, 0.00000000]); neither interval supports greater instability.
- High overlap instead reduced mean C^pos by 0.00147597 (95% CI [-0.00238331, -0.00052985]) and C^flow by 0.00309075 (95% CI [-0.00558926, -0.00080477]) in this implementation.
- Step 5 mapped 27 overlap x liquidity x drawdown cells (135 episodes): forced flow occurred in 0/27 cells and 0/135 episodes, so the baseline regime grid found no cascade region.
- Step 10 used 200 train and 50 episodes per evaluation split; default-label base rates were 0.050000 train, 0.058816 in-distribution, 0 held-out-family, and 0.107429 held-out-impact.
- Onset-label base rates were 0.008929 train, 0.008449 in-distribution, 0 held-out-family, and 0.012041 held-out-impact.
- For the default label in-distribution, GRU PR-AUC was 0.7964 (95% CI [0.7571, 0.8297]) versus 0.1094 [0.0959, 0.1228] for the trivial reference; GRU Brier was 0.0229 [0.0204, 0.0253] and recall at FAR <= 0.05 was 0.9077 [0.8859, 0.9327].
- Under held-out impact, default-label GRU PR-AUC was 0.7598 [0.7380, 0.7798], Brier 0.0480 [0.0438, 0.0523], and recall 0.7633 [0.7309, 0.7955].
- For onset, GRU PR-AUC was 0.2487 [0.2137, 0.2973] in-distribution and 0.1214 [0.1043, 0.1456] under held-out impact, against base rates 0.008449 and 0.012041.
- Only 11.6% of in-distribution and 30.9% of held-out-impact default events contained forced flow (onset: 11.6% and 27.5%), so the target mostly captures broader agent selling, not forced-liquidation cascades.
- These runs show predictive signal within this selected simulator and some impact-mechanism transfer; zero held-out-family events make family generalization untestable, and the results do not establish causality, real-market validity, or that greater overlap increases crowding.

### Pre-registration, round 3 (written 2026-10-26, before running)

Motivation, from the research lead's review of `experiment_zero_v2.md`: the held-out-family split had zero events and so was uninformative; only about 10% of default-label events overlap forced flow; agents are deterministic functions of public prices, so the observer may be replicating known strategy rules; GRU results came from a single training seed.

Fixed setting: `configs/experiment_zero.yaml` (impact 0.03, leverage ×20, drawdown limit 0.20, Amendment 1 risk rule), horizon 10, training-only 0.95 score quantile for every label.

1. **Held-out-architecture split.** Train population: momentum 6, reversion 6, linear 9, MLP 9. Test population: momentum 6, reversion 6, GBDT 18. Same total count (30) and aggregate q_max. Seeds: train 5000–5199, in-distribution test 5500–5549, held-out-architecture test 6500–6549. The split is informative only if the held-out test default-label base rate is at least 1%; otherwise it is reported as uninformative and no generalization claim is made from it.
2. **Labels:** default, onset, and forced-flow (score uses only forced agent selling), each with its own training-only threshold.
3. **Strategy-replication control.** A baseline whose features are the target exposures and order signals that the baseline momentum and reversion rules (config lookback, k, q_max, tanh rule, no per-agent dispersion) would produce from public prices, plus their lagged changes, fed to LightGBM with the same settings as the LightGBM observer.
4. **GRU seed variance.** Five GRU training seeds (6000–6004). The primary GRU score is the seed-averaged predicted probability; per-seed metrics and the min–max range are also reported.
5. **Observers:** trivial reference, logistic, LightGBM, replicator, GRU (5 seeds and ensemble), on the v2 splits (same seeds as v2) and the new architecture split.
6. **Primary comparisons:** paired episode-bootstrap 95% CIs of PR-AUC differences, GRU ensemble minus replicator and GRU ensemble minus LightGBM, on every split and label.
7. **Decision rules, fixed now:**
   - If GRU ensemble minus replicator has a CI that includes or lies below 0 on a split, the observer's advantage on that split is reported as consistent with strategy replication.
   - Evidence for H2-style generalization requires, on the held-out-architecture split (if informative), the GRU ensemble to beat both the trivial reference and the replicator with paired CIs excluding 0, on the default label. Results on the onset and forced-flow labels are reported as secondary.
   - Forced-flow-label results with fewer than 30 test events in a split are reported as underpowered.
8. No simulator parameter, threshold rule, observer hyperparameter or seed range is changed after viewing round-3 metrics.

### 2026-10-27: round-3 generalization and replication controls (simulator evidence only)

- **Simulator evidence only:** these results come from the uncalibrated synthetic HerdLab model, not real markets, funds, or live trading.
- The held-out-architecture default split was informative: 378/24,500 positives, a 0.015429 base rate above the pre-registered 1% minimum.
- On that split, GRU-ensemble default-label PR-AUC was 0.6571 (95% CI [0.4096, 0.7874]), versus 0.0396 [0.0140, 0.0772] for the trivial reference and 0.0223 [0.0076, 0.0437] for the strategy replicator.
- Paired GRU-minus-trivial and GRU-minus-replicator differences there were 0.6175 [0.3823, 0.7314] and 0.6348 [0.4141, 0.7551]; the pre-registered H2-style generalization rule therefore passed in this simulator.
- GRU also exceeded LightGBM on held-out-architecture/default by 0.0539 [0.0272, 0.0995], while its five-seed PR-AUC range was 0.6091–0.6727.
- V2 default-label base rates were 0.058816 in-distribution, 0 in held-out families, and 0.107429 under held-out impact; the zero-event family split remains uninformative.
- V2 default GRU PR-AUC was 0.8048 [0.7700, 0.8345] in-distribution and 0.7433 [0.7212, 0.7641] under held-out impact.
- Its paired advantage over the replicator was 0.5676 [0.5344, 0.5977] in-distribution and 0.4175 [0.3823, 0.4554] under held-out impact; both exclude zero above.
- Its paired advantage over LightGBM on those default splits was 0.0909 [0.0724, 0.1098] and 0.1315 [0.1077, 0.1564].
- Onset-label GRU-minus-replicator intervals also excluded zero wherever events existed; held-out-architecture/onset was 0.1170 [0.0770, 0.1741] at a 0.004000 base rate.
- Forced-flow base rates were 0.045347, 0.000490, and 0.076776 across the three V2 test splits, and 0.050857 and 0.046939 across the architecture test splits.
- V2 held-out-family forced flow had only 12 events and is underpowered by the pre-registered 30-event rule; its apparent metrics support no conclusion.
- On V2 held-out-impact forced flow, GRU minus replicator was -0.0037 [-0.0232, 0.0177], so the pre-registered rule reports the advantage as consistent with strategy replication on that split.
- On held-out-architecture forced flow, GRU PR-AUC was 0.1646 [0.1134, 0.2089], with paired gains of 0.0599 [0.0260, 0.0927] over the replicator and 0.0366 [0.0131, 0.0602] over LightGBM.
- The five-seed ranges, including 0.6672–0.7598 for V2 held-out-impact/default, show meaningful training-seed variation but do not overturn the ensemble rule outcomes.
- These runs show predictive public-data signal and architecture/impact transfer under this fixed simulator; they do not establish identifiable hidden positions, causality, unseen-strategy or market-mechanics generalization, or real-market validity.

### Pre-registration, round 4 (written 2026-10-28, before running)

Motivation, from the review of round 3: every agent decides only from lagged public prices, so hidden crowding is in principle a function of public history and the inverse problem is easier than the spec intends. `feature_overlap` affects only the factor agents, so H1 has not been tested. The spec's feature ablations have not been run.

Fixed setting: `configs/experiment_zero.yaml` (impact 0.03, leverage ×20, drawdown limit 0.20, Amendment 1 risk rule), horizon 10, training-only 0.95 score quantile computed separately within every condition.

1. **Private signals.** At step t, agent i receives s_{i,t} = ρ·z_{t+1} + sqrt(1−ρ²)·ε_{i,t}, where z_{t+1} is the next exogenous return divided by its configured volatility and ε_{i,t} ~ N(0,1). Every agent adds β·σ̂_t·s_{i,t} to its forecast before the tanh rule, with β = 1. The signal noise comes from its own RNG stream, so the exogenous path does not change with ρ. With ρ = 0 the signal is switched off entirely, and the run must match round 3 exactly. Signals are private state and never reach the public record or the observer.
2. **Signal-strength sweep.** ρ ∈ {0, 0.1, 0.25, 0.5}. The same seeds are used at every ρ: train 8000–8199, test 8500–8549. Observers: trivial, logistic, LightGBM, replicator, and a GRU ensemble over seeds 6000–6002. Labels: default (primary) and forced-flow (secondary).
3. **Feature ablations** on the ρ = 0 data. LightGBM and GRU seed 6000 are refit with one public feature group removed at a time: order flow (imbalance), volatility, liquidity (the liquidity proxy, its change and volume/liquidity), and volume. Two simulator ablations reuse the same seeds: impact switched off (impact_sensitivity = 0) and forced deleveraging switched off (risk rules disabled). For these two, report base rates and observer metrics, with thresholds refit within each condition.
4. **Overlap for learned agents (H1).** A new mode `agents.learned_overlap_mode: lag_weights` applies the factor agents' shared/individual lag weighting to the lagged-return inputs of the linear, GBDT and MLP agents. With private signals on, the noise is split as ε_{i,t} = sqrt(ω)·ε^common_t + sqrt(1−ω)·ε^idio_{i,t}, where ω is `feature_overlap`. The default mode `none` keeps round-3 behavior and is used for items 2–3. Sweep: ω ∈ {0, 0.25, 0.5, 0.75, 1.0} at ρ = 0.25, with 30 episodes per level on seeds 7000–7029, shared across levels. Outcomes per episode: mean C^pos, mean C^flow, the default-label event rate using the threshold from the ω = 0.5 level's own episodes, maximum price drawdown, and the forced-order fraction.
5. **Decision rules, fixed now:**
   - *Private information degrades inference* if, on the default label, the bootstrap CI of [(GRU − trivial) at ρ = 0] − [(GRU − trivial) at ρ = 0.5] lies above 0. Test episodes are resampled jointly across ρ, since they share seeds. A condition with a test base rate below 1% is reported as uninformative.
   - The round-3 replication rule (GRU minus replicator) is applied at each ρ.
   - *A feature group is used* if the paired CI of full minus ablated PR-AUC excludes 0 and is positive. This is reported per observer.
   - *H1 is supported in this simulator* if the episode-bootstrap CI of the OLS slope of episode mean C^pos on ω lies above 0, and the event-rate slope CI also lies above 0. If only the C^pos slope is positive, the report says that overlap raises alignment without raising unwinds.
   - Forced-flow results with fewer than 30 test events are reported as underpowered.
6. No ρ, β, ω, seed range, threshold rule or observer hyperparameter is changed after viewing round-4 metrics. If total runtime would exceed about 60 minutes, the run stops and reports instead of cutting episodes.

### 2026-10-28: round-4 private-information, ablation, and overlap results (simulator evidence only)

- **Simulator evidence only:** all results are from the uncalibrated synthetic HerdLab model, not real markets, funds, or live trading.
- Default-label test base rates were 0.049959, 0.053061, 0.054776, and 0.054898 at ρ = 0, 0.1, 0.25, and 0.5, so every default condition cleared the pre-registered 1% informativeness floor.
- GRU-ensemble default PR-AUC was 0.9564 [0.9355, 0.9694] at ρ = 0 and 0.3158 [0.2801, 0.3511] at ρ = 0.5; the corresponding trivial-reference values were 0.0897 [0.0528, 0.1324] and 0.0738 [0.0654, 0.0843].
- The jointly resampled degradation statistic, [(GRU − trivial) at ρ = 0] − [(GRU − trivial) at ρ = 0.5], was 0.6246 [0.5803, 0.6632]; private information therefore degraded inference by the pre-registered rule.
- GRU-minus-replicator default-label CIs excluded zero above at every ρ: estimates were 0.8733, 0.1509, 0.1850, and 0.2076, so none was reported as consistent with strategy replication by that rule.
- Liquidity was the only default-label feature group reported as used: full-minus-ablated PR-AUC was 0.0072 [0.0040, 0.0106] for LightGBM and 0.0084 [0.0041, 0.0137] for GRU seed 6000.
- Order flow, volatility, and volume were not reported as used for the default label by either observer; no feature group met the use rule for forced flow.
- Forced-flow test base rates were 0.062000 at ρ = 0 but only 0.000571, 0, and 0.000571 at ρ = 0.1, 0.25, and 0.5.
- Only ρ = 0 forced flow met the 30-event power rule (1,519 events); ρ = 0.1, 0.25, 0.5, impact-off, and risk-off had 14, 0, 14, 24, and 0 events and were underpowered.
- With impact off, default-label base rate was 0.048000 and GRU PR-AUC was 0.2594 [0.2198, 0.2982]; with risk off they were 0.049918 and 0.9516 [0.9280, 0.9651].
- In the overlap sweep, mean C^pos rose from 0.24687875 at ω = 0 to 0.83329869 at ω = 1, while default-label event rate rose from 0.00034014 to 0.14163265.
- The C^pos slope was 0.5641311472 [0.5573096332, 0.5708205978] and the event-rate slope was 0.1421496599 [0.1342034014, 0.1498231293]; both CIs lay above zero, so H1 was supported in this simulator by the pre-registered rule.
- Maximum price drawdown also rose from 0.10941507 to 0.21288511 across the overlap endpoints, but forced-order fractions were zero except 0.00001111 at ω = 0.
- These runs show that added private information weakens the public observer under the fixed degradation comparison, and that mechanically increased learned-agent overlap raises alignment and the broad agent-selling event rate in this simulator.
- They do not show that overlap produced forced-liquidation cascades, identify causal mechanisms behind the observer scores, establish transfer beyond this simulator, or provide evidence about real-market crowding.

### Pre-registration, round 5 (written 2026-11-05, before running)

Motivation, from the review of round 4. GRU PR-AUC fell from 0.96 at ρ = 0 to about 0.21 at ρ = 0.1, then rose to 0.32 at ρ = 0.5. Because ρ = 0 switches the signal term off entirely, the round-4 degradation contrast mixes "noise added to forecasts" with "information added to forecasts". Forced flow also almost vanished at ρ > 0. Separately, the ρ = 0 score (0.96) is well above round 3's in-distribution 0.80. Round 3's v2 cohort used the population in `configs/heldout_families.yaml` and `configs/heldout_impact.yaml` with seeds 1000–2049, while round 4 used `configs/experiment_zero.yaml` with seeds 8000–8549, so population and seeds were both changed.

The noise explanation was formed after seeing round-4 results, so this round is a test of a post-hoc hypothesis. Written prediction: at fixed β, a larger ρ gives agents more common information, so crowding should become easier to see, not harder. This is the opposite of the round-4 rule's reading.

Fixed setting unless stated: `configs/experiment_zero.yaml`, horizon 10, training-only 0.95 score quantile within every condition, train seeds 8000–8199, test seeds 8500–8549, GRU ensemble over seeds 6000–6002.

1. **Reconciliation.** A 2×2 design crosses population (round-3 v2 in-distribution cohort, `experiment_zero.yaml`) with seed set (round-3 v2 train and in-distribution seeds, round-4 seeds), with no signal. Observers: LightGBM and GRU seed 6000, default label. Each cell where both population and seeds match an earlier run must reproduce that run's LightGBM PR-AUC to 4 decimal places; a mismatch is reported as a reproducibility failure, not explained away.
2. **Noise-only control.** A new mode `agents.private_signal.mode` takes `off`, `signal` or `noise_only`. `noise_only` adds β·σ̂_t·ε_{i,t} with ε_{i,t} ~ N(0,1), which is the ρ = 0 limit of the signal term with unit variance and no information. Existing configs keep their behavior: `strength: 0` still means off unless `mode: noise_only` is set explicitly.
3. **Grid.** β ∈ {0.1, 0.25, 0.5, 1.0} × {noise-only, ρ = 0.25, ρ = 0.5}, plus the off baseline: 13 conditions. Observers: trivial, logistic, LightGBM, replicator, GRU ensemble. Labels: default (primary) and forced-flow (secondary). Report at each β the ratio of the standard deviation of the signal term to that of the agent's base forecast, pooled by agent family.
4. **Forced-flow stress conditions.** β = 0.25 × ρ ∈ {0.25, 0.5} × `risk.max_drawdown` ∈ {0.05, 0.10}: 4 more conditions with the same observers and labels. These values are fixed now and are not selected from results.
5. **Overlap under stress.** The round-4 ω sweep (ω ∈ {0, 0.25, 0.5, 0.75, 1.0}, `learned_overlap_mode: lag_weights`, seeds 7000–7029) rerun at ρ = 0.25, β = 0.25, `max_drawdown` 0.05, with the same outcomes plus forced-flow event count per level.
6. **Decision rules, fixed now.** GRU gain means GRU ensemble PR-AUC minus trivial PR-AUC. All cross-condition contrasts resample test episodes jointly, since seeds are shared.
   - *Noise effect at β:* (gain off) − (gain noise-only). Noise degrades inference at β if the CI lies above 0.
   - *Information effect at β and ρ:* (gain at ρ) − (gain noise-only). Private information makes inference harder at that cell only if the CI lies below 0, and easier only if it lies above 0.
   - The round-4 "private information degrades inference" reading is reported as *not supported* if, at β = 1, the noise effect CI lies above 0 and the information effect CI at ρ = 0.5 does not lie below 0.
   - *Swamping:* the report states the smallest β whose noise effect CI includes 0, or says none.
   - The replication rule applies at every condition. Conditions with a test base rate below 1% are uninformative. Forced-flow results with fewer than 30 test events are underpowered.
   - *H1 for forced selling* is supported only if the bootstrap CI of the slope of episode forced-order fraction on ω lies above 0 and at least three ω levels have 30 or more forced-flow events in total across their episodes. Otherwise the report says forced selling was not tested adequately.
7. No β, ρ, ω, drawdown limit, seed range, threshold rule or observer hyperparameter is changed after viewing round-5 metrics. If total runtime would exceed about 75 minutes, the run stops and reports instead of cutting episodes.

### Amendment 2 (written 2026-11-05, after the R5-3 runtime stop, before any grid metric existed)

The R5-3 step stopped before running because a conservative estimate of 81 minutes exceeded the roughly 75-minute limit in item 7. The limit was a guard against unplanned compute, not a scientific parameter, so it is raised to about 120 minutes for R5-3 and R5-4 combined. No condition, seed, episode count, threshold rule, observer or decision rule changes. The reconciliation results from R5-2 were available when this was written, but they do not bear on the runtime limit.

### 2026-11-05: round-5 noise, information, reconciliation, and stress results (simulator evidence only)

- **Simulator evidence only:** this synthesis uses the completed [reconciliation](reports/round5_reconcile.md), [grid](reports/experiment_zero_round5.md), and [stress-overlap](reports/overlap_sweep_stress.md) reports from uncalibrated synthetic HerdLab; no real-market evidence is claimed.
- The grid used 200 training and 50 test episodes per condition; all intervals below are 95% whole-episode bootstrap CIs, with shared seeds resampled jointly for cross-condition contrasts. Gain means GRU-ensemble PR-AUC minus trivial-reference PR-AUC.
- Both historical LightGBM reconciliation checks passed to four decimals: round 3, 0.7138 [0.6713, 0.7509] at test base rate 0.058816; round 4, 0.9510 [0.9273, 0.9655] at 0.049959.
- Crossing seeds gave LightGBM 0.6935 [0.6603, 0.7246] for the round-3 population on round-4 seeds (rate 0.054449), and 0.9515 [0.9124, 0.9728] for experiment-zero on round-3 seeds (rate 0.055184): descriptively, the score gap tracks population more than seed set.
- Default-label noise effects (gain off minus gain noise-only) were 0.2064 [0.0849, 0.3691], 0.2595 [0.1560, 0.4105], 0.4426 [0.3680, 0.5419], and 0.7416 [0.6927, 0.7770] at β = 0.1, 0.25, 0.5, 1; noise degrades inference at every tested β, and the registered swamping outcome is “none.”
- At β = 1, default information effects (gain signal minus gain noise-only) were 0.0557 [0.0254, 0.0874] at ρ = 0.25 and 0.1170 [0.0797, 0.1547] at ρ = 0.5: private information makes inference easier, and the round-4 “private information degrades inference” reading is not supported by the registered rule.
- Default test base rates for off, β = 1 noise-only, and β = 1 / ρ = 0.5 were respectively 0.049959 (CI [0.0302, 0.0705]), 0.050000 [0.0449, 0.0548], and 0.054898 [0.0503, 0.0600]; all 17 default conditions cleared the 1% informativeness floor.
- Information instead made default inference harder at β = 0.1 / ρ = 0.5 (-0.1930 [-0.3233, -0.0798]), β = 0.25 / ρ = 0.5 (-0.3778 [-0.4856, -0.2160]), and both β = 0.5 cells (-0.2147 [-0.3091, -0.1133] and -0.2008 [-0.2769, -0.1004] for ρ = 0.25 and 0.5).
- At ρ = 0.25, β = 0.1 and 0.25 established no directional default information effect: -0.0650 [-0.2108, 0.0825] and -0.1223 [-0.2427, 0.0186]. Thus the written prediction that information makes crowding easier to see is not uniformly supported across β.
- Even β = 0.1 noise-only had signal-term/base-forecast standard-deviation ratios 1.8660 [1.8259, 1.9147] for momentum and 1.8775 [1.8401, 1.9202] for reversion, versus 0.1050 [0.1046, 0.1054] for linear agents; β does not imply equal relative perturbations across families.
- All default conditions and all informative forced-flow conditions were “not consistent with strategy replication by the registered rule”; this is a comparison with the specified replicator, not proof that strategy replication is impossible.
- Secondary forced-flow noise effects were positive at β = 0.1 and 0.25 (0.0463 [0.0051, 0.0874] and 0.0402 [0.0015, 0.0840]); information made inference easier at β = 0.1 for both ρ values, but no direction was established at β = 0.25 / ρ = 0.25.
- Forced flow was uninformative at β = 0.25 / ρ = 0.5 and every β ≥ 0.5 grid cell; β = 0.5 signal cells and all β = 1 cells were also underpowered (0, 0, 28, 0, and 14 positive windows respectively). Off had 1,519 positives, base rate 0.0620 [0.0493, 0.0744].
- All four β = 0.25 stress cells cleared both forced-flow gates: rates were 0.0622 [0.0511, 0.0744], 0.0449 [0.0341, 0.0575], 0.0591 [0.0439, 0.0742], and 0.0162 [0.0097, 0.0234], with 1,524/1,101/1,449/398 positives for (ρ, drawdown) = (0.25, 0.05), (0.25, 0.10), (0.5, 0.05), (0.5, 0.10).
- In the stress-overlap sweep, alignment and default-event-rate slopes were positive: 0.1389595815 [0.1218977474, 0.1564824234] and 0.0951292517 [0.0824761905, 0.1075102041]; default event rates rose from 0.01843537 at ω = 0 to 0.11034014 at ω = 1.
- The forced-order-fraction slope was negative, -0.0050933333 [-0.0058524889, -0.0043955556], despite four levels clearing 30 forced-flow positives (totals 2,841/1,312/199/42/14); the registered H1 outcome is “forced selling was not tested adequately.” The failure is the slope criterion, not an absence of events at every level.
- Sweep counts include overlapping positive windows, not distinct liquidation episodes; forced-order fraction includes buys and sells. Its ω = 0.5 calibration is an in-sample diagnostic, and extending that calibration to forced flow remains flagged for lead review; prior duplicate/interrupted runtime accounting also remains open.
- These results separate noise and information effects under the fixed simulator and show increased alignment and broad selling with overlap, but do not establish uniquely identifiable hidden positions, increased forced-liquidation cascades, generalization beyond the tested settings, or real-market validity; no research settings were changed for this synthesis.

### Pre-registration, round 6 (written 2026-11-06, before running)

Motivation, from the review of round 5. Population moved GRU PR-AUC by 0.16 to 0.19 while seeds moved it by at most 0.035, and the cause is unknown. Forced selling fell as overlap rose, which contradicts the usual crowding story, and the round-5 H1 rule labelled that clear negative slope "not tested adequately". A single β also perturbed momentum and reversion agents about ten times more than linear agents, so the noise-versus-information comparison was uneven across families.

Fixed setting unless stated: `configs/experiment_zero.yaml`, horizon 10, training-only 0.95 score quantile within every condition, train seeds 8000–8199, test seeds 8500–8549, GRU ensemble over seeds 6000–6002, observers trivial, LightGBM, replicator and GRU ensemble. Default label is primary and forced-flow secondary. GRU gain means GRU ensemble PR-AUC minus trivial PR-AUC. Cross-condition contrasts resample test seeds jointly. No signal unless stated.

1. **Population decomposition.** Seven populations: the 6/6/6/6/6 baseline; five leave-one-family-out populations of 24 agents; and the round-3 v2 mix (momentum, linear and MLP, 10 each), which drops reversion and GBDT together. In the 24-agent populations, `q_max`, `order_limit` and `initial_cash` are scaled by 30/24, so aggregate capital and per-agent leverage match the baseline. The baseline must reproduce round 5's off condition (LightGBM 0.9510).
2. **Mechanism sweep.** Round-5 stress setting: ρ = 0.25, β = 0.25, `max_drawdown` 0.05, `learned_overlap_mode: lag_weights`, seeds 7000–7029. `feature_overlap` currently sets both lag-weight overlap and signal-noise sharing, so these become separate settings. Run the round-5 five-level ω sweep, plus a 2×2 of lag overlap {0, 1} × noise sharing {0, 1}. Episode outcomes: drawdown breaches, fraction of agents that ever breach, largest number of agents liquidating at once, mean agent maximum drawdown, cross-sectional dispersion of agent P&L, mean agent return, and forced-flow window count at the existing zero threshold.
3. **Per-family signal scaling.** A new option `agents.private_signal.scale: family` replaces σ̂_t in the signal term with a fixed constant per family: the pooled standard deviation of that family's base forecast in the off condition on training seeds only. β then equals the intended signal-to-forecast ratio for every family. Default `sigma` keeps earlier behavior. Grid: β ∈ {0.25, 0.5, 1.0} × {noise-only, ρ = 0.25, ρ = 0.5}, compared against the off condition: 9 new conditions.
4. **Decision rules, fixed now.**
   - *Family effect:* a family contributes to visibility if (gain baseline) − (gain with it removed) has a CI above 0, and suppresses visibility if the CI lies below 0. A family *explains the population gap* if that contrast's point estimate has the same sign as, and is at least half of, (gain baseline) − (gain v2).
   - *Interaction:* if neither reversion nor GBDT alone explains the gap but the v2 contrast has a CI above 0, the report says the gap needs both changes or something else in the v2 mix, and does not name a family.
   - *Forced-selling direction:* the slope of episode forced-order fraction on ω is reported as *increases*, *decreases* or *inconclusive* from whether the CI lies above 0, below 0 or spans 0. Levels with fewer than 30 forced-flow windows are named in the report.
   - *Mechanism candidates:* M1, agents draw down less: the mean agent maximum drawdown slope on ω has a CI below 0. M2, idiosyncratic losers: the P&L dispersion slope has a CI below 0. M3, agents profit from shared flow: the mean agent return slope has a CI above 0. More than one may hold. These are associations, not causes.
   - *Knob attribution:* in the 2×2, the main effect of each setting on forced-order fraction uses episode bootstrap with shared seeds. A setting drives the decline if its effect's CI lies below 0. This is the only causal claim allowed here, and it is limited to this simulator.
   - *Fair scaling:* the round-5 noise and information rules apply unchanged to the scaled grid. The report states whether each β's information effect changed direction compared with the nearest round-5 β, as a description only.
   - The replication, 1% base-rate and 30-event rules apply throughout.
5. No population, β, ρ, ω, drawdown limit, seed range, threshold rule or observer hyperparameter is changed after viewing round-6 metrics. Runtime limit: about 130 minutes in total for one uninterrupted pass through items 1 and 3. Over that, the run stops and reports, and cached conditions are reused rather than recomputed.

### 2026-11-07: round-6 population, mechanism, and scaled-signal results (simulator evidence only)

The implementation agent hit a usage limit after R6-3. The research lead finished R6-4 to R6-6 with the design unchanged. Reports: `docs/reports/round6_population.md`, `docs/reports/round6_mechanism.md`, `docs/reports/round6_scaled_grid.md`.

- **Deviation.** The scaled-grid estimate of 57 minutes was about half the real 125.17 minutes. With the 25.80-minute population run, the total was about 151 minutes against the ~130-minute limit in item 5. The runner doesn't enforce the limit, so the last two conditions finished past it instead of stopping. No setting changed, and the results are kept with this deviation recorded.
- The baseline reproduced round 5's off condition: LightGBM 0.9510 [0.9281, 0.9653], GRU 0.9564 [0.9355, 0.9694], default base rate 0.0500 [0.0302, 0.0705].
- Population, default label, GRU PR-AUC: without momentum 0.7291, without reversion 0.8042, without MLP 0.9010, without linear 0.9103, without GBDT 0.9307, v2 0.7647. Test base rates were 0.0408–0.0622.
- Gain lost on removal: momentum 0.3088 [0.2264, 0.4176], reversion 0.1505 [0.0880, 0.2212] and MLP 0.0623 [0.0338, 0.0932] contribute to visibility. Linear 0.0277 [-0.0218, 0.1181] and GBDT 0.0093 [-0.0238, 0.0478] have no established effect. The v2 gap is 0.2050 [0.1644, 0.2442].
- The gap rule says both momentum and reversion "explain the population gap". The momentum half is a flaw in the rule: v2 has more momentum agents than the baseline, so removing momentum can't describe how v2 differs. Reversion, which v2 does drop, is the sensible candidate, but no run removes reversion alone from v2. The verdict stands as registered and this note qualifies it.
- Population and forced flow: without momentum is uninformative, and no other family has an established effect.
- Mechanism: forced selling decreases with ω, slope -0.0050933333 [-0.0058791333, -0.0043990889]. Forced-flow windows at the five ω levels were 2,841/1,312/199/42/14, and ω = 1 is under 30 windows. ω = 0.75 and 1 are below the 1% floor. Those counts include overlapping windows.
- The factorial attributes the decline to noise sharing: -0.0050455556 [-0.0056744722, -0.0044755556]. Lag overlap is inconclusive: -0.0002122222 [-0.0005378056, 0.0000977778]. This is the only causal claim, and it is limited to this simulator. Shared noise isn't the same thing as shared strategy, so this does not show that crowding protects agents.
- M1 to M3 are all supported as associations. Mean agent maximum drawdown fell (slope -0.0231 [-0.0260, -0.0206]), P&L dispersion fell (-2636 [-3022, -2251]), and mean agent return rose (0.4528 [0.4043, 0.5022]). The fraction of agents ever breaching fell from 0.224 to 0.003. With independent noise, the forced sellers look like agents whose noise happened to lose. Whether agents profit from shared flow was not tested.
- Family scaling, default label, noise effect: 0.2256 [0.1049, 0.3876], 0.3507 [0.2460, 0.4700], 0.7774 [0.7326, 0.8120] at β = 0.25, 0.5, 1. Noise degrades inference at every β, as in round 5. Default base rates were 0.0420–0.0554, and swamping is "none".
- Family scaling, information effect: at β = 0.25 no direction is established (ρ = 0.25: 0.0954 [-0.0420, 0.2485]; ρ = 0.5: -0.0728 [-0.2205, 0.0741]). At β = 0.5 information makes inference harder (-0.1378 [-0.2594, -0.0039]; -0.2554 [-0.3465, -0.1332]). At β = 1 it makes inference easier (0.0593 [0.0327, 0.0805]; 0.0945 [0.0651, 0.1224]).
- Compared with round 5: the β = 0.5 and β = 1 directions hold. The strong round-5 result at β = 0.25 / ρ = 0.5 (-0.3778 [-0.4856, -0.2160]) doesn't survive scaling as an established effect, though its sign is still negative. The report's "direction changed" means the rule outcome changed. This comparison is descriptive, because the same numeric β meant a different signal size in round 5.
- Scaling worked only partly. Measured signal-to-forecast ratios were 0.21–0.26 at β = 0.25 and 0.42–0.62 at β = 0.5, but 0.80–5.79 at β = 1. Linear agents' base-forecast spread falls to about 0.17 of its off-condition size with β = 1 noise, for reasons not tested. So the β = 1 "easier" result is not the equal-ratio comparison the design intended, and it occurs where learned agents' forecasts are dominated by the signal term.
- Forced flow in the scaled grid: noise degrades inference only at β = 1 (0.0366 [0.0033, 0.0756]). No information effect has a direction, β = 1 / ρ = 0.25 is uninformative (rate 0.0092), and β = 1 / ρ = 0.5 is also underpowered (28 positives). Five conditions have a training threshold of exactly 0.
- All informative conditions were "not consistent with strategy replication by the registered rule", apart from without_reversion and without_gbdt on forced flow, which were "consistent with strategy replication". This compares against the specified replicator only.
- Open: the mechanism step's definitions of breach, drawdown, dispersion and zero-position liquidation need review, as does the zero forced-flow threshold.
- These results show which families drive visibility in one market setting, and which knob reduces forced selling in this simulator. They do not establish that private information has a single direction of effect, that crowding reduces forced selling, generalization beyond these settings, or anything about real markets.

