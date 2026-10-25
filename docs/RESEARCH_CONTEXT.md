# HerdLab: research context

HerdLab: Inferring and Predicting Crowded Trading in Multi-Agent Financial Markets
(subtitle: Learning Latent Strategy Crowding from Aggregate Market Observations)

Source: project specification v0.1, October 2026. Nothing below has been tested yet. Every hypothesis is open, and no experimental results exist.

## Central question

Can an external observer infer the hidden collective behavior of competing trading algorithms from public market activity, and predict when their synchronized decisions will destabilize prices?

The goal is not price prediction. Intended outputs: an open-source research framework, an experimentally supported finding (positive or negative), and possibly a workshop paper.

## Hypotheses (none assumed true)

- **H1, shared objectives produce correlated trading.** Independently trained agents with overlapping information may develop correlated positions even with different architectures. Test by varying overlap in training data, forecasting targets and risk objectives.
- **H2, crowded positions leave detectable market signatures (core contribution).** An observer with market data only may infer aggregate exposure and predict future synchronized order flow, and this may generalize to unseen agent populations.
- **H3, crowding increases vulnerability under specific liquidity conditions.** Synchronized agents may amplify exogenous shocks, especially when risk constraints force simultaneous deleveraging. Test whether an observer can forecast elevated liquidation risk before the price dislocation.

H1 establishes the phenomenon, H3 motivates why prediction matters, and H2 is the focus. A convincing negative result (hidden crowding is fundamentally hard to infer outside the training environment) counts as a useful outcome.

## Key methodological commitments

- The simulator knows everything; the observer sees only what an outside participant could legitimately observe. This separation is enforced at the data-interface level.
- Start with a falsifiable experiment (Experiment Zero, see `EXPERIMENTS.md`), not a large RL environment.
- Don't replay historical returns and add impact on top, since real returns already include real trading effects. Use historical data to calibrate volatility, liquidity regimes and background flow, then generate counterfactual trajectories under stated assumptions.
- Train on one market ecology, test on another (out-of-distribution evaluation is the headline).
- Simulated results are never presented as real market behavior.

## Identifiability

Different agent populations can produce the same public trajectory, so private positions are not uniquely recoverable. The defensible target is probabilistic inference about aggregate states and future behavior, with uncertainty estimates.

## Scientific risks

1. **Artificial crowding.** Identical data and objectives make similar trades trivial. Needs systematic heterogeneity and matched controls.
2. **Market-impact shortcut.** The observer could recover current flow from impact it already caused. Primary targets must be future flow or persistent hidden exposure, using only earlier observations.
3. **Simulator overfitting.** A network could learn our impact formula or risk thresholds. Hold out simulator mechanisms. If predicted crowding disappears when the impact model changes, the observer learned artifacts; design for this case explicitly.
4. **External validity.** A positive result in HerdLab does not show that real funds can be reverse-engineered. State this limitation explicitly.

Out of scope: active manipulation, intentionally misleading orders, spoofing. Only controlled simulated stress tests and passive observation of real markets.

## Real-data role

| Claim | Simulation | Public historical data |
| --- | --- | --- |
| Different models develop correlated positions | Directly testable | Generally not observable |
| Hidden crowding can be reconstructed | Testable against ground truth | Not directly verifiable |
| Market-flow features predict future extreme selling | Testable | Testable for aggregate public trade flow |
| Crowding causes larger liquidity shocks | Controlled counterfactual possible | Difficult to establish causally |

Real data can show whether an observable warning signal is useful outside the simulator. It cannot show that real AI strategies caused a given price move.

## Literature review (not started)

Organize around five questions:

| Group | Question |
| --- | --- |
| Quant crowding | What empirical evidence shows strategies converge? |
| Market impact and liquidation cascades | How does synchronized trading destabilize prices? |
| Inverse market inference | What private behavior can be reconstructed from public information? |
| Multi-agent learning | When do independent learners coordinate or converge? |
| Adversarial robustness | How do trading models behave under distribution shift and strategic interaction? |

References already named in the spec: ABIDES (github.com/abides-sim/abides), ABIDES-MARL (arXiv:2511.02016), Binance public data (github.com/binance/binance-public-data), LOBSTER sample data.

## Open questions

- Exact threshold definition for an "extreme agent-selling" episode (to be fixed from training data only).
- Form of the concave impact function g.
- Whether a latent-state sequence model adds anything over GRU/LightGBM baselines (build only after simple models show signal).
- Broader framing beyond finance (networks of AI agents, MARL) is speculative motivation, not a claim.
