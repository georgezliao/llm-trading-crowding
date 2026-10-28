# HerdLab: architecture

Planned design from spec v0.1. No code exists yet; update this file when implementation diverges.

## Data flow

```
Market data & feature engine (real returns, vol, volume, order flow)
        |  calibrates exogenous processes
        v
Agents (factor, ML, temporal)  <-->  Multi-agent market simulator
                                     (execution, impact, liquidity, risk limits, forced liquidation)
                                        |                         |
                               Private ground truth        Public observations
                               (positions, orders,         (prices, volume, imbalance,
                                leverage, crowding)          volatility)
                                        |                         |
                                        +--> labels only    Inverse observer model
                                                                  |
                                                             Evaluation
```

The observer code path must never be able to import or receive private state. Labels flow to training as targets only.

## Agents

Default population of 30 (experimental defaults, not claims about real fund composition):

| Family | Count | Strategy |
| --- | --- | --- |
| Momentum | 6 | Follow recent returns/trends |
| Mean reversion | 6 | Trade against short-term deviations |
| Linear ML | 6 | Ridge/logistic return forecast |
| Gradient-boosted trees | 6 | LightGBM forecast |
| Neural networks | 6 | MLP or small GRU forecast |

Each agent gets common market features plus agent-specific data and parameters; overlap is varied in a controlled way.

Decision model for agent i at time t:

- Forecast: r̂_{i,t+1} = f_{θ_i}(X_{≤t})
- Optional private signal: s_{i,t} = ρ z_{t+1} + sqrt(1−ρ²) ε_{i,t}, where z_{t+1} is the next exogenous return divided by its configured regime volatility and ε_{i,t} is agent-specific N(0,1) noise.
- Adjusted forecast: r̂'_{i,t+1} = r̂_{i,t+1} + β σ̂_t s_{i,t}. This adjustment is applied to every agent family before the common exposure rule.
- Target exposure: q*_{i,t} = q_i^max · tanh( r̂'_{i,t+1} / (k_i σ̂_t + ε) )
- Order: u_{i,t} = clip( q*_{i,t} − q_{i,t}, −u_i^max, u_i^max )

Optional constraints: position limits, execution costs, delay, max participation in volume.

`agents.private_signal` configures `strength` (ρ), `beta`, and a `rng_stream` index. Episode seed streams 0 and 1 remain the exogenous-path and population-construction streams; the private signal uses its configured independent stream (2 by default). At ρ = 0 signal generation is bypassed, exact zeros are recorded, and the round-3 decision path is retained bit-for-bit. Signal row t is aligned with the decision at t and uses the exogenous innovation entering r_{t+1}, making this deliberately private predictive information rather than a public causal feature.

Signals are serialized only in `PrivateRecord`, alongside per-agent positions, orders, risk state, and labels. `PublicRecord`, `PublicObservation`, the strict observer loader, and observer feature builders have no signal field.

Risk management is required. On breaching a drawdown or margin threshold, an agent reduces exposure to zero over `liquidation_steps` decisions. The completed schedule is followed by `cooldown_steps` decisions with zero target exposure; strategy orders resume after the cooldown and the drawdown peak resets to current equity at that point. A risk-overridden order is marked forced only when its quantity is nonzero. Without forced reduction, synchronized forced selling cannot appear.

## Market simulator

Phase A (first): aggregate price impact.

- Q_t = Σ_i u_{i,t} + Q_t^background
- r_{t+1} = r_{t+1}^exo + λ_t · g(Q_t / D_t)
- r^exo: exogenous shocks; D_t: available liquidity; λ_t: impact sensitivity; g: linear or the concave signed-power form sign(x)|x|^α with 0 < α < 1.
- Phase A execution convention: all agent orders at step t fill completely at the pre-return mid P_t; proportional transaction costs are deducted from cash, then post-fill positions are marked at P_{t+1}. Thus ΔE_i = q_{i,t+1}(P_{t+1}−P_t)−cost_i. This is a simulation convention, not a real-market execution claim.

Agents observe resulting prices and update, giving the feedback loop price decline → risk constraints → synchronized selling → more impact. This loop is possible, not guaranteed; the simulation decides.

Phase B (later): full limit order book with market makers, cancellations, queues, latency, depth. ABIDES / ABIDES-MARL as references.

## Observer inputs and targets

Public features at t: lagged returns r_t..r_{t−k}, σ̂_t, volume V_t, order-flow imbalance OFI_t, liquidity proxies. OFI is computed only from observable past executed trades, never from agents' intended orders.

Hidden labels from ground truth:

- Position alignment: C^pos_t = |Σ_i q_{i,t}| / Σ_i |q_{i,t}|, defined as 0 when all positions are zero.
- Flow synchronization: C^flow_t = |Σ_i u_{i,t}| / Σ_i |u_{i,t}|, defined as 0 when all orders are zero.
- Also rolling correlations and concentration of risk exposure.

These measure different things: similar positions without simultaneous trading, or same-direction trading from different positions.

Tasks:

| Task | Predicts | Difficulty |
| --- | --- | --- |
| A. State inference | Current C^pos_t | Medium (diagnostic) |
| B. Flow forecasting | Next h-period net agent buying/selling | Hard |
| C. Unwind forecasting | P(Y_{t,h}=1 \| X_{≤t}), large endogenous sell episode within h | Hardest (headline) |

Y_{t,h}=1 when `max(0, -Σ_{s=t+1}^{t+h} agent_flow_s) / Σ_{s=t+1}^{t+h} D_s` exceeds a fixed threshold. Only complete, strictly future windows are labeled. The threshold is a specified empirical quantile of these scores pooled from training episodes only (95th percentile by default); validation/test episodes never enter the fit. Future price declines are reported as a separate outcome.

The original window label above remains the default. Two alternative
private-ground-truth labels are available for diagnostics and ablations:

- **Onset:** first compute the same qualifying-window sequence. A qualifying
  window is positive only when no qualifying window was active at any of the
  preceding `h` prediction times. This removes repeated positives caused by
  overlapping future windows around one unwind; the score and training-only
  threshold are unchanged.
- **Forced flow:** replace `agent_flow_s` in the numerator with the sum of only
  negative agent orders marked as forced at step `s`. Strategy sells and
  forced buys contribute zero, so the score is gross forced selling divided by
  future liquidity. Forced flags and this derived score remain private and may
  be used only to construct targets, never observer inputs.

Models: logistic/linear baseline, LightGBM, GRU/temporal CNN; latent-state probabilistic sequence model only after baselines show signal.

## Real data

Start with Binance BTC/USDT then ETH/USDT spot (trades, aggTrades, klines), 1-minute bars, 6–12 months for calibration. Spot timestamps from January 2025 onward are in microseconds; ingestion must handle both units. LOBSTER samples later for order-book feature validation.

## Repository layout (planned)

```
herdlab/
  configs/      baseline.yaml, stress_test.yaml, generalization.yaml
  data/         ingest.py, features.py
  agents/       base.py, factor.py, learned.py, risk.py
  market/       simulator.py, impact.py, execution.py
  inference/    observer.py, targets.py, baselines.py
  experiments/  emergence.py, identification.py, generalization.py, stress.py
  evaluation/   metrics.py, plotting.py
tests/
```

## Technology

Python + NumPy/Polars (simulator deterministic given a seed), PyTorch, LightGBM, DuckDB + Parquet, Hydra or YAML configs, pytest, Streamlit/Plotly once experiments work. Single language; move the event loop to C++/Rust only if profiling justifies it.

Tests must cover position accounting, cash conservation under execution, timestamps, risk rules, and public/private data separation.
