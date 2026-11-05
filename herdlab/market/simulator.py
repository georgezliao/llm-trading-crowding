"""Single-asset aggregate-impact market simulator.

At step ``t`` agents observe public information available at price ``P_t`` and
submit orders ``u_{i,t}``.  Every agent order is filled completely at the
pre-return mid price ``P_t``.  Proportional transaction costs are deducted from
cash separately.  Aggregate signed flow then determines the next log return::

    Q_t = sum_i u_{i,t} + Q_background,t
    r_{t+1} = r_exogenous,t+1 + impact_sensitivity * g(Q_t / D_t)
    P_{t+1} = P_t * exp(r_{t+1})

Positions after the fills are marked to ``P_{t+1}``.  Consequently, for each
agent, ``E_{t+1} - E_t = q_{i,t+1} * (P_{t+1} - P_t) - costs_{i,t}``.
The pre-return fill convention is intentionally simple and should not be read
as a claim about real execution quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from herdlab.agents.base import Agent, PublicObservation
from herdlab.agents.private_signal import private_signal_config
from herdlab.agents.risk import RiskLimits, RiskManager
from herdlab.market.exogenous import ExogenousPath
from herdlab.market.impact import ImpactFunction, LinearImpact, SignedPowerImpact
from herdlab.market.records import PrivateRecord, PublicRecord, split_simulation_records


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SimulationResult:
    """Complete simulator state used to construct separate records later.

    State arrays (prices, positions, cash, equity, and leverage) include the
    initial state at index zero.  Flow arrays have one row per executed step.
    This combined internal result is not an observer-facing data interface.
    """

    prices: FloatArray
    returns: FloatArray
    fill_prices: FloatArray
    aggregate_flow: FloatArray
    volume: FloatArray
    orders: FloatArray
    forced: BoolArray
    transaction_costs: FloatArray
    positions: FloatArray
    cash: FloatArray
    equity: FloatArray
    leverage: FloatArray
    signals: FloatArray
    base_forecasts: FloatArray | None
    signal_terms: FloatArray


class MarketSimulator:
    """Run factor agents against a supplied exogenous path."""

    def __init__(
        self,
        config: Mapping[str, Any],
        agents: Sequence[Agent],
        *,
        impact_function: ImpactFunction | None = None,
        private_signals: FloatArray | None = None,
    ) -> None:
        if not agents:
            raise ValueError("agents must not be empty")
        self.config = config
        self.agents = list(agents)
        self.initial_price, self.initial_cash, self.cost_rate = _accounting_config(
            config
        )
        market = config["market"]
        self.impact_sensitivity = float(market["impact_sensitivity"])
        if not np.isfinite(self.impact_sensitivity) or self.impact_sensitivity < 0.0:
            raise ValueError("market.impact_sensitivity must be finite and nonnegative")
        self.impact_function = impact_function or _impact_from_config(market)
        self.private_signal = private_signal_config(config)
        self._private_signals = private_signals
        self.risk_enabled = _risk_enabled(config)
        if self.risk_enabled:
            limits = RiskLimits.from_config(config)
            self.risk_managers: list[RiskManager | None] = [
                RiskManager(limits) for _ in self.agents
            ]
        else:
            self.risk_managers = [None for _ in self.agents]
        self.volatility_lookback = int(config["simulation"]["realized_vol_lookback"])
        if self.volatility_lookback <= 0:
            raise ValueError("simulation.realized_vol_lookback must be positive")

    def run(self, exogenous: ExogenousPath) -> SimulationResult:
        """Execute all steps in ``exogenous`` and return full internal state."""

        n_steps = _validate_exogenous(exogenous)
        configured_steps = int(self.config["simulation"]["n_steps"])
        if configured_steps != n_steps:
            raise ValueError("exogenous path length must equal simulation.n_steps")

        n_agents = len(self.agents)
        signals = _validated_private_signals(
            self._private_signals,
            enabled=self.private_signal.enabled,
            n_steps=n_steps,
            n_agents=n_agents,
        )
        base_forecasts = np.full((n_steps, n_agents), np.nan)
        signal_terms = np.zeros((n_steps, n_agents), dtype=np.float64)
        prices = np.empty(n_steps + 1, dtype=np.float64)
        returns = np.empty(n_steps, dtype=np.float64)
        fill_prices = np.empty(n_steps, dtype=np.float64)
        aggregate_flow = np.empty(n_steps, dtype=np.float64)
        volume = np.empty(n_steps, dtype=np.float64)
        orders = np.empty((n_steps, n_agents), dtype=np.float64)
        forced = np.empty((n_steps, n_agents), dtype=np.bool_)
        costs = np.empty((n_steps, n_agents), dtype=np.float64)
        positions = np.empty((n_steps + 1, n_agents), dtype=np.float64)
        cash = np.empty((n_steps + 1, n_agents), dtype=np.float64)
        equity = np.empty((n_steps + 1, n_agents), dtype=np.float64)
        leverage = np.empty((n_steps + 1, n_agents), dtype=np.float64)

        prices[0] = self.initial_price
        positions[0] = np.asarray(
            [agent.position for agent in self.agents], dtype=np.float64
        )
        cash[0] = self.initial_cash
        equity[0] = cash[0] + positions[0] * prices[0]
        leverage[0] = _leverage_vector(positions[0], prices[0], equity[0])

        for step in range(n_steps):
            realized_volatility = _realized_volatility(
                returns[:step], self.volatility_lookback
            )
            previous_volume = 0.0 if step == 0 else volume[step - 1]
            previous_imbalance = 0.0 if step == 0 else aggregate_flow[step - 1]
            observation = PublicObservation(
                step=step,
                price=float(prices[step]),
                returns=returns[:step].copy(),
                realized_volatility=realized_volatility,
                volume=float(previous_volume),
                order_flow_imbalance=float(previous_imbalance),
                liquidity_proxy=float(exogenous.liquidity[step]),
            )

            for index, (agent, risk_manager) in enumerate(
                zip(self.agents, self.risk_managers, strict=True)
            ):
                # A custom decision override may not produce a base forecast.
                agent.last_base_forecast = None
                if not self.private_signal.enabled:
                    # Preserve the round-3 decision call exactly when private
                    # information is switched off.
                    strategy_order = agent.decide(observation)
                else:
                    adjustment = (
                        self.private_signal.beta
                        * realized_volatility
                        * signals[step, index]
                    )
                    signal_terms[step, index] = adjustment
                    strategy_order = agent.decide(
                        observation, forecast_adjustment=float(adjustment)
                    )
                if agent.last_base_forecast is not None:
                    base_forecasts[step, index] = agent.last_base_forecast
                if risk_manager is None:
                    orders[step, index] = strategy_order
                    forced[step, index] = False
                else:
                    decision = risk_manager.apply(
                        strategy_order,
                        position=float(positions[step, index]),
                        price=float(prices[step]),
                        equity=float(equity[step, index]),
                    )
                    orders[step, index] = decision.quantity
                    forced[step, index] = decision.forced

            fill_prices[step] = prices[step]
            costs[step] = self.cost_rate * np.abs(orders[step]) * fill_prices[step]
            cash[step + 1] = (
                cash[step] - orders[step] * fill_prices[step] - costs[step]
            )
            positions[step + 1] = positions[step] + orders[step]
            for agent, order in zip(self.agents, orders[step], strict=True):
                agent.apply_fill(float(order))

            aggregate_flow[step] = (
                float(orders[step].sum()) + exogenous.background_flow[step]
            )
            volume[step] = float(np.abs(orders[step]).sum()) + abs(
                exogenous.background_flow[step]
            )
            impact = self.impact_sensitivity * float(
                self.impact_function(
                    aggregate_flow[step] / exogenous.liquidity[step]
                )
            )
            returns[step] = exogenous.returns[step] + impact
            prices[step + 1] = prices[step] * np.exp(returns[step])
            if not np.isfinite(prices[step + 1]) or prices[step + 1] <= 0.0:
                raise FloatingPointError("simulated price became non-finite or non-positive")

            equity[step + 1] = cash[step + 1] + positions[step + 1] * prices[step + 1]
            leverage[step + 1] = _leverage_vector(
                positions[step + 1], prices[step + 1], equity[step + 1]
            )

        return SimulationResult(
            prices=prices,
            returns=returns,
            fill_prices=fill_prices,
            aggregate_flow=aggregate_flow,
            volume=volume,
            orders=orders,
            forced=forced,
            transaction_costs=costs,
            positions=positions,
            cash=cash,
            equity=equity,
            leverage=leverage,
            signals=signals,
            base_forecasts=(
                base_forecasts if np.all(np.isfinite(base_forecasts)) else None
            ),
            signal_terms=signal_terms,
        )

    def run_records(
        self, exogenous: ExogenousPath
    ) -> tuple[PublicRecord, PrivateRecord]:
        """Execute an episode and return its separated public/private outputs."""

        result = self.run(exogenous)
        return split_simulation_records(
            result,
            exogenous,
            realized_volatility_lookback=self.volatility_lookback,
        )


def simulate_market(
    config: Mapping[str, Any],
    agents: Sequence[Agent],
    exogenous: ExogenousPath,
    *,
    impact_function: ImpactFunction | None = None,
    private_signals: FloatArray | None = None,
) -> SimulationResult:
    """Convenience wrapper around :class:`MarketSimulator`."""

    return MarketSimulator(
        config,
        agents,
        impact_function=impact_function,
        private_signals=private_signals,
    ).run(exogenous)


def simulate_market_records(
    config: Mapping[str, Any],
    agents: Sequence[Agent],
    exogenous: ExogenousPath,
    *,
    impact_function: ImpactFunction | None = None,
    private_signals: FloatArray | None = None,
) -> tuple[PublicRecord, PrivateRecord]:
    """Run one episode and return only the explicitly separated records."""

    return MarketSimulator(
        config,
        agents,
        impact_function=impact_function,
        private_signals=private_signals,
    ).run_records(exogenous)


def _accounting_config(config: Mapping[str, Any]) -> tuple[float, float, float]:
    initial_price = float(config["simulation"]["initial_price"])
    initial_cash = float(config["agents"]["initial_cash"])
    cost_rate = float(config["agents"]["transaction_cost_rate"])
    if not np.isfinite(initial_price) or initial_price <= 0.0:
        raise ValueError("simulation.initial_price must be finite and positive")
    if not np.isfinite(initial_cash) or initial_cash <= 0.0:
        raise ValueError("agents.initial_cash must be finite and positive")
    if not np.isfinite(cost_rate) or cost_rate < 0.0:
        raise ValueError("agents.transaction_cost_rate must be finite and nonnegative")
    return initial_price, initial_cash, cost_rate


def _risk_enabled(config: Mapping[str, Any]) -> bool:
    risk = config["agents"]["risk"]
    enabled = risk.get("enabled", True)
    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("agents.risk.enabled must be a boolean")
    return bool(enabled)


def _impact_from_config(config: Mapping[str, Any]) -> ImpactFunction:
    model = str(config["impact_model"])
    if model == "linear":
        return LinearImpact()
    if model in {"signed_power", "concave"}:
        return SignedPowerImpact(float(config["impact_exponent"]))
    raise ValueError(f"unknown market.impact_model: {model}")


def _validate_exogenous(exogenous: ExogenousPath) -> int:
    arrays = (
        exogenous.returns,
        exogenous.regimes,
        exogenous.jump_shocks,
        exogenous.liquidity,
        exogenous.background_flow,
    )
    if any(np.asarray(values).ndim != 1 for values in arrays):
        raise ValueError("all exogenous path arrays must be one-dimensional")
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("all exogenous path arrays must have one equal positive length")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("all exogenous path values must be finite")
    if np.any(exogenous.liquidity <= 0.0):
        raise ValueError("exogenous liquidity must be positive")
    return next(iter(lengths))


def _validated_private_signals(
    signals: FloatArray | None,
    *,
    enabled: bool,
    n_steps: int,
    n_agents: int,
) -> FloatArray:
    expected_shape = (n_steps, n_agents)
    if signals is None:
        if enabled:
            raise ValueError(
                "private_signals are required when private signals are enabled"
            )
        return np.zeros(expected_shape, dtype=np.float64)

    values = np.asarray(signals, dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(f"private_signals must have shape {expected_shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("private_signals must be finite")
    if not enabled and np.any(values != 0.0):
        raise ValueError("private_signals must be zero when private signals are disabled")
    return values.copy()


def _realized_volatility(returns: FloatArray, lookback: int) -> float:
    window = returns[-lookback:]
    if window.size < 2:
        return 0.0
    return float(np.std(window, ddof=1))


def _leverage_vector(
    positions: FloatArray, price: float, equity: FloatArray
) -> FloatArray:
    gross_exposure = np.abs(positions) * price
    return np.divide(
        gross_exposure,
        equity,
        out=np.full_like(gross_exposure, np.inf),
        where=equity > 0.0,
    )


__all__ = [
    "MarketSimulator",
    "SimulationResult",
    "simulate_market",
    "simulate_market_records",
]
