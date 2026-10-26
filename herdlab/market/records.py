"""Separate public market observations from private simulator ground truth."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from herdlab.market.exogenous import ExogenousPath
    from herdlab.market.simulator import SimulationResult


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class PublicRecord:
    """Observer-safe output for one simulated episode.

    ``executed_imbalance`` is the signed flow of *all* simulated participants:
    agent fills plus background flow.  It must not be replaced with an
    agent-only aggregate.  Prices include the initial value, while all other
    arrays have one value per executed step.
    """

    prices: FloatArray
    returns: FloatArray
    volume: FloatArray
    executed_imbalance: FloatArray
    realized_volatility: FloatArray
    liquidity_proxy: FloatArray

    def __post_init__(self) -> None:
        step_arrays = (
            self.returns,
            self.volume,
            self.executed_imbalance,
            self.realized_volatility,
            self.liquidity_proxy,
        )
        _validate_float_arrays(step_arrays, ndim=1, label="public step arrays")
        n_steps = len(self.returns)
        if any(len(array) != n_steps for array in step_arrays):
            raise ValueError("all public step arrays must have equal length")
        if np.asarray(self.prices).ndim != 1 or len(self.prices) != n_steps + 1:
            raise ValueError("prices must be one-dimensional with n_steps + 1 values")
        if not np.all(np.isfinite(self.prices)) or np.any(self.prices <= 0.0):
            raise ValueError("prices must be finite and positive")
        if np.any(self.volume < 0.0):
            raise ValueError("volume must be nonnegative")
        if np.any(self.realized_volatility < 0.0):
            raise ValueError("realized_volatility must be nonnegative")
        if np.any(self.liquidity_proxy <= 0.0):
            raise ValueError("liquidity_proxy must be positive")


@dataclass(frozen=True)
class PrivateRecord:
    """Per-agent ground truth that must never enter the observer data path."""

    positions: FloatArray
    orders: FloatArray
    forced: BoolArray
    cash: FloatArray
    equity: FloatArray
    leverage: FloatArray
    transaction_costs: FloatArray
    agent_net_flow: FloatArray
    position_crowding: FloatArray
    flow_crowding: FloatArray

    def __post_init__(self) -> None:
        state_arrays = (self.positions, self.cash, self.equity)
        flow_arrays = (self.orders, self.transaction_costs)
        _validate_float_arrays(state_arrays, ndim=2, label="private state arrays")
        _validate_float_arrays(flow_arrays, ndim=2, label="private flow arrays")

        leverage = np.asarray(self.leverage)
        if leverage.ndim != 2 or np.any(np.isnan(leverage)) or np.any(leverage < 0.0):
            raise ValueError("leverage must be two-dimensional, nonnegative, and not NaN")

        n_steps, n_agents = self.orders.shape
        if n_steps == 0 or n_agents == 0:
            raise ValueError("private records must contain steps and agents")
        if any(array.shape != (n_steps, n_agents) for array in flow_arrays):
            raise ValueError("all private flow arrays must have shape (steps, agents)")
        if any(
            np.asarray(array).shape != (n_steps + 1, n_agents)
            for array in (*state_arrays, leverage)
        ):
            raise ValueError(
                "all private state arrays must have shape (steps + 1, agents)"
            )
        forced = np.asarray(self.forced)
        if forced.dtype != np.bool_ or forced.shape != (
            n_steps,
            n_agents,
        ):
            raise ValueError("forced must be boolean with shape (steps, agents)")
        if np.asarray(self.agent_net_flow).shape != (n_steps,):
            raise ValueError("agent_net_flow must have shape (steps,)")
        if not np.all(np.isfinite(self.agent_net_flow)):
            raise ValueError("agent_net_flow must be finite")
        if not np.allclose(self.agent_net_flow, self.orders.sum(axis=1)):
            raise ValueError("agent_net_flow must equal the sum of per-agent orders")
        if np.asarray(self.position_crowding).shape != (n_steps + 1,):
            raise ValueError("position_crowding must have shape (steps + 1,)")
        if np.asarray(self.flow_crowding).shape != (n_steps,):
            raise ValueError("flow_crowding must have shape (steps,)")
        for name, values in (
            ("position_crowding", self.position_crowding),
            ("flow_crowding", self.flow_crowding),
        ):
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if np.any(self.transaction_costs < 0.0):
            raise ValueError("transaction_costs must be nonnegative")


def split_simulation_records(
    result: SimulationResult,
    exogenous: ExogenousPath,
    *,
    realized_volatility_lookback: int,
) -> tuple[PublicRecord, PrivateRecord]:
    """Project an internal simulation result into public and private records.

    Realized volatility at step ``t`` uses returns strictly before ``t``,
    matching the causal value supplied to agents when they make that step's
    decision.
    """

    n_steps = len(result.returns)
    if realized_volatility_lookback <= 0:
        raise ValueError("realized_volatility_lookback must be positive")
    if len(exogenous.liquidity) != n_steps or len(exogenous.background_flow) != n_steps:
        raise ValueError("exogenous path length must match simulation result")

    realized_volatility = np.asarray(
        [
            _realized_volatility(result.returns[:step], realized_volatility_lookback)
            for step in range(n_steps)
        ],
        dtype=np.float64,
    )
    agent_net_flow = np.asarray(result.orders.sum(axis=1), dtype=np.float64)
    # Imported here to keep observer-safe record definitions independent of
    # private inference targets at module import time.
    from herdlab.inference.targets import flow_crowding, position_crowding

    executed_imbalance = agent_net_flow + np.asarray(
        exogenous.background_flow, dtype=np.float64
    )
    if not np.allclose(executed_imbalance, result.aggregate_flow):
        raise ValueError("aggregate flow is inconsistent with agent and background flow")

    public = PublicRecord(
        prices=np.asarray(result.prices, dtype=np.float64).copy(),
        returns=np.asarray(result.returns, dtype=np.float64).copy(),
        volume=np.asarray(result.volume, dtype=np.float64).copy(),
        executed_imbalance=executed_imbalance.copy(),
        realized_volatility=realized_volatility,
        liquidity_proxy=np.asarray(exogenous.liquidity, dtype=np.float64).copy(),
    )
    private = PrivateRecord(
        positions=np.asarray(result.positions, dtype=np.float64).copy(),
        orders=np.asarray(result.orders, dtype=np.float64).copy(),
        forced=np.asarray(result.forced, dtype=np.bool_).copy(),
        cash=np.asarray(result.cash, dtype=np.float64).copy(),
        equity=np.asarray(result.equity, dtype=np.float64).copy(),
        leverage=np.asarray(result.leverage, dtype=np.float64).copy(),
        transaction_costs=np.asarray(result.transaction_costs, dtype=np.float64).copy(),
        agent_net_flow=agent_net_flow.copy(),
        position_crowding=position_crowding(result.positions),
        flow_crowding=flow_crowding(result.orders),
    )
    return public, private


def public_field_names() -> tuple[str, ...]:
    """Return the complete serialized schema accepted by observer loaders."""

    return tuple(field.name for field in fields(PublicRecord))


def _validate_float_arrays(
    arrays: tuple[FloatArray, ...], *, ndim: int, label: str
) -> None:
    if any(np.asarray(array).ndim != ndim for array in arrays):
        raise ValueError(f"{label} must be {ndim}-dimensional")
    if any(not np.issubdtype(np.asarray(array).dtype, np.number) for array in arrays):
        raise ValueError(f"{label} must be numeric")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"{label} must be finite")


def _realized_volatility(returns: FloatArray, lookback: int) -> float:
    window = returns[-lookback:]
    if window.size < 2:
        return 0.0
    return float(np.std(window, ddof=1))


__all__ = [
    "PrivateRecord",
    "PublicRecord",
    "public_field_names",
    "split_simulation_records",
]
