"""Causal feature extraction from observer-safe public episode records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from herdlab.inference.observer_data import PublicRecordSource, load_public_record


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PublicFeatures:
    """A time-aligned public feature matrix.

    Row ``t`` is computed exclusively from public observations at indices no
    later than ``t``.  Early return lags with no available history are zero.
    """

    values: FloatArray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2 or values.shape[1] != len(self.names):
            raise ValueError("feature values must have one column per name")
        if not np.all(np.isfinite(values)):
            raise ValueError("feature values must be finite")


def build_public_features(
    source: PublicRecordSource, *, return_lookback: int = 10
) -> PublicFeatures:
    """Build public-only Task C features without looking beyond each row.

    The features are lagged returns (including the current observed return),
    current realized volatility, log volume, observable signed imbalance
    scaled by liquidity, log liquidity, the current liquidity change, and
    volume scaled by liquidity.  No private record type is accepted here.
    """

    if isinstance(return_lookback, bool) or not isinstance(
        return_lookback, (int, np.integer)
    ):
        raise TypeError("return_lookback must be an integer")
    lookback = int(return_lookback)
    if lookback <= 0:
        raise ValueError("return_lookback must be positive")

    public = load_public_record(source)
    returns = np.asarray(public.returns, dtype=np.float64)
    n_steps = len(returns)
    lagged = np.zeros((n_steps, lookback), dtype=np.float64)
    for lag in range(lookback):
        if lag < n_steps:
            lagged[lag:, lag] = returns[: n_steps - lag]

    volume = np.asarray(public.volume, dtype=np.float64)
    liquidity = np.asarray(public.liquidity_proxy, dtype=np.float64)
    imbalance = np.asarray(public.executed_imbalance, dtype=np.float64)
    liquidity_change = np.zeros(n_steps, dtype=np.float64)
    if n_steps > 1:
        liquidity_change[1:] = liquidity[1:] / liquidity[:-1] - 1.0

    current = np.column_stack(
        (
            np.asarray(public.realized_volatility, dtype=np.float64),
            np.log1p(volume),
            imbalance / liquidity,
            np.log(liquidity),
            liquidity_change,
            volume / liquidity,
        )
    )
    names = tuple(f"return_lag_{lag}" for lag in range(lookback)) + (
        "realized_volatility",
        "log1p_volume",
        "imbalance_to_liquidity",
        "log_liquidity",
        "liquidity_change",
        "volume_to_liquidity",
    )
    return PublicFeatures(values=np.column_stack((lagged, current)), names=names)


def build_strategy_replication_features(
    source: PublicRecordSource,
    config: Mapping[str, Any],
) -> PublicFeatures:
    """Reconstruct the configured baseline factor strategies from prices.

    Row ``t`` matches the strategy decision made at simulator step ``t``:
    only price returns completed before that step are used.  The two shadow
    strategies have the configured common (undispersed) parameters, start
    flat, and treat every clipped strategy order as filled.  Risk overrides
    are deliberately absent because they require private agent state.

    The returned columns contain momentum and mean-reversion target exposures
    and clipped order signals, followed by their causal one-step changes.
    Although ``source`` is a complete public record, prices are the only
    record field read by this feature builder.
    """

    public = load_public_record(source)
    lookback, volatility_lookback, q_max, k, order_limit, epsilon = (
        _strategy_replication_config(config)
    )
    prices = np.asarray(public.prices, dtype=np.float64)
    returns = np.diff(np.log(prices))
    n_steps = returns.size

    targets = np.zeros((n_steps, 2), dtype=np.float64)
    orders = np.zeros((n_steps, 2), dtype=np.float64)
    positions = np.zeros(2, dtype=np.float64)
    directions = np.asarray((1.0, -1.0), dtype=np.float64)

    for step in range(n_steps):
        history = returns[:step]
        signal_window = history[-lookback:]
        signal = 0.0 if signal_window.size == 0 else float(signal_window.mean())
        volatility_window = history[-volatility_lookback:]
        volatility = (
            0.0
            if volatility_window.size < 2
            else float(np.std(volatility_window, ddof=1))
        )
        forecasts = directions * signal
        targets[step] = q_max * np.tanh(forecasts / (k * volatility + epsilon))
        orders[step] = np.clip(
            targets[step] - positions, -order_limit, order_limit
        )
        positions += orders[step]

    signals = np.column_stack((targets, orders))
    changes = np.zeros_like(signals)
    if n_steps > 1:
        changes[1:] = np.diff(signals, axis=0)
    names = (
        "momentum_target_exposure",
        "reversion_target_exposure",
        "momentum_order_signal",
        "reversion_order_signal",
        "momentum_target_change",
        "reversion_target_change",
        "momentum_order_change",
        "reversion_order_change",
    )
    return PublicFeatures(values=np.column_stack((signals, changes)), names=names)


def _strategy_replication_config(
    config: Mapping[str, Any],
) -> tuple[int, int, float, float, float, float]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    agents = config.get("agents", config)
    if not isinstance(agents, Mapping):
        raise TypeError("config agents section must be a mapping")

    lookback = _positive_integer(agents.get("lookback"), "agents.lookback")
    simulation = config.get("simulation", {})
    if simulation and not isinstance(simulation, Mapping):
        raise TypeError("config simulation section must be a mapping")
    volatility_lookback = _positive_integer(
        simulation.get("realized_vol_lookback", lookback),
        "simulation.realized_vol_lookback",
    )
    q_max = _positive_float(agents.get("q_max"), "agents.q_max")
    k = _positive_float(agents.get("risk_aversion_k"), "agents.risk_aversion_k")
    order_limit = _positive_float(
        agents.get("order_limit"), "agents.order_limit"
    )
    epsilon = _positive_float(agents.get("epsilon", 1.0e-8), "agents.epsilon")
    return lookback, volatility_lookback, q_max, k, order_limit, epsilon


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


# A concise alias for callers that already operate in the inference package.
extract_features = build_public_features


__all__ = [
    "PublicFeatures",
    "build_public_features",
    "build_strategy_replication_features",
    "extract_features",
]
