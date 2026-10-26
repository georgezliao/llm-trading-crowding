"""Causal feature extraction from observer-safe public episode records."""

from __future__ import annotations

from dataclasses import dataclass

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


# A concise alias for callers that already operate in the inference package.
extract_features = build_public_features


__all__ = ["PublicFeatures", "build_public_features", "extract_features"]
