"""Per-episode diagnostics for simulated forced-deleveraging cascades.

This evaluation module is allowed to consume private simulator ground truth.
It is deliberately separate from the observer feature path.  Prices and total
executed absolute flow come from the matching public record because those
quantities are not present in :class:`~herdlab.market.records.PrivateRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from herdlab.market.records import PrivateRecord, PublicRecord


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CascadeDiagnostics:
    """Summary of forced reductions and crowding in one simulated episode."""

    fraction_steps_with_forced_order: float
    max_simultaneously_forced_agents: int
    mean_simultaneously_forced_agents: float
    forced_share_of_absolute_agent_flow: float
    agent_flow_share_of_total_absolute_flow: float
    price_max_drawdown: float
    position_crowding_quantiles: Mapping[float, float]
    flow_crowding_quantiles: Mapping[float, float]
    fraction_positive_windows_with_forced_flow: float


def episode_cascade_diagnostics(
    public: PublicRecord,
    private: PrivateRecord,
    labels: ArrayLike,
    *,
    horizon: int,
    quantiles: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> CascadeDiagnostics:
    """Compute cascade diagnostics for one episode.

    ``labels[t]`` is interpreted using HerdLab's existing unwind-target
    convention: it describes the strictly future window ``t+1`` through
    ``t+horizon``.  A positive window contains forced flow when at least one
    order in that window is both marked forced and nonzero.  If there are no
    positive windows, the corresponding fraction is defined as zero.

    Absolute total flow is ``public.volume``.  Under the Phase A simulator
    convention this is gross absolute agent flow plus absolute background
    flow, avoiding cancellation between buyers and sellers.
    """

    n_steps = len(private.orders)
    if len(public.returns) != n_steps:
        raise ValueError("public and private records must have the same step count")
    h = _validate_horizon(horizon, n_steps)
    binary_labels = _validate_labels(labels, expected_length=n_steps - h)
    probabilities = _validate_quantiles(quantiles)

    forced_counts = np.count_nonzero(private.forced, axis=1)
    forced_flow = np.where(private.forced, np.abs(private.orders), 0.0)
    absolute_agent_flow = np.abs(private.orders)

    forced_step_mask = forced_counts > 0
    positive_window_forced = _positive_windows_with_forced_flow(
        binary_labels,
        np.any((private.forced) & (absolute_agent_flow > 0.0), axis=1),
        horizon=h,
    )

    return CascadeDiagnostics(
        fraction_steps_with_forced_order=float(np.mean(forced_step_mask)),
        max_simultaneously_forced_agents=int(np.max(forced_counts)),
        mean_simultaneously_forced_agents=float(np.mean(forced_counts)),
        forced_share_of_absolute_agent_flow=_safe_share(
            float(forced_flow.sum()), float(absolute_agent_flow.sum())
        ),
        agent_flow_share_of_total_absolute_flow=_safe_share(
            float(absolute_agent_flow.sum()), float(np.sum(public.volume))
        ),
        price_max_drawdown=max_drawdown(public.prices),
        position_crowding_quantiles=_quantile_mapping(
            private.position_crowding, probabilities
        ),
        flow_crowding_quantiles=_quantile_mapping(
            private.flow_crowding, probabilities
        ),
        fraction_positive_windows_with_forced_flow=(
            float(np.mean(positive_window_forced))
            if positive_window_forced.size
            else 0.0
        ),
    )


def max_drawdown(prices: ArrayLike) -> float:
    """Return the largest peak-to-subsequent-trough fractional price loss."""

    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("prices must be a nonempty vector")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("prices must be finite and positive")
    running_peak = np.maximum.accumulate(values)
    return float(np.max((running_peak - values) / running_peak))


def _positive_windows_with_forced_flow(
    labels: NDArray[np.bool_], forced_flow_steps: NDArray[np.bool_], *, horizon: int
) -> NDArray[np.bool_]:
    positive_indices = np.flatnonzero(labels)
    return np.asarray(
        [
            np.any(forced_flow_steps[index + 1 : index + horizon + 1])
            for index in positive_indices
        ],
        dtype=np.bool_,
    )


def _safe_share(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def _quantile_mapping(
    values: ArrayLike, probabilities: tuple[float, ...]
) -> Mapping[float, float]:
    estimates = np.quantile(np.asarray(values, dtype=np.float64), probabilities)
    return {
        probability: float(estimate)
        for probability, estimate in zip(probabilities, estimates, strict=True)
    }


def _validate_horizon(horizon: int, n_steps: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be an integer")
    value = int(horizon)
    if value <= 0 or value >= n_steps:
        raise ValueError("horizon must be positive and smaller than episode length")
    return value


def _validate_labels(
    labels: ArrayLike, *, expected_length: int
) -> NDArray[np.bool_]:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(
            "labels must cover every complete strictly-future horizon window"
        )
    if not np.all(np.isin(values, (0, 1, False, True))):
        raise ValueError("labels must be binary")
    return np.asarray(values, dtype=np.bool_)


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    probabilities = tuple(float(value) for value in quantiles)
    if not probabilities:
        raise ValueError("quantiles must not be empty")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("quantiles must be finite and in [0, 1]")
    if len(set(probabilities)) != len(probabilities):
        raise ValueError("quantiles must not contain duplicates")
    return probabilities


# A concise alias for callers that already establish episode context.
cascade_diagnostics = episode_cascade_diagnostics


__all__ = [
    "CascadeDiagnostics",
    "cascade_diagnostics",
    "episode_cascade_diagnostics",
    "max_drawdown",
]
