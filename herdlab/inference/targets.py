"""Private ground-truth targets for crowding and unwind prediction.

This module consumes simulator-private positions and orders.  It is deliberately
separate from :mod:`herdlab.inference.observer_data`, whose inputs are restricted
to public records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class UnwindEpisode:
    """Private agent flow and matching public liquidity for one episode."""

    agent_net_flow: ArrayLike
    liquidity: ArrayLike


UnwindEpisodeLike: TypeAlias = UnwindEpisode | tuple[ArrayLike, ArrayLike]


def position_crowding(positions: ArrayLike) -> FloatArray:
    """Return position alignment ``|sum(q_i)| / sum(|q_i|)`` by time.

    A row with no exposure has alignment zero.  The final axis is interpreted
    as agents, so a matrix shaped ``(time, agents)`` produces one value per
    time step.
    """

    return _alignment(positions, name="positions")


def flow_crowding(orders: ArrayLike) -> FloatArray:
    """Return signed-flow synchronization by time, with zero for no flow."""

    return _alignment(orders, name="orders")


def agent_net_flow(orders: ArrayLike) -> FloatArray:
    """Sum per-agent signed orders without adding background market flow."""

    values = _agent_matrix(orders, name="orders")
    return np.asarray(values.sum(axis=-1), dtype=np.float64)


def unwind_scores(
    agent_flow: ArrayLike,
    liquidity: ArrayLike,
    *,
    horizon: int,
) -> FloatArray:
    """Compute future agent net-selling intensity for complete windows.

    For each valid time ``t``, the score is

    ``max(0, -sum(flow[t+1:t+h+1])) / sum(liquidity[t+1:t+h+1])``.

    Thus the current step is excluded, preserving the forecasting boundary,
    and the returned array has length ``n_steps - horizon``.  Buying within a
    window offsets selling because the target is *net* selling.
    """

    flow, depth = _flow_and_liquidity(agent_flow, liquidity)
    h = _validate_horizon(horizon, len(flow))
    flow_sums = _future_window_sums(flow, h)
    liquidity_sums = _future_window_sums(depth, h)
    return np.maximum(0.0, -flow_sums) / liquidity_sums


def unwind_labels(
    agent_flow: ArrayLike,
    liquidity: ArrayLike,
    *,
    horizon: int,
    threshold: float,
) -> BoolArray:
    """Return ``Y[t,h]`` for complete future windows at a fixed threshold."""

    cutoff = float(threshold)
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("threshold must be finite and nonnegative")
    return np.asarray(
        unwind_scores(agent_flow, liquidity, horizon=horizon) > cutoff,
        dtype=np.bool_,
    )


def fit_unwind_threshold(
    training_episodes: Iterable[UnwindEpisodeLike],
    *,
    horizon: int,
    quantile: float = 0.95,
) -> float:
    """Fit a score quantile using only the supplied training episodes.

    Each item is either :class:`UnwindEpisode` or an ``(agent_flow,
    liquidity)`` pair.  Callers must pass only training episodes; requiring the
    episode collection explicitly prevents this helper from consulting test
    data or module-level state.
    """

    probability = float(quantile)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile must be between zero and one")

    episode_scores: list[FloatArray] = []
    for episode in training_episodes:
        if isinstance(episode, UnwindEpisode):
            flow, liquidity = episode.agent_net_flow, episode.liquidity
        else:
            try:
                flow, liquidity = episode
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "each training episode must be UnwindEpisode or a "
                    "(agent_flow, liquidity) pair"
                ) from error
        episode_scores.append(unwind_scores(flow, liquidity, horizon=horizon))

    if not episode_scores:
        raise ValueError("training_episodes must not be empty")
    scores = np.concatenate(episode_scores)
    return float(np.quantile(scores, probability))


def _alignment(values: ArrayLike, *, name: str) -> FloatArray:
    matrix = _agent_matrix(values, name=name)
    gross = np.abs(matrix).sum(axis=-1)
    aligned = np.divide(
        np.abs(matrix.sum(axis=-1)),
        gross,
        out=np.zeros_like(gross, dtype=np.float64),
        where=gross > 0.0,
    )
    # Guard against tiny floating-point excursions above one.
    return np.asarray(np.clip(aligned, 0.0, 1.0), dtype=np.float64)


def _agent_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim < 1 or matrix.shape[-1] == 0:
        raise ValueError(f"{name} must have a nonempty final agent axis")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _flow_and_liquidity(
    agent_flow: ArrayLike, liquidity: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    flow = np.asarray(agent_flow, dtype=np.float64)
    depth = np.asarray(liquidity, dtype=np.float64)
    if flow.ndim != 1 or depth.ndim != 1 or flow.shape != depth.shape:
        raise ValueError("agent_flow and liquidity must be equal-length vectors")
    if not np.all(np.isfinite(flow)) or not np.all(np.isfinite(depth)):
        raise ValueError("agent_flow and liquidity must be finite")
    if np.any(depth <= 0.0):
        raise ValueError("liquidity must be positive")
    return flow, depth


def _validate_horizon(horizon: int, n_steps: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be an integer")
    h = int(horizon)
    if h <= 0 or h >= n_steps:
        raise ValueError("horizon must be positive and smaller than episode length")
    return h


def _future_window_sums(values: FloatArray, horizon: int) -> FloatArray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    starts = np.arange(1, len(values) - horizon + 1)
    return cumulative[starts + horizon] - cumulative[starts]


# Explicit compute_* aliases make call sites read naturally while retaining the
# concise names used in the architecture.
compute_position_crowding = position_crowding
compute_flow_crowding = flow_crowding
compute_agent_net_flow = agent_net_flow
compute_unwind_labels = unwind_labels


__all__ = [
    "UnwindEpisode",
    "agent_net_flow",
    "compute_agent_net_flow",
    "compute_flow_crowding",
    "compute_position_crowding",
    "compute_unwind_labels",
    "fit_unwind_threshold",
    "flow_crowding",
    "position_crowding",
    "unwind_labels",
    "unwind_scores",
]
