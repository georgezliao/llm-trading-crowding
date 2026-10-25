"""Exogenous market-state processes used by the synthetic simulator.

The processes in this module depend only on their configuration and the
``numpy.random.Generator`` supplied by the caller.  In particular, they do not
consume agent state or agent orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ExogenousPath:
    """A realized path of public market drivers.

    Every array has one value per simulation step.  ``returns`` contains the
    regime-volatility innovation plus ``jump_shocks``.  Liquidity is generated
    as a floored log-AR(1) process, while background flow is a zero-mean AR(1)
    process.
    """

    returns: FloatArray
    regimes: IntArray
    jump_shocks: FloatArray
    liquidity: FloatArray
    background_flow: FloatArray


def generate_exogenous_path(
    config: Mapping[str, Any], rng: np.random.Generator
) -> ExogenousPath:
    """Generate all exogenous drivers for one simulation episode.

    Args:
        config: Full HerdLab configuration, including ``simulation.n_steps``
            and the ``exogenous`` section from ``configs/baseline.yaml``.
        rng: Explicit random-number generator owned by the caller.

    Returns:
        An :class:`ExogenousPath` whose arrays all have shape ``(n_steps,)``.

    The regime recorded at step ``t`` determines that step's volatility.  A
    transition is then drawn for step ``t + 1``.  Jump shocks are independent,
    zero-mean Gaussian innovations gated by the configured jump probability.
    """

    n_steps, exogenous = _validated_config(config)
    transition = np.asarray(exogenous["transition_matrix"], dtype=np.float64)
    volatility = np.asarray(exogenous["regime_volatility"], dtype=np.float64)

    regimes = np.empty(n_steps, dtype=np.int64)
    regimes[0] = int(exogenous["initial_regime"])
    for step in range(1, n_steps):
        previous = regimes[step - 1]
        regimes[step] = int(rng.random() >= transition[previous, 0])

    gaussian_returns = rng.normal(0.0, volatility[regimes], size=n_steps)
    jump_mask = rng.random(n_steps) < float(exogenous["jump_probability"])
    jump_shocks = np.zeros(n_steps, dtype=np.float64)
    jump_shocks[jump_mask] = rng.normal(
        0.0, float(exogenous["jump_scale"]), size=int(jump_mask.sum())
    )

    liquidity_config = exogenous["liquidity"]
    liquidity = _liquidity_path(n_steps, liquidity_config, rng)

    flow_config = exogenous["background_flow"]
    background_flow = _background_flow_path(n_steps, flow_config, rng)

    return ExogenousPath(
        returns=gaussian_returns + jump_shocks,
        regimes=regimes,
        jump_shocks=jump_shocks,
        liquidity=liquidity,
        background_flow=background_flow,
    )


def _liquidity_path(
    n_steps: int, config: Mapping[str, Any], rng: np.random.Generator
) -> FloatArray:
    mean = float(config["mean"])
    persistence = float(config["persistence"])
    innovation_scale = float(config["innovation_scale"])
    minimum = float(config["minimum"])

    log_deviation = np.empty(n_steps, dtype=np.float64)
    log_deviation[0] = rng.normal(0.0, innovation_scale)
    for step in range(1, n_steps):
        log_deviation[step] = (
            persistence * log_deviation[step - 1]
            + rng.normal(0.0, innovation_scale)
        )
    return np.maximum(minimum, mean * np.exp(log_deviation))


def _background_flow_path(
    n_steps: int, config: Mapping[str, Any], rng: np.random.Generator
) -> FloatArray:
    scale = float(config["scale"])
    persistence = float(config["persistence"])

    flow = np.empty(n_steps, dtype=np.float64)
    flow[0] = rng.normal(0.0, scale)
    for step in range(1, n_steps):
        flow[step] = persistence * flow[step - 1] + rng.normal(0.0, scale)
    return flow


def _validated_config(
    config: Mapping[str, Any],
) -> tuple[int, Mapping[str, Any]]:
    """Extract and validate the parameters required by this module."""

    simulation = config["simulation"]
    exogenous = config["exogenous"]
    n_steps = int(simulation["n_steps"])
    if n_steps <= 0:
        raise ValueError("simulation.n_steps must be positive")

    transition = np.asarray(exogenous["transition_matrix"], dtype=np.float64)
    if transition.shape != (2, 2):
        raise ValueError("exogenous.transition_matrix must have shape (2, 2)")
    if np.any(transition < 0.0) or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("each transition-matrix row must be nonnegative and sum to 1")

    volatility = np.asarray(exogenous["regime_volatility"], dtype=np.float64)
    if volatility.shape != (2,) or np.any(volatility < 0.0):
        raise ValueError("exogenous.regime_volatility must contain two nonnegative values")
    if int(exogenous["initial_regime"]) not in (0, 1):
        raise ValueError("exogenous.initial_regime must be 0 or 1")

    jump_probability = float(exogenous["jump_probability"])
    jump_scale = float(exogenous["jump_scale"])
    if not 0.0 <= jump_probability <= 1.0:
        raise ValueError("exogenous.jump_probability must be in [0, 1]")
    if jump_scale < 0.0:
        raise ValueError("exogenous.jump_scale must be nonnegative")

    liquidity = exogenous["liquidity"]
    if float(liquidity["mean"]) <= 0.0 or float(liquidity["minimum"]) <= 0.0:
        raise ValueError("liquidity mean and minimum must be positive")
    if not 0.0 <= float(liquidity["persistence"]) < 1.0:
        raise ValueError("liquidity persistence must be in [0, 1)")
    if float(liquidity["innovation_scale"]) < 0.0:
        raise ValueError("liquidity innovation_scale must be nonnegative")

    background_flow = exogenous["background_flow"]
    if abs(float(background_flow["persistence"])) >= 1.0:
        raise ValueError("background-flow persistence must have magnitude below 1")
    if float(background_flow["scale"]) < 0.0:
        raise ValueError("background-flow scale must be nonnegative")

    return n_steps, exogenous
