"""Causal factor strategies and their heterogeneous population builder."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from herdlab.agents.base import Agent, PublicObservation


FloatArray = NDArray[np.float64]


class _FactorAgent(Agent):
    def __init__(
        self,
        *,
        lookback: int,
        lag_weights: FloatArray | None = None,
        **kwargs: float,
    ) -> None:
        super().__init__(**kwargs)
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        self.lookback = int(lookback)
        self.lag_weights = _validated_weights(lag_weights, self.lookback)

    def _signal(self, observation: PublicObservation) -> float:
        return _recent_return_signal(
            observation.returns, self.lookback, self.lag_weights
        )


class MomentumAgent(_FactorAgent):
    """Forecast continuation from a weighted mean of recent public returns."""

    def forecast(self, observation: PublicObservation) -> float:
        return self._signal(observation)


class MeanReversionAgent(_FactorAgent):
    """Forecast reversal of the recent weighted cumulative direction."""

    def forecast(self, observation: PublicObservation) -> float:
        return -self._signal(observation)


def build_factor_population(
    config: Mapping[str, Any], rng: np.random.Generator
) -> list[Agent]:
    """Build momentum and mean-reversion agents from configuration.

    ``parameter_dispersion`` is the standard deviation of centered log-normal
    perturbations to lookback, ``q_max``, ``risk_aversion_k``, and
    ``order_limit``.  Lag weights mix a population-shared draw with an
    agent-specific draw according to ``feature_overlap``: zero gives wholly
    agent-specific weights and one gives wholly shared weights.  When
    dispersion is zero, no random perturbations are applied, so agents within
    a family are exactly identical.

    The function accepts either the full project configuration or its
    ``agents`` section.  All randomness comes from the supplied generator.
    """

    agent_config = config["agents"] if "agents" in config else config
    population_size, momentum_count, dispersion, overlap = _validate_config(
        agent_config
    )

    base_lookback = int(agent_config["lookback"])
    base_q_max = float(agent_config["q_max"])
    base_k = float(agent_config["risk_aversion_k"])
    base_u_max = float(agent_config["order_limit"])
    epsilon = float(agent_config.get("epsilon", 1.0e-8))

    if dispersion == 0.0:
        shared_scores = np.ones(base_lookback, dtype=np.float64)
    else:
        shared_scores = rng.lognormal(
            mean=-0.5 * dispersion**2,
            sigma=dispersion,
            size=base_lookback,
        )

    population: list[Agent] = []
    for index in range(population_size):
        lookback = max(1, int(round(_disperse(base_lookback, dispersion, rng))))
        q_max = _disperse(base_q_max, dispersion, rng)
        k = _disperse(base_k, dispersion, rng)
        u_max = _disperse(base_u_max, dispersion, rng)
        weights = _lag_weights(lookback, shared_scores, overlap, dispersion, rng)

        agent_type = MomentumAgent if index < momentum_count else MeanReversionAgent
        population.append(
            agent_type(
                lookback=lookback,
                lag_weights=weights,
                q_max=q_max,
                k=k,
                u_max=u_max,
                epsilon=epsilon,
            )
        )
    return population


def _recent_return_signal(
    returns: FloatArray, lookback: int, lag_weights: FloatArray
) -> float:
    history = np.asarray(returns, dtype=np.float64)
    if history.ndim != 1:
        raise ValueError("observation returns must be one-dimensional")
    if not np.all(np.isfinite(history)):
        raise ValueError("observation returns must be finite")
    available = min(history.size, lookback)
    if available == 0:
        return 0.0
    weights = lag_weights[-available:]
    return float(np.dot(history[-available:], weights / weights.sum()))


def _validated_weights(weights: FloatArray | None, lookback: int) -> FloatArray:
    if weights is None:
        return np.full(lookback, 1.0 / lookback, dtype=np.float64)
    result = np.asarray(weights, dtype=np.float64)
    if result.shape != (lookback,):
        raise ValueError("lag_weights must have one value per lookback step")
    if np.any(result < 0.0) or not np.all(np.isfinite(result)):
        raise ValueError("lag_weights must be finite and nonnegative")
    if result.sum() <= 0.0:
        raise ValueError("lag_weights must have a positive sum")
    return result / result.sum()


def _disperse(base: float, dispersion: float, rng: np.random.Generator) -> float:
    if dispersion == 0.0:
        return float(base)
    return float(base * rng.lognormal(-0.5 * dispersion**2, dispersion))


def _lag_weights(
    lookback: int,
    shared_scores: FloatArray,
    overlap: float,
    dispersion: float,
    rng: np.random.Generator,
) -> FloatArray:
    if dispersion == 0.0:
        return np.full(lookback, 1.0 / lookback, dtype=np.float64)

    if lookback <= shared_scores.size:
        shared = shared_scores[-lookback:]
    else:
        shared = np.pad(
            shared_scores, (lookback - shared_scores.size, 0), mode="edge"
        )
    individual = rng.lognormal(
        mean=-0.5 * dispersion**2, sigma=dispersion, size=lookback
    )
    weights = overlap * shared + (1.0 - overlap) * individual
    return weights / weights.sum()


def _validate_config(
    config: Mapping[str, Any],
) -> tuple[int, int, float, float]:
    population_size = int(config["population_size"])
    momentum_fraction = float(config["momentum_fraction"])
    dispersion = float(config["parameter_dispersion"])
    overlap = float(config["feature_overlap"])

    if population_size <= 0:
        raise ValueError("agents.population_size must be positive")
    if not 0.0 <= momentum_fraction <= 1.0:
        raise ValueError("agents.momentum_fraction must be in [0, 1]")
    if dispersion < 0.0 or not np.isfinite(dispersion):
        raise ValueError("agents.parameter_dispersion must be finite and nonnegative")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("agents.feature_overlap must be in [0, 1]")

    for key in ("lookback", "q_max", "risk_aversion_k", "order_limit"):
        value = float(config[key])
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"agents.{key} must be finite and positive")
    epsilon = float(config.get("epsilon", 1.0e-8))
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("agents.epsilon must be finite and positive")

    momentum_count = int(round(population_size * momentum_fraction))
    return population_size, momentum_count, dispersion, overlap


__all__ = ["MeanReversionAgent", "MomentumAgent", "build_factor_population"]
