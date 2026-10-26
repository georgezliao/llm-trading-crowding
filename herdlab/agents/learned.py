"""Causal learned trading agents and mixed-population construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from herdlab.agents.base import Agent, PublicObservation
from herdlab.agents.factor import build_factor_population


FloatArray = NDArray[np.float64]


class LinearMLAgent(Agent):
    """Ridge return forecaster fitted on causal lagged-return examples.

    At decision step ``t``, ``observation.returns`` contains returns strictly
    before the order at ``t``.  A training row predicting return ``r_j`` uses
    only ``[r_{j-lookback}, ..., r_{j-1}]``.  The live forecast uses the most
    recent completed lag window.  Thus neither fitting nor prediction reads a
    return from the future of its target.

    The ridge model is refitted at most every ``refit_interval`` decision
    steps.  Only the most recent ``training_window`` eligible examples are
    used.  Before ``min_samples`` examples exist, the forecast is zero.
    """

    def __init__(
        self,
        *,
        lookback: int,
        ridge_alpha: float,
        refit_interval: int,
        training_window: int,
        min_samples: int,
        **kwargs: float,
    ) -> None:
        super().__init__(**kwargs)
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be finite and nonnegative")
        if refit_interval <= 0:
            raise ValueError("refit_interval must be positive")
        if training_window <= 0:
            raise ValueError("training_window must be positive")
        if min_samples <= 0 or min_samples > training_window:
            raise ValueError("min_samples must be in [1, training_window]")

        self.lookback = int(lookback)
        self.ridge_alpha = float(ridge_alpha)
        self.refit_interval = int(refit_interval)
        self.training_window = int(training_window)
        self.min_samples = int(min_samples)
        self._coefficients = np.zeros(self.lookback, dtype=np.float64)
        self._intercept = 0.0
        self._last_fit_step: int | None = None
        self._fit_sample_count = 0

    @property
    def coefficients(self) -> FloatArray:
        """Return a copy of the current ridge coefficients."""

        return self._coefficients.copy()

    @property
    def intercept(self) -> float:
        return self._intercept

    @property
    def last_fit_step(self) -> int | None:
        return self._last_fit_step

    @property
    def fit_sample_count(self) -> int:
        return self._fit_sample_count

    def forecast(self, observation: PublicObservation) -> float:
        history = _validated_history(observation.returns)
        features, targets = _lagged_examples(history, self.lookback)
        should_refit = (
            targets.size >= self.min_samples
            and (
                self._last_fit_step is None
                or observation.step - self._last_fit_step >= self.refit_interval
            )
        )
        if should_refit:
            self._fit(features[-self.training_window :], targets[-self.training_window :])
            self._last_fit_step = int(observation.step)

        if history.size < self.lookback or self._last_fit_step is None:
            return 0.0
        current = history[-self.lookback :]
        return float(self._intercept + current @ self._coefficients)

    def _fit(self, features: FloatArray, targets: FloatArray) -> None:
        feature_mean = features.mean(axis=0)
        target_mean = float(targets.mean())
        centered_features = features - feature_mean
        centered_targets = targets - target_mean
        gram = centered_features.T @ centered_features
        penalty = self.ridge_alpha * np.eye(self.lookback, dtype=np.float64)
        rhs = centered_features.T @ centered_targets
        try:
            coefficients = np.linalg.solve(gram + penalty, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(gram + penalty) @ rhs
        self._coefficients = np.asarray(coefficients, dtype=np.float64)
        self._intercept = float(target_mean - feature_mean @ self._coefficients)
        self._fit_sample_count = int(targets.size)


def build_linear_population(
    config: Mapping[str, Any], rng: np.random.Generator
) -> list[LinearMLAgent]:
    """Build the configured linear-agent family using the supplied RNG."""

    agents = config["agents"] if "agents" in config else config
    learned = agents.get("linear", {})
    count = int(learned.get("count", 0))
    if count < 0:
        raise ValueError("agents.linear.count must be nonnegative")

    dispersion = float(agents["parameter_dispersion"])
    if dispersion < 0.0 or not np.isfinite(dispersion):
        raise ValueError("agents.parameter_dispersion must be finite and nonnegative")

    population: list[LinearMLAgent] = []
    for _ in range(count):
        population.append(
            LinearMLAgent(
                lookback=int(learned["lookback"]),
                ridge_alpha=float(learned["ridge_alpha"]),
                refit_interval=int(learned["refit_interval"]),
                training_window=int(learned["training_window"]),
                min_samples=int(learned["min_samples"]),
                q_max=_disperse(float(agents["q_max"]), dispersion, rng),
                k=_disperse(float(agents["risk_aversion_k"]), dispersion, rng),
                u_max=_disperse(float(agents["order_limit"]), dispersion, rng),
                epsilon=float(agents.get("epsilon", 1.0e-8)),
            )
        )
    return population


def build_population(config: Mapping[str, Any], rng: np.random.Generator) -> list[Agent]:
    """Build factor and linear agents while preserving total population size."""

    agents = config["agents"] if "agents" in config else config
    total = int(agents["population_size"])
    linear_count = int(agents.get("linear", {}).get("count", 0))
    factor_count = total - linear_count
    if factor_count < 0:
        raise ValueError("agents.linear.count cannot exceed agents.population_size")

    factor_config = deepcopy(dict(config))
    if "agents" in config:
        factor_config["agents"] = deepcopy(dict(agents))
        factor_config["agents"]["population_size"] = factor_count
    else:
        factor_config["population_size"] = factor_count

    population: list[Agent] = []
    if factor_count:
        population.extend(build_factor_population(factor_config, rng))
    population.extend(build_linear_population(config, rng))
    return population


def _validated_history(returns: FloatArray) -> FloatArray:
    history = np.asarray(returns, dtype=np.float64)
    if history.ndim != 1:
        raise ValueError("observation returns must be one-dimensional")
    if not np.all(np.isfinite(history)):
        raise ValueError("observation returns must be finite")
    return history


def _lagged_examples(history: FloatArray, lookback: int) -> tuple[FloatArray, FloatArray]:
    sample_count = max(0, history.size - lookback)
    features = np.empty((sample_count, lookback), dtype=np.float64)
    for row, target_index in enumerate(range(lookback, history.size)):
        features[row] = history[target_index - lookback : target_index]
    return features, history[lookback:].copy()


def _disperse(base: float, dispersion: float, rng: np.random.Generator) -> float:
    if dispersion == 0.0:
        return base
    return float(base * rng.lognormal(-0.5 * dispersion**2, dispersion))


__all__ = ["LinearMLAgent", "build_linear_population", "build_population"]
