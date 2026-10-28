"""Causal learned trading agents and mixed-population construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from herdlab.agents.base import Agent, PublicObservation
from herdlab.agents.factor import (
    _lag_weights,
    _validated_weights,
    build_factor_population,
)
from herdlab.agents.overlap import learned_overlap_mode


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
        lag_weights: FloatArray | None = None,
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
        self.lag_weights = (
            None
            if lag_weights is None
            else _validated_weights(lag_weights, self.lookback)
        )
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
        features, targets = _lagged_examples(
            history, self.lookback, self.lag_weights
        )
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
        current = _weighted_lag_inputs(
            history[-self.lookback :], self.lag_weights
        )
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


class LightGBMAgent(Agent):
    """Small LightGBM return forecaster with causal rolling refits."""

    def __init__(
        self,
        *,
        lookback: int,
        refit_interval: int,
        training_window: int,
        min_samples: int,
        n_estimators: int,
        learning_rate: float,
        num_leaves: int,
        random_seed: int,
        lag_weights: FloatArray | None = None,
        **kwargs: float,
    ) -> None:
        super().__init__(**kwargs)
        _validate_rolling_parameters(
            lookback, refit_interval, training_window, min_samples
        )
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if num_leaves < 2:
            raise ValueError("num_leaves must be at least two")

        self.lookback = int(lookback)
        self.refit_interval = int(refit_interval)
        self.training_window = int(training_window)
        self.min_samples = int(min_samples)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.random_seed = int(random_seed)
        self.lag_weights = (
            None
            if lag_weights is None
            else _validated_weights(lag_weights, self.lookback)
        )
        self._model: lgb.LGBMRegressor | None = None
        self._last_fit_step: int | None = None
        self._fit_sample_count = 0

    @property
    def last_fit_step(self) -> int | None:
        return self._last_fit_step

    @property
    def fit_sample_count(self) -> int:
        return self._fit_sample_count

    def forecast(self, observation: PublicObservation) -> float:
        history = _validated_history(observation.returns)
        features, targets = _lagged_examples(
            history, self.lookback, self.lag_weights
        )
        if _should_refit(
            targets.size,
            self.min_samples,
            observation.step,
            self._last_fit_step,
            self.refit_interval,
        ):
            self._fit(
                features[-self.training_window :], targets[-self.training_window :]
            )
            self._last_fit_step = int(observation.step)

        if history.size < self.lookback or self._model is None:
            return 0.0
        current = _weighted_lag_inputs(
            history[-self.lookback :], self.lag_weights
        )
        prediction = self._model.predict(current.reshape(1, -1))
        return float(prediction[0])

    def _fit(self, features: FloatArray, targets: FloatArray) -> None:
        self._model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=3,
            min_child_samples=5,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0e-4,
            random_state=self.random_seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        self._model.fit(features, targets)
        self._fit_sample_count = int(targets.size)


class MLPAgent(Agent):
    """One-hidden-layer NumPy MLP trained by deterministic full-batch descent."""

    def __init__(
        self,
        *,
        lookback: int,
        refit_interval: int,
        training_window: int,
        min_samples: int,
        hidden_size: int,
        epochs: int,
        learning_rate: float,
        l2: float,
        rng: np.random.Generator,
        lag_weights: FloatArray | None = None,
        **kwargs: float,
    ) -> None:
        super().__init__(**kwargs)
        _validate_rolling_parameters(
            lookback, refit_interval, training_window, min_samples
        )
        if hidden_size <= 0 or epochs <= 0:
            raise ValueError("hidden_size and epochs must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(l2) or l2 < 0.0:
            raise ValueError("l2 must be finite and nonnegative")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

        self.lookback = int(lookback)
        self.refit_interval = int(refit_interval)
        self.training_window = int(training_window)
        self.min_samples = int(min_samples)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.lag_weights = (
            None
            if lag_weights is None
            else _validated_weights(lag_weights, self.lookback)
        )
        self._rng = rng
        self._weights_1: FloatArray | None = None
        self._bias_1: FloatArray | None = None
        self._weights_2: FloatArray | None = None
        self._bias_2 = 0.0
        self._feature_mean = np.zeros(self.lookback, dtype=np.float64)
        self._feature_scale = np.ones(self.lookback, dtype=np.float64)
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._last_fit_step: int | None = None
        self._fit_sample_count = 0

    @property
    def last_fit_step(self) -> int | None:
        return self._last_fit_step

    @property
    def fit_sample_count(self) -> int:
        return self._fit_sample_count

    def forecast(self, observation: PublicObservation) -> float:
        history = _validated_history(observation.returns)
        features, targets = _lagged_examples(
            history, self.lookback, self.lag_weights
        )
        if _should_refit(
            targets.size,
            self.min_samples,
            observation.step,
            self._last_fit_step,
            self.refit_interval,
        ):
            self._fit(
                features[-self.training_window :], targets[-self.training_window :]
            )
            self._last_fit_step = int(observation.step)

        if history.size < self.lookback or self._weights_1 is None:
            return 0.0
        current_lags = _weighted_lag_inputs(
            history[-self.lookback :], self.lag_weights
        )
        current = (current_lags - self._feature_mean) / self._feature_scale
        hidden = np.tanh(current @ self._weights_1 + self._bias_1)
        scaled_prediction = float(hidden @ self._weights_2 + self._bias_2)
        return self._target_mean + self._target_scale * scaled_prediction

    def _fit(self, features: FloatArray, targets: FloatArray) -> None:
        self._feature_mean = features.mean(axis=0)
        feature_std = features.std(axis=0)
        self._feature_scale = np.where(feature_std > 1.0e-12, feature_std, 1.0)
        self._target_mean = float(targets.mean())
        target_std = float(targets.std())
        self._target_scale = target_std if target_std > 1.0e-12 else 1.0
        x = (features - self._feature_mean) / self._feature_scale
        y = (targets - self._target_mean) / self._target_scale

        limit_1 = np.sqrt(6.0 / (self.lookback + self.hidden_size))
        weights_1 = self._rng.uniform(
            -limit_1, limit_1, size=(self.lookback, self.hidden_size)
        )
        bias_1 = np.zeros(self.hidden_size, dtype=np.float64)
        limit_2 = np.sqrt(6.0 / (self.hidden_size + 1))
        weights_2 = self._rng.uniform(-limit_2, limit_2, size=self.hidden_size)
        bias_2 = 0.0

        sample_count = targets.size
        for _ in range(self.epochs):
            hidden = np.tanh(x @ weights_1 + bias_1)
            predictions = hidden @ weights_2 + bias_2
            error = predictions - y
            gradient_2 = hidden.T @ error / sample_count + self.l2 * weights_2
            gradient_bias_2 = float(error.mean())
            hidden_gradient = np.outer(error, weights_2) * (1.0 - hidden**2)
            gradient_1 = x.T @ hidden_gradient / sample_count + self.l2 * weights_1
            gradient_bias_1 = hidden_gradient.mean(axis=0)
            weights_2 -= self.learning_rate * gradient_2
            bias_2 -= self.learning_rate * gradient_bias_2
            weights_1 -= self.learning_rate * gradient_1
            bias_1 -= self.learning_rate * gradient_bias_1

        self._weights_1 = weights_1
        self._bias_1 = bias_1
        self._weights_2 = weights_2
        self._bias_2 = bias_2
        self._fit_sample_count = int(sample_count)


def build_linear_population(
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    shared_lag_scores: FloatArray | None = None,
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
    mode = learned_overlap_mode(agents)
    shared_lag_scores = _resolve_shared_lag_scores(
        agents, rng, mode, shared_lag_scores
    )

    population: list[LinearMLAgent] = []
    for _ in range(count):
        lag_weights = _learned_lag_weights(
            int(learned["lookback"]),
            agents,
            rng,
            mode,
            shared_lag_scores,
        )
        population.append(
            LinearMLAgent(
                lookback=int(learned["lookback"]),
                ridge_alpha=float(learned["ridge_alpha"]),
                refit_interval=int(learned["refit_interval"]),
                training_window=int(learned["training_window"]),
                min_samples=int(learned["min_samples"]),
                lag_weights=lag_weights,
                q_max=_disperse(float(agents["q_max"]), dispersion, rng),
                k=_disperse(float(agents["risk_aversion_k"]), dispersion, rng),
                u_max=_disperse(float(agents["order_limit"]), dispersion, rng),
                epsilon=float(agents.get("epsilon", 1.0e-8)),
            )
        )
    return population


def build_gbdt_population(
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    shared_lag_scores: FloatArray | None = None,
) -> list[LightGBMAgent]:
    """Build the configured LightGBM family using deterministic agent seeds."""

    agents = config["agents"] if "agents" in config else config
    learned = agents.get("gbdt", {})
    count = _nonnegative_count(learned, "gbdt")
    dispersion = _validated_dispersion(agents)
    mode = learned_overlap_mode(agents)
    shared_lag_scores = _resolve_shared_lag_scores(
        agents, rng, mode, shared_lag_scores
    )
    population: list[LightGBMAgent] = []
    for _ in range(count):
        lag_weights = _learned_lag_weights(
            int(learned["lookback"]),
            agents,
            rng,
            mode,
            shared_lag_scores,
        )
        population.append(
            LightGBMAgent(
                lookback=int(learned["lookback"]),
                refit_interval=int(learned["refit_interval"]),
                training_window=int(learned["training_window"]),
                min_samples=int(learned["min_samples"]),
                n_estimators=int(learned["n_estimators"]),
                learning_rate=float(learned["learning_rate"]),
                num_leaves=int(learned["num_leaves"]),
                random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                lag_weights=lag_weights,
                q_max=_disperse(float(agents["q_max"]), dispersion, rng),
                k=_disperse(float(agents["risk_aversion_k"]), dispersion, rng),
                u_max=_disperse(float(agents["order_limit"]), dispersion, rng),
                epsilon=float(agents.get("epsilon", 1.0e-8)),
            )
        )
    return population


def build_mlp_population(
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    shared_lag_scores: FloatArray | None = None,
) -> list[MLPAgent]:
    """Build the configured small NumPy-MLP family."""

    agents = config["agents"] if "agents" in config else config
    learned = agents.get("mlp", {})
    count = _nonnegative_count(learned, "mlp")
    dispersion = _validated_dispersion(agents)
    mode = learned_overlap_mode(agents)
    shared_lag_scores = _resolve_shared_lag_scores(
        agents, rng, mode, shared_lag_scores
    )
    population: list[MLPAgent] = []
    for _ in range(count):
        lag_weights = _learned_lag_weights(
            int(learned["lookback"]),
            agents,
            rng,
            mode,
            shared_lag_scores,
        )
        model_rng = np.random.default_rng(
            int(rng.integers(0, np.iinfo(np.int64).max))
        )
        population.append(
            MLPAgent(
                lookback=int(learned["lookback"]),
                refit_interval=int(learned["refit_interval"]),
                training_window=int(learned["training_window"]),
                min_samples=int(learned["min_samples"]),
                hidden_size=int(learned["hidden_size"]),
                epochs=int(learned["epochs"]),
                learning_rate=float(learned["learning_rate"]),
                l2=float(learned["l2"]),
                rng=model_rng,
                lag_weights=lag_weights,
                q_max=_disperse(float(agents["q_max"]), dispersion, rng),
                k=_disperse(float(agents["risk_aversion_k"]), dispersion, rng),
                u_max=_disperse(float(agents["order_limit"]), dispersion, rng),
                epsilon=float(agents.get("epsilon", 1.0e-8)),
            )
        )
    return population


def build_population(config: Mapping[str, Any], rng: np.random.Generator) -> list[Agent]:
    """Build all five configured families and validate their explicit counts."""

    agents = config["agents"] if "agents" in config else config
    total = int(agents["population_size"])
    momentum_count = _nonnegative_count(agents.get("momentum", {}), "momentum")
    reversion_count = _nonnegative_count(agents.get("reversion", {}), "reversion")
    linear_count = _nonnegative_count(agents.get("linear", {}), "linear")
    gbdt_count = _nonnegative_count(agents.get("gbdt", {}), "gbdt")
    mlp_count = _nonnegative_count(agents.get("mlp", {}), "mlp")
    configured_total = (
        momentum_count + reversion_count + linear_count + gbdt_count + mlp_count
    )
    if configured_total != total:
        raise ValueError(
            "configured family counts must sum to agents.population_size "
            f"({configured_total} != {total})"
        )

    mode = learned_overlap_mode(agents)
    shared_lag_scores = _resolve_shared_lag_scores(agents, rng, mode, None)

    factor_config = deepcopy(dict(config))
    factor_count = momentum_count + reversion_count
    if "agents" in config:
        factor_config["agents"] = deepcopy(dict(agents))
        factor_config["agents"]["population_size"] = factor_count
        factor_config["agents"]["momentum_fraction"] = (
            momentum_count / factor_count if factor_count else 0.0
        )
    else:
        factor_config["population_size"] = factor_count
        factor_config["momentum_fraction"] = (
            momentum_count / factor_count if factor_count else 0.0
        )

    population: list[Agent] = []
    if factor_count:
        population.extend(build_factor_population(factor_config, rng))
    population.extend(
        build_linear_population(
            config, rng, shared_lag_scores=shared_lag_scores
        )
    )
    population.extend(
        build_gbdt_population(config, rng, shared_lag_scores=shared_lag_scores)
    )
    population.extend(
        build_mlp_population(config, rng, shared_lag_scores=shared_lag_scores)
    )
    return population


def _validate_rolling_parameters(
    lookback: int, refit_interval: int, training_window: int, min_samples: int
) -> None:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if refit_interval <= 0:
        raise ValueError("refit_interval must be positive")
    if training_window <= 0:
        raise ValueError("training_window must be positive")
    if min_samples <= 0 or min_samples > training_window:
        raise ValueError("min_samples must be in [1, training_window]")


def _should_refit(
    sample_count: int,
    min_samples: int,
    step: int,
    last_fit_step: int | None,
    refit_interval: int,
) -> bool:
    return sample_count >= min_samples and (
        last_fit_step is None or step - last_fit_step >= refit_interval
    )


def _nonnegative_count(config: Mapping[str, Any], family: str) -> int:
    count = int(config.get("count", 0))
    if count < 0:
        raise ValueError(f"agents.{family}.count must be nonnegative")
    return count


def _validated_dispersion(agents: Mapping[str, Any]) -> float:
    dispersion = float(agents["parameter_dispersion"])
    if dispersion < 0.0 or not np.isfinite(dispersion):
        raise ValueError("agents.parameter_dispersion must be finite and nonnegative")
    return dispersion


def _validated_history(returns: FloatArray) -> FloatArray:
    history = np.asarray(returns, dtype=np.float64)
    if history.ndim != 1:
        raise ValueError("observation returns must be one-dimensional")
    if not np.all(np.isfinite(history)):
        raise ValueError("observation returns must be finite")
    return history


def _lagged_examples(
    history: FloatArray,
    lookback: int,
    lag_weights: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    sample_count = max(0, history.size - lookback)
    features = np.empty((sample_count, lookback), dtype=np.float64)
    for row, target_index in enumerate(range(lookback, history.size)):
        features[row] = _weighted_lag_inputs(
            history[target_index - lookback : target_index], lag_weights
        )
    return features, history[lookback:].copy()


def _weighted_lag_inputs(
    values: FloatArray, lag_weights: FloatArray | None
) -> FloatArray:
    if lag_weights is None:
        return values
    return values * lag_weights


def _resolve_shared_lag_scores(
    agents: Mapping[str, Any],
    rng: np.random.Generator,
    mode: str,
    shared_lag_scores: FloatArray | None,
) -> FloatArray | None:
    if mode == "none":
        return None
    if shared_lag_scores is not None:
        return shared_lag_scores

    lookbacks = [
        int(agents[family]["lookback"])
        for family in ("linear", "gbdt", "mlp")
        if _nonnegative_count(agents.get(family, {}), family) > 0
    ]
    if not lookbacks:
        return np.empty(0, dtype=np.float64)
    dispersion = _validated_dispersion(agents)
    if dispersion == 0.0:
        return np.ones(max(lookbacks), dtype=np.float64)
    return rng.lognormal(
        mean=-0.5 * dispersion**2,
        sigma=dispersion,
        size=max(lookbacks),
    )


def _learned_lag_weights(
    lookback: int,
    agents: Mapping[str, Any],
    rng: np.random.Generator,
    mode: str,
    shared_lag_scores: FloatArray | None,
) -> FloatArray | None:
    if mode == "none":
        return None
    if shared_lag_scores is None or shared_lag_scores.size == 0:
        raise ValueError("shared lag scores are required in lag_weights mode")
    overlap = float(agents["feature_overlap"])
    if not np.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
        raise ValueError("agents.feature_overlap must be finite and in [0, 1]")
    return _lag_weights(
        lookback,
        shared_lag_scores,
        overlap,
        _validated_dispersion(agents),
        rng,
    )


def _disperse(base: float, dispersion: float, rng: np.random.Generator) -> float:
    if dispersion == 0.0:
        return base
    return float(base * rng.lognormal(-0.5 * dispersion**2, dispersion))


__all__ = [
    "LightGBMAgent",
    "LinearMLAgent",
    "MLPAgent",
    "build_gbdt_population",
    "build_linear_population",
    "build_mlp_population",
    "build_population",
]
