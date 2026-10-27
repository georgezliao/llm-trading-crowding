"""Episode-safe baseline observers for ten-step unwind forecasting (Task C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, TypeVar

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from herdlab.inference.features import (
    build_public_features,
    build_strategy_replication_features,
)
from herdlab.inference.observer_data import PublicRecordSource, load_public_record


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
DEFAULT_UNWIND_HORIZON = 10


@dataclass(frozen=True)
class TaskCEpisode:
    """One public episode paired only with its precomputed target labels.

    Labels are kept as a separate input: observer code never receives private
    positions, per-agent orders, or a :class:`PrivateRecord`.
    """

    public: PublicRecordSource
    labels: ArrayLike
    episode_id: str = ""
    horizon: int = DEFAULT_UNWIND_HORIZON

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if not np.all(np.isin(labels, (0, 1, False, True))):
            raise ValueError("labels must be binary")
        if isinstance(self.horizon, bool) or not isinstance(
            self.horizon, (int, np.integer)
        ):
            raise TypeError("horizon must be an integer")
        public = load_public_record(self.public)
        if self.horizon <= 0 or len(labels) != len(public.returns) - self.horizon:
            raise ValueError(
                "labels must cover every complete strictly-future horizon window"
            )


EpisodeT = TypeVar("EpisodeT")


def split_episodes(
    episodes: Sequence[EpisodeT],
    *,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[tuple[EpisodeT, ...], tuple[EpisodeT, ...]]:
    """Randomly split whole episodes, never overlapping their time steps."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if len(episodes) < 2:
        raise ValueError("at least two episodes are required")
    fraction = float(test_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    n_test = min(len(episodes) - 1, max(1, int(np.ceil(len(episodes) * fraction))))
    permutation = rng.permutation(len(episodes))
    test_indices = set(int(index) for index in permutation[:n_test])
    train = tuple(
        episode
        for index, episode in enumerate(episodes)
        if index not in test_indices
    )
    test = tuple(
        episode for index, episode in enumerate(episodes) if index in test_indices
    )
    return train, test


class _ProbabilityEstimator(Protocol):
    def fit(self, features: FloatArray, labels: BoolArray) -> object: ...

    def predict_proba(self, features: FloatArray) -> FloatArray: ...


class _BaseObserver:
    """Shared public-feature handling for independently fitted baselines."""

    def __init__(self, *, return_lookback: int = 10) -> None:
        self.return_lookback = return_lookback
        self._estimator: _ProbabilityEstimator | None = None
        self._constant_probability: float | None = None

    def _make_estimator(self) -> _ProbabilityEstimator:
        raise NotImplementedError

    def _features(self, episode: TaskCEpisode) -> FloatArray:
        return np.asarray(
            build_public_features(
                episode.public, return_lookback=self.return_lookback
            ).values[: len(episode.labels)],
            dtype=np.float64,
        )

    def fit(self, episodes: Sequence[TaskCEpisode]) -> _BaseObserver:
        if not episodes:
            raise ValueError("episodes must not be empty")
        features = np.concatenate([self._features(episode) for episode in episodes])
        labels = np.concatenate(
            [np.asarray(episode.labels, dtype=np.bool_) for episode in episodes]
        )
        unique = np.unique(labels)
        if len(unique) == 1:
            self._constant_probability = float(unique[0])
            self._estimator = None
        else:
            self._constant_probability = None
            self._estimator = self._make_estimator()
            self._estimator.fit(features, labels)
        return self

    def predict_proba(self, episodes: Sequence[TaskCEpisode]) -> tuple[FloatArray, ...]:
        """Return positive-class probabilities separately for each episode."""

        if self._estimator is None and self._constant_probability is None:
            raise RuntimeError("observer must be fitted before prediction")
        predictions: list[FloatArray] = []
        for episode in episodes:
            features = self._features(episode)
            if self._constant_probability is not None:
                probability = np.full(len(features), self._constant_probability)
            else:
                assert self._estimator is not None
                probability = np.asarray(
                    self._estimator.predict_proba(features)[:, 1], dtype=np.float64
                )
            predictions.append(probability)
        return tuple(predictions)


class LogisticObserver(_BaseObserver):
    """Standardized L2 logistic-regression Task C observer."""

    def _make_estimator(self) -> _ProbabilityEstimator:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
            ]
        )


class TrivialReferenceObserver(LogisticObserver):
    """Logistic reference using only recent absolute OFI and realized vol.

    OFI is the observer-safe executed imbalance divided by the public
    liquidity proxy.  Its absolute value is lagged causally through the
    configured lookback; the only other input is current realized volatility.
    """

    def _features(self, episode: TaskCEpisode) -> FloatArray:
        public = load_public_record(episode.public)
        imbalance = np.abs(
            np.asarray(public.executed_imbalance, dtype=np.float64)
            / np.asarray(public.liquidity_proxy, dtype=np.float64)
        )
        n_steps = len(imbalance)
        lagged = np.zeros((n_steps, self.return_lookback), dtype=np.float64)
        for lag in range(self.return_lookback):
            if lag < n_steps:
                lagged[lag:, lag] = imbalance[: n_steps - lag]
        features = np.column_stack(
            (
                lagged,
                np.asarray(public.realized_volatility, dtype=np.float64),
            )
        )
        return np.asarray(features[: len(episode.labels)], dtype=np.float64)


class LightGBMObserver(_BaseObserver):
    """Small deterministic LightGBM Task C observer."""

    def __init__(
        self,
        *,
        return_lookback: int = 10,
        rng: np.random.Generator,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
    ) -> None:
        super().__init__(return_lookback=return_lookback)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self.random_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)

    def _make_estimator(self) -> _ProbabilityEstimator:
        return LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )


class StrategyReplicationObserver(LightGBMObserver):
    """LightGBM on price-only reconstructions of the baseline strategies.

    The estimator configuration is inherited unchanged from
    :class:`LightGBMObserver`; only its feature matrix differs.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        rng: np.random.Generator,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        self.config = config
        super().__init__(
            return_lookback=1,
            rng=rng,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
        )

    def _features(self, episode: TaskCEpisode) -> FloatArray:
        features = build_strategy_replication_features(
            episode.public, self.config
        ).values
        return np.asarray(features[: len(episode.labels)], dtype=np.float64)


def _episode_arrays(
    episode: TaskCEpisode, *, return_lookback: int
) -> tuple[FloatArray, BoolArray]:
    features = build_public_features(
        episode.public, return_lookback=return_lookback
    ).values
    labels = np.asarray(episode.labels, dtype=np.bool_)
    return np.asarray(features[: len(labels)], dtype=np.float64), labels


def _stack_episodes(
    episodes: Sequence[TaskCEpisode], *, return_lookback: int
) -> tuple[FloatArray, BoolArray]:
    if not episodes:
        raise ValueError("episodes must not be empty")
    arrays = [
        _episode_arrays(episode, return_lookback=return_lookback)
        for episode in episodes
    ]
    return (
        np.concatenate([features for features, _ in arrays]),
        np.concatenate([labels for _, labels in arrays]),
    )


# Explicit name used by experiment code and the architecture.
episode_train_test_split = split_episodes


__all__ = [
    "DEFAULT_UNWIND_HORIZON",
    "LightGBMObserver",
    "LogisticObserver",
    "StrategyReplicationObserver",
    "TaskCEpisode",
    "TrivialReferenceObserver",
    "episode_train_test_split",
    "split_episodes",
]
