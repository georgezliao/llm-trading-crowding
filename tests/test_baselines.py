"""Tests for Task C baseline fitting and episode-level splitting."""

from __future__ import annotations

import numpy as np

from herdlab.inference.baselines import (
    LightGBMObserver,
    LogisticObserver,
    TaskCEpisode,
    TrivialReferenceObserver,
    split_episodes,
)
from herdlab.market.records import PublicRecord


def _episode(index: int, *, n_steps: int = 16, horizon: int = 3) -> TaskCEpisode:
    returns = np.linspace(-0.03, 0.03, n_steps) + index * 0.001
    public = PublicRecord(
        prices=100.0 * np.exp(np.r_[0.0, np.cumsum(returns)]),
        returns=returns,
        volume=np.arange(1, n_steps + 1, dtype=np.float64),
        executed_imbalance=np.linspace(-10.0, 10.0, n_steps),
        realized_volatility=np.linspace(0.0, 0.1, n_steps),
        liquidity_proxy=np.full(n_steps, 100.0 + index),
    )
    labels = np.arange(n_steps - horizon) % 2 == 0
    return TaskCEpisode(public, labels, episode_id=str(index), horizon=horizon)


def test_split_keeps_complete_episodes_disjoint_and_is_deterministic() -> None:
    episodes = tuple(_episode(index) for index in range(5))
    train, test = split_episodes(
        episodes, test_fraction=0.4, rng=np.random.default_rng(7)
    )
    repeated = split_episodes(
        episodes, test_fraction=0.4, rng=np.random.default_rng(7)
    )

    assert len(train) == 3
    assert len(test) == 2
    assert {item.episode_id for item in train}.isdisjoint(
        item.episode_id for item in test
    )
    assert [[item.episode_id for item in part] for part in (train, test)] == [
        [item.episode_id for item in part] for part in repeated
    ]


def test_logistic_and_lightgbm_return_one_probability_per_label() -> None:
    episodes = tuple(_episode(index) for index in range(3))
    observers = (
        TrivialReferenceObserver(return_lookback=3),
        LogisticObserver(return_lookback=3),
        LightGBMObserver(
            return_lookback=3,
            rng=np.random.default_rng(4),
            n_estimators=5,
        ),
    )
    for observer in observers:
        predictions = observer.fit(episodes[:2]).predict_proba(episodes[2:])
        assert len(predictions) == 1
        assert predictions[0].shape == np.asarray(episodes[2].labels).shape
        assert np.all((0.0 <= predictions[0]) & (predictions[0] <= 1.0))
