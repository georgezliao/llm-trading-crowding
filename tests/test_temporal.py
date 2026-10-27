"""Tests for the causal, deterministic temporal observer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from herdlab.inference.baselines import TaskCEpisode
from herdlab.inference.temporal import GRUEnsembleObserver, GRUObserver
from herdlab.market.records import PublicRecord


def _episode(index: int, *, n_steps: int = 18, horizon: int = 3) -> TaskCEpisode:
    phase = np.arange(n_steps, dtype=np.float64) + index
    returns = 0.02 * np.sin(phase / 3.0)
    public = PublicRecord(
        prices=100.0 * np.exp(np.r_[0.0, np.cumsum(returns)]),
        returns=returns,
        volume=2.0 + np.abs(np.cos(phase)),
        executed_imbalance=np.sin(phase / 2.0),
        realized_volatility=0.01 + 0.005 * np.abs(np.sin(phase / 4.0)),
        liquidity_proxy=100.0 + phase,
    )
    labels = np.sin(phase[: n_steps - horizon] / 2.0) < 0.0
    return TaskCEpisode(public, labels, episode_id=f"episode-{index}", horizon=horizon)


def _observer(seed: int = 7) -> GRUObserver:
    return GRUObserver(
        return_lookback=3,
        sequence_length=4,
        hidden_size=5,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=8,
        patience=3,
        validation_fraction=0.25,
        seed=seed,
    )


def test_gru_output_shape_and_training_only_validation_split() -> None:
    training = tuple(_episode(index) for index in range(4))
    test = _episode(99)
    observer = _observer().fit(training)
    predictions = observer.predict_proba((test,))

    assert len(predictions) == 1
    assert predictions[0].shape == np.asarray(test.labels).shape
    assert np.all((0.0 <= predictions[0]) & (predictions[0] <= 1.0))
    assert set(observer.training_episode_ids_ or ()).isdisjoint(
        observer.validation_episode_ids_ or ()
    )
    assert set(observer.training_episode_ids_ or ()) | set(
        observer.validation_episode_ids_ or ()
    ) == {episode.episode_id for episode in training}
    assert test.episode_id not in (observer.validation_episode_ids_ or ())
    assert observer.epochs_trained_ is not None
    assert observer.epochs_trained_ <= observer.max_epochs
    assert torch.get_num_threads() == 1


def test_gru_predictions_are_causal() -> None:
    observer = _observer().fit(tuple(_episode(index) for index in range(4)))
    original = _episode(20)
    cutoff = 7
    public = original.public
    changed_public = replace(
        public,
        returns=np.r_[public.returns[: cutoff + 1], np.full(10, 9.0)],
        volume=np.r_[public.volume[: cutoff + 1], np.full(10, 999.0)],
        executed_imbalance=np.r_[
            public.executed_imbalance[: cutoff + 1], np.full(10, -999.0)
        ],
        realized_volatility=np.r_[
            public.realized_volatility[: cutoff + 1], np.full(10, 8.0)
        ],
        liquidity_proxy=np.r_[
            public.liquidity_proxy[: cutoff + 1], np.full(10, 777.0)
        ],
    )
    changed = TaskCEpisode(
        changed_public,
        original.labels,
        episode_id="changed",
        horizon=original.horizon,
    )

    original_predictions = observer.predict_proba((original,))[0]
    changed_predictions = observer.predict_proba((changed,))[0]
    np.testing.assert_array_equal(
        original_predictions[: cutoff + 1], changed_predictions[: cutoff + 1]
    )
    assert not np.array_equal(
        original_predictions[cutoff + 1 :], changed_predictions[cutoff + 1 :]
    )


def test_gru_fit_is_deterministic_for_a_fixed_seed() -> None:
    training = tuple(_episode(index) for index in range(4))
    evaluation = (_episode(30), _episode(31))

    first = _observer(seed=11).fit(training)
    first_predictions = first.predict_proba(evaluation)
    second = _observer(seed=11).fit(training)
    second_predictions = second.predict_proba(evaluation)

    for left, right in zip(first_predictions, second_predictions, strict=True):
        np.testing.assert_array_equal(left, right)
    assert first.epochs_trained_ == second.epochs_trained_
    assert first.best_validation_loss_ == second.best_validation_loss_


def test_gru_ensemble_equals_mean_of_per_seed_probabilities() -> None:
    training = tuple(_episode(index) for index in range(4))
    evaluation = (_episode(40), _episode(41))
    ensemble = GRUEnsembleObserver(
        seeds=(21, 22),
        return_lookback=3,
        sequence_length=4,
        hidden_size=5,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=8,
        patience=3,
        validation_fraction=0.25,
    ).fit(training)

    per_seed = ensemble.predict_proba_per_seed(evaluation)
    averaged = ensemble.predict_proba(evaluation)

    assert tuple(observer.seed for observer in ensemble.observers_) == (21, 22)
    for episode_index, probabilities in enumerate(averaged):
        expected = np.mean(
            np.stack([prediction[episode_index] for prediction in per_seed]), axis=0
        )
        np.testing.assert_array_equal(probabilities, expected)
