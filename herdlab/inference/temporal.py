"""Small causal GRU observer for public market-feature sequences."""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from herdlab.inference.baselines import TaskCEpisode, split_episodes
from herdlab.inference.features import build_public_features


FloatArray = NDArray[np.float64]


class _CausalGRU(nn.Module):
    """A one-directional GRU followed by one logit at the window endpoint."""

    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor) -> Tensor:
        states, _ = self.gru(features)
        return self.output(states[:, -1]).squeeze(-1)


class GRUObserver:
    """Predict Task C labels from fixed-length causal public-feature windows.

    ``fit`` receives training episodes only and automatically holds out whole
    episodes for early stopping.  It never accepts a separate validation or
    test set, which prevents an evaluation episode from being used by the
    stopping rule accidentally.  Feature scaling is fitted on the retained
    training episodes, not the validation episodes.

    The implementation is deliberately small and CPU-only.  Every fit resets
    PyTorch's seed and uses a single CPU thread; batches are visited in fixed
    order, making repeated fits with the same inputs and seed deterministic.
    """

    def __init__(
        self,
        *,
        return_lookback: int = 10,
        sequence_length: int = 20,
        hidden_size: int = 16,
        learning_rate: float = 1.0e-3,
        batch_size: int = 512,
        max_epochs: int = 40,
        patience: int = 5,
        min_delta: float = 1.0e-5,
        validation_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.return_lookback = _positive_integer(return_lookback, "return_lookback")
        self.sequence_length = _positive_integer(sequence_length, "sequence_length")
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.max_epochs = _positive_integer(max_epochs, "max_epochs")
        self.patience = _positive_integer(patience, "patience")
        self.learning_rate = _positive_finite(learning_rate, "learning_rate")
        self.min_delta = _nonnegative_finite(min_delta, "min_delta")
        self.validation_fraction = float(validation_fraction)
        if not np.isfinite(self.validation_fraction) or not (
            0.0 < self.validation_fraction < 1.0
        ):
            raise ValueError("validation_fraction must be between zero and one")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        self.seed = int(seed)
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")

        self._model: _CausalGRU | None = None
        self._feature_mean: FloatArray | None = None
        self._feature_scale: FloatArray | None = None
        self.epochs_trained_: int | None = None
        self.best_validation_loss_: float | None = None
        self.training_episode_ids_: tuple[str, ...] | None = None
        self.validation_episode_ids_: tuple[str, ...] | None = None

    def fit(self, episodes: Sequence[TaskCEpisode]) -> GRUObserver:
        """Fit using an episode-level validation split of ``episodes``."""

        if len(episodes) < 2:
            raise ValueError("at least two training episodes are required")
        if not all(isinstance(episode, TaskCEpisode) for episode in episodes):
            raise TypeError("episodes must contain only TaskCEpisode objects")

        training, validation = split_episodes(
            episodes,
            test_fraction=self.validation_fraction,
            rng=np.random.default_rng(self.seed),
        )
        training_arrays = [_public_episode_array(item, self.return_lookback) for item in training]
        validation_arrays = [
            _public_episode_array(item, self.return_lookback) for item in validation
        ]
        if any(len(features) == 0 for features, _ in training_arrays + validation_arrays):
            raise ValueError("each episode must contain at least one labeled time step")

        feature_rows = np.concatenate([features for features, _ in training_arrays])
        self._feature_mean = feature_rows.mean(axis=0)
        standard_deviation = feature_rows.std(axis=0)
        self._feature_scale = np.where(standard_deviation > 1.0e-12, standard_deviation, 1.0)

        training_features, training_labels = self._windowed_arrays(training_arrays)
        validation_features, validation_labels = self._windowed_arrays(validation_arrays)

        _configure_torch(self.seed)
        model = _CausalGRU(training_features.shape[-1], self.hidden_size).to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_function = nn.BCEWithLogitsLoss()
        train_x = torch.from_numpy(training_features)
        train_y = torch.from_numpy(training_labels)
        validation_x = torch.from_numpy(validation_features)
        validation_y = torch.from_numpy(validation_labels)

        best_loss = np.inf
        best_state: dict[str, Tensor] | None = None
        stale_epochs = 0
        epochs_trained = 0
        for epoch in range(self.max_epochs):
            model.train()
            for start in range(0, len(train_x), self.batch_size):
                stop = start + self.batch_size
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(train_x[start:stop]), train_y[start:stop])
                loss.backward()
                optimizer.step()

            validation_loss = _mean_loss(
                model,
                validation_x,
                validation_y,
                batch_size=self.batch_size,
                loss_function=loss_function,
            )
            epochs_trained = epoch + 1
            if validation_loss < best_loss - self.min_delta:
                best_loss = validation_loss
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self.epochs_trained_ = epochs_trained
        self.best_validation_loss_ = float(best_loss)
        self.training_episode_ids_ = tuple(item.episode_id for item in training)
        self.validation_episode_ids_ = tuple(item.episode_id for item in validation)
        return self

    def predict_proba(self, episodes: Sequence[TaskCEpisode]) -> tuple[FloatArray, ...]:
        """Return one positive-class probability per label in each episode."""

        if self._model is None or self._feature_mean is None or self._feature_scale is None:
            raise RuntimeError("observer must be fitted before prediction")
        if not all(isinstance(episode, TaskCEpisode) for episode in episodes):
            raise TypeError("episodes must contain only TaskCEpisode objects")

        predictions: list[FloatArray] = []
        self._model.eval()
        with torch.no_grad():
            for episode in episodes:
                features, _ = _public_episode_array(episode, self.return_lookback)
                normalized = (features - self._feature_mean) / self._feature_scale
                windows = torch.from_numpy(
                    _causal_windows(normalized, self.sequence_length)
                )
                chunks: list[Tensor] = []
                for start in range(0, len(windows), self.batch_size):
                    logits = self._model(windows[start : start + self.batch_size])
                    chunks.append(torch.sigmoid(logits))
                if chunks:
                    probability = torch.cat(chunks).cpu().numpy().astype(np.float64)
                else:
                    probability = np.empty(0, dtype=np.float64)
                predictions.append(probability)
        return tuple(predictions)

    def _windowed_arrays(
        self, arrays: Sequence[tuple[FloatArray, NDArray[np.float32]]]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        assert self._feature_mean is not None and self._feature_scale is not None
        windows = []
        labels = []
        for features, episode_labels in arrays:
            normalized = (features - self._feature_mean) / self._feature_scale
            windows.append(_causal_windows(normalized, self.sequence_length))
            labels.append(episode_labels)
        return np.concatenate(windows), np.concatenate(labels)


class GRUEnsembleObserver:
    """Seed-averaged ensemble of independently trained GRU observers.

    One :class:`GRUObserver` is fitted for every seed.  Per-seed predictions
    remain available for uncertainty reporting, while :meth:`predict_proba`
    returns their arithmetic mean as the primary ensemble probability.
    """

    def __init__(
        self,
        *,
        seeds: Sequence[int],
        return_lookback: int = 10,
        sequence_length: int = 20,
        hidden_size: int = 16,
        learning_rate: float = 1.0e-3,
        batch_size: int = 512,
        max_epochs: int = 40,
        patience: int = 5,
        min_delta: float = 1.0e-5,
        validation_fraction: float = 0.2,
    ) -> None:
        if not seeds:
            raise ValueError("seeds must not be empty")
        self.seeds = tuple(_seed(seed) for seed in seeds)
        self.return_lookback = return_lookback
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.validation_fraction = validation_fraction
        self.observers_: tuple[GRUObserver, ...] = ()

    def fit(self, episodes: Sequence[TaskCEpisode]) -> GRUEnsembleObserver:
        """Fit one independent GRU to ``episodes`` for each configured seed."""

        self.observers_ = tuple(
            GRUObserver(
                return_lookback=self.return_lookback,
                sequence_length=self.sequence_length,
                hidden_size=self.hidden_size,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                patience=self.patience,
                min_delta=self.min_delta,
                validation_fraction=self.validation_fraction,
                seed=seed,
            ).fit(episodes)
            for seed in self.seeds
        )
        return self

    def predict_proba_per_seed(
        self, episodes: Sequence[TaskCEpisode]
    ) -> tuple[tuple[FloatArray, ...], ...]:
        """Return episode probabilities for each seed, in configured order."""

        if not self.observers_:
            raise RuntimeError("observer must be fitted before prediction")
        return tuple(observer.predict_proba(episodes) for observer in self.observers_)

    def predict_proba(self, episodes: Sequence[TaskCEpisode]) -> tuple[FloatArray, ...]:
        """Return the arithmetic mean of per-seed probabilities by episode."""

        per_seed = self.predict_proba_per_seed(episodes)
        return tuple(
            np.mean(
                np.stack(
                    [seed_predictions[index] for seed_predictions in per_seed], axis=0
                ),
                axis=0,
            )
            for index in range(len(episodes))
        )


def _public_episode_array(
    episode: TaskCEpisode, return_lookback: int
) -> tuple[FloatArray, NDArray[np.float32]]:
    """Extract only public features plus the separately supplied target."""

    features = build_public_features(
        episode.public, return_lookback=return_lookback
    ).values
    labels = np.asarray(episode.labels, dtype=np.float32)
    return np.asarray(features[: len(labels)], dtype=np.float64), labels


def _causal_windows(features: FloatArray, sequence_length: int) -> NDArray[np.float32]:
    """Build windows ending at each row, zero-padding unavailable history."""

    n_steps, n_features = features.shape
    padded = np.zeros((n_steps + sequence_length - 1, n_features), dtype=np.float32)
    padded[sequence_length - 1 :] = features.astype(np.float32, copy=False)
    view = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=sequence_length, axis=0
    )
    return np.ascontiguousarray(view.transpose(0, 2, 1))


def _mean_loss(
    model: _CausalGRU,
    features: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    loss_function: nn.BCEWithLogitsLoss,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            loss = loss_function(model(features[start:stop]), labels[start:stop])
            total += float(loss) * (stop - start)
    return total / len(features)


def _configure_torch(seed: int) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(seed)


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("seed must be an integer")
    result = int(value)
    if not 0 <= result < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


__all__ = ["GRUEnsembleObserver", "GRUObserver"]
