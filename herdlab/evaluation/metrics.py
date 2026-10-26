"""Binary unwind metrics with episode-level bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score


FloatArray = NDArray[np.float64]
Metric = Callable[[ArrayLike, ArrayLike], float]


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float


def pr_auc(y_true: ArrayLike, y_probability: ArrayLike) -> float:
    """Return area under the stepwise precision-recall curve (AP)."""

    labels, probabilities = _binary_inputs(y_true, y_probability)
    if not np.any(labels):
        raise ValueError("PR-AUC requires at least one positive label")
    return float(average_precision_score(labels, probabilities))


def brier_score(y_true: ArrayLike, y_probability: ArrayLike) -> float:
    """Return the mean squared error of probabilistic predictions."""

    labels, probabilities = _binary_inputs(y_true, y_probability)
    return float(np.mean((probabilities - labels.astype(np.float64)) ** 2))


def recall_at_false_alarm_rate(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    max_false_alarm_rate: float = 0.05,
) -> float:
    """Maximum recall at a threshold whose false-positive rate is bounded."""

    labels, probabilities = _binary_inputs(y_true, y_probability)
    limit = float(max_false_alarm_rate)
    if not np.isfinite(limit) or not 0.0 <= limit <= 1.0:
        raise ValueError("max_false_alarm_rate must be between zero and one")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("recall at false-alarm rate requires both classes")

    best_recall = 0.0
    thresholds = np.concatenate(([np.inf], np.unique(probabilities)[::-1]))
    for threshold in thresholds:
        predicted = probabilities >= threshold
        false_alarm_rate = np.count_nonzero(predicted & ~labels) / negatives
        if false_alarm_rate <= limit:
            recall = np.count_nonzero(predicted & labels) / positives
            best_recall = max(best_recall, recall)
    return float(best_recall)


def binary_metrics(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    max_false_alarm_rate: float = 0.05,
) -> dict[str, float]:
    """Compute the required Task C point metrics."""

    return {
        "pr_auc": pr_auc(y_true, y_probability),
        "brier": brier_score(y_true, y_probability),
        "recall_at_false_alarm_rate": recall_at_false_alarm_rate(
            y_true,
            y_probability,
            max_false_alarm_rate=max_false_alarm_rate,
        ),
    }


def episode_bootstrap_ci(
    labels_by_episode: Sequence[ArrayLike],
    probabilities_by_episode: Sequence[ArrayLike],
    metric: Metric,
    *,
    rng: np.random.Generator,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Bootstrap a pooled metric by resampling whole episodes.

    All time steps from a sampled episode move together.  An episode can be
    selected more than once, as required by the ordinary nonparametric
    bootstrap; overlapping windows are never treated as independent units.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    labels, probabilities = _episode_inputs(
        labels_by_episode, probabilities_by_episode
    )
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples <= 0
    ):
        raise ValueError("n_resamples must be a positive integer")
    level = float(confidence)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must be between zero and one")

    estimate = float(metric(np.concatenate(labels), np.concatenate(probabilities)))
    samples = np.empty(int(n_resamples), dtype=np.float64)
    sample_index = 0
    attempts = 0
    max_attempts = max(100, int(n_resamples) * 100)
    while sample_index < int(n_resamples) and attempts < max_attempts:
        attempts += 1
        selected = rng.integers(0, len(labels), size=len(labels))
        sample_labels = np.concatenate([labels[index] for index in selected])
        sample_probabilities = np.concatenate(
            [probabilities[index] for index in selected]
        )
        try:
            value = float(metric(sample_labels, sample_probabilities))
        except ValueError:
            # Rare-event episode bootstraps can draw no positives or no
            # negatives.  Such replicates do not define PR-AUC or constrained
            # recall and are redrawn as complete episode samples.
            continue
        if not np.isfinite(value):
            continue
        samples[sample_index] = value
        sample_index += 1
    if sample_index < int(n_resamples):
        raise ValueError("metric is undefined for too many bootstrap resamples")
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return ConfidenceInterval(estimate=estimate, lower=float(lower), upper=float(upper))


def episode_bootstrap_metrics(
    labels_by_episode: Sequence[ArrayLike],
    probabilities_by_episode: Sequence[ArrayLike],
    *,
    rng: np.random.Generator,
    max_false_alarm_rate: float = 0.05,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> Mapping[str, ConfidenceInterval]:
    """Return episode-bootstrap intervals for all required Task C metrics."""

    metrics: dict[str, Metric] = {
        "pr_auc": pr_auc,
        "brier": brier_score,
        "recall_at_false_alarm_rate": lambda labels, probabilities: (
            recall_at_false_alarm_rate(
                labels,
                probabilities,
                max_false_alarm_rate=max_false_alarm_rate,
            )
        ),
    }
    return {
        name: episode_bootstrap_ci(
            labels_by_episode,
            probabilities_by_episode,
            metric,
            rng=rng,
            n_resamples=n_resamples,
            confidence=confidence,
        )
        for name, metric in metrics.items()
    }


def _binary_inputs(
    y_true: ArrayLike, y_probability: ArrayLike
) -> tuple[NDArray[np.bool_], FloatArray]:
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_probability, dtype=np.float64)
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or labels.shape != probabilities.shape
    ):
        raise ValueError("labels and probabilities must be equal-length vectors")
    if len(labels) == 0:
        raise ValueError("labels and probabilities must not be empty")
    if not np.all(np.isin(labels, (0, 1, False, True))):
        raise ValueError("labels must be binary")
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return np.asarray(labels, dtype=np.bool_), probabilities


def _episode_inputs(
    labels_by_episode: Sequence[ArrayLike],
    probabilities_by_episode: Sequence[ArrayLike],
) -> tuple[list[NDArray[np.bool_]], list[FloatArray]]:
    if len(labels_by_episode) == 0 or len(labels_by_episode) != len(
        probabilities_by_episode
    ):
        raise ValueError("episode collections must be nonempty and equal length")
    checked = [
        _binary_inputs(labels, probabilities)
        for labels, probabilities in zip(labels_by_episode, probabilities_by_episode)
    ]
    return [item[0] for item in checked], [item[1] for item in checked]


# Alternative spelling retained for clarity at call sites.
recall_at_fixed_false_alarm_rate = recall_at_false_alarm_rate


__all__ = [
    "ConfidenceInterval",
    "binary_metrics",
    "brier_score",
    "episode_bootstrap_ci",
    "episode_bootstrap_metrics",
    "pr_auc",
    "recall_at_false_alarm_rate",
    "recall_at_fixed_false_alarm_rate",
]
