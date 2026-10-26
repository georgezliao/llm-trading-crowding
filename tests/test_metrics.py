"""Tests for Task C point metrics and episode-bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from herdlab.evaluation.metrics import (
    brier_score,
    episode_bootstrap_ci,
    pr_auc,
    recall_at_false_alarm_rate,
)


def test_metrics_on_known_toy_inputs() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.35, 0.8])

    assert pr_auc(labels, probabilities) == pytest.approx(5.0 / 6.0)
    assert brier_score(labels, probabilities) == pytest.approx(0.158125)
    assert recall_at_false_alarm_rate(
        labels, probabilities, max_false_alarm_rate=0.0
    ) == pytest.approx(0.5)
    assert recall_at_false_alarm_rate(
        labels, probabilities, max_false_alarm_rate=0.5
    ) == pytest.approx(1.0)


def test_bootstrap_resamples_episodes_and_is_deterministic() -> None:
    labels = [np.asarray([0, 1]), np.asarray([1, 0]), np.asarray([0, 1])]
    probabilities = [
        np.asarray([0.2, 0.8]),
        np.asarray([0.7, 0.3]),
        np.asarray([0.4, 0.6]),
    ]
    first = episode_bootstrap_ci(
        labels,
        probabilities,
        brier_score,
        rng=np.random.default_rng(9),
        n_resamples=100,
    )
    second = episode_bootstrap_ci(
        labels,
        probabilities,
        brier_score,
        rng=np.random.default_rng(9),
        n_resamples=100,
    )

    assert first == second
    assert first.lower <= first.estimate <= first.upper
