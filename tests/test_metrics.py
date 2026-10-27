"""Tests for Task C point metrics and episode-bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from herdlab.evaluation.metrics import (
    brier_score,
    episode_bootstrap_ci,
    episode_bootstrap_metrics,
    paired_pr_auc_difference_ci,
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


def test_bootstrap_reports_undefined_class_metrics_when_no_events() -> None:
    labels = [np.zeros(3, dtype=np.bool_), np.zeros(2, dtype=np.bool_)]
    probabilities = [np.full(3, 0.1), np.full(2, 0.2)]

    metrics = episode_bootstrap_metrics(
        labels,
        probabilities,
        rng=np.random.default_rng(3),
        n_resamples=20,
    )

    assert np.isnan(metrics["pr_auc"].estimate)
    assert np.isnan(metrics["recall_at_false_alarm_rate"].estimate)
    assert metrics["brier"].estimate == pytest.approx(0.022)


def test_paired_pr_auc_difference_is_exactly_zero_for_identical_predictions() -> None:
    labels = [np.asarray([0, 1, 0]), np.asarray([1, 0, 1])]
    probabilities = [np.asarray([0.1, 0.8, 0.3]), np.asarray([0.7, 0.2, 0.9])]

    result = paired_pr_auc_difference_ci(
        labels,
        probabilities,
        probabilities,
        rng=np.random.default_rng(12),
        n_resamples=100,
    )

    assert result.estimate == 0.0
    assert result.lower == 0.0
    assert result.upper == 0.0
