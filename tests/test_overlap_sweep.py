"""Tests for the pre-registered learned-agent overlap sweep and H1 rule."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from herdlab.evaluation.metrics import ConfidenceInterval
from herdlab.experiments.overlap_sweep import (
    apply_h1_rule,
    episode_bootstrap_slope_ci,
    overlap_config,
)


def test_overlap_config_sets_only_registered_overlap_controls() -> None:
    base = {
        "agents": {
            "feature_overlap": 0.5,
            "learned_overlap_mode": "none",
            "private_signal": {"strength": 0.0, "beta": 1.0, "rng_stream": 2},
            "q_max": 200.0,
        },
        "market": {"impact_sensitivity": 0.03},
    }
    original = deepcopy(base)

    condition = overlap_config(base, 0.75)

    assert base == original
    assert condition["agents"]["feature_overlap"] == 0.75
    assert condition["agents"]["learned_overlap_mode"] == "lag_weights"
    assert condition["agents"]["private_signal"]["strength"] == 0.25
    assert condition["agents"]["private_signal"]["beta"] == 1.0
    assert condition["market"] == base["market"]


def test_episode_bootstrap_slope_recovers_linear_episode_outcomes() -> None:
    overlaps = (0.0, 0.25, 0.5, 0.75, 1.0)
    episode_offsets = np.array([-0.2, 0.0, 0.1, 0.4])
    outcomes = np.asarray(
        [2.0 * overlap + episode_offsets for overlap in overlaps], dtype=np.float64
    )

    interval = episode_bootstrap_slope_ci(
        overlaps,
        outcomes,
        rng=np.random.default_rng(12),
        n_resamples=100,
    )

    assert interval.estimate == pytest.approx(2.0)
    assert interval.lower == pytest.approx(2.0)
    assert interval.upper == pytest.approx(2.0)


def test_h1_rule_supports_only_two_positive_intervals() -> None:
    positive = ConfidenceInterval(0.2, 0.1, 0.3)
    nonpositive = ConfidenceInterval(0.0, -0.1, 0.1)

    supported = apply_h1_rule(positive, positive)
    alignment_only = apply_h1_rule(positive, nonpositive)
    unsupported = apply_h1_rule(nonpositive, positive)

    assert "H1 is supported in this simulator" in supported
    assert "overlap raises alignment without raising unwinds" in alignment_only
    assert "H1 is not supported in this simulator" in unsupported


def test_h1_rule_rejects_undefined_intervals() -> None:
    with pytest.raises(ValueError, match="position_slope must be finite"):
        apply_h1_rule(
            ConfidenceInterval(np.nan, np.nan, np.nan),
            ConfidenceInterval(0.2, 0.1, 0.3),
        )
