"""Tests for the pre-registered round-4 rules and joint resampling."""

from __future__ import annotations

import numpy as np
import pytest

from herdlab.evaluation.metrics import ConfidenceInterval
from herdlab.experiments.experiment_zero_round4 import (
    FeatureDecisionInput,
    ForcedFlowDecisionInput,
    SignalDecisionInput,
    apply_decision_rules,
    joint_signal_degradation_ci,
)


def _signals(*, endpoint_base_rate: float = 0.05) -> tuple[SignalDecisionInput, ...]:
    positive = ConfidenceInterval(0.2, 0.1, 0.3)
    return tuple(
        SignalDecisionInput(
            strength=strength,
            default_base_rate=endpoint_base_rate if strength == 0.5 else 0.05,
            gru_minus_replicator=positive,
        )
        for strength in (0.0, 0.1, 0.25, 0.5)
    )


def _features() -> tuple[FeatureDecisionInput, ...]:
    return (
        FeatureDecisionInput(
            "default",
            "order_flow",
            "lightgbm",
            ConfidenceInterval(0.1, 0.02, 0.2),
        ),
        FeatureDecisionInput(
            "default",
            "volume",
            "gru_6000",
            ConfidenceInterval(0.01, -0.02, 0.04),
        ),
    )


def test_rules_cover_positive_and_nonpositive_feature_use() -> None:
    outcomes = apply_decision_rules(
        _signals(),
        _features(),
        (ForcedFlowDecisionInput("rho=0/forced_flow", 30),),
        degradation_interval=ConfidenceInterval(0.08, 0.01, 0.15),
    )

    assert "private information degrades inference" in outcomes[0]
    assert any("order_flow" in value and "reported as used" in value for value in outcomes)
    assert any("volume" in value and "not reported as used" in value for value in outcomes)
    assert any("meets the 30-event minimum" in value for value in outcomes)


def test_rules_flag_uninformative_and_underpowered_cases() -> None:
    outcomes = apply_decision_rules(
        _signals(endpoint_base_rate=0.009),
        _features(),
        (ForcedFlowDecisionInput("risk_off/forced_flow", 0),),
        degradation_interval=ConfidenceInterval(0.2, 0.1, 0.3),
    )

    assert "comparison is reported as uninformative" in outcomes[0]
    assert any("rho=0.5/default" in value and "uninformative" in value for value in outcomes)
    assert any("0 test events is fewer than 30" in value for value in outcomes)
    assert any("reported as underpowered" in value for value in outcomes)


def test_rules_reject_duplicate_registered_strength() -> None:
    duplicated = (*_signals(), _signals()[0])

    with pytest.raises(ValueError, match="each registered rho once"):
        apply_decision_rules(
            duplicated,
            _features(),
            (ForcedFlowDecisionInput("rho=0/forced_flow", 30),),
            degradation_interval=ConfidenceInterval(0.08, 0.01, 0.15),
        )


def test_joint_resample_statistic_on_toy_inputs() -> None:
    labels = (np.array([0, 1, 0, 1], dtype=bool),)
    strong = (np.array([0.1, 0.9, 0.2, 0.8]),)
    weak = (np.array([0.8, 0.7, 0.6, 0.5]),)

    interval = joint_signal_degradation_ci(
        labels,
        strong,
        weak,
        labels,
        weak,
        weak,
        rng=np.random.default_rng(4),
        n_resamples=20,
    )

    assert interval.estimate > 0.0
    assert interval.lower == interval.estimate
    assert interval.upper == interval.estimate


def test_joint_resample_uses_shared_episode_draws() -> None:
    labels = (
        np.array([0, 1, 0, 1], dtype=bool),
        np.array([0, 0, 1, 1], dtype=bool),
    )
    first = (
        np.array([0.1, 0.9, 0.2, 0.8]),
        np.array([0.2, 0.1, 0.8, 0.7]),
    )
    second = (
        np.array([0.7, 0.6, 0.5, 0.4]),
        np.array([0.8, 0.7, 0.2, 0.1]),
    )

    interval = joint_signal_degradation_ci(
        labels,
        first,
        second,
        labels,
        first,
        second,
        rng=np.random.default_rng(5),
        n_resamples=50,
    )

    assert interval == ConfidenceInterval(0.0, 0.0, 0.0)
