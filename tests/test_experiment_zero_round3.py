"""Tests for the pre-registered Experiment Zero round-3 runner."""

from __future__ import annotations

import pytest

from herdlab.evaluation.metrics import ConfidenceInterval
from herdlab.experiments.experiment_zero_round3 import (
    DecisionInputs,
    apply_decision_rules,
    load_architecture_configs,
)


def _input(
    *,
    cohort: str = "architecture",
    split: str = "heldout_architecture",
    label: str = "default",
    events: int = 40,
    base_rate: float = 0.05,
    replicator: ConfidenceInterval = ConfidenceInterval(0.1, 0.02, 0.2),
    trivial: ConfidenceInterval = ConfidenceInterval(0.2, 0.1, 0.3),
) -> DecisionInputs:
    return DecisionInputs(
        cohort=cohort,
        split=split,
        label=label,
        event_count=events,
        base_rate=base_rate,
        gru_minus_replicator=replicator,
        gru_minus_trivial=trivial,
    )


def test_architecture_configs_keep_selected_setting_and_fixed_populations() -> None:
    configs = load_architecture_configs()

    assert set(configs) == {"training", "in_distribution", "heldout_architecture"}
    for config in configs.values():
        assert config["market"]["impact_sensitivity"] == pytest.approx(0.03)
        assert config["agents"]["q_max"] == pytest.approx(200.0)
        assert config["agents"]["order_limit"] == pytest.approx(40.0)
        assert config["agents"]["population_size"] == 30
    assert configs["training"]["agents"]["momentum"]["count"] == 6
    assert configs["training"]["agents"]["linear"]["count"] == 9
    assert configs["training"]["agents"]["mlp"]["count"] == 9
    assert configs["heldout_architecture"]["agents"]["gbdt"]["count"] == 18


def test_rules_mechanically_flag_replication_and_h2_success() -> None:
    includes_zero = ConfidenceInterval(0.01, -0.03, 0.04)
    inputs = (
        _input(
            cohort="v2",
            split="in_distribution",
            replicator=includes_zero,
        ),
        _input(),
    )

    outcomes = apply_decision_rules(inputs)

    assert "consistent with strategy replication" in outcomes[0]
    assert "not reported as consistent with strategy replication" in outcomes[1]
    assert any("finds evidence for H2-style generalization" in value for value in outcomes)


def test_rules_cover_uninformative_architecture_and_underpowered_forced_flow() -> None:
    inputs = (
        _input(base_rate=0.009),
        _input(
            cohort="v2",
            split="heldout_families",
            label="forced_flow",
            events=29,
            base_rate=0.002,
            replicator=ConfidenceInterval(-0.1, -0.2, -0.01),
        ),
    )

    outcomes = apply_decision_rules(inputs)

    assert any("split is reported as uninformative" in value for value in outcomes)
    assert any("29 test events is fewer than 30" in value for value in outcomes)
    assert any("reported as underpowered" in value for value in outcomes)
