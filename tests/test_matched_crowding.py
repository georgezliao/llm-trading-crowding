"""Tests for the matched low/high crowding experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import herdlab.experiments.matched_crowding as matched_crowding
from herdlab.experiments.matched_crowding import (
    main,
    overlap_configs,
    paired_episode_bootstrap_ci,
    prepare_matched_inputs,
)


PROJECT_ROOT = Path(__file__).parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


def _short_config() -> dict:
    config = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    config["simulation"]["n_steps"] = 12
    config["agents"]["population_size"] = 2
    config["agents"]["momentum"]["count"] = 2
    config["agents"]["reversion"]["count"] = 0
    for family in ("linear", "gbdt", "mlp"):
        config["agents"][family]["count"] = 0
    return config


def test_pairing_uses_identical_exogenous_arrays_and_matched_capacity() -> None:
    low, high = overlap_configs(_short_config())

    exogenous, low_agents, high_agents = prepare_matched_inputs(low, high, seed=2718)
    repeated_exogenous, _, _ = prepare_matched_inputs(low, high, seed=2718)

    for field in exogenous.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(exogenous, field), getattr(repeated_exogenous, field)
        )
    assert [type(agent) for agent in low_agents] == [
        type(agent) for agent in high_agents
    ]
    assert sum(agent.q_max for agent in low_agents) == sum(
        agent.q_max for agent in high_agents
    )
    assert low["agents"]["feature_overlap"] == 0.0
    assert high["agents"]["feature_overlap"] == 1.0


def test_pair_simulations_receive_the_same_exogenous_path_object(monkeypatch) -> None:
    low, high = overlap_configs(_short_config())
    observed_paths = []
    simulate = matched_crowding.simulate_market_records

    def capture_path(config, agents, exogenous):
        observed_paths.append(exogenous)
        return simulate(config, agents, exogenous)

    monkeypatch.setattr(matched_crowding, "simulate_market_records", capture_path)
    matched_crowding.run_matched_episode(low, high, seed=99)

    assert len(observed_paths) == 2
    assert observed_paths[0] is observed_paths[1]


def test_paired_bootstrap_uses_high_minus_low_episode_differences() -> None:
    result = paired_episode_bootstrap_ci(
        [1.0, 2.0, 4.0],
        [2.0, 4.0, 7.0],
        rng=np.random.default_rng(4),
        n_resamples=200,
    )

    assert result.low_mean == pytest.approx(7.0 / 3.0)
    assert result.high_mean == pytest.approx(13.0 / 3.0)
    assert result.estimate == pytest.approx(2.0)
    assert result.lower <= result.estimate <= result.upper


def test_cli_runs_two_short_pairs_and_writes_report(tmp_path: Path) -> None:
    config_path = tmp_path / "short.yaml"
    report_path = tmp_path / "matched.md"
    config_path.write_text(yaml.safe_dump(_short_config()), encoding="utf-8")

    main(
        [
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--n-episodes",
            "2",
            "--seed-start",
            "31",
            "--bootstrap-resamples",
            "100",
        ]
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# Matched crowding comparison" in report
    assert "2 consecutive seeds (`31-32`)" in report
    assert "Paired difference" in report
    assert "Excess realized volatility" in report
    assert "| 31 |" in report
    assert "| 32 |" in report
