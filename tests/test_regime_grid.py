"""Tests for the small cascade regime-grid experiment."""

from __future__ import annotations

from pathlib import Path

import yaml

from herdlab.experiments.regime_grid import main, regime_config, run_regime_grid


PROJECT_ROOT = Path(__file__).parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


def _short_config() -> dict:
    config = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    config["simulation"]["n_steps"] = 12
    config["targets"]["unwind_horizon"] = 3
    config["agents"]["population_size"] = 2
    config["agents"]["momentum"]["count"] = 1
    config["agents"]["reversion"]["count"] = 1
    for family in ("linear", "gbdt", "mlp"):
        config["agents"][family]["count"] = 0
    return config


def test_regime_config_scales_liquidity_without_mutating_baseline() -> None:
    base = _short_config()
    original_mean = base["exogenous"]["liquidity"]["mean"]
    original_minimum = base["exogenous"]["liquidity"]["minimum"]

    varied = regime_config(
        base, overlap=0.25, liquidity_scale=0.5, drawdown_limit=0.1
    )

    assert varied["agents"]["feature_overlap"] == 0.25
    assert varied["agents"]["risk"]["max_drawdown"] == 0.1
    assert varied["exogenous"]["liquidity"]["mean"] == 0.5 * original_mean
    assert varied["exogenous"]["liquidity"]["minimum"] == 0.5 * original_minimum
    assert base["exogenous"]["liquidity"]["mean"] == original_mean
    assert base["exogenous"]["liquidity"]["minimum"] == original_minimum


def test_grid_runner_runs_one_by_one_by_two_grid(tmp_path: Path) -> None:
    config_path = tmp_path / "short.yaml"
    config_path.write_text(yaml.safe_dump(_short_config()), encoding="utf-8")

    result = run_regime_grid(
        config_path,
        overlaps=[0.5],
        liquidity_scales=[1.0],
        drawdown_limits=[0.1, 0.2],
        n_episodes=2,
        seed_start=17,
    )

    assert len(result.cells) == 2
    assert [cell.drawdown_limit for cell in result.cells] == [0.1, 0.2]
    assert all([episode.seed for episode in cell.episodes] == [17, 18] for cell in result.cells)
    assert all(len(cell.episodes) == 2 for cell in result.cells)
    assert all(
        0.0 <= episode.diagnostics.forced_share_of_absolute_agent_flow <= 1.0
        for cell in result.cells
        for episode in cell.episodes
    )


def test_cli_writes_one_by_one_by_two_grid_report(tmp_path: Path) -> None:
    config_path = tmp_path / "short.yaml"
    report_path = tmp_path / "regime_grid.md"
    config_path.write_text(yaml.safe_dump(_short_config()), encoding="utf-8")

    main(
        [
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--overlaps",
            "0.5",
            "--liquidity-scales",
            "1.0",
            "--drawdown-limits",
            "0.1",
            "0.2",
            "--n-episodes",
            "2",
            "--seed-start",
            "23",
        ]
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# Cascade regime grid" in report
    assert "Grid: 1 x 1 x 2 = 2 cells" in report
    assert "same seeds (`23-24`)" in report
    assert "Forced-step fraction" in report
    assert "C^pos p50" in report
    assert "| 0.5 | 1 | 0.1 |" in report
    assert "| 0.5 | 1 | 0.2 |" in report
