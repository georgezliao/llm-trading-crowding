"""Tests for the pre-registered impact x leverage selector and config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from herdlab.experiments.impact_leverage_grid import (
    CellSummary,
    eligible_cells,
    experiment_config,
    grid_config,
    select_cell,
)


PROJECT_ROOT = Path(__file__).parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
QUANTILES = {0.0: 0.0, 0.25: 0.25, 0.5: 0.5, 0.75: 0.75, 1.0: 1.0}


def _cell(
    *,
    impact: float,
    multiplier: float,
    drawdown: float = 0.20,
    episodes: int = 10,
    forced_episodes: int = 5,
    forced_step_fraction: float = 0.10,
    median_drawdown: float = 0.30,
    equity_valid: bool = True,
) -> CellSummary:
    return CellSummary(
        impact_sensitivity=impact,
        leverage_multiplier=multiplier,
        drawdown_limit=drawdown,
        episode_count=episodes,
        forced_episode_count=forced_episodes,
        mean_forced_step_fraction=forced_step_fraction,
        max_simultaneously_forced_agents=2,
        mean_simultaneously_forced_agents=0.5,
        mean_forced_share_of_absolute_agent_flow=0.2,
        mean_agent_flow_share_of_total_absolute_flow=0.3,
        mean_price_max_drawdown=median_drawdown,
        median_price_max_drawdown=median_drawdown,
        mean_fraction_positive_windows_with_forced_flow=0.4,
        position_crowding_quantiles=QUANTILES,
        flow_crowding_quantiles=QUANTILES,
        minimum_agent_equity=1.0 if equity_valid else -1.0,
        all_equity_finite_and_nonnegative=equity_valid,
    )


def test_selector_applies_preregistered_rule_and_tie_breaks() -> None:
    cells = [
        _cell(impact=0.01, multiplier=5.0),
        _cell(impact=0.003, multiplier=20.0),
        _cell(impact=0.003, multiplier=5.0),
        _cell(impact=0.001, multiplier=20.0, forced_episodes=4),
        _cell(impact=0.001, multiplier=5.0, drawdown=0.10),
        _cell(impact=0.001, multiplier=1.0, forced_step_fraction=0.100001),
        _cell(impact=0.003, multiplier=1.0, median_drawdown=0.300001),
        _cell(impact=0.01, multiplier=1.0, equity_valid=False),
    ]

    eligible = eligible_cells(cells)
    assert [(cell.impact_sensitivity, cell.leverage_multiplier) for cell in eligible] == [
        (0.003, 5.0),
        (0.003, 20.0),
        (0.01, 5.0),
    ]
    assert select_cell(cells) is eligible[0]


def test_selector_returns_none_when_no_cell_is_eligible() -> None:
    cells = [
        _cell(impact=0.001, multiplier=1.0, forced_episodes=4),
        _cell(impact=0.003, multiplier=5.0, episodes=9),
        _cell(impact=0.01, multiplier=20.0, equity_valid=False),
    ]

    assert eligible_cells(cells) == ()
    assert select_cell(cells) is None


def test_grid_and_experiment_configs_change_only_preregistered_values() -> None:
    baseline = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    original = deepcopy(baseline)
    cell_config = grid_config(
        baseline,
        impact_sensitivity=0.03,
        leverage_multiplier=5.0,
        drawdown_limit=0.10,
    )

    assert baseline == original
    assert cell_config["market"]["impact_sensitivity"] == 0.03
    assert cell_config["agents"]["q_max"] == 50.0
    assert cell_config["agents"]["order_limit"] == 10.0
    assert cell_config["agents"]["risk"]["max_drawdown"] == 0.10

    selected = _cell(impact=0.003, multiplier=5.0)
    generated = experiment_config(baseline, selected)
    expected = deepcopy(baseline)
    expected["market"]["impact_sensitivity"] = 0.003
    expected["agents"]["q_max"] = 50.0
    expected["agents"]["order_limit"] = 10.0
    assert generated == expected
    assert baseline == original
