"""Tests for the Amendment 1 grid report wrapper."""

from __future__ import annotations

from pathlib import Path

from herdlab.experiments.impact_leverage_grid import CellSummary
from herdlab.experiments.impact_leverage_grid_amended import (
    parse_original_drawdown_rows,
    render_drawdown_comparison,
)


PROJECT_ROOT = Path(__file__).parents[1]
ORIGINAL_REPORT = PROJECT_ROOT / "docs" / "reports" / "impact_leverage_grid.md"
QUANTILES = {0.0: 0.0, 0.25: 0.25, 0.5: 0.5, 0.75: 0.75, 1.0: 1.0}


def _summary(impact: float, multiplier: float) -> CellSummary:
    return CellSummary(
        impact_sensitivity=impact,
        leverage_multiplier=multiplier,
        drawdown_limit=0.20,
        episode_count=10,
        forced_episode_count=6,
        mean_forced_step_fraction=0.02,
        max_simultaneously_forced_agents=2,
        mean_simultaneously_forced_agents=0.1,
        mean_forced_share_of_absolute_agent_flow=0.03,
        mean_agent_flow_share_of_total_absolute_flow=0.4,
        mean_price_max_drawdown=0.2,
        median_price_max_drawdown=0.2,
        mean_fraction_positive_windows_with_forced_flow=0.5,
        position_crowding_quantiles=QUANTILES,
        flow_crowding_quantiles=QUANTILES,
        minimum_agent_equity=1.0,
        all_equity_finite_and_nonnegative=True,
    )


def test_parse_original_report_returns_all_dd_020_cells() -> None:
    rows = parse_original_drawdown_rows(
        ORIGINAL_REPORT.read_text(encoding="utf-8")
    )

    assert len(rows) == 15
    assert (rows[0].impact_sensitivity, rows[0].leverage_multiplier) == (
        0.001,
        1.0,
    )
    assert rows[0].forced_episodes == "0/10"
    assert rows[-1].median_price_max_drawdown == 0.251561
    assert all(row.eligible == "no" for row in rows)


def test_comparison_uses_unchanged_eligibility_rule() -> None:
    original = """| Impact | Leverage m | Drawdown limit | Forced episodes | Forced-step fraction | Max agents forced | Mean agents forced | Forced / agent abs. flow | Agent / total abs. flow | Mean max drawdown | Median max drawdown | Y=1 windows with forced flow | Min equity | Equity valid | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 0.003 | 5 | 0.2 | 0/10 | 0.000000 | 0 | 0.000000 | 0.000000 | 0.7 | 0.1 | 0.1 | 0.0 | 9000 | yes | no |
"""
    old_rows = parse_original_drawdown_rows(original)
    comparison = render_drawdown_comparison((_summary(0.003, 5.0),), old_rows)

    assert "| 0.003 | 5 | 0/10 | 6/10 |" in comparison
    assert "| no | yes |" in comparison
