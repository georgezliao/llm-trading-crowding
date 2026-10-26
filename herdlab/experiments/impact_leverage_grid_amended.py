"""Rerun the pre-registered grid under Amendment 1 and compare results.

The grid, episode seeds, eligibility rule, and selection rule are imported
unchanged from :mod:`herdlab.experiments.impact_leverage_grid`.  This wrapper
only changes the report destination and adds the required comparison with the
original grid report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from herdlab.experiments.impact_leverage_grid import (
    DEFAULT_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    ELIGIBLE_DRAWDOWN_LIMIT,
    CellSummary,
    ImpactLeverageGridRun,
    render_report,
    run_impact_leverage_grid,
    write_experiment_config,
)
from herdlab.experiments.run_episode import load_config


DEFAULT_ORIGINAL_REPORT = Path("docs/reports/impact_leverage_grid.md")
DEFAULT_AMENDED_REPORT = Path(
    "docs/reports/impact_leverage_grid_amended.md"
)


@dataclass(frozen=True)
class OriginalDrawdownRow:
    """Selection-relevant values parsed from one original report row."""

    impact_sensitivity: float
    leverage_multiplier: float
    forced_episodes: str
    forced_step_fraction: float
    forced_share_of_absolute_agent_flow: float
    median_price_max_drawdown: float
    eligible: str


def parse_original_drawdown_rows(
    report: str,
    *,
    drawdown_limit: float = ELIGIBLE_DRAWDOWN_LIMIT,
) -> tuple[OriginalDrawdownRow, ...]:
    """Parse the requested drawdown slice from the original full-map table."""

    lines = report.splitlines()
    header_prefix = "| Impact | Leverage m | Drawdown limit |"
    try:
        header_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith(header_prefix)
        )
    except StopIteration as error:
        raise ValueError("original report has no full diagnostic map") from error

    header = _markdown_cells(lines[header_index])
    rows: list[OriginalDrawdownRow] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        values = _markdown_cells(line)
        if len(values) != len(header):
            raise ValueError("original full-map row does not match its header")
        row = dict(zip(header, values, strict=True))
        if float(row["Drawdown limit"]) != drawdown_limit:
            continue
        rows.append(
            OriginalDrawdownRow(
                impact_sensitivity=float(row["Impact"]),
                leverage_multiplier=float(row["Leverage m"]),
                forced_episodes=row["Forced episodes"],
                forced_step_fraction=float(row["Forced-step fraction"]),
                forced_share_of_absolute_agent_flow=float(
                    row["Forced / agent abs. flow"]
                ),
                median_price_max_drawdown=float(row["Median max drawdown"]),
                eligible=row["Eligible"],
            )
        )
    if not rows:
        raise ValueError(
            f"original report has no drawdown-limit {drawdown_limit:g} rows"
        )
    return tuple(rows)


def render_amended_report(
    result: ImpactLeverageGridRun,
    *,
    original_report: str,
    config_path: str | Path,
    workers: int,
    horizon: int,
) -> str:
    """Render the full amended map plus the pre-amendment comparison."""

    report = render_report(
        result,
        config_path=config_path,
        workers=workers,
        horizon=horizon,
    )
    report = report.replace(
        "# Pre-registered impact x leverage x drawdown grid",
        "# Pre-registered impact x leverage x drawdown grid, Amendment 1 rerun",
        1,
    )
    report = report.replace(
        "The grid and selection rule were pre-registered in "
        "`docs/EXPERIMENTS.md` before this run.",
        "This is the Amendment 1 rerun specified in `docs/EXPERIMENTS.md`; "
        "the grid, seeds, episode count, eligibility rule, and selection rule "
        "are unchanged from the original run.",
        1,
    )
    amendment_line = (
        "- Model version: Amendment 1 adds the pre-specified 20-step "
        "post-liquidation cooldown, peak-equity reset on resumption, and "
        "nonzero-quantity condition for forced-order labels."
    )
    report = report.replace(
        f"- Base config: `{config_path}`",
        f"- Base config: `{config_path}`\n{amendment_line}",
        1,
    )

    comparison = render_drawdown_comparison(
        result.summaries,
        parse_original_drawdown_rows(original_report),
    )
    marker = "\n## Crowding distributions by cell\n"
    if marker not in report:
        raise ValueError("amended report has no crowding-distribution section")
    return report.replace(marker, f"\n{comparison}\n{marker}", 1)


def render_drawdown_comparison(
    amended_summaries: Sequence[CellSummary],
    original_rows: Sequence[OriginalDrawdownRow],
) -> str:
    """Render selection-relevant dd=0.20 rows before and after Amendment 1."""

    amended = {
        (cell.impact_sensitivity, cell.leverage_multiplier): cell
        for cell in amended_summaries
        if cell.drawdown_limit == ELIGIBLE_DRAWDOWN_LIMIT
    }
    original = {
        (row.impact_sensitivity, row.leverage_multiplier): row
        for row in original_rows
    }
    if amended.keys() != original.keys():
        raise ValueError(
            "original and amended dd=0.20 rows must have identical cells"
        )

    lines = [
        "## dd=0.20 side-by-side with the original grid",
        "",
        "These are the 15 rows to which the unchanged eligibility rule can apply. `Original` values come directly from `docs/reports/impact_leverage_grid.md`; `amended` values are from this rerun.",
        "",
        "| Impact | Leverage m | Forced episodes original | amended | Forced-step fraction original | amended | Forced / agent abs. flow original | amended | Median max drawdown original | amended | Eligible original | amended |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for key in sorted(amended):
        old = original[key]
        new = amended[key]
        new_eligible = "yes" if _is_selected_as_eligible(new) else "no"
        lines.append(
            f"| {new.impact_sensitivity:g} | {new.leverage_multiplier:g} | "
            f"{old.forced_episodes} | "
            f"{new.forced_episode_count}/{new.episode_count} | "
            f"{old.forced_step_fraction:.6f} | "
            f"{new.mean_forced_step_fraction:.6f} | "
            f"{old.forced_share_of_absolute_agent_flow:.6f} | "
            f"{new.mean_forced_share_of_absolute_agent_flow:.6f} | "
            f"{old.median_price_max_drawdown:.6f} | "
            f"{new.median_price_max_drawdown:.6f} | "
            f"{old.eligible} | {new_eligible} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the unchanged grid under the amended model and write its outputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--original-report", default=str(DEFAULT_ORIGINAL_REPORT))
    parser.add_argument("--report", default=str(DEFAULT_AMENDED_REPORT))
    parser.add_argument(
        "--experiment-config", default=str(DEFAULT_EXPERIMENT_CONFIG)
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    result = run_impact_leverage_grid(args.config, workers=args.workers)
    base_config = load_config(args.config)
    report = render_amended_report(
        result,
        original_report=Path(args.original_report).read_text(encoding="utf-8"),
        config_path=args.config,
        workers=args.workers,
        horizon=int(base_config["targets"]["unwind_horizon"]),
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    if result.selected is not None:
        write_experiment_config(
            args.experiment_config,
            base_config,
            result.selected,
        )


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_selected_as_eligible(cell: CellSummary) -> bool:
    # Import locally to emphasize that this is the unchanged step-9b rule.
    from herdlab.experiments.impact_leverage_grid import is_eligible

    return is_eligible(cell)


if __name__ == "__main__":
    main()


__all__ = [
    "OriginalDrawdownRow",
    "main",
    "parse_original_drawdown_rows",
    "render_amended_report",
    "render_drawdown_comparison",
]
