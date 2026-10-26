"""Run the pre-registered impact x leverage x drawdown diagnostic grid.

The command-line entry point deliberately exposes no options that can alter
the grid, seed set, episode count, diagnostic threshold, or selection rule.
Those values were fixed in ``docs/EXPERIMENTS.md`` before this experiment was
run.  Private simulator state is consumed only for diagnostics and selection;
it is never exposed through the observer data path.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import product
import multiprocessing
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from herdlab.evaluation.diagnostics import (
    CascadeDiagnostics,
    episode_cascade_diagnostics,
)
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.inference.targets import fit_unwind_threshold, unwind_labels


DEFAULT_CONFIG = Path("configs/baseline.yaml")
DEFAULT_REPORT = Path("docs/reports/impact_leverage_grid.md")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment_zero.yaml")

# Pre-registered values.  Keep these synchronized with docs/EXPERIMENTS.md.
IMPACT_SENSITIVITIES = (0.001, 0.003, 0.01, 0.03, 0.1)
LEVERAGE_MULTIPLIERS = (1.0, 5.0, 20.0)
DRAWDOWN_LIMITS = (0.05, 0.10, 0.20)
SEEDS = tuple(range(10))
THRESHOLD_QUANTILE = 0.95
REPORT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)

# Pre-registered eligibility and selection rule.
ELIGIBLE_DRAWDOWN_LIMIT = 0.20
REQUIRED_EPISODE_COUNT = 10
MIN_FORCED_EPISODES = 5
MAX_MEAN_FORCED_STEP_FRACTION = 0.10
MAX_MEDIAN_PRICE_DRAWDOWN = 0.30


@dataclass(frozen=True)
class GridEpisode:
    """One seeded episode and its private cascade diagnostics."""

    seed: int
    diagnostics: CascadeDiagnostics
    has_nonzero_forced_flow: bool
    minimum_agent_equity: float
    equity_is_finite_and_nonnegative: bool


@dataclass(frozen=True)
class ImpactLeverageCell:
    """The ten matched-seed episodes for one pre-registered grid cell."""

    impact_sensitivity: float
    leverage_multiplier: float
    drawdown_limit: float
    episodes: tuple[GridEpisode, ...]


@dataclass(frozen=True)
class CellSummary:
    """Cell-level values used by both reporting and mechanical selection."""

    impact_sensitivity: float
    leverage_multiplier: float
    drawdown_limit: float
    episode_count: int
    forced_episode_count: int
    mean_forced_step_fraction: float
    max_simultaneously_forced_agents: int
    mean_simultaneously_forced_agents: float
    mean_forced_share_of_absolute_agent_flow: float
    mean_agent_flow_share_of_total_absolute_flow: float
    mean_price_max_drawdown: float
    median_price_max_drawdown: float
    mean_fraction_positive_windows_with_forced_flow: float
    position_crowding_quantiles: Mapping[float, float]
    flow_crowding_quantiles: Mapping[float, float]
    minimum_agent_equity: float
    all_equity_finite_and_nonnegative: bool


@dataclass(frozen=True)
class ImpactLeverageGridRun:
    """A complete diagnostic grid and the fixed threshold it used."""

    threshold: float
    cells: tuple[ImpactLeverageCell, ...]
    summaries: tuple[CellSummary, ...]
    eligible: tuple[CellSummary, ...]
    selected: CellSummary | None


@dataclass(frozen=True)
class _RawEpisode:
    impact_sensitivity: float
    leverage_multiplier: float
    drawdown_limit: float
    seed: int
    diagnostics_without_labels: CascadeDiagnostics
    has_nonzero_forced_flow: bool
    minimum_agent_equity: float
    equity_is_finite_and_nonnegative: bool
    agent_net_flow: np.ndarray
    liquidity: np.ndarray
    forced_flow_steps: np.ndarray


def grid_config(
    base_config: Mapping[str, Any],
    *,
    impact_sensitivity: float,
    leverage_multiplier: float,
    drawdown_limit: float,
) -> dict[str, Any]:
    """Apply exactly one pre-registered cell over an isolated baseline copy."""

    impact = _finite_nonnegative(impact_sensitivity, "impact_sensitivity")
    multiplier = _positive_finite(leverage_multiplier, "leverage_multiplier")
    drawdown = _drawdown_value(drawdown_limit)
    config = deepcopy(dict(base_config))
    config["market"]["impact_sensitivity"] = impact
    config["agents"]["q_max"] = float(config["agents"]["q_max"]) * multiplier
    config["agents"]["order_limit"] = (
        float(config["agents"]["order_limit"]) * multiplier
    )
    config["agents"]["risk"]["max_drawdown"] = drawdown
    return config


def run_impact_leverage_grid(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    workers: int = 1,
    impact_sensitivities: Sequence[float] = IMPACT_SENSITIVITIES,
    leverage_multipliers: Sequence[float] = LEVERAGE_MULTIPLIERS,
    drawdown_limits: Sequence[float] = DRAWDOWN_LIMITS,
    seeds: Sequence[int] = SEEDS,
    threshold_quantile: float = THRESHOLD_QUANTILE,
) -> ImpactLeverageGridRun:
    """Run a diagnostic grid, using explicit tasks and deterministic seeds.

    The optional axes support focused tests.  The CLI always calls this with
    the exact pre-registered module constants above.
    """

    worker_count = _positive_integer(workers, "workers")
    impacts = _unique_axis(
        impact_sensitivities,
        "impact_sensitivities",
        lambda value: _finite_nonnegative(value, "impact_sensitivity"),
    )
    multipliers = _unique_axis(
        leverage_multipliers,
        "leverage_multipliers",
        lambda value: _positive_finite(value, "leverage_multiplier"),
    )
    drawdowns = _unique_axis(
        drawdown_limits, "drawdown_limits", _drawdown_value
    )
    seed_values = tuple(_nonnegative_integer(seed, "seed") for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be nonempty and unique")
    probability = float(threshold_quantile)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("threshold_quantile must be finite and in [0, 1]")

    base_config = load_config(config_path)
    horizon = int(base_config["targets"]["unwind_horizon"])
    tasks = [
        (
            grid_config(
                base_config,
                impact_sensitivity=impact,
                leverage_multiplier=multiplier,
                drawdown_limit=drawdown,
            ),
            impact,
            multiplier,
            drawdown,
            seed,
            horizon,
        )
        for impact, multiplier, drawdown, seed in product(
            impacts, multipliers, drawdowns, seed_values
        )
    ]
    if worker_count == 1:
        raw_episodes = [_run_grid_episode(task) for task in tasks]
    else:
        with multiprocessing.Pool(processes=worker_count) as pool:
            raw_episodes = pool.map(_run_grid_episode, tasks)

    threshold = fit_unwind_threshold(
        (
            (episode.agent_net_flow, episode.liquidity)
            for episode in raw_episodes
        ),
        horizon=horizon,
        quantile=probability,
    )

    completed: list[GridEpisode] = []
    for raw in raw_episodes:
        labels = unwind_labels(
            raw.agent_net_flow,
            raw.liquidity,
            horizon=horizon,
            threshold=threshold,
        )
        positive_indices = np.flatnonzero(labels)
        positive_with_forced = np.asarray(
            [
                np.any(raw.forced_flow_steps[index + 1 : index + horizon + 1])
                for index in positive_indices
            ],
            dtype=np.bool_,
        )
        fraction = (
            float(np.mean(positive_with_forced))
            if positive_with_forced.size
            else 0.0
        )
        completed.append(
            GridEpisode(
                seed=raw.seed,
                diagnostics=replace(
                    raw.diagnostics_without_labels,
                    fraction_positive_windows_with_forced_flow=fraction,
                ),
                has_nonzero_forced_flow=raw.has_nonzero_forced_flow,
                minimum_agent_equity=raw.minimum_agent_equity,
                equity_is_finite_and_nonnegative=(
                    raw.equity_is_finite_and_nonnegative
                ),
            )
        )

    cells: list[ImpactLeverageCell] = []
    episode_index = 0
    for impact, multiplier, drawdown in product(impacts, multipliers, drawdowns):
        count = len(seed_values)
        cells.append(
            ImpactLeverageCell(
                impact_sensitivity=impact,
                leverage_multiplier=multiplier,
                drawdown_limit=drawdown,
                episodes=tuple(completed[episode_index : episode_index + count]),
            )
        )
        episode_index += count

    cell_values = tuple(cells)
    summaries = tuple(summarize_cell(cell) for cell in cell_values)
    eligible = eligible_cells(summaries)
    return ImpactLeverageGridRun(
        threshold=threshold,
        cells=cell_values,
        summaries=summaries,
        eligible=eligible,
        selected=select_cell(summaries),
    )


def summarize_cell(cell: ImpactLeverageCell) -> CellSummary:
    """Aggregate independent episode diagnostics for one cell."""

    if not cell.episodes:
        raise ValueError("grid cells must contain at least one episode")
    diagnostics = tuple(episode.diagnostics for episode in cell.episodes)

    def mean(attribute: str) -> float:
        return float(np.mean([getattr(item, attribute) for item in diagnostics]))

    return CellSummary(
        impact_sensitivity=cell.impact_sensitivity,
        leverage_multiplier=cell.leverage_multiplier,
        drawdown_limit=cell.drawdown_limit,
        episode_count=len(cell.episodes),
        forced_episode_count=sum(
            episode.has_nonzero_forced_flow for episode in cell.episodes
        ),
        mean_forced_step_fraction=mean("fraction_steps_with_forced_order"),
        max_simultaneously_forced_agents=max(
            item.max_simultaneously_forced_agents for item in diagnostics
        ),
        mean_simultaneously_forced_agents=mean(
            "mean_simultaneously_forced_agents"
        ),
        mean_forced_share_of_absolute_agent_flow=mean(
            "forced_share_of_absolute_agent_flow"
        ),
        mean_agent_flow_share_of_total_absolute_flow=mean(
            "agent_flow_share_of_total_absolute_flow"
        ),
        mean_price_max_drawdown=mean("price_max_drawdown"),
        median_price_max_drawdown=float(
            np.median([item.price_max_drawdown for item in diagnostics])
        ),
        mean_fraction_positive_windows_with_forced_flow=mean(
            "fraction_positive_windows_with_forced_flow"
        ),
        position_crowding_quantiles={
            quantile: float(
                np.mean(
                    [item.position_crowding_quantiles[quantile] for item in diagnostics]
                )
            )
            for quantile in REPORT_QUANTILES
        },
        flow_crowding_quantiles={
            quantile: float(
                np.mean(
                    [item.flow_crowding_quantiles[quantile] for item in diagnostics]
                )
            )
            for quantile in REPORT_QUANTILES
        },
        minimum_agent_equity=min(
            episode.minimum_agent_equity for episode in cell.episodes
        ),
        all_equity_finite_and_nonnegative=all(
            episode.equity_is_finite_and_nonnegative for episode in cell.episodes
        ),
    )


def is_eligible(cell: CellSummary) -> bool:
    """Apply the pre-registered eligibility rule without judgment or tuning."""

    return (
        cell.drawdown_limit == ELIGIBLE_DRAWDOWN_LIMIT
        and cell.episode_count == REQUIRED_EPISODE_COUNT
        and cell.forced_episode_count >= MIN_FORCED_EPISODES
        and cell.mean_forced_step_fraction <= MAX_MEAN_FORCED_STEP_FRACTION
        and cell.median_price_max_drawdown <= MAX_MEDIAN_PRICE_DRAWDOWN
        and cell.all_equity_finite_and_nonnegative
    )


def eligible_cells(cells: Sequence[CellSummary]) -> tuple[CellSummary, ...]:
    """Return eligible cells in deterministic selection priority order."""

    return tuple(
        sorted(
            (cell for cell in cells if is_eligible(cell)),
            key=lambda cell: (
                cell.impact_sensitivity,
                cell.leverage_multiplier,
            ),
        )
    )


def select_cell(cells: Sequence[CellSummary]) -> CellSummary | None:
    """Select lowest impact, breaking ties by lowest leverage multiplier."""

    eligible = eligible_cells(cells)
    return eligible[0] if eligible else None


def experiment_config(
    base_config: Mapping[str, Any], selected: CellSummary
) -> dict[str, Any]:
    """Create baseline with only the selected impact and leverage scaling."""

    config = deepcopy(dict(base_config))
    config["market"]["impact_sensitivity"] = selected.impact_sensitivity
    config["agents"]["q_max"] = (
        float(config["agents"]["q_max"]) * selected.leverage_multiplier
    )
    config["agents"]["order_limit"] = (
        float(config["agents"]["order_limit"]) * selected.leverage_multiplier
    )
    return config


def write_experiment_config(
    path: str | Path,
    base_config: Mapping[str, Any],
    selected: CellSummary,
) -> Path:
    """Write the selected baseline-derived Experiment Zero configuration."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by the pre-registered impact x leverage grid.\n"
        "# The selected values must be applied identically to every v2 split.\n"
    )
    output.write_text(
        header
        + yaml.safe_dump(
            experiment_config(base_config, selected),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return output


def render_report(
    result: ImpactLeverageGridRun,
    *,
    config_path: str | Path,
    workers: int,
    horizon: int,
) -> str:
    """Render the complete map, eligible cells, and mechanical selection."""

    if len(result.cells) != len(result.summaries):
        raise ValueError("result cells and summaries must have equal length")
    lines = [
        "# Pre-registered impact x leverage x drawdown grid",
        "",
        "This report describes synthetic output from the uncalibrated HerdLab simulator. It is not evidence about real market behavior. The grid and selection rule were pre-registered in `docs/EXPERIMENTS.md` before this run.",
        "",
        "## Design",
        "",
        f"- Base config: `{config_path}`",
        f"- Grid: 5 x 3 x 3 = {len(result.cells)} cells; 10 episodes per cell; shared seeds `0-9` ({sum(len(cell.episodes) for cell in result.cells)} total episodes)",
        f"- Impact sensitivities: `{_values_text(IMPACT_SENSITIVITIES)}`",
        f"- Leverage multipliers: `{_values_text(LEVERAGE_MULTIPLIERS)}`; each multiplier scales both baseline `q_max` and `order_limit`",
        f"- Drawdown limits: `{_values_text(DRAWDOWN_LIMITS)}`",
        f"- Parallel workers: {workers}",
        f"- Unwind horizon: {horizon} steps. The step-1 diagnostic threshold is the pooled score quantile {THRESHOLD_QUANTILE:.2f}, with fitted value `{result.threshold:.10g}`; this map makes no held-out prediction claim.",
        "- Each diagnostic value is the unweighted mean of episode-level values, except maximum agents forced (cell maximum), median max drawdown (episode median), and minimum equity (minimum over every agent-time value).",
        "",
        "## Pre-registered eligibility and selection",
        "",
        "A cell is eligible only when `max_drawdown = 0.20`, at least 5/10 episodes have nonzero forced flow, mean forced-step fraction is at most 0.10, median price max drawdown is at most 0.30, and every agent-equity value is finite and nonnegative. Selection takes the lowest impact sensitivity, breaking ties by the lowest leverage multiplier.",
        "",
    ]
    if result.eligible:
        lines.extend(
            [
                f"Eligible cells: {len(result.eligible)}.",
                "",
                "| Impact | Leverage m | Forced episodes | Mean forced-step fraction | Median max drawdown | Equity valid |",
                "| ---: | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        for cell in result.eligible:
            lines.append(
                f"| {cell.impact_sensitivity:g} | {cell.leverage_multiplier:g} | "
                f"{cell.forced_episode_count}/{cell.episode_count} | "
                f"{cell.mean_forced_step_fraction:.6f} | "
                f"{cell.median_price_max_drawdown:.6f} | yes |"
            )
        selected = result.selected
        if selected is None:
            raise ValueError("eligible cells require a selected cell")
        lines.extend(
            [
                "",
                f"Selected cell: impact sensitivity `{selected.impact_sensitivity:g}`, leverage multiplier `{selected.leverage_multiplier:g}`, drawdown limit `{selected.drawdown_limit:g}`.",
                "",
                "`configs/experiment_zero.yaml` was generated from the baseline by changing only `market.impact_sensitivity` and scaling both `agents.q_max` and `agents.order_limit` by the selected multiplier. The selected values are to be applied identically to training and every test split.",
            ]
        )
    else:
        lines.extend(
            [
                "Eligible cells: 0.",
                "",
                "Selection result: **null**. No cell satisfied the pre-registered eligibility rule, so `configs/experiment_zero.yaml` was not created.",
            ]
        )

    lines.extend(
        [
            "",
            "## Full diagnostic map",
            "",
            "| Impact | Leverage m | Drawdown limit | Forced episodes | Forced-step fraction | Max agents forced | Mean agents forced | Forced / agent abs. flow | Agent / total abs. flow | Mean max drawdown | Median max drawdown | Y=1 windows with forced flow | Min equity | Equity valid | Eligible |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
        ]
    )
    eligible_set = {
        (cell.impact_sensitivity, cell.leverage_multiplier, cell.drawdown_limit)
        for cell in result.eligible
    }
    for cell in result.summaries:
        key = (
            cell.impact_sensitivity,
            cell.leverage_multiplier,
            cell.drawdown_limit,
        )
        lines.append(
            f"| {cell.impact_sensitivity:g} | {cell.leverage_multiplier:g} | "
            f"{cell.drawdown_limit:g} | "
            f"{cell.forced_episode_count}/{cell.episode_count} | "
            f"{cell.mean_forced_step_fraction:.6f} | "
            f"{cell.max_simultaneously_forced_agents} | "
            f"{cell.mean_simultaneously_forced_agents:.6f} | "
            f"{cell.mean_forced_share_of_absolute_agent_flow:.6f} | "
            f"{cell.mean_agent_flow_share_of_total_absolute_flow:.6f} | "
            f"{cell.mean_price_max_drawdown:.6f} | "
            f"{cell.median_price_max_drawdown:.6f} | "
            f"{cell.mean_fraction_positive_windows_with_forced_flow:.6f} | "
            f"{cell.minimum_agent_equity:.6f} | "
            f"{'yes' if cell.all_equity_finite_and_nonnegative else 'no'} | "
            f"{'yes' if key in eligible_set else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Crowding distributions by cell",
            "",
            "| Impact | Leverage m | Drawdown limit | C^pos min | C^pos p25 | C^pos p50 | C^pos p75 | C^pos max | C^flow min | C^flow p25 | C^flow p50 | C^flow p75 | C^flow max |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cell in result.summaries:
        position = " | ".join(
            f"{cell.position_crowding_quantiles[quantile]:.6f}"
            for quantile in REPORT_QUANTILES
        )
        flow = " | ".join(
            f"{cell.flow_crowding_quantiles[quantile]:.6f}"
            for quantile in REPORT_QUANTILES
        )
        lines.append(
            f"| {cell.impact_sensitivity:g} | {cell.leverage_multiplier:g} | "
            f"{cell.drawdown_limit:g} | {position} | {flow} |"
        )

    lines.extend(
        [
            "",
            "Forced flow, crowding, and equity checks use private simulator ground truth only for this diagnostic experiment; they are not observer inputs. All 45 cells are retained in the map, including ineligible cells.",
            "",
            "## Deviations and open questions",
            "",
            "No deviations from the pre-registered grid, thresholds, seeds, episode count, eligibility rule, or selection rule. No open questions.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """Run exactly the pre-registered grid and write its report/config."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--experiment-config", default=str(DEFAULT_EXPERIMENT_CONFIG)
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    result = run_impact_leverage_grid(args.config, workers=args.workers)
    base_config = load_config(args.config)
    report = render_report(
        result,
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


def _run_grid_episode(
    task: tuple[dict[str, Any], float, float, float, int, int],
) -> _RawEpisode:
    config, impact, multiplier, drawdown, seed, horizon = task
    public, private = run_episode(config, seed)
    empty_labels = np.zeros(len(private.orders) - horizon, dtype=np.bool_)
    diagnostics = episode_cascade_diagnostics(
        public,
        private,
        empty_labels,
        horizon=horizon,
        quantiles=REPORT_QUANTILES,
    )
    forced_flow_steps = np.any(
        private.forced & (np.abs(private.orders) > 0.0), axis=1
    )
    equity = np.asarray(private.equity, dtype=np.float64)
    return _RawEpisode(
        impact_sensitivity=impact,
        leverage_multiplier=multiplier,
        drawdown_limit=drawdown,
        seed=seed,
        diagnostics_without_labels=diagnostics,
        has_nonzero_forced_flow=bool(np.any(forced_flow_steps)),
        minimum_agent_equity=float(np.min(equity)),
        equity_is_finite_and_nonnegative=bool(
            np.all(np.isfinite(equity)) and np.all(equity >= 0.0)
        ),
        agent_net_flow=np.asarray(private.agent_net_flow, dtype=np.float64),
        liquidity=np.asarray(public.liquidity_proxy, dtype=np.float64),
        forced_flow_steps=np.asarray(forced_flow_steps, dtype=np.bool_),
    )


def _unique_axis(
    values: Sequence[float], name: str, validator: Any
) -> tuple[float, ...]:
    result = tuple(validator(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _drawdown_value(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("drawdown_limit must be finite and in [0, 1)")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _values_text(values: Sequence[float]) -> str:
    return ", ".join(f"{value:g}" for value in values)


if __name__ == "__main__":
    main()


__all__ = [
    "CellSummary",
    "GridEpisode",
    "ImpactLeverageCell",
    "ImpactLeverageGridRun",
    "eligible_cells",
    "experiment_config",
    "grid_config",
    "is_eligible",
    "main",
    "render_report",
    "run_impact_leverage_grid",
    "select_cell",
    "summarize_cell",
    "write_experiment_config",
]
