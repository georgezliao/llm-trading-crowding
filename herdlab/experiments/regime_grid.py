"""Map simulated cascade diagnostics over a small pre-specified regime grid.

The grid varies feature overlap, the scale of the entire exogenous liquidity
path, and the agents' drawdown limit.  It is descriptive: the same episode
seeds are reused in every cell, the baseline configuration is never edited,
and the results are not used to select a preferred simulator setting.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from herdlab.evaluation.diagnostics import (
    CascadeDiagnostics,
    episode_cascade_diagnostics,
)
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.inference.targets import fit_unwind_threshold, unwind_labels
from herdlab.market.records import PrivateRecord, PublicRecord


DEFAULT_CONFIG = Path("configs/baseline.yaml")
DEFAULT_REPORT = Path("docs/reports/regime_grid.md")
DEFAULT_OVERLAPS = (0.0, 0.5, 1.0)
DEFAULT_LIQUIDITY_SCALES = (0.5, 1.0, 2.0)
DEFAULT_DRAWDOWN_LIMITS = (0.1, 0.2, 0.3)
DEFAULT_N_EPISODES = 5
DEFAULT_SEED_START = 0
DEFAULT_THRESHOLD_QUANTILE = 0.95
REPORT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
MAX_AXIS_VALUES = 3
MAX_EPISODES_PER_CELL = 5


@dataclass(frozen=True)
class GridEpisode:
    """One seeded episode and its private cascade diagnostics."""

    seed: int
    diagnostics: CascadeDiagnostics
    has_nonzero_forced_flow: bool


@dataclass(frozen=True)
class RegimeGridCell:
    """All repeated episodes at one point in the three-dimensional grid."""

    overlap: float
    liquidity_scale: float
    drawdown_limit: float
    episodes: tuple[GridEpisode, ...]

    @property
    def forced_episode_count(self) -> int:
        return sum(episode.has_nonzero_forced_flow for episode in self.episodes)


@dataclass(frozen=True)
class RegimeGridRun:
    """A complete grid and the fixed diagnostic label threshold it used."""

    threshold: float
    cells: tuple[RegimeGridCell, ...]


@dataclass(frozen=True)
class _RawGridEpisode:
    overlap: float
    liquidity_scale: float
    drawdown_limit: float
    seed: int
    public: PublicRecord
    private: PrivateRecord


def regime_config(
    base_config: Mapping[str, Any],
    *,
    overlap: float,
    liquidity_scale: float,
    drawdown_limit: float,
) -> dict[str, Any]:
    """Return an isolated grid-cell config without mutating ``base_config``.

    Liquidity scale multiplies both ``exogenous.liquidity.mean`` and
    ``exogenous.liquidity.minimum``.  Scaling both values preserves the same
    normalized liquidity innovations and floor behavior across matched seeds.
    """

    overlap_value = _overlap_value(overlap)
    scale_value = _positive_finite(liquidity_scale, "liquidity_scale")
    drawdown_value = _drawdown_value(drawdown_limit)

    config = deepcopy(dict(base_config))
    config["agents"]["feature_overlap"] = overlap_value
    config["agents"]["risk"]["max_drawdown"] = drawdown_value
    liquidity = config["exogenous"]["liquidity"]
    liquidity["mean"] = float(liquidity["mean"]) * scale_value
    liquidity["minimum"] = float(liquidity["minimum"]) * scale_value
    return config


def run_regime_grid(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    overlaps: Sequence[float] = DEFAULT_OVERLAPS,
    liquidity_scales: Sequence[float] = DEFAULT_LIQUIDITY_SCALES,
    drawdown_limits: Sequence[float] = DEFAULT_DRAWDOWN_LIMITS,
    n_episodes: int = DEFAULT_N_EPISODES,
    seed_start: int = DEFAULT_SEED_START,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
) -> RegimeGridRun:
    """Run at most a 3x3x3 grid with common seed numbers in every cell."""

    overlap_values = _grid_axis(overlaps, "overlaps", _overlap_value)
    scale_values = _grid_axis(
        liquidity_scales,
        "liquidity_scales",
        lambda value: _positive_finite(value, "liquidity_scale"),
    )
    drawdown_values = _grid_axis(
        drawdown_limits, "drawdown_limits", _drawdown_value
    )
    count = _episode_count(n_episodes)
    first_seed = _nonnegative_integer(seed_start, "seed_start")
    probability = float(threshold_quantile)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("threshold_quantile must be finite and in [0, 1]")

    base_config = load_config(config_path)
    horizon = int(base_config["targets"]["unwind_horizon"])
    raw_episodes: list[_RawGridEpisode] = []
    for overlap, liquidity_scale, drawdown_limit in product(
        overlap_values, scale_values, drawdown_values
    ):
        config = regime_config(
            base_config,
            overlap=overlap,
            liquidity_scale=liquidity_scale,
            drawdown_limit=drawdown_limit,
        )
        for seed in range(first_seed, first_seed + count):
            public, private = run_episode(config, seed)
            raw_episodes.append(
                _RawGridEpisode(
                    overlap=overlap,
                    liquidity_scale=liquidity_scale,
                    drawdown_limit=drawdown_limit,
                    seed=seed,
                    public=public,
                    private=private,
                )
            )

    threshold = fit_unwind_threshold(
        (
            (episode.private.agent_net_flow, episode.public.liquidity_proxy)
            for episode in raw_episodes
        ),
        horizon=horizon,
        quantile=probability,
    )

    cells: list[RegimeGridCell] = []
    for overlap, liquidity_scale, drawdown_limit in product(
        overlap_values, scale_values, drawdown_values
    ):
        cell_episodes: list[GridEpisode] = []
        for raw in raw_episodes:
            if (
                raw.overlap,
                raw.liquidity_scale,
                raw.drawdown_limit,
            ) != (overlap, liquidity_scale, drawdown_limit):
                continue
            labels = unwind_labels(
                raw.private.agent_net_flow,
                raw.public.liquidity_proxy,
                horizon=horizon,
                threshold=threshold,
            )
            has_forced_flow = bool(
                np.any(raw.private.forced & (np.abs(raw.private.orders) > 0.0))
            )
            cell_episodes.append(
                GridEpisode(
                    seed=raw.seed,
                    diagnostics=episode_cascade_diagnostics(
                        raw.public,
                        raw.private,
                        labels,
                        horizon=horizon,
                        quantiles=REPORT_QUANTILES,
                    ),
                    has_nonzero_forced_flow=has_forced_flow,
                )
            )
        cells.append(
            RegimeGridCell(
                overlap=overlap,
                liquidity_scale=liquidity_scale,
                drawdown_limit=drawdown_limit,
                episodes=tuple(cell_episodes),
            )
        )
    return RegimeGridRun(threshold=threshold, cells=tuple(cells))


def render_report(
    result: RegimeGridRun,
    *,
    config_path: str | Path,
    horizon: int,
    threshold_quantile: float,
) -> str:
    """Render every step-1 diagnostic for each grid cell as Markdown."""

    if not result.cells or not result.cells[0].episodes:
        raise ValueError("result must contain cells with episodes")
    seeds = tuple(episode.seed for episode in result.cells[0].episodes)
    overlaps = _ordered_unique(cell.overlap for cell in result.cells)
    scales = _ordered_unique(cell.liquidity_scale for cell in result.cells)
    drawdowns = _ordered_unique(cell.drawdown_limit for cell in result.cells)
    forced_cells = sum(cell.forced_episode_count > 0 for cell in result.cells)
    forced_episodes = sum(cell.forced_episode_count for cell in result.cells)
    total_episodes = sum(len(cell.episodes) for cell in result.cells)

    lines = [
        "# Cascade regime grid",
        "",
        "This report maps synthetic output from the uncalibrated HerdLab simulator. It is not evidence about real market behavior, and the grid is descriptive rather than a parameter-selection exercise. `configs/baseline.yaml` was not changed based on these results.",
        "",
        "## Design",
        "",
        f"- Base config: `{config_path}`",
        f"- Grid: {len(overlaps)} x {len(scales)} x {len(drawdowns)} = {len(result.cells)} cells; {len(seeds)} episodes per cell using the same seeds (`{_seed_text(seeds)}`)",
        f"- Feature-overlap values: `{_values_text(overlaps)}`",
        f"- Liquidity-scale values: `{_values_text(scales)}`. Each multiplier is applied to both the baseline liquidity mean and minimum, preserving matched normalized innovations and floor behavior.",
        f"- Drawdown-limit values: `{_values_text(drawdowns)}`",
        "- Reusing a seed holds return, jump, background-flow, agent-parameter, and normalized liquidity innovations common across cells; the liquidity level changes only by its stated multiplier.",
        "- `feature_overlap` affects the 12 momentum/reversion agents in the current baseline; the 18 learned agents do not consume this knob.",
        f"- Unwind horizon: {horizon} steps. One pooled diagnostic threshold was fitted across all grid episodes at quantile {threshold_quantile:.2f}: `{result.threshold:.10g}`. This report makes no held-out prediction claim.",
        "- Each table value is the unweighted mean of the per-episode step-1 diagnostic, except max agents forced, which is the maximum across episodes. Crowding entries average the per-episode quantile estimates.",
        "",
        "## Observed map",
        "",
        f"Nonzero forced flow occurred in {forced_cells} / {len(result.cells)} cells and {forced_episodes} / {total_episodes} cell-episodes. Cells with zero measured forced flow are retained; no grid value was selected or promoted from this map.",
        "",
        "## Forced-flow and market diagnostics by cell",
        "",
        "| Overlap | Liquidity scale | Drawdown limit | Forced episodes | Forced-step fraction | Max agents forced | Mean agents forced | Forced / agent abs. flow | Agent / total abs. flow | Max drawdown | Y=1 windows with forced flow |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in result.cells:
        summary = _summarize_cell(cell)
        lines.append(
            f"| {cell.overlap:g} | {cell.liquidity_scale:g} | {cell.drawdown_limit:g} | "
            f"{cell.forced_episode_count}/{len(cell.episodes)} | "
            f"{summary.fraction_steps_with_forced_order:.6f} | "
            f"{summary.max_simultaneously_forced_agents} | "
            f"{summary.mean_simultaneously_forced_agents:.6f} | "
            f"{summary.forced_share_of_absolute_agent_flow:.6f} | "
            f"{summary.agent_flow_share_of_total_absolute_flow:.6f} | "
            f"{summary.price_max_drawdown:.6f} | "
            f"{summary.fraction_positive_windows_with_forced_flow:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Crowding distributions by cell",
            "",
            "| Overlap | Liquidity scale | Drawdown limit | C^pos min | C^pos p25 | C^pos p50 | C^pos p75 | C^pos max | C^flow min | C^flow p25 | C^flow p50 | C^flow p75 | C^flow max |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cell in result.cells:
        summary = _summarize_cell(cell)
        position = " | ".join(
            f"{summary.position_crowding_quantiles[quantile]:.6f}"
            for quantile in REPORT_QUANTILES
        )
        flow = " | ".join(
            f"{summary.flow_crowding_quantiles[quantile]:.6f}"
            for quantile in REPORT_QUANTILES
        )
        lines.append(
            f"| {cell.overlap:g} | {cell.liquidity_scale:g} | "
            f"{cell.drawdown_limit:g} | {position} | {flow} |"
        )
    lines.extend(
        [
            "",
            "Forced flow and crowding use private simulator ground truth only for this diagnostic map; they are not observer inputs. Y uses strictly future agent net flow and the single pooled threshold stated above.",
            "",
        ]
    )
    return "\n".join(lines)


def _summarize_cell(cell: RegimeGridCell) -> CascadeDiagnostics:
    if not cell.episodes:
        raise ValueError("grid cells must contain at least one episode")
    diagnostics = tuple(episode.diagnostics for episode in cell.episodes)

    def mean(attribute: str) -> float:
        return float(np.mean([getattr(item, attribute) for item in diagnostics]))

    return CascadeDiagnostics(
        fraction_steps_with_forced_order=mean("fraction_steps_with_forced_order"),
        max_simultaneously_forced_agents=max(
            item.max_simultaneously_forced_agents for item in diagnostics
        ),
        mean_simultaneously_forced_agents=mean(
            "mean_simultaneously_forced_agents"
        ),
        forced_share_of_absolute_agent_flow=mean(
            "forced_share_of_absolute_agent_flow"
        ),
        agent_flow_share_of_total_absolute_flow=mean(
            "agent_flow_share_of_total_absolute_flow"
        ),
        price_max_drawdown=mean("price_max_drawdown"),
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
        fraction_positive_windows_with_forced_flow=mean(
            "fraction_positive_windows_with_forced_flow"
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the documented regime grid and write its Markdown report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--overlaps", nargs="+", type=float, default=DEFAULT_OVERLAPS)
    parser.add_argument(
        "--liquidity-scales",
        nargs="+",
        type=float,
        default=DEFAULT_LIQUIDITY_SCALES,
    )
    parser.add_argument(
        "--drawdown-limits",
        nargs="+",
        type=float,
        default=DEFAULT_DRAWDOWN_LIMITS,
    )
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    args = parser.parse_args(argv)

    result = run_regime_grid(
        args.config,
        overlaps=args.overlaps,
        liquidity_scales=args.liquidity_scales,
        drawdown_limits=args.drawdown_limits,
        n_episodes=args.n_episodes,
        seed_start=args.seed_start,
    )
    config = load_config(args.config)
    report = render_report(
        result,
        config_path=args.config,
        horizon=int(config["targets"]["unwind_horizon"]),
        threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def _grid_axis(values: Sequence[float], name: str, validator: Any) -> tuple[float, ...]:
    result = tuple(validator(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) > MAX_AXIS_VALUES:
        raise ValueError(f"{name} must contain at most {MAX_AXIS_VALUES} values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _overlap_value(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("overlap must be finite and in [0, 1]")
    return result


def _drawdown_value(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("drawdown_limit must be finite and in [0, 1)")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _episode_count(value: int) -> int:
    result = _nonnegative_integer(value, "n_episodes")
    if result == 0 or result > MAX_EPISODES_PER_CELL:
        raise ValueError(
            f"n_episodes must be between 1 and {MAX_EPISODES_PER_CELL}"
        )
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _ordered_unique(values: Sequence[float] | Any) -> tuple[float, ...]:
    return tuple(dict.fromkeys(values))


def _values_text(values: Sequence[float]) -> str:
    return ", ".join(f"{value:g}" for value in values)


def _seed_text(seeds: Sequence[int]) -> str:
    values = tuple(int(seed) for seed in seeds)
    if len(values) > 1 and values == tuple(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ", ".join(str(seed) for seed in values)


if __name__ == "__main__":
    main()


__all__ = [
    "GridEpisode",
    "RegimeGridCell",
    "RegimeGridRun",
    "main",
    "regime_config",
    "render_report",
    "run_regime_grid",
]
