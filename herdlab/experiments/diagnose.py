"""Generate baseline episodes and report forced-deleveraging diagnostics.

The report consumes private simulator state only inside this diagnostic
experiment.  It is descriptive simulator evidence, not observer input and not
evidence about real market behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from herdlab.evaluation.diagnostics import (
    CascadeDiagnostics,
    episode_cascade_diagnostics,
)
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.inference.targets import fit_unwind_threshold, unwind_labels
from herdlab.market.records import PrivateRecord, PublicRecord


DEFAULT_CONFIG = Path("configs/baseline.yaml")
DEFAULT_REPORT = Path("docs/reports/cascade_diagnostics.md")
DEFAULT_N_EPISODES = 20
DEFAULT_SEED_START = 0
DEFAULT_THRESHOLD_QUANTILE = 0.95
REPORT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class EpisodeDiagnostic:
    """One seeded episode and its diagnostic summary."""

    seed: int
    n_steps: int
    forced_step_count: int
    has_nonzero_forced_flow: bool
    diagnostics: CascadeDiagnostics


@dataclass(frozen=True)
class DiagnosticRun:
    """All values needed to render the baseline diagnostic report."""

    threshold: float
    episodes: tuple[EpisodeDiagnostic, ...]
    classification: str

    @property
    def forced_episode_count(self) -> int:
        return sum(episode.has_nonzero_forced_flow for episode in self.episodes)

    @property
    def forced_step_count(self) -> int:
        return sum(episode.forced_step_count for episode in self.episodes)

    @property
    def total_step_count(self) -> int:
        return sum(episode.n_steps for episode in self.episodes)


def run_diagnostics(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    n_episodes: int = DEFAULT_N_EPISODES,
    seed_start: int = DEFAULT_SEED_START,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
) -> DiagnosticRun:
    """Simulate consecutive baseline seeds and compute private diagnostics."""

    if isinstance(n_episodes, bool) or n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if isinstance(seed_start, bool) or seed_start < 0:
        raise ValueError("seed_start must be nonnegative")

    config = load_config(config_path)
    horizon = int(config["targets"]["unwind_horizon"])
    records: list[tuple[int, PublicRecord, PrivateRecord]] = []
    for seed in range(seed_start, seed_start + n_episodes):
        public, private = run_episode(config, seed)
        records.append((seed, public, private))

    threshold = fit_unwind_threshold(
        (
            (private.agent_net_flow, public.liquidity_proxy)
            for _, public, private in records
        ),
        horizon=horizon,
        quantile=threshold_quantile,
    )

    episodes: list[EpisodeDiagnostic] = []
    for seed, public, private in records:
        labels = unwind_labels(
            private.agent_net_flow,
            public.liquidity_proxy,
            horizon=horizon,
            threshold=threshold,
        )
        forced_flow = np.where(private.forced, np.abs(private.orders), 0.0)
        forced_steps = np.any(private.forced, axis=1)
        episodes.append(
            EpisodeDiagnostic(
                seed=seed,
                n_steps=len(private.orders),
                forced_step_count=int(np.count_nonzero(forced_steps)),
                has_nonzero_forced_flow=bool(np.any(forced_flow > 0.0)),
                diagnostics=episode_cascade_diagnostics(
                    public,
                    private,
                    labels,
                    horizon=horizon,
                    quantiles=REPORT_QUANTILES,
                ),
            )
        )

    episode_values = tuple(episodes)
    return DiagnosticRun(
        threshold=threshold,
        episodes=episode_values,
        classification=classify_forced_deleveraging(episode_values),
    )


def classify_forced_deleveraging(
    episodes: Sequence[EpisodeDiagnostic],
) -> str:
    """Classify forced flow by its incidence across independent episodes."""

    if not episodes:
        raise ValueError("episodes must not be empty")
    incidence = np.mean(
        [episode.has_nonzero_forced_flow for episode in episodes], dtype=np.float64
    )
    if incidence == 0.0:
        return "absent"
    if incidence < 0.5:
        return "rare"
    return "common"


def render_report(
    result: DiagnosticRun,
    *,
    config_path: str | Path,
    horizon: int,
    threshold_quantile: float,
) -> str:
    """Render the actual per-episode diagnostic values as Markdown."""

    if not result.episodes:
        raise ValueError("result must contain at least one episode")
    seeds = tuple(episode.seed for episode in result.episodes)
    mean = _mean_diagnostics(result.episodes)
    classification_statement = _classification_statement(result)

    lines = [
        "# Baseline cascade diagnostics",
        "",
        "This report describes synthetic output from the uncalibrated HerdLab simulator. It is not evidence about real market behavior. No simulator parameter, risk rule, threshold rule, or label definition was changed for this run.",
        "",
        "## Configuration",
        "",
        f"- Baseline config: `{config_path}`",
        f"- Episodes: {len(result.episodes)} consecutive seeds (`{_seed_text(seeds)}`), {result.episodes[0].n_steps} steps each",
        f"- Unwind horizon: {horizon} steps",
        f"- Unwind threshold: pooled diagnostic-score quantile {threshold_quantile:.2f}, fitted value `{result.threshold:.10g}`. Only these diagnostic episodes were used; this report makes no held-out prediction claim.",
        "- Crowding columns report the 0th, 25th, 50th, 75th, and 100th percentiles within each episode.",
        "",
        "## Finding",
        "",
        classification_statement,
        "",
        f"Across all episodes, {result.forced_step_count} / {result.total_step_count} steps had an order marked forced, and {result.forced_episode_count} / {len(result.episodes)} episodes contained nonzero forced flow.",
        "Classification rule fixed for this report: **absent** means no episode contains nonzero forced flow; otherwise **rare** means fewer than half of episodes contain it, and **common** means at least half contain it.",
        "",
        "Mean episode-level values: "
        f"forced-step fraction `{mean['forced_step_fraction']:.6f}`, "
        f"forced share of absolute agent flow `{mean['forced_flow_share']:.6f}`, "
        f"agent-flow share of total absolute flow `{mean['agent_total_flow_share']:.6f}`, "
        f"maximum drawdown `{mean['max_drawdown']:.6f}`, and "
        f"positive-window forced-flow fraction `{mean['positive_window_forced']:.6f}`.",
        "",
        "## Forced-flow and market diagnostics",
        "",
        "| Seed | Forced steps | Forced-step fraction | Max agents forced | Mean agents forced | Forced / agent abs. flow | Agent / total abs. flow | Max drawdown | Y=1 windows with forced flow |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode in result.episodes:
        diagnostic = episode.diagnostics
        lines.append(
            f"| {episode.seed} | {episode.forced_step_count} | "
            f"{diagnostic.fraction_steps_with_forced_order:.6f} | "
            f"{diagnostic.max_simultaneously_forced_agents} | "
            f"{diagnostic.mean_simultaneously_forced_agents:.6f} | "
            f"{diagnostic.forced_share_of_absolute_agent_flow:.6f} | "
            f"{diagnostic.agent_flow_share_of_total_absolute_flow:.6f} | "
            f"{diagnostic.price_max_drawdown:.6f} | "
            f"{diagnostic.fraction_positive_windows_with_forced_flow:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Crowding distributions",
            "",
            "| Seed | C^pos min | C^pos p25 | C^pos p50 | C^pos p75 | C^pos max | C^flow min | C^flow p25 | C^flow p50 | C^flow p75 | C^flow max |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for episode in result.episodes:
        position = episode.diagnostics.position_crowding_quantiles
        flow = episode.diagnostics.flow_crowding_quantiles
        position_values = " | ".join(
            f"{position[quantile]:.6f}" for quantile in REPORT_QUANTILES
        )
        flow_values = " | ".join(
            f"{flow[quantile]:.6f}" for quantile in REPORT_QUANTILES
        )
        lines.append(
            f"| {episode.seed} | {position_values} | {flow_values} |"
        )

    lines.extend(
        [
            "",
            "The Y-window column is diagnostic only: Y uses strictly future agent net flow and the threshold above, while forced-flow membership is read from private ground truth. Private state is not supplied to an observer model.",
            "",
        ]
    )
    return "\n".join(lines)


def _classification_statement(result: DiagnosticRun) -> str:
    if result.classification == "absent":
        detail = "no episode contained nonzero forced flow"
    elif result.classification == "rare":
        detail = "fewer than half of episodes contained nonzero forced flow"
    else:
        detail = "at least half of episodes contained nonzero forced flow"
    return f"**Forced deleveraging is {result.classification}**: {detail}."


def _mean_diagnostics(
    episodes: Sequence[EpisodeDiagnostic],
) -> dict[str, float]:
    diagnostics = [episode.diagnostics for episode in episodes]
    return {
        "forced_step_fraction": float(
            np.mean([item.fraction_steps_with_forced_order for item in diagnostics])
        ),
        "forced_flow_share": float(
            np.mean([item.forced_share_of_absolute_agent_flow for item in diagnostics])
        ),
        "agent_total_flow_share": float(
            np.mean([item.agent_flow_share_of_total_absolute_flow for item in diagnostics])
        ),
        "max_drawdown": float(
            np.mean([item.price_max_drawdown for item in diagnostics])
        ),
        "positive_window_forced": float(
            np.mean(
                [
                    item.fraction_positive_windows_with_forced_flow
                    for item in diagnostics
                ]
            )
        ),
    }


def _seed_text(seeds: Sequence[int]) -> str:
    values = tuple(int(seed) for seed in seeds)
    if len(values) > 1 and values == tuple(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ", ".join(str(seed) for seed in values)


def main(argv: Sequence[str] | None = None) -> None:
    """Run baseline diagnostics and write the Markdown report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    args = parser.parse_args(argv)

    result = run_diagnostics(
        args.config,
        n_episodes=args.n_episodes,
        seed_start=args.seed_start,
    )
    config = load_config(args.config)
    horizon = int(config["targets"]["unwind_horizon"])
    report = render_report(
        result,
        config_path=args.config,
        horizon=horizon,
        threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "DiagnosticRun",
    "EpisodeDiagnostic",
    "classify_forced_deleveraging",
    "main",
    "render_report",
    "run_diagnostics",
]
