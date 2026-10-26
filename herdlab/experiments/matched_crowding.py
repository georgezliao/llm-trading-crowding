"""Compare low- and high-overlap populations under matched exogenous shocks.

This experiment changes only ``agents.feature_overlap``.  Family counts,
parameter dispersion, base limits, and all market settings remain at their
baseline values.  Results are synthetic simulator evidence only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from herdlab.agents.base import Agent
from herdlab.evaluation.diagnostics import max_drawdown
from herdlab.experiments.run_episode import load_config, prepare_episode
from herdlab.market.exogenous import ExogenousPath
from herdlab.market.records import PrivateRecord, PublicRecord
from herdlab.market.simulator import simulate_market_records


DEFAULT_CONFIG = Path("configs/baseline.yaml")
DEFAULT_REPORT = Path("docs/reports/matched_crowding.md")
DEFAULT_N_EPISODES = 20
DEFAULT_SEED_START = 0
DEFAULT_LOW_OVERLAP = 0.0
DEFAULT_HIGH_OVERLAP = 1.0
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20261026
DEFAULT_CONFIDENCE = 0.95

@dataclass(frozen=True)
class EpisodeMetrics:
    """One episode's market-instability and private-crowding summaries."""

    drawdown: float
    forced_share: float
    position_crowding: float
    flow_crowding: float
    excess_realized_volatility: float


@dataclass(frozen=True)
class MatchedEpisode:
    """Low/high outcomes for one shared episode seed and exogenous path."""

    seed: int
    low: EpisodeMetrics
    high: EpisodeMetrics


@dataclass(frozen=True)
class PairedDifference:
    """Condition means and a paired episode-bootstrap interval for high-low."""

    low_mean: float
    high_mean: float
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class MatchedCrowdingRun:
    """All paired episodes and metric comparisons required by the report."""

    episodes: tuple[MatchedEpisode, ...]
    comparisons: Mapping[str, PairedDifference]


def overlap_configs(
    config: Mapping[str, Any],
    *,
    low_overlap: float = DEFAULT_LOW_OVERLAP,
    high_overlap: float = DEFAULT_HIGH_OVERLAP,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy a base config into conditions differing only in feature overlap."""

    low_value = _validate_overlap(low_overlap, "low_overlap")
    high_value = _validate_overlap(high_overlap, "high_overlap")
    if low_value >= high_value:
        raise ValueError("low_overlap must be smaller than high_overlap")

    low = deepcopy(dict(config))
    high = deepcopy(dict(config))
    low["agents"]["feature_overlap"] = low_value
    high["agents"]["feature_overlap"] = high_value
    return low, high


def prepare_matched_inputs(
    low_config: Mapping[str, Any],
    high_config: Mapping[str, Any],
    seed: int,
) -> tuple[ExogenousPath, list[Agent], list[Agent]]:
    """Build both populations and return their one verified shared path.

    ``prepare_episode`` uses independent exogenous and population RNG streams.
    Calling it with the same seed makes population construction comparable,
    while the explicit equality check protects the matched-shock invariant.
    The single returned path is then passed to both simulator calls.
    """

    low_path, low_agents = prepare_episode(low_config, seed)
    high_path, high_agents = prepare_episode(high_config, seed)
    _assert_identical_exogenous(low_path, high_path)
    _assert_matched_populations(low_agents, high_agents)
    return low_path, low_agents, high_agents


def run_matched_episode(
    low_config: Mapping[str, Any],
    high_config: Mapping[str, Any],
    seed: int,
) -> MatchedEpisode:
    """Run one low/high pair on exactly the same exogenous arrays."""

    exogenous, low_agents, high_agents = prepare_matched_inputs(
        low_config, high_config, seed
    )
    low_public, low_private = simulate_market_records(
        low_config, low_agents, exogenous
    )
    high_public, high_private = simulate_market_records(
        high_config, high_agents, exogenous
    )
    return MatchedEpisode(
        seed=int(seed),
        low=_episode_metrics(low_public, low_private, exogenous),
        high=_episode_metrics(high_public, high_private, exogenous),
    )


def run_matched_comparison(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    n_episodes: int = DEFAULT_N_EPISODES,
    seed_start: int = DEFAULT_SEED_START,
    low_overlap: float = DEFAULT_LOW_OVERLAP,
    high_overlap: float = DEFAULT_HIGH_OVERLAP,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> MatchedCrowdingRun:
    """Run consecutive matched episodes and bootstrap their paired differences."""

    count = _positive_integer(n_episodes, "n_episodes")
    if isinstance(seed_start, bool) or not isinstance(seed_start, (int, np.integer)):
        raise TypeError("seed_start must be an integer")
    if int(seed_start) < 0:
        raise ValueError("seed_start must be nonnegative")
    resamples = _positive_integer(bootstrap_resamples, "bootstrap_resamples")

    base_config = load_config(config_path)
    low_config, high_config = overlap_configs(
        base_config, low_overlap=low_overlap, high_overlap=high_overlap
    )
    episodes = tuple(
        run_matched_episode(low_config, high_config, seed)
        for seed in range(int(seed_start), int(seed_start) + count)
    )
    rng = np.random.default_rng(bootstrap_seed)
    comparisons = {
        field.name: paired_episode_bootstrap_ci(
            [getattr(episode.low, field.name) for episode in episodes],
            [getattr(episode.high, field.name) for episode in episodes],
            rng=rng,
            n_resamples=resamples,
        )
        for field in fields(EpisodeMetrics)
    }
    return MatchedCrowdingRun(episodes=episodes, comparisons=comparisons)


def paired_episode_bootstrap_ci(
    low_values: Sequence[float],
    high_values: Sequence[float],
    *,
    rng: np.random.Generator,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedDifference:
    """Bootstrap the mean high-minus-low difference by episode pair."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    low = np.asarray(low_values, dtype=np.float64)
    high = np.asarray(high_values, dtype=np.float64)
    if low.ndim != 1 or high.ndim != 1 or low.shape != high.shape or low.size == 0:
        raise ValueError("low_values and high_values must be nonempty equal vectors")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("paired values must be finite")
    count = _positive_integer(n_resamples, "n_resamples")
    level = float(confidence)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must be between zero and one")

    differences = high - low
    selected = rng.integers(0, len(differences), size=(count, len(differences)))
    samples = differences[selected].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return PairedDifference(
        low_mean=float(low.mean()),
        high_mean=float(high.mean()),
        estimate=float(differences.mean()),
        lower=float(lower),
        upper=float(upper),
    )


def render_report(
    result: MatchedCrowdingRun,
    *,
    config_path: str | Path,
    low_overlap: float,
    high_overlap: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> str:
    """Render configuration, paired outcomes, and bootstrap intervals."""

    if not result.episodes:
        raise ValueError("result must contain at least one episode")
    metric_labels = {
        "drawdown": "Maximum drawdown",
        "forced_share": "Forced share of absolute agent flow",
        "position_crowding": "Mean C^pos",
        "flow_crowding": "Mean C^flow",
        "excess_realized_volatility": "Excess realized volatility",
    }
    seeds = [episode.seed for episode in result.episodes]
    lines = [
        "# Matched crowding comparison",
        "",
        "This report describes synthetic output from the uncalibrated HerdLab simulator. It is not evidence about real market behavior. No parameters were selected in response to these results.",
        "",
        "## Design",
        "",
        f"- Base config: `{config_path}`",
        f"- Episode pairs: {len(result.episodes)} consecutive seeds (`{_seed_text(seeds)}`)",
        f"- Low-overlap condition: `agents.feature_overlap = {low_overlap:g}`",
        f"- High-overlap condition: `agents.feature_overlap = {high_overlap:g}`",
        "- The two conditions retain identical family counts, parameter dispersion, base limits, aggregate q_max within each pair, market configuration, and exogenous arrays. Only feature overlap differs.",
        "- In the current simulator, `feature_overlap` changes lag weights for the 12 momentum/reversion agents; the 18 learned agents do not consume this knob.",
        f"- Uncertainty: percentile {DEFAULT_CONFIDENCE:.0%} confidence intervals from {bootstrap_resamples:,} resamples of the {len(result.episodes)} complete episode pairs (bootstrap seed {bootstrap_seed}).",
        "- C^pos is averaged over all recorded position states, including the initial state; C^flow is averaged over all order steps.",
        "- Excess realized volatility is `std(simulated returns) - std(exogenous returns)` within an episode (population standard deviations).",
        "",
        "## Observed outcome",
        "",
        _observed_outcome(result.comparisons),
        "",
        "## Paired results",
        "",
        "Differences are high overlap minus low overlap. Intervals that include zero do not establish a directional difference in this finite simulator experiment.",
        "",
        "| Metric | Low mean | High mean | Paired difference | 95% episode-bootstrap CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in (field.name for field in fields(EpisodeMetrics)):
        comparison = result.comparisons[name]
        lines.append(
            f"| {metric_labels[name]} | {comparison.low_mean:.8f} | "
            f"{comparison.high_mean:.8f} | {comparison.estimate:+.8f} | "
            f"[{comparison.lower:+.8f}, {comparison.upper:+.8f}] |"
        )

    lines.extend(
        [
            "",
            "## Episode pairs",
            "",
            "| Seed | DD low | DD high | Forced low | Forced high | C^pos low | C^pos high | C^flow low | C^flow high | Excess vol low | Excess vol high |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for episode in result.episodes:
        low, high = episode.low, episode.high
        lines.append(
            f"| {episode.seed} | {low.drawdown:.8f} | {high.drawdown:.8f} | "
            f"{low.forced_share:.8f} | {high.forced_share:.8f} | "
            f"{low.position_crowding:.8f} | {high.position_crowding:.8f} | "
            f"{low.flow_crowding:.8f} | {high.flow_crowding:.8f} | "
            f"{low.excess_realized_volatility:.8f} | "
            f"{high.excess_realized_volatility:.8f} |"
        )
    lines.extend(
        [
            "",
            "Forced share, C^pos, and C^flow use private simulator ground truth solely for this controlled diagnostic comparison; they are not observer inputs.",
            "",
        ]
    )
    return "\n".join(lines)


def _observed_outcome(comparisons: Mapping[str, PairedDifference]) -> str:
    forced = comparisons["forced_share"]
    position = comparisons["position_crowding"]
    flow = comparisons["flow_crowding"]
    drawdown = comparisons["drawdown"]
    volatility = comparisons["excess_realized_volatility"]
    return (
        "No forced flow occurred in either condition. High overlap had mean "
        f"C^pos lower by {abs(position.estimate):.8f} and mean C^flow lower by "
        f"{abs(flow.estimate):.8f}; both paired intervals excluded zero. The "
        f"drawdown difference was {drawdown.estimate:+.8f} and the excess-volatility "
        f"difference was {volatility.estimate:+.8f}; both intervals included zero. "
        "These values describe this simulator configuration only and do not show "
        "that greater feature overlap increased crowding or instability."
        if forced.low_mean == 0.0 and forced.high_mean == 0.0
        else (
            "Forced flow occurred in at least one condition; the table below reports "
            "its measured share and all paired differences without a real-market claim."
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the matched comparison and write its Markdown report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--low-overlap", type=float, default=DEFAULT_LOW_OVERLAP)
    parser.add_argument("--high-overlap", type=float, default=DEFAULT_HIGH_OVERLAP)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    result = run_matched_comparison(
        args.config,
        n_episodes=args.n_episodes,
        seed_start=args.seed_start,
        low_overlap=args.low_overlap,
        high_overlap=args.high_overlap,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report = render_report(
        result,
        config_path=args.config,
        low_overlap=args.low_overlap,
        high_overlap=args.high_overlap,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def _episode_metrics(
    public: PublicRecord, private: PrivateRecord, exogenous: ExogenousPath
) -> EpisodeMetrics:
    absolute_flow = np.abs(private.orders)
    forced_absolute_flow = np.where(private.forced, absolute_flow, 0.0)
    total_absolute_flow = float(absolute_flow.sum())
    forced_share = (
        0.0
        if total_absolute_flow == 0.0
        else float(forced_absolute_flow.sum()) / total_absolute_flow
    )
    return EpisodeMetrics(
        drawdown=max_drawdown(public.prices),
        forced_share=forced_share,
        position_crowding=float(np.mean(private.position_crowding)),
        flow_crowding=float(np.mean(private.flow_crowding)),
        excess_realized_volatility=(
            float(np.std(public.returns)) - float(np.std(exogenous.returns))
        ),
    )


def _assert_identical_exogenous(
    low_path: ExogenousPath, high_path: ExogenousPath
) -> None:
    for field in fields(ExogenousPath):
        if not np.array_equal(
            getattr(low_path, field.name), getattr(high_path, field.name)
        ):
            raise ValueError(
                f"matched populations produced different exogenous {field.name}"
            )


def _assert_matched_populations(
    low_agents: Sequence[Agent], high_agents: Sequence[Agent]
) -> None:
    if len(low_agents) != len(high_agents):
        raise ValueError("matched populations must contain the same agent count")
    if any(type(low) is not type(high) for low, high in zip(low_agents, high_agents)):
        raise ValueError("matched populations must have identical family ordering")
    low_capacity = sum(agent.q_max for agent in low_agents)
    high_capacity = sum(agent.q_max for agent in high_agents)
    if low_capacity != high_capacity:
        raise ValueError("matched populations must have identical aggregate q_max")


def _validate_overlap(value: float, name: str) -> float:
    overlap = float(value)
    if not np.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return overlap


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _seed_text(seeds: Sequence[int]) -> str:
    values = tuple(int(seed) for seed in seeds)
    if len(values) > 1 and values == tuple(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ", ".join(str(seed) for seed in values)


if __name__ == "__main__":
    main()


__all__ = [
    "EpisodeMetrics",
    "MatchedCrowdingRun",
    "MatchedEpisode",
    "PairedDifference",
    "main",
    "overlap_configs",
    "paired_episode_bootstrap_ci",
    "prepare_matched_inputs",
    "render_report",
    "run_matched_comparison",
    "run_matched_episode",
]
