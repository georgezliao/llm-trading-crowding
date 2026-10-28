"""Run the pre-registered learned-agent overlap sweep for round 4.

This diagnostic experiment may consume private simulator ground truth to form
episode outcomes.  It does not construct or evaluate an observer, and its
results are synthetic-simulator evidence only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, fields
import multiprocessing
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from herdlab.evaluation.diagnostics import max_drawdown
from herdlab.evaluation.metrics import ConfidenceInterval
from herdlab.experiments.experiment_zero import _seed_text
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.inference.targets import fit_unwind_threshold, unwind_labels


DEFAULT_CONFIG = Path("configs/experiment_zero.yaml")
DEFAULT_REPORT = Path("docs/reports/overlap_sweep.md")
OVERLAPS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = tuple(range(7000, 7030))
PRIVATE_SIGNAL_STRENGTH = 0.25
LEARNED_OVERLAP_MODE = "lag_weights"
THRESHOLD_SOURCE_OVERLAP = 0.5
DEFAULT_THRESHOLD_QUANTILE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20261028
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class EpisodeOutcome:
    """All pre-registered outcomes for one complete simulated episode."""

    mean_position_crowding: float
    mean_flow_crowding: float
    event_rate: float
    maximum_price_drawdown: float
    forced_order_fraction: float


@dataclass(frozen=True)
class OverlapEpisode:
    """One matched seed's outcomes at one learned-agent overlap level."""

    seed: int
    outcomes: EpisodeOutcome


@dataclass(frozen=True)
class OverlapCondition:
    """All independently seeded episodes at one overlap level."""

    overlap: float
    episodes: tuple[OverlapEpisode, ...]


@dataclass(frozen=True)
class OverlapSweepResult:
    """The fitted diagnostic threshold, episode outcomes, and slope CIs."""

    threshold: float
    conditions: tuple[OverlapCondition, ...]
    slopes: Mapping[str, ConfidenceInterval]
    h1_outcome: str


@dataclass(frozen=True)
class _RawEpisode:
    overlap: float
    seed: int
    prices: np.ndarray
    liquidity: np.ndarray
    agent_net_flow: np.ndarray
    position_crowding: np.ndarray
    flow_crowding: np.ndarray
    forced: np.ndarray


def overlap_config(base: Mapping[str, Any], overlap: float) -> dict[str, Any]:
    """Return one registered condition without mutating the base config."""

    value = float(overlap)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("overlap must be finite and in [0, 1]")
    config = deepcopy(dict(base))
    config["agents"]["feature_overlap"] = value
    config["agents"]["learned_overlap_mode"] = LEARNED_OVERLAP_MODE
    config["agents"]["private_signal"]["strength"] = PRIVATE_SIGNAL_STRENGTH
    return config


def run_overlap_sweep(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    workers: int = 10,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> OverlapSweepResult:
    """Run all 150 registered episodes and analyze their matched outcomes."""

    worker_count = _positive_integer(workers, "workers")
    resample_count = _positive_integer(bootstrap_resamples, "bootstrap_resamples")
    probability = float(threshold_quantile)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("threshold_quantile must be finite and in [0, 1]")

    base = load_config(config_path)
    tasks = tuple(
        (overlap_config(base, overlap), overlap, seed)
        for overlap in OVERLAPS
        for seed in SEEDS
    )
    if worker_count == 1:
        raw = tuple(_simulate_episode(task) for task in tasks)
    else:
        with multiprocessing.Pool(processes=worker_count) as pool:
            raw = tuple(pool.map(_simulate_episode, tasks))

    horizon = int(base["targets"]["unwind_horizon"])
    return analyze_overlap_episodes(
        raw,
        horizon=horizon,
        threshold_quantile=probability,
        bootstrap_resamples=resample_count,
        bootstrap_seed=bootstrap_seed,
    )


def analyze_overlap_episodes(
    raw_episodes: Sequence[_RawEpisode],
    *,
    horizon: int,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> OverlapSweepResult:
    """Fit the registered omega=0.5 threshold and summarize every episode."""

    raw_by_key = {(episode.overlap, episode.seed): episode for episode in raw_episodes}
    expected = {(overlap, seed) for overlap in OVERLAPS for seed in SEEDS}
    if len(raw_episodes) != len(expected) or set(raw_by_key) != expected:
        raise ValueError("raw episodes must contain every registered overlap/seed once")
    probability = float(threshold_quantile)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("threshold_quantile must be finite and in [0, 1]")
    resample_count = _positive_integer(bootstrap_resamples, "bootstrap_resamples")

    threshold_source = tuple(
        raw_by_key[(THRESHOLD_SOURCE_OVERLAP, seed)] for seed in SEEDS
    )
    threshold = fit_unwind_threshold(
        (
            (episode.agent_net_flow, episode.liquidity)
            for episode in threshold_source
        ),
        horizon=horizon,
        quantile=probability,
    )

    conditions: list[OverlapCondition] = []
    for overlap in OVERLAPS:
        episodes: list[OverlapEpisode] = []
        for seed in SEEDS:
            raw = raw_by_key[(overlap, seed)]
            labels = unwind_labels(
                raw.agent_net_flow,
                raw.liquidity,
                horizon=horizon,
                threshold=threshold,
            )
            episodes.append(
                OverlapEpisode(
                    seed=seed,
                    outcomes=EpisodeOutcome(
                        mean_position_crowding=float(
                            np.mean(raw.position_crowding)
                        ),
                        mean_flow_crowding=float(np.mean(raw.flow_crowding)),
                        event_rate=float(np.mean(labels)),
                        maximum_price_drawdown=max_drawdown(raw.prices),
                        forced_order_fraction=float(np.mean(raw.forced)),
                    ),
                )
            )
        conditions.append(OverlapCondition(overlap, tuple(episodes)))

    slopes: dict[str, ConfidenceInterval] = {}
    for outcome_index, field in enumerate(fields(EpisodeOutcome)):
        values = np.asarray(
            [
                [getattr(episode.outcomes, field.name) for episode in condition.episodes]
                for condition in conditions
            ],
            dtype=np.float64,
        )
        slopes[field.name] = episode_bootstrap_slope_ci(
            OVERLAPS,
            values,
            rng=np.random.default_rng(bootstrap_seed + outcome_index),
            n_resamples=resample_count,
        )

    h1_outcome = apply_h1_rule(
        slopes["mean_position_crowding"], slopes["event_rate"]
    )
    return OverlapSweepResult(
        threshold=threshold,
        conditions=tuple(conditions),
        slopes=slopes,
        h1_outcome=h1_outcome,
    )


def episode_bootstrap_slope_ci(
    overlaps: Sequence[float],
    outcomes: np.ndarray,
    *,
    rng: np.random.Generator,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ConfidenceInterval:
    """Bootstrap an OLS outcome-on-overlap slope by matched episode seed.

    ``outcomes`` has shape ``(overlap levels, episode seeds)``.  Each bootstrap
    replicate selects seed columns with replacement and uses the same selected
    columns at every overlap, preserving the registered common-random-number
    pairing.  OLS includes an intercept and treats every selected episode-level
    outcome as one observation.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    x = np.asarray(overlaps, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("overlaps must contain at least two finite values")
    if np.unique(x).size < 2:
        raise ValueError("overlaps must contain at least two distinct values")
    if y.ndim != 2 or y.shape[0] != x.size or y.shape[1] == 0:
        raise ValueError("outcomes must have shape (overlap levels, episodes)")
    if not np.all(np.isfinite(y)):
        raise ValueError("outcomes must be finite")
    count = _positive_integer(n_resamples, "n_resamples")
    level = float(confidence)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must be finite and between zero and one")

    centered = x - np.mean(x)
    denominator = float(np.sum(centered**2))
    estimate = float(np.dot(centered, np.mean(y, axis=1)) / denominator)
    selected = rng.integers(0, y.shape[1], size=(count, y.shape[1]))
    bootstrap_means = np.mean(y[:, selected], axis=2).T
    samples = bootstrap_means @ centered / denominator
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    return ConfidenceInterval(estimate, float(lower), float(upper))


def apply_h1_rule(
    position_slope: ConfidenceInterval,
    event_rate_slope: ConfidenceInterval,
) -> str:
    """Apply the pre-registered H1 rule without discretionary interpretation."""

    _validate_interval(position_slope, "position_slope")
    _validate_interval(event_rate_slope, "event_rate_slope")
    position_positive = position_slope.lower > 0.0
    event_positive = event_rate_slope.lower > 0.0
    if position_positive and event_positive:
        return (
            "H1 rule — H1 is supported in this simulator: the episode-bootstrap "
            "CIs of both the OLS slope of episode mean C^pos on omega and the "
            "event-rate slope lie above 0."
        )
    if position_positive:
        return (
            "H1 rule — the episode-bootstrap CI of the OLS slope of episode "
            "mean C^pos on omega lies above 0, but the event-rate slope CI "
            "includes or lies below 0; overlap raises alignment without raising "
            "unwinds."
        )
    return (
        "H1 rule — H1 is not supported in this simulator: the episode-bootstrap "
        "CI of the OLS slope of episode mean C^pos on omega includes or lies "
        "below 0, so the pre-registered joint condition is not met."
    )


def render_report(
    result: OverlapSweepResult,
    *,
    runtime_seconds: float,
    workers: int,
    config_path: str | Path = DEFAULT_CONFIG,
    horizon: int = 10,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> str:
    """Render the disclaimer, every episode, slope CIs, and rule outcome."""

    if tuple(condition.overlap for condition in result.conditions) != OVERLAPS:
        raise ValueError("result must contain the registered overlap conditions")
    outcome_labels = {
        "mean_position_crowding": "Mean C^pos",
        "mean_flow_crowding": "Mean C^flow",
        "event_rate": "Default-label event rate",
        "maximum_price_drawdown": "Maximum price drawdown",
        "forced_order_fraction": "Forced-order fraction",
    }
    lines = [
        "# Learned-agent overlap sweep (H1)",
        "",
        "**Synthetic-simulator disclaimer:** Every result in this report comes from the uncalibrated synthetic HerdLab simulator. It is not evidence about real markets, real funds, or live trading behavior.",
        "",
        "## Configuration and runtime",
        "",
        f"- Fixed config: `{config_path}`; {len(SEEDS)} episodes per overlap, {len(OVERLAPS) * len(SEEDS)} episodes total, seeds `{_seed_text(SEEDS)}` shared across all overlap levels.",
        f"- Overlap omega: {', '.join(f'{value:g}' for value in OVERLAPS)}; `agents.learned_overlap_mode = {LEARNED_OVERLAP_MODE}`; private-signal strength rho = {PRIVATE_SIGNAL_STRENGTH:g} and beta unchanged at 1.",
        f"- Default-label horizon: {horizon}; the fixed threshold is the {threshold_quantile:.2f} score quantile fitted only from the omega={THRESHOLD_SOURCE_OVERLAP:g} episodes, with fitted value `{result.threshold:.10g}`.",
        "- Episode mean C^pos includes the initial position state; mean C^flow covers all order steps. Event rates use only complete strictly-future windows.",
        "- Forced-order fraction is the fraction of all agent-step decisions marked forced. Under Amendment 1, a forced flag requires a nonzero forced order.",
        f"- OLS includes an intercept. Percentile 95% CIs use {bootstrap_resamples:,} whole-episode bootstrap resamples (seed {bootstrap_seed}); each replicate resamples matched seed identities jointly across omega levels.",
        f"- Measured wall runtime: {runtime_seconds / 60.0:.2f} minutes with {workers} episode workers.",
        "",
        "## Outcomes by overlap",
        "",
        "Each entry is the unweighted mean of the 30 episode-level outcomes at that overlap.",
        "",
        "| omega | Mean C^pos | Mean C^flow | Default-label event rate | Maximum price drawdown | Forced-order fraction |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in result.conditions:
        means = _condition_means(condition)
        lines.append(
            f"| {condition.overlap:g} | {means.mean_position_crowding:.8f} | "
            f"{means.mean_flow_crowding:.8f} | {means.event_rate:.8f} | "
            f"{means.maximum_price_drawdown:.8f} | "
            f"{means.forced_order_fraction:.8f} |"
        )

    lines.extend(
        [
            "",
            "## OLS slopes with episode-bootstrap confidence intervals",
            "",
            "Slopes are in outcome units per one-unit increase in omega. The H1 decision uses only the mean-C^pos and default-label-event-rate rows; the other slopes are descriptive.",
            "",
            "| Episode outcome | OLS slope | 95% CI lower | 95% CI upper |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for field in fields(EpisodeOutcome):
        interval = result.slopes[field.name]
        lines.append(
            f"| {outcome_labels[field.name]} | {interval.estimate:.10f} | "
            f"{interval.lower:.10f} | {interval.upper:.10f} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered H1 rule outcome (verbatim)",
            "",
            result.h1_outcome,
            "",
            "## Episode outcomes",
            "",
            "The following five tables contain the pre-registered outcomes for every independent episode.",
        ]
    )
    for condition in result.conditions:
        lines.extend(
            [
                "",
                f"### omega = {condition.overlap:g}",
                "",
                "| Seed | Mean C^pos | Mean C^flow | Default-label event rate | Maximum price drawdown | Forced-order fraction |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for episode in condition.episodes:
            outcome = episode.outcomes
            lines.append(
                f"| {episode.seed} | {outcome.mean_position_crowding:.8f} | "
                f"{outcome.mean_flow_crowding:.8f} | {outcome.event_rate:.8f} | "
                f"{outcome.maximum_price_drawdown:.8f} | "
                f"{outcome.forced_order_fraction:.8f} |"
            )
    lines.extend(
        [
            "",
            "## Deviations and limits",
            "",
            "- No rho, beta, omega, seed range, threshold rule, episode count, outcome definition, or bootstrap-resample count was changed after viewing metrics.",
            "- These results test an association induced inside this fixed simulator. They do not establish causality or external validity, and they are not evidence of real-market crowding or unwinds.",
            "",
        ]
    )
    return "\n".join(lines)


def _simulate_episode(task: tuple[dict[str, Any], float, int]) -> _RawEpisode:
    config, overlap, seed = task
    public, private = run_episode(config, seed)
    return _RawEpisode(
        overlap=overlap,
        seed=seed,
        prices=np.asarray(public.prices, dtype=np.float64),
        liquidity=np.asarray(public.liquidity_proxy, dtype=np.float64),
        agent_net_flow=np.asarray(private.agent_net_flow, dtype=np.float64),
        position_crowding=np.asarray(private.position_crowding, dtype=np.float64),
        flow_crowding=np.asarray(private.flow_crowding, dtype=np.float64),
        forced=np.asarray(private.forced, dtype=np.bool_),
    )


def _condition_means(condition: OverlapCondition) -> EpisodeOutcome:
    if not condition.episodes:
        raise ValueError("condition episodes must not be empty")
    return EpisodeOutcome(
        **{
            field.name: float(
                np.mean(
                    [getattr(episode.outcomes, field.name) for episode in condition.episodes]
                )
            )
            for field in fields(EpisodeOutcome)
        }
    )


def _validate_interval(interval: ConfidenceInterval, name: str) -> None:
    values = (interval.estimate, interval.lower, interval.upper)
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"{name} must be finite")
    if interval.lower > interval.upper:
        raise ValueError(f"{name} lower bound must not exceed its upper bound")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    result = run_overlap_sweep(args.config, workers=args.workers)
    runtime_seconds = time.perf_counter() - started
    config = load_config(args.config)
    report = render_report(
        result,
        runtime_seconds=runtime_seconds,
        workers=args.workers,
        config_path=args.config,
        horizon=int(config["targets"]["unwind_horizon"]),
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "EpisodeOutcome",
    "OverlapCondition",
    "OverlapEpisode",
    "OverlapSweepResult",
    "analyze_overlap_episodes",
    "apply_h1_rule",
    "episode_bootstrap_slope_ci",
    "main",
    "overlap_config",
    "render_report",
    "run_overlap_sweep",
]
