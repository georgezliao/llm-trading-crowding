"""Round-5 overlap stress diagnostic; leaves the round-4 runner unchanged."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import multiprocessing
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from herdlab.evaluation.metrics import ConfidenceInterval
from herdlab.experiments import overlap_sweep as legacy
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.inference.targets import fit_unwind_threshold, forced_selling_flow, unwind_labels

DEFAULT_REPORT = Path("docs/reports/overlap_sweep_stress.md")


@dataclass(frozen=True)
class StressEpisode:
    diagnostic: legacy._RawEpisode
    forced_flow: np.ndarray


@dataclass(frozen=True)
class StressResult:
    overlap: legacy.OverlapSweepResult
    forced_threshold: float
    forced_counts: Mapping[tuple[float, int], int]
    h1_outcome: str


def stress_config(base, overlap):
    config = legacy.overlap_config(base, overlap)
    config["agents"]["private_signal"].update(mode="signal", beta=0.25)
    config["agents"]["risk"]["max_drawdown"] = 0.05
    return config


def apply_forced_h1_rule(slope: ConfidenceInterval, event_counts: Sequence[int]) -> str:
    """Apply round-5 item 6, including strict CI and inclusive count boundaries."""
    legacy._validate_interval(slope, "forced_order_fraction_slope")
    counts = tuple(event_counts)
    if len(counts) != len(legacy.OVERLAPS):
        raise ValueError("one event count is required for each of the five omega levels")
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0 for n in counts):
        raise ValueError("event counts must be nonnegative integers")
    if slope.lower > 0 and sum(n >= 30 for n in counts) >= 3:
        return "H1 for forced selling is supported in this simulator."
    return "forced selling was not tested adequately."


def simulate_stress_episode(task):
    config, overlap, seed = task
    public, private = run_episode(config, seed)
    return StressEpisode(
        legacy._RawEpisode(
            overlap, seed, public.prices, public.liquidity_proxy,
            private.agent_net_flow, private.position_crowding,
            private.flow_crowding, private.forced,
        ),
        forced_selling_flow(private.orders, private.forced),
    )


def analyze_stress_episodes(raw, *, horizon, bootstrap_resamples=legacy.DEFAULT_BOOTSTRAP_RESAMPLES):
    # The legacy analysis validates complete, unique matched overlap/seed pairs.
    overlap = legacy.analyze_overlap_episodes(
        [episode.diagnostic for episode in raw], horizon=horizon,
        bootstrap_resamples=bootstrap_resamples,
    )
    threshold = fit_unwind_threshold(
        ((episode.forced_flow, episode.diagnostic.liquidity) for episode in raw
         if episode.diagnostic.overlap == legacy.THRESHOLD_SOURCE_OVERLAP),
        horizon=horizon, quantile=legacy.DEFAULT_THRESHOLD_QUANTILE,
    )
    counts = {
        (episode.diagnostic.overlap, episode.diagnostic.seed): int(np.count_nonzero(
            unwind_labels(episode.forced_flow, episode.diagnostic.liquidity,
                          horizon=horizon, threshold=threshold)
        )) for episode in raw
    }
    totals = [sum(counts[(omega, seed)] for seed in legacy.SEEDS) for omega in legacy.OVERLAPS]
    return StressResult(overlap, threshold, counts, apply_forced_h1_rule(
        overlap.slopes["forced_order_fraction"], totals))


def run_stress_sweep(config_path=legacy.DEFAULT_CONFIG, *, workers=10):
    workers = legacy._positive_integer(workers, "workers")
    base = load_config(config_path)
    tasks = [(stress_config(base, omega), omega, seed)
             for omega in legacy.OVERLAPS for seed in legacy.SEEDS]
    if workers == 1:
        raw = list(map(simulate_stress_episode, tasks))
    else:
        with multiprocessing.Pool(workers) as pool:
            raw = pool.map(simulate_stress_episode, tasks)
    return analyze_stress_episodes(raw, horizon=int(base["targets"]["unwind_horizon"]))


def render_report(result, *, runtime_seconds, workers, config_path=legacy.DEFAULT_CONFIG, horizon=10):
    lines = [
        "# Round-5 overlap under stress", "",
        "**Synthetic-simulator disclaimer:** Every result comes from the uncalibrated synthetic HerdLab simulator, not real markets, funds, or live trading behavior.", "",
        "## Configuration and runtime", "",
        f"Config: `{config_path}` with rho=0.25, beta=0.25, signal mode, max_drawdown=0.05, and learned_overlap_mode=lag_weights. All other settings are unchanged. Omega levels: 0, 0.25, 0.5, 0.75, 1. Seeds 7000–7029 at every level: 30 episodes per level, 150 total.", "",
        f"Measured wall runtime: {runtime_seconds / 60:.2f} minutes; {workers} episode workers. Command: `.venv/bin/python -m herdlab.experiments.overlap_sweep_stress --workers {workers}`.", "",
        f"Horizon {horizon}; separate 0.95 score quantiles for default and forced-flow labels, fitted only on the omega=0.5 calibration episodes and held fixed across omega. Default threshold: {result.overlap.threshold:.12g}; forced-flow threshold: {result.forced_threshold:.12g}.", "",
        "Calibration follows the round-4 sweep's own-episode convention: omega=0.5 is also summarized, so its event rates are in-sample diagnostics, not held-out predictions. Forced-flow calibration extends that convention to the new label. Counts are positive complete strictly-future windows (including overlaps), not distinct liquidation episodes. Forced-flow scores include only negative forced orders; forced-order fraction includes all nonzero forced buys and sells, as registered.", "",
        f"OLS includes an intercept; 95% percentile CIs use {legacy.DEFAULT_BOOTSTRAP_RESAMPLES:,} whole-episode resamples, jointly resampling seed identities across omega. RNG seeds are {legacy.DEFAULT_BOOTSTRAP_SEED} + outcome index in table order. Mean C^pos includes the initial state; mean C^flow covers order steps.", "",
        "## Mean outcomes and event totals", "",
        "| omega | Mean C^pos | Mean C^flow | Default event rate | Max drawdown | Forced-order fraction | Forced-flow events |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in result.overlap.conditions:
        values = legacy._condition_means(condition)
        total = sum(result.forced_counts[(condition.overlap, seed)] for seed in legacy.SEEDS)
        lines.append(f"| {condition.overlap:g} | " + " | ".join(
            f"{getattr(values, field.name):.8f}" for field in fields(legacy.EpisodeOutcome)
        ) + f" | {total} |")
    lines += ["", "## OLS slopes with 95% episode-bootstrap CIs", "",
              "| Outcome | Slope | Lower | Upper |", "| --- | ---: | ---: | ---: |"]
    for name, ci in result.overlap.slopes.items():
        lines.append(f"| {name} | {ci.estimate:.10f} | {ci.lower:.10f} | {ci.upper:.10f} |")
    lines += ["", "## Forced-selling H1 rule outcome (verbatim)", "",
              result.h1_outcome, "",
              "Support requires the forced-order-fraction slope CI to lie strictly above zero and at least three omega levels to have at least 30 forced-flow events each. Other slopes are descriptive for this round.", ""]
    for condition in result.overlap.conditions:
        lines += [f"## omega = {condition.overlap:g}", "",
                  "| Seed | Mean C^pos | Mean C^flow | Default event rate | Max drawdown | Forced-order fraction | Forced-flow events |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for episode in condition.episodes:
            lines.append(f"| {episode.seed} | " + " | ".join(
                f"{getattr(episode.outcomes, field.name):.8f}" for field in fields(legacy.EpisodeOutcome)
            ) + f" | {result.forced_counts[(condition.overlap, episode.seed)]} |")
        lines.append("")
    lines += ["## Deviations, open questions, and limits", "",
              "No simulator equation, existing label definition, seed, episode count, or parameter was changed after viewing metrics. Round-4 code and report remain unchanged. No observer receives private diagnostics.", "",
              "Open question for the research lead: item 5 does not separately specify the added forced-flow calibration source; this implementation inherits the round-4 omega=0.5 source and quantile, using a separate forced-flow threshold. Confirm this interpretation before drawing a research conclusion.", "",
              "R5-3 reports 97.57 minutes of retained condition compute, plus duplicated/interrupted attempts whose budget accounting remains an open question. The runtime above is additional R5-4 wall time. These diagnostic associations do not establish real-market validity or causality.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(legacy.DEFAULT_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    result = run_stress_sweep(args.config, workers=args.workers)
    report = render_report(result, runtime_seconds=time.perf_counter() - start,
                           workers=args.workers, config_path=args.config,
                           horizon=int(load_config(args.config)["targets"]["unwind_horizon"]))
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(result.h1_outcome, flush=True)


if __name__ == "__main__":
    main()
