"""Round-5 item 1: cross historical populations and seeds without signals."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import yaml

from herdlab.evaluation.metrics import ConfidenceInterval, episode_bootstrap_metrics
from herdlab.experiments.experiment_zero import load_v2_configs, _seed_text
from herdlab.experiments.experiment_zero_round3 import (
    EventSummary, V2_TRAINING_SEEDS, V2_IN_DISTRIBUTION_SEEDS,
    _simulate, _task_episodes, _event_summary, _fit_gru_seeds, _format_ci,
)
from herdlab.experiments.experiment_zero_round4 import (
    TRAINING_SEEDS, TEST_SEEDS, _fit_threshold,
)
from herdlab.experiments.run_episode import load_config
from herdlab.inference.baselines import LightGBMObserver

SEED_SETS = {
    "round3": (V2_TRAINING_SEEDS, V2_IN_DISTRIBUTION_SEEDS),
    "round4": (TRAINING_SEEDS, TEST_SEEDS),
}
# Frozen published default/in-distribution point estimates, not CI endpoints.
REFERENCES = {("round3_v2", "round3"): 0.7138,
              ("experiment_zero", "round4"): 0.9510}
RUNTIME_ESTIMATE_MINUTES = 10.0
DEFAULT_REPORT = Path("docs/reports/round5_reconcile.md")


@dataclass(frozen=True)
class CellResult:
    population: str
    seed_set: str
    config: Mapping[str, Any]
    training_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    threshold: float
    training_events: EventSummary
    test_events: EventSummary
    metrics: Mapping[str, ConfidenceInterval]
    gru_detail: str


def population_configs() -> dict[str, dict[str, Any]]:
    """Use the historical composition helper, including all inherited settings."""
    configs = {
        "round3_v2": load_v2_configs()["training"],
        "experiment_zero": load_config("configs/experiment_zero.yaml"),
    }
    result = deepcopy(configs)
    for config in result.values():
        config["agents"]["private_signal"]["mode"] = "off"
        config["agents"]["private_signal"]["strength"] = 0.0
        config["agents"]["learned_overlap_mode"] = "none"
    return result


def reproduction_status(value: float, reference: float) -> str:
    """The registration requires agreement after rounding to four decimals."""
    if np.isfinite(value) and f"{value:.4f}" == f"{reference:.4f}":
        return "PASS"
    return "REPRODUCIBILITY FAILURE"


def run_reconciliation(*, workers: int = 10) -> tuple[CellResult, ...]:
    cells = []
    for population, config in population_configs().items():
        for seed_set, (training_seeds, test_seeds) in SEED_SETS.items():
            print(f"Starting {population}/{seed_set}: 200 train + 50 test episodes", flush=True)
            training = _simulate(config, training_seeds, workers=workers)
            test = _simulate(config, test_seeds, workers=workers)
            threshold = _fit_threshold(training, "default", 10, 0.95)
            train_tasks = _task_episodes(training, label_name="default", horizon=10,
                                        threshold=threshold)
            test_tasks = _task_episodes(test, label_name="default", horizon=10,
                                       threshold=threshold)
            # Neither observer receives EpisodeSample or private diagnostics.
            lightgbm = LightGBMObserver(return_lookback=10,
                                       rng=np.random.default_rng(5000)).fit(train_tasks)
            predictions = {"lightgbm": lightgbm.predict_proba(test_tasks)}
            _, detail, gru_predictions = _fit_gru_seeds(
                train_tasks, {"test": test_tasks}, seeds=(6000,), workers=1,
                return_lookback=10, gru_kwargs=None,
            )[0]
            predictions["gru_6000"] = gru_predictions["test"]
            labels = tuple(np.asarray(task.labels, dtype=bool) for task in test_tasks)
            metrics = {
                name: episode_bootstrap_metrics(
                    labels, values, rng=np.random.default_rng(9000 + len(cells) * 100 + i),
                    n_resamples=1000,
                )["pr_auc"]
                for i, (name, values) in enumerate(predictions.items())
            }
            cells.append(CellResult(population, seed_set, deepcopy(config),
                                    tuple(training_seeds), tuple(test_seeds), threshold,
                                    _event_summary(train_tasks), _event_summary(test_tasks),
                                    metrics, detail))
            print(f"Finished {population}/{seed_set}: LightGBM {_format_ci(metrics['lightgbm'])}",
                  flush=True)
    return tuple(cells)


def render_report(cells: tuple[CellResult, ...], *, runtime_seconds: float) -> str:
    lines = [
        "# Round 5: population and seed reconciliation", "",
        "**Synthetic-simulator disclaimer:** These are results from the uncalibrated synthetic HerdLab simulator, not evidence about real markets, funds, or live trading.", "",
        "## Design and runtime", "",
        f"Pre-run estimate for R5-2: {RUNTIME_ESTIMATE_MINUTES:g} minutes, below the registered ~75-minute stop threshold. Measured runtime: {runtime_seconds / 60:.2f} minutes. Later round-5 experiments are not included in this estimate or run.", "",
        "Four independently fitted cells cross two populations and two seed sets. Each uses 200 training and 50 test episodes of 500 steps. Default label only: horizon 10, training-only 0.95 score quantile fitted separately per cell. Signals are explicitly off; learned overlap mode is none.", "",
        "Observers retain historical defaults: public return lookback 10, LightGBM RNG seed 5000, GRU training seed 6000 (single model, not ensemble). PR-AUC uses 1,000 whole-test-episode percentile bootstrap resamples, 95% CIs; bootstrap seeds are 9000 + 100 × cell index + observer index in the order below. Base rates are descriptive. No overlapping time steps are independently resampled.", "",
        "## 2×2 results", "",
        "| Population | Round-3 seeds | Round-4 seeds |",
        "| --- | --- | --- |",
    ]
    for population in ("round3_v2", "experiment_zero"):
        entries = []
        for seed_set in SEED_SETS:
            cell = next(c for c in cells if (c.population, c.seed_set) == (population, seed_set))
            entries.append(f"Train rate {cell.training_events.base_rate:.6f}; test rate {cell.test_events.base_rate:.6f}; LightGBM {_format_ci(cell.metrics['lightgbm'])}; GRU 6000 {_format_ci(cell.metrics['gru_6000'])}")
        lines.append(f"| {population} | " + " | ".join(entries) + " |")
    lines += ["", "## Counts, thresholds, seeds, and training details", "",
              "| Population / seeds | Train positives / labels | Test positives / labels | Threshold |",
              "| --- | ---: | ---: | ---: |"]
    for c in cells:
        lines.append(f"| {c.population} / {c.seed_set} | {c.training_events.event_count} / {c.training_events.label_count} | {c.test_events.event_count} / {c.test_events.label_count} | {c.threshold:.12g} |")
    for c in cells:
        lines.append(f"\n- {c.population}/{c.seed_set}: train `{_seed_text(c.training_seeds)}`; test `{_seed_text(c.test_seeds)}`; GRU {c.gru_detail}.")
    lines += ["", "## Historical LightGBM reproducibility checks", "",
              "Agreement is required to four decimal places, regardless of CI overlap. References are the default-label in-distribution row in `experiment_zero_round3.md` and the rho=0/default row in `experiment_zero_round4.md`.", "",
              "| Matching cell | Published PR-AUC | Rerun PR-AUC | Outcome |",
              "| --- | ---: | ---: | --- |"]
    for c in cells:
        reference = REFERENCES.get((c.population, c.seed_set))
        if reference is not None:
            value = c.metrics["lightgbm"].estimate
            lines.append(f"| {c.population}/{c.seed_set} | {reference:.4f} | {value:.4f} | {reproduction_status(value, reference)} |")
    lines += ["", "## Exact resolved configurations", "",
              "Round-3 v2 is composed by `load_v2_configs()` from the selected experiment config and the training population override in `heldout_families.yaml`; the held-out impact override does not apply to this in-distribution cohort. Counts are momentum/reversion/linear/GBDT/MLP = 10/0/10/0/10. The experiment-zero population is 6/6/6/6/6. Both have 30 agents and aggregate configured q_max 6000. Full resolved configs below include every inherited population, risk, and market setting, with the explicit no-signal override."]
    for population in ("round3_v2", "experiment_zero"):
        subset = [c for c in cells if c.population == population]
        lines += ["", f"### {population}: applies to both seed-set cells", "",
                  "```yaml", yaml.safe_dump(dict(subset[0].config), sort_keys=False).rstrip(), "```"]
    lines += ["", "## Scope, deviations, and open questions", "",
              "Only round-5 item 1 was run. No simulator equation, label, threshold rule, observer hyperparameter, or registered seed range was changed. Private agent flows are reduced to labels before observer construction; observers receive only public records and targets. The historical check concerns LightGBM point estimates, not bootstrap endpoints or GRU ensemble scores. This 2×2 table separates population and seed changes descriptively; it does not establish real-market generalization or a causal crowding mechanism.", "",
              "No research-design deviations or open questions. Any failed check above is a reproducibility failure and requires investigation; it is not waived by uncertainty intervals.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    print(f"Pre-run R5-2 runtime estimate: {RUNTIME_ESTIMATE_MINUTES:g} minutes "
          "(four cells; historical round-3 full run took 20.80 minutes).", flush=True)
    if RUNTIME_ESTIMATE_MINUTES > 75:
        raise SystemExit("Runtime stop: estimate exceeds registered 75-minute limit")
    started = time.perf_counter()
    cells = run_reconciliation(workers=args.workers)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(cells, runtime_seconds=time.perf_counter() - started),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
