"""Run the fully pre-registered third round of Experiment Zero.

Private simulator state is reduced to label inputs before observers are built.
Every observer receives only public records paired with already-computed labels.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from herdlab.evaluation.metrics import (
    ConfidenceInterval,
    episode_bootstrap_metrics,
    paired_pr_auc_difference_ci,
)
from herdlab.experiments.experiment_zero import (
    _deep_merge,
    _population_override,
    _seed_text,
    _selected_values,
    load_v2_configs,
)
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.inference.baselines import (
    LightGBMObserver,
    LogisticObserver,
    StrategyReplicationObserver,
    TaskCEpisode,
    TrivialReferenceObserver,
)
from herdlab.inference.targets import fit_unwind_threshold, forced_selling_flow, unwind_labels
from herdlab.inference.temporal import GRUObserver
from herdlab.market.records import PublicRecord


DEFAULT_SELECTED_CONFIG = Path("configs/experiment_zero.yaml")
DEFAULT_FAMILY_CONFIG = Path("configs/heldout_families.yaml")
DEFAULT_IMPACT_CONFIG = Path("configs/heldout_impact.yaml")
DEFAULT_ARCHITECTURE_CONFIG = Path("configs/heldout_architecture.yaml")
DEFAULT_REPORT = Path("docs/reports/experiment_zero_round3.md")

V2_TRAINING_SEEDS = tuple(range(1000, 1200))
V2_IN_DISTRIBUTION_SEEDS = tuple(range(2000, 2050))
V2_HELDOUT_FAMILY_SEEDS = tuple(range(3000, 3050))
V2_HELDOUT_IMPACT_SEEDS = tuple(range(4000, 4050))
ARCHITECTURE_TRAINING_SEEDS = tuple(range(5000, 5200))
ARCHITECTURE_IN_DISTRIBUTION_SEEDS = tuple(range(5500, 5550))
HELDOUT_ARCHITECTURE_SEEDS = tuple(range(6500, 6550))
GRU_SEEDS = tuple(range(6000, 6005))
LABEL_NAMES = ("default", "onset", "forced_flow")


@dataclass(frozen=True)
class EpisodeSample:
    """Public episode plus only the private vectors required to make targets."""

    public: PublicRecord
    agent_net_flow: np.ndarray
    forced_selling_flow: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        flow = np.asarray(self.agent_net_flow, dtype=np.float64)
        forced = np.asarray(self.forced_selling_flow, dtype=np.float64)
        if flow.ndim != 1 or forced.shape != flow.shape:
            raise ValueError("target flow vectors must be equal-length vectors")
        if not np.all(np.isfinite(flow)) or not np.all(np.isfinite(forced)):
            raise ValueError("target flow vectors must be finite")
        if np.any(forced > 0.0):
            raise ValueError("forced_selling_flow must contain sells or zero only")


@dataclass(frozen=True)
class EventSummary:
    event_count: int
    label_count: int

    @property
    def base_rate(self) -> float:
        return self.event_count / self.label_count


@dataclass(frozen=True)
class EvaluationResult:
    events: EventSummary
    metrics: Mapping[str, Mapping[str, ConfidenceInterval]]
    paired_differences: Mapping[str, ConfidenceInterval]
    gru_pr_auc_range: tuple[float, float]


@dataclass(frozen=True)
class LabelResult:
    threshold: float
    training_events: EventSummary
    evaluation: Mapping[str, EvaluationResult]
    gru_training_details: Mapping[int, str]


@dataclass(frozen=True)
class CohortResult:
    labels: Mapping[str, LabelResult]


@dataclass(frozen=True)
class Round3Result:
    cohorts: Mapping[str, CohortResult]


@dataclass(frozen=True)
class DecisionInputs:
    """Minimal measured inputs consumed by the fixed decision rules."""

    cohort: str
    split: str
    label: str
    event_count: int
    base_rate: float
    gru_minus_replicator: ConfidenceInterval
    gru_minus_trivial: ConfidenceInterval | None = None


def load_architecture_configs(
    selected_config_path: str | Path = DEFAULT_SELECTED_CONFIG,
    architecture_config_path: str | Path = DEFAULT_ARCHITECTURE_CONFIG,
) -> dict[str, dict[str, Any]]:
    """Compose the selected setting with both architecture populations."""

    selected = deepcopy(dict(load_config(selected_config_path)))
    specification = load_config(architecture_config_path)
    train = _deep_merge(selected, _population_override(specification, "train"))
    test = _deep_merge(selected, _population_override(specification, "test"))
    if _selected_values(train) != _selected_values(selected):
        raise ValueError("selected impact/leverage values changed in architecture train")
    if _selected_values(test) != _selected_values(selected):
        raise ValueError("selected impact/leverage values changed in architecture test")
    return {
        "training": train,
        "in_distribution": deepcopy(train),
        "heldout_architecture": test,
    }


def apply_decision_rules(inputs: Sequence[DecisionInputs]) -> tuple[str, ...]:
    """Apply the pre-registered round-3 rules without discretionary tuning."""

    if not inputs:
        raise ValueError("decision inputs must not be empty")
    outcomes: list[str] = []
    for item in inputs:
        interval = item.gru_minus_replicator
        prefix = (
            f"Strategy-replication rule — {item.cohort}/{item.split}/{item.label}: "
        )
        if not _finite_interval(interval):
            outcomes.append(
                prefix
                + "the paired GRU-ensemble-minus-replicator PR-AUC interval is "
                "undefined, so the rule cannot establish an observer advantage."
            )
        elif interval.lower <= 0.0:
            outcomes.append(
                prefix
                + "the paired GRU-ensemble-minus-replicator PR-AUC interval "
                "includes or lies below 0; the observer's advantage is reported "
                "as consistent with strategy replication."
            )
        else:
            outcomes.append(
                prefix
                + "the paired GRU-ensemble-minus-replicator PR-AUC interval "
                "excludes 0 above; the observer's advantage is not reported as "
                "consistent with strategy replication by this rule."
            )

    architecture = [
        item
        for item in inputs
        if item.cohort == "architecture"
        and item.split == "heldout_architecture"
        and item.label == "default"
    ]
    if len(architecture) != 1:
        raise ValueError("exactly one held-out-architecture default result is required")
    heldout = architecture[0]
    if heldout.base_rate < 0.01:
        outcomes.append(
            "H2-style generalization rule — held-out architecture/default: the "
            f"base rate is {heldout.base_rate:.6f}, below 0.010000; the split is "
            "reported as uninformative and no generalization claim is made."
        )
    elif heldout.gru_minus_trivial is None or not (
        _strictly_positive(heldout.gru_minus_trivial)
        and _strictly_positive(heldout.gru_minus_replicator)
    ):
        outcomes.append(
            "H2-style generalization rule — held-out architecture/default: the "
            f"base rate is {heldout.base_rate:.6f}, so the split is informative, "
            "but the GRU ensemble did not beat both the trivial reference and "
            "replicator with paired intervals excluding 0 above; the pre-registered "
            "rule finds no evidence for H2-style generalization."
        )
    else:
        outcomes.append(
            "H2-style generalization rule — held-out architecture/default: the "
            f"base rate is {heldout.base_rate:.6f}, and the GRU ensemble beat both "
            "the trivial reference and replicator with paired intervals excluding "
            "0 above; the pre-registered rule finds evidence for H2-style "
            "generalization in this simulator."
        )

    for item in inputs:
        if item.label != "forced_flow":
            continue
        if item.event_count < 30:
            outcomes.append(
                f"Forced-flow power rule — {item.cohort}/{item.split}: "
                f"{item.event_count} test events is fewer than 30; the result is "
                "reported as underpowered."
            )
        else:
            outcomes.append(
                f"Forced-flow power rule — {item.cohort}/{item.split}: "
                f"{item.event_count} test events meets the 30-event minimum; the "
                "result is not flagged as underpowered by this rule."
            )
    return tuple(outcomes)


def run_round3(
    selected_config_path: str | Path = DEFAULT_SELECTED_CONFIG,
    *,
    family_config_path: str | Path = DEFAULT_FAMILY_CONFIG,
    impact_config_path: str | Path = DEFAULT_IMPACT_CONFIG,
    architecture_config_path: str | Path = DEFAULT_ARCHITECTURE_CONFIG,
    workers: int = 10,
    gru_workers: int = 5,
    horizon: int = 10,
    threshold_quantile: float = 0.95,
    return_lookback: int = 10,
    false_alarm_rate: float = 0.05,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 7000,
    lightgbm_seed: int = 5000,
    gru_seeds: Sequence[int] = GRU_SEEDS,
    gru_kwargs: Mapping[str, Any] | None = None,
) -> Round3Result:
    """Generate every fixed split and evaluate the complete observer set."""

    v2_configs = load_v2_configs(
        selected_config_path, family_config_path, impact_config_path
    )
    architecture_configs = load_architecture_configs(
        selected_config_path, architecture_config_path
    )
    specifications = {
        "v2": (
            v2_configs,
            {
                "training": V2_TRAINING_SEEDS,
                "in_distribution": V2_IN_DISTRIBUTION_SEEDS,
                "heldout_families": V2_HELDOUT_FAMILY_SEEDS,
                "heldout_impact": V2_HELDOUT_IMPACT_SEEDS,
            },
        ),
        "architecture": (
            architecture_configs,
            {
                "training": ARCHITECTURE_TRAINING_SEEDS,
                "in_distribution": ARCHITECTURE_IN_DISTRIBUTION_SEEDS,
                "heldout_architecture": HELDOUT_ARCHITECTURE_SEEDS,
            },
        ),
    }
    cohorts: dict[str, CohortResult] = {}
    for cohort_index, (cohort_name, (configs, seed_sets)) in enumerate(
        specifications.items()
    ):
        samples = {
            name: _simulate(configs[name], seeds, workers=workers)
            for name, seeds in seed_sets.items()
        }
        cohorts[cohort_name] = evaluate_cohort(
            samples["training"],
            {name: value for name, value in samples.items() if name != "training"},
            replicator_config=configs["training"],
            horizon=horizon,
            threshold_quantile=threshold_quantile,
            return_lookback=return_lookback,
            false_alarm_rate=false_alarm_rate,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + cohort_index * 100_000,
            lightgbm_seed=lightgbm_seed,
            gru_seeds=gru_seeds,
            gru_workers=gru_workers,
            gru_kwargs=gru_kwargs,
        )
    return Round3Result(cohorts=cohorts)


def evaluate_cohort(
    training: Sequence[EpisodeSample],
    evaluation_sets: Mapping[str, Sequence[EpisodeSample]],
    *,
    replicator_config: Mapping[str, Any],
    horizon: int,
    threshold_quantile: float,
    return_lookback: int,
    false_alarm_rate: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    lightgbm_seed: int,
    gru_seeds: Sequence[int],
    gru_workers: int,
    gru_kwargs: Mapping[str, Any] | None = None,
) -> CohortResult:
    """Fit label-specific training thresholds and all fixed observers."""

    if not training or not evaluation_sets or any(not value for value in evaluation_sets.values()):
        raise ValueError("training and evaluation sets must be nonempty")
    labels: dict[str, LabelResult] = {}
    for label_index, label_name in enumerate(LABEL_NAMES):
        training_flow = _target_flow(training, label_name)
        threshold = fit_unwind_threshold(
            (
                (flow, sample.public.liquidity_proxy)
                for flow, sample in zip(training_flow, training, strict=True)
            ),
            horizon=horizon,
            quantile=threshold_quantile,
        )
        training_tasks = _task_episodes(
            training, label_name=label_name, horizon=horizon, threshold=threshold
        )
        evaluation_tasks = {
            name: _task_episodes(
                samples, label_name=label_name, horizon=horizon, threshold=threshold
            )
            for name, samples in evaluation_sets.items()
        }
        predictions: dict[str, dict[str, tuple[np.ndarray, ...]]] = {
            name: {} for name in evaluation_tasks
        }
        observers = {
            "trivial_reference": TrivialReferenceObserver(
                return_lookback=return_lookback
            ),
            "logistic_regression": LogisticObserver(
                return_lookback=return_lookback
            ),
            "lightgbm": LightGBMObserver(
                return_lookback=return_lookback,
                rng=np.random.default_rng(lightgbm_seed + label_index),
            ),
            "replicator": StrategyReplicationObserver(
                replicator_config,
                rng=np.random.default_rng(lightgbm_seed + label_index),
            ),
        }
        for model_name, observer in observers.items():
            observer.fit(training_tasks)
            for split_name, tasks in evaluation_tasks.items():
                predictions[split_name][model_name] = observer.predict_proba(tasks)

        gru_results = _fit_gru_seeds(
            training_tasks,
            evaluation_tasks,
            seeds=gru_seeds,
            workers=gru_workers,
            return_lookback=return_lookback,
            gru_kwargs=gru_kwargs,
        )
        details: dict[int, str] = {}
        for seed, detail, split_predictions in gru_results:
            details[seed] = detail
            for split_name, values in split_predictions.items():
                predictions[split_name][f"gru_{seed}"] = values
        for split_name in evaluation_tasks:
            predictions[split_name]["gru_ensemble"] = tuple(
                np.mean(
                    np.stack(
                        [
                            predictions[split_name][f"gru_{seed}"][episode_index]
                            for seed in gru_seeds
                        ]
                    ),
                    axis=0,
                )
                for episode_index in range(len(evaluation_tasks[split_name]))
            )

        evaluations: dict[str, EvaluationResult] = {}
        for split_index, (split_name, tasks) in enumerate(evaluation_tasks.items()):
            episode_labels = tuple(
                np.asarray(task.labels, dtype=np.bool_) for task in tasks
            )
            metrics: dict[str, Mapping[str, ConfidenceInterval]] = {}
            for model_index, (model_name, model_predictions) in enumerate(
                predictions[split_name].items()
            ):
                metrics[model_name] = episode_bootstrap_metrics(
                    episode_labels,
                    model_predictions,
                    rng=np.random.default_rng(
                        bootstrap_seed
                        + label_index * 10_000
                        + split_index * 100
                        + model_index
                    ),
                    max_false_alarm_rate=false_alarm_rate,
                    n_resamples=bootstrap_resamples,
                )
            comparisons = {
                comparator: _paired_difference_or_undefined(
                    episode_labels,
                    predictions[split_name]["gru_ensemble"],
                    predictions[split_name][comparator],
                    rng=np.random.default_rng(
                        bootstrap_seed
                        + 50_000
                        + label_index * 1_000
                        + split_index * 10
                        + comparison_index
                    ),
                    n_resamples=bootstrap_resamples,
                )
                for comparison_index, comparator in enumerate(
                    ("replicator", "lightgbm", "trivial_reference")
                )
            }
            seed_estimates = [
                metrics[f"gru_{seed}"]["pr_auc"].estimate for seed in gru_seeds
            ]
            finite_estimates = [value for value in seed_estimates if np.isfinite(value)]
            pr_range = (
                (min(finite_estimates), max(finite_estimates))
                if finite_estimates
                else (np.nan, np.nan)
            )
            evaluations[split_name] = EvaluationResult(
                events=_event_summary(tasks),
                metrics=metrics,
                paired_differences=comparisons,
                gru_pr_auc_range=pr_range,
            )
        labels[label_name] = LabelResult(
            threshold=threshold,
            training_events=_event_summary(training_tasks),
            evaluation=evaluations,
            gru_training_details=details,
        )
    return CohortResult(labels=labels)


def decision_inputs(result: Round3Result) -> tuple[DecisionInputs, ...]:
    """Extract the measured values required by :func:`apply_decision_rules`."""

    values = []
    for cohort_name, cohort in result.cohorts.items():
        for label_name, label in cohort.labels.items():
            for split_name, evaluation in label.evaluation.items():
                values.append(
                    DecisionInputs(
                        cohort=cohort_name,
                        split=split_name,
                        label=label_name,
                        event_count=evaluation.events.event_count,
                        base_rate=evaluation.events.base_rate,
                        gru_minus_replicator=evaluation.paired_differences["replicator"],
                        gru_minus_trivial=evaluation.paired_differences[
                            "trivial_reference"
                        ],
                    )
                )
    return tuple(values)


def render_report(
    result: Round3Result,
    *,
    runtime_seconds: float,
    workers: int,
    gru_workers: int,
    horizon: int = 10,
    threshold_quantile: float = 0.95,
    return_lookback: int = 10,
    false_alarm_rate: float = 0.05,
    bootstrap_resamples: int = 1000,
) -> str:
    """Render all fixed settings, measured tables, and mechanical outcomes."""

    lines = [
        "# Experiment Zero, round 3",
        "",
        "**Synthetic-simulator disclaimer:** Every result in this report comes from the uncalibrated synthetic HerdLab simulator. It is not evidence about real markets, real funds, or live trading behavior.",
        "",
        "## Configuration and runtime",
        "",
        "- Fixed selected setting: `configs/experiment_zero.yaml` (linear impact sensitivity 0.03, q_max 200, order limit 40, drawdown limit 0.20, and the Amendment 1 cooldown rule).",
        "- V2 cohort: `configs/heldout_families.yaml` and `configs/heldout_impact.yaml`; 200 training episodes and 50 episodes in each of in-distribution, held-out-family, and held-out-impact.",
        "- Architecture cohort: `configs/heldout_architecture.yaml`; train population momentum 6/reversion 6/linear 9/MLP 9; held-out population momentum 6/reversion 6/GBDT 18; 200 training episodes and 50 episodes in each evaluation split.",
        f"- Episode seeds: v2 training `{_seed_text(V2_TRAINING_SEEDS)}`, in-distribution `{_seed_text(V2_IN_DISTRIBUTION_SEEDS)}`, held-out-family `{_seed_text(V2_HELDOUT_FAMILY_SEEDS)}`, held-out-impact `{_seed_text(V2_HELDOUT_IMPACT_SEEDS)}`; architecture training `{_seed_text(ARCHITECTURE_TRAINING_SEEDS)}`, in-distribution `{_seed_text(ARCHITECTURE_IN_DISTRIBUTION_SEEDS)}`, held-out-architecture `{_seed_text(HELDOUT_ARCHITECTURE_SEEDS)}`.",
        f"- Labels: default, onset, and forced-flow; horizon {horizon}; each cohort/label threshold is its own training-only {threshold_quantile:.2f} score quantile.",
        f"- Observers: trivial reference, logistic regression, LightGBM, price-only strategy replicator, GRU seeds `{_seed_text(GRU_SEEDS)}`, and their probability ensemble. Public-feature return lookback is {return_lookback}.",
        f"- Metrics: PR-AUC, Brier score, and recall at FAR <= {false_alarm_rate:.2f}; 95% percentile CIs use {bootstrap_resamples} whole-episode resamples. Paired PR-AUC differences use the same resampled episodes for both observers.",
        f"- Parallelism: {workers} simulator workers and {gru_workers} concurrent GRU seed workers.",
        f"- Measured wall runtime: {runtime_seconds / 60.0:.2f} minutes.",
        "",
        "## Training-only thresholds",
        "",
        "| Cohort | Label | Threshold |",
        "| --- | --- | ---: |",
    ]
    for cohort_name, cohort in result.cohorts.items():
        for label_name, label in cohort.labels.items():
            lines.append(f"| {cohort_name} | {label_name} | {label.threshold:.10g} |")

    lines.extend(
        [
            "",
            "## Base rates and event counts",
            "",
            "Counts and observed base rates are descriptive; inferential CIs are reported in the observer and paired-difference tables below.",
            "",
            "| Cohort | Label | Split | Events / labels | Base rate |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for cohort_name, cohort in result.cohorts.items():
        for label_name, label in cohort.labels.items():
            lines.append(
                _event_row(cohort_name, label_name, "training", label.training_events)
            )
            for split_name, evaluation in label.evaluation.items():
                lines.append(
                    _event_row(cohort_name, label_name, split_name, evaluation.events)
                )

    metric_order = ("pr_auc", "brier", "recall_at_false_alarm_rate")
    lines.extend(
        [
            "",
            "## Observer metrics",
            "",
            "Every entry is point estimate [episode-bootstrap 95% CI]. Higher is better for PR-AUC and recall; lower is better for Brier score.",
        ]
    )
    for cohort_name, cohort in result.cohorts.items():
        for label_name, label in cohort.labels.items():
            for split_name, evaluation in label.evaluation.items():
                lines.extend(
                    [
                        "",
                        f"### {cohort_name} / {label_name} / {split_name}",
                        "",
                        "| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |",
                        "| --- | ---: | ---: | ---: |",
                    ]
                )
                for model_name, metrics in evaluation.metrics.items():
                    values = " | ".join(_format_ci(metrics[name]) for name in metric_order)
                    lines.append(f"| {model_name} | {values} |")

    lines.extend(
        [
            "",
            "## Per-seed GRU PR-AUC ranges",
            "",
            "The range is the minimum–maximum of the five per-seed point estimates; each contributing per-seed estimate and CI appears in the observer tables above.",
            "",
            "| Cohort | Label | Split | GRU seed min–max PR-AUC |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for cohort_name, cohort in result.cohorts.items():
        for label_name, label in cohort.labels.items():
            for split_name, evaluation in label.evaluation.items():
                low, high = evaluation.gru_pr_auc_range
                value = "N/A" if not np.all(np.isfinite((low, high))) else f"{low:.4f}–{high:.4f}"
                lines.append(f"| {cohort_name} | {label_name} | {split_name} | {value} |")

    lines.extend(
        [
            "",
            "## Paired PR-AUC differences",
            "",
            "Entries are GRU ensemble minus comparator, with paired whole-episode-bootstrap 95% CIs.",
            "",
            "| Cohort | Label | Split | Minus replicator (95% CI) | Minus LightGBM (95% CI) | Minus trivial (95% CI) |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for cohort_name, cohort in result.cohorts.items():
        for label_name, label in cohort.labels.items():
            for split_name, evaluation in label.evaluation.items():
                differences = evaluation.paired_differences
                lines.append(
                    f"| {cohort_name} | {label_name} | {split_name} | "
                    f"{_format_ci(differences['replicator'])} | "
                    f"{_format_ci(differences['lightgbm'])} | "
                    f"{_format_ci(differences['trivial_reference'])} |"
                )

    lines.extend(["", "## GRU training detail", ""])
    for cohort_name, cohort in result.cohorts.items():
        for label_name, label in cohort.labels.items():
            details = "; ".join(
                f"seed {seed}: {detail}"
                for seed, detail in label.gru_training_details.items()
            )
            lines.append(f"- {cohort_name}/{label_name}: {details}.")

    lines.extend(["", "## Pre-registered rule outcomes (verbatim)", ""])
    lines.extend(f"- {outcome}" for outcome in apply_decision_rules(decision_inputs(result)))
    lines.extend(
        [
            "",
            "## Caveats, deviations, and open questions",
            "",
            "- Observer inputs contain only public records. Agent-only flow and forced flags are reduced to labels before observer construction.",
            "- Thresholds use training episodes only. Confidence intervals resample whole evaluation episodes, never overlapping time steps independently.",
            "- No simulator setting, label rule, observer hyperparameter, seed range, or episode count was changed after viewing round-3 metrics.",
            "- No deviations from the pre-registration and no open questions.",
            "",
        ]
    )
    return "\n".join(lines)


def _simulate(
    config: Mapping[str, Any], seeds: Sequence[int], *, workers: int
) -> tuple[EpisodeSample, ...]:
    if isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be positive")
    tasks = [(deepcopy(dict(config)), int(seed)) for seed in seeds]
    if workers == 1:
        return tuple(_simulate_one(task) for task in tasks)
    with multiprocessing.Pool(processes=workers) as pool:
        return tuple(pool.map(_simulate_one, tasks))


def _simulate_one(task: tuple[dict[str, Any], int]) -> EpisodeSample:
    config, seed = task
    public, private = run_episode(config, seed)
    return EpisodeSample(
        public=public,
        agent_net_flow=np.asarray(private.agent_net_flow, dtype=np.float64),
        forced_selling_flow=forced_selling_flow(private.orders, private.forced),
        seed=seed,
    )


def _target_flow(
    samples: Sequence[EpisodeSample], label_name: str
) -> tuple[np.ndarray, ...]:
    if label_name not in LABEL_NAMES:
        raise ValueError(f"unknown label {label_name!r}")
    return tuple(
        sample.forced_selling_flow
        if label_name == "forced_flow"
        else sample.agent_net_flow
        for sample in samples
    )


def _task_episodes(
    samples: Sequence[EpisodeSample],
    *,
    label_name: str,
    horizon: int,
    threshold: float,
) -> tuple[TaskCEpisode, ...]:
    variant = "onset" if label_name == "onset" else "window"
    flows = _target_flow(samples, label_name)
    return tuple(
        TaskCEpisode(
            public=sample.public,
            labels=unwind_labels(
                flow,
                sample.public.liquidity_proxy,
                horizon=horizon,
                threshold=threshold,
                variant=variant,
            ),
            episode_id=str(sample.seed),
            horizon=horizon,
        )
        for sample, flow in zip(samples, flows, strict=True)
    )


def _event_summary(tasks: Sequence[TaskCEpisode]) -> EventSummary:
    return EventSummary(
        event_count=sum(int(np.asarray(task.labels, dtype=np.bool_).sum()) for task in tasks),
        label_count=sum(len(np.asarray(task.labels)) for task in tasks),
    )


def _fit_gru_seeds(
    training: Sequence[TaskCEpisode],
    evaluation: Mapping[str, Sequence[TaskCEpisode]],
    *,
    seeds: Sequence[int],
    workers: int,
    return_lookback: int,
    gru_kwargs: Mapping[str, Any] | None,
) -> tuple[tuple[int, str, Mapping[str, tuple[np.ndarray, ...]]], ...]:
    if not seeds:
        raise ValueError("GRU seeds must not be empty")
    tasks = [
        (
            tuple(training),
            {name: tuple(value) for name, value in evaluation.items()},
            int(seed),
            return_lookback,
            dict(gru_kwargs or {}),
        )
        for seed in seeds
    ]
    if workers == 1:
        results = [_fit_one_gru(task) for task in tasks]
    else:
        if isinstance(workers, bool) or workers <= 0:
            raise ValueError("GRU workers must be positive")
        with multiprocessing.Pool(processes=min(workers, len(tasks))) as pool:
            results = pool.map(_fit_one_gru, tasks)
    return tuple(results)


def _fit_one_gru(
    task: tuple[
        tuple[TaskCEpisode, ...],
        Mapping[str, tuple[TaskCEpisode, ...]],
        int,
        int,
        dict[str, Any],
    ]
) -> tuple[int, str, Mapping[str, tuple[np.ndarray, ...]]]:
    training, evaluation, seed, return_lookback, kwargs = task
    observer = GRUObserver(
        return_lookback=return_lookback, seed=seed, **kwargs
    ).fit(training)
    predictions = {
        name: observer.predict_proba(episodes) for name, episodes in evaluation.items()
    }
    detail = (
        f"epochs={observer.epochs_trained_}, "
        f"best_validation_loss={observer.best_validation_loss_:.8g}"
    )
    return seed, detail, predictions


def _paired_difference_or_undefined(
    labels: Sequence[np.ndarray],
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    *,
    rng: np.random.Generator,
    n_resamples: int,
) -> ConfidenceInterval:
    if not np.any(np.concatenate(labels)):
        return ConfidenceInterval(np.nan, np.nan, np.nan)
    return paired_pr_auc_difference_ci(
        labels, first, second, rng=rng, n_resamples=n_resamples
    )


def _finite_interval(interval: ConfidenceInterval) -> bool:
    return bool(np.all(np.isfinite((interval.estimate, interval.lower, interval.upper))))


def _strictly_positive(interval: ConfidenceInterval) -> bool:
    return _finite_interval(interval) and interval.lower > 0.0


def _event_row(
    cohort: str, label: str, split: str, summary: EventSummary
) -> str:
    return (
        f"| {cohort} | {label} | {split} | {summary.event_count} / "
        f"{summary.label_count} | {summary.base_rate:.6f} |"
    )


def _format_ci(interval: ConfidenceInterval) -> str:
    if not _finite_interval(interval):
        return "N/A"
    return f"{interval.estimate:.4f} [{interval.lower:.4f}, {interval.upper:.4f}]"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_SELECTED_CONFIG))
    parser.add_argument("--family-config", default=str(DEFAULT_FAMILY_CONFIG))
    parser.add_argument("--impact-config", default=str(DEFAULT_IMPACT_CONFIG))
    parser.add_argument("--architecture-config", default=str(DEFAULT_ARCHITECTURE_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--gru-workers", type=int, default=5)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    result = run_round3(
        args.config,
        family_config_path=args.family_config,
        impact_config_path=args.impact_config,
        architecture_config_path=args.architecture_config,
        workers=args.workers,
        gru_workers=args.gru_workers,
    )
    runtime_seconds = time.perf_counter() - started
    if runtime_seconds > 45.0 * 60.0:
        raise RuntimeError(
            f"round-3 runtime {runtime_seconds / 60.0:.2f} minutes exceeded the "
            "45-minute cap; no report was written and episode counts were not cut"
        )
    report = render_report(
        result,
        runtime_seconds=runtime_seconds,
        workers=args.workers,
        gru_workers=args.gru_workers,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "DecisionInputs",
    "EpisodeSample",
    "Round3Result",
    "apply_decision_rules",
    "decision_inputs",
    "evaluate_cohort",
    "load_architecture_configs",
    "main",
    "render_report",
    "run_round3",
]
