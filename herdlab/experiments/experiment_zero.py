"""Run the pre-specified Experiment Zero v2 observer comparison.

Private simulator output is reduced to target inputs and forced-flow
diagnostics before any observer is constructed. Every observer receives only
``TaskCEpisode`` objects containing a public record and precomputed labels.
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

from herdlab.evaluation.metrics import ConfidenceInterval, episode_bootstrap_metrics
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.inference.baselines import (
    LightGBMObserver,
    LogisticObserver,
    TaskCEpisode,
    TrivialReferenceObserver,
)
from herdlab.inference.targets import fit_unwind_threshold, unwind_labels
from herdlab.inference.temporal import GRUObserver
from herdlab.market.records import PublicRecord


DEFAULT_SELECTED_CONFIG = Path("configs/experiment_zero.yaml")
DEFAULT_FAMILY_CONFIG = Path("configs/heldout_families.yaml")
DEFAULT_IMPACT_CONFIG = Path("configs/heldout_impact.yaml")
DEFAULT_REPORT = Path("docs/reports/experiment_zero_v2.md")
TRAINING_SEEDS = tuple(range(1000, 1200))
IN_DISTRIBUTION_SEEDS = tuple(range(2000, 2050))
HELDOUT_FAMILY_SEEDS = tuple(range(3000, 3050))
HELDOUT_IMPACT_SEEDS = tuple(range(4000, 4050))
LABEL_VARIANTS = ("default", "onset")


@dataclass(frozen=True)
class EpisodeSample:
    """Public episode plus the minimum private data needed for targets."""

    public: PublicRecord
    agent_net_flow: np.ndarray
    forced_flow_steps: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        flow = np.asarray(self.agent_net_flow)
        forced = np.asarray(self.forced_flow_steps)
        if flow.ndim != 1 or forced.shape != flow.shape or forced.dtype != np.bool_:
            raise ValueError(
                "agent_net_flow and boolean forced_flow_steps must be equal-length vectors"
            )


@dataclass(frozen=True)
class EventSummary:
    """Label prevalence and private forced-flow overlap for one split."""

    event_count: int
    label_count: int
    forced_event_count: int

    @property
    def event_base_rate(self) -> float:
        return self.event_count / self.label_count

    @property
    def fraction_events_with_forced_flow(self) -> float:
        if self.event_count == 0:
            return 0.0
        return self.forced_event_count / self.event_count


@dataclass(frozen=True)
class EvaluationSetResult:
    """One evaluation ecology's event summary and observer metrics."""

    events: EventSummary
    metrics: Mapping[str, Mapping[str, ConfidenceInterval]]

    @property
    def event_count(self) -> int:
        return self.events.event_count

    @property
    def label_count(self) -> int:
        return self.events.label_count

    @property
    def event_base_rate(self) -> float:
        return self.events.event_base_rate


@dataclass(frozen=True)
class LabelVariantResult:
    """Training and evaluation results for one target-label variant."""

    training: EventSummary
    evaluation: Mapping[str, EvaluationSetResult]
    observer_details: Mapping[str, str]


@dataclass(frozen=True)
class ExperimentZeroResult:
    """All measured outputs from Experiment Zero v2."""

    threshold: float
    labels: Mapping[str, LabelVariantResult]


def load_v2_configs(
    selected_config_path: str | Path = DEFAULT_SELECTED_CONFIG,
    family_config_path: str | Path = DEFAULT_FAMILY_CONFIG,
    impact_config_path: str | Path = DEFAULT_IMPACT_CONFIG,
) -> dict[str, dict[str, Any]]:
    """Compose selected parameters with the two matched step-7 splits.

    The familiar population is the ``train`` side of the family split. The
    impact holdout applies only the impact-test override to that same familiar
    population, so family composition is not changed in the impact test.
    """

    selected = deepcopy(dict(load_config(selected_config_path)))
    family_spec = load_config(family_config_path)
    impact_spec = load_config(impact_config_path)
    family_train = _population_override(family_spec, "train")
    family_test = _population_override(family_spec, "test")
    impact_test = _population_override(impact_spec, "test")

    familiar = _deep_merge(selected, family_train)
    configs = {
        "training": familiar,
        "in_distribution": deepcopy(familiar),
        "heldout_families": _deep_merge(selected, family_test),
        "heldout_impact": _deep_merge(familiar, impact_test),
    }
    selected_values = _selected_values(selected)
    for name, config in configs.items():
        if _selected_values(config) != selected_values:
            raise ValueError(f"selected impact/leverage values changed in {name}")
    return configs


def run_experiment_zero(
    selected_config_path: str | Path = DEFAULT_SELECTED_CONFIG,
    *,
    family_config_path: str | Path = DEFAULT_FAMILY_CONFIG,
    impact_config_path: str | Path = DEFAULT_IMPACT_CONFIG,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
    in_distribution_seeds: Sequence[int] = IN_DISTRIBUTION_SEEDS,
    heldout_family_seeds: Sequence[int] = HELDOUT_FAMILY_SEEDS,
    heldout_impact_seeds: Sequence[int] = HELDOUT_IMPACT_SEEDS,
    workers: int = 10,
    horizon: int = 10,
    threshold_quantile: float = 0.95,
    return_lookback: int = 10,
    false_alarm_rate: float = 0.05,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 7000,
    lightgbm_seed: int = 5000,
    gru_seed: int = 6000,
    gru_kwargs: Mapping[str, Any] | None = None,
) -> ExperimentZeroResult:
    """Generate all four episode sets in parallel and evaluate observers."""

    seed_sets = {
        "training": training_seeds,
        "in_distribution": in_distribution_seeds,
        "heldout_families": heldout_family_seeds,
        "heldout_impact": heldout_impact_seeds,
    }
    if any(not seeds for seeds in seed_sets.values()):
        raise ValueError("all episode seed sets must be nonempty")
    configs = load_v2_configs(
        selected_config_path, family_config_path, impact_config_path
    )
    samples = {
        name: _simulate(configs[name], seeds, workers=workers)
        for name, seeds in seed_sets.items()
    }
    return evaluate_samples(
        samples.pop("training"),
        samples,
        horizon=horizon,
        threshold_quantile=threshold_quantile,
        return_lookback=return_lookback,
        false_alarm_rate=false_alarm_rate,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        lightgbm_seed=lightgbm_seed,
        gru_seed=gru_seed,
        gru_kwargs=gru_kwargs,
    )


def evaluate_samples(
    training: Sequence[EpisodeSample],
    evaluation_sets: Mapping[str, Sequence[EpisodeSample]],
    *,
    horizon: int,
    threshold_quantile: float,
    return_lookback: int,
    false_alarm_rate: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    lightgbm_seed: int,
    gru_seed: int = 6000,
    gru_kwargs: Mapping[str, Any] | None = None,
) -> ExperimentZeroResult:
    """Fit the training-only threshold and evaluate both target variants."""

    if not training or not evaluation_sets or any(
        not samples for samples in evaluation_sets.values()
    ):
        raise ValueError("training and evaluation sets must be nonempty")

    threshold = fit_unwind_threshold(
        (
            (sample.agent_net_flow, sample.public.liquidity_proxy)
            for sample in training
        ),
        horizon=horizon,
        quantile=threshold_quantile,
    )
    label_results: dict[str, LabelVariantResult] = {}
    for variant_index, variant_name in enumerate(LABEL_VARIANTS):
        target_variant = "window" if variant_name == "default" else "onset"
        training_tasks = _task_episodes(
            training,
            horizon=horizon,
            threshold=threshold,
            variant=target_variant,
        )
        evaluation_tasks = {
            name: _task_episodes(
                samples,
                horizon=horizon,
                threshold=threshold,
                variant=target_variant,
            )
            for name, samples in evaluation_sets.items()
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
                rng=np.random.default_rng(lightgbm_seed + variant_index),
            ),
            "gru": GRUObserver(
                return_lookback=return_lookback,
                seed=gru_seed + variant_index,
                **dict(gru_kwargs or {}),
            ),
        }

        metric_results: dict[
            str, dict[str, Mapping[str, ConfidenceInterval]]
        ] = {name: {} for name in evaluation_tasks}
        observer_details: dict[str, str] = {}
        for model_index, (model_name, observer) in enumerate(observers.items()):
            observer.fit(training_tasks)
            if isinstance(observer, GRUObserver):
                observer_details[model_name] = (
                    f"epochs={observer.epochs_trained_}, "
                    f"best_validation_loss={observer.best_validation_loss_:.8g}"
                )
            for set_index, (set_name, tasks) in enumerate(
                evaluation_tasks.items()
            ):
                predictions = observer.predict_proba(tasks)
                labels = tuple(
                    np.asarray(task.labels, dtype=np.bool_) for task in tasks
                )
                metric_seed = (
                    bootstrap_seed
                    + variant_index * 1000
                    + model_index * 100
                    + set_index
                )
                metric_results[set_name][model_name] = episode_bootstrap_metrics(
                    labels,
                    predictions,
                    rng=np.random.default_rng(metric_seed),
                    max_false_alarm_rate=false_alarm_rate,
                    n_resamples=bootstrap_resamples,
                )

        evaluation = {
            name: EvaluationSetResult(
                events=_event_summary(samples, evaluation_tasks[name], horizon=horizon),
                metrics=metric_results[name],
            )
            for name, samples in evaluation_sets.items()
        }
        label_results[variant_name] = LabelVariantResult(
            training=_event_summary(training, training_tasks, horizon=horizon),
            evaluation=evaluation,
            observer_details=observer_details,
        )
    return ExperimentZeroResult(threshold=threshold, labels=label_results)


def render_report(
    result: ExperimentZeroResult,
    *,
    selected_config_path: str | Path,
    family_config_path: str | Path,
    impact_config_path: str | Path,
    training_seeds: Sequence[int],
    in_distribution_seeds: Sequence[int],
    heldout_family_seeds: Sequence[int],
    heldout_impact_seeds: Sequence[int],
    workers: int,
    n_steps: int,
    horizon: int,
    threshold_quantile: float,
    return_lookback: int,
    false_alarm_rate: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    lightgbm_seed: int,
    gru_seed: int,
    selected_values: tuple[float, float, float],
    runtime_seconds: float,
) -> str:
    """Render only measured v2 outputs and fixed run settings."""

    impact, q_max, order_limit = selected_values
    lines = [
        "# Experiment Zero v2",
        "",
        "This report contains synthetic output from the uncalibrated HerdLab simulator. It is not evidence about real market behavior.",
        "",
        "## Configuration",
        "",
        f"- Selected config: `{selected_config_path}`; impact sensitivity `{impact:g}`, q_max `{q_max:g}`, order limit `{order_limit:g}`.",
        f"- Step-7 splits: `{family_config_path}` and `{impact_config_path}`. Training and in-distribution use the family split's familiar population (momentum/linear/MLP, 10 each); held-out family uses reversion/GBDT (15 each); held-out impact applies only the concave-impact override to the familiar population.",
        "- The selected impact sensitivity, q_max, and order limit are identical in every split.",
        f"- Episodes: {len(training_seeds)} training, {len(in_distribution_seeds)} in-distribution, {len(heldout_family_seeds)} held-out family, {len(heldout_impact_seeds)} held-out impact; {n_steps} steps each; {workers} generation workers.",
        f"- Seeds: training `{_seed_text(training_seeds)}`; in-distribution `{_seed_text(in_distribution_seeds)}`; held-out family `{_seed_text(heldout_family_seeds)}`; held-out impact `{_seed_text(heldout_impact_seeds)}`.",
        f"- Labels: default future-window and onset; horizon {horizon}. Both use the training-only score quantile {threshold_quantile:.2f}, fitted threshold `{result.threshold:.10g}`.",
        f"- Observers: trivial standardized logistic on {return_lookback} causal lags of absolute public OFI plus realized volatility; full-feature logistic; LightGBM (seed {lightgbm_seed}); small GRU with training-episode validation (seeds {gru_seed} and {gru_seed + 1}).",
        f"- Metrics: PR-AUC with event base rate, Brier score, recall at FAR <= {false_alarm_rate:.2f}, and fraction of labeled events whose future window contains nonzero forced flow.",
        f"- Uncertainty: 95% percentile intervals from {bootstrap_resamples} whole-episode bootstrap resamples; seed family begins at {bootstrap_seed}.",
        f"- Measured wall runtime: {runtime_seconds / 60.0:.2f} minutes.",
        "",
        "## Event rates and forced-flow overlap",
        "",
        "Forced-flow overlap uses private ground truth for diagnostics only and is not an observer input. A positive event overlaps forced flow when its strictly future horizon contains a nonzero forced order.",
        "",
        "| Label | Split | Events / labels | Base rate | Events with forced flow | Fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    display_names = {
        "in_distribution": "In-distribution",
        "heldout_families": "Held-out family",
        "heldout_impact": "Held-out impact",
    }
    for label_name, label_result in result.labels.items():
        lines.append(_event_row(label_name.title(), "Training", label_result.training))
        for split_name, evaluation in label_result.evaluation.items():
            lines.append(
                _event_row(
                    label_name.title(),
                    display_names.get(split_name, split_name),
                    evaluation.events,
                )
            )

    lines.extend(
        [
            "",
            "## Observer metrics",
            "",
            "Values in brackets are episode-bootstrap 95% CIs. Higher is better for PR-AUC and recall; lower is better for Brier score.",
        ]
    )
    metric_order = ("pr_auc", "brier", "recall_at_false_alarm_rate")
    for label_name, label_result in result.labels.items():
        lines.extend(["", f"### {label_name.title()} label"])
        for split_name, evaluation in label_result.evaluation.items():
            lines.extend(
                [
                    "",
                    f"#### {display_names.get(split_name, split_name)}",
                    "",
                    "| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for model_name, metrics in evaluation.metrics.items():
                lines.append(
                    "| "
                    + model_name.replace("_", " ").title()
                    + " | "
                    + " | ".join(_format_ci(metrics[name]) for name in metric_order)
                    + " |"
                )
        if label_result.observer_details:
            lines.extend(
                [
                    "",
                    "Training detail: "
                    + "; ".join(
                        f"{name.upper()} {detail}"
                        for name, detail in label_result.observer_details.items()
                    )
                    + ".",
                ]
            )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Every number above comes from this simulator run; no public historical market data were used.",
            "- The 95th-percentile threshold mechanically sets the pooled default-label training base rate near 5%; onset suppresses nearby qualifying windows and therefore has a lower base rate.",
            "- Confidence intervals resample 50 complete evaluation episodes. Overlapping prediction windows are never resampled as independent observations.",
            "- PR-AUC and recall are reported as N/A when a label/split has no positive events (or, for recall, lacks either class); Brier score remains defined and bootstrapped.",
            "- The impact holdout changes the impact function from linear to concave while retaining the familiar agent population. The family holdout changes agent families while retaining population size and configured aggregate capacity.",
            "- Forced-flow overlap is descriptive private ground truth, not a causal attribution and not an observer feature.",
            "- No simulator parameter, threshold rule, or observer hyperparameter was changed after viewing these metrics.",
            "",
            "## Deviations and open questions",
            "",
            "No episode-count reduction or other run deviation. No open questions.",
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
        samples = [_simulate_one(task) for task in tasks]
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            samples = pool.map(_simulate_one, tasks)
    return tuple(samples)


def _simulate_one(task: tuple[dict[str, Any], int]) -> EpisodeSample:
    config, seed = task
    public, private = run_episode(config, seed)
    forced_flow_steps = np.any(
        np.asarray(private.forced, dtype=np.bool_)
        & (np.asarray(private.orders, dtype=np.float64) != 0.0),
        axis=1,
    )
    return EpisodeSample(
        public=public,
        agent_net_flow=np.asarray(private.agent_net_flow, dtype=np.float64),
        forced_flow_steps=np.asarray(forced_flow_steps, dtype=np.bool_),
        seed=seed,
    )


def _task_episodes(
    samples: Sequence[EpisodeSample],
    *,
    horizon: int,
    threshold: float,
    variant: str,
) -> tuple[TaskCEpisode, ...]:
    return tuple(
        TaskCEpisode(
            public=sample.public,
            labels=unwind_labels(
                sample.agent_net_flow,
                sample.public.liquidity_proxy,
                horizon=horizon,
                threshold=threshold,
                variant=variant,
            ),
            episode_id=str(sample.seed),
            horizon=horizon,
        )
        for sample in samples
    )


def _event_summary(
    samples: Sequence[EpisodeSample],
    tasks: Sequence[TaskCEpisode],
    *,
    horizon: int,
) -> EventSummary:
    if len(samples) != len(tasks):
        raise ValueError("samples and tasks must have equal length")
    event_count = 0
    label_count = 0
    forced_event_count = 0
    for sample, task in zip(samples, tasks, strict=True):
        labels = np.asarray(task.labels, dtype=np.bool_)
        event_count += int(labels.sum())
        label_count += len(labels)
        for index in np.flatnonzero(labels):
            forced_event_count += int(
                np.any(sample.forced_flow_steps[index + 1 : index + horizon + 1])
            )
    return EventSummary(event_count, label_count, forced_event_count)


def _population_override(
    specification: Mapping[str, Any], population: str
) -> Mapping[str, Any]:
    populations = specification.get("populations")
    if not isinstance(populations, Mapping) or population not in populations:
        raise ValueError(f"population config has no {population!r} override")
    override = populations[population]
    if not isinstance(override, Mapping):
        raise ValueError(f"population {population!r} override must be a mapping")
    return override


def _deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = deepcopy(value)
    return result


def _selected_values(config: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(config["market"]["impact_sensitivity"]),
        float(config["agents"]["q_max"]),
        float(config["agents"]["order_limit"]),
    )


def _event_row(label: str, split: str, summary: EventSummary) -> str:
    return (
        f"| {label} | {split} | {summary.event_count} / {summary.label_count} | "
        f"{summary.event_base_rate:.6f} | {summary.forced_event_count} / "
        f"{summary.event_count} | {summary.fraction_events_with_forced_flow:.6f} |"
    )


def _format_ci(interval: ConfidenceInterval) -> str:
    if not np.all(
        np.isfinite((interval.estimate, interval.lower, interval.upper))
    ):
        return "N/A"
    return f"{interval.estimate:.4f} [{interval.lower:.4f}, {interval.upper:.4f}]"


def _seed_text(seeds: Sequence[int]) -> str:
    values = tuple(int(seed) for seed in seeds)
    if len(values) > 1 and values == tuple(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ", ".join(str(seed) for seed in values)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fixed v2 experiment and write its Markdown report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_SELECTED_CONFIG))
    parser.add_argument("--family-config", default=str(DEFAULT_FAMILY_CONFIG))
    parser.add_argument("--impact-config", default=str(DEFAULT_IMPACT_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)

    selected_path = Path(args.config)
    if not selected_path.exists():
        raise FileNotFoundError(
            "configs/experiment_zero.yaml is required; leave PLAN step 10 unchecked"
        )

    started = time.perf_counter()
    result = run_experiment_zero(
        selected_path,
        family_config_path=args.family_config,
        impact_config_path=args.impact_config,
        workers=args.workers,
    )
    runtime_seconds = time.perf_counter() - started
    configs = load_v2_configs(selected_path, args.family_config, args.impact_config)
    report = render_report(
        result,
        selected_config_path=selected_path,
        family_config_path=args.family_config,
        impact_config_path=args.impact_config,
        training_seeds=TRAINING_SEEDS,
        in_distribution_seeds=IN_DISTRIBUTION_SEEDS,
        heldout_family_seeds=HELDOUT_FAMILY_SEEDS,
        heldout_impact_seeds=HELDOUT_IMPACT_SEEDS,
        workers=args.workers,
        n_steps=int(configs["training"]["simulation"]["n_steps"]),
        horizon=10,
        threshold_quantile=0.95,
        return_lookback=10,
        false_alarm_rate=0.05,
        bootstrap_resamples=1000,
        bootstrap_seed=7000,
        lightgbm_seed=5000,
        gru_seed=6000,
        selected_values=_selected_values(configs["training"]),
        runtime_seconds=runtime_seconds,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "EpisodeSample",
    "EvaluationSetResult",
    "EventSummary",
    "ExperimentZeroResult",
    "LabelVariantResult",
    "evaluate_samples",
    "load_v2_configs",
    "main",
    "render_report",
    "run_experiment_zero",
]
