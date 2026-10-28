"""Run the pre-registered signal sweep and ablations for round 4.

Private simulator state is reduced to target-flow vectors before observer
construction.  Observer implementations receive only public records paired
with already-computed labels.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from herdlab.evaluation.metrics import (
    ConfidenceInterval,
    episode_bootstrap_metrics,
    paired_pr_auc_difference_ci,
    pr_auc,
)
from herdlab.experiments.experiment_zero import _seed_text
from herdlab.experiments.experiment_zero_round3 import (
    EpisodeSample,
    EventSummary,
    _event_summary,
    _finite_interval,
    _fit_gru_seeds,
    _format_ci,
    _paired_difference_or_undefined,
    _simulate,
    _task_episodes,
)
from herdlab.experiments.run_episode import load_config
from herdlab.inference.baselines import (
    LightGBMObserver,
    LogisticObserver,
    StrategyReplicationObserver,
    TaskCEpisode,
    TrivialReferenceObserver,
)
from herdlab.inference.features import FEATURE_GROUP_COLUMNS, FeatureGroup
from herdlab.inference.targets import fit_unwind_threshold


DEFAULT_SELECTED_CONFIG = Path("configs/experiment_zero.yaml")
DEFAULT_REPORT = Path("docs/reports/experiment_zero_round4.md")
TRAINING_SEEDS = tuple(range(8000, 8200))
TEST_SEEDS = tuple(range(8500, 8550))
GRU_SEEDS = tuple(range(6000, 6003))
SIGNAL_STRENGTHS = (0.0, 0.1, 0.25, 0.5)
LABEL_NAMES = ("default", "forced_flow")
FEATURE_GROUPS: tuple[FeatureGroup, ...] = tuple(FEATURE_GROUP_COLUMNS)  # type: ignore[assignment]
RUNTIME_ESTIMATE_MINUTES = 48.0


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
    evaluation: EvaluationResult
    gru_training_details: Mapping[int, str]


@dataclass(frozen=True)
class ConditionResult:
    labels: Mapping[str, LabelResult]


@dataclass(frozen=True)
class FeatureAblationResult:
    full_metrics: Mapping[str, Mapping[str, ConfidenceInterval]]
    ablated_metrics: Mapping[str, Mapping[str, Mapping[str, ConfidenceInterval]]]
    full_minus_ablated: Mapping[str, Mapping[str, ConfidenceInterval]]
    gru_training_details: Mapping[str, str]


@dataclass(frozen=True)
class Round4Result:
    signal_conditions: Mapping[float, ConditionResult]
    simulator_ablations: Mapping[str, ConditionResult]
    feature_ablations: Mapping[str, FeatureAblationResult]
    private_information_degradation: ConfidenceInterval


@dataclass(frozen=True)
class SignalDecisionInput:
    strength: float
    default_base_rate: float
    gru_minus_replicator: ConfidenceInterval


@dataclass(frozen=True)
class FeatureDecisionInput:
    label: str
    feature_group: str
    observer: str
    full_minus_ablated: ConfidenceInterval


@dataclass(frozen=True)
class ForcedFlowDecisionInput:
    condition: str
    event_count: int


@dataclass(frozen=True)
class _LabelBundle:
    training_tasks: tuple[TaskCEpisode, ...]
    test_tasks: tuple[TaskCEpisode, ...]
    predictions: Mapping[str, tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class _ConditionBundle:
    labels: Mapping[str, _LabelBundle]


def apply_decision_rules(
    signal_inputs: Sequence[SignalDecisionInput],
    feature_inputs: Sequence[FeatureDecisionInput],
    forced_flow_inputs: Sequence[ForcedFlowDecisionInput],
    *,
    degradation_interval: ConfidenceInterval,
) -> tuple[str, ...]:
    """Apply every pre-registered R4-4 rule without discretionary tuning."""

    by_strength = {item.strength: item for item in signal_inputs}
    if len(signal_inputs) != len(SIGNAL_STRENGTHS) or set(by_strength) != set(
        SIGNAL_STRENGTHS
    ):
        raise ValueError("signal decision inputs must contain each registered rho once")
    outcomes: list[str] = []

    endpoints = (by_strength[0.0], by_strength[0.5])
    if any(item.default_base_rate < 0.01 for item in endpoints):
        rates = ", ".join(
            f"rho={item.strength:g}: {item.default_base_rate:.6f}"
            for item in endpoints
        )
        outcomes.append(
            "Private-information degradation rule — default label: at least one "
            f"endpoint has a test base rate below 0.010000 ({rates}); the "
            "comparison is reported as uninformative."
        )
    elif not _finite_interval(degradation_interval):
        outcomes.append(
            "Private-information degradation rule — default label: the jointly "
            "resampled difference-of-differences interval is undefined, so the "
            "rule cannot establish degradation."
        )
    elif degradation_interval.lower > 0.0:
        outcomes.append(
            "Private-information degradation rule — default label: the jointly "
            "resampled CI of [(GRU minus trivial) at rho=0] minus [(GRU minus "
            "trivial) at rho=0.5] lies above 0; private information degrades "
            "inference by the pre-registered rule."
        )
    else:
        outcomes.append(
            "Private-information degradation rule — default label: the jointly "
            "resampled CI of [(GRU minus trivial) at rho=0] minus [(GRU minus "
            "trivial) at rho=0.5] includes or lies below 0; the pre-registered "
            "rule does not find that private information degrades inference."
        )

    for item in signal_inputs:
        prefix = f"Strategy-replication rule — rho={item.strength:g}/default: "
        if item.default_base_rate < 0.01:
            outcomes.append(
                prefix
                + f"the test base rate is {item.default_base_rate:.6f}, below "
                "0.010000; the condition is reported as uninformative."
            )
        elif not _finite_interval(item.gru_minus_replicator):
            outcomes.append(
                prefix
                + "the paired GRU-ensemble-minus-replicator PR-AUC interval is "
                "undefined, so the rule cannot establish an observer advantage."
            )
        elif item.gru_minus_replicator.lower <= 0.0:
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

    if not feature_inputs:
        raise ValueError("feature decision inputs must not be empty")
    for item in feature_inputs:
        interval = item.full_minus_ablated
        prefix = (
            f"Feature-use rule — {item.label}/{item.observer}/"
            f"{item.feature_group}: "
        )
        if _finite_interval(interval) and interval.lower > 0.0:
            outcomes.append(
                prefix
                + "the paired full-minus-ablated PR-AUC interval excludes 0 "
                "above; the feature group is reported as used."
            )
        else:
            outcomes.append(
                prefix
                + "the paired full-minus-ablated PR-AUC interval does not exclude "
                "0 above; the feature group is not reported as used."
            )

    if not forced_flow_inputs:
        raise ValueError("forced-flow decision inputs must not be empty")
    for item in forced_flow_inputs:
        if item.event_count < 30:
            outcomes.append(
                f"Forced-flow power rule — {item.condition}: {item.event_count} "
                "test events is fewer than 30; the result is reported as "
                "underpowered."
            )
        else:
            outcomes.append(
                f"Forced-flow power rule — {item.condition}: {item.event_count} "
                "test events meets the 30-event minimum; the result is not "
                "flagged as underpowered by this rule."
            )
    return tuple(outcomes)


def joint_signal_degradation_ci(
    rho_zero_labels: Sequence[np.ndarray],
    rho_zero_gru: Sequence[np.ndarray],
    rho_zero_trivial: Sequence[np.ndarray],
    rho_half_labels: Sequence[np.ndarray],
    rho_half_gru: Sequence[np.ndarray],
    rho_half_trivial: Sequence[np.ndarray],
    *,
    rng: np.random.Generator,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Bootstrap the registered difference-of-differences jointly by seed.

    The same episode indices are selected in both rho conditions on every
    replicate because test seeds are shared across the sweep.
    """

    collections = (
        rho_zero_labels,
        rho_zero_gru,
        rho_zero_trivial,
        rho_half_labels,
        rho_half_gru,
        rho_half_trivial,
    )
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not collections[0] or any(len(value) != len(collections[0]) for value in collections):
        raise ValueError("all joint-resample inputs must have the same nonzero episode count")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples <= 0
    ):
        raise ValueError("n_resamples must be a positive integer")
    level = float(confidence)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must be between zero and one")

    arrays = tuple(tuple(np.asarray(item) for item in collection) for collection in collections)

    def statistic(selected: np.ndarray) -> float:
        pooled = [np.concatenate([collection[index] for index in selected]) for collection in arrays]
        return float(
            (pr_auc(pooled[0], pooled[1]) - pr_auc(pooled[0], pooled[2]))
            - (pr_auc(pooled[3], pooled[4]) - pr_auc(pooled[3], pooled[5]))
        )

    all_indices = np.arange(len(arrays[0]), dtype=np.int64)
    estimate = statistic(all_indices)
    samples = np.empty(int(n_resamples), dtype=np.float64)
    accepted = 0
    attempts = 0
    while accepted < int(n_resamples) and attempts < int(n_resamples) * 100:
        attempts += 1
        selected = rng.integers(0, len(arrays[0]), size=len(arrays[0]))
        try:
            value = statistic(selected)
        except ValueError:
            continue
        if np.isfinite(value):
            samples[accepted] = value
            accepted += 1
    if accepted < int(n_resamples):
        raise ValueError("joint PR-AUC statistic is undefined for too many resamples")
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    return ConfidenceInterval(estimate, float(lower), float(upper))


def run_round4(
    selected_config_path: str | Path = DEFAULT_SELECTED_CONFIG,
    *,
    workers: int = 10,
    gru_workers: int = 3,
    horizon: int = 10,
    threshold_quantile: float = 0.95,
    return_lookback: int = 10,
    false_alarm_rate: float = 0.05,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 9000,
    lightgbm_seed: int = 5000,
    gru_seeds: Sequence[int] = GRU_SEEDS,
    gru_kwargs: Mapping[str, Any] | None = None,
) -> Round4Result:
    """Generate all registered R4-4 conditions and fit their observers."""

    base_config = deepcopy(dict(load_config(selected_config_path)))
    signal_results: dict[float, ConditionResult] = {}
    signal_bundles: dict[float, _ConditionBundle] = {}
    for condition_index, strength in enumerate(SIGNAL_STRENGTHS):
        config = _condition_config(base_config, strength=strength)
        result, bundle = _run_condition(
            config,
            workers=workers,
            gru_workers=gru_workers,
            horizon=horizon,
            threshold_quantile=threshold_quantile,
            return_lookback=return_lookback,
            false_alarm_rate=false_alarm_rate,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + condition_index * 100_000,
            lightgbm_seed=lightgbm_seed,
            gru_seeds=gru_seeds,
            gru_kwargs=gru_kwargs,
        )
        signal_results[strength] = result
        signal_bundles[strength] = bundle

    feature_ablations = _run_feature_ablations(
        signal_results[0.0],
        signal_bundles[0.0],
        return_lookback=return_lookback,
        false_alarm_rate=false_alarm_rate,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 500_000,
        lightgbm_seed=lightgbm_seed,
        gru_kwargs=gru_kwargs,
    )

    simulator_results: dict[str, ConditionResult] = {}
    for ablation_index, ablation_name in enumerate(("impact_off", "risk_off")):
        config = _condition_config(base_config, strength=0.0, simulator_ablation=ablation_name)
        result, _ = _run_condition(
            config,
            workers=workers,
            gru_workers=gru_workers,
            horizon=horizon,
            threshold_quantile=threshold_quantile,
            return_lookback=return_lookback,
            false_alarm_rate=false_alarm_rate,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + (6 + ablation_index) * 100_000,
            lightgbm_seed=lightgbm_seed,
            gru_seeds=gru_seeds,
            gru_kwargs=gru_kwargs,
        )
        simulator_results[ablation_name] = result

    zero = signal_bundles[0.0].labels["default"]
    half = signal_bundles[0.5].labels["default"]
    degradation = joint_signal_degradation_ci(
        tuple(np.asarray(task.labels, dtype=np.bool_) for task in zero.test_tasks),
        zero.predictions["gru_ensemble"],
        zero.predictions["trivial_reference"],
        tuple(np.asarray(task.labels, dtype=np.bool_) for task in half.test_tasks),
        half.predictions["gru_ensemble"],
        half.predictions["trivial_reference"],
        rng=np.random.default_rng(bootstrap_seed + 900_000),
        n_resamples=bootstrap_resamples,
    )
    return Round4Result(
        signal_conditions=signal_results,
        simulator_ablations=simulator_results,
        feature_ablations=feature_ablations,
        private_information_degradation=degradation,
    )


def _run_condition(
    config: Mapping[str, Any],
    *,
    workers: int,
    gru_workers: int,
    horizon: int,
    threshold_quantile: float,
    return_lookback: int,
    false_alarm_rate: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    lightgbm_seed: int,
    gru_seeds: Sequence[int],
    gru_kwargs: Mapping[str, Any] | None,
) -> tuple[ConditionResult, _ConditionBundle]:
    training = _simulate(config, TRAINING_SEEDS, workers=workers)
    test = _simulate(config, TEST_SEEDS, workers=workers)
    labels: dict[str, LabelResult] = {}
    bundles: dict[str, _LabelBundle] = {}
    for label_index, label_name in enumerate(LABEL_NAMES):
        threshold = _fit_threshold(training, label_name, horizon, threshold_quantile)
        training_tasks = _task_episodes(
            training, label_name=label_name, horizon=horizon, threshold=threshold
        )
        test_tasks = _task_episodes(
            test, label_name=label_name, horizon=horizon, threshold=threshold
        )
        predictions: dict[str, tuple[np.ndarray, ...]] = {}
        observers = {
            "trivial_reference": TrivialReferenceObserver(return_lookback=return_lookback),
            "logistic_regression": LogisticObserver(return_lookback=return_lookback),
            "lightgbm": LightGBMObserver(
                return_lookback=return_lookback,
                rng=np.random.default_rng(lightgbm_seed + label_index),
            ),
            "replicator": StrategyReplicationObserver(
                config, rng=np.random.default_rng(lightgbm_seed + label_index)
            ),
        }
        for model_name, observer in observers.items():
            predictions[model_name] = observer.fit(training_tasks).predict_proba(test_tasks)

        details: dict[int, str] = {}
        for seed, detail, split_predictions in _fit_gru_seeds(
            training_tasks,
            {"test": test_tasks},
            seeds=gru_seeds,
            workers=gru_workers,
            return_lookback=return_lookback,
            gru_kwargs=gru_kwargs,
        ):
            details[seed] = detail
            predictions[f"gru_{seed}"] = split_predictions["test"]
        predictions["gru_ensemble"] = tuple(
            np.mean(
                np.stack([predictions[f"gru_{seed}"][index] for seed in gru_seeds]),
                axis=0,
            )
            for index in range(len(test_tasks))
        )

        episode_labels = tuple(np.asarray(task.labels, dtype=np.bool_) for task in test_tasks)
        metrics = {
            model_name: episode_bootstrap_metrics(
                episode_labels,
                values,
                rng=np.random.default_rng(
                    bootstrap_seed + label_index * 10_000 + model_index
                ),
                max_false_alarm_rate=false_alarm_rate,
                n_resamples=bootstrap_resamples,
            )
            for model_index, (model_name, values) in enumerate(predictions.items())
        }
        comparisons = {
            comparator: _paired_difference_or_undefined(
                episode_labels,
                predictions["gru_ensemble"],
                predictions[comparator],
                rng=np.random.default_rng(
                    bootstrap_seed + 50_000 + label_index * 100 + index
                ),
                n_resamples=bootstrap_resamples,
            )
            for index, comparator in enumerate(
                ("replicator", "lightgbm", "trivial_reference")
            )
        }
        seed_estimates = [metrics[f"gru_{seed}"]["pr_auc"].estimate for seed in gru_seeds]
        finite = [value for value in seed_estimates if np.isfinite(value)]
        pr_range = (min(finite), max(finite)) if finite else (np.nan, np.nan)
        labels[label_name] = LabelResult(
            threshold=threshold,
            training_events=_event_summary(training_tasks),
            evaluation=EvaluationResult(
                events=_event_summary(test_tasks),
                metrics=metrics,
                paired_differences=comparisons,
                gru_pr_auc_range=pr_range,
            ),
            gru_training_details=details,
        )
        bundles[label_name] = _LabelBundle(training_tasks, test_tasks, predictions)
    return ConditionResult(labels), _ConditionBundle(bundles)


def _run_feature_ablations(
    full_result: ConditionResult,
    full_bundle: _ConditionBundle,
    *,
    return_lookback: int,
    false_alarm_rate: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    lightgbm_seed: int,
    gru_kwargs: Mapping[str, Any] | None,
) -> Mapping[str, FeatureAblationResult]:
    results: dict[str, FeatureAblationResult] = {}
    for label_index, label_name in enumerate(LABEL_NAMES):
        bundle = full_bundle.labels[label_name]
        episode_labels = tuple(
            np.asarray(task.labels, dtype=np.bool_) for task in bundle.test_tasks
        )
        full_predictions = {
            "lightgbm": bundle.predictions["lightgbm"],
            "gru_6000": bundle.predictions["gru_6000"],
        }
        ablated_metrics: dict[str, Mapping[str, Mapping[str, ConfidenceInterval]]] = {}
        differences: dict[str, Mapping[str, ConfidenceInterval]] = {}
        training_details: dict[str, str] = {}
        for group_index, group in enumerate(FEATURE_GROUPS):
            lightgbm = LightGBMObserver(
                return_lookback=return_lookback,
                rng=np.random.default_rng(lightgbm_seed + label_index),
                exclude_feature_groups=(group,),
            ).fit(bundle.training_tasks)
            predictions = {"lightgbm": lightgbm.predict_proba(bundle.test_tasks)}
            kwargs = dict(gru_kwargs or {})
            kwargs["exclude_feature_groups"] = (group,)
            gru_result = _fit_gru_seeds(
                bundle.training_tasks,
                {"test": bundle.test_tasks},
                seeds=(6000,),
                workers=1,
                return_lookback=return_lookback,
                gru_kwargs=kwargs,
            )[0]
            _, detail, split_predictions = gru_result
            predictions["gru_6000"] = split_predictions["test"]
            training_details[group] = detail
            ablated_metrics[group] = {
                observer: episode_bootstrap_metrics(
                    episode_labels,
                    values,
                    rng=np.random.default_rng(
                        bootstrap_seed
                        + label_index * 10_000
                        + group_index * 100
                        + observer_index
                    ),
                    max_false_alarm_rate=false_alarm_rate,
                    n_resamples=bootstrap_resamples,
                )
                for observer_index, (observer, values) in enumerate(predictions.items())
            }
            differences[group] = {
                observer: _paired_difference_or_undefined(
                    episode_labels,
                    full_predictions[observer],
                    predictions[observer],
                    rng=np.random.default_rng(
                        bootstrap_seed
                        + 50_000
                        + label_index * 1_000
                        + group_index * 10
                        + observer_index
                    ),
                    n_resamples=bootstrap_resamples,
                )
                for observer_index, observer in enumerate(full_predictions)
            }
        full_metrics = {
            observer: full_result.labels[label_name].evaluation.metrics[observer]
            for observer in full_predictions
        }
        results[label_name] = FeatureAblationResult(
            full_metrics=full_metrics,
            ablated_metrics=ablated_metrics,
            full_minus_ablated=differences,
            gru_training_details=training_details,
        )
    return results


def _fit_threshold(
    samples: Sequence[EpisodeSample],
    label_name: str,
    horizon: int,
    quantile: float,
) -> float:
    flows = tuple(
        sample.forced_selling_flow if label_name == "forced_flow" else sample.agent_net_flow
        for sample in samples
    )
    return fit_unwind_threshold(
        (
            (flow, sample.public.liquidity_proxy)
            for flow, sample in zip(flows, samples, strict=True)
        ),
        horizon=horizon,
        quantile=quantile,
    )


def _condition_config(
    base: Mapping[str, Any],
    *,
    strength: float,
    simulator_ablation: str | None = None,
) -> dict[str, Any]:
    config = deepcopy(dict(base))
    config["agents"]["private_signal"]["strength"] = float(strength)
    config["agents"]["learned_overlap_mode"] = "none"
    if simulator_ablation == "impact_off":
        config["market"]["impact_sensitivity"] = 0.0
    elif simulator_ablation == "risk_off":
        config["agents"]["risk"]["enabled"] = False
    elif simulator_ablation is not None:
        raise ValueError(f"unknown simulator ablation {simulator_ablation!r}")
    return config


def decision_inputs(
    result: Round4Result,
) -> tuple[
    tuple[SignalDecisionInput, ...],
    tuple[FeatureDecisionInput, ...],
    tuple[ForcedFlowDecisionInput, ...],
]:
    signals = tuple(
        SignalDecisionInput(
            strength=strength,
            default_base_rate=condition.labels["default"].evaluation.events.base_rate,
            gru_minus_replicator=condition.labels["default"].evaluation.paired_differences[
                "replicator"
            ],
        )
        for strength, condition in result.signal_conditions.items()
    )
    features = tuple(
        FeatureDecisionInput(label_name, group, observer, interval)
        for label_name, label in result.feature_ablations.items()
        for group, observers in label.full_minus_ablated.items()
        for observer, interval in observers.items()
    )
    forced = tuple(
        [
            ForcedFlowDecisionInput(
                f"rho={strength:g}/forced_flow",
                condition.labels["forced_flow"].evaluation.events.event_count,
            )
            for strength, condition in result.signal_conditions.items()
        ]
        + [
            ForcedFlowDecisionInput(
                f"{name}/forced_flow",
                condition.labels["forced_flow"].evaluation.events.event_count,
            )
            for name, condition in result.simulator_ablations.items()
        ]
    )
    return signals, features, forced


def render_report(
    result: Round4Result,
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
    """Render configuration, every measured table, and mechanical outcomes."""

    lines = [
        "# Experiment Zero, round 4: signal sweep and ablations",
        "",
        "**Synthetic-simulator disclaimer:** Every result in this report comes from the uncalibrated synthetic HerdLab simulator. It is not evidence about real markets, real funds, or live trading behavior.",
        "",
        "## Runtime estimate, configuration, and measured runtime",
        "",
        f"- Pre-run combined R4-4 plus R4-5 runtime estimate: {RUNTIME_ESTIMATE_MINUTES:.0f} minutes, below the pre-registered ~60-minute stop threshold. The estimate used the measured 20.80-minute round-3 run and the registered episode/model workloads; no episode, seed, or resample count was cut.",
        f"- Measured R4-4 wall runtime: {runtime_seconds / 60.0:.2f} minutes.",
        "- Fixed selected setting: `configs/experiment_zero.yaml`; signal beta 1; learned overlap mode `none`; 500 steps per episode.",
        f"- Signal strengths: {', '.join(f'{value:g}' for value in SIGNAL_STRENGTHS)}; training seeds `{_seed_text(TRAINING_SEEDS)}` and test seeds `{_seed_text(TEST_SEEDS)}` at every strength.",
        f"- Labels: default (primary) and forced-flow (secondary); horizon {horizon}; a separate training-only {threshold_quantile:.2f} threshold is fitted within every condition.",
        f"- Observers: trivial, logistic, LightGBM, replicator, GRU seeds `{_seed_text(GRU_SEEDS)}`, and the seed-averaged GRU ensemble; return lookback {return_lookback}.",
        "- Feature ablations on rho=0: order flow, volatility, liquidity, and volume removed one group at a time; LightGBM and GRU seed 6000 refitted.",
        "- Simulator ablations on rho=0: impact sensitivity set to 0, or risk rules disabled; the same training and test seeds and full observer suite were used.",
        f"- Metrics: PR-AUC, Brier, and recall at FAR <= {false_alarm_rate:.2f}; 95% percentile CIs use {bootstrap_resamples} whole-episode resamples. Paired and cross-rho comparisons keep episode pairing.",
        f"- Parallelism: {workers} simulator workers and {gru_workers} concurrent GRU workers.",
        "",
        "## Thresholds and base rates",
        "",
        "Base rates are descriptive. Observer and comparison uncertainty is reported with whole-episode-bootstrap CIs below.",
        "",
        "| Condition | Label | Training threshold | Training events / labels | Training base rate | Test events / labels | Test base rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_conditions = [
        *(
            (f"rho={strength:g}", value)
            for strength, value in result.signal_conditions.items()
        ),
        *result.simulator_ablations.items(),
    ]
    for condition_name, condition in all_conditions:
        for label_name, label in condition.labels.items():
            train = label.training_events
            test = label.evaluation.events
            lines.append(
                f"| {condition_name} | {label_name} | {label.threshold:.10g} | "
                f"{train.event_count} / {train.label_count} | {train.base_rate:.6f} | "
                f"{test.event_count} / {test.label_count} | {test.base_rate:.6f} |"
            )

    metric_order = ("pr_auc", "brier", "recall_at_false_alarm_rate")
    lines.extend(
        [
            "",
            "## Observer metrics",
            "",
            "Entries are point estimate [episode-bootstrap 95% CI]. Higher is better for PR-AUC and recall; lower is better for Brier.",
        ]
    )
    for condition_name, condition in all_conditions:
        for label_name, label in condition.labels.items():
            lines.extend(
                [
                    "",
                    f"### {condition_name} / {label_name}",
                    "",
                    "| Observer | PR-AUC (95% CI) | Brier (95% CI) | Recall at FAR <= 0.05 (95% CI) |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for observer, metrics in label.evaluation.metrics.items():
                values = " | ".join(_format_ci(metrics[name]) for name in metric_order)
                lines.append(f"| {observer} | {values} |")

    lines.extend(
        [
            "",
            "## Per-seed GRU PR-AUC ranges",
            "",
            "Each range is the minimum–maximum of the three per-seed point estimates; the contributing estimates and CIs appear above.",
            "",
            "| Condition | Label | GRU seed min–max PR-AUC |",
            "| --- | --- | ---: |",
        ]
    )
    for condition_name, condition in all_conditions:
        for label_name, label in condition.labels.items():
            low, high = label.evaluation.gru_pr_auc_range
            value = "N/A" if not np.all(np.isfinite((low, high))) else f"{low:.4f}–{high:.4f}"
            lines.append(f"| {condition_name} | {label_name} | {value} |")

    lines.extend(
        [
            "",
            "## Main paired PR-AUC differences",
            "",
            "Entries are GRU ensemble minus comparator with paired whole-episode-bootstrap 95% CIs.",
            "",
            "| Condition | Label | Minus replicator (95% CI) | Minus LightGBM (95% CI) | Minus trivial (95% CI) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for condition_name, condition in all_conditions:
        for label_name, label in condition.labels.items():
            values = label.evaluation.paired_differences
            lines.append(
                f"| {condition_name} | {label_name} | {_format_ci(values['replicator'])} | "
                f"{_format_ci(values['lightgbm'])} | {_format_ci(values['trivial_reference'])} |"
            )

    lines.extend(
        [
            "",
            "## Private-information comparison",
            "",
            "The registered statistic is [(GRU ensemble minus trivial) at rho=0] minus [(GRU ensemble minus trivial) at rho=0.5]. The bootstrap selects the same test-seed episodes in both conditions on every replicate.",
            "",
            "| Statistic | Estimate and jointly resampled 95% CI |",
            "| --- | ---: |",
            f"| Difference of differences | {_format_ci(result.private_information_degradation)} |",
            "",
            "## Feature ablations at rho=0",
            "",
            "Full and ablated entries are PR-AUC point estimate [episode-bootstrap 95% CI]. Differences use paired whole-episode resamples.",
            "",
            "| Label | Removed group | Observer | Full PR-AUC (95% CI) | Ablated PR-AUC (95% CI) | Full minus ablated (95% CI) | Test base rate |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label_name, ablation in result.feature_ablations.items():
        base_rate = result.signal_conditions[0.0].labels[label_name].evaluation.events.base_rate
        for group, observer_metrics in ablation.ablated_metrics.items():
            for observer, metrics in observer_metrics.items():
                lines.append(
                    f"| {label_name} | {group} | {observer} | "
                    f"{_format_ci(ablation.full_metrics[observer]['pr_auc'])} | "
                    f"{_format_ci(metrics['pr_auc'])} | "
                    f"{_format_ci(ablation.full_minus_ablated[group][observer])} | "
                    f"{base_rate:.6f} |"
                )

    lines.extend(["", "## GRU training detail", ""])
    for condition_name, condition in all_conditions:
        for label_name, label in condition.labels.items():
            details = "; ".join(
                f"seed {seed}: {detail}" for seed, detail in label.gru_training_details.items()
            )
            lines.append(f"- {condition_name}/{label_name}: {details}.")
    for label_name, ablation in result.feature_ablations.items():
        details = "; ".join(
            f"without {group}: {detail}"
            for group, detail in ablation.gru_training_details.items()
        )
        lines.append(f"- feature ablations/{label_name}/GRU seed 6000: {details}.")

    signals, features, forced = decision_inputs(result)
    outcomes = apply_decision_rules(
        signals,
        features,
        forced,
        degradation_interval=result.private_information_degradation,
    )
    lines.extend(["", "## Pre-registered rule outcomes (verbatim)", ""])
    lines.extend(f"- {outcome}" for outcome in outcomes)
    lines.extend(
        [
            "",
            "## Caveats, deviations, and open questions",
            "",
            "- Observer inputs contain only public records. Agent-only flow and forced flags are reduced to targets before observer construction.",
            "- Thresholds use training episodes only. CIs resample whole episodes, never overlapping time steps independently.",
            "- The rho=0 condition uses the registered private-signal bypass and therefore preserves the round-3 simulator path.",
            "- No rho, beta, seed range, threshold, observer hyperparameter, episode count, or resample count was changed after viewing metrics.",
            "- No deviations from the pre-registration and no open questions.",
            "",
        ]
    )
    return "\n".join(lines)


def _runtime_stop_report(estimated_minutes: float) -> str:
    return "\n".join(
        [
            "# Experiment Zero, round 4: runtime stop",
            "",
            "**Synthetic-simulator disclaimer:** No round-4 result was generated; the planned work concerns only the uncalibrated synthetic HerdLab simulator.",
            "",
            f"Pre-run combined R4-4 plus R4-5 runtime estimate: {estimated_minutes:.2f} minutes.",
            "",
            "This exceeds the pre-registered ~60-minute threshold, so the run stopped before generating episodes or viewing metrics. No episode counts were cut.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_SELECTED_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--gru-workers", type=int, default=3)
    args = parser.parse_args(argv)

    print(
        f"Pre-run combined R4-4 plus R4-5 runtime estimate: "
        f"{RUNTIME_ESTIMATE_MINUTES:.0f} minutes.",
        flush=True,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if RUNTIME_ESTIMATE_MINUTES > 60.0:
        report_path.write_text(
            _runtime_stop_report(RUNTIME_ESTIMATE_MINUTES), encoding="utf-8"
        )
        return

    started = time.perf_counter()
    result = run_round4(
        args.config, workers=args.workers, gru_workers=args.gru_workers
    )
    runtime_seconds = time.perf_counter() - started
    report_path.write_text(
        render_report(
            result,
            runtime_seconds=runtime_seconds,
            workers=args.workers,
            gru_workers=args.gru_workers,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FeatureDecisionInput",
    "ForcedFlowDecisionInput",
    "Round4Result",
    "SignalDecisionInput",
    "apply_decision_rules",
    "decision_inputs",
    "joint_signal_degradation_ci",
    "main",
    "render_report",
    "run_round4",
]
