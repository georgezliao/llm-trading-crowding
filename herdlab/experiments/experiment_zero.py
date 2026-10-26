"""Run the small Experiment Zero baseline pipeline smoke test.

This module deliberately keeps private simulator output local to target
construction.  Baseline observers receive only :class:`TaskCEpisode` objects,
which pair observer-safe public records with already-computed binary labels.
The checked-in defaults are fixed smoke-test choices, not tuned research
hyperparameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from herdlab.evaluation.metrics import ConfidenceInterval, episode_bootstrap_metrics
from herdlab.experiments.generate import load_population_config
from herdlab.experiments.run_episode import run_episode
from herdlab.inference.baselines import (
    LightGBMObserver,
    LogisticObserver,
    TaskCEpisode,
)
from herdlab.inference.targets import fit_unwind_threshold, unwind_labels
from herdlab.market.records import PublicRecord


DEFAULT_CONFIG = Path("configs/heldout_families.yaml")
DEFAULT_REPORT = Path("docs/reports/experiment_zero_smoke.md")
TRAINING_SEEDS = tuple(range(1000, 1008))
IN_DISTRIBUTION_SEEDS = tuple(range(2000, 2004))
HELDOUT_FAMILY_SEEDS = tuple(range(3000, 3004))


@dataclass(frozen=True)
class EpisodeSample:
    """The minimum data retained between simulation and target construction."""

    public: PublicRecord
    agent_net_flow: np.ndarray
    seed: int


@dataclass(frozen=True)
class EvaluationSetResult:
    """Event frequency and baseline uncertainty for one evaluation ecology."""

    event_count: int
    label_count: int
    metrics: Mapping[str, Mapping[str, ConfidenceInterval]]

    @property
    def event_base_rate(self) -> float:
        return self.event_count / self.label_count


@dataclass(frozen=True)
class ExperimentZeroResult:
    """All outputs needed to render the reproducible smoke-test report."""

    threshold: float
    training_event_count: int
    training_label_count: int
    evaluation: Mapping[str, EvaluationSetResult]

    @property
    def training_event_base_rate(self) -> float:
        return self.training_event_count / self.training_label_count


def run_experiment_zero(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
    in_distribution_seeds: Sequence[int] = IN_DISTRIBUTION_SEEDS,
    heldout_family_seeds: Sequence[int] = HELDOUT_FAMILY_SEEDS,
    horizon: int = 10,
    threshold_quantile: float = 0.95,
    return_lookback: int = 10,
    false_alarm_rate: float = 0.05,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 4000,
    lightgbm_seed: int = 5000,
) -> ExperimentZeroResult:
    """Simulate fixed episode sets and evaluate both step-14 observers."""

    if not training_seeds or not in_distribution_seeds or not heldout_family_seeds:
        raise ValueError("training and both evaluation seed sets must be nonempty")

    familiar_config = load_population_config(config_path, "train")
    heldout_config = load_population_config(config_path, "test")
    training = _simulate(familiar_config, training_seeds)
    in_distribution = _simulate(familiar_config, in_distribution_seeds)
    heldout = _simulate(heldout_config, heldout_family_seeds)
    return evaluate_samples(
        training,
        {
            "in_distribution": in_distribution,
            "heldout_families": heldout,
        },
        horizon=horizon,
        threshold_quantile=threshold_quantile,
        return_lookback=return_lookback,
        false_alarm_rate=false_alarm_rate,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        lightgbm_seed=lightgbm_seed,
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
) -> ExperimentZeroResult:
    """Fit a training-only target threshold and evaluate public observers."""

    if not training or not evaluation_sets or any(not items for items in evaluation_sets.values()):
        raise ValueError("training and evaluation sets must be nonempty")

    threshold = fit_unwind_threshold(
        (
            (sample.agent_net_flow, sample.public.liquidity_proxy)
            for sample in training
        ),
        horizon=horizon,
        quantile=threshold_quantile,
    )
    training_tasks = _task_episodes(training, horizon=horizon, threshold=threshold)
    evaluation_tasks = {
        name: _task_episodes(samples, horizon=horizon, threshold=threshold)
        for name, samples in evaluation_sets.items()
    }

    observers = {
        "logistic_regression": LogisticObserver(return_lookback=return_lookback),
        "lightgbm": LightGBMObserver(
            return_lookback=return_lookback,
            rng=np.random.default_rng(lightgbm_seed),
        ),
    }
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    metric_results: dict[str, dict[str, Mapping[str, ConfidenceInterval]]] = {
        name: {} for name in evaluation_tasks
    }
    for model_name, observer in observers.items():
        observer.fit(training_tasks)
        for set_name, tasks in evaluation_tasks.items():
            predictions = observer.predict_proba(tasks)
            labels = tuple(np.asarray(task.labels, dtype=np.bool_) for task in tasks)
            metric_results[set_name][model_name] = episode_bootstrap_metrics(
                labels,
                predictions,
                rng=bootstrap_rng,
                max_false_alarm_rate=false_alarm_rate,
                n_resamples=bootstrap_resamples,
            )

    evaluation: dict[str, EvaluationSetResult] = {}
    for set_name, tasks in evaluation_tasks.items():
        labels = np.concatenate([np.asarray(task.labels, dtype=np.bool_) for task in tasks])
        evaluation[set_name] = EvaluationSetResult(
            event_count=int(labels.sum()),
            label_count=len(labels),
            metrics=metric_results[set_name],
        )
    training_labels = np.concatenate(
        [np.asarray(task.labels, dtype=np.bool_) for task in training_tasks]
    )
    return ExperimentZeroResult(
        threshold=threshold,
        training_event_count=int(training_labels.sum()),
        training_label_count=len(training_labels),
        evaluation=evaluation,
    )


def render_report(
    result: ExperimentZeroResult,
    *,
    config_path: str | Path,
    training_seeds: Sequence[int],
    in_distribution_seeds: Sequence[int],
    heldout_family_seeds: Sequence[int],
    n_steps: int,
    horizon: int,
    threshold_quantile: float,
    return_lookback: int,
    false_alarm_rate: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    lightgbm_seed: int,
) -> str:
    """Render actual run outputs and their fixed configuration as Markdown."""

    lines = [
        "# Experiment Zero pipeline smoke test",
        "",
        "This is a pipeline smoke test on synthetic simulator output, not a research result and not evidence about real market behavior. The run used fixed, untuned development defaults solely to exercise generation, training-only target construction, public-only observer fitting, held-out-family evaluation, and episode-level uncertainty.",
        "",
        "## Configuration",
        "",
        f"- Population config: `{config_path}` (`train` = momentum + linear + MLP; `test` = mean reversion + GBDT)",
        f"- Episode length: {n_steps} steps; unwind horizon: {horizon}; return lookback: {return_lookback}",
        f"- Training seeds: `{_seed_text(training_seeds)}`",
        f"- In-distribution test seeds: `{_seed_text(in_distribution_seeds)}`",
        f"- Held-out-family test seeds: `{_seed_text(heldout_family_seeds)}`",
        f"- Event threshold: training-score quantile {threshold_quantile:.2f}, fitted value `{result.threshold:.10g}`; test episodes were excluded from fitting",
        f"- Metrics: PR-AUC, Brier score, and recall at false-alarm rate <= {false_alarm_rate:.2f}",
        f"- Uncertainty: {int((1.0 - 0.05) * 100)}% percentile CI from {bootstrap_resamples} whole-episode bootstrap resamples (seed {bootstrap_seed})",
        f"- Observer seeds/settings: LightGBM RNG seed {lightgbm_seed}; step-14 default observer hyperparameters",
        "",
        "## Event base rates",
        "",
        "| Split | Events / labels | Base rate |",
        "| --- | ---: | ---: |",
        f"| Training | {result.training_event_count} / {result.training_label_count} | {result.training_event_base_rate:.4f} |",
    ]
    display_names = {
        "in_distribution": "In-distribution test",
        "heldout_families": "Held-out-family test",
    }
    for name, evaluation in result.evaluation.items():
        lines.append(
            f"| {display_names.get(name, name)} | {evaluation.event_count} / {evaluation.label_count} | {evaluation.event_base_rate:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Baseline metrics",
            "",
            "Intervals in brackets are episode-bootstrap 95% CIs. Higher is better for PR-AUC and recall; lower is better for Brier score.",
        ]
    )
    for set_name, evaluation in result.evaluation.items():
        lines.extend(
            [
                "",
                f"### {display_names.get(set_name, set_name)}",
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
                + " | ".join(
                    _format_ci(metrics[name])
                    for name in ("pr_auc", "brier", "recall_at_false_alarm_rate")
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- All observations are generated by the uncalibrated HerdLab simulator; no real market data are used.",
            "- There are only four episodes in each test ecology, so the episode-bootstrap intervals are coarse and unstable; overlapping time windows are not treated as independent bootstrap units.",
            "- The held-out-family check changes agent families and population size but not market mechanics or exogenous-process parameters.",
            "- The event definition is relative to this run's training distribution. The roughly 5% training rate follows mechanically from the 95th-percentile threshold and is not an empirical prevalence estimate.",
            "- No configuration, threshold rule, simulator equation, or model hyperparameter was adjusted after observing these metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def _simulate(
    config: Mapping[str, object], seeds: Sequence[int]
) -> tuple[EpisodeSample, ...]:
    samples: list[EpisodeSample] = []
    for seed in seeds:
        public, private = run_episode(config, int(seed))
        samples.append(
            EpisodeSample(
                public=public,
                agent_net_flow=np.asarray(private.agent_net_flow, dtype=np.float64),
                seed=int(seed),
            )
        )
    return tuple(samples)


def _task_episodes(
    samples: Sequence[EpisodeSample], *, horizon: int, threshold: float
) -> tuple[TaskCEpisode, ...]:
    return tuple(
        TaskCEpisode(
            public=sample.public,
            labels=unwind_labels(
                sample.agent_net_flow,
                sample.public.liquidity_proxy,
                horizon=horizon,
                threshold=threshold,
            ),
            episode_id=str(sample.seed),
            horizon=horizon,
        )
        for sample in samples
    )


def _format_ci(interval: ConfidenceInterval) -> str:
    return f"{interval.estimate:.4f} [{interval.lower:.4f}, {interval.upper:.4f}]"


def _seed_text(seeds: Sequence[int]) -> str:
    values = tuple(int(seed) for seed in seeds)
    if len(values) > 1 and values == tuple(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ", ".join(str(seed) for seed in values)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fixed smoke experiment and write its Markdown report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)

    result = run_experiment_zero(args.config)
    familiar_config = load_population_config(args.config, "train")
    n_steps = int(familiar_config["simulation"]["n_steps"])
    report = render_report(
        result,
        config_path=args.config,
        training_seeds=TRAINING_SEEDS,
        in_distribution_seeds=IN_DISTRIBUTION_SEEDS,
        heldout_family_seeds=HELDOUT_FAMILY_SEEDS,
        n_steps=n_steps,
        horizon=10,
        threshold_quantile=0.95,
        return_lookback=10,
        false_alarm_rate=0.05,
        bootstrap_resamples=1000,
        bootstrap_seed=4000,
        lightgbm_seed=5000,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "EpisodeSample",
    "EvaluationSetResult",
    "ExperimentZeroResult",
    "evaluate_samples",
    "main",
    "render_report",
    "run_experiment_zero",
]
