"""Tests for the fixed Experiment Zero v2 pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from herdlab.experiments.experiment_zero import (
    EpisodeSample,
    evaluate_samples,
    load_v2_configs,
)
from herdlab.inference.targets import fit_unwind_threshold
from herdlab.market.records import PublicRecord


def _sample(seed: int, *, flow_scale: float = 1.0) -> EpisodeSample:
    n_steps = 36
    phase = np.arange(n_steps, dtype=np.float64) + seed * 0.37
    returns = 0.002 * np.sin(phase / 3.0)
    liquidity = 90.0 + 10.0 * np.cos(phase / 5.0)
    flow = flow_scale * (3.0 * np.sin(phase / 2.0) - 1.0)
    public = PublicRecord(
        prices=100.0 * np.exp(np.r_[0.0, np.cumsum(returns)]),
        returns=returns,
        volume=20.0 + np.abs(flow),
        executed_imbalance=flow + 2.0 * np.cos(phase),
        realized_volatility=np.linspace(0.0, 0.01, n_steps),
        liquidity_proxy=liquidity,
    )
    forced = np.zeros(n_steps, dtype=np.bool_)
    forced[10 + seed % 5] = True
    return EpisodeSample(
        public=public,
        agent_net_flow=flow,
        forced_flow_steps=forced,
        seed=seed,
    )


def test_selected_parameters_apply_to_every_matched_split() -> None:
    configs = load_v2_configs()

    assert set(configs) == {
        "training",
        "in_distribution",
        "heldout_families",
        "heldout_impact",
    }
    for config in configs.values():
        assert config["market"]["impact_sensitivity"] == pytest.approx(0.03)
        assert config["agents"]["q_max"] == pytest.approx(200.0)
        assert config["agents"]["order_limit"] == pytest.approx(40.0)
        assert config["agents"]["population_size"] == 30
    assert configs["training"]["agents"]["momentum"]["count"] == 10
    assert configs["heldout_families"]["agents"]["reversion"]["count"] == 15
    assert configs["heldout_impact"]["agents"] == configs["training"]["agents"]
    assert configs["training"]["market"]["impact_model"] == "linear"
    assert configs["heldout_impact"]["market"]["impact_model"] == "concave"


def test_evaluation_uses_training_only_threshold_and_reports_both_ecologies() -> None:
    training = tuple(_sample(seed) for seed in range(4))
    in_distribution = tuple(_sample(seed) for seed in range(10, 13))
    # Deliberately extreme held-out flow would move the threshold if it leaked.
    heldout = tuple(_sample(seed, flow_scale=100.0) for seed in range(20, 23))
    expected_threshold = fit_unwind_threshold(
        ((item.agent_net_flow, item.public.liquidity_proxy) for item in training),
        horizon=3,
        quantile=0.75,
    )

    result = evaluate_samples(
        training,
        {"in_distribution": in_distribution, "heldout_families": heldout},
        horizon=3,
        threshold_quantile=0.75,
        return_lookback=3,
        false_alarm_rate=0.10,
        bootstrap_resamples=20,
        bootstrap_seed=7,
        lightgbm_seed=8,
        gru_seed=9,
        gru_kwargs={
            "sequence_length": 4,
            "hidden_size": 4,
            "batch_size": 32,
            "max_epochs": 2,
            "patience": 1,
            "validation_fraction": 0.25,
        },
    )

    assert result.threshold == expected_threshold
    assert set(result.labels) == {"default", "onset"}
    for label_result in result.labels.values():
        assert set(label_result.evaluation) == {
            "in_distribution",
            "heldout_families",
        }
        assert 0.0 <= label_result.training.fraction_events_with_forced_flow <= 1.0
        for evaluation in label_result.evaluation.values():
            assert 0.0 < evaluation.event_base_rate < 1.0
            assert set(evaluation.metrics) == {
                "trivial_reference",
                "logistic_regression",
                "lightgbm",
                "gru",
            }
            assert 0.0 <= evaluation.events.fraction_events_with_forced_flow <= 1.0
            for metrics in evaluation.metrics.values():
                assert set(metrics) == {
                    "pr_auc",
                    "brier",
                    "recall_at_false_alarm_rate",
                }
                for interval in metrics.values():
                    assert (
                        0.0
                        <= interval.lower
                        <= interval.estimate
                        <= interval.upper
                        <= 1.0
                    )
