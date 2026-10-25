"""Tests for factor forecasts and heterogeneous population construction."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import yaml

from herdlab.agents.base import PublicObservation
from herdlab.agents.factor import (
    MeanReversionAgent,
    MomentumAgent,
    build_factor_population,
)


def _observation(returns: list[float]) -> PublicObservation:
    return PublicObservation(
        step=len(returns) - 1,
        price=100.0,
        returns=np.asarray(returns, dtype=np.float64),
        realized_volatility=0.01,
        volume=100.0,
        order_flow_imbalance=0.0,
        liquidity_proxy=1000.0,
    )


def _agent(
    agent_type: type[MomentumAgent] | type[MeanReversionAgent],
) -> MomentumAgent | MeanReversionAgent:
    return agent_type(lookback=3, q_max=10.0, k=2.0, u_max=10.0)


def _baseline_config() -> dict:
    with open("configs/baseline.yaml", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_momentum_follows_a_trend_and_reversion_opposes_it() -> None:
    upward_trend = _observation([-0.02, 0.01, 0.02, 0.03])
    downward_trend = _observation([0.02, -0.01, -0.02, -0.03])

    momentum = _agent(MomentumAgent)
    reversion = _agent(MeanReversionAgent)

    assert momentum.forecast(upward_trend) > 0.0
    assert momentum.forecast(downward_trend) < 0.0
    assert reversion.forecast(upward_trend) < 0.0
    assert reversion.forecast(downward_trend) > 0.0


def test_zero_dispersion_gives_identical_agents_within_each_family() -> None:
    config = deepcopy(_baseline_config())
    config["agents"]["population_size"] = 8
    config["agents"]["momentum_fraction"] = 0.5
    config["agents"]["parameter_dispersion"] = 0.0

    population = build_factor_population(config, np.random.default_rng(11))

    for family in (population[:4], population[4:]):
        first = family[0]
        for agent in family[1:]:
            assert type(agent) is type(first)
            assert agent.lookback == first.lookback
            assert agent.q_max == first.q_max
            assert agent.k == first.k
            assert agent.u_max == first.u_max
            np.testing.assert_array_equal(agent.lag_weights, first.lag_weights)


def test_population_builder_is_deterministic_and_dispersion_varies_parameters() -> None:
    config = _baseline_config()
    first = build_factor_population(config, np.random.default_rng(42))
    second = build_factor_population(config, np.random.default_rng(42))

    assert len(first) == config["agents"]["population_size"]
    assert sum(isinstance(agent, MeanReversionAgent) for agent in first) == 15
    for left, right in zip(first, second, strict=True):
        assert type(left) is type(right)
        assert (left.lookback, left.q_max, left.k, left.u_max) == (
            right.lookback,
            right.q_max,
            right.k,
            right.u_max,
        )
        np.testing.assert_array_equal(left.lag_weights, right.lag_weights)
    assert len({agent.q_max for agent in first}) > 1


def test_feature_overlap_controls_shared_lag_weights() -> None:
    config = deepcopy(_baseline_config())
    config["agents"]["population_size"] = 4
    config["agents"]["momentum_fraction"] = 1.0
    config["agents"]["parameter_dispersion"] = 0.001

    config["agents"]["feature_overlap"] = 1.0
    shared = build_factor_population(config, np.random.default_rng(7))
    assert all(agent.lookback == shared[0].lookback for agent in shared)
    for agent in shared[1:]:
        np.testing.assert_array_equal(agent.lag_weights, shared[0].lag_weights)

    config["agents"]["feature_overlap"] = 0.0
    individual = build_factor_population(config, np.random.default_rng(7))
    assert any(
        not np.array_equal(agent.lag_weights, individual[0].lag_weights)
        for agent in individual[1:]
    )
