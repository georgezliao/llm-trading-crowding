"""Tests for pre-registered private exogenous-return signals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields

import numpy as np
import yaml

from herdlab.agents.base import Agent, PublicObservation
from herdlab.agents.factor import MeanReversionAgent, MomentumAgent
from herdlab.agents.learned import LightGBMAgent, LinearMLAgent, MLPAgent, build_population
from herdlab.agents.private_signal import generate_private_signals
from herdlab.experiments.run_episode import prepare_episode, run_episode
from herdlab.inference.features import (
    build_public_features,
    build_strategy_replication_features,
)
from herdlab.market.exogenous import ExogenousPath, generate_exogenous_path
from herdlab.market.records import PublicRecord
from herdlab.market.simulator import simulate_market_records


def _config(n_steps: int = 60) -> dict:
    with open("configs/experiment_zero.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["simulation"]["n_steps"] = n_steps
    return config


def test_zero_strength_matches_round3_episode_bit_identically() -> None:
    config = _config(n_steps=50)
    round3_config = deepcopy(config)
    del round3_config["agents"]["private_signal"]
    del round3_config["agents"]["learned_overlap_mode"]
    seed = 8341

    exogenous_seed, agent_seed = np.random.SeedSequence(seed).spawn(2)
    old_exogenous = generate_exogenous_path(
        round3_config, np.random.default_rng(exogenous_seed)
    )
    old_agents = build_population(round3_config, np.random.default_rng(agent_seed))
    old_public, old_private = simulate_market_records(
        round3_config, old_agents, old_exogenous
    )
    new_public, new_private = run_episode(config, seed)

    for field in fields(PublicRecord):
        np.testing.assert_array_equal(
            getattr(old_public, field.name), getattr(new_public, field.name)
        )
    for name in (
        "positions",
        "orders",
        "forced",
        "cash",
        "equity",
        "leverage",
        "transaction_costs",
        "agent_net_flow",
        "position_crowding",
        "flow_crowding",
    ):
        np.testing.assert_array_equal(
            getattr(old_private, name), getattr(new_private, name)
        )
    np.testing.assert_array_equal(new_private.signals, 0.0)


def test_signal_strength_does_not_change_exogenous_path() -> None:
    off = _config(n_steps=37)
    on = deepcopy(off)
    on["agents"]["private_signal"]["strength"] = 0.5

    off_path, _ = prepare_episode(off, seed=121)
    on_path, _ = prepare_episode(on, seed=121)

    for field in off_path.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(off_path, field), getattr(on_path, field)
        )


def test_empirical_signal_exogenous_correlation_matches_strength() -> None:
    n_steps = 100_000
    strength = 0.25
    config = _config(n_steps=n_steps)
    config["agents"]["private_signal"]["strength"] = strength
    config["exogenous"]["regime_volatility"] = [0.01, 0.01]
    z = np.random.default_rng(41).normal(size=n_steps)
    exogenous = ExogenousPath(
        returns=0.01 * z,
        regimes=np.zeros(n_steps, dtype=np.int64),
        jump_shocks=np.zeros(n_steps),
        liquidity=np.ones(n_steps),
        background_flow=np.zeros(n_steps),
    )

    signals = generate_private_signals(
        config, exogenous, n_agents=3, rng=np.random.default_rng(42)
    )

    correlations = [np.corrcoef(signals[:, index], z)[0, 1] for index in range(3)]
    np.testing.assert_allclose(correlations, strength, atol=0.01)


def test_all_agent_families_use_common_adjusted_decision_rule() -> None:
    for family in (
        MomentumAgent,
        MeanReversionAgent,
        LinearMLAgent,
        LightGBMAgent,
        MLPAgent,
    ):
        assert family.decide is Agent.decide

    class ZeroForecastAgent(Agent):
        def forecast(self, observation: PublicObservation) -> float:
            return 0.0

    agent = ZeroForecastAgent(q_max=10.0, k=2.0, u_max=10.0)
    observation = PublicObservation(
        step=3,
        price=100.0,
        returns=np.asarray([0.01, -0.02, 0.03]),
        realized_volatility=0.02,
        volume=1.0,
        order_flow_imbalance=0.0,
        liquidity_proxy=1000.0,
    )
    expected = agent.target_position(0.006, observation.realized_volatility)
    assert agent.decide(observation, forecast_adjustment=0.006) == expected


def test_signals_are_private_and_absent_from_observer_features() -> None:
    config = _config(n_steps=40)
    config["agents"]["private_signal"]["strength"] = 0.5
    public, private = run_episode(config, seed=19)
    public_names = {field.name for field in fields(public)}
    features = build_public_features(public)
    replication = build_strategy_replication_features(public, config)

    assert private.signals is not None
    assert np.any(private.signals != 0.0)
    assert not any("signal" in name for name in public_names)
    assert not any("signal" in name for name in features.names)
    assert not any("private_signal" in name for name in replication.names)
    assert not hasattr(public, "signals")
