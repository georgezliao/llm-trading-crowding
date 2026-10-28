"""Tests for learned lag overlap and overlap-split private-signal noise."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields

import numpy as np
import yaml

from herdlab.agents.learned import (
    LightGBMAgent,
    LinearMLAgent,
    MLPAgent,
    build_population,
)
from herdlab.agents.private_signal import generate_private_signals
from herdlab.experiments.run_episode import run_episode
from herdlab.market.exogenous import ExogenousPath


def _config(n_steps: int = 80) -> dict:
    with open("configs/experiment_zero.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["simulation"]["n_steps"] = n_steps
    config["agents"]["learned_overlap_mode"] = "lag_weights"
    config["agents"]["private_signal"]["strength"] = 0.25
    return config


def _zero_exogenous(n_steps: int) -> ExogenousPath:
    return ExogenousPath(
        returns=np.zeros(n_steps, dtype=np.float64),
        regimes=np.zeros(n_steps, dtype=np.int64),
        jump_shocks=np.zeros(n_steps, dtype=np.float64),
        liquidity=np.ones(n_steps, dtype=np.float64),
        background_flow=np.zeros(n_steps, dtype=np.float64),
    )


def _learned_agents(config: dict, seed: int = 17) -> list:
    population = build_population(config, np.random.default_rng(seed))
    return [
        agent
        for agent in population
        if isinstance(agent, (LinearMLAgent, LightGBMAgent, MLPAgent))
    ]


def test_full_overlap_gives_identical_learned_weights_and_signal_noise() -> None:
    config = _config(n_steps=2_000)
    config["agents"]["feature_overlap"] = 1.0

    learned = _learned_agents(config)
    assert learned
    assert learned[0].lag_weights is not None
    for agent in learned[1:]:
        np.testing.assert_array_equal(agent.lag_weights, learned[0].lag_weights)

    signals = generate_private_signals(
        config,
        _zero_exogenous(2_000),
        n_agents=len(learned),
        rng=np.random.default_rng(91),
    )
    for column in range(1, signals.shape[1]):
        np.testing.assert_array_equal(signals[:, column], signals[:, 0])


def test_zero_overlap_gives_independent_signal_noise() -> None:
    n_steps = 100_000
    config = _config(n_steps=n_steps)
    config["agents"]["feature_overlap"] = 0.0

    signals = generate_private_signals(
        config,
        _zero_exogenous(n_steps),
        n_agents=4,
        rng=np.random.default_rng(92),
    )
    correlations = np.corrcoef(signals, rowvar=False)
    off_diagonal = correlations[np.triu_indices(4, k=1)]
    assert np.max(np.abs(off_diagonal)) < 0.02


def test_lag_weights_mode_is_deterministic_per_seed() -> None:
    config = _config(n_steps=55)
    config["agents"]["feature_overlap"] = 0.5

    first_agents = _learned_agents(config, seed=123)
    second_agents = _learned_agents(config, seed=123)
    for left, right in zip(first_agents, second_agents, strict=True):
        np.testing.assert_array_equal(left.lag_weights, right.lag_weights)

    first_public, first_private = run_episode(deepcopy(config), seed=456)
    second_public, second_private = run_episode(deepcopy(config), seed=456)
    for field in fields(first_public):
        np.testing.assert_array_equal(
            getattr(first_public, field.name), getattr(second_public, field.name)
        )
    for field in fields(first_private):
        np.testing.assert_array_equal(
            getattr(first_private, field.name), getattr(second_private, field.name)
        )
