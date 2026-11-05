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


# Captured from the unmodified round-4 implementation, seed 8341, full 500-step
# experiment_zero.yaml episodes (all five families, including rolling refits).
def test_existing_configs_match_frozen_round4_episodes() -> None:
    import hashlib
    import json
    from pathlib import Path

    expected = json.loads(
        (Path(__file__).parent / "fixtures/round4_episode_hashes.json").read_text()
    )
    for condition, hashes in expected.items():
        overlap, strength = condition.split("/")
        config = _config(n_steps=500)
        config["agents"]["learned_overlap_mode"] = overlap
        config["agents"]["private_signal"]["strength"] = float(strength)
        public, private = run_episode(config, seed=8341)
        for name, digest in hashes.items():
            record = public if hasattr(public, name) else private
            actual = hashlib.sha256(getattr(record, name).tobytes()).hexdigest()
            assert actual == digest, (condition, name)


def test_noise_only_is_unit_variance_and_uninformative() -> None:
    from dataclasses import replace

    config = _config(n_steps=100_000)
    exogenous = generate_exogenous_path(config, np.random.default_rng(41))
    for overlap_mode in ("none", "lag_weights"):
        config["agents"]["learned_overlap_mode"] = overlap_mode
        config["agents"]["private_signal"].update(mode="noise_only", strength=0.0)
        signals = generate_private_signals(
            config, exogenous, 3, np.random.default_rng(42)
        )
        np.testing.assert_allclose(signals.var(axis=0), 1.0, atol=0.02)
        np.testing.assert_allclose(signals.mean(axis=0), 0.0, atol=0.01)
        correlations = [
            np.corrcoef(signals[:, i], exogenous.returns)[0, 1] for i in range(3)
        ]
        np.testing.assert_allclose(correlations, 0.0, atol=0.01)
        # Noise ignores strength and future returns, even when supplied nonzero.
        config["agents"]["private_signal"]["strength"] = 0.9
        changed = generate_private_signals(
            config, replace(exogenous, returns=-2 * exogenous.returns),
            3, np.random.default_rng(42),
        )
        np.testing.assert_array_equal(signals, changed)


def test_modes_preserve_exogenous_path_and_off_bypasses_rng() -> None:
    from herdlab.agents.private_signal import private_signal_config
    from herdlab.experiments.run_episode import prepare_episode_with_private_signals
    import pytest

    config = _config(n_steps=30)
    reference, _, _ = prepare_episode_with_private_signals(config, 121)
    for mode in ("off", "signal", "noise_only"):
        config["agents"]["private_signal"].update(mode=mode, strength=0.25)
        path, _, signals = prepare_episode_with_private_signals(config, 121)
        for field in fields(path):
            np.testing.assert_array_equal(
                getattr(reference, field.name), getattr(path, field.name)
            )
        if mode == "off":
            np.testing.assert_array_equal(signals, 0.0)
            rng = np.random.default_rng(12)
            generate_private_signals(config, path, 3, rng)
            assert rng.normal() == np.random.default_rng(12).normal()
        else:
            assert np.any(signals != 0.0)
    config["agents"]["private_signal"]["mode"] = "invalid"
    with pytest.raises(ValueError, match="mode"):
        private_signal_config(config)
    config["agents"]["private_signal"].update(mode="signal", strength=0.0)
    assert not private_signal_config(config).enabled


def test_forecast_diagnostics_are_exact_and_private(tmp_path, monkeypatch) -> None:
    from dataclasses import asdict
    from herdlab.experiments.run_episode import (
        prepare_episode_with_private_signals, write_episode,
    )
    from herdlab.inference.observer_data import load_public_record
    import pytest

    config = _config(n_steps=60)
    config["agents"]["private_signal"].update(mode="noise_only", beta=0.25)
    path, agents, signals = prepare_episode_with_private_signals(config, 19)
    captured = [[] for agent in agents]
    for index, agent in enumerate(agents):
        original = agent.forecast

        def capture(observation, original=original, index=index):
            value = original(observation)
            captured[index].append(value)
            return value

        monkeypatch.setattr(agent, "forecast", capture)
    public, private = simulate_market_records(
        config, agents, path, private_signals=signals
    )
    np.testing.assert_array_equal(private.base_forecasts, np.asarray(captured).T)
    np.testing.assert_array_equal(
        private.signal_terms,
        0.25 * public.realized_volatility[:, None] * private.signals,
    )
    # Each model forecast is called once per step, including risk overrides.
    assert private.base_forecasts.shape == (60, len(agents))
    assert np.any(private.signal_terms != 0.0)
    names = {field.name for field in fields(public)}
    names.update(field.name for field in fields(PublicObservation))
    names.update(build_public_features(public).names)
    names.update(build_strategy_replication_features(public, config).names)
    assert not {"base_forecasts", "signal_terms", "signals"} & names
    monkeypatch.chdir(tmp_path)
    public_path, private_path = write_episode(public, private, "runs/diagnostics")
    load_public_record(public_path)
    with np.load(private_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["base_forecasts"], private.base_forecasts)
        np.testing.assert_array_equal(archive["signal_terms"], private.signal_terms)
    # Strict observer loading rejects a public archive contaminated by diagnostics.
    np.savez(tmp_path / "contaminated.npz", **asdict(public),
             base_forecasts=private.base_forecasts)
    with pytest.raises(ValueError):
        load_public_record(tmp_path / "contaminated.npz")


def test_explicit_off_matches_legacy_zero_strength() -> None:
    config = _config(n_steps=40)
    legacy = run_episode(config, 19)
    config["agents"]["private_signal"].update(mode="off", strength=0.25)
    explicit = run_episode(config, 19)
    for old, new in zip(legacy, explicit, strict=True):
        for field in fields(old):
            np.testing.assert_array_equal(
                getattr(old, field.name), getattr(new, field.name)
            )
    np.testing.assert_array_equal(explicit[1].signal_terms, 0.0)


def test_noise_only_requires_supplied_private_signals() -> None:
    import pytest

    config = _config(n_steps=10)
    config["agents"]["private_signal"].update(mode="noise_only", strength=0.0)
    path, agents = prepare_episode(config, 19)
    with pytest.raises(ValueError, match="private_signals are required"):
        simulate_market_records(config, agents, path)
