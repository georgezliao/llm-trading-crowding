"""Registered population composition, capital matching, and seeded execution."""
from copy import deepcopy
from dataclasses import fields

import numpy as np
import pytest

from herdlab.experiments.round5_reconcile import population_configs as historical_configs
from herdlab.experiments.round6_populations import FAMILIES, population_configs
from herdlab.experiments.run_episode import prepare_episode, run_episode

POPULATIONS = ("baseline", *(f"without_{f}" for f in FAMILIES), "round3_v2")


def test_resolved_populations_match_registration_and_history():
    configs = population_configs()
    historical = historical_configs()
    assert tuple(configs) == POPULATIONS
    assert configs["baseline"] == historical["experiment_zero"]
    assert configs["round3_v2"] == historical["round3_v2"]
    baseline = configs["baseline"]["agents"]
    for name, config in configs.items():
        agents = config["agents"]
        omitted = name.removeprefix("without_") if name.startswith("without_") else None
        expected = ([10, 0, 10, 0, 10] if name == "round3_v2" else
                    [0 if family == omitted else 6 for family in FAMILIES])
        assert [agents[f]["count"] for f in FAMILIES] == expected
        assert agents["population_size"] == sum(expected)
        for field in ("q_max", "order_limit", "initial_cash"):
            assert agents[field] == baseline[field] * (30 / 24 if omitted else 1)
            assert sum(expected) * agents[field] == pytest.approx(30 * baseline[field])
        assert agents["risk"] == baseline["risk"]
        assert agents["risk_aversion_k"] == baseline["risk_aversion_k"]
        price = config["simulation"]["initial_price"]
        assert agents["q_max"] * price / agents["initial_cash"] == pytest.approx(
            baseline["q_max"] * price / baseline["initial_cash"])
        # Undo only the registered edits: no other model settings may change.
        restored = deepcopy(config)
        for family in FAMILIES:
            restored["agents"][family]["count"] = 6
        for field in ("population_size", "q_max", "order_limit", "initial_cash"):
            restored["agents"][field] = baseline[field]
        assert restored == configs["baseline"]


def test_configs_are_independent_across_conditions_and_calls():
    configs = population_configs()
    original = deepcopy(configs)
    configs["without_linear"]["agents"]["risk"]["max_leverage"] = 99
    configs["baseline"]["agents"]["mlp"]["count"] = 99
    for name in POPULATIONS:
        if name not in ("without_linear", "baseline"):
            assert configs[name] == original[name]
    assert population_configs() == original


@pytest.mark.parametrize("name", POPULATIONS)
def test_seeded_populations_and_episodes(name):
    config = population_configs()[name]
    config["simulation"]["n_steps"] = 45
    exogenous, agents = prepare_episode(config, 8000)
    expected_types = ("MomentumAgent", "MeanReversionAgent", "LinearMLAgent",
                      "LightGBMAgent", "MLPAgent")
    assert [type(agent).__name__ for agent in agents] == [
        kind for family, kind in zip(FAMILIES, expected_types)
        for _ in range(config["agents"][family]["count"])
    ]
    baseline = population_configs()["baseline"]
    baseline["simulation"]["n_steps"] = 45
    baseline_exogenous, _ = prepare_episode(baseline, 8000)
    for field in fields(exogenous):
        np.testing.assert_array_equal(getattr(exogenous, field.name),
                                      getattr(baseline_exogenous, field.name))
    first = run_episode(config, 8000)
    repeated = run_episode(config, 8000)
    for record, repeat in zip(first, repeated):
        for field in fields(record):
            np.testing.assert_array_equal(getattr(record, field.name),
                                          getattr(repeat, field.name))
    assert first[1].positions.shape[1] == config["agents"]["population_size"]
    np.testing.assert_array_equal(first[1].cash[0], config["agents"]["initial_cash"])
    different = run_episode(config, 8001)
    assert not np.array_equal(first[0].returns, different[0].returns)
