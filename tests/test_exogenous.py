"""Tests for synthetic exogenous market drivers."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import yaml

from herdlab.market.exogenous import generate_exogenous_path


def _baseline_config() -> dict:
    with open("configs/baseline.yaml", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_exogenous_path_is_deterministic_by_seed() -> None:
    config = _baseline_config()

    first = generate_exogenous_path(config, np.random.default_rng(817))
    second = generate_exogenous_path(config, np.random.default_rng(817))
    different = generate_exogenous_path(config, np.random.default_rng(818))

    for field in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert not np.array_equal(first.returns, different.returns)


def test_exogenous_shapes_and_positive_liquidity() -> None:
    config = _baseline_config()
    config["simulation"]["n_steps"] = 73

    path = generate_exogenous_path(config, np.random.default_rng(9))

    for field in path.__dataclass_fields__:
        assert getattr(path, field).shape == (73,)
    assert np.all(path.liquidity > 0.0)
    assert np.all(path.liquidity >= config["exogenous"]["liquidity"]["minimum"])


def test_empirical_regime_transitions_match_configured_probabilities() -> None:
    config = deepcopy(_baseline_config())
    config["simulation"]["n_steps"] = 100_000
    configured = np.asarray(config["exogenous"]["transition_matrix"])

    regimes = generate_exogenous_path(config, np.random.default_rng(42)).regimes

    empirical = np.empty((2, 2))
    for previous in (0, 1):
        next_states = regimes[1:][regimes[:-1] == previous]
        empirical[previous] = np.bincount(next_states, minlength=2) / len(next_states)
    np.testing.assert_allclose(empirical, configured, atol=0.01)
