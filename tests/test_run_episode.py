"""End-to-end tests for the single-episode experiment runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from herdlab.agents.factor import build_factor_population
from herdlab.experiments.run_episode import main, prepare_episode
from herdlab.inference.observer_data import load_public_record
from herdlab.market.exogenous import ExogenousPath, generate_exogenous_path
from herdlab.market.records import private_field_names, public_field_names
from herdlab.market.simulator import simulate_market_records


BASELINE_CONFIG = Path(__file__).parents[1] / "configs" / "baseline.yaml"


def _config(n_steps: int = 40) -> dict:
    with BASELINE_CONFIG.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["simulation"]["n_steps"] = n_steps
    return config


def _run_cli(tmp_path: Path, monkeypatch, *, seed: int, name: str) -> Path:
    monkeypatch.chdir(tmp_path)
    output_dir = Path("runs") / name
    main(
        [
            "--config",
            str(BASELINE_CONFIG),
            "--seed",
            str(seed),
            "--out",
            str(output_dir),
        ]
    )
    return tmp_path / output_dir


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def test_cli_writes_separate_deterministic_archives(tmp_path, monkeypatch) -> None:
    first = _run_cli(tmp_path, monkeypatch, seed=19, name="first")
    second = _run_cli(tmp_path, monkeypatch, seed=19, name="second")
    different = _run_cli(tmp_path, monkeypatch, seed=20, name="different")

    first_public = _archive(first / "public.npz")
    first_private = _archive(first / "private.npz")
    second_public = _archive(second / "public.npz")
    second_private = _archive(second / "private.npz")
    different_public = _archive(different / "public.npz")
    different_private = _archive(different / "private.npz")

    assert set(first_public) == set(public_field_names())
    assert set(first_private) == set(private_field_names())
    assert set(first_public).isdisjoint(first_private)
    for name in first_public:
        np.testing.assert_array_equal(first_public[name], second_public[name])
    for name in first_private:
        np.testing.assert_array_equal(first_private[name], second_private[name])
    assert any(
        not np.array_equal(first_public[name], different_public[name])
        for name in first_public
    )
    assert any(
        not np.array_equal(first_private[name], different_private[name])
        for name in first_private
    )
    load_public_record(first / "public.npz")


def test_different_populations_share_identical_exogenous_path() -> None:
    first_config = _config(n_steps=31)
    first_config["agents"]["population_size"] = 2
    first_config["agents"]["momentum"]["count"] = 1
    first_config["agents"]["reversion"]["count"] = 1
    for family in ("linear", "gbdt", "mlp"):
        first_config["agents"][family]["count"] = 0

    second_config = deepcopy(first_config)
    second_config["agents"]["population_size"] = 5
    second_config["agents"]["momentum"]["count"] = 0
    second_config["agents"]["reversion"]["count"] = 2
    second_config["agents"]["linear"]["count"] = 1
    second_config["agents"]["gbdt"]["count"] = 1
    second_config["agents"]["mlp"]["count"] = 1
    second_config["agents"]["parameter_dispersion"] = 0.35
    second_config["agents"]["feature_overlap"] = 0.9

    first_path, first_population = prepare_episode(first_config, seed=9127)
    second_path, second_population = prepare_episode(second_config, seed=9127)

    assert len(first_population) == 2
    assert len(second_population) == 5
    for field in first_path.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(first_path, field), getattr(second_path, field)
        )


def test_future_return_perturbation_does_not_change_orders_through_t() -> None:
    config = _config()
    seed = 73
    cutoff = 18

    original_rng = np.random.default_rng(seed)
    original_path = generate_exogenous_path(config, original_rng)
    original_agents = build_factor_population(config, original_rng)
    _, original_private = simulate_market_records(
        config, original_agents, original_path
    )

    changed_returns = original_path.returns.copy()
    changed_returns[cutoff + 1 :] += np.linspace(
        0.01, 0.03, len(changed_returns) - cutoff - 1
    )
    changed_path: ExogenousPath = replace(original_path, returns=changed_returns)

    changed_rng = np.random.default_rng(seed)
    generate_exogenous_path(config, changed_rng)  # reproduce RNG state for agents
    changed_agents = build_factor_population(config, changed_rng)
    _, changed_private = simulate_market_records(config, changed_agents, changed_path)

    np.testing.assert_array_equal(
        original_private.orders[: cutoff + 1],
        changed_private.orders[: cutoff + 1],
    )
    assert not np.array_equal(
        original_private.orders[cutoff + 2 :],
        changed_private.orders[cutoff + 2 :],
    )
