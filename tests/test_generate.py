"""Tests for named-population batch episode generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from herdlab.agents.factor import MeanReversionAgent, MomentumAgent
from herdlab.agents.learned import (
    LightGBMAgent,
    LinearMLAgent,
    MLPAgent,
    build_population,
)
from herdlab.experiments.generate import (
    generate_episodes,
    load_population_config,
    main,
)
from herdlab.inference.observer_data import load_public_record
from herdlab.market.records import private_field_names


PROJECT_ROOT = Path(__file__).parents[1]
HELDOUT_CONFIG = PROJECT_ROOT / "configs" / "heldout_families.yaml"
HELDOUT_IMPACT_CONFIG = PROJECT_ROOT / "configs" / "heldout_impact.yaml"
HELDOUT_ARCHITECTURE_CONFIG = (
    PROJECT_ROOT / "configs" / "heldout_architecture.yaml"
)
EXPERIMENT_ZERO_CONFIG = PROJECT_ROOT / "configs" / "experiment_zero.yaml"


def _configured_aggregate_q_max(config: dict) -> float:
    agents = config["agents"]
    return int(agents["population_size"]) * float(agents["q_max"])


def test_heldout_config_resolves_named_family_populations() -> None:
    train_config = load_population_config(HELDOUT_CONFIG, "train")
    test_config = load_population_config(HELDOUT_CONFIG, "test")
    train = build_population(
        train_config, np.random.default_rng(1)
    )
    test = build_population(
        test_config, np.random.default_rng(1)
    )

    assert len(train) == len(test) == 30
    assert _configured_aggregate_q_max(train_config) == 300.0
    assert _configured_aggregate_q_max(test_config) == 300.0

    assert sum(isinstance(agent, MomentumAgent) for agent in train) == 10
    assert sum(isinstance(agent, LinearMLAgent) for agent in train) == 10
    assert sum(isinstance(agent, MLPAgent) for agent in train) == 10
    assert not any(isinstance(agent, (MeanReversionAgent, LightGBMAgent)) for agent in train)

    assert sum(isinstance(agent, MeanReversionAgent) for agent in test) == 15
    assert sum(isinstance(agent, LightGBMAgent) for agent in test) == 15
    assert not any(isinstance(agent, (MomentumAgent, LinearMLAgent, MLPAgent)) for agent in test)


def test_heldout_impact_changes_only_impact_function() -> None:
    train = load_population_config(HELDOUT_IMPACT_CONFIG, "train")
    test = load_population_config(HELDOUT_IMPACT_CONFIG, "test")

    assert train["agents"] == test["agents"]
    assert train["agents"]["population_size"] == 30
    assert _configured_aggregate_q_max(train) == _configured_aggregate_q_max(test)
    assert train["market"] == {
        "impact_model": "linear",
        "impact_sensitivity": 0.001,
    }
    assert test["market"] == {
        "impact_model": "concave",
        "impact_sensitivity": 0.001,
        "impact_exponent": 0.5,
    }


def test_heldout_architecture_config_matches_preregistered_populations() -> None:
    selected = yaml.safe_load(EXPERIMENT_ZERO_CONFIG.read_text(encoding="utf-8"))
    train = load_population_config(HELDOUT_ARCHITECTURE_CONFIG, "train")
    test = load_population_config(HELDOUT_ARCHITECTURE_CONFIG, "test")

    expected_counts = {
        "train": {
            "momentum": 6,
            "reversion": 6,
            "linear": 9,
            "gbdt": 0,
            "mlp": 9,
        },
        "test": {
            "momentum": 6,
            "reversion": 6,
            "linear": 0,
            "gbdt": 18,
            "mlp": 0,
        },
    }
    for name, config in (("train", train), ("test", test)):
        counts = {
            family: int(config["agents"][family]["count"])
            for family in expected_counts[name]
        }
        assert counts == expected_counts[name]
        assert sum(counts.values()) == config["agents"]["population_size"] == 30

        assert config["market"]["impact_sensitivity"] == selected["market"][
            "impact_sensitivity"
        ]
        assert config["agents"]["q_max"] == selected["agents"]["q_max"]
        assert config["agents"]["order_limit"] == selected["agents"]["order_limit"]
        assert config["agents"]["risk"]["max_leverage"] == selected["agents"][
            "risk"
        ]["max_leverage"]

    assert _configured_aggregate_q_max(train) == _configured_aggregate_q_max(test)


def test_cli_writes_correct_manifest_and_small_batch(tmp_path, monkeypatch) -> None:
    baseline = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8")
    )
    baseline["simulation"]["n_steps"] = 8
    baseline["agents"]["population_size"] = 2
    for family in ("linear", "gbdt", "mlp"):
        baseline["agents"][family]["count"] = 0
    baseline["agents"]["momentum"]["count"] = 1
    baseline["agents"]["reversion"]["count"] = 1

    base_path = tmp_path / "small_base.yaml"
    populations_path = tmp_path / "populations.yaml"
    base_path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
    populations_path.write_text(
        yaml.safe_dump(
            {
                "base_config": base_path.name,
                "populations": {"tiny": {"agents": {"feature_overlap": 0.25}}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    main(
        [
            "--config",
            str(populations_path),
            "--population",
            "tiny",
            "--n-episodes",
            "2",
            "--seed-start",
            "41",
            "--out",
            "runs/batch",
            "--workers",
            "2",
        ]
    )

    manifest_path = tmp_path / "runs" / "batch" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "format_version": 1,
        "population": "tiny",
        "n_episodes": 2,
        "seed_start": 41,
        "seed_end": 42,
        "source_config": str(populations_path.resolve()),
        "episodes": [
            {
                "episode": 0,
                "seed": 41,
                "public": "episode_0000_seed_41/public.npz",
                "private": "episode_0000_seed_41/private.npz",
            },
            {
                "episode": 1,
                "seed": 42,
                "public": "episode_0001_seed_42/public.npz",
                "private": "episode_0001_seed_42/private.npz",
            },
        ],
    }

    for entry in manifest["episodes"]:
        load_public_record(manifest_path.parent / entry["public"])
        with np.load(manifest_path.parent / entry["private"], allow_pickle=False) as archive:
            assert set(archive.files) == set(private_field_names())


def test_parallel_generation_matches_serial_for_same_three_seeds(
    tmp_path, monkeypatch
) -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8")
    )
    config["simulation"]["n_steps"] = 8
    config["agents"]["population_size"] = 2
    for family in ("linear", "gbdt", "mlp"):
        config["agents"][family]["count"] = 0
    config["agents"]["momentum"]["count"] = 1
    config["agents"]["reversion"]["count"] = 1

    monkeypatch.chdir(tmp_path)
    serial_manifest_path = generate_episodes(
        config,
        population="tiny",
        n_episodes=3,
        seed_start=71,
        out="runs/serial",
        workers=1,
    )
    parallel_manifest_path = generate_episodes(
        config,
        population="tiny",
        n_episodes=3,
        seed_start=71,
        out="runs/parallel",
        workers=2,
    )

    serial_manifest = json.loads(serial_manifest_path.read_text(encoding="utf-8"))
    parallel_manifest = json.loads(parallel_manifest_path.read_text(encoding="utf-8"))
    assert serial_manifest == parallel_manifest

    for entry in serial_manifest["episodes"]:
        for record_name in ("public", "private"):
            with (
                np.load(
                    serial_manifest_path.parent / entry[record_name],
                    allow_pickle=False,
                ) as serial,
                np.load(
                    parallel_manifest_path.parent / entry[record_name],
                    allow_pickle=False,
                ) as parallel,
            ):
                assert serial.files == parallel.files
                for field in serial.files:
                    np.testing.assert_array_equal(serial[field], parallel[field])
