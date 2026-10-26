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
from herdlab.experiments.generate import load_population_config, main
from herdlab.inference.observer_data import load_public_record
from herdlab.market.records import private_field_names


PROJECT_ROOT = Path(__file__).parents[1]
HELDOUT_CONFIG = PROJECT_ROOT / "configs" / "heldout_families.yaml"


def test_heldout_config_resolves_named_family_populations() -> None:
    train = build_population(
        load_population_config(HELDOUT_CONFIG, "train"), np.random.default_rng(1)
    )
    test = build_population(
        load_population_config(HELDOUT_CONFIG, "test"), np.random.default_rng(1)
    )

    assert len(train) == 18
    assert sum(isinstance(agent, MomentumAgent) for agent in train) == 6
    assert sum(isinstance(agent, LinearMLAgent) for agent in train) == 6
    assert sum(isinstance(agent, MLPAgent) for agent in train) == 6
    assert not any(isinstance(agent, (MeanReversionAgent, LightGBMAgent)) for agent in train)

    assert len(test) == 12
    assert sum(isinstance(agent, MeanReversionAgent) for agent in test) == 6
    assert sum(isinstance(agent, LightGBMAgent) for agent in test) == 6
    assert not any(isinstance(agent, (MomentumAgent, LinearMLAgent, MLPAgent)) for agent in test)


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
