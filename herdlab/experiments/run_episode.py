"""Run and serialize one reproducible synthetic HerdLab episode.

The public and private records are deliberately written to different archives.
The public archive contains exactly the schema accepted by the observer-data
loader; private ground truth never shares a file with observer inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from herdlab.agents.base import Agent
from herdlab.agents.learned import build_population
from herdlab.agents.private_signal import (
    generate_private_signals,
    private_signal_config,
)
from herdlab.market.exogenous import ExogenousPath, generate_exogenous_path
from herdlab.market.records import PrivateRecord, PublicRecord
from herdlab.market.simulator import simulate_market_records


def prepare_episode(
    config: Mapping[str, Any], seed: int
) -> tuple[ExogenousPath, list[Agent]]:
    """Build deterministic episode inputs from independent random streams.

    ``seed`` identifies the whole episode.  NumPy ``SeedSequence`` child
    streams isolate exogenous market draws from all population construction
    draws, so changing agent counts, families, or parameters cannot advance
    or otherwise alter the exogenous stream.  Reusing the episode seed across
    population configurations therefore provides common random numbers.
    """

    signal_settings = private_signal_config(config)
    episode_seeds = np.random.SeedSequence(seed).spawn(
        signal_settings.rng_stream + 1
    )
    exogenous_seed, agent_seed = episode_seeds[:2]
    exogenous_rng = np.random.default_rng(exogenous_seed)
    agent_rng = np.random.default_rng(agent_seed)
    exogenous = generate_exogenous_path(config, exogenous_rng)
    agents = build_population(config, agent_rng)
    return exogenous, agents


def prepare_episode_with_private_signals(
    config: Mapping[str, Any], seed: int
) -> tuple[ExogenousPath, list[Agent], np.ndarray]:
    """Build episode inputs, including agent-only signals on their own stream."""

    signal_settings = private_signal_config(config)
    episode_seeds = np.random.SeedSequence(seed).spawn(
        signal_settings.rng_stream + 1
    )
    exogenous_rng = np.random.default_rng(episode_seeds[0])
    agent_rng = np.random.default_rng(episode_seeds[1])
    exogenous = generate_exogenous_path(config, exogenous_rng)
    agents = build_population(config, agent_rng)
    signal_rng = np.random.default_rng(episode_seeds[signal_settings.rng_stream])
    signals = generate_private_signals(config, exogenous, len(agents), signal_rng)
    return exogenous, agents, signals


def run_episode(
    config: Mapping[str, Any], seed: int
) -> tuple[PublicRecord, PrivateRecord]:
    """Simulate one episode from deterministic exogenous and agent streams."""

    exogenous, agents, signals = prepare_episode_with_private_signals(config, seed)
    return simulate_market_records(
        config, agents, exogenous, private_signals=signals
    )


def write_episode(
    public: PublicRecord, private: PrivateRecord, out: str | Path
) -> tuple[Path, Path]:
    """Write separated records beneath the local ``runs/`` directory."""

    runs_root = (Path.cwd() / "runs").resolve()
    output_dir = Path(out).resolve()
    if not output_dir.is_relative_to(runs_root):
        raise ValueError("--out must resolve to a directory under runs/")

    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "public.npz"
    private_path = output_dir / "private.npz"
    np.savez_compressed(public_path, **asdict(public))
    np.savez_compressed(private_path, **asdict(private))
    return public_path, private_path


def load_config(path: str | Path) -> Mapping[str, Any]:
    """Load one YAML experiment configuration."""

    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, Mapping):
        raise ValueError("configuration must be a YAML mapping")
    return config


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point for a single episode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--seed", required=True, type=int, help="episode RNG seed")
    parser.add_argument(
        "--out", required=True, help="output directory beneath runs/"
    )
    args = parser.parse_args(argv)

    public, private = run_episode(load_config(args.config), args.seed)
    write_episode(public, private, args.out)


if __name__ == "__main__":
    main()


__all__ = [
    "load_config",
    "main",
    "prepare_episode",
    "prepare_episode_with_private_signals",
    "run_episode",
    "write_episode",
]
