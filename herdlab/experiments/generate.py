"""Generate a reproducible collection of separated simulation episodes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from herdlab.experiments.run_episode import load_config, run_episode, write_episode


def load_population_config(
    path: str | Path, population: str
) -> dict[str, Any]:
    """Resolve one named population by merging it over its base config.

    Population files contain a ``base_config`` path and a ``populations``
    mapping.  Paths are resolved relative to the population file, so checked-in
    experiment definitions do not depend on the caller's working directory.
    """

    population_path = Path(path).resolve()
    specification = load_config(population_path)
    base_value = specification.get("base_config")
    populations = specification.get("populations")
    if not isinstance(base_value, str) or not base_value:
        raise ValueError("population config must define a nonempty base_config")
    if not isinstance(populations, Mapping):
        raise ValueError("population config must define a populations mapping")
    if population not in populations:
        available = ", ".join(sorted(str(name) for name in populations))
        raise ValueError(
            f"unknown population {population!r}; available populations: {available}"
        )
    override = populations[population]
    if not isinstance(override, Mapping):
        raise ValueError(f"population {population!r} must be a mapping")

    base_path = (population_path.parent / base_value).resolve()
    base = load_config(base_path)
    return _deep_merge(base, override)


def generate_episodes(
    config: Mapping[str, Any],
    *,
    population: str,
    n_episodes: int,
    seed_start: int,
    out: str | Path,
    source_config: str | Path | None = None,
) -> Path:
    """Generate consecutive seeded episodes and write a JSON manifest.

    Each episode has its own directory and separate ``public.npz`` and
    ``private.npz`` archives.  Paths recorded in the manifest are relative to
    the manifest directory, allowing an entire run directory to be moved.
    """

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if seed_start < 0:
        raise ValueError("seed_start must be nonnegative")
    if not population:
        raise ValueError("population must be nonempty")

    runs_root = (Path.cwd() / "runs").resolve()
    output_dir = Path(out).resolve()
    if not output_dir.is_relative_to(runs_root):
        raise ValueError("--out must resolve to a directory under runs/")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for index, seed in enumerate(range(seed_start, seed_start + n_episodes)):
        episode_dir = output_dir / f"episode_{index:04d}_seed_{seed}"
        public, private = run_episode(config, seed)
        public_path, private_path = write_episode(public, private, episode_dir)
        entries.append(
            {
                "episode": index,
                "seed": seed,
                "public": str(public_path.relative_to(output_dir)),
                "private": str(private_path.relative_to(output_dir)),
            }
        )

    manifest: dict[str, Any] = {
        "format_version": 1,
        "population": population,
        "n_episodes": n_episodes,
        "seed_start": seed_start,
        "seed_end": seed_start + n_episodes - 1,
        "episodes": entries,
    }
    if source_config is not None:
        manifest["source_config"] = str(Path(source_config).resolve())
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge mapping values without mutating either input."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = deepcopy(value)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point for batch episode generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="named-population YAML configuration"
    )
    parser.add_argument("--population", required=True, help="population name")
    parser.add_argument(
        "--n-episodes", required=True, type=int, help="number of episodes"
    )
    parser.add_argument(
        "--seed-start", required=True, type=int, help="first consecutive RNG seed"
    )
    parser.add_argument(
        "--out", required=True, help="output directory beneath runs/"
    )
    args = parser.parse_args(argv)

    config = load_population_config(args.config, args.population)
    generate_episodes(
        config,
        population=args.population,
        n_episodes=args.n_episodes,
        seed_start=args.seed_start,
        out=args.out,
        source_config=args.config,
    )


if __name__ == "__main__":
    main()


__all__ = ["generate_episodes", "load_population_config", "main"]
