"""Configuration helpers for the pre-registered learned-agent overlap mode."""

from __future__ import annotations

from typing import Any, Literal, Mapping, cast


LearnedOverlapMode = Literal["none", "lag_weights"]


def learned_overlap_mode(config: Mapping[str, Any]) -> LearnedOverlapMode:
    """Return the validated overlap mode, defaulting old configurations to off."""

    agents = config["agents"] if "agents" in config else config
    mode = str(agents.get("learned_overlap_mode", "none"))
    if mode not in {"none", "lag_weights"}:
        raise ValueError(
            "agents.learned_overlap_mode must be 'none' or 'lag_weights'"
        )
    return cast(LearnedOverlapMode, mode)


__all__ = ["LearnedOverlapMode", "learned_overlap_mode"]
