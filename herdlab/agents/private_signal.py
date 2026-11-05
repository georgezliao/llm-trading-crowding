"""Private one-step exogenous-return signals for trading agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from numpy.typing import NDArray

from herdlab.agents.overlap import learned_overlap_mode

if TYPE_CHECKING:
    from herdlab.market.exogenous import ExogenousPath


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PrivateSignalConfig:
    """Validated private-signal parameters and episode RNG stream index."""

    strength: float
    beta: float
    rng_stream: int
    mode: str = "signal"

    @property
    def enabled(self) -> bool:
        return self.mode == "noise_only" or (
            self.mode == "signal" and self.strength > 0.0
        )


def private_signal_config(config: Mapping[str, Any]) -> PrivateSignalConfig:
    """Return private-signal settings, defaulting missing round-3 config to off."""

    agents = config["agents"] if "agents" in config else config
    raw = agents.get("private_signal", {})
    if not isinstance(raw, Mapping):
        raise TypeError("agents.private_signal must be a mapping")

    strength = float(raw.get("strength", 0.0))
    mode = raw.get("mode", "signal")
    if mode not in ("off", "signal", "noise_only"):
        raise ValueError("agents.private_signal.mode must be off, signal, or noise_only")
    beta = float(raw.get("beta", 1.0))
    stream_value = raw.get("rng_stream", 2)
    if isinstance(stream_value, bool) or not isinstance(
        stream_value, (int, np.integer)
    ):
        raise TypeError("agents.private_signal.rng_stream must be an integer")
    rng_stream = int(stream_value)

    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("agents.private_signal.strength must be finite and in [0, 1]")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("agents.private_signal.beta must be finite and nonnegative")
    if rng_stream < 2:
        raise ValueError(
            "agents.private_signal.rng_stream must be at least 2 to remain "
            "separate from exogenous and population streams"
        )
    return PrivateSignalConfig(strength=strength, beta=beta, rng_stream=rng_stream, mode=mode)


def generate_private_signals(
    config: Mapping[str, Any],
    exogenous: ExogenousPath,
    n_agents: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate ``s[i,t]`` from the next exogenous return and private noise.

    Array row ``t`` is available only to agents deciding at step ``t``.  When
    disabled, this returns exact zeros without drawing from ``rng``.
    Explicit noise_only mode ignores strength and returns unit-variance noise.
    """

    if isinstance(n_agents, bool) or not isinstance(n_agents, (int, np.integer)):
        raise TypeError("n_agents must be an integer")
    agent_count = int(n_agents)
    if agent_count <= 0:
        raise ValueError("n_agents must be positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    settings = private_signal_config(config)
    exogenous_returns = np.asarray(exogenous.returns, dtype=np.float64)
    regimes = np.asarray(exogenous.regimes)
    if exogenous_returns.ndim != 1 or regimes.shape != exogenous_returns.shape:
        raise ValueError("exogenous returns and regimes must be equal-length vectors")
    n_steps = exogenous_returns.size
    if not settings.enabled:
        return np.zeros((n_steps, agent_count), dtype=np.float64)

    mode = learned_overlap_mode(config)
    if mode == "none":
        # Preserve the round-3/R4-1 signal draw exactly in the default mode.
        noise = rng.normal(0.0, 1.0, size=(n_steps, agent_count))
    else:
        agents = config["agents"] if "agents" in config else config
        overlap = float(agents["feature_overlap"])
        if not np.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise ValueError("agents.feature_overlap must be finite and in [0, 1]")
        common = rng.normal(0.0, 1.0, size=n_steps)
        idiosyncratic = rng.normal(0.0, 1.0, size=(n_steps, agent_count))
        noise = (
            np.sqrt(overlap) * common[:, None]
            + np.sqrt(1.0 - overlap) * idiosyncratic
        )
    if settings.mode == "noise_only":
        return noise

    configured_volatility = np.asarray(
        config["exogenous"]["regime_volatility"], dtype=np.float64
    )
    if configured_volatility.shape != (2,):
        raise ValueError("exogenous.regime_volatility must contain two values")
    if np.any((regimes < 0) | (regimes >= configured_volatility.size)):
        raise ValueError("exogenous regimes contain an unknown state")
    step_volatility = configured_volatility[regimes.astype(np.int64)]
    if np.any(step_volatility <= 0.0) or not np.all(np.isfinite(step_volatility)):
        raise ValueError(
            "configured regime volatility must be finite and positive when "
            "private signals are enabled"
        )

    standardized_return = exogenous_returns / step_volatility
    if not np.all(np.isfinite(standardized_return)):
        raise ValueError("standardized exogenous returns must be finite")
    residual_scale = np.sqrt(1.0 - settings.strength**2)
    return (
        settings.strength * standardized_return[:, None]
        + residual_scale * noise
    )


__all__ = [
    "PrivateSignalConfig",
    "generate_private_signals",
    "private_signal_config",
]
