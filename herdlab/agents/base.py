"""Base interface and public inputs for trading agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PublicObservation:
    """Causal market information available to every agent at one step.

    ``returns`` contains the public return history through ``step``.  The
    remaining values are the latest public market observations.  Per-agent
    positions, orders, cash, equity, and risk state deliberately do not belong
    in this object.
    """

    step: int
    price: float
    returns: FloatArray
    realized_volatility: float
    volume: float
    order_flow_imbalance: float
    liquidity_proxy: float


class Agent(ABC):
    """Base trading agent with the common exposure-sizing decision rule."""

    def __init__(
        self,
        *,
        q_max: float,
        k: float,
        u_max: float,
        epsilon: float = 1.0e-8,
        initial_position: float = 0.0,
    ) -> None:
        if q_max <= 0.0:
            raise ValueError("q_max must be positive")
        if k <= 0.0:
            raise ValueError("k must be positive")
        if u_max <= 0.0:
            raise ValueError("u_max must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not np.isfinite(initial_position):
            raise ValueError("initial_position must be finite")

        self.q_max = float(q_max)
        self.k = float(k)
        self.u_max = float(u_max)
        self.epsilon = float(epsilon)
        self.position = float(initial_position)

    @abstractmethod
    def forecast(self, observation: PublicObservation) -> float:
        """Forecast the next return using only the supplied public history."""

    def target_position(self, forecast: float, volatility: float) -> float:
        """Return ``q_max * tanh(forecast / (k * volatility + epsilon))``."""

        if not np.isfinite(forecast):
            raise ValueError("forecast must be finite")
        if not np.isfinite(volatility) or volatility < 0.0:
            raise ValueError("volatility must be finite and nonnegative")

        scaled_forecast = forecast / (self.k * volatility + self.epsilon)
        return float(self.q_max * np.tanh(scaled_forecast))

    def decide(
        self,
        observation: PublicObservation,
        *,
        forecast_adjustment: float = 0.0,
    ) -> float:
        """Return the clipped order implied by the agent's current position.

        Calling this method does not assume execution or mutate the position.
        The simulator records an executed fill separately with
        :meth:`apply_fill`.  ``forecast_adjustment`` is private agent input;
        it is added after the family forecast and before the shared tanh rule.
        """

        if not np.isfinite(forecast_adjustment):
            raise ValueError("forecast_adjustment must be finite")
        target = self.target_position(
            self.forecast(observation) + forecast_adjustment,
            observation.realized_volatility,
        )
        return float(np.clip(target - self.position, -self.u_max, self.u_max))

    def apply_fill(self, executed_order: float) -> None:
        """Update private position state after an executed order."""

        if not np.isfinite(executed_order):
            raise ValueError("executed_order must be finite")
        self.position += float(executed_order)


__all__ = ["Agent", "PublicObservation"]
