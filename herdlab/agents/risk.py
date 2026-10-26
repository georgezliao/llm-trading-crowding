"""Private per-agent risk limits and forced exposure reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RiskLimits:
    """Limits that trigger a forced reduction toward zero exposure."""

    max_drawdown: float
    max_leverage: float
    liquidation_steps: int
    cooldown_steps: int = 20

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_drawdown) or not 0.0 <= self.max_drawdown < 1.0:
            raise ValueError("max_drawdown must be finite and in [0, 1)")
        if not np.isfinite(self.max_leverage) or self.max_leverage <= 0.0:
            raise ValueError("max_leverage must be finite and positive")
        if isinstance(self.liquidation_steps, (bool, np.bool_)) or not isinstance(
            self.liquidation_steps, (int, np.integer)
        ):
            raise ValueError("liquidation_steps must be an integer")
        if self.liquidation_steps <= 0:
            raise ValueError("liquidation_steps must be positive")
        if isinstance(self.cooldown_steps, (bool, np.bool_)) or not isinstance(
            self.cooldown_steps, (int, np.integer)
        ):
            raise ValueError("cooldown_steps must be an integer")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be nonnegative")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RiskLimits:
        """Build limits from a full, agents, or risk configuration mapping."""

        section = config.get("agents", config)
        risk = section.get("risk", section)
        return cls(
            max_drawdown=float(risk["max_drawdown"]),
            max_leverage=float(risk["max_leverage"]),
            liquidation_steps=int(risk["liquidation_steps"]),
            cooldown_steps=int(risk.get("cooldown_steps", 20)),
        )


@dataclass(frozen=True)
class OrderDecision:
    """An order quantity together with its private strategy/forced label."""

    quantity: float
    forced: bool


class RiskManager:
    """Apply drawdown and gross-leverage limits to one agent's orders.

    Drawdown is measured from the highest equity supplied so far.  Gross
    leverage is ``abs(position) * price / equity`` (infinite for non-positive
    equity).  On a breach, the current exposure is divided over the remaining
    liquidation decisions so that, if fills match orders, it reaches zero on
    the configured final decision.  A liquidation already under way completes
    even if leverage falls below its limit.  It is followed by a configured
    cooldown targeting zero exposure; when strategy trading resumes, peak
    equity is reset to the then-current equity.

    This object contains private agent state and must not be passed to observer
    code.
    """

    def __init__(
        self, limits: RiskLimits, *, initial_peak_equity: float | None = None
    ) -> None:
        if initial_peak_equity is not None:
            if not np.isfinite(initial_peak_equity) or initial_peak_equity <= 0.0:
                raise ValueError("initial_peak_equity must be finite and positive")
            self._peak_equity: float | None = float(initial_peak_equity)
        else:
            self._peak_equity = None
        self.limits = limits
        self._liquidation_steps_remaining = 0
        self._cooldown_steps_remaining = 0
        self._reset_peak_on_resume = False

    @property
    def peak_equity(self) -> float | None:
        """Highest strictly positive equity observed by this manager."""

        return self._peak_equity

    @property
    def liquidation_steps_remaining(self) -> int:
        """Number of forced reduction decisions left in the active schedule."""

        return self._liquidation_steps_remaining

    @property
    def cooldown_steps_remaining(self) -> int:
        """Number of zero-exposure decisions left in the cooldown."""

        return self._cooldown_steps_remaining

    def drawdown(self, equity: float) -> float:
        """Update peak equity and return current fractional drawdown."""

        _require_finite("equity", equity)
        equity = float(equity)
        if self._peak_equity is None:
            if equity <= 0.0:
                raise ValueError("the first observed equity must be positive")
            self._peak_equity = equity
        elif equity > self._peak_equity:
            self._peak_equity = equity
        return max(0.0, (self._peak_equity - equity) / self._peak_equity)

    @staticmethod
    def leverage(*, position: float, price: float, equity: float) -> float:
        """Return gross leverage for a single-asset position."""

        _require_finite("position", position)
        _require_finite("price", price)
        _require_finite("equity", equity)
        if price <= 0.0:
            raise ValueError("price must be positive")
        if equity <= 0.0:
            return float("inf")
        return abs(float(position)) * float(price) / float(equity)

    def apply(
        self,
        strategy_order: float,
        *,
        position: float,
        price: float,
        equity: float,
    ) -> OrderDecision:
        """Return the strategy order or a forced order that overrides it."""

        _require_finite("strategy_order", strategy_order)
        _require_finite("position", position)

        if self._cooldown_steps_remaining > 0:
            _require_finite("price", price)
            _require_finite("equity", equity)
            if price <= 0.0:
                raise ValueError("price must be positive")
            quantity = -float(position)
            self._cooldown_steps_remaining -= 1
            if self._cooldown_steps_remaining == 0:
                self._reset_peak_on_resume = True
            return OrderDecision(quantity=quantity, forced=quantity != 0.0)

        if self._reset_peak_on_resume:
            _require_finite("equity", equity)
            if equity > 0.0:
                self._peak_equity = float(equity)
            self._reset_peak_on_resume = False

        current_drawdown = self.drawdown(equity)
        current_leverage = self.leverage(
            position=position, price=price, equity=equity
        )
        breached = (
            current_drawdown > self.limits.max_drawdown
            or current_leverage > self.limits.max_leverage
        )

        if breached and self._liquidation_steps_remaining == 0:
            self._liquidation_steps_remaining = self.limits.liquidation_steps

        if breached or self._liquidation_steps_remaining > 0:
            remaining = max(self._liquidation_steps_remaining, 1)
            quantity = -float(position) / remaining
            if self._liquidation_steps_remaining > 0:
                self._liquidation_steps_remaining -= 1
                if self._liquidation_steps_remaining == 0:
                    self._cooldown_steps_remaining = self.limits.cooldown_steps
                    self._reset_peak_on_resume = self.limits.cooldown_steps == 0
            return OrderDecision(quantity=quantity, forced=quantity != 0.0)

        return OrderDecision(quantity=float(strategy_order), forced=False)


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


__all__ = ["OrderDecision", "RiskLimits", "RiskManager"]
