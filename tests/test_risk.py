"""Tests for private per-agent drawdown and leverage limits."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from herdlab.agents.risk import RiskLimits, RiskManager


def _limits(**overrides: float | int) -> RiskLimits:
    values: dict[str, float | int] = {
        "max_drawdown": 0.10,
        "max_leverage": 2.0,
        "liquidation_steps": 4,
    }
    values.update(overrides)
    return RiskLimits(**values)  # type: ignore[arg-type]


def test_drawdown_breach_forces_orders_until_exposure_is_reduced() -> None:
    manager = RiskManager(_limits())
    position = 8.0

    healthy = manager.apply(
        strategy_order=1.5, position=position, price=10.0, equity=100.0
    )
    assert healthy.quantity == 1.5
    assert healthy.forced is False

    quantities = []
    for _ in range(4):
        decision = manager.apply(
            strategy_order=2.0, position=position, price=10.0, equity=89.0
        )
        assert decision.forced is True
        assert decision.quantity < 0.0
        previous_magnitude = abs(position)
        position += decision.quantity
        assert abs(position) < previous_magnitude
        quantities.append(decision.quantity)

    assert quantities == pytest.approx([-2.0, -2.0, -2.0, -2.0])
    assert position == pytest.approx(0.0)


def test_forced_reduction_works_for_a_short_position() -> None:
    manager = RiskManager(
        _limits(max_leverage=1.0, liquidation_steps=2),
        initial_peak_equity=100.0,
    )
    position = -3.0

    first = manager.apply(
        strategy_order=-5.0, position=position, price=50.0, equity=100.0
    )
    position += first.quantity
    second = manager.apply(
        strategy_order=-5.0, position=position, price=50.0, equity=100.0
    )
    position += second.quantity

    assert first.forced and second.forced
    assert first.quantity == pytest.approx(1.5)
    assert second.quantity == pytest.approx(1.5)
    assert position == pytest.approx(0.0)


def test_liquidation_schedule_finishes_after_leverage_returns_within_limit() -> None:
    manager = RiskManager(
        _limits(max_leverage=1.0, liquidation_steps=2),
        initial_peak_equity=100.0,
    )

    first = manager.apply(
        strategy_order=4.0, position=3.0, price=50.0, equity=100.0
    )
    second = manager.apply(
        strategy_order=4.0,
        position=1.5,
        price=50.0,
        equity=100.0,
    )
    resumed = manager.apply(
        strategy_order=0.75, position=0.0, price=50.0, equity=100.0
    )

    assert first == pytest.approx(first)  # exercise immutable value return
    assert first.forced and first.quantity == pytest.approx(-1.5)
    assert second.forced and second.quantity == pytest.approx(-1.5)
    assert resumed.forced is False
    assert resumed.quantity == 0.75


def test_limits_load_from_full_baseline_config() -> None:
    with open("configs/baseline.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    copied = deepcopy(config)

    limits = RiskLimits.from_config(copied)

    assert limits == RiskLimits(
        max_drawdown=0.20, max_leverage=2.0, liquidation_steps=5
    )
