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
        "cooldown_steps": 2,
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


def test_liquidation_cooldown_then_resume_resets_peak_equity() -> None:
    manager = RiskManager(
        _limits(max_leverage=1.0, liquidation_steps=2, cooldown_steps=2),
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
    cooldown = [
        manager.apply(
            strategy_order=0.75, position=0.0, price=50.0, equity=85.0
        ),
        manager.apply(
            strategy_order=0.75, position=0.0, price=50.0, equity=82.0
        ),
    ]
    resumed = manager.apply(
        strategy_order=0.75, position=0.0, price=50.0, equity=82.0
    )

    assert first == pytest.approx(first)  # exercise immutable value return
    assert first.forced and first.quantity == pytest.approx(-1.5)
    assert second.forced and second.quantity == pytest.approx(-1.5)
    assert [decision.quantity for decision in cooldown] == [0.0, 0.0]
    assert all(decision.forced is False for decision in cooldown)
    assert resumed.forced is False
    assert resumed.quantity == 0.75
    assert manager.peak_equity == 82.0


def test_agent_can_breach_recover_and_breach_again_in_one_episode() -> None:
    manager = RiskManager(
        _limits(liquidation_steps=1, cooldown_steps=1),
        initial_peak_equity=100.0,
    )

    first_breach = manager.apply(
        strategy_order=1.0, position=2.0, price=10.0, equity=89.0
    )
    cooldown = manager.apply(
        strategy_order=1.0, position=0.0, price=10.0, equity=89.0
    )
    resumed = manager.apply(
        strategy_order=1.0, position=0.0, price=10.0, equity=90.0
    )
    second_breach = manager.apply(
        strategy_order=1.0, position=1.0, price=10.0, equity=80.0
    )

    assert first_breach.forced is True
    assert cooldown.quantity == 0.0 and cooldown.forced is False
    assert resumed.quantity == 1.0 and resumed.forced is False
    assert manager.peak_equity == 90.0
    assert second_breach.quantity == -1.0 and second_breach.forced is True


def test_zero_quantity_risk_order_is_not_marked_forced() -> None:
    manager = RiskManager(
        _limits(liquidation_steps=1), initial_peak_equity=100.0
    )

    decision = manager.apply(
        strategy_order=2.0, position=0.0, price=10.0, equity=89.0
    )

    assert decision.quantity == 0.0
    assert decision.forced is False


def test_limits_load_from_full_baseline_config() -> None:
    with open("configs/baseline.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    copied = deepcopy(config)

    limits = RiskLimits.from_config(copied)

    assert limits == RiskLimits(
        max_drawdown=0.20,
        max_leverage=2.0,
        liquidation_steps=5,
        cooldown_steps=20,
    )
