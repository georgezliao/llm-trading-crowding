"""Tests for the common agent decision model."""

from __future__ import annotations

import numpy as np

from herdlab.agents.base import Agent, PublicObservation


class FixedForecastAgent(Agent):
    """Test agent whose forecast is fixed in advance."""

    def __init__(self, forecast: float, **kwargs: float) -> None:
        super().__init__(**kwargs)
        self.fixed_forecast = forecast

    def forecast(self, observation: PublicObservation) -> float:
        return self.fixed_forecast


def _observation(volatility: float) -> PublicObservation:
    return PublicObservation(
        step=3,
        price=101.5,
        returns=np.array([0.01, -0.02, 0.005]),
        realized_volatility=volatility,
        volume=125.0,
        order_flow_imbalance=-0.15,
        liquidity_proxy=900.0,
    )


def test_target_and_unclipped_order_match_hand_computation() -> None:
    agent = FixedForecastAgent(
        forecast=0.03,
        q_max=10.0,
        k=2.0,
        u_max=10.0,
        epsilon=0.01,
        initial_position=1.25,
    )
    expected_target = 10.0 * np.tanh(0.03 / (2.0 * 0.01 + 0.01))

    assert agent.target_position(0.03, 0.01) == expected_target
    assert agent.decide(_observation(0.01)) == expected_target - 1.25


def test_order_is_clipped_in_both_directions() -> None:
    buyer = FixedForecastAgent(
        forecast=1.0, q_max=10.0, k=1.0, u_max=2.0, epsilon=0.01
    )
    seller = FixedForecastAgent(
        forecast=-1.0,
        q_max=10.0,
        k=1.0,
        u_max=2.0,
        epsilon=0.01,
        initial_position=1.0,
    )

    assert buyer.decide(_observation(0.0)) == 2.0
    assert seller.decide(_observation(0.0)) == -2.0


def test_apply_fill_changes_the_position_used_by_next_decision() -> None:
    agent = FixedForecastAgent(
        forecast=0.0, q_max=10.0, k=2.0, u_max=3.0, initial_position=2.0
    )

    assert agent.decide(_observation(0.01)) == -2.0
    agent.apply_fill(-1.5)
    assert agent.position == 0.5
    assert agent.decide(_observation(0.01)) == -0.5


def test_public_observation_contains_no_private_agent_fields() -> None:
    field_names = set(PublicObservation.__dataclass_fields__)

    assert field_names == {
        "step",
        "price",
        "returns",
        "realized_volatility",
        "volume",
        "order_flow_imbalance",
        "liquidity_proxy",
    }
