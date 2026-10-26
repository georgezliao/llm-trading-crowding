"""Accounting and price-formation tests for the aggregate simulator."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import yaml

from herdlab.agents.base import Agent, PublicObservation
from herdlab.market.exogenous import ExogenousPath
from herdlab.market.simulator import simulate_market


class FixedOrderAgent(Agent):
    """Submit a requested order without relying on a forecast in these tests."""

    def __init__(self, orders: list[float], *, initial_position: float = 0.0) -> None:
        super().__init__(
            q_max=100.0,
            k=1.0,
            u_max=100.0,
            initial_position=initial_position,
        )
        self._orders = iter(orders)

    def forecast(self, observation: PublicObservation) -> float:
        return 0.0

    def decide(self, observation: PublicObservation) -> float:
        return float(next(self._orders))


def _config(n_steps: int, *, cost_rate: float, impact: float = 0.0) -> dict:
    with open("configs/baseline.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["simulation"]["n_steps"] = n_steps
    config["agents"]["transaction_cost_rate"] = cost_rate
    config["market"]["impact_sensitivity"] = impact
    return config


def _path(returns: list[float], *, background_flow: float = 0.0) -> ExogenousPath:
    n_steps = len(returns)
    return ExogenousPath(
        returns=np.asarray(returns, dtype=np.float64),
        regimes=np.zeros(n_steps, dtype=np.int64),
        jump_shocks=np.zeros(n_steps, dtype=np.float64),
        liquidity=np.full(n_steps, 1000.0, dtype=np.float64),
        background_flow=np.full(n_steps, background_flow, dtype=np.float64),
    )


def test_equity_conservation_with_zero_cost_and_zero_impact() -> None:
    agents = [
        FixedOrderAgent([2.0, -0.5, 1.0], initial_position=1.0),
        FixedOrderAgent([-1.0, 0.25, 0.0], initial_position=-2.0),
    ]
    result = simulate_market(
        _config(3, cost_rate=0.0), agents, _path([0.02, -0.01, 0.015])
    )

    price_changes = np.diff(result.prices)[:, None]
    expected_change = result.positions[1:] * price_changes
    np.testing.assert_allclose(np.diff(result.equity, axis=0), expected_change)
    np.testing.assert_allclose(
        np.diff(result.equity.sum(axis=1)), expected_change.sum(axis=1)
    )


def test_transaction_costs_reduce_equity_by_the_charged_amount() -> None:
    no_cost = simulate_market(
        _config(2, cost_rate=0.0),
        [FixedOrderAgent([3.0, -1.0])],
        _path([0.01, -0.02]),
    )
    with_cost = simulate_market(
        _config(2, cost_rate=0.002),
        [FixedOrderAgent([3.0, -1.0])],
        _path([0.01, -0.02]),
    )

    cumulative_cost = with_cost.transaction_costs.sum(axis=0)
    np.testing.assert_allclose(
        no_cost.equity[-1] - with_cost.equity[-1], cumulative_cost
    )
    assert with_cost.equity[-1, 0] < no_cost.equity[-1, 0]


def test_aggregate_flow_drives_impact_and_includes_background_flow() -> None:
    config = _config(1, cost_rate=0.0, impact=0.5)
    result = simulate_market(
        config,
        [FixedOrderAgent([2.0])],
        _path([0.01], background_flow=3.0),
    )

    assert result.aggregate_flow[0] == pytest.approx(5.0)
    assert result.volume[0] == pytest.approx(5.0)
    assert result.returns[0] == pytest.approx(0.01 + 0.5 * 5.0 / 1000.0)
    assert result.fill_prices[0] == pytest.approx(config["simulation"]["initial_price"])


def test_exogenous_path_length_must_match_configuration() -> None:
    config = deepcopy(_config(2, cost_rate=0.0))
    with pytest.raises(ValueError, match="path length"):
        simulate_market(config, [FixedOrderAgent([0.0])], _path([0.0]))
