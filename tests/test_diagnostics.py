"""Tests for per-episode private cascade diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from herdlab.evaluation.diagnostics import episode_cascade_diagnostics, max_drawdown
from herdlab.market.records import PrivateRecord, PublicRecord


def _hand_built_records() -> tuple[PublicRecord, PrivateRecord]:
    orders = np.asarray(
        [
            [2.0, -1.0],
            [-3.0, -1.0],
            [-2.0, 0.0],
            [1.0, 1.0],
        ]
    )
    forced = np.asarray(
        [
            [False, False],
            [True, True],
            [True, False],
            [False, False],
        ]
    )
    public = PublicRecord(
        prices=np.asarray([100.0, 120.0, 90.0, 96.0, 80.0]),
        returns=np.zeros(4),
        volume=np.asarray([4.0, 5.0, 4.0, 3.0]),
        executed_imbalance=np.asarray([2.0, -3.0, -1.0, 3.0]),
        realized_volatility=np.zeros(4),
        liquidity_proxy=np.ones(4),
    )
    private = PrivateRecord(
        positions=np.zeros((5, 2)),
        orders=orders,
        forced=forced,
        cash=np.ones((5, 2)),
        equity=np.ones((5, 2)),
        leverage=np.zeros((5, 2)),
        transaction_costs=np.zeros((4, 2)),
        agent_net_flow=orders.sum(axis=1),
        position_crowding=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        flow_crowding=np.asarray([1.0 / 3.0, 1.0, 1.0, 1.0]),
    )
    return public, private


def test_episode_diagnostics_have_known_answers() -> None:
    public, private = _hand_built_records()

    result = episode_cascade_diagnostics(
        public,
        private,
        labels=[True, False],
        horizon=2,
        quantiles=(0.0, 0.5, 1.0),
    )

    assert result.fraction_steps_with_forced_order == 0.5
    assert result.max_simultaneously_forced_agents == 2
    assert result.mean_simultaneously_forced_agents == 0.75
    assert result.forced_share_of_absolute_agent_flow == pytest.approx(6.0 / 11.0)
    assert result.agent_flow_share_of_total_absolute_flow == pytest.approx(11.0 / 16.0)
    assert result.price_max_drawdown == pytest.approx(1.0 / 3.0)
    assert result.position_crowding_quantiles == {0.0: 0.0, 0.5: 0.5, 1.0: 1.0}
    assert result.flow_crowding_quantiles == pytest.approx(
        {0.0: 1.0 / 3.0, 0.5: 1.0, 1.0: 1.0}
    )
    assert result.fraction_positive_windows_with_forced_flow == 1.0


def test_positive_windows_use_strictly_future_forced_flow() -> None:
    public, private = _hand_built_records()

    result = episode_cascade_diagnostics(
        public,
        private,
        labels=[False, True, True],
        horizon=1,
        quantiles=(0.5,),
    )

    # Positive windows t=1 and t=2 inspect steps 2 and 3.  The first contains
    # forced flow and the second does not; the forced orders at t=1 are current
    # to the first positive label and therefore excluded.
    assert result.fraction_positive_windows_with_forced_flow == 0.5


def test_zero_flow_and_no_positive_windows_have_zero_shares() -> None:
    public, private = _hand_built_records()
    zero_private = PrivateRecord(
        positions=private.positions,
        orders=np.zeros_like(private.orders),
        forced=np.zeros_like(private.forced),
        cash=private.cash,
        equity=private.equity,
        leverage=private.leverage,
        transaction_costs=private.transaction_costs,
        agent_net_flow=np.zeros(4),
        position_crowding=private.position_crowding,
        flow_crowding=np.zeros(4),
    )
    zero_public = PublicRecord(
        prices=public.prices,
        returns=public.returns,
        volume=np.zeros(4),
        executed_imbalance=np.zeros(4),
        realized_volatility=public.realized_volatility,
        liquidity_proxy=public.liquidity_proxy,
    )

    result = episode_cascade_diagnostics(
        zero_public, zero_private, labels=[False, False], horizon=2
    )

    assert result.forced_share_of_absolute_agent_flow == 0.0
    assert result.agent_flow_share_of_total_absolute_flow == 0.0
    assert result.fraction_positive_windows_with_forced_flow == 0.0


def test_diagnostic_input_validation() -> None:
    public, private = _hand_built_records()

    assert max_drawdown([100.0, 110.0, 121.0]) == 0.0
    with pytest.raises(ValueError, match="strictly-future"):
        episode_cascade_diagnostics(public, private, labels=[True], horizon=2)
    with pytest.raises(ValueError, match="quantiles"):
        episode_cascade_diagnostics(
            public, private, labels=[True, False], horizon=2, quantiles=()
        )
