"""Tests for causal, observer-safe public feature construction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from herdlab.agents.base import PublicObservation
from herdlab.agents.factor import MeanReversionAgent, MomentumAgent
from herdlab.inference.features import (
    FEATURE_GROUP_COLUMNS,
    build_public_features,
    build_strategy_replication_features,
)
from herdlab.market.records import PrivateRecord, PublicRecord


def _public_record(n_steps: int = 8) -> PublicRecord:
    returns = np.linspace(-0.02, 0.03, n_steps)
    return PublicRecord(
        prices=100.0 * np.exp(np.concatenate(([0.0], np.cumsum(returns)))),
        returns=returns,
        volume=np.arange(1, n_steps + 1, dtype=np.float64),
        executed_imbalance=np.linspace(-4.0, 3.0, n_steps),
        realized_volatility=np.linspace(0.0, 0.2, n_steps),
        liquidity_proxy=np.linspace(100.0, 170.0, n_steps),
    )


def _replication_config() -> dict:
    return {
        "simulation": {"realized_vol_lookback": 3},
        "agents": {
            "lookback": 3,
            "q_max": 10.0,
            "risk_aversion_k": 2.0,
            "order_limit": 2.5,
            "epsilon": 0.01,
        },
    }


def test_public_features_use_only_observations_at_or_before_t() -> None:
    public = _public_record()
    cutoff = 3
    changed = replace(
        public,
        returns=np.r_[public.returns[: cutoff + 1], np.full(4, 99.0)],
        volume=np.r_[public.volume[: cutoff + 1], np.full(4, 999.0)],
        executed_imbalance=np.r_[
            public.executed_imbalance[: cutoff + 1], np.full(4, -999.0)
        ],
        realized_volatility=np.r_[
            public.realized_volatility[: cutoff + 1], np.full(4, 88.0)
        ],
        liquidity_proxy=np.r_[
            public.liquidity_proxy[: cutoff + 1], np.full(4, 777.0)
        ],
    )

    original_features = build_public_features(public, return_lookback=3)
    changed_features = build_public_features(changed, return_lookback=3)

    assert original_features.names == changed_features.names
    np.testing.assert_array_equal(
        original_features.values[: cutoff + 1],
        changed_features.values[: cutoff + 1],
    )
    assert not np.array_equal(
        original_features.values[cutoff + 1 :],
        changed_features.values[cutoff + 1 :],
    )


@pytest.mark.parametrize("excluded_group", FEATURE_GROUP_COLUMNS)
def test_feature_group_ablation_removes_only_the_group_columns(
    excluded_group: str,
) -> None:
    public = _public_record()
    full = build_public_features(public, return_lookback=3)
    ablated = build_public_features(
        public,
        return_lookback=3,
        exclude_groups=(excluded_group,),
    )
    removed = set(FEATURE_GROUP_COLUMNS[excluded_group])  # type: ignore[index]
    retained_indices = [
        index for index, name in enumerate(full.names) if name not in removed
    ]

    assert all(name.startswith("return_lag_") for name in ablated.names[:3])
    assert ablated.names == tuple(full.names[index] for index in retained_indices)
    np.testing.assert_array_equal(ablated.values, full.values[:, retained_indices])


def test_ablated_public_features_remain_causal() -> None:
    public = _public_record()
    cutoff = 3
    changed = replace(
        public,
        returns=np.r_[public.returns[: cutoff + 1], np.full(4, 99.0)],
        volume=np.r_[public.volume[: cutoff + 1], np.full(4, 999.0)],
        executed_imbalance=np.r_[
            public.executed_imbalance[: cutoff + 1], np.full(4, -999.0)
        ],
        realized_volatility=np.r_[
            public.realized_volatility[: cutoff + 1], np.full(4, 88.0)
        ],
        liquidity_proxy=np.r_[
            public.liquidity_proxy[: cutoff + 1], np.full(4, 777.0)
        ],
    )

    original = build_public_features(
        public, return_lookback=3, exclude_groups=("liquidity",)
    )
    perturbed = build_public_features(
        changed, return_lookback=3, exclude_groups=("liquidity",)
    )

    assert original.names == perturbed.names
    np.testing.assert_array_equal(
        original.values[: cutoff + 1], perturbed.values[: cutoff + 1]
    )


def test_feature_group_ablation_rejects_unknown_groups_and_returns() -> None:
    with pytest.raises(ValueError, match="unknown feature group"):
        build_public_features(_public_record(), exclude_groups=("returns",))


def test_feature_builder_rejects_private_records() -> None:
    private = PrivateRecord(
        positions=np.zeros((3, 1)),
        orders=np.zeros((2, 1)),
        forced=np.zeros((2, 1), dtype=np.bool_),
        cash=np.ones((3, 1)),
        equity=np.ones((3, 1)),
        leverage=np.zeros((3, 1)),
        transaction_costs=np.zeros((2, 1)),
        agent_net_flow=np.zeros(2),
        position_crowding=np.zeros(3),
        flow_crowding=np.zeros(2),
    )
    with pytest.raises(TypeError, match="only a PublicRecord"):
        build_public_features(private)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="only a PublicRecord"):
        build_strategy_replication_features(  # type: ignore[arg-type]
            private, _replication_config()
        )


def test_strategy_replication_features_are_causal_and_price_only() -> None:
    public = _public_record()
    cutoff = 4
    changed_prices = np.asarray(public.prices).copy()
    changed_prices[cutoff + 1 :] *= np.linspace(
        1.1, 1.5, changed_prices.size - cutoff - 1
    )
    changed = replace(
        public,
        prices=changed_prices,
        returns=np.full_like(public.returns, 7.0),
        volume=np.full_like(public.volume, 999.0),
        executed_imbalance=np.full_like(public.executed_imbalance, -999.0),
        realized_volatility=np.full_like(public.realized_volatility, 8.0),
        liquidity_proxy=np.full_like(public.liquidity_proxy, 777.0),
    )

    original = build_strategy_replication_features(public, _replication_config())
    perturbed = build_strategy_replication_features(changed, _replication_config())

    assert original.names == perturbed.names
    np.testing.assert_array_equal(
        original.values[: cutoff + 1], perturbed.values[: cutoff + 1]
    )
    assert not np.array_equal(
        original.values[cutoff + 1 :], perturbed.values[cutoff + 1 :]
    )


def test_strategy_replication_matches_factor_agent_rule() -> None:
    log_returns = np.asarray([0.02, -0.01, 0.03, 0.01, -0.02], dtype=np.float64)
    public = PublicRecord(
        prices=100.0 * np.exp(np.r_[0.0, np.cumsum(log_returns)]),
        returns=np.zeros_like(log_returns),
        volume=np.zeros_like(log_returns),
        executed_imbalance=np.zeros_like(log_returns),
        realized_volatility=np.zeros_like(log_returns),
        liquidity_proxy=np.ones_like(log_returns),
    )
    config = _replication_config()
    common = {
        "lookback": 3,
        "q_max": 10.0,
        "k": 2.0,
        "u_max": 2.5,
        "epsilon": 0.01,
    }
    agents = (MomentumAgent(**common), MeanReversionAgent(**common))
    expected_signals = []
    for step in range(log_returns.size):
        history = log_returns[:step]
        volatility_window = history[-3:]
        volatility = (
            0.0
            if volatility_window.size < 2
            else float(np.std(volatility_window, ddof=1))
        )
        observation = PublicObservation(
            step=step,
            price=float(public.prices[step]),
            returns=history,
            realized_volatility=volatility,
            volume=0.0,
            order_flow_imbalance=0.0,
            liquidity_proxy=1.0,
        )
        targets = np.asarray(
            [
                agent.target_position(agent.forecast(observation), volatility)
                for agent in agents
            ]
        )
        orders = np.asarray([agent.decide(observation) for agent in agents])
        for agent, order in zip(agents, orders, strict=True):
            agent.apply_fill(float(order))
        expected_signals.append(np.r_[targets, orders])

    expected = np.asarray(expected_signals)
    expected_changes = np.zeros_like(expected)
    expected_changes[1:] = np.diff(expected, axis=0)
    actual = build_strategy_replication_features(public, config)

    np.testing.assert_allclose(
        actual.values, np.column_stack((expected, expected_changes))
    )
