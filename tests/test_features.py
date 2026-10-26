"""Tests for causal, observer-safe public feature construction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from herdlab.inference.features import build_public_features
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
