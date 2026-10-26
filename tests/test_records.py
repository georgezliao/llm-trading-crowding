"""Tests for the hard public/private episode-data boundary."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import yaml

from herdlab.agents.base import Agent, PublicObservation
from herdlab.inference.observer_data import load_public_record
from herdlab.market.exogenous import ExogenousPath
from herdlab.market.records import PrivateRecord, PublicRecord
from herdlab.market.simulator import simulate_market_records


class FixedOrderAgent(Agent):
    def __init__(self, orders: list[float]) -> None:
        super().__init__(q_max=100.0, k=1.0, u_max=100.0)
        self._orders = iter(orders)

    def forecast(self, observation: PublicObservation) -> float:
        return 0.0

    def decide(self, observation: PublicObservation) -> float:
        return float(next(self._orders))


def _records() -> tuple[PublicRecord, PrivateRecord]:
    with open("configs/baseline.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["simulation"]["n_steps"] = 2
    path = ExogenousPath(
        returns=np.asarray([0.01, -0.02]),
        regimes=np.zeros(2, dtype=np.int64),
        jump_shocks=np.zeros(2),
        liquidity=np.asarray([1000.0, 900.0]),
        background_flow=np.asarray([4.0, -3.0]),
    )
    return simulate_market_records(
        config,
        [FixedOrderAgent([2.0, -1.0]), FixedOrderAgent([1.0, -2.0])],
        path,
    )


def test_public_record_schema_has_no_agent_only_fields() -> None:
    public, private = _records()
    public_names = {field.name for field in fields(public)}
    forbidden_fragments = {"agent", "position", "order", "forced", "cash", "equity", "leverage"}

    assert not any(
        fragment in name
        for name in public_names
        for fragment in forbidden_fragments
    )
    np.testing.assert_allclose(public.executed_imbalance, [7.0, -6.0])
    np.testing.assert_allclose(private.agent_net_flow, [3.0, -3.0])
    np.testing.assert_allclose(private.position_crowding, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(private.flow_crowding, [1.0, 1.0])
    assert not np.array_equal(public.executed_imbalance, private.agent_net_flow)


def test_observer_loader_returns_public_record_and_rejects_private_data(
    tmp_path,
) -> None:
    public, private = _records()
    public_path = tmp_path / "public.npz"
    np.savez(public_path, **vars(public))

    loaded = load_public_record(public_path)
    assert isinstance(loaded, PublicRecord)
    np.testing.assert_array_equal(loaded.executed_imbalance, public.executed_imbalance)
    with pytest.raises(TypeError, match="only a PublicRecord"):
        load_public_record(private)  # type: ignore[arg-type]


def test_observer_loader_rejects_archive_containing_private_field(tmp_path) -> None:
    public, private = _records()
    mixed_path = tmp_path / "mixed.npz"
    np.savez(mixed_path, **vars(public), agent_net_flow=private.agent_net_flow)

    with pytest.raises(ValueError, match="not a public-record archive"):
        load_public_record(mixed_path)
