"""Tests for simulator-private crowding and future-unwind targets."""

from __future__ import annotations

import numpy as np
import pytest

from herdlab.inference.targets import (
    UnwindEpisode,
    agent_net_flow,
    fit_unwind_threshold,
    flow_crowding,
    position_crowding,
    unwind_labels,
    unwind_scores,
)


def test_crowding_bounds_and_complete_same_sign_alignment() -> None:
    positions = np.asarray([[1.0, 2.0, 3.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    orders = np.asarray([[-2.0, -1.0, -4.0], [1.0, -1.0, 0.0]])

    position_values = position_crowding(positions)
    flow_values = flow_crowding(orders)

    assert position_values[0] == 1.0
    assert flow_values[0] == 1.0
    assert position_values[2] == 0.0
    assert np.all((0.0 <= position_values) & (position_values <= 1.0))
    assert np.all((0.0 <= flow_values) & (flow_values <= 1.0))


def test_agent_net_flow_excludes_any_background_component() -> None:
    orders = np.asarray([[2.0, -1.0], [-3.0, -4.0]])
    np.testing.assert_array_equal(agent_net_flow(orders), [1.0, -7.0])


def test_unwind_score_uses_strictly_future_net_flow_and_liquidity() -> None:
    flow = np.asarray([-100.0, -2.0, -4.0, 9.0, -8.0])
    liquidity = np.asarray([1.0, 10.0, 20.0, 30.0, 40.0])

    scores = unwind_scores(flow, liquidity, horizon=2)

    # The large current-step sale is excluded.  Future buying offsets selling.
    np.testing.assert_allclose(scores, [6.0 / 30.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        unwind_labels(flow, liquidity, horizon=2, threshold=0.1),
        [True, False, False],
    )


def test_threshold_is_fit_only_from_episodes_passed_as_training_data() -> None:
    train = [
        UnwindEpisode(
            agent_net_flow=np.asarray([0.0, -1.0, -2.0, -3.0]),
            liquidity=np.ones(4),
        )
    ]
    disjoint_test = UnwindEpisode(
        agent_net_flow=np.asarray([0.0, -1000.0, -1000.0, -1000.0]),
        liquidity=np.ones(4),
    )

    threshold = fit_unwind_threshold(train, horizon=1, quantile=1.0)
    assert threshold == 3.0
    assert fit_unwind_threshold(train, horizon=1, quantile=1.0) == threshold
    assert fit_unwind_threshold([disjoint_test], horizon=1, quantile=1.0) == 1000.0


def test_target_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="liquidity must be positive"):
        unwind_scores([0.0, -1.0], [1.0, 0.0], horizon=1)
    with pytest.raises(ValueError, match="training_episodes must not be empty"):
        fit_unwind_threshold([], horizon=1)
