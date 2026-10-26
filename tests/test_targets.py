"""Tests for simulator-private crowding and future-unwind targets."""

from __future__ import annotations

import numpy as np
import pytest

from herdlab.inference.targets import (
    UnwindEpisode,
    agent_net_flow,
    fit_unwind_threshold,
    forced_selling_flow,
    forced_unwind_labels,
    forced_unwind_scores,
    flow_crowding,
    onset_unwind_labels,
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


def test_onset_labels_keep_only_qualifying_windows_separated_by_horizon() -> None:
    flow = np.asarray([0.0, 0.0, -2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0])
    liquidity = np.ones_like(flow)

    default = unwind_labels(flow, liquidity, horizon=2, threshold=0.75)
    onset = onset_unwind_labels(flow, liquidity, horizon=2, threshold=0.75)

    np.testing.assert_array_equal(
        default,
        [True, True, False, False, True, True, False],
    )
    np.testing.assert_array_equal(
        onset,
        [True, False, False, False, True, False, False],
    )
    np.testing.assert_array_equal(
        unwind_labels(
            flow,
            liquidity,
            horizon=2,
            threshold=0.75,
            variant="onset",
        ),
        onset,
    )


def test_forced_flow_variant_uses_only_forced_agent_selling() -> None:
    orders = np.asarray(
        [
            [-100.0, 0.0],
            [-3.0, -7.0],
            [-4.0, 9.0],
            [8.0, -6.0],
            [0.0, 0.0],
        ]
    )
    forced = np.asarray(
        [
            [True, False],
            [True, False],
            [False, True],
            [True, True],
            [False, False],
        ],
        dtype=np.bool_,
    )
    liquidity = np.asarray([1.0, 10.0, 20.0, 30.0, 40.0])

    # Current forced selling is excluded.  The strategy sells (-7, -4) and
    # forced buy (+9, +8) do not enter the forced-selling flow.
    np.testing.assert_array_equal(
        forced_selling_flow(orders, forced),
        [-100.0, -3.0, 0.0, -6.0, 0.0],
    )
    np.testing.assert_allclose(
        forced_unwind_scores(orders, forced, liquidity, horizon=2),
        [3.0 / 30.0, 6.0 / 50.0, 6.0 / 70.0],
    )
    np.testing.assert_array_equal(
        forced_unwind_labels(
            orders,
            forced,
            liquidity,
            horizon=2,
            threshold=0.09,
        ),
        [True, True, False],
    )


def test_target_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="liquidity must be positive"):
        unwind_scores([0.0, -1.0], [1.0, 0.0], horizon=1)
    with pytest.raises(ValueError, match="training_episodes must not be empty"):
        fit_unwind_threshold([], horizon=1)
    with pytest.raises(ValueError, match="variant"):
        unwind_labels(
            [0.0, -1.0],
            [1.0, 1.0],
            horizon=1,
            threshold=0.0,
            variant="other",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="forced must be boolean"):
        forced_selling_flow([[-1.0]], [[1]])
