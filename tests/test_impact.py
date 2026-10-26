"""Tests for aggregate price-impact functions."""

from __future__ import annotations

import numpy as np
import pytest

from herdlab.market.impact import LinearImpact, SignedPowerImpact


@pytest.mark.parametrize("impact", [LinearImpact(), SignedPowerImpact(0.5)])
def test_impact_is_zero_at_zero(impact: LinearImpact | SignedPowerImpact) -> None:
    assert impact(0.0) == 0.0


@pytest.mark.parametrize("impact", [LinearImpact(), SignedPowerImpact(0.6)])
def test_impact_has_odd_symmetry(
    impact: LinearImpact | SignedPowerImpact,
) -> None:
    positive_flow = np.array([0.0, 0.01, 0.5, 1.0, 4.0])

    np.testing.assert_allclose(impact(-positive_flow), -impact(positive_flow))


@pytest.mark.parametrize("impact", [LinearImpact(), SignedPowerImpact(0.4)])
def test_impact_is_monotonic(impact: LinearImpact | SignedPowerImpact) -> None:
    ordered_flow = np.linspace(-10.0, 10.0, 1001)

    impacted = impact(ordered_flow)

    assert np.all(np.diff(impacted) > 0.0)


def test_linear_impact_returns_normalized_flow() -> None:
    flow = np.array([-2.5, 0.0, 1.25])

    np.testing.assert_array_equal(LinearImpact()(flow), flow)
    assert LinearImpact()(1.25) == 1.25


def test_signed_power_impact_uses_configured_exponent() -> None:
    flow = np.array([-16.0, -1.0, 0.0, 1.0, 16.0])

    np.testing.assert_allclose(
        SignedPowerImpact(0.5)(flow),
        np.array([-4.0, -1.0, 0.0, 1.0, 4.0]),
    )


@pytest.mark.parametrize("exponent", [-1.0, 0.0, 1.0, 2.0, np.inf, np.nan])
def test_signed_power_requires_a_strictly_concave_exponent(exponent: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        SignedPowerImpact(exponent)
