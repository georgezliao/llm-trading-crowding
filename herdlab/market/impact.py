"""Pluggable price-impact functions for normalized signed order flow.

Impact functions operate on ``Q_t / D_t``; the market simulator applies the
separate impact-sensitivity coefficient.  Implementations support both scalar
values and NumPy arrays to make their mathematical properties easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]


class ImpactFunction(Protocol):
    """Callable interface for dimensionless aggregate price impact."""

    @overload
    def __call__(self, normalized_flow: float) -> float: ...

    @overload
    def __call__(self, normalized_flow: ArrayLike) -> FloatArray: ...

    def __call__(
        self, normalized_flow: float | ArrayLike
    ) -> float | FloatArray: ...


@dataclass(frozen=True)
class LinearImpact:
    """Linear impact ``g(x) = x``."""

    @overload
    def __call__(self, normalized_flow: float) -> float: ...

    @overload
    def __call__(self, normalized_flow: ArrayLike) -> FloatArray: ...

    def __call__(self, normalized_flow: float | ArrayLike) -> float | FloatArray:
        values = np.asarray(normalized_flow, dtype=np.float64)
        return _restore_scalar(values)


@dataclass(frozen=True)
class SignedPowerImpact:
    """Concave impact ``g(x) = sign(x) * abs(x) ** exponent``.

    The exponent is strictly between zero and one, which gives diminishing
    marginal impact as the magnitude of normalized order flow increases.
    """

    exponent: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.exponent) or not 0.0 < self.exponent < 1.0:
            raise ValueError("exponent must be finite and strictly between 0 and 1")

    @overload
    def __call__(self, normalized_flow: float) -> float: ...

    @overload
    def __call__(self, normalized_flow: ArrayLike) -> FloatArray: ...

    def __call__(self, normalized_flow: float | ArrayLike) -> float | FloatArray:
        values = np.asarray(normalized_flow, dtype=np.float64)
        impacted = np.sign(values) * np.power(np.abs(values), self.exponent)
        return _restore_scalar(impacted)


def _restore_scalar(values: FloatArray) -> float | FloatArray:
    """Return a Python float for scalar input and an array otherwise."""

    if values.ndim == 0:
        return float(values)
    return values


__all__ = ["ImpactFunction", "LinearImpact", "SignedPowerImpact"]
