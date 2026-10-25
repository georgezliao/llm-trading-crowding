"""Trading-agent interfaces and strategy implementations."""

from herdlab.agents.base import Agent, PublicObservation
from herdlab.agents.factor import (
    MeanReversionAgent,
    MomentumAgent,
    build_factor_population,
)

__all__ = [
    "Agent",
    "MeanReversionAgent",
    "MomentumAgent",
    "PublicObservation",
    "build_factor_population",
]
