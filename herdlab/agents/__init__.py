"""Trading-agent interfaces and strategy implementations."""

from herdlab.agents.base import Agent, PublicObservation
from herdlab.agents.factor import (
    MeanReversionAgent,
    MomentumAgent,
    build_factor_population,
)
from herdlab.agents.learned import (
    LightGBMAgent,
    LinearMLAgent,
    MLPAgent,
    build_gbdt_population,
    build_linear_population,
    build_mlp_population,
    build_population,
)
from herdlab.agents.risk import OrderDecision, RiskLimits, RiskManager

__all__ = [
    "Agent",
    "LightGBMAgent",
    "LinearMLAgent",
    "MLPAgent",
    "MeanReversionAgent",
    "MomentumAgent",
    "OrderDecision",
    "PublicObservation",
    "RiskLimits",
    "RiskManager",
    "build_factor_population",
    "build_gbdt_population",
    "build_linear_population",
    "build_mlp_population",
    "build_population",
]
