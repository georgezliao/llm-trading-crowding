"""Resolved populations for round-6 item 1 (no experiments are run here)."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from herdlab.experiments.round5_reconcile import population_configs as historical_configs

FAMILIES = ("momentum", "reversion", "linear", "gbdt", "mlp")


def population_configs() -> dict[str, dict[str, Any]]:
    """Return independent baseline, five omissions, and historical v2 configs.

    Each omission retains six agents from each other family, scaling configured
    capacity, order limit and cash by 30/24. Thus configured aggregate capacity
    and capital, and capacity/cash ratios, match the baseline. The simulator's
    existing random parameter dispersion is untouched: realized sums of drawn
    capacities need not match between populations. Risk limits are unchanged.
    Episode seeds remain caller-supplied through the existing episode runner.
    """
    historical = historical_configs()
    baseline = historical["experiment_zero"]
    configs = {"baseline": baseline}
    for family in FAMILIES:
        config = deepcopy(baseline)
        agents = config["agents"]
        agents[family]["count"] = 0
        agents["population_size"] = 24
        for field in ("q_max", "order_limit", "initial_cash"):
            agents[field] *= 30 / 24
        configs[f"without_{family}"] = config
    configs["round3_v2"] = historical["round3_v2"]
    return configs
