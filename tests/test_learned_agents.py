"""Tests for the causal rolling ridge agent and population integration."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import yaml

from herdlab.agents.base import PublicObservation
from herdlab.agents.learned import LinearMLAgent, build_population


def _observation(returns: np.ndarray) -> PublicObservation:
    return PublicObservation(
        step=int(returns.size),
        price=100.0,
        returns=returns.copy(),
        realized_volatility=float(np.std(returns[-20:])) if returns.size else 0.0,
        volume=100.0,
        order_flow_imbalance=0.0,
        liquidity_proxy=1000.0,
    )


def _agent(**overrides: float | int) -> LinearMLAgent:
    parameters: dict[str, float | int] = {
        "lookback": 1,
        "ridge_alpha": 1.0e-6,
        "refit_interval": 1,
        "training_window": 200,
        "min_samples": 10,
        "q_max": 10.0,
        "k": 2.0,
        "u_max": 10.0,
    }
    parameters.update(overrides)
    return LinearMLAgent(**parameters)


def _baseline_config() -> dict:
    with open("configs/baseline.yaml", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_future_returns_cannot_affect_earlier_forecasts_or_fit() -> None:
    rng = np.random.default_rng(8)
    returns = rng.normal(0.0, 0.01, size=80)
    changed = returns.copy()
    changed[51:] += 0.2
    first = _agent(refit_interval=5)
    second = _agent(refit_interval=5)

    for end in range(1, 51):
        left = first.forecast(_observation(returns[:end]))
        right = second.forecast(_observation(changed[:end]))
        assert left == right
        np.testing.assert_array_equal(first.coefficients, second.coefficients)
        assert first.intercept == second.intercept


def test_linear_agent_is_deterministic_and_refits_on_fixed_cadence() -> None:
    returns = np.linspace(-0.02, 0.03, 60, dtype=np.float64)
    first = _agent(refit_interval=7)
    second = _agent(refit_interval=7)
    first_forecasts = []
    second_forecasts = []
    fit_steps = []

    previous_fit = None
    for end in range(1, returns.size + 1):
        observation = _observation(returns[:end])
        first_forecasts.append(first.forecast(observation))
        second_forecasts.append(second.forecast(observation))
        if first.last_fit_step != previous_fit:
            fit_steps.append(first.last_fit_step)
            previous_fit = first.last_fit_step

    np.testing.assert_array_equal(first_forecasts, second_forecasts)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert fit_steps == list(range(11, 61, 7))


def test_ridge_learns_sensible_ar1_forecasts() -> None:
    rng = np.random.default_rng(31)
    returns = np.empty(500, dtype=np.float64)
    returns[0] = rng.normal()
    for index in range(1, returns.size):
        returns[index] = 0.8 * returns[index - 1] + rng.normal(scale=0.15)

    agent = _agent(training_window=400, min_samples=30)
    predictions = []
    targets = []
    for end in range(1, returns.size):
        prediction = agent.forecast(_observation(returns[:end]))
        if end >= 100:
            predictions.append(prediction)
            targets.append(returns[end])

    assert 0.7 < agent.coefficients[0] < 0.9
    model_mse = np.mean((np.asarray(predictions) - targets) ** 2)
    zero_mse = np.mean(np.asarray(targets) ** 2)
    assert model_mse < 0.5 * zero_mse


def test_mixed_population_contains_configured_linear_agents() -> None:
    config = deepcopy(_baseline_config())
    config["agents"]["population_size"] = 10
    config["agents"]["linear"]["count"] = 2
    first = build_population(config, np.random.default_rng(4))
    second = build_population(config, np.random.default_rng(4))

    assert len(first) == 10
    assert sum(isinstance(agent, LinearMLAgent) for agent in first) == 2
    for left, right in zip(first, second, strict=True):
        assert type(left) is type(right)
        assert (left.q_max, left.k, left.u_max) == (right.q_max, right.k, right.u_max)
