"""Per-family signal scaling: compatibility, ratios, leakage and description."""
from copy import deepcopy
from dataclasses import fields

import numpy as np
import pytest

from herdlab.agents.private_signal import private_signal_config
from herdlab.experiments import round6_scaled_grid as g
from herdlab.experiments.experiment_zero_round5 import FAMILIES, forecast_moments
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.market.records import PublicRecord

CONSTANTS = {'momentum': 2e-4, 'reversion': 2e-4, 'linear': 2e-3, 'gbdt': 8e-4, 'mlp': 2e-3}


def _short(config):
    config = deepcopy(config)
    config['simulation']['n_steps'] = 300
    return config


def _base():
    return dict(load_config('configs/experiment_zero.yaml'))


def test_explicit_sigma_matches_omitted_scale():
    config = _short(g.condition_configs(_base())['beta=0.5/rho=0.25'])
    explicit = deepcopy(config)
    explicit['agents']['private_signal']['scale'] = 'sigma'
    first, second = run_episode(config, 8000), run_episode(explicit, 8000)
    np.testing.assert_array_equal(first[0].returns, second[0].returns)
    np.testing.assert_array_equal(first[1].signal_terms, second[1].signal_terms)


def test_family_term_uses_each_family_constant():
    configs = g.scaled_configs(_base(), CONSTANTS)
    config = _short(configs['beta=0.5/noise'])
    _, private = run_episode(config, 8000)
    start = 0
    for family in FAMILIES:
        count = config['agents'][family]['count']
        expected = 0.5 * CONSTANTS[family] * private.signals[:, start:start + count]
        np.testing.assert_allclose(private.signal_terms[:, start:start + count], expected)
        start += count


def test_ratios_track_beta_across_families():
    base = _short(_base())
    off = g.condition_configs(base)['off']
    public, private = run_episode(off, 8000)
    counts = [off['agents'][family]['count'] for family in FAMILIES]
    moments = forecast_moments(private.base_forecasts, np.zeros_like(private.base_forecasts), counts)
    constants = g.family_constants(moments[None])
    config = g.scaled_configs(base, constants)['beta=0.25/noise']
    public, private = run_episode(config, 8000)
    moments = forecast_moments(private.base_forecasts, private.signal_terms, counts)
    n, b, b2, s, s2 = moments.T
    ratios = np.sqrt((s2 / n - (s / n) ** 2) / (b2 / n - (b / n) ** 2))
    assert ratios.max() / ratios.min() < 3.0


def test_constants_use_only_supplied_moments():
    moments = np.zeros((2, 5, 5))
    moments[..., 0] = 4.0
    moments[..., 2] = 4.0
    moments[..., 4] = 1.0
    assert set(g.family_constants(moments).values()) == {1.0}
    moments[:, 0, 2] = 0.0
    with pytest.raises(ValueError):
        g.family_constants(moments)


def test_family_scale_requires_valid_constants():
    config = g.scaled_configs(_base(), CONSTANTS)['beta=1/noise']
    assert private_signal_config(config).scale == 'family'
    broken = deepcopy(config)
    broken['agents']['private_signal']['family_scales'] = None
    with pytest.raises(ValueError):
        private_signal_config(broken)
    broken['agents']['private_signal'].update(family_scales={'momentum': -1.0})
    with pytest.raises(ValueError):
        private_signal_config(broken)


def test_grid_has_off_plus_nine_conditions():
    configs = g.scaled_configs(_base(), CONSTANTS)
    assert len(configs) == 10 and 'off' in configs
    assert configs['off']['agents']['private_signal'].get('scale', 'sigma') == 'sigma'


def test_no_new_public_field():
    names = {field.name for field in fields(PublicRecord)}
    assert not names & {'signals', 'signal_terms', 'base_forecasts', 'family_scales'}


def test_direction_description():
    harder = 'private information makes inference harder'
    easier = 'private information makes inference easier'
    none = 'no directional information effect established'
    assert g.direction_change(harder, harder) == 'same direction'
    assert g.direction_change(easier, harder) == 'direction changed'
    assert g.direction_change(none, harder) == 'direction changed'
    assert g.direction_change(none, none) == 'same direction'
    assert g.direction_change('uninformative', harder) == 'not comparable'
