"""Toy inputs for registered R5-3 rules, pairing, scales, and cache resumption."""
from copy import deepcopy

import numpy as np
import pytest

from herdlab.evaluation.metrics import ConfidenceInterval as CI
from herdlab.experiments.experiment_zero_round5 import (
    condition_configs, effect_rule, replication_rule, round4_reading_rule,
    swamping_rule, contrast_ci, forecast_moments, scale_ratios,
    cached_condition, simulate_with_scales,
)
from herdlab.experiments.run_episode import load_config

POS = CI(.2, .1, .3)
NEG = CI(-.2, -.3, -.1)
ZERO = CI(0, -.1, .1)


def test_noise_driven_reading_and_swamping():
    assert effect_rule(POS) == 'noise degrades inference'
    assert round4_reading_rule(POS, POS) == 'not supported'
    assert round4_reading_rule(POS, ZERO) == 'not supported'
    assert 'not rejected' in round4_reading_rule(POS, NEG)
    assert swamping_rule({1: POS, .5: ZERO, .1: ZERO}).endswith(': 0.1')
    assert swamping_rule({.1: NEG, 1: POS}) == 'none'
    assert swamping_rule({.1: CI(0, 0, .1)}).endswith(': 0.1')


@pytest.mark.parametrize('interval,expected', [(NEG, 'harder'), (POS, 'easier'), (ZERO, 'no directional')])
def test_information_direction(interval, expected):
    assert expected in effect_rule(interval, information=True)


def test_uninformative_underpowered_and_boundary_cases():
    assert effect_rule(POS, informative=False) == 'uninformative'
    assert round4_reading_rule(POS, POS, informative=False) == 'uninformative'
    assert replication_rule(POS, .009, forced_events=29) == 'uninformative; underpowered'
    assert replication_rule(POS, .01, forced_events=29) == 'underpowered'
    assert replication_rule(ZERO, .01, forced_events=30) == 'consistent with strategy replication'
    assert replication_rule(NEG, .05) == 'consistent with strategy replication'
    assert replication_rule(POS, .01, forced_events=30).startswith('not consistent')
    undefined = CI(np.nan, np.nan, np.nan)
    assert effect_rule(undefined) == 'undefined'
    assert round4_reading_rule(POS, undefined) == 'undefined'
    assert replication_rule(undefined, .05).startswith('undefined')
    assert 'not established' in effect_rule(CI(.1, 0, .2))


def toy_cell():
    return dict(seeds=(8500, 8501),
                labels=(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])),
                gru=(np.array([.1, .9, .2, .8]), np.array([.6, .2, .8, .4])),
                trivial=(np.full(4, .5), np.full(4, .5)))


def test_joint_resamples_cancel_identical_cells_and_preserve_direction():
    first = toy_cell()
    assert contrast_ci(first, first, rng=np.random.default_rng(4), n_resamples=50) == CI(0, 0, 0)
    second = deepcopy(first)
    second['gru'] = second['trivial']
    positive = contrast_ci(first, second, rng=np.random.default_rng(5), n_resamples=50)
    negative = contrast_ci(second, first, rng=np.random.default_rng(5), n_resamples=50)
    assert positive.estimate > 0
    assert negative.estimate == -positive.estimate
    assert negative.lower == pytest.approx(-positive.upper)
    second['seeds'] = (8501, 8500)
    with pytest.raises(ValueError, match='ordered test seeds'):
        contrast_ci(first, second, rng=np.random.default_rng(4))


def test_no_event_contrast_is_undefined():
    cell = toy_cell()
    cell['labels'] = (np.zeros(4), np.zeros(4))
    assert np.isnan(contrast_ci(cell, cell, rng=np.random.default_rng(1)).estimate)


def test_grid_is_exact_and_does_not_mutate_base():
    base = load_config('configs/experiment_zero.yaml')
    original = deepcopy(base)
    configs = condition_configs(base)
    assert base == original
    assert len(configs) == 17
    assert sum(name.startswith('stress/') for name in configs) == 4
    assert configs['off']['agents']['private_signal']['mode'] == 'off'
    for name, config in configs.items():
        assert config['agents']['learned_overlap_mode'] == 'none'
        if name.startswith('stress/'):
            assert config['agents']['risk']['max_drawdown'] in (.05, .10)
            assert config['agents']['private_signal']['beta'] == .25


def test_scales_pool_samples_not_episode_ratios():
    b = np.array([[1., 2.], [3., 8.]])
    c = np.array([[10., -2.], [5., 0.]])
    moments = np.stack([forecast_moments(b, 2*b, [1, 1]),
                        forecast_moments(c, 3*c, [1, 1])])
    expected = np.std(np.concatenate([2*b, 3*c]), axis=0) / np.std(np.concatenate([b, c]), axis=0)
    np.testing.assert_allclose(scale_ratios(moments), expected)


def test_cache_hit_and_invalidated_key(tmp_path):
    calls = []
    def compute():
        calls.append(1)
        return {'result': np.arange(3)}
    path = tmp_path / 'cell.pkl'
    _, hit = cached_condition(path, 'first', compute)
    assert not hit
    value, hit = cached_condition(path, 'first', compute)
    assert hit and len(calls) == 1
    np.testing.assert_array_equal(value['result'], np.arange(3))
    _, hit = cached_condition(path, 'changed', compute)
    assert not hit and len(calls) == 2
    assert not path.with_suffix('.tmp').exists()


def test_diagnostics_remain_outside_observer_samples():
    config = dict(load_config('configs/experiment_zero.yaml'))
    config['simulation']['n_steps'] = 30
    samples, moments = simulate_with_scales(config, (8,), 1)
    assert moments.shape == (1, 5, 5)
    assert not hasattr(samples[0], 'base_forecasts')
    assert not hasattr(samples[0].public, 'signal_terms')
    assert np.all(scale_ratios(moments) == 0)


def test_report_includes_every_cell_label_and_verbatim_rules(monkeypatch):
    from herdlab.experiments import experiment_zero_round5 as r5
    from herdlab.experiments.experiment_zero_round3 import EventSummary
    from herdlab.experiments.experiment_zero_round4 import ConditionResult, LabelResult, EvaluationResult
    monkeypatch.setattr(r5, 'contrast_ci', lambda *args, **kwargs: POS)
    base = dict(load_config('configs/experiment_zero.yaml'))
    configs = condition_configs(base)
    events = EventSummary(50, 1000)
    evaluation = EvaluationResult(events,
        {'gru_ensemble': {name: POS for name in ('pr_auc', 'brier', 'recall_at_false_alarm_rate')}},
        {name: POS for name in ('replicator', 'lightgbm', 'trivial_reference')}, (.1, .3))
    result = ConditionResult({label: LabelResult(.1, events, evaluation, {6000: 'toy'})
                              for label in ('default', 'forced_flow')})
    cells = {name: dict(result=result, seconds=1, scales=(POS,)*5,
                       rate_cis={label: POS for label in result.labels},
                       contrasts={label: toy_cell() for label in result.labels}) for name in configs}
    report = r5.render_report(base, configs, cells, runtime_seconds=17, workers=10, gru_workers=3)
    assert 'Synthetic-simulator disclaimer' in report
    assert 'runtime stop' not in report
    assert report.count('### beta=') == 24
    assert report.count('### stress/') == 8
    assert 'Round-4 private information degrades inference reading: not supported.' in report
    assert 'Swamping — none.' in report
    for name in configs:
        for label in result.labels:
            assert f'### {name} / {label}' in report
            assert f'Replication / informativeness / power — {name}/{label}:' in report
