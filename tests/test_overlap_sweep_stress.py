"""Registered stress settings, forced-selling rule, and calibration isolation."""
from copy import deepcopy

import numpy as np
import pytest

from herdlab.evaluation.metrics import ConfidenceInterval as CI
from herdlab.experiments import overlap_sweep as legacy
from herdlab.experiments import overlap_sweep_stress as stress
from herdlab.experiments.run_episode import load_config


@pytest.mark.parametrize("lower,counts,supported", [
    (0.01, [30, 30, 30, 0, 0], True),
    (0.01, [30, 30, 29, 0, 0], False),
    (0.0, [100] * 5, False),
    (-0.1, [100] * 5, False),
    (0.01, [0] * 5, False),
])
def test_forced_rule_boundaries(lower, counts, supported):
    result = stress.apply_forced_h1_rule(CI(0.1, lower, 0.2), counts)
    assert result == ("H1 for forced selling is supported in this simulator." if supported
                      else "forced selling was not tested adequately.")


@pytest.mark.parametrize("ci,counts", [
    (CI(np.nan, 0, 1), [30] * 5),
    (CI(1, 2, 0), [30] * 5),
    (CI(1, 0, 2), [30] * 4),
    (CI(1, 0, 2), [30, 30, 30, 30, -1]),
    (CI(1, 0, 2), [30.5] * 5),
])
def test_forced_rule_invalid_inputs(ci, counts):
    with pytest.raises(ValueError):
        stress.apply_forced_h1_rule(ci, counts)


def test_stress_config_only_changes_registered_controls():
    base = load_config(legacy.DEFAULT_CONFIG)
    original = deepcopy(base)
    expected = legacy.overlap_config(base, 0.75)
    expected['agents']['private_signal'].update(mode='signal', beta=0.25)
    expected['agents']['risk']['max_drawdown'] = 0.05
    assert stress.stress_config(base, 0.75) == expected
    assert base == original
    assert legacy.overlap_config(base, 0.75)['agents']['private_signal']['beta'] == 1


def toy_episodes():
    return [stress.StressEpisode(
        legacy._RawEpisode(omega, seed, np.ones(5), np.ones(4),
                           np.zeros(4), np.zeros(5), np.zeros(4),
                           np.full((4, 1), omega > 0.5, dtype=bool)),
        np.array([0., -1., -2., -3.]) if omega == 0.5 else np.full(4, -4.),
    ) for omega in legacy.OVERLAPS for seed in legacy.SEEDS]


def test_forced_threshold_uses_only_midpoint_and_report_has_all_rows():
    raw = toy_episodes()
    result = stress.analyze_stress_episodes(raw, horizon=1, bootstrap_resamples=20)
    assert result.forced_threshold == 3
    assert result.forced_counts[(0.5, 7000)] == 0  # strict > quantile
    assert result.forced_counts[(0., 7000)] == 3
    assert result.overlap.threshold == 0  # separate default threshold
    changed = [stress.StressEpisode(e.diagnostic, e.forced_flow * 100)
               if e.diagnostic.overlap != 0.5 else e for e in raw]
    assert stress.analyze_stress_episodes(changed, horizon=1, bootstrap_resamples=20).forced_threshold == 3
    text = stress.render_report(result, runtime_seconds=1, workers=1)
    assert result.h1_outcome in text
    assert 'Synthetic-simulator disclaimer' in text
    for omega in legacy.OVERLAPS:
        assert f'## omega = {omega:g}' in text
    assert text.count('| 7000 |') == 5


def test_analysis_rejects_missing_or_duplicate_episode():
    raw = toy_episodes()
    with pytest.raises(ValueError, match='every registered'):
        stress.analyze_stress_episodes(raw[:-1], horizon=1)
    with pytest.raises(ValueError, match='every registered'):
        stress.analyze_stress_episodes(raw[:-1] + [raw[0]], horizon=1)
