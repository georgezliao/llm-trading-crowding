"""Compatibility, separation, private diagnostics and registered mechanism rules."""
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from herdlab.agents.private_signal import generate_private_signals, private_signal_config
from herdlab.agents.risk import RiskLimits, RiskManager
from herdlab.evaluation.metrics import ConfidenceInterval as CI
from herdlab.experiments import round6_mechanism as m
from herdlab.experiments.overlap_sweep_stress import stress_config
from herdlab.experiments.run_episode import load_config, run_episode, prepare_episode, write_episode
from herdlab.inference.observer_data import load_public_record
from herdlab.inference.features import build_public_features, build_strategy_replication_features
from herdlab.agents.base import PublicObservation


def test_round5_stress_bit_identical():
    frozen = json.loads(Path('tests/fixtures/round5_stress_hashes.json').read_text())
    for omega, hashes in frozen.items():
        config = stress_config(load_config(m.legacy.DEFAULT_CONFIG), float(omega))
        public, private = run_episode(config, 7000)
        for name, digest in hashes.items():
            record = public if hasattr(public, name) else private
            assert hashlib.sha256(getattr(record, name).tobytes()).hexdigest() == digest
        # Explicit diagonal settings must also match the historical values.
        explicit = m.mechanism_config(load_config(m.legacy.DEFAULT_CONFIG), float(omega), float(omega))
        for before, after in zip((public, private), run_episode(explicit, 7000)):
            for field in fields(before):
                np.testing.assert_array_equal(getattr(before, field.name), getattr(after, field.name))


def test_noise_override_independent_of_lag_weights_and_exogenous():
    base = load_config(m.legacy.DEFAULT_CONFIG)
    base['simulation']['n_steps'] = 100
    a = m.mechanism_config(base, 0, 0)
    b = m.mechanism_config(base, 1, 0)
    exo_a, agents_a = prepare_episode(a, 7000)
    exo_b, agents_b = prepare_episode(b, 7000)
    for field in fields(exo_a):
        np.testing.assert_array_equal(getattr(exo_a, field.name), getattr(exo_b, field.name))
    signals_a = generate_private_signals(a, exo_a, 30, np.random.default_rng(17))
    np.testing.assert_array_equal(signals_a, generate_private_signals(b, exo_b, 30, np.random.default_rng(17)))
    b = m.mechanism_config(base, 0, 1)
    _, agents_b = prepare_episode(b, 7000)
    for aa, bb in zip(agents_a, agents_b):
        # Population construction cannot see the noise-sharing override.
        np.testing.assert_array_equal(aa.lag_weights, bb.lag_weights)
    shared = generate_private_signals(b, exo_a, 30, np.random.default_rng(17))
    np.testing.assert_array_equal(shared, np.repeat(shared[:, :1], 30, axis=1))
    assert not np.array_equal(signals_a[:, 0], signals_a[:, 1])
    for invalid in (-.1, 1.1, np.nan, np.inf):
        b['agents']['private_signal']['noise_sharing'] = invalid
        with pytest.raises(ValueError, match='noise_sharing'):
            private_signal_config(b)


def test_risk_logs_new_drawdown_schedules_without_double_counting():
    manager = RiskManager(RiskLimits(.05, 100, 2, 1), initial_peak_equity=100)
    assert not manager.apply(1, position=2, price=1, equity=100).forced
    manager.apply(1, position=2, price=1, equity=90)
    assert manager.last_drawdown_breach and manager.last_liquidating
    manager.apply(1, position=1, price=1, equity=90)
    assert not manager.last_drawdown_breach and manager.last_liquidating
    manager.apply(1, position=0, price=1, equity=90)
    assert not manager.last_drawdown_breach and not manager.last_liquidating
    manager.apply(1, position=0, price=1, equity=90)
    assert not manager.last_drawdown_breach and not manager.last_liquidating
    leverage = RiskManager(RiskLimits(.5, 1, 2, 0), initial_peak_equity=100)
    leverage.apply(0, position=200, price=1, equity=100)
    assert leverage.last_liquidating and not leverage.last_drawdown_breach


def test_private_logging_serialization_and_outcome_definitions(tmp_path, monkeypatch):
    config = m.mechanism_config(load_config(m.legacy.DEFAULT_CONFIG), 0, 0)
    config['simulation']['n_steps'] = 20
    public, private = run_episode(config, 7000)
    # Hand-controlled equity paths and risk flags, independent of simulator formulas.
    equity = np.full_like(private.equity, 100.)
    equity[1:, 0] = 80.
    equity[1:, 1] = 120.
    breaches = np.zeros_like(private.forced)
    breaches[2, 0] = breaches[7, 0] = breaches[2, 1] = True
    liquidating = breaches.copy()
    private = replace(private, equity=equity, drawdown_breaches=breaches,
                      liquidating=liquidating, orders=np.zeros_like(private.orders),
                      forced=np.zeros_like(private.forced), agent_net_flow=np.zeros(20))
    outcome = m.episode_outcome(public, private, 10)
    assert outcome.drawdown_breaches == 3
    assert outcome.fraction_ever_breached == 2/30
    assert outcome.max_simultaneously_liquidating == 2
    assert outcome.mean_agent_max_drawdown == .2/30
    assert outcome.mean_agent_return == 0
    assert outcome.pnl_dispersion == pytest.approx(np.sqrt(800/30))
    assert outcome.forced_flow_windows == 0 and outcome.forced_flow_rate == 0
    monkeypatch.chdir(tmp_path)
    pub_path, priv_path = write_episode(public, private, tmp_path / 'runs/logging')
    with np.load(priv_path) as archive:
        np.testing.assert_array_equal(archive['drawdown_breaches'], breaches)
        np.testing.assert_array_equal(archive['liquidating'], liquidating)
    load_public_record(pub_path)
    names = {f.name for record in (public, PublicObservation) for f in fields(record)}
    names.update(build_public_features(public).names)
    names.update(build_strategy_replication_features(public, config).names)
    assert not {'drawdown_breaches', 'liquidating'} & names
    np.savez(tmp_path / 'bad.npz', **asdict(public), liquidating=liquidating)
    with pytest.raises(ValueError):
        load_public_record(tmp_path / 'bad.npz')
    with pytest.raises(ValueError, match='boolean'):
        replace(private, liquidating=liquidating.astype(float))


@pytest.mark.parametrize('ci,direction', [(CI(2,1,3),'increases'), (CI(-2,-3,-1),'decreases'),
    (CI(0,-1,1),'inconclusive'), (CI(0,0,1),'inconclusive'), (CI(0,-1,0),'inconclusive')])
def test_all_rule_outcomes(ci, direction):
    assert m.forced_selling_direction(ci) == direction
    for name in ('M1','M2','M3'):
        wanted = 'increases' if name == 'M3' else 'decreases'
        state = 'supported' if direction == wanted else 'inconclusive' if direction == 'inconclusive' else 'not supported'
        assert m.mechanism_rule(name, ci) == f'{name}: {state} (association, not cause).'
    text = m.knob_rule('noise sharing', ci)
    assert text == ('noise sharing: drives the decline in this simulator.' if direction == 'decreases' else
                    'noise sharing: inconclusive.' if direction == 'inconclusive' else
                    'noise sharing: does not drive the decline; increases forced-order fraction in this simulator.')


def test_joint_seed_factorial_and_slope_bootstrap():
    offsets = np.arange(30.) * 100
    cells = {(0.,0.): offsets, (0.,1.): offsets - 4,
             (1.,0.): offsets + 2, (1.,1.): offsets - 2}
    effects = m.knob_effects(cells, n_resamples=100)
    assert effects['lag overlap'] == CI(2,2,2)
    assert effects['noise sharing'] == CI(-4,-4,-4)
    y = np.array([offsets - 3*w for w in m.legacy.OVERLAPS])
    ci = m.legacy.episode_bootstrap_slope_ci(m.legacy.OVERLAPS, y,
                                           rng=np.random.default_rng(1), n_resamples=100)
    np.testing.assert_allclose([ci.estimate,ci.lower,ci.upper], -3, atol=1e-12)
    with pytest.raises(ValueError):
        m.knob_effects({})
    with pytest.raises(ValueError):
        m.forced_selling_direction(CI(0,np.nan,1))
    with pytest.raises(ValueError):
        m.analyze({})


def test_analysis_all_mechanisms_and_report_gates():
    conditions = {}
    for lag, noise in m.CONDITIONS:
        conditions[(lag,noise)] = {seed: m.MechanismOutcome(
            1-noise, 2, .1, 2, 1-lag, 1-lag, lag, 0, 0.) for seed in m.legacy.SEEDS}
    slopes, effects, rules = m.analyze(conditions, n_resamples=100)
    assert rules[0] == 'Forced-selling direction: decreases.'
    assert all('supported (association' in rule for rule in rules[1:4])
    report = m.render_report(conditions, runtime=1, compute_seconds=1, workers=1)
    assert all(rule in report for rule in rules)
    assert 'uninformative (<1%)' in report and 'underpowered (<30)' in report
    assert 'fewer than 30 forced-flow windows: 0, 0.25, 0.5, 0.75, 1.' in report
    del conditions[(0.,0.)][7000]
    with pytest.raises(ValueError, match='seed'):
        m.analyze(conditions)


def test_condition_cache_resumes_without_simulation_and_rejects_stale(tmp_path, monkeypatch):
    class SerialPool:
        def __init__(self, workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, tasks):
            return map(function, tasks)

    calls = []
    row = m.MechanismOutcome(0., 0, 0., 0, 0., 0., 0., 0, 0.)

    def fake_simulate(task):
        calls.append(task[1])
        return task[1], row

    monkeypatch.setattr(m.multiprocessing, 'Pool', SerialPool)
    monkeypatch.setattr(m, '_simulate', fake_simulate)
    base = load_config(m.legacy.DEFAULT_CONFIG)
    first, seconds = m.run_conditions(base, tmp_path, 1)
    assert len(calls) == 210
    calls.clear()
    second, retained = m.run_conditions(base, tmp_path, 1)
    assert not calls and first == second and retained == seconds
    base['simulation']['n_steps'] += 1
    with pytest.raises(ValueError, match='stale cache'):
        m.run_conditions(base, tmp_path, 1)
