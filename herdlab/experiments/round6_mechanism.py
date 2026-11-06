"""Registered round-6 paired-seed mechanism diagnostics (private evaluation only)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from herdlab.evaluation.metrics import ConfidenceInterval
from herdlab.experiments import overlap_sweep as legacy
from herdlab.experiments.overlap_sweep_stress import stress_config
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.inference.targets import forced_selling_flow, unwind_labels

CONDITIONS = tuple((w, w) for w in legacy.OVERLAPS) + ((0., 1.), (1., 0.))
BOOTSTRAP_SEED = 20261106


@dataclass(frozen=True)
class MechanismOutcome:
    forced_order_fraction: float
    drawdown_breaches: int
    fraction_ever_breached: float
    max_simultaneously_liquidating: int
    mean_agent_max_drawdown: float
    pnl_dispersion: float
    mean_agent_return: float
    forced_flow_windows: int
    forced_flow_rate: float


def mechanism_config(base, lag_overlap, noise_sharing):
    config = stress_config(base, lag_overlap)
    config['agents']['private_signal']['noise_sharing'] = noise_sharing
    return config


def episode_outcome(public, private, horizon):
    """Reduce private diagnostics; all equity rows include the initial/final state."""
    if private.drawdown_breaches is None or private.liquidating is None:
        raise ValueError('mechanism logging is required')
    peaks = np.maximum.accumulate(private.equity, axis=0)
    drawdowns = (peaks - private.equity) / peaks
    pnl = private.equity[-1] - private.equity[0]
    labels = unwind_labels(forced_selling_flow(private.orders, private.forced),
                           public.liquidity_proxy, horizon=horizon, threshold=0.)
    return MechanismOutcome(
        float(private.forced.mean()), int(private.drawdown_breaches.sum()),
        float(private.drawdown_breaches.any(axis=0).mean()),
        int(private.liquidating.sum(axis=1).max()),
        float(drawdowns.max(axis=0).mean()), float(pnl.std(ddof=0)),
        float((pnl / private.equity[0]).mean()), int(labels.sum()), float(labels.mean()),
    )


def forced_selling_direction(ci):
    legacy._validate_interval(ci, 'forced-order fraction slope')
    return 'increases' if ci.lower > 0 else 'decreases' if ci.upper < 0 else 'inconclusive'


def mechanism_rule(name, ci):
    if name not in ('M1', 'M2', 'M3'):
        raise ValueError('unknown mechanism')
    direction = forced_selling_direction(ci)
    wanted = 'increases' if name == 'M3' else 'decreases'
    outcome = ('supported' if direction == wanted else
               'inconclusive' if direction == 'inconclusive' else 'not supported')
    return f'{name}: {outcome} (association, not cause).'


def knob_rule(name, ci):
    direction = forced_selling_direction(ci)
    if direction == 'decreases':
        return f'{name}: drives the decline in this simulator.'
    if direction == 'increases':
        return f'{name}: does not drive the decline; increases forced-order fraction in this simulator.'
    return f'{name}: inconclusive.'


def mean_ci(values, *, rng, n_resamples=legacy.DEFAULT_BOOTSTRAP_RESAMPLES):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('values must be a nonempty finite vector')
    count = legacy._positive_integer(n_resamples, 'n_resamples')
    samples = values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    return ConfidenceInterval(float(values.mean()), float(lo), float(hi))


def knob_effects(cells, *, n_resamples=legacy.DEFAULT_BOOTSTRAP_RESAMPLES):
    """Average high-minus-low effects, pairing seeds across all four cells."""
    if set(cells) != {(0., 0.), (0., 1.), (1., 0.), (1., 1.)}:
        raise ValueError('exactly four factorial cells required')
    a, b, c, d = (np.asarray(cells[k], dtype=float) for k in
                   ((0., 0.), (0., 1.), (1., 0.), (1., 1.)))
    if not a.ndim == 1 or not a.size or any(v.shape != a.shape for v in (b, c, d)):
        raise ValueError('factorial cells must have matched episode vectors')
    return {name: mean_ci(value, rng=np.random.default_rng(BOOTSTRAP_SEED + 100 + i),
                           n_resamples=n_resamples)
            for i, (name, value) in enumerate((
                ('lag overlap', ((c - a) + (d - b)) / 2),
                ('noise sharing', ((b - a) + (d - c)) / 2)))}


def analyze(conditions, *, n_resamples=legacy.DEFAULT_BOOTSTRAP_RESAMPLES):
    if set(conditions) != set(CONDITIONS):
        raise ValueError('all seven unique conditions required')
    if any(set(rows) != set(legacy.SEEDS) for rows in conditions.values()):
        raise ValueError('every registered seed required exactly once per condition')
    slopes = {}
    for i, field in enumerate(fields(MechanismOutcome)):
        values = np.array([[getattr(conditions[(w, w)][seed], field.name)
                            for seed in legacy.SEEDS] for w in legacy.OVERLAPS])
        slopes[field.name] = legacy.episode_bootstrap_slope_ci(
            legacy.OVERLAPS, values, rng=np.random.default_rng(BOOTSTRAP_SEED + i),
            n_resamples=n_resamples)
    effects = knob_effects({key: [conditions[key][seed].forced_order_fraction
                                  for seed in legacy.SEEDS]
                            for key in ((0., 0.), (0., 1.), (1., 0.), (1., 1.))},
                           n_resamples=n_resamples)
    rules = [f'Forced-selling direction: {forced_selling_direction(slopes["forced_order_fraction"])}.']
    rules += [mechanism_rule(name, slopes[outcome]) for name, outcome in (
        ('M1', 'mean_agent_max_drawdown'), ('M2', 'pnl_dispersion'), ('M3', 'mean_agent_return'))]
    rules += [knob_rule(name, ci) for name, ci in effects.items()]
    return slopes, effects, rules


def _simulate(task):
    config, seed = task
    public, private = run_episode(config, seed)
    return seed, episode_outcome(public, private, int(config['targets']['unwind_horizon']))


def fingerprint(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(json.dumps(legacy.SEEDS).encode())
    # Includes simulator, agent, target and runner code; stale caches fail closed.
    for path in sorted(Path('herdlab').rglob('*.py')):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_conditions(base, cache_dir, workers):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    conditions, compute_seconds = {}, 0.
    workers = legacy._positive_integer(workers, 'workers')
    with multiprocessing.Pool(workers) as pool:
        for lag, noise in CONDITIONS:
            config = mechanism_config(base, lag, noise)
            signature = fingerprint(config)
            path = cache_dir / f'lag_{lag:g}_noise_{noise:g}.json'
            if path.exists():
                data = json.loads(path.read_text())
                if data['fingerprint'] != signature or data['config'] != config:
                    raise ValueError(f'stale cache: {path}')
                print(f'Reusing {path}', flush=True)
            else:
                start = time.perf_counter()
                rows = dict(pool.map(_simulate, [(config, seed) for seed in legacy.SEEDS]))
                data = dict(fingerprint=signature, config=config,
                            seconds=time.perf_counter() - start,
                            episodes={str(seed): asdict(value) for seed, value in rows.items()})
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(data, indent=2) + '\n')
                temporary.replace(path)
                print(f'Finished lag={lag:g}, noise={noise:g}: {data["seconds"]:.2f}s', flush=True)
            conditions[(lag, noise)] = {int(seed): MechanismOutcome(**row)
                                        for seed, row in data['episodes'].items()}
            compute_seconds += data['seconds']
    return conditions, compute_seconds


def render_report(conditions, *, runtime, compute_seconds, workers):
    slopes, effects, rules = analyze(conditions)
    lines = ['# Round-6 mechanism diagnostics', '',
        '**Synthetic-simulator disclaimer:** All results are from the uncalibrated synthetic HerdLab simulator. They are not evidence of real-market behavior, funds, or live trading.', '',
        '## Configuration and execution', '',
        '`configs/experiment_zero.yaml`; 500 steps, 30 agents; rho=0.25, beta=0.25, max_drawdown=0.05, learned_overlap_mode=lag_weights. Seeds 7000–7029 shared across every condition. Five coupled omega levels plus lag overlap {0,1} × noise sharing {0,1}; diagonal corners reused: 210 unique episodes.', '',
        f'Runtime estimate before execution: 2–4 minutes. Measured invocation wall time {runtime / 60:.2f} minutes; retained condition compute {compute_seconds / 60:.2f} minutes; {workers} workers. Command: `.venv/bin/python -m herdlab.experiments.round6_mechanism --workers {workers}`. Completed conditions are cached under `runs/round6_mechanism_cache/` with config/source fingerprints and private episode outcomes.', '',
        'Forced-flow threshold is the explicitly registered existing zero threshold, fixed without fitting on these diagnostic episodes; horizon 10, complete strictly-future windows only. Counts include overlapping windows, not independent events. Forced-order fraction includes nonzero forced buys and sells.', '',
        'Drawdown breaches count new liquidation schedules triggered with drawdown above the risk limit (including simultaneous leverage breaches); ongoing schedules and cooldown do not add breaches. Fraction ever breached uses those drawdown events. Simultaneously liquidating counts active liquidation decisions, including zero exposure, excluding cooldown. Mean agent maximum drawdown uses each full equity path and its running all-time peak, without the risk-manager cooldown reset. P&L dispersion is the population standard deviation across agents of terminal minus initial equity in currency units; mean return is the mean of those P&Ls divided by initial agent equity. All diagnostics stay private.', '',
        'OLS includes an intercept. All 95% percentile CIs use 10,000 whole-episode resamples; slope and factorial contrasts jointly resample seed identities across conditions. Slope RNG seeds: 20261106 + outcome row index; factorial seeds: 20261206/20261207. Main effects are high minus low averaged equally over the other knob.', '',
        '## Condition means and forced-flow gates', '',
        '| Lag | Noise | ' + ' | '.join(f.name for f in fields(MechanismOutcome)) + ' | Total windows | Gate |',
        '| ' + ' | '.join(['---:'] * 12) + ' | --- |']
    low = []
    for key in CONDITIONS:
        rows = list(conditions[key].values())
        means = [np.mean([getattr(row, f.name) for row in rows]) for f in fields(MechanismOutcome)]
        count = sum(row.forced_flow_windows for row in rows)
        gates = []
        if means[-1] < .01:
            gates.append('uninformative (<1%)')
        if count < 30:
            gates.append('underpowered (<30)')
            if key[0] == key[1]:
                low.append(f'{key[0]:g}')
        lines.append('| ' + ' | '.join([f'{v:g}' for v in key] +
                     [f'{v:.8f}' for v in means] + [str(count), ', '.join(gates) or 'clears both']) + ' |')
    lines += ['', 'Sweep levels with fewer than 30 forced-flow windows: ' + (', '.join(low) or 'none') + '.', '',
        'These gates flag the forced-flow label. The registered continuous-outcome slope and knob rules below are reported even for sparse windows; this avoids reintroducing the round-5 direction-rule problem. No observer is fitted in item 2, so the replication rule is not applicable.', '',
        '## Slopes with 95% CIs', '', '| Outcome | Slope | Lower | Upper |', '| --- | ---: | ---: | ---: |']
    for name, ci in slopes.items():
        lines.append(f'| {name} | {ci.estimate:.10f} | {ci.lower:.10f} | {ci.upper:.10f} |')
    lines += ['', '## Factorial main effects on forced-order fraction', '',
              '| Setting | Effect | Lower | Upper |', '| --- | ---: | ---: | ---: |']
    for name, ci in effects.items():
        lines.append(f'| {name} | {ci.estimate:.10f} | {ci.lower:.10f} | {ci.upper:.10f} |')
    lines += ['', '## Rule outcomes (verbatim)', ''] + [line for rule in rules for line in (rule, '')]
    lines += ['M1–M3 may hold together and describe associations only. Knob attribution is the sole causal interpretation allowed here and is limited to these interventions in this simulator.', '',
        '## Deviations, definitions and open questions', '',
        'No equation, label, threshold, population, seed, or parameter was tuned after viewing results. Missing noise_sharing retains the historical mode-dependent coupling and RNG draw order; frozen full-length pre-change stress hashes cover all five omega levels. New logging does not affect decisions. No later step was implemented.', '',
        'For research-lead review: the registration does not specify breach episode versus exceedance-step counts, reset versus all-time drawdown, terminal versus pathwise P&L dispersion, or whether zero-exposure liquidation decisions count as liquidating. The explicit operational definitions above were fixed before execution. Confirm them before interpreting M1–M3. The general 1%/30-window gates are displayed without suppressing the specifically registered continuous-outcome rules; no replication claim is made.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--cache', default='runs/round6_mechanism_cache')
    parser.add_argument('--report', default='docs/reports/round6_mechanism.md')
    args = parser.parse_args(argv)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.perf_counter()
        conditions, compute = run_conditions(load_config(legacy.DEFAULT_CONFIG), cache, args.workers)
        report = render_report(conditions, runtime=time.perf_counter() - start,
                               compute_seconds=compute, workers=args.workers)
        Path(args.report).write_text(report)
        print('\n'.join(analyze(conditions)[2]), flush=True)


if __name__ == '__main__':
    main()
