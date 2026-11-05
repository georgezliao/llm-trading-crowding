"""Registered round-5 grid, diagnostic scales, and joint-episode rules.

Only public records and targets enter the reused observer fitting path. Private
forecast moments are computed in simulation workers and used only in the report.
Caches are local trusted pickle files, atomically replaced after each condition.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np
import yaml

from herdlab.evaluation.metrics import ConfidenceInterval
from herdlab.experiments.experiment_zero_round3 import EpisodeSample, _format_ci, _finite_interval
from herdlab.experiments.experiment_zero_round4 import (
    TRAINING_SEEDS, TEST_SEEDS, GRU_SEEDS, LABEL_NAMES, _run_condition,
    joint_signal_degradation_ci,
)
from herdlab.experiments.run_episode import load_config, run_episode
from herdlab.inference.targets import forced_selling_flow

BETAS = (0.1, 0.25, 0.5, 1.0)
FAMILIES = ('momentum', 'reversion', 'linear', 'gbdt', 'mlp')
DEFAULT_REPORT = Path('docs/reports/experiment_zero_round5.md')


def condition_configs(base):
    """Return the 13 main and four stress cells in a stable registered order."""
    conditions = {}

    def add(name, mode, beta, rho, drawdown=None):
        config = deepcopy(base)
        config['agents']['learned_overlap_mode'] = 'none'
        config['agents']['private_signal'].update(mode=mode, beta=beta, strength=rho)
        if drawdown is not None:
            config['agents']['risk']['max_drawdown'] = drawdown
        conditions[name] = config

    add('off', 'off', 1.0, 0.0)
    for beta in BETAS:
        add(f'beta={beta:g}/noise', 'noise_only', beta, 0.0)
        for rho in (0.25, 0.5):
            add(f'beta={beta:g}/rho={rho:g}', 'signal', beta, rho)
    for rho in (0.25, 0.5):
        for drawdown in (0.05, 0.10):
            add(f'stress/rho={rho:g}/drawdown={drawdown:g}', 'signal', 0.25, rho, drawdown)
    return conditions


def forecast_moments(base, signal, counts):
    """Per family [n, sum(base), sum(base²), sum(signal), sum(signal²)]."""
    if base is None or signal is None or base.shape != signal.shape:
        raise ValueError('forecast diagnostics must be present and aligned')
    if sum(counts) != base.shape[1]:
        raise ValueError('family counts must match forecast columns')
    output = []
    start = 0
    for count in counts:
        b = base[:, start:start + count].ravel()
        s = signal[:, start:start + count].ravel()
        output.append([b.size, b.sum(), np.dot(b, b), s.sum(), np.dot(s, s)])
        start += count
    return np.asarray(output, dtype=float)


def scale_ratios(moments):
    pooled = np.sum(moments, axis=0)
    n, b, b2, s, s2 = pooled.T
    with np.errstate(divide='ignore', invalid='ignore'):
        bv = np.maximum(b2 / n - (b / n)**2, 0)
        sv = np.maximum(s2 / n - (s / n)**2, 0)
        return np.sqrt(sv / bv)


def scale_intervals(moments, *, rng, n_resamples=1000):
    estimate = scale_ratios(moments)
    draws = np.asarray([scale_ratios(moments[rng.integers(len(moments), size=len(moments))])
                        for _ in range(n_resamples)])
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return tuple(ConfidenceInterval(float(e), float(l), float(h))
                 for e, l, h in zip(estimate, low, high, strict=True))


def _simulate_one(task):
    config, seed = task
    public, private = run_episode(config, seed)
    sample = EpisodeSample(public, private.agent_net_flow,
                           forced_selling_flow(private.orders, private.forced), seed)
    counts = [config['agents'][family]['count'] for family in FAMILIES]
    return sample, forecast_moments(private.base_forecasts, private.signal_terms, counts)


def simulate_with_scales(config, seeds, workers):
    tasks = [(config, seed) for seed in seeds]
    if workers == 1:
        rows = list(map(_simulate_one, tasks))
    else:
        with multiprocessing.Pool(workers) as pool:
            rows = pool.map(_simulate_one, tasks)
    return tuple(row[0] for row in rows), np.stack([row[1] for row in rows])


def contrast_ci(first, second, *, rng, n_resamples=1000):
    """Gain(first) minus gain(second), with the same episode draws in both."""
    if tuple(first['seeds']) != tuple(second['seeds']):
        raise ValueError('joint contrasts require identical ordered test seeds')
    if any(np.unique(np.concatenate(cell['labels'])).size < 2 for cell in (first, second)):
        return ConfidenceInterval(np.nan, np.nan, np.nan)
    return joint_signal_degradation_ci(
        first['labels'], first['gru'], first['trivial'],
        second['labels'], second['gru'], second['trivial'],
        rng=rng, n_resamples=n_resamples,
    )


def effect_rule(interval, *, information=False, informative=True):
    if not informative:
        return 'uninformative'
    if not _finite_interval(interval):
        return 'undefined'
    if information:
        if interval.upper < 0:
            return 'private information makes inference harder'
        if interval.lower > 0:
            return 'private information makes inference easier'
        return 'no directional information effect established'
    return 'noise degrades inference' if interval.lower > 0 else 'noise degradation not established'


def replication_rule(interval, base_rate, *, forced_events=None):
    flags = []
    if base_rate < 0.01:
        flags.append('uninformative')
    if forced_events is not None and forced_events < 30:
        flags.append('underpowered')
    if flags:
        return '; '.join(flags)
    if not _finite_interval(interval):
        return 'undefined; observer advantage cannot be established'
    if interval.lower <= 0:
        return 'consistent with strategy replication'
    return 'not consistent with strategy replication by the registered rule'


def round4_reading_rule(noise, information, *, informative=True):
    if not informative:
        return 'uninformative'
    if not (_finite_interval(noise) and _finite_interval(information)):
        return 'undefined'
    if noise.lower > 0 and information.upper >= 0:
        return 'not supported'
    return 'not rejected by this rule (does not establish support)'


def swamping_rule(noise):
    candidates = [beta for beta, interval in noise.items()
                  if _finite_interval(interval) and interval.lower <= 0 <= interval.upper]
    return f'smallest beta whose noise effect CI includes 0: {min(candidates):g}' if candidates else 'none'


def cache_key(config, index):
    # Changes to simulator, observer, or runner implementation invalidate caches.
    digest = hashlib.sha256()
    for path in sorted(Path('herdlab').rglob('*.py')):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(dict(config=config, index=index, train=TRAINING_SEEDS,
                                  test=TEST_SEEDS, gru=GRU_SEEDS), sort_keys=True).encode())
    return digest.hexdigest()


def cached_condition(path, key, compute):
    if path.exists():
        with path.open('rb') as handle:
            stored = pickle.load(handle)
        if stored['key'] == key:
            return stored['value'], True
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        pickle.dump({'key': key, 'value': value}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temporary.replace(path)
    return value, False


def run_condition(config, index, workers, gru_workers):
    started = time.perf_counter()
    training, train_moments = simulate_with_scales(config, TRAINING_SEEDS, workers)
    test, test_moments = simulate_with_scales(config, TEST_SEEDS, workers)
    result, bundle = _run_condition(
        config, workers=workers, gru_workers=gru_workers, horizon=10,
        threshold_quantile=0.95, return_lookback=10, false_alarm_rate=0.05,
        bootstrap_resamples=1000, bootstrap_seed=9000 + index * 100_000,
        lightgbm_seed=5000, gru_seeds=GRU_SEEDS, gru_kwargs=None,
        samples=(training, test),
    )
    contrasts = {}
    rate_cis = {}
    for j, label in enumerate(LABEL_NAMES):
        tasks = bundle.labels[label].test_tasks
        predictions = bundle.labels[label].predictions
        labels = tuple(np.asarray(task.labels, dtype=bool) for task in tasks)
        contrasts[label] = dict(seeds=TEST_SEEDS, labels=labels,
                                gru=predictions['gru_ensemble'],
                                trivial=predictions['trivial_reference'])
        counts = np.array([x.sum() for x in labels])
        sizes = np.array([len(x) for x in labels])
        rng = np.random.default_rng(2_000_000 + index * 10 + j)
        draws = rng.integers(len(labels), size=(1000, len(labels)))
        lo, hi = np.quantile(counts[draws].sum(axis=1) / sizes[draws].sum(axis=1), [0.025, 0.975])
        rate_cis[label] = ConfidenceInterval(counts.sum() / sizes.sum(), lo, hi)
    return dict(result=result, contrasts=contrasts, rate_cis=rate_cis,
                scales=scale_intervals(np.concatenate((train_moments, test_moments)),
                                       rng=np.random.default_rng(3_000_000 + index)),
                seconds=time.perf_counter() - started)


def render_report(base, configs, cells, *, runtime_seconds, workers, gru_workers):
    lines = ['# Experiment Zero, round 5: noise and information', '',
             '**Synthetic-simulator disclaimer:** All results are from the uncalibrated synthetic HerdLab simulator, not real markets, funds, or live trading.', '',
             '## Configuration, seeds, and runtime', '',
             'This tests a post-hoc noise hypothesis formed after round 4. Amendment 2 approved the 81.06-minute estimate and raises the combined R5-3/R5-4 allowance to approximately 120 minutes. R5-4 is not run here.', '',
             f'Measured invocation wall time: {runtime_seconds / 60:.2f} minutes. Sum of cached condition compute times: {sum(c["seconds"] for c in cells.values()) / 60:.2f} minutes. Parallelism: {workers} simulator workers, {gru_workers} GRU workers.', '',
             'Each of 17 conditions uses 200 training episodes (8000–8199) and 50 test episodes (8500–8549), 500 steps each. Horizon 10; separate training-only 0.95 thresholds for default (primary) and forced-flow (secondary). Return lookback 10; unchanged round-4 observer defaults; GRU seeds 6000–6002; LightGBM/replicator RNG seeds 5000 + label index.', '',
             'All CIs are 95% percentile intervals from 1,000 whole-episode resamples. Cross-condition gain differences resample the same ordered test seeds jointly. Metric RNG seeds: 9000 + 100000 × condition index + round-4 helper offsets; contrasts: 4000000 + 100 × label index + contrast index; scale: 3000000 + condition index; test base rate: 2000000 + 10 × condition index + label index. Indices start at zero in displayed order.', '',
             'Finished conditions are atomically cached under `runs/round5_cache/`, including predictions needed for joint contrasts, diagnostic CIs, and compute times. Cache fingerprints cover resolved configuration, seeds, and Python source; mismatches are recomputed.', '',
             '### Base configuration', '', '```yaml', yaml.safe_dump(base, sort_keys=False).rstrip(), '```', '',
             'Every condition forces learned_overlap_mode=none. Overrides below are the complete changes to the base configuration.', '',
             '| Condition | Signal mode | Beta | Rho | Max drawdown | Compute minutes |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for name, config in configs.items():
        signal = config['agents']['private_signal']
        lines.append(f'| {name} | {signal["mode"]} | {signal["beta"]:g} | {signal["strength"]:g} | {config["agents"]["risk"]["max_drawdown"]:g} | {cells[name]["seconds"] / 60:.2f} |')
    lines += ['', '## Thresholds and base rates', '',
              '| Condition | Label | Threshold | Train positives / labels | Train rate | Test positives / labels | Test rate (95% CI) |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, cell in cells.items():
        for label, value in cell['result'].labels.items():
            train, test = value.training_events, value.evaluation.events
            lines.append(f'| {name} | {label} | {value.threshold:.12g} | {train.event_count}/{train.label_count} | {train.base_rate:.6f} | {test.event_count}/{test.label_count} | {_format_ci(cell["rate_cis"][label])} |')
    lines += ['', '## Forecast scale ratios', '',
              'std(signal term) / std(base forecast), pooled across all decision times and agents within each family over the 250 training and test episodes, including warm-up and risk-overridden decisions. CIs resample whole episodes and recompute the pooled standard deviations. These private diagnostics never enter an observer. Undefined ratios are N/A.', '',
              '| Condition | ' + ' | '.join(FAMILIES) + ' |', '| --- | ' + ' | '.join(['---:'] * 5) + ' |']
    for name, cell in cells.items():
        lines.append(f'| {name} | ' + ' | '.join(_format_ci(x) for x in cell['scales']) + ' |')
    lines += ['', '## Observer metrics', '', 'Entries are estimate [95% CI]. PR-AUC and recall are higher-is-better; Brier is lower-is-better. Base rates and power flags apply to every observer in a cell.']
    outcomes = []
    for name, cell in cells.items():
        for label, value in cell['result'].labels.items():
            ev = value.evaluation
            lines += ['', f'### {name} / {label}', '',
                      f'Test base rate: {ev.events.base_rate:.6f}; positive labels: {ev.events.event_count}.', '',
                      '| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |', '| --- | ---: | ---: | ---: |']
            for observer, metrics in ev.metrics.items():
                lines.append(f'| {observer} | ' + ' | '.join(_format_ci(metrics[m]) for m in ('pr_auc', 'brier', 'recall_at_false_alarm_rate')) + ' |')
            lines += ['', 'GRU seed min–max PR-AUC: ' + str(ev.gru_pr_auc_range) + '.', '',
                      '| GRU ensemble minus | Paired PR-AUC difference (95% CI) |', '| --- | ---: |']
            for comparator, interval in ev.paired_differences.items():
                lines.append(f'| {comparator} | {_format_ci(interval)} |')
            lines += ['', 'Training: ' + '; '.join(f'{seed}: {detail}' for seed, detail in value.gru_training_details.items()) + '.']
            status = replication_rule(ev.paired_differences['replicator'], ev.events.base_rate,
                                      forced_events=ev.events.event_count if label == 'forced_flow' else None)
            outcomes.append(f'Replication / informativeness / power — {name}/{label}: {status}.')
    lines += ['', '## Joint gain contrasts', '',
              'Gain = GRU ensemble PR-AUC minus trivial PR-AUC. Noise effect = gain(off) − gain(noise). Information effect = gain(signal) − gain(noise). The primary interpretation uses default labels; secondary forced-flow results retain power warnings.', '',
              '| Label | Contrast | Gain difference (95% CI) | Test base rates (first / second) | Outcome |', '| --- | --- | ---: | ---: | --- |']
    for j, label in enumerate(LABEL_NAMES):
        noise_intervals, information_intervals = {}, {}
        k = 0
        def informative(names):
            return all(cells[n]['result'].labels[label].evaluation.events.base_rate >= 0.01 for n in names)
        for beta in BETAS:
            noise_name = f'beta={beta:g}/noise'
            for rho in (None, 0.25, 0.5):
                first = 'off' if rho is None else f'beta={beta:g}/rho={rho:g}'
                interval = contrast_ci(cells[first]['contrasts'][label], cells[noise_name]['contrasts'][label],
                                       rng=np.random.default_rng(4_000_000 + 100 * j + k))
                k += 1
                if rho is None:
                    noise_intervals[beta] = interval
                else:
                    information_intervals[beta, rho] = interval
                status = effect_rule(interval, information=rho is not None,
                                     informative=informative((first, noise_name)))
                if label == 'forced_flow' and any(cells[n]['result'].labels[label].evaluation.events.event_count < 30 for n in (first, noise_name)):
                    status = 'underpowered; ' + status
                title = f'noise beta={beta:g}' if rho is None else f'information beta={beta:g}, rho={rho:g}'
                rates = [cells[n]['result'].labels[label].evaluation.events.base_rate for n in (first, noise_name)]
                lines.append(f'| {label} | {title} | {_format_ci(interval)} | {rates[0]:.6f} / {rates[1]:.6f} | {status} |')
                outcomes.append(f'{label} / {title}: {status}.')
        if label == 'default':
            outcomes.append('Round-4 private information degrades inference reading: ' + round4_reading_rule(
                noise_intervals[1.0], information_intervals[1.0, 0.5],
                informative=informative(('off', 'beta=1/noise', 'beta=1/rho=0.5'))) + '.')
            outcomes.append('Swamping — ' + swamping_rule(noise_intervals) + '.')
    lines += ['', '## Pre-registered rule outcomes (verbatim)', '']
    lines += ['- ' + outcome for outcome in outcomes]
    lines += ['', '## Scope and open questions', '',
              'No simulator equation, label definition, threshold rule, scientific setting, observer hyperparameter, or registered seed count was changed. Observers receive public records and targets only. Positive-label counts include overlapping future windows; uncertainty resamples episodes, not windows. Undefined PR-AUC is N/A, not zero. Informativeness (<1%) and forced-flow power (<30 positives) flags limit interpretation. The forced-selling H1 rule belongs to R5-4 and is not evaluated here.', '',
              'No research-design deviations or open questions. This experiment cannot establish real-market validity or identify hidden real fund positions.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--gru-workers', type=int, default=3)
    parser.add_argument('--cache-dir', type=Path, default=Path('runs/round5_cache'))
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    base = dict(load_config('configs/experiment_zero.yaml'))
    configs = condition_configs(base)
    cells = {}
    print('Approved estimate 81.06 minutes; Amendment 2 combined allowance ~120 minutes.', flush=True)
    for index, (name, config) in enumerate(configs.items()):
        print(f'Starting {index + 1}/17: {name}', flush=True)
        cells[name], hit = cached_condition(
            args.cache_dir / f'condition_{index:02d}.pkl', cache_key(config, index),
            lambda: run_condition(config, index, args.workers, args.gru_workers))
        print(f'Finished {name}; cache_hit={hit}; compute_minutes={cells[name]["seconds"] / 60:.2f}', flush=True)
    report = render_report(base, configs, cells, runtime_seconds=time.perf_counter() - started,
                           workers=args.workers, gru_workers=args.gru_workers)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding='utf-8')
    print(f'COMPLETE: {args.report}; wall_minutes={(time.perf_counter() - started) / 60:.2f}', flush=True)


if __name__ == '__main__':
    main()
