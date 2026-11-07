"""Registered round-6 per-family signal scaling grid; simulator evidence only.

Family constants come from off-condition training seeds only. Forecast moments
stay in the simulation workers and the report, never in observer inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import yaml

from herdlab.experiments.experiment_zero_round3 import _format_ci
from herdlab.experiments.experiment_zero_round4 import LABEL_NAMES, TRAINING_SEEDS
from herdlab.experiments.experiment_zero_round5 import (
    FAMILIES, cache_key, cached_condition, condition_configs, contrast_ci, effect_rule,
    replication_rule, run_condition, simulate_with_scales, swamping_rule,
)
from herdlab.experiments.run_episode import load_config

BETAS = (0.25, 0.5, 1.0)
RHOS = (0.25, 0.5)
RUNTIME_ESTIMATE_MINUTES = 57
DEFAULT_REPORT = Path('docs/reports/round6_scaled_grid.md')
# Default-label information outcomes at the same numeric beta, from
# docs/reports/experiment_zero_round5.md.
ROUND5_INFORMATION = {
    (0.25, 0.25): 'no directional information effect established',
    (0.25, 0.5): 'private information makes inference harder',
    (0.5, 0.25): 'private information makes inference harder',
    (0.5, 0.5): 'private information makes inference harder',
    (1.0, 0.25): 'private information makes inference easier',
    (1.0, 0.5): 'private information makes inference easier',
}


def family_constants(moments):
    """Pooled base-forecast standard deviation per family from episode moments."""
    n, base, base_squared, _, _ = np.sum(moments, axis=0).T
    variance = np.maximum(base_squared / n - (base / n) ** 2, 0.0)
    if not np.all(variance > 0.0):
        raise ValueError('every family needs a nonzero base-forecast variance')
    return {family: float(value) for family, value in zip(FAMILIES, np.sqrt(variance), strict=True)}


def scaled_configs(base, constants):
    """Return the off condition followed by the nine registered scaled cells."""
    conditions = {'off': condition_configs(base)['off']}

    def add(name, mode, beta, rho):
        config = deepcopy(conditions['off'])
        config['agents']['private_signal'].update(
            mode=mode, beta=beta, strength=rho, scale='family', family_scales=dict(constants))
        conditions[name] = config

    for beta in BETAS:
        add(f'beta={beta:g}/noise', 'noise_only', beta, 0.0)
        for rho in RHOS:
            add(f'beta={beta:g}/rho={rho:g}', 'signal', beta, rho)
    return conditions


def direction_change(current, previous):
    """Describe a sign change between two information outcomes without testing it."""
    def sign(outcome):
        return {'harder': -1, 'easier': 1}.get(outcome.split()[-1], 0)

    if 'makes' not in current and 'no directional' not in current:
        return 'not comparable'
    return 'same direction' if sign(current) == sign(previous) else 'direction changed'


def _events(cell, label):
    return cell['result'].labels[label].evaluation.events


def render_report(base, constants, configs, cells, *, runtime_seconds, workers, gru_workers):
    lines = ['# Round 6: per-family signal scaling', '',
             '**Synthetic-simulator disclaimer:** All results are from the uncalibrated synthetic HerdLab simulator, not real markets, funds, or live trading.', '',
             '## Configuration, seeds, and runtime', '',
             f'Runtime estimate before running: {RUNTIME_ESTIMATE_MINUTES} minutes, inside the ~130-minute limit once added to the {28.80:.2f}-minute population run. Measured invocation wall time: {runtime_seconds / 60:.2f} minutes. Sum of condition compute times: {sum(c["seconds"] for c in cells.values()) / 60:.2f} minutes. Workers: {workers} simulator, {gru_workers} GRU.', '',
             'This step was implemented by the research lead after the implementation agent hit a usage limit. The design is unchanged from the pre-registration.', '',
             'Each condition uses 200 training episodes (8000–8199) and 50 test episodes (8500–8549), 500 steps each. Horizon 10, separate training-only 0.95 quantile per condition, GRU seeds 6000–6002. CIs are 95% percentile intervals from 1,000 whole-episode resamples, and cross-condition contrasts resample the same test seeds jointly.', '',
             '### Family constants', '',
             'Pooled standard deviation of each family\'s base forecast in the off condition, on training seeds only, including warm-up decisions. The signal term is β × constant × s.', '',
             '| Family | Constant |', '| --- | ---: |']
    lines += [f'| {family} | {constants[family]:.6g} |' for family in FAMILIES]
    lines += ['', '### Base configuration', '', '```yaml', yaml.safe_dump(base, sort_keys=False).rstrip(), '```', '',
              '## Thresholds and base rates', '',
              '| Condition | Label | Threshold | Test positives / labels | Test rate (95% CI) |', '| --- | --- | ---: | ---: | ---: |']
    for name, cell in cells.items():
        for label, value in cell['result'].labels.items():
            test = value.evaluation.events
            lines.append(f'| {name} | {label} | {value.threshold:.12g} | {test.event_count}/{test.label_count} | {_format_ci(cell["rate_cis"][label])} |')
    lines += ['', '## Forecast scale ratios', '',
              'std(signal term) / std(base forecast), pooled over the 250 episodes of each condition. Base forecasts respond to the signal through prices, so ratios need not equal β exactly.', '',
              '| Condition | ' + ' | '.join(FAMILIES) + ' |', '| --- | ' + ' | '.join(['---:'] * 5) + ' |']
    lines += [f'| {name} | ' + ' | '.join(_format_ci(x) for x in cell['scales']) + ' |' for name, cell in cells.items()]
    lines += ['', '## Observer metrics', '']
    outcomes = []
    for name, cell in cells.items():
        for label, value in cell['result'].labels.items():
            ev = value.evaluation
            lines += [f'### {name} / {label}', '', f'Test base rate: {ev.events.base_rate:.6f}; positive labels: {ev.events.event_count}.', '',
                      '| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |', '| --- | ---: | ---: | ---: |']
            lines += [f'| {observer} | ' + ' | '.join(_format_ci(metrics[m]) for m in ('pr_auc', 'brier', 'recall_at_false_alarm_rate')) + ' |'
                      for observer, metrics in ev.metrics.items()]
            lines += ['', 'GRU seed min–max PR-AUC: ' + str(ev.gru_pr_auc_range) + '.', '']
            lines += [f'GRU ensemble minus {comparator}: {_format_ci(interval)}.' for comparator, interval in ev.paired_differences.items()]
            lines.append('')
            status = replication_rule(ev.paired_differences['replicator'], ev.events.base_rate,
                                      forced_events=ev.events.event_count if label == 'forced_flow' else None)
            outcomes.append(f'Replication / informativeness / power — {name}/{label}: {status}.')
    lines += ['## Joint gain contrasts', '',
              'Gain = GRU ensemble PR-AUC minus trivial PR-AUC. Noise effect = gain(off) − gain(noise). Information effect = gain(signal) − gain(noise). Compared with round 5 is descriptive, at the same numeric β, and that β meant a different size of signal in round 5.', '',
              '| Label | Contrast | Gain difference (95% CI) | Outcome | Compared with round 5 |', '| --- | --- | ---: | --- | --- |']
    for j, label in enumerate(LABEL_NAMES):
        noise_intervals = {}
        k = 0
        for beta in BETAS:
            noise_name = f'beta={beta:g}/noise'
            for rho in (None, *RHOS):
                first = 'off' if rho is None else f'beta={beta:g}/rho={rho:g}'
                interval = contrast_ci(cells[first]['contrasts'][label], cells[noise_name]['contrasts'][label],
                                       rng=np.random.default_rng(5_000_000 + 100 * j + k))
                k += 1
                names = (first, noise_name)
                informative = all(_events(cells[n], label).base_rate >= 0.01 for n in names)
                status = effect_rule(interval, information=rho is not None, informative=informative)
                if label == 'forced_flow' and any(_events(cells[n], label).event_count < 30 for n in names):
                    status = 'underpowered; ' + status
                if rho is None:
                    noise_intervals[beta] = interval
                compared = '—'
                if rho is not None and label == 'default':
                    compared = direction_change(status, ROUND5_INFORMATION[beta, rho])
                title = f'noise beta={beta:g}' if rho is None else f'information beta={beta:g}, rho={rho:g}'
                lines.append(f'| {label} | {title} | {_format_ci(interval)} | {status} | {compared} |')
                outcomes.append(f'{label} / {title}: {status}.')
        if label == 'default':
            outcomes.append('Swamping — ' + swamping_rule(noise_intervals) + '.')
    lines += ['', '## Pre-registered rule outcomes (verbatim)', '']
    lines += ['- ' + outcome for outcome in outcomes]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--gru-workers', type=int, default=3)
    parser.add_argument('--cache-dir', type=Path, default=Path('runs/round6_scaled_cache'))
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    base = dict(load_config('configs/experiment_zero.yaml'))
    off = condition_configs(base)['off']
    constants, _ = cached_condition(
        args.cache_dir / 'constants.pkl', cache_key(off, 'constants'),
        lambda: family_constants(simulate_with_scales(off, TRAINING_SEEDS, args.workers)[1]))
    print(f'Family constants: {constants}', flush=True)
    configs = scaled_configs(base, constants)
    cells = {}
    for index, (name, config) in enumerate(configs.items()):
        print(f'Starting {index + 1}/{len(configs)}: {name}', flush=True)
        cells[name], hit = cached_condition(
            args.cache_dir / f'condition_{index:02d}.pkl', cache_key(config, index),
            lambda: run_condition(config, index, args.workers, args.gru_workers))
        print(f'Finished {name}; cache_hit={hit}; compute_minutes={cells[name]["seconds"] / 60:.2f}', flush=True)
    report = render_report(base, constants, configs, cells, runtime_seconds=time.perf_counter() - started,
                           workers=args.workers, gru_workers=args.gru_workers)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding='utf-8')
    print(f'COMPLETE: {args.report}; wall_minutes={(time.perf_counter() - started) / 60:.2f}', flush=True)


if __name__ == '__main__':
    main()
