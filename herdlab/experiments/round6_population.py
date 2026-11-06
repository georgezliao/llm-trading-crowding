"""Registered round-6 population decomposition; simulator evidence only."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import yaml

from herdlab.experiments.experiment_zero_round3 import _finite_interval, _format_ci
from herdlab.experiments.experiment_zero_round4 import LABEL_NAMES
from herdlab.experiments.experiment_zero_round5 import (
    cache_key, cached_condition, contrast_ci, replication_rule, run_condition,
)
from herdlab.experiments.round5_reconcile import reproduction_status
from herdlab.experiments.round6_populations import FAMILIES, population_configs

RUNTIME_ESTIMATE_MINUTES = 45
SCALED_GRID_RESERVE_MINUTES = 60
RUNTIME_LIMIT_MINUTES = 130


def eligibility(base_rates, forced_events=None):
    flags = []
    if any(rate < .01 for rate in base_rates):
        flags.append('uninformative')
    if forced_events is not None and any(count < 30 for count in forced_events):
        flags.append('underpowered')
    return '; '.join(flags)


def family_rule(interval, *, base_rates, forced_events=None):
    flag = eligibility(base_rates, forced_events)
    if flag:
        return flag
    if not _finite_interval(interval):
        return 'undefined'
    if interval.lower > 0:
        return 'contributes to visibility'
    if interval.upper < 0:
        return 'suppresses visibility'
    return 'no directional family effect established'


def gap_rule(interval, gap, *, base_rates, forced_events=None):
    flag = eligibility(base_rates, forced_events)
    if flag:
        return flag
    if not (_finite_interval(interval) and _finite_interval(gap)):
        return 'undefined'
    # This registered rule uses point estimates, not significance of the family CI.
    explains = interval.estimate * gap.estimate > 0 and abs(interval.estimate) >= abs(gap.estimate) / 2
    return 'explains the population gap' if explains else 'does not explain the population gap'


def interaction_rule(reversion, gbdt, gap, *, base_rates, forced_events=None):
    flag = eligibility(base_rates, forced_events)
    if flag:
        return flag
    if not all(_finite_interval(x) for x in (reversion, gbdt, gap)):
        return 'undefined'
    statuses = [gap_rule(x, gap, base_rates=base_rates) for x in (reversion, gbdt)]
    if all(x == 'does not explain the population gap' for x in statuses) and gap.lower > 0:
        return 'the gap needs both changes or something else in the v2 mix'
    return 'interaction not established by the registered rule'


def render_report(configs, cells, *, runtime_seconds, workers, gru_workers):
    lines = ['# Round 6: population decomposition', '',
        '**Synthetic-simulator disclaimer:** Uncalibrated synthetic HerdLab results only; no evidence about real markets, funds, or live trading.', '',
        '## Configuration, seeds, and runtime', '',
        f'Pre-run estimate: {RUNTIME_ESTIMATE_MINUTES} minutes for R6-2, plus a {SCALED_GRID_RESERVE_MINUTES}-minute planning reserve for R6-5; combined 105 minutes versus the approximately 130-minute limit. R6-5 is not run here.', '',
        f'Measured invocation through analysis: {runtime_seconds / 60:.2f} minutes. Retained condition compute: {sum(c["seconds"] for c in cells.values()) / 60:.2f} minutes. Workers: {workers} simulation, {gru_workers} GRU.', '',
        'Each population uses 200 train seeds 8000–8199 and 50 test seeds 8500–8549, 500 steps each. Signals off; horizon 10; separate training-only 0.95 quantile thresholds for default (primary) and forced_flow (secondary). Return lookback 10; GRU seeds 6000–6002; LightGBM and replicator seeds 5000 + label index. Observer hyperparameters are unchanged from round 5.', '',
        'All intervals are 95% percentile CIs with 1,000 whole-test-episode resamples. Shared test seeds are resampled jointly for gain contrasts. Gain = GRU ensemble PR-AUC minus trivial PR-AUC. Metric seed: 9000 + 100000 × population index + round-4 helper offsets; base-rate seed: 2000000 + 10 × population index + label index; contrast seed: 6000000 + 100 × label index + contrast index. Indices start at zero in displayed order.', '',
        'The unchanged round-5 runner also computes ancillary logistic and individual-GRU metrics; the four registered observers are reported below. Unused forecast-scale diagnostics are not observer inputs. Atomic condition caches live in runs/round6_population_cache; fingerprints cover resolved configs, seed sets and Python source. The baseline is freshly fitted unless a matching round-6 cache exists.', '',
        '## Baseline reproducibility', '']
    baseline = cells['baseline']['result'].labels['default'].evaluation.metrics['lightgbm']['pr_auc']
    lines += [f'LightGBM default PR-AUC: {_format_ci(baseline)}; reference 0.9510; {reproduction_status(baseline.estimate, .9510)} (four decimal places).', '',
        '## Base rates and thresholds', '',
        '| Population | Label | Threshold | Train positives / labels | Train rate | Test positives / labels | Test rate (95% CI) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    outcomes = []
    for name, cell in cells.items():
        for label, value in cell['result'].labels.items():
            train, test = value.training_events, value.evaluation.events
            lines.append(f'| {name} | {label} | {value.threshold:.12g} | {train.event_count}/{train.label_count} | {train.base_rate:.6f} | {test.event_count}/{test.label_count} | {_format_ci(cell["rate_cis"][label])} |')
    lines += ['', '## Observer metrics', '', 'PR-AUC and recall are higher-is-better; Brier is lower-is-better. Recall uses FAR <= 0.05.']
    for name, cell in cells.items():
        for label, value in cell['result'].labels.items():
            ev = value.evaluation
            lines += ['', f'### {name} / {label}', '',
                '| Observer | PR-AUC | Brier | Recall at FAR <= 0.05 |', '| --- | ---: | ---: | ---: |']
            for observer in ('trivial_reference', 'lightgbm', 'replicator', 'gru_ensemble'):
                lines.append(f'| {observer} | ' + ' | '.join(_format_ci(ev.metrics[observer][m]) for m in ('pr_auc', 'brier', 'recall_at_false_alarm_rate')) + ' |')
            lines += ['', '| GRU ensemble minus | Paired PR-AUC difference (95% CI) |', '| --- | ---: |']
            for comparator, interval in ev.paired_differences.items():
                lines.append(f'| {comparator} | {_format_ci(interval)} |')
            lines += ['', f'GRU seed PR-AUC range: {ev.gru_pr_auc_range}.', '',
                      'Training: ' + '; '.join(f'{seed}: {detail}' for seed, detail in value.gru_training_details.items()) + '.']
            outcomes.append(f'{name}/{label}: ' + replication_rule(ev.paired_differences['replicator'], ev.events.base_rate, forced_events=ev.events.event_count if label == 'forced_flow' else None))
    lines += ['', '## Joint gain contrasts', '',
        'Every contrast is gain(baseline) minus gain(other). Gap explanation requires the same point-estimate sign and at least half the absolute v2 contrast; it does not require a significant family CI. Interaction is the registered qualitative rule, not a factorial interaction estimate.', '',
        '| Label | Other population | Gain difference (95% CI) | Family rule | Gap rule |', '| --- | --- | ---: | --- | --- |']
    for j, label in enumerate(LABEL_NAMES):
        intervals = {name: contrast_ci(cells['baseline']['contrasts'][label], cell['contrasts'][label], rng=np.random.default_rng(6_000_000 + 100*j + k))
                     for k, (name, cell) in enumerate(list(cells.items())[1:])}
        def gates(names):
            events = [cells[n]['result'].labels[label].evaluation.events for n in names]
            return dict(base_rates=[e.base_rate for e in events], forced_events=[e.event_count for e in events] if label == 'forced_flow' else None)
        for family in FAMILIES:
            name = f'without_{family}'
            effect = family_rule(intervals[name], **gates(('baseline', name)))
            gap = gap_rule(intervals[name], intervals['round3_v2'], **gates(('baseline', name, 'round3_v2')))
            lines.append(f'| {label} | {name} | {_format_ci(intervals[name])} | {effect} | {gap} |')
            outcomes += [f'{label}/{family}: {effect}', f'{label}/{family}: {gap}']
        lines.append(f'| {label} | round3_v2 | {_format_ci(intervals["round3_v2"])} | population gap | — |')
        outcomes.append(f'{label}/interaction: ' + interaction_rule(intervals['without_reversion'], intervals['without_gbdt'], intervals['round3_v2'], **gates(('baseline', 'without_reversion', 'without_gbdt', 'round3_v2'))))
    lines += ['', '## Pre-registered rule outcomes (verbatim)', ''] + ['- ' + x for x in outcomes]
    lines += ['', '## Resolved configurations', '', 'Omissions scale configured q_max, order_limit and initial_cash by 30/24. Risk/leverage limits are unchanged. Configured aggregate capital and capacity match; realized dispersed capacities need not match exactly.']
    for name, config in configs.items():
        lines += ['', f'### {name}', '', f'Condition compute minutes: {cells[name]["seconds"] / 60:.2f}.', '', '```yaml', yaml.safe_dump(config, sort_keys=False).rstrip(), '```']
    lines += ['', '## Scope, deviations, and open questions', '',
        'No simulator equation, label, threshold, population, seed range or observer hyperparameter was changed. Private state is reduced to targets before observer fitting. Positive counts are overlapping future windows, not distinct liquidation episodes; bootstrap units are whole episodes. Rules are withheld for any required population below 1% base rate or, for forced flow, below 30 positives. No multiplicity adjustment was registered. Family removal also redistributes capacity, and the v2 mix changes remaining family counts; these results do not uniquely identify a causal family mechanism or hidden real-world positions.', '',
        'Implementation detail: the reused round-5 runner fits an ancillary logistic observer; it is excluded from the registered tables and decisions. No research-design deviations or open questions.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--gru-workers', type=int, default=3)
    parser.add_argument('--cache-dir', type=Path, default=Path('runs/round6_population_cache'))
    parser.add_argument('--report', type=Path, default=Path('docs/reports/round6_population.md'))
    args = parser.parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with (args.cache_dir / 'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A population run already holds the lock; refusing duplicate.')
        started = time.perf_counter()
        print('Estimate: 45 minutes; later scaled-grid reserve 60 minutes; combined limit ~130 minutes.', flush=True)
        configs, cells = population_configs(), {}
        for index, (name, config) in enumerate(configs.items()):
            if (time.perf_counter() - started) / 60 + SCALED_GRID_RESERVE_MINUTES > RUNTIME_LIMIT_MINUTES:
                args.report.write_text('# Round 6 population: runtime stop\n\nSynthetic simulator only. Runtime plus the scaled-grid reserve exceeds 130 minutes. Finished conditions remain cached; no episode counts were cut.\n')
                raise SystemExit('Runtime stop; resume from finished condition caches.')
            print(f'Starting {index + 1}/7: {name}', flush=True)
            cells[name], hit = cached_condition(args.cache_dir / f'condition_{index:02d}.pkl', cache_key(config, index), lambda: run_condition(config, index, args.workers, args.gru_workers))
            print(f'Finished {name}; cache_hit={hit}; compute_minutes={cells[name]["seconds"] / 60:.2f}', flush=True)
        report = render_report(configs, cells, runtime_seconds=time.perf_counter() - started, workers=args.workers, gru_workers=args.gru_workers)
        # Include contrast/report time in measured invocation time, without redoing bootstraps.
        elapsed = time.perf_counter() - started
        report += f'\nCompleted invocation including report analysis: {elapsed / 60:.2f} minutes.\n'
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f'COMPLETE: {args.report}; wall_minutes={elapsed / 60:.2f}', flush=True)


if __name__ == '__main__':
    main()
