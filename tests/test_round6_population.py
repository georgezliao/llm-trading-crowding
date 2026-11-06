"""Toy tests for the fixed round-6 population decision rules."""
import numpy as np
import pytest
from herdlab.evaluation.metrics import ConfidenceInterval as CI
from herdlab.experiments.round6_population import family_rule, gap_rule, interaction_rule

POS = CI(.2, .1, .3)
NEG = CI(-.2, -.3, -.1)
SMALL = CI(.04, -.01, .09)
RATES = dict(base_rates=[.05, .05, .05, .05])


def test_family_directions_and_zero_boundaries():
    assert family_rule(POS, **RATES) == 'contributes to visibility'
    assert family_rule(NEG, **RATES) == 'suppresses visibility'
    for interval in (SMALL, CI(.1, 0, .2), CI(-.1, -.2, 0)):
        assert family_rule(interval, **RATES) == 'no directional family effect established'


def test_explains_gap_uses_sign_and_half_point_estimate_including_equality():
    for interval, gap in [(CI(.1, -.1, .3), POS), (CI(-.1, -.3, .1), NEG)]:
        assert gap_rule(interval, gap, **RATES) == 'explains the population gap'
    for interval, gap in [(SMALL, POS), (POS, NEG), (NEG, POS), (POS, CI(0, -.1, .1))]:
        assert gap_rule(interval, gap, **RATES) == 'does not explain the population gap'


def test_interaction_requires_both_small_family_effects_and_positive_gap_ci():
    assert interaction_rule(SMALL, SMALL, POS, **RATES) == 'the gap needs both changes or something else in the v2 mix'
    for rev, gbdt, gap in [(POS, SMALL, POS), (SMALL, POS, POS), (SMALL, SMALL, NEG), (SMALL, SMALL, CI(.2, 0, .3))]:
        assert interaction_rule(rev, gbdt, gap, **RATES) == 'interaction not established by the registered rule'


@pytest.mark.parametrize('function,args', [(family_rule, (POS,)), (gap_rule, (POS, POS)), (interaction_rule, (SMALL, SMALL, POS))])
def test_gates_and_undefined(function, args):
    assert function(*args, base_rates=[.01, .009]) == 'uninformative'
    assert function(*args, base_rates=[.01, .05], forced_events=[30, 29]) == 'underpowered'
    assert function(*args, base_rates=[.009], forced_events=[29]) == 'uninformative; underpowered'
    assert function(*args, base_rates=[.01], forced_events=[30]) not in ('uninformative', 'underpowered')
    assert function(CI(np.nan, np.nan, np.nan), *args[1:], **RATES) == 'undefined'


def test_report_covers_populations_labels_and_verbatim_rules(monkeypatch):
    from herdlab.experiments import round6_population as r6
    from herdlab.experiments.experiment_zero_round3 import EventSummary
    from herdlab.experiments.experiment_zero_round4 import ConditionResult, LabelResult, EvaluationResult
    monkeypatch.setattr(r6, 'contrast_ci', lambda *args, **kwargs: POS)
    configs = r6.population_configs()
    events = EventSummary(50, 1000)
    metrics = {name: {metric: POS for metric in ('pr_auc', 'brier', 'recall_at_false_alarm_rate')}
               for name in ('trivial_reference', 'lightgbm', 'replicator', 'gru_ensemble')}
    metrics['lightgbm']['pr_auc'] = CI(.95104, .92, .97)
    ev = EvaluationResult(events, metrics, {n: POS for n in ('replicator', 'lightgbm', 'trivial_reference')}, (.1, .3))
    result = ConditionResult({label: LabelResult(.1, events, ev, {6000: 'toy'}) for label in r6.LABEL_NAMES})
    cells = {name: dict(result=result, seconds=1, rate_cis={label: POS for label in r6.LABEL_NAMES},
                       contrasts={label: {} for label in r6.LABEL_NAMES}) for name in configs}
    report = r6.render_report(configs, cells, runtime_seconds=7, workers=10, gru_workers=3)
    assert 'Synthetic-simulator disclaimer' in report
    assert 'reference 0.9510; PASS' in report
    assert 'whole-test-episode' in report
    for name in configs:
        for label in r6.LABEL_NAMES:
            assert f'### {name} / {label}' in report
            assert f'- {name}/{label}: not consistent with strategy replication' in report
    for family in r6.FAMILIES:
        assert f'- default/{family}: contributes to visibility' in report
        assert f'- default/{family}: explains the population gap' in report
    assert '- default/interaction: interaction not established by the registered rule' in report
