"""Guard the registered crossing, historical comparisons, and target boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from herdlab.evaluation.metrics import ConfidenceInterval
from herdlab.experiments import round5_reconcile as reconcile


def test_registered_populations_and_seed_ranges():
    configs = reconcile.population_configs()
    families = ("momentum", "reversion", "linear", "gbdt", "mlp")
    assert [configs["round3_v2"]["agents"][f]["count"] for f in families] == [10, 0, 10, 0, 10]
    assert [configs["experiment_zero"]["agents"][f]["count"] for f in families] == [6] * 5
    for config in configs.values():
        assert config["agents"]["private_signal"]["mode"] == "off"
        assert config["agents"]["private_signal"]["strength"] == 0
        assert config["agents"]["q_max"] == 200
        assert config["market"]["impact_sensitivity"] == 0.03
    assert reconcile.SEED_SETS == {
        "round3": (tuple(range(1000, 1200)), tuple(range(2000, 2050))),
        "round4": (tuple(range(8000, 8200)), tuple(range(8500, 8550))),
    }
    configs["round3_v2"]["agents"]["q_max"] = 0
    assert configs["experiment_zero"]["agents"]["q_max"] == 200
    assert reconcile.population_configs()["round3_v2"]["agents"]["q_max"] == 200


@pytest.mark.parametrize("value,expected", [
    (0.713849, "PASS"), (0.713851, "REPRODUCIBILITY FAILURE"),
    (float("nan"), "REPRODUCIBILITY FAILURE"),
    (float("inf"), "REPRODUCIBILITY FAILURE"),
])
def test_reproduction_is_four_decimal_agreement(value, expected):
    assert reconcile.reproduction_status(value, 0.7138) == expected


def test_runner_crosses_all_cells_and_fits_training_only_threshold(monkeypatch):
    simulations = []
    thresholds = []
    tasks = []
    def simulate(config, seeds, *, workers):
        result = (object(),)
        simulations.append((config, seeds, result))
        return result
    def threshold(samples, label, horizon, quantile):
        assert samples is simulations[-2][2]  # training, never test
        assert (label, horizon, quantile) == ("default", 10, 0.95)
        thresholds.append(float(len(thresholds) + 1))
        return thresholds[-1]
    def task_episodes(samples, *, label_name, horizon, threshold):
        assert threshold == thresholds[-1]
        result = (SimpleNamespace(labels=np.array([False, True])),)
        tasks.append(result)
        return result
    class Observer:
        def __init__(self, *, return_lookback, rng):
            assert return_lookback == 10
            assert rng.random() == np.random.default_rng(5000).random()
        def fit(self, training):
            assert training is tasks[-2]
            return self
        def predict_proba(self, test):
            assert test is tasks[-1]
            return (np.array([0.1, 0.9]),)
    def gru(training, evaluation, **kwargs):
        assert training is tasks[-2]
        assert evaluation["test"] is tasks[-1]
        assert kwargs["seeds"] == (6000,)
        return ((6000, "epochs=1", {"test": (np.array([0.1, 0.9]),)}),)
    def metrics(labels, values, *, rng, n_resamples):
        assert n_resamples == 1000
        assert len(labels) == 1  # whole episodes, not individual timesteps
        return {"pr_auc": ConfidenceInterval(0.5, 0.4, 0.6)}
    monkeypatch.setattr(reconcile, "_simulate", simulate)
    monkeypatch.setattr(reconcile, "_fit_threshold", threshold)
    monkeypatch.setattr(reconcile, "_task_episodes", task_episodes)
    monkeypatch.setattr(reconcile, "LightGBMObserver", Observer)
    monkeypatch.setattr(reconcile, "_fit_gru_seeds", gru)
    monkeypatch.setattr(reconcile, "episode_bootstrap_metrics", metrics)
    cells = reconcile.run_reconciliation(workers=1)
    assert [(c.population, c.seed_set) for c in cells] == [
        (p, s) for p in reconcile.population_configs() for s in reconcile.SEED_SETS]
    assert len(simulations) == 8
    assert [c.threshold for c in cells] == [1, 2, 3, 4]
    report = reconcile.render_report(cells, runtime_seconds=60)
    assert report.count("REPRODUCIBILITY FAILURE") == 2
    assert "0.7138" in report and "0.9510" in report
    assert report.count("```yaml") == 2
    assert "Synthetic-simulator disclaimer" in report
    assert "Train rate 0.500000; test rate 0.500000" in report
    assert "1000-1199" in report
