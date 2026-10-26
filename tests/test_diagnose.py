"""End-to-end test for the baseline cascade-diagnostic CLI."""

from __future__ import annotations

from pathlib import Path

import yaml

from herdlab.experiments.diagnose import main


PROJECT_ROOT = Path(__file__).parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


def test_cli_runs_two_short_episodes_and_writes_actual_table(tmp_path: Path) -> None:
    config = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    config["simulation"]["n_steps"] = 12
    config["targets"]["unwind_horizon"] = 3
    config["agents"]["population_size"] = 2
    config["agents"]["momentum"]["count"] = 1
    config["agents"]["reversion"]["count"] = 1
    for family in ("linear", "gbdt", "mlp"):
        config["agents"][family]["count"] = 0

    config_path = tmp_path / "short_baseline.yaml"
    report_path = tmp_path / "cascade_diagnostics.md"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    main(
        [
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--n-episodes",
            "2",
            "--seed-start",
            "41",
        ]
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# Baseline cascade diagnostics" in report
    assert "2 consecutive seeds (`41-42`), 12 steps each" in report
    assert "**Forced deleveraging is" in report
    assert "| 41 |" in report
    assert "| 42 |" in report
    assert "C^pos p50" in report
    assert "C^flow p50" in report
