# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for corrected trial-level score artifacts."""

import importlib.util
import math
import sys
from pathlib import Path


def test_corrected_replay_cell_uses_trial_rewards_for_uncertainty(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("campaign_critical_difference_plot", Path(__file__).with_name("plot.py"))
    assert spec is not None and spec.loader is not None
    plot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plot
    spec.loader.exec_module(plot)
    replay = {
        "model": "open-athena/Snowball-Test",
        "benchmark": "olympiadbench",
        "num_trials": 3,
        "num_judge_failed": 0,
        "trials": [{"corrected_correct": value} for value in (True, False, True)],
    }
    monkeypatch.setattr(plot, "read_json", lambda _: replay)
    cell = plot.Cell(replay["model"], replay["benchmark"], 0.667, "s3://bucket/regrade/model.json")

    statistics = plot.cell_statistics(cell, {})

    assert statistics["trial_count"] == 3
    assert math.isclose(statistics["raw_reward_mean"], 2 / 3)
    assert math.isclose(statistics["sem"], 1 / 3)
    assert statistics["sem_basis"] == "corrected_replay_trial_rewards"
