# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for corrected trial-level score artifacts."""

import importlib.util
import math
import sys
from pathlib import Path


def load_plot():
    spec = importlib.util.spec_from_file_location(
        "campaign_critical_difference_plot", Path(__file__).with_name("plot.py")
    )
    assert spec is not None and spec.loader is not None
    plot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plot
    spec.loader.exec_module(plot)
    return plot


def test_corrected_replay_cell_uses_trial_rewards_for_uncertainty(monkeypatch) -> None:
    plot = load_plot()
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


def test_cruxeval_counts_only_output_tasks(monkeypatch) -> None:
    plot = load_plot()

    class SampleTable:
        def to_pylist(self, **_kwargs):
            return [
                {"doc": '{"cruxeval_task": "input"}', "metrics": {"pass_rate": 0}, "filter": None},
                {"doc": '{"cruxeval_task": "output"}', "metrics": {"pass_rate": 1}, "filter": None},
                {"doc": '{"cruxeval_task": "output"}', "metrics": {"pass_rate": 0}, "filter": None},
            ]

    class SampleView:
        def __init__(self, _url):
            pass

        def scan(self, _table, **_kwargs):
            return SampleTable()

    monkeypatch.setattr(plot, "ReadView", SampleView)
    cell = plot.Cell("model", "cruxeval", 0.5, "s3://bucket/results")

    rewards, selector = plot.continuous_rewards(cell)
    count, basis = plot.scored_count(cell)

    assert rewards.tolist() == [1.0, 0.0]
    assert selector == "metrics:pass_rate:None"
    assert count == 2
    assert basis == "cruxeval_output_sample_rewards"
