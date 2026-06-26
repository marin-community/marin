# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep helpers: grid fan-out and metric-based selection.

The trials here are toy artifacts whose payload *is* their metrics dict, so the
tests exercise the real fan-out/fan-in mechanics (lower + run + reduce) without
training anything.
"""

import pytest
from marin.execution.artifact import Artifact
from marin.execution.lazy import Dataset, Recipe, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.sweep import grid, select, sweep


def _trial(learning_rate: float, weight_decay: float) -> Dataset:
    """A toy trial whose artifact payload is its own metrics: loss = lr + wd."""
    return Dataset(
        name=f"trials/lr{learning_rate}-wd{weight_decay}",
        version="v1",
        recipe=Recipe(
            fn=lambda config: config,
            build_config=lambda ctx, lr=learning_rate, wd=weight_decay: {
                "learning_rate": lr,
                "weight_decay": wd,
                "loss": lr + wd,
            },
        ),
    )


def test_grid_is_the_cartesian_product():
    assert grid(learning_rate=[0.1, 0.2], weight_decay=[0.0, 0.5]) == [
        {"learning_rate": 0.1, "weight_decay": 0.0},
        {"learning_rate": 0.1, "weight_decay": 0.5},
        {"learning_rate": 0.2, "weight_decay": 0.0},
        {"learning_rate": 0.2, "weight_decay": 0.5},
    ]


def test_sweep_gives_each_grid_point_a_distinct_identity():
    trials = sweep(_trial, learning_rate=[1e-4, 1e-3], weight_decay=[0.1])
    assert len(trials) == 2
    assert trials[0].fingerprint() != trials[1].fingerprint()


def test_select_records_the_lowest_loss_trial(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    # losses: 0.1, 0.3 (lr.1), 0.3, 0.5 (lr.3) -> min is lr=0.1, wd=0.0
    trials = sweep(_trial, learning_rate=[0.1, 0.3], weight_decay=[0.0, 0.2])
    best = select("sweeps/best", "v1", trials, metric="loss", mode="min")

    StepRunner().run([lower(best)])

    result = Artifact.from_path(f"{tmp_path}/sweeps/best/v1")
    assert result["score"] == pytest.approx(0.1)
    assert result["metrics"]["learning_rate"] == pytest.approx(0.1)
    assert result["winner_path"] == f"{tmp_path}/trials/lr0.1-wd0.0/v1"
    assert len(result["scores"]) == 4


def test_select_max_inverts_the_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    trials = sweep(_trial, learning_rate=[0.1, 0.3], weight_decay=[0.0])
    best = select("sweeps/top", "v1", trials, metric="loss", mode="max")

    StepRunner().run([lower(best)])

    assert Artifact.from_path(f"{tmp_path}/sweeps/top/v1")["score"] == pytest.approx(0.3)


def test_select_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        select("sweeps/x", "v1", [_trial(0.1, 0.0)], metric="loss", mode="best")


def test_select_rejects_an_empty_sweep():
    with pytest.raises(ValueError):
        select("sweeps/x", "v1", [], metric="loss")
