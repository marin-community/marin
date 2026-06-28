# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep helpers: grid fan-out and metric-based selection.

The toy trials here *write* their metrics to their own output, the way a real
training run does (``train_lm`` mirrors them to ``tracker_metrics.jsonl``), so the
tests exercise the real fan-out/fan-in mechanics — lower, run, then read each
trial's recorded metrics back with the selection's reader and reduce — without
training anything.
"""

import json

import fsspec
import pytest
from marin.execution.artifact import Checkpoint
from marin.execution.lazy import Lazy, Recipe, resolve
from marin.experiment.sweep import Selection, grid, select, sweep


def _write_metrics(config: dict) -> None:
    """Write a ``train_lm``-shaped metrics record to the trial's output."""
    fs, _, _ = fsspec.get_fs_token_paths(config["out"])
    fs.makedirs(config["out"], exist_ok=True)
    record = {
        "config": {},
        "summary": {"loss": config["loss"], "learning_rate": config["lr"], "weight_decay": config["wd"]},
    }
    with fs.open(f"{config['out']}/tracker_metrics.jsonl", "w") as f:
        f.write(json.dumps(record) + "\n")


def _trial(learning_rate: float, weight_decay: float) -> Lazy[Checkpoint]:
    """A toy trial that records ``loss = lr + wd`` to its output, like a real run."""
    return Lazy(
        name=f"trials/lr{learning_rate}-wd{weight_decay}",
        version="v1",
        result_type=Checkpoint,
        recipe=Recipe(
            fn=_write_metrics,
            build_config=lambda ctx, lr=learning_rate, wd=weight_decay: {
                "out": ctx.out,
                "loss": lr + wd,
                "lr": lr,
                "wd": wd,
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

    result = resolve(best)

    assert isinstance(result, Selection)
    assert result.score == pytest.approx(0.1)
    assert result.winner == "trials/lr0.1-wd0.0@v1"
    assert result.metrics["learning_rate"] == pytest.approx(0.1)
    assert result.winner_path == f"{tmp_path}/trials/lr0.1-wd0.0/v1"
    assert len(result.scores) == 4


def test_select_max_inverts_the_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    trials = sweep(_trial, learning_rate=[0.1, 0.3], weight_decay=[0.0])
    best = select("sweeps/top", "v1", trials, metric="loss", mode="max")

    assert resolve(best).score == pytest.approx(0.3)


def test_select_returns_a_selection_handle():
    # The reduced sweep is a Lazy[Selection] addressed at the given name@version.
    best = select("sweeps/x", "v1", [_trial(0.1, 0.0)], metric="loss")
    assert (best.name, best.version, best.result_type) == ("sweeps/x", "v1", Selection)


def test_select_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        select("sweeps/x", "v1", [_trial(0.1, 0.0)], metric="loss", mode="best")


def test_select_rejects_an_empty_sweep():
    with pytest.raises(ValueError):
        select("sweeps/x", "v1", [], metric="loss")


def test_select_rejects_colliding_trial_identities():
    # Two trials that share name@version would silently collapse in the selection.
    same = _trial(0.1, 0.0)
    with pytest.raises(ValueError, match="distinct name@version"):
        select("sweeps/x", "v1", [same, _trial(0.1, 0.0)], metric="loss")
