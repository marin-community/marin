# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep helpers: grid fan-out, and selection as ordinary code over resolved outputs.

The toy trials here *write* their metrics to their own output, the way a real training run
does (``train_lm`` mirrors them to ``tracker_metrics.jsonl``), so the selection test exercises
the real fan-out/fan-in mechanics — run each trial, read its recorded metrics back through
``LevanterCheckpoint.training_metrics()``, and reduce — without training anything. There is no
framework ``select``: selection is the reduction the experiment writes.
"""

import json

import fsspec
import pytest
from marin.execution.lazy import ArtifactStep, resolve
from marin.experiment.sweep import grid, sweep
from marin.training.training import LevanterCheckpoint


def _write_metrics(config: dict) -> None:
    """Write a ``train_lm``-shaped metrics record to the trial's output."""
    fs, _, _ = fsspec.get_fs_token_paths(config["out"])
    fs.makedirs(config["out"], exist_ok=True)
    record = {"config": {}, "summary": {"eval/loss": config["loss"], "train/loss": config["loss"]}}
    with fs.open(f"{config['out']}/tracker_metrics.jsonl", "w") as f:
        f.write(json.dumps(record) + "\n")


def _trial(learning_rate: float, weight_decay: float) -> ArtifactStep[LevanterCheckpoint]:
    """A toy trial that records ``loss = lr + wd`` to its output, like a real run."""
    return ArtifactStep(
        name=f"trials/lr{learning_rate}-wd{weight_decay}",
        version="2026.06.28",
        artifact_type=LevanterCheckpoint,
        run=_write_metrics,
        build_config=lambda ctx, lr=learning_rate, wd=weight_decay: {"out": ctx.output_path, "loss": lr + wd},
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


def test_selection_is_user_code_over_resolved_metrics(tmp_path, monkeypatch):
    """The replacement for the old ``select``: run the sweep, read each trial's metrics through
    its typed artifact, and reduce — no framework verb involved."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    # losses: 0.1, 0.3 (lr.1), 0.3, 0.5 (lr.3) -> min is lr=0.1, wd=0.0
    trials = sweep(_trial, learning_rate=[0.1, 0.3], weight_decay=[0.0, 0.2])

    scored = [(trial, resolve(trial).training_metrics().eval_loss) for trial in trials]
    best, best_loss = min(scored, key=lambda pair: pair[1])

    assert best_loss == pytest.approx(0.1)
    assert best.name == "trials/lr0.1-wd0.0"
