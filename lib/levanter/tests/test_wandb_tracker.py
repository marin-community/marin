# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from levanter.tracker.wandb import WandbTracker


class _FakeRun:
    def __init__(self, step: int):
        self.step = step
        self.logged: list[tuple[dict[str, object], int | None, bool | None]] = []

    def log(self, metrics, step=None, commit=None):
        self.logged.append((dict(metrics), step, commit))


def _make_tracker(run_step: int) -> WandbTracker:
    tracker = WandbTracker.__new__(WandbTracker)
    tracker.run = _FakeRun(run_step)
    tracker._last_warning_step = -500
    tracker._replicate_path = None
    return tracker


def test_wandb_tracker_clamps_stale_resumed_step(caplog):
    tracker = _make_tracker(run_step=15712)

    with caplog.at_level(logging.WARNING):
        tracker.log({"train/loss": 1.23}, step=15549, commit=None)

    assert tracker.run.logged == [({"train/loss": 1.23}, 15712, None)]
    assert "Clamping resumed W&B log forward" in caplog.text
    assert "caller=" in caplog.text


def test_wandb_tracker_only_warns_once_per_interval(caplog):
    tracker = _make_tracker(run_step=15712)

    with caplog.at_level(logging.WARNING):
        tracker.log({"metric/a": 1}, step=15549, commit=None)
        tracker.log({"metric/b": 2}, step=15550, commit=None)

    assert tracker.run.logged == [
        ({"metric/a": 1}, 15712, None),
        ({"metric/b": 2}, 15712, None),
    ]
    assert caplog.text.count("Clamping resumed W&B log forward") == 1
