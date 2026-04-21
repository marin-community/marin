# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from levanter.callbacks._straggler import StragglerReporter, _gather_durations


@dataclass
class FakeState:
    step: int = 10


@dataclass
class FakeStepInfo:
    step_duration: float
    state: FakeState = None
    loss: float = 0.0

    def __post_init__(self):
        if self.state is None:
            self.state = FakeState()

    step = property(lambda self: int(self.state.step) - 1)


def test_ewma_single_rank_initial():
    """First observation should set EWMA to the raw duration."""
    reporter = StragglerReporter(top_k=1)
    # Single-process: _gather_durations returns {0: duration}.
    with patch("levanter.tracker.log"):
        reporter._report({0: 1.5}, step=1)

    assert reporter._rank_ewma[0] == 1.5


def test_ewma_update():
    """EWMA should blend new observation with history."""
    reporter = StragglerReporter(ewma_alpha=0.5, top_k=1)
    with patch("levanter.tracker.log"):
        reporter._report({0: 2.0}, step=1)
        reporter._report({0: 4.0}, step=2)

    # EWMA: 0.5 * 4.0 + 0.5 * 2.0 = 3.0
    assert reporter._rank_ewma[0] == pytest.approx(3.0)


def test_report_min_median_max():
    """_report should compute correct min/median/max and log them."""
    reporter = StragglerReporter(top_k=2)
    logged = {}

    def capture_log(metrics, step):
        logged.update(metrics)

    with patch("levanter.tracker.log", side_effect=capture_log):
        reporter._report({0: 1.0, 1: 2.0, 2: 3.0}, step=5)

    assert logged["straggler/min_duration"] == pytest.approx(1.0)
    assert logged["straggler/median_duration"] == pytest.approx(2.0)
    assert logged["straggler/max_duration"] == pytest.approx(3.0)
    assert logged["straggler/spread"] == pytest.approx(2.0)


def test_top_k_slowest_ranks():
    """Top-k should identify the slowest ranks by EWMA."""
    reporter = StragglerReporter(top_k=2, ewma_alpha=1.0)
    logged = {}

    def capture_log(metrics, step):
        logged.update(metrics)

    with patch("levanter.tracker.log", side_effect=capture_log):
        reporter._report({0: 1.0, 1: 5.0, 2: 3.0, 3: 4.0}, step=10)

    # With alpha=1.0, EWMA == raw duration. Slowest: rank 1 (5.0), rank 3 (4.0)
    assert logged["straggler/slow_rank_0_id"] == 1
    assert logged["straggler/slow_rank_0_ewma"] == pytest.approx(5.0)
    assert logged["straggler/slow_rank_1_id"] == 3
    assert logged["straggler/slow_rank_1_ewma"] == pytest.approx(4.0)


def test_chronic_vs_oneoff_straggler():
    """EWMA should distinguish chronic stragglers from one-off stalls."""
    reporter = StragglerReporter(top_k=2, ewma_alpha=0.3)

    with patch("levanter.tracker.log"):
        # Rank 1 is consistently slow
        for _ in range(10):
            reporter._report({0: 1.0, 1: 3.0}, step=1)

        # One-off stall on rank 0
        reporter._report({0: 10.0, 1: 3.0}, step=2)

    # Rank 1 has been chronically slow; rank 0 had one spike.
    # After many steps with alpha=0.3, rank 1's EWMA should be close to 3.0.
    # Rank 0's EWMA should spike but still reflect history.
    assert reporter._rank_ewma[1] > 2.5
    # Rank 0 spike: 0.3 * 10 + 0.7 * prev (~1.0) = 3.0 + 0.7 ≈ 3.7
    # But rank 1 has been at 3.0 for 10 steps, so they're comparable.
    # The key test: after one more normal step, rank 1 stays slow.
    with patch("levanter.tracker.log"):
        reporter._report({0: 1.0, 1: 3.0}, step=3)

    # Rank 0 should decay back down, rank 1 stays high.
    assert reporter._rank_ewma[1] > reporter._rank_ewma[0]


def test_gather_durations_single_process():
    """Single-process gather returns the local duration."""
    result = _gather_durations(0, 2.5, 1)
    assert result == {0: 2.5}


def test_custom_prefix():
    """Custom prefix should appear in logged metric keys."""
    reporter = StragglerReporter(top_k=1, prefix="perf")
    logged = {}

    def capture_log(metrics, step):
        logged.update(metrics)

    with patch("levanter.tracker.log", side_effect=capture_log):
        reporter._report({0: 1.0}, step=1)

    assert "perf/min_duration" in logged
    assert "perf/max_duration" in logged


def test_on_step_single_process():
    """on_step should work end-to-end in single-process mode."""
    reporter = StragglerReporter(top_k=1)
    info = FakeStepInfo(step_duration=1.5)

    with patch("levanter.tracker.log"):
        with patch("jax.process_count", return_value=1):
            with patch("jax.process_index", return_value=0):
                reporter.on_step(info)

    assert len(reporter._local_durations) == 1
    assert reporter._local_durations[0] == 1.5
    assert reporter._rank_ewma[0] == pytest.approx(1.5)
