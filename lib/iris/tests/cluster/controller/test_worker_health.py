# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for WorkerHealthTracker.

These tests exercise the three pieces of non-obvious behavior:

- Exponential decay of accumulated score
- Weight differences between signals observable through threshold crossing
- Forget / snapshot APIs consumed by the reaper
"""

import pytest

from iris.cluster.controller.worker_health import HealthSignal, WorkerHealthTracker
from iris.cluster.types import WorkerId


class FakeClock:
    def __init__(self) -> None:
        self.now_ms = 0

    def __call__(self) -> int:
        return self.now_ms

    def advance(self, seconds: float) -> None:
        self.now_ms += int(seconds * 1000)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def tracker(clock: FakeClock) -> WorkerHealthTracker:
    return WorkerHealthTracker(half_life_s=60.0, threshold=10.0, clock=clock)


def test_score_halves_every_half_life(tracker: WorkerHealthTracker, clock: FakeClock) -> None:
    wid = WorkerId("w-1")
    tracker.bump(wid, HealthSignal.RPC_FAILURE)
    clock.advance(60.0)
    assert tracker.current_score(wid) == pytest.approx(0.5)
    clock.advance(60.0)
    assert tracker.current_score(wid) == pytest.approx(0.25)


def test_bump_adds_weight_to_decayed_score(tracker: WorkerHealthTracker, clock: FakeClock) -> None:
    """New bumps are added on top of the already-decayed score, not reset to weight."""
    wid = WorkerId("w-1")
    tracker.bump(wid, HealthSignal.RPC_FAILURE)
    clock.advance(60.0)  # 1.0 -> 0.5
    assert tracker.bump(wid, HealthSignal.RPC_FAILURE) == pytest.approx(1.5)


def test_spaced_bumps_never_cross_threshold(clock: FakeClock) -> None:
    """One failure per hour with 1-hour half-life converges to 2.0 — stays under threshold.

    Without decay, any worker alive long enough would eventually be reaped.
    The geometric series 1/(1-0.5) caps at 2.0, well below threshold 10.0.
    """
    tracker = WorkerHealthTracker(half_life_s=3600.0, threshold=10.0, clock=clock)
    wid = WorkerId("w-1")
    for _ in range(100):
        tracker.bump(wid, HealthSignal.RPC_FAILURE)
        clock.advance(3600.0)
    assert tracker.current_score(wid) < 3.0
    assert tracker.workers_over_threshold() == []


def test_ten_rpc_failures_cross_threshold(tracker: WorkerHealthTracker) -> None:
    """Preserves the legacy 10-strike RPC behavior: 9 are not enough, 10th trips."""
    wid = WorkerId("w-1")
    for _ in range(9):
        tracker.bump(wid, HealthSignal.RPC_FAILURE)
    assert tracker.workers_over_threshold() == []
    tracker.bump(wid, HealthSignal.RPC_FAILURE)
    over = tracker.workers_over_threshold()
    assert [w for w, _ in over] == [wid]
    assert over[0][1] == pytest.approx(10.0)


def test_build_failures_alone_need_twenty_to_cross(tracker: WorkerHealthTracker) -> None:
    """BUILD_FAILED has half weight: threshold requires ~2x as many hits."""
    wid = WorkerId("w-1")
    for _ in range(19):
        tracker.bump(wid, HealthSignal.TASK_BUILD_FAILED)
    assert tracker.workers_over_threshold() == []
    tracker.bump(wid, HealthSignal.TASK_BUILD_FAILED)
    assert [w for w, _ in tracker.workers_over_threshold()] == [wid]


def test_build_and_rpc_failures_combine(tracker: WorkerHealthTracker) -> None:
    """8 RPC (weight 1.0) + 4 BUILD (weight 0.5) = 10.0 -> trips threshold."""
    wid = WorkerId("w-1")
    for _ in range(8):
        tracker.bump(wid, HealthSignal.RPC_FAILURE)
    for _ in range(4):
        tracker.bump(wid, HealthSignal.TASK_BUILD_FAILED)
    over = tracker.workers_over_threshold()
    assert len(over) == 1
    assert over[0][1] == pytest.approx(10.0)


def test_forget_removes_worker_from_threshold_list(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
    for _ in range(10):
        tracker.bump(wid, HealthSignal.RPC_FAILURE)
    assert tracker.workers_over_threshold()
    tracker.forget(wid)
    assert tracker.current_score(wid) == 0.0
    assert tracker.workers_over_threshold() == []


def test_forget_many_drops_listed_workers_only(tracker: WorkerHealthTracker) -> None:
    a, b, c = WorkerId("a"), WorkerId("b"), WorkerId("c")
    for wid in (a, b, c):
        for _ in range(10):
            tracker.bump(wid, HealthSignal.RPC_FAILURE)
    tracker.forget_many([a, c])
    assert [wid for wid, _ in tracker.workers_over_threshold()] == [b]


def test_per_worker_scores_are_independent(tracker: WorkerHealthTracker, clock: FakeClock) -> None:
    a, b = WorkerId("a"), WorkerId("b")
    tracker.bump(a, HealthSignal.RPC_FAILURE)
    clock.advance(60.0)  # a decays to 0.5
    tracker.bump(b, HealthSignal.RPC_FAILURE)
    assert tracker.current_score(a) == pytest.approx(0.5)
    assert tracker.current_score(b) == pytest.approx(1.0)


def test_snapshot_reports_decayed_scores(tracker: WorkerHealthTracker, clock: FakeClock) -> None:
    """snapshot() is used by diagnostics — must return decayed, not stored, values."""
    a, b = WorkerId("a"), WorkerId("b")
    tracker.bump(a, HealthSignal.RPC_FAILURE)
    tracker.bump(b, HealthSignal.RPC_FAILURE)
    clock.advance(60.0)
    assert tracker.snapshot() == {a: pytest.approx(0.5), b: pytest.approx(0.5)}
