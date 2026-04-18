# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for WorkerHealthTracker."""

import math

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


def test_unknown_worker_score_is_zero(tracker: WorkerHealthTracker) -> None:
    assert tracker.current_score(WorkerId("w-1")) == 0.0
    assert tracker.workers_over_threshold() == []


def test_single_bump_adds_weight(tracker: WorkerHealthTracker) -> None:
    score = tracker.bump(WorkerId("w-1"), HealthSignal.RPC_FAILURE)
    assert score == pytest.approx(1.0)
    assert tracker.current_score(WorkerId("w-1")) == pytest.approx(1.0)


def test_repeated_bumps_accumulate_without_time_advance(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
    for _ in range(10):
        tracker.bump(wid, HealthSignal.RPC_FAILURE)
    assert tracker.current_score(wid) == pytest.approx(10.0)
    over = tracker.workers_over_threshold()
    assert over == [(wid, pytest.approx(10.0))]


def test_score_decays_with_time(tracker: WorkerHealthTracker, clock: FakeClock) -> None:
    wid = WorkerId("w-1")
    tracker.bump(wid, HealthSignal.RPC_FAILURE)
    # Half-life is 60s; after 60s, score should halve.
    clock.advance(60.0)
    assert tracker.current_score(wid) == pytest.approx(0.5)
    # After another 60s (total 120s, two half-lives), score is 0.25.
    clock.advance(60.0)
    assert tracker.current_score(wid) == pytest.approx(0.25)


def test_decayed_score_accumulates_correctly(tracker: WorkerHealthTracker, clock: FakeClock) -> None:
    wid = WorkerId("w-1")
    tracker.bump(wid, HealthSignal.RPC_FAILURE)  # score=1 at t=0
    clock.advance(60.0)  # score decays to 0.5
    new = tracker.bump(wid, HealthSignal.RPC_FAILURE)  # 0.5 + 1.0 = 1.5
    assert new == pytest.approx(1.5)
    assert tracker.current_score(wid) == pytest.approx(1.5)


def test_spaced_bumps_stay_below_threshold(clock: FakeClock) -> None:
    """Evidence spread over many half-lives should not accumulate indefinitely."""
    tracker = WorkerHealthTracker(half_life_s=60.0, threshold=10.0, clock=clock)
    wid = WorkerId("w-1")
    # A bump every full half-life: score converges to 2.0 (geometric series 1/(1-0.5)).
    for _ in range(100):
        tracker.bump(wid, HealthSignal.RPC_FAILURE)
        clock.advance(60.0)
    assert tracker.current_score(wid) < 3.0
    assert tracker.workers_over_threshold() == []


def test_rapid_bumps_cross_threshold(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
    for signal in [
        HealthSignal.RPC_FAILURE,
        HealthSignal.TASK_WORKER_FAILED,
    ] * 6:
        tracker.bump(wid, signal)
    # 12 bumps of weight 1.0 with no decay = 12.0 >= 10.0
    over = tracker.workers_over_threshold()
    assert len(over) == 1
    assert over[0][0] == wid
    assert over[0][1] >= 10.0


def test_forget_drops_worker(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
    for _ in range(10):
        tracker.bump(wid, HealthSignal.RPC_FAILURE)
    assert tracker.workers_over_threshold()
    tracker.forget(wid)
    assert tracker.current_score(wid) == 0.0
    assert tracker.workers_over_threshold() == []


def test_forget_many_drops_multiple(tracker: WorkerHealthTracker) -> None:
    a, b, c = WorkerId("a"), WorkerId("b"), WorkerId("c")
    for wid in (a, b, c):
        for _ in range(10):
            tracker.bump(wid, HealthSignal.RPC_FAILURE)
    assert len(tracker.workers_over_threshold()) == 3
    tracker.forget_many([a, c])
    remaining = tracker.workers_over_threshold()
    assert [wid for wid, _ in remaining] == [b]


def test_multiple_workers_tracked_independently(tracker: WorkerHealthTracker, clock: FakeClock) -> None:
    a, b = WorkerId("a"), WorkerId("b")
    tracker.bump(a, HealthSignal.RPC_FAILURE)
    clock.advance(30.0)
    tracker.bump(b, HealthSignal.RPC_FAILURE)
    # A has decayed ~ exp(-ln2 * 0.5) ≈ 0.707
    assert tracker.current_score(a) == pytest.approx(math.exp(-math.log(2) * 0.5))
    assert tracker.current_score(b) == pytest.approx(1.0)


def test_snapshot_returns_decayed_scores(tracker: WorkerHealthTracker, clock: FakeClock) -> None:
    a, b = WorkerId("a"), WorkerId("b")
    tracker.bump(a, HealthSignal.RPC_FAILURE)
    tracker.bump(b, HealthSignal.RPC_FAILURE)
    clock.advance(60.0)
    snap = tracker.snapshot()
    assert snap == {a: pytest.approx(0.5), b: pytest.approx(0.5)}


def test_threshold_exposed_on_tracker() -> None:
    tracker = WorkerHealthTracker(half_life_s=60.0, threshold=7.5)
    assert tracker.threshold == 7.5


@pytest.mark.parametrize("bad_half_life", [0.0, -1.0])
def test_non_positive_half_life_rejected(bad_half_life: float) -> None:
    with pytest.raises(AssertionError):
        WorkerHealthTracker(half_life_s=bad_half_life, threshold=10.0)


@pytest.mark.parametrize("bad_threshold", [0.0, -1.0])
def test_non_positive_threshold_rejected(bad_threshold: float) -> None:
    with pytest.raises(AssertionError):
        WorkerHealthTracker(half_life_s=60.0, threshold=bad_threshold)
