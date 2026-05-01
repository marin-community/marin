# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for WorkerHealthTracker.

Exercises the two independent termination paths:
- Consecutive ping failures (reset by healthy ping)
- Build failures (monotonic counter, independent of pings)
"""

import pytest
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import WorkerId


@pytest.fixture
def tracker() -> WorkerHealthTracker:
    return WorkerHealthTracker(ping_threshold=10, build_threshold=10)


def test_ping_failure_threshold_boundary(tracker: WorkerHealthTracker) -> None:
    """9 consecutive failures are not enough; 10th trips; a healthy ping resets and requires 10 more."""
    wid = WorkerId("w-1")
    for _ in range(9):
        tracker.ping(wid, healthy=False)
    assert tracker.workers_over_threshold() == []
    tracker.ping(wid, healthy=False)
    assert tracker.workers_over_threshold() == [wid]

    tracker.forget(wid)
    for _ in range(9):
        tracker.ping(wid, healthy=False)
    tracker.ping(wid, healthy=True)  # reset
    for _ in range(9):
        tracker.ping(wid, healthy=False)
    assert tracker.workers_over_threshold() == []
    tracker.ping(wid, healthy=False)
    assert tracker.workers_over_threshold() == [wid]


def test_build_failure_threshold_boundary(tracker: WorkerHealthTracker) -> None:
    """9 build failures are not enough; 10th trips; healthy pings do not reset the counter."""
    wid = WorkerId("w-1")
    for _ in range(9):
        tracker.build_failed(wid)
    assert tracker.workers_over_threshold() == []
    tracker.build_failed(wid)
    assert tracker.workers_over_threshold() == [wid]

    tracker.ping(wid, healthy=True)
    assert tracker.workers_over_threshold() == [wid]  # not reset by healthy ping


def test_ping_and_build_failures_are_independent(tracker: WorkerHealthTracker) -> None:
    """A worker can trip via either path; tripping one does not affect the other counter."""
    wid = WorkerId("w-1")
    for _ in range(5):
        tracker.ping(wid, healthy=False)
    for _ in range(5):
        tracker.build_failed(wid)
    assert tracker.workers_over_threshold() == []
    tracker.ping(wid, healthy=False)  # 6 ping failures, still < 10
    assert tracker.workers_over_threshold() == []


def test_forget_removes_worker(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
    for _ in range(10):
        tracker.ping(wid, healthy=False)
    assert tracker.workers_over_threshold()
    tracker.forget(wid)
    assert tracker.workers_over_threshold() == []


def test_forget_many_drops_only_listed_workers(tracker: WorkerHealthTracker) -> None:
    a, b, c = WorkerId("a"), WorkerId("b"), WorkerId("c")
    for wid in (a, b, c):
        for _ in range(10):
            tracker.ping(wid, healthy=False)
    tracker.forget_many([a, c])
    assert tracker.workers_over_threshold() == [b]


def test_per_worker_counters_are_independent(tracker: WorkerHealthTracker) -> None:
    a, b = WorkerId("a"), WorkerId("b")
    for _ in range(10):
        tracker.ping(a, healthy=False)
    assert tracker.workers_over_threshold() == [a]
    assert b not in tracker.workers_over_threshold()


def test_snapshot_reports_both_counters(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
    for _ in range(3):
        tracker.ping(wid, healthy=False)
    for _ in range(2):
        tracker.build_failed(wid)
    assert tracker.snapshot() == {wid: (3, 2)}
