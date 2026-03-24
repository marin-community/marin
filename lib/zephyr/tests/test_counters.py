# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Zephyr user-defined counters: worker API, heartbeat plumbing, and coordinator aggregation."""

import threading

from zephyr import counters
from zephyr.execution import JobStatus, WorkerContext, ZephyrWorker, _worker_ctx_var


class FakeWorker:
    """Minimal WorkerContext implementation for testing counters."""

    def __init__(self):
        self._counters: dict[str, int] = {}
        self._counters_lock = threading.Lock()

    def get_shared(self, name: str):
        raise NotImplementedError

    def increment_counter(self, name: str, value: int = 1) -> None:
        with self._counters_lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def get_counter_snapshot(self) -> dict[str, int]:
        with self._counters_lock:
            return dict(self._counters)


def test_counters_increment_and_snapshot():
    """increment() accumulates in-memory; get_counter_snapshot() returns current values."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.increment("docs", 10)
        counters.increment("docs", 5)
        counters.increment("errors", 1)

        snapshot = counters.get_counters()
        assert snapshot == {"docs": 15, "errors": 1}
    finally:
        _worker_ctx_var.reset(token)


def test_counters_noop_outside_worker():
    """increment() is a no-op when not inside a Zephyr worker context."""
    token = _worker_ctx_var.set(None)
    try:
        counters.increment("anything", 999)  # should not raise
        assert counters.get_counters() == {}
    finally:
        _worker_ctx_var.reset(token)


def test_counters_changed_since_last_report():
    """counters_changed_since_last_report detects changes correctly."""
    from zephyr.counters import counters_changed_since_last_report, _last_reported, _report_lock

    # Clean up any state from previous tests
    with _report_lock:
        _last_reported.pop("test-worker", None)

    assert counters_changed_since_last_report("test-worker", {"docs": 10}) is True
    assert counters_changed_since_last_report("test-worker", {"docs": 10}) is False
    assert counters_changed_since_last_report("test-worker", {"docs": 20}) is True
    assert counters_changed_since_last_report("test-worker", {"docs": 20}) is False


def test_job_status_has_counters():
    """JobStatus dataclass includes counters field."""
    status = JobStatus(
        stage="stage0",
        completed=5,
        total=10,
        retries=0,
        in_flight=3,
        queue_depth=2,
        done=False,
        fatal_error=None,
        workers={},
        counters={"docs": 100, "bytes": 5000},
    )
    assert status.counters == {"docs": 100, "bytes": 5000}


def test_job_status_default_empty_counters():
    """JobStatus.counters defaults to empty dict."""
    status = JobStatus(
        stage="stage0",
        completed=0,
        total=0,
        retries=0,
        in_flight=0,
        queue_depth=0,
        done=False,
        fatal_error=None,
        workers={},
    )
    assert status.counters == {}
