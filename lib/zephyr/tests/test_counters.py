# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Zephyr user-defined counters: worker API, heartbeat plumbing, and coordinator aggregation."""

import threading

from zephyr import counters
from zephyr.execution import _worker_ctx_var


class FakeWorker:
    """Minimal WorkerContext implementation for testing counters."""

    def __init__(self):
        self._counters: dict[str, int] = {}
        self._counters_lock = threading.Lock()
        self._last_reported_counters: dict[str, int] = {}

    def get_shared(self, name: str):
        raise NotImplementedError

    def increment_counter(self, name: str, value: int = 1) -> None:
        with self._counters_lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def get_counter_snapshot(self) -> dict[str, int]:
        with self._counters_lock:
            return dict(self._counters)

    def _counters_changed(self) -> bool:
        with self._counters_lock:
            current = dict(self._counters)
        if current == self._last_reported_counters:
            return False
        self._last_reported_counters = current
        return True


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


def test_counters_changed():
    """_counters_changed detects changes and deduplicates correctly."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        assert worker._counters_changed() is False  # empty → empty, no change

        counters.increment("docs", 10)
        assert worker._counters_changed() is True
        assert worker._counters_changed() is False  # same value, no change

        counters.increment("docs", 10)
        assert worker._counters_changed() is True
        assert worker._counters_changed() is False
    finally:
        _worker_ctx_var.reset(token)
