# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Zephyr user-defined counters: worker API and heartbeat plumbing."""

from zephyr import counters
from zephyr.execution import ZephyrExecutionResult
from zephyr.worker_context import CounterSnapshot, _worker_ctx_var


class FakeWorker:
    """Minimal WorkerContext implementation for testing counters."""

    def __init__(self):
        self._counters: dict[str, int] = {}
        self._generation: int = 0

    def get_shared(self, name: str):
        raise NotImplementedError

    def increment_counter(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def set_counter(self, name: str, value: int) -> None:
        self._counters[name] = value

    def get_counter_snapshot(self) -> CounterSnapshot:
        self._generation += 1
        return CounterSnapshot(counters=dict(self._counters), generation=self._generation)


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


def test_counters_set():
    """set() overwrites the counter value rather than accumulating."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.increment("visits", 5)
        counters.set("visits", 2)  # overwrites, not 5+2
        assert counters.get_counters() == {"visits": 2}

        counters.set("mem_bytes", 1024)
        counters.set("mem_bytes", 2048)  # replaces previous value
        assert counters.get_counters()["mem_bytes"] == 2048
    finally:
        _worker_ctx_var.reset(token)


def test_set_noop_outside_worker():
    """set() is a no-op when not inside a Zephyr worker context."""
    token = _worker_ctx_var.set(None)
    try:
        counters.set("anything", 999)  # should not raise
        assert counters.get_counters() == {}
    finally:
        _worker_ctx_var.reset(token)


def test_zephyr_execution_result_fields():
    """ZephyrExecutionResult exposes both results and counters."""
    result = ZephyrExecutionResult(results=["a.jsonl", "b.jsonl"], counters={"docs": 7})
    assert result.results == ["a.jsonl", "b.jsonl"]
    assert result.counters == {"docs": 7}


def test_zephyr_execution_result_empty():
    """ZephyrExecutionResult handles empty results and counters (e.g. dry_run)."""
    result = ZephyrExecutionResult(results=[], counters={})
    assert result.results == []
    assert result.counters == {}
