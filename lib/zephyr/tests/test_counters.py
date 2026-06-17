# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Zephyr user-defined counters: worker API and heartbeat plumbing."""

import logging
import threading

from zephyr import counters
from zephyr.counters import ScopedCounters
from zephyr.execution import ZephyrCoordinator, ZephyrExecutionResult
from zephyr.worker_context import Aggregation, CounterEntry, CounterSnapshot, _worker_ctx_var


class FakeWorker:
    """Minimal WorkerContext implementation for testing counters."""

    def __init__(self, stage_name: str = "test_stage"):
        self._counters: dict[str, CounterEntry] = {}
        self._generation: int = 0
        self._stage_name = stage_name

    def get_shared(self, name: str):
        raise NotImplementedError

    def increment_counter(self, name: str, value: int = 1, stage: str | None = None) -> None:
        if name in self._counters:
            self._counters[name].value += value
        else:
            self._counters[name] = CounterEntry(value, stage=stage)

    def set_counter(self, name: str, value: int | float, stage: str | None = None) -> None:
        if name in self._counters:
            entry = self._counters[name]
            entry.value = value
            entry.count = 1
            entry.stage = stage
        else:
            self._counters[name] = CounterEntry(value, stage=stage)

    def update_counter(self, name: str, value: int | float, stage: str | None = None) -> None:
        entry = self._counters.get(name)
        if entry is None or entry.count == 0:
            if entry is None:
                self._counters[name] = CounterEntry(value, stage=stage)
            else:
                entry.value = value
                entry.stage = stage
                entry.count = 1
            return
        entry.merge(CounterEntry(value, entry.aggregation, stage, count=1))

    def set_aggregation(self, name: str, agg: Aggregation) -> None:
        if name in self._counters:
            self._counters[name].aggregation = agg
        else:
            self._counters[name] = CounterEntry(0, aggregation=agg, count=0)

    def current_stage_name(self) -> str:
        return self._stage_name

    def get_counter_snapshot(self) -> CounterSnapshot:
        self._generation += 1
        return CounterSnapshot(
            counters={k: CounterEntry(e.value, e.aggregation, e.stage, e.count) for k, e in self._counters.items()},
            generation=self._generation,
        )


def _make_coordinator(
    completed: list[CounterSnapshot],
    inflight: list[CounterSnapshot] | None = None,
) -> ZephyrCoordinator:
    """Build a minimal ZephyrCoordinator seeded with canned snapshots for testing."""
    coord = ZephyrCoordinator.__new__(ZephyrCoordinator)
    coord._lock = threading.Lock()
    coord._completed_counters = list(completed)
    coord._worker_counters = {str(i): s for i, s in enumerate(inflight or [])}
    return coord


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


def test_counters_set_counter():
    """set_counter() overwrites the counter value rather than accumulating."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.increment("visits", 5)
        counters.set_counter("visits", 2)  # overwrites, not 5+2
        assert counters.get_counters() == {"visits": 2}

        counters.set_counter("mem_bytes", 1024)
        counters.set_counter("mem_bytes", 2048)  # replaces previous value
        assert counters.get_counters()["mem_bytes"] == 2048
    finally:
        _worker_ctx_var.reset(token)


def test_set_counter_noop_outside_worker():
    """set_counter() is a no-op when not inside a Zephyr worker context."""
    token = _worker_ctx_var.set(None)
    try:
        counters.set_counter("anything", 999)  # should not raise
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


def test_legacy_aliases_delegate_to_pipeline():
    """Module-level increment/set_counter/get_counters delegate to pipeline scope."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.increment("x", 3)
        counters.set_counter("y", 7)
        result = counters.get_counters()
        assert result == {"x": 3, "y": 7}
        assert counters.pipeline.get_counters() == {"x": 3, "y": 7}
    finally:
        _worker_ctx_var.reset(token)


def test_stage_scoped_counters():
    """Stage-scoped counters carry stage=<name> on the entry, not a key prefix."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        s = counters.stage("tokenize")
        s.update_counter("tokens", 100)
        s.update_counter("tokens", 50)

        assert s.get_counters() == {"tokens": 150}

        # The underlying entry carries stage metadata, not a prefixed key
        assert worker._counters["tokens"].stage == "tokenize"
        assert worker._counters["tokens"].value == 150
    finally:
        _worker_ctx_var.reset(token)


def test_stage_scope_isolated_from_pipeline():
    """Stage and pipeline counters with the same name are stored under the same key but different stage."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.pipeline.update_counter("metric", 1)
        # pipeline counter is stored with stage=None; updating via stage("stg")
        # would share the same key "metric" so we use a different name here
        counters.stage("stg").update_counter("stg_metric", 2)

        assert counters.pipeline.get_counters() == {"metric": 1}
        assert counters.stage("stg").get_counters() == {"stg_metric": 2}
    finally:
        _worker_ctx_var.reset(token)


def test_current_stage_uses_worker_stage_name():
    """current_stage() returns a ScopedCounters for the worker's current stage."""
    worker = FakeWorker(stage_name="my_stage")
    token = _worker_ctx_var.set(worker)
    try:
        cs = counters.current_stage()
        cs.update_counter("items", 5)

        assert cs.get_counters() == {"items": 5}
        assert worker._counters["items"].stage == "my_stage"
    finally:
        _worker_ctx_var.reset(token)


def test_current_stage_noop_outside_worker():
    """current_stage() outside a worker returns pipeline scope (all ops are no-ops)."""
    token = _worker_ctx_var.set(None)
    try:
        cs = counters.current_stage()
        assert isinstance(cs, ScopedCounters)
        cs.update_counter("x", 99)  # no-op
        assert cs.get_counters() == {}
    finally:
        _worker_ctx_var.reset(token)


def test_aggregation_sum_default():
    """Counters with no aggregation hint are summed."""
    coord = _make_coordinator(
        [
            CounterSnapshot(counters={"a": CounterEntry(10)}, generation=1),
            CounterSnapshot(counters={"a": CounterEntry(20)}, generation=2),
        ]
    )
    assert coord.get_counters() == {"a": 30}


def test_aggregation_max():
    coord = _make_coordinator(
        [
            CounterSnapshot(counters={"peak": CounterEntry(100, Aggregation.MAX)}, generation=1),
            CounterSnapshot(counters={"peak": CounterEntry(200, Aggregation.MAX)}, generation=2),
            CounterSnapshot(counters={"peak": CounterEntry(50, Aggregation.MAX)}, generation=3),
        ]
    )
    assert coord.get_counters() == {"peak": 200}


def test_aggregation_min():
    coord = _make_coordinator(
        [
            CounterSnapshot(counters={"latency": CounterEntry(300, Aggregation.MIN)}, generation=1),
            CounterSnapshot(counters={"latency": CounterEntry(100, Aggregation.MIN)}, generation=2),
            CounterSnapshot(counters={"latency": CounterEntry(200, Aggregation.MIN)}, generation=3),
        ]
    )
    assert coord.get_counters() == {"latency": 100}


def test_aggregation_average():
    coord = _make_coordinator(
        [
            CounterSnapshot(counters={"cpu_pct": CounterEntry(80, Aggregation.AVERAGE)}, generation=1),
            CounterSnapshot(counters={"cpu_pct": CounterEntry(60, Aggregation.AVERAGE)}, generation=2),
            CounterSnapshot(counters={"cpu_pct": CounterEntry(40, Aggregation.AVERAGE)}, generation=3),
        ]
    )
    assert coord.get_counters() == {"cpu_pct": 60}


def test_aggregation_mixed():
    """Multiple counters can use different aggregations in the same snapshot set."""
    coord = _make_coordinator(
        [
            CounterSnapshot(
                counters={"total": CounterEntry(10), "peak": CounterEntry(100, Aggregation.MAX)},
                generation=1,
            ),
            CounterSnapshot(
                counters={"total": CounterEntry(20), "peak": CounterEntry(50, Aggregation.MAX)},
                generation=2,
            ),
        ]
    )
    assert coord.get_counters() == {"total": 30, "peak": 100}


def test_aggregation_conflict_keeps_first_and_warns(caplog):
    """Conflicting aggregations keep the first-seen mode and warn, never raising.

    Stats collection must not be able to fault the execution path, so a counter
    that disagrees on aggregation across snapshots (only possible via user
    ``set_aggregation`` misuse) resolves to the first-seen mode rather than
    raising.
    """
    coord = _make_coordinator(
        [
            CounterSnapshot(counters={"x": CounterEntry(1, Aggregation.MAX)}, generation=1),
            CounterSnapshot(counters={"x": CounterEntry(2, Aggregation.MIN)}, generation=2),
        ]
    )
    with caplog.at_level(logging.WARNING):
        result = coord.get_counters()
    # First-seen aggregation (MAX) wins over the later MIN.
    assert result == {"x": 2}
    assert any("conflicting aggregations" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Coordinator stage filter
# ---------------------------------------------------------------------------


def test_coordinator_stage_filter():
    """get_counters(stage=...) filters to entries with that stage."""
    coord = _make_coordinator(
        [
            CounterSnapshot(
                counters={
                    "errors": CounterEntry(5, stage=None),
                    "tokens": CounterEntry(100, stage="tokenize"),
                },
                generation=1,
            ),
        ]
    )
    assert coord.get_counters() == {"errors": 5, "tokens": 100}
    assert coord.get_counters(stage="tokenize") == {"tokens": 100}
    assert coord.get_counters(stage="nonexistent") == {}


def test_coordinator_stage_filter_aggregates_same_name_across_stages():
    """get_counters() (no filter) aggregates same-name counters from different stages."""
    coord = _make_coordinator(
        [
            CounterSnapshot(counters={"errors": CounterEntry(3, stage=None)}, generation=1),
            CounterSnapshot(counters={"errors": CounterEntry(7, stage="tokenize")}, generation=2),
        ]
    )
    assert coord.get_counters() == {"errors": 10}


# ---------------------------------------------------------------------------
# update_counter
# ---------------------------------------------------------------------------


def test_update_counter_sum():
    """update_counter with SUM aggregation behaves like increment."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.pipeline.set_aggregation("hits", Aggregation.SUM)
        counters.pipeline.update_counter("hits", 10)
        counters.pipeline.update_counter("hits", 5)
        assert counters.pipeline.get_counters()["hits"] == 15
    finally:
        _worker_ctx_var.reset(token)


def test_update_counter_max():
    """update_counter with MAX keeps the running maximum."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.pipeline.set_aggregation("peak", Aggregation.MAX)
        counters.pipeline.update_counter("peak", 50)
        counters.pipeline.update_counter("peak", 200)
        counters.pipeline.update_counter("peak", 100)
        assert counters.pipeline.get_counters()["peak"] == 200
    finally:
        _worker_ctx_var.reset(token)


def test_update_counter_min():
    """update_counter with MIN keeps the running minimum."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.pipeline.set_aggregation("latency", Aggregation.MIN)
        counters.pipeline.update_counter("latency", 300)
        counters.pipeline.update_counter("latency", 100)
        counters.pipeline.update_counter("latency", 200)
        assert counters.pipeline.get_counters()["latency"] == 100
    finally:
        _worker_ctx_var.reset(token)


def test_update_counter_average():
    """update_counter with AVERAGE maintains a rolling average."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.pipeline.set_aggregation("cpu_pct", Aggregation.AVERAGE)
        counters.pipeline.update_counter("cpu_pct", 80)  # count=1, avg=80
        counters.pipeline.update_counter("cpu_pct", 60)  # count=2, avg=70
        counters.pipeline.update_counter("cpu_pct", 40)  # count=3, avg=60
        assert counters.pipeline.get_counters()["cpu_pct"] == 60
    finally:
        _worker_ctx_var.reset(token)


def test_update_counter_first_call_initialises():
    """First update_counter call on a new key sets value regardless of aggregation."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.pipeline.set_aggregation("x", Aggregation.MAX)
        counters.pipeline.update_counter("x", 42)
        assert counters.pipeline.get_counters()["x"] == 42
    finally:
        _worker_ctx_var.reset(token)


def test_update_counter_noop_outside_worker():
    """update_counter is a no-op when called outside a worker context."""
    token = _worker_ctx_var.set(None)
    try:
        counters.pipeline.update_counter("anything", 999)
        assert counters.pipeline.get_counters() == {}
    finally:
        _worker_ctx_var.reset(token)


def test_set_counter_resets_average_count():
    """set_counter resets the rolling-average count so subsequent update_counters start fresh."""
    worker = FakeWorker()
    token = _worker_ctx_var.set(worker)
    try:
        counters.pipeline.set_aggregation("avg", Aggregation.AVERAGE)
        counters.pipeline.update_counter("avg", 100)
        counters.pipeline.update_counter("avg", 200)  # avg=150, count=2
        counters.pipeline.set_counter("avg", 0)  # reset; count back to 1
        counters.pipeline.update_counter("avg", 100)  # avg=(0+100)//2=50, count=2
        assert counters.pipeline.get_counters()["avg"] == 50
    finally:
        _worker_ctx_var.reset(token)
