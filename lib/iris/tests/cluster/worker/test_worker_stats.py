# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the iris.worker stats namespace emission path.

Stage 5 of the stats service: each worker registers ``iris.worker`` once
on startup and writes one row per heartbeat. These tests use a fake
``LogClient`` plus a fake ``Table`` so we can assert directly against the
rows the worker produces, without spinning up a real finelog server.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from iris.cluster.worker.stats import (
    WORKER_STATS_NAMESPACE,
    IrisWorkerStat,
)
from iris.cluster.worker.worker import Worker, WorkerConfig, _StatsState
from iris.rpc import worker_pb2
from rigging.timing import Duration


class _FakeTable:
    """In-memory Table standing in for ``finelog.client.Table``."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.rows: list[Any] = []
        self.closed = False

    def write(self, rows):
        self.rows.extend(rows)

    def flush(self, timeout=None):
        return True

    def close(self):
        self.closed = True


class _FakeLogClient:
    """Minimal LogClient stand-in: tracks get_table calls and Tables."""

    def __init__(self, *, register_error: BaseException | None = None) -> None:
        self.register_calls: list[tuple[str, type]] = []
        self._register_error = register_error
        self.tables: dict[str, _FakeTable] = {}

    def get_table(self, namespace: str, schema):
        self.register_calls.append((namespace, schema))
        if self._register_error is not None:
            raise self._register_error
        if namespace not in self.tables:
            self.tables[namespace] = _FakeTable(namespace)
        return self.tables[namespace]

    def write_batch(self, key, messages):
        pass

    def flush(self, timeout=None):
        return True

    def close(self):
        for tbl in self.tables.values():
            tbl.close()


def _build_worker(tmp_path) -> Worker:
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        cache_dir=tmp_path / "cache",
        worker_id="w-1",
        poll_interval=Duration.from_seconds(0.1),
    )
    # Worker.__init__ reads a number of components; pass minimal stubs.
    from unittest.mock import MagicMock

    runtime = MagicMock()
    runtime.discover_containers.return_value = []
    runtime.remove_all_iris_containers.return_value = 0
    bundle_store = MagicMock()
    worker = Worker(config, bundle_store=bundle_store, container_runtime=runtime)
    return worker


def test_emits_stat_per_heartbeat(tmp_path):
    worker = _build_worker(tmp_path)
    fake_client = _FakeLogClient()
    worker._log_client = fake_client

    try:
        # Drive a single heartbeat; the worker should register iris.worker
        # and append a row to the Table.
        resp = worker.handle_ping(worker_pb2.Worker.PingRequest())
        assert isinstance(resp, worker_pb2.Worker.PingResponse)
    finally:
        worker.stop()

    assert fake_client.register_calls, "expected get_table to be called"
    namespace, schema = fake_client.register_calls[0]
    assert namespace == WORKER_STATS_NAMESPACE
    assert schema is IrisWorkerStat
    table = fake_client.tables[WORKER_STATS_NAMESPACE]
    # Exactly one row per heartbeat — no synthetic register-time row.
    assert len(table.rows) == 1
    heartbeat_row = table.rows[0]
    assert heartbeat_row.worker_id == "w-1"
    assert heartbeat_row.address  # whatever resolve_address picks
    # ``ts`` is a tz-naive datetime representing the UTC moment of the
    # heartbeat (see worker._now_dt). The stats namespace stores
    # ``pa.timestamp("ms")``, which has no tz; we encode that convention
    # at the worker boundary.
    assert isinstance(heartbeat_row.ts, datetime)
    assert heartbeat_row.ts.tzinfo is None


def test_ts_round_trips_through_pyarrow_at_ms_precision():
    """A known UTC ts written to a pa.timestamp('ms') column must come
    back exactly (after ms truncation) — this is the contract the
    controller-side ``WHERE ts > (now() AT TIME ZONE 'UTC')::TIMESTAMP``
    query relies on.
    """
    import pyarrow as pa

    fixed = datetime(2026, 4, 29, 12, 34, 56, 789_000, tzinfo=timezone.utc)
    naive_utc = fixed.astimezone(timezone.utc).replace(tzinfo=None)
    arr = pa.array([naive_utc], type=pa.timestamp("ms"))
    round_tripped = arr.to_pylist()[0]
    # Naive in, naive out; the stored value matches the UTC wall-clock at
    # ms precision.
    assert round_tripped.tzinfo is None
    assert round_tripped == naive_utc


def test_register_failure_does_not_break_ping(tmp_path):
    """A schema-register error must not propagate from ``handle_ping``."""
    from finelog.client import StatsError

    worker = _build_worker(tmp_path)
    worker._log_client = _FakeLogClient(register_error=StatsError("boom"))

    try:
        # Ping must succeed even when the stats register raises. The next
        # ping must also not retry register (one-shot give-up).
        worker.handle_ping(worker_pb2.Worker.PingRequest())
        worker.handle_ping(worker_pb2.Worker.PingRequest())
        assert worker._stats_state is _StatsState.FAILED
        assert worker._stats_table is None
    finally:
        worker.stop()


def test_subsequent_heartbeats_reuse_table(tmp_path):
    worker = _build_worker(tmp_path)
    fake_client = _FakeLogClient()
    worker._log_client = fake_client

    try:
        worker.handle_ping(worker_pb2.Worker.PingRequest())
        worker.handle_ping(worker_pb2.Worker.PingRequest())
        worker.handle_ping(worker_pb2.Worker.PingRequest())
    finally:
        worker.stop()

    # get_table is called exactly once; subsequent pings reuse the cached
    # Table.
    assert len(fake_client.register_calls) == 1
    table = fake_client.tables[WORKER_STATS_NAMESPACE]
    # One row per heartbeat. No synthetic row is emitted at register time.
    assert len(table.rows) == 3


def test_register_schema_bug_propagates(tmp_path):
    """A non-transport error (e.g. a schema-inference bug) must propagate
    out of ``handle_ping`` rather than getting swallowed by the soft-fail
    path. This is the contract that lets such bugs surface in CI rather
    than silently producing an empty stats namespace in production.
    """

    class _BrokenLogClient(_FakeLogClient):
        def get_table(self, namespace, schema):  # type: ignore[override]
            self.register_calls.append((namespace, schema))
            raise TypeError("schema field has unsupported type")

    worker = _build_worker(tmp_path)
    worker._log_client = _BrokenLogClient()

    try:
        with pytest.raises(TypeError):
            worker.handle_ping(worker_pb2.Worker.PingRequest())
    finally:
        worker.stop()


def test_stat_dataclass_registers_with_ts_as_key():
    """Smoke-test schema inference for the IrisWorkerStat dataclass.

    A schema with ``key_column = "ts"`` should infer cleanly via the
    finelog ``schema_from_dataclass`` path. This catches accidental
    field-type drift that would silently break worker emission.
    """
    from finelog.client import schema_from_dataclass

    schema = schema_from_dataclass(IrisWorkerStat)
    assert schema.key_column == "ts"
    column_names = [c.name for c in schema.columns]
    assert "ts" in column_names
    assert "worker_id" in column_names
    assert "cpu_pct" in column_names
