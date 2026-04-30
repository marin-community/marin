# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LogClient buffering, Table semantics, and RemoteLogHandler.

These tests exercise the public client surface. They use
``LogClient.connect`` plus a monkeypatched ``LogServiceClientSync`` /
``StatsServiceClientSync`` to drive the underlying RPC behavior without
spinning a real server. Integration tests covering the wire path live in
``tests/test_stats_service.py`` and ``tests/test_stats_query_drop.py``.
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.client import LogClient, RemoteLogHandler, schema_from_dataclass
from finelog.client import log_client as log_client_mod
from finelog.errors import (
    InvalidNamespaceError,
    QueryResultTooLargeError,
    SchemaConflictError,
    SchemaValidationError,
)
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.store.schema import Column, ColumnType, Schema, schema_to_proto

# ---------------------------------------------------------------------------
# RemoteLogHandler tests against a tiny fake LogClient.
# ---------------------------------------------------------------------------


class FakeLogClient:
    """Minimal LogClient stand-in for the handler tests."""

    def __init__(self, *, fail: bool = False) -> None:
        self.batches: list[tuple[str, list[logging_pb2.LogEntry]]] = []
        self._fail = fail

    def write_batch(self, key: str, messages: list[logging_pb2.LogEntry]) -> None:
        self.batches.append((key, list(messages)))
        if self._fail:
            raise ConnectionError("server unavailable")

    def flush(self, timeout: float | None = None) -> bool:
        return True

    def close(self) -> None:
        pass


def test_handler_writes_batches():
    client = FakeLogClient()
    handler = RemoteLogHandler(client, key="test")
    log = logging.getLogger("test_handler_push")
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        log.info("hello")
        assert len(client.batches) == 1
        assert client.batches[0][1][0].data.endswith("hello")
    finally:
        log.removeHandler(handler)
        handler.close()


def test_no_deadlock_on_write_failure():
    """Handler re-entry on write failure must not deadlock."""
    client = FakeLogClient(fail=True)
    handler = RemoteLogHandler(client, key="test")
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    done = threading.Event()

    def log_one():
        try:
            logging.getLogger("test_deadlock").info("trigger flush")
        finally:
            done.set()

    t = threading.Thread(target=log_one)
    t.start()
    finished = done.wait(timeout=2.0)
    root.removeHandler(handler)
    handler.close()
    t.join(timeout=1.0)
    assert finished, "RemoteLogHandler deadlocked on write failure"


# ---------------------------------------------------------------------------
# LogClient end-to-end tests using monkeypatched RPC clients.
# ---------------------------------------------------------------------------


class _FakeLogServiceClient:
    """Records pushes; can be seeded with errors to raise."""

    def __init__(self, address, **_kwargs):
        self.address = address
        self.pushes: list[logging_pb2.PushLogsRequest] = []
        self.fetches: list[logging_pb2.FetchLogsRequest] = []
        self.errors: list[Exception] = []

    def push_logs(self, request):
        if self.errors:
            raise self.errors.pop(0)
        self.pushes.append(request)
        return logging_pb2.PushLogsResponse()

    def fetch_logs(self, request):
        self.fetches.append(request)
        return logging_pb2.FetchLogsResponse()

    def close(self):
        pass


class _FakeStatsServiceClient:
    """Records register/write/drop/query; tracks state for round-trip tests."""

    def __init__(self, address, **_kwargs):
        self.address = address
        self.registered: dict[str, stats_pb2.Schema] = {}
        self.writes: list[stats_pb2.WriteRowsRequest] = []
        self.drops: list[str] = []
        self.queries: list[str] = []
        self.errors: list[Exception] = []
        # Optional: a callable that returns a pa.Table given the SQL string.
        # Tests that exercise Table.query install this; the default behaves
        # like an empty result set.
        self.query_handler = None

    def register_table(self, request):
        self.registered[request.namespace] = request.schema
        return stats_pb2.RegisterTableResponse(effective_schema=request.schema)

    def write_rows(self, request):
        if self.errors:
            raise self.errors.pop(0)
        self.writes.append(request)
        # Decode the IPC stream so tests can introspect rows.
        return stats_pb2.WriteRowsResponse(rows_written=_decode_ipc_row_count(request.arrow_ipc))

    def drop_table(self, request):
        self.drops.append(request.namespace)
        return stats_pb2.DropTableResponse()

    def query(self, request):
        self.queries.append(request.sql)
        if self.errors:
            raise self.errors.pop(0)
        if self.query_handler is None:
            table = pa.table({})
        else:
            table = self.query_handler(request.sql)
        sink = io.BytesIO()
        with paipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return stats_pb2.QueryResponse(arrow_ipc=sink.getvalue(), row_count=table.num_rows)

    def close(self):
        pass


def _decode_ipc_row_count(blob: bytes) -> int:
    reader = paipc.open_stream(pa.BufferReader(blob))
    table = reader.read_all()
    return table.num_rows


@pytest.fixture
def tracked_clients(monkeypatch):
    """Patch the RPC client classes to record every constructed instance."""
    log_clients: list[_FakeLogServiceClient] = []
    stats_clients: list[_FakeStatsServiceClient] = []

    def log_factory(address, timeout_ms=10_000, interceptors=()):
        c = _FakeLogServiceClient(address, timeout_ms=timeout_ms, interceptors=interceptors)
        log_clients.append(c)
        return c

    def stats_factory(address, timeout_ms=10_000, interceptors=()):
        c = _FakeStatsServiceClient(address, timeout_ms=timeout_ms, interceptors=interceptors)
        stats_clients.append(c)
        return c

    monkeypatch.setattr(log_client_mod, "LogServiceClientSync", log_factory)
    monkeypatch.setattr(log_client_mod, "StatsServiceClientSync", stats_factory)
    return log_clients, stats_clients


def test_connect_returns_usable_client(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        client.write_batch("key", [logging_pb2.LogEntry(source="t", data="hi")])
        assert client.flush(timeout=5.0) is True
        log_clients, _ = tracked_clients
        assert log_clients and log_clients[0].pushes[0].key == "key"
    finally:
        client.close()


def test_close_is_idempotent(tracked_clients):
    client = LogClient.connect("http://h:1")
    client.close()
    # Second call must not raise. write after close raises.
    client.close()
    with pytest.raises(RuntimeError):
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="x")])


def test_connect_accepts_host_port_tuple(tracked_clients):
    client = LogClient.connect(("h", 1234))
    try:
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="x")])
        assert client.flush(timeout=5.0) is True
        log_clients, _ = tracked_clients
        assert log_clients[0].address == "http://h:1234"
    finally:
        client.close()


def test_resolver_runs_per_resolve(tracked_clients):
    addresses = iter(["http://primary:1", "http://secondary:1"])
    resolver_calls: list[str] = []

    def resolver(url: str) -> str:
        resolver_calls.append(url)
        return next(addresses)

    client = LogClient.connect("/system/log-server", resolver=resolver)
    try:
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="x")])
        assert client.flush(timeout=5.0) is True
    finally:
        client.close()
    assert resolver_calls == ["/system/log-server"]


def test_invalidates_on_connection_refused(tracked_clients, monkeypatch):
    """Retryable failure invalidates the cached client; the next send re-resolves.

    The client retries with exponential backoff. To keep this test
    deterministic (rather than racing a 0.5s timer) we shrink the
    initial backoff to ~0 so the bg flush thread retries immediately;
    ``flush()`` then blocks until the row is acknowledged, which is the
    deterministic signal that the retry landed.
    """
    monkeypatch.setattr(log_client_mod, "_BACKOFF_INITIAL", 1e-9)
    monkeypatch.setattr(log_client_mod, "_BACKOFF_MAX", 1e-9)
    log_clients, _ = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="primer")])
        assert client.flush(timeout=5.0) is True
        # Seed a single retryable error; the next send will fail and trigger
        # invalidation, then the retry succeeds against a freshly resolved client.
        log_clients[0].errors.append(ConnectError(Code.UNAVAILABLE, "down"))
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="retry")])
        assert client.flush(timeout=5.0) is True
        assert len(log_clients) >= 2, "expected re-resolution to construct a new client"
        assert any(p.entries[0].data == "retry" for p in log_clients[1].pushes)
    finally:
        client.close()


def test_query_round_trips(tracked_clients):
    log_clients, _ = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        client.query(logging_pb2.FetchLogsRequest(source="key", max_lines=10))
        assert log_clients[0].fetches[0].source == "key"
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Stats-side: get_table, write, drop_table.
# ---------------------------------------------------------------------------


@dataclass
class WorkerStat:
    worker_id: str
    timestamp_ms: int
    mem_bytes: int
    note: str | None = None


def test_get_table_with_dataclass_round_trips(tracked_clients):
    _, stats_clients = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        assert table.namespace == "iris.worker"
        assert tuple(c.name for c in table.schema.columns) == ("worker_id", "timestamp_ms", "mem_bytes", "note")
        # Writes are routed through the Table buffer; flush forces a send.
        table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=128, note="ok")])
        assert table.flush(timeout=5.0) is True
        # Decode the IPC bytes to confirm the row landed correctly.
        write_req = stats_clients[0].writes[0]
        decoded = paipc.open_stream(pa.BufferReader(write_req.arrow_ipc)).read_all()
        assert decoded.num_rows == 1
        assert decoded.column_names == ["worker_id", "timestamp_ms", "mem_bytes", "note"]
        assert decoded.column("worker_id").to_pylist() == ["w-1"]
    finally:
        client.close()


def test_get_table_with_explicit_schema(tracked_clients):
    schema = Schema(
        columns=(
            Column(name="ts", type=ColumnType.INT64, nullable=False),
            Column(name="value", type=ColumnType.FLOAT64, nullable=False),
        ),
        key_column="ts",
    )
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.metric", schema)
        assert table.schema.key_column == "ts"
        table.write([SimpleNamespace(ts=1, value=1.5)])
        assert table.flush(timeout=5.0) is True
    finally:
        client.close()


def test_get_table_rejects_log_namespace(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        with pytest.raises(InvalidNamespaceError):
            client.get_table("log", WorkerStat)
    finally:
        client.close()


def test_drop_table_calls_server(tracked_clients):
    _, stats_clients = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        client.get_table("iris.worker", WorkerStat)
        client.drop_table("iris.worker")
        assert stats_clients[0].drops == ["iris.worker"]
    finally:
        client.close()


def test_drop_table_rejects_log_namespace(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        with pytest.raises(InvalidNamespaceError):
            client.drop_table("log")
    finally:
        client.close()


def test_drop_table_unknown_is_no_op(tracked_clients, monkeypatch):
    _, stats_clients = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        # Pre-create the stats client and seed it to raise NOT_FOUND.
        client.get_table("iris.worker", WorkerStat)
        original_drop = stats_clients[0].drop_table

        def fail_drop(request):
            raise ConnectError(Code.NOT_FOUND, "namespace not registered")

        stats_clients[0].drop_table = fail_drop  # type: ignore[method-assign]
        client.drop_table("iris.unknown")  # must not raise
        stats_clients[0].drop_table = original_drop  # type: ignore[method-assign]
    finally:
        client.close()


def test_get_table_propagates_schema_conflict(tracked_clients, monkeypatch):
    _, stats_clients = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        # Touch the stats client so it exists.
        client.get_table("iris.metric", WorkerStat)

        def conflict(request):
            raise ConnectError(Code.FAILED_PRECONDITION, "type mismatch")

        stats_clients[0].register_table = conflict  # type: ignore[method-assign]
        with pytest.raises(SchemaConflictError):
            client.get_table("iris.other", WorkerStat)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Table.query: SQL round-trip via the StatsService.
# ---------------------------------------------------------------------------


def test_table_query_round_trips(tracked_clients):
    _, stats_clients = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)

        def handler(_sql: str) -> pa.Table:
            return pa.table({"worker_id": ["w-1", "w-2"], "mem_bytes": [10, 20]})

        stats_clients[0].query_handler = handler
        result = table.query('SELECT worker_id, mem_bytes FROM "iris.worker"')
        assert result.column_names == ["worker_id", "mem_bytes"]
        assert result.column("worker_id").to_pylist() == ["w-1", "w-2"]
        assert stats_clients[0].queries == ['SELECT worker_id, mem_bytes FROM "iris.worker"']
    finally:
        client.close()


def test_table_query_raises_on_too_large(tracked_clients):
    _, stats_clients = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        stats_clients[0].query_handler = lambda _sql: pa.table({"worker_id": ["w"] * 5})
        with pytest.raises(QueryResultTooLargeError):
            table.query('SELECT * FROM "iris.worker"', max_rows=2)
    finally:
        client.close()


def test_table_query_translates_invalid_argument(tracked_clients):
    _, stats_clients = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        stats_clients[0].errors.append(ConnectError(Code.INVALID_ARGUMENT, "syntax error"))
        with pytest.raises(SchemaValidationError):
            table.query("not valid sql")
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Table buffering and overflow.
# ---------------------------------------------------------------------------


def test_close_drains_pending_log_rows(tracked_clients):
    log_clients, _ = tracked_clients
    client = LogClient.connect("http://h:1")
    entry = logging_pb2.LogEntry(source="t", data="line")
    client.write_batch("k", [entry, entry])
    client.close()
    # Close drains, so the entries land before close returns.
    assert log_clients[0].pushes
    total = sum(len(p.entries) for p in log_clients[0].pushes)
    assert total == 2


def test_table_overflow_drops_oldest(tracked_clients, caplog):
    """Saturate beyond the row cap; oldest rows are dropped, no block.

    `_trim_oldest_locked` runs synchronously inside `Table.write()` once
    the queue exceeds the cap, so the warning is emitted on the calling
    thread — no need to wait on the bg flush thread. We also block the
    bg thread's send so the queue cannot drain underneath us, otherwise
    a fast bg flush would empty the queue between writes and the
    overflow trigger would never fire.
    """
    _, stats_clients = tracked_clients
    client = LogClient.connect("http://h:1")
    try:
        client.get_table("iris.worker", WorkerStat)  # creates the stats RPC client
        flush_lock = threading.Event()

        def blocked_write(request):
            flush_lock.wait()
            return stats_pb2.WriteRowsResponse(rows_written=0)

        stats_clients[0].write_rows = blocked_write  # type: ignore[method-assign]
        table = client._tables["iris.worker"]
        table._max_buffer_rows = 4
        table._max_buffer_bytes = 1024
        table._batch_rows = 1_000_000  # disable size-triggered notify; rely on time-based wakeup only
        # Bypass the rate limiter so the very first overflow logs.
        table._overflow_log_limiter = log_client_mod.RateLimiter(interval_seconds=0)
        client_logger = logging.getLogger("finelog.client.log_client")
        client_logger.addHandler(caplog.handler)
        client_logger.setLevel(logging.WARNING)
        try:
            for i in range(20):
                table.write([WorkerStat(worker_id=f"w-{i}", timestamp_ms=i, mem_bytes=i)])
            # Overflow trim runs in the calling thread; assert immediately.
            assert any("buffer overflow" in r.message for r in caplog.records)
            # Only the four most-recent rows survive; older ones were dropped.
            with table._cond:
                surviving_ids = [item.payload.worker_id for item in table._queue]
            assert surviving_ids == ["w-16", "w-17", "w-18", "w-19"]
        finally:
            client_logger.removeHandler(caplog.handler)
            flush_lock.set()  # unblock the bg thread so close() returns
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Dataclass schema inference unit tests.
# ---------------------------------------------------------------------------


def test_schema_from_dataclass_basic():
    @dataclass
    class Stat:
        worker_id: str
        timestamp_ms: int
        mem_bytes: int
        note: str | None = None

    s = schema_from_dataclass(Stat)
    assert s.key_column == ""
    names = [c.name for c in s.columns]
    assert names == ["worker_id", "timestamp_ms", "mem_bytes", "note"]
    note_col = next(c for c in s.columns if c.name == "note")
    assert note_col.nullable is True


def test_schema_from_dataclass_classvar_key():
    @dataclass
    class Stat:
        key_column: ClassVar[str] = "ts"
        worker_id: str
        ts: int
        mem_bytes: int

    s = schema_from_dataclass(Stat)
    assert s.key_column == "ts"


def test_schema_from_dataclass_rejects_unsupported_type():
    @dataclass
    class Stat:
        worker_id: str
        timestamp_ms: int
        labels: list[str]

    with pytest.raises(SchemaValidationError):
        schema_from_dataclass(Stat)


# ---------------------------------------------------------------------------
# Wire path: end-to-end with an in-process LogServiceImpl.
# ---------------------------------------------------------------------------


def test_remote_log_handler_writes_via_log_client(tracked_clients):
    client = LogClient.connect("http://h:1")
    handler = RemoteLogHandler(client, key="proc")
    log = logging.getLogger("e2e_handler")
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    try:
        log.info("end-to-end")
        assert client.flush(timeout=5.0) is True
        log_clients, _ = tracked_clients
        assert log_clients[0].pushes[0].key == "proc"
    finally:
        log.removeHandler(handler)
        handler.close()
        client.close()


def test_table_flush_waits_for_in_flight(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=1) for _ in range(10)])
        assert table.flush(timeout=5.0) is True
        # All rows landed exactly once.
        _, stats_clients = tracked_clients
        total_rows = sum(_decode_ipc_row_count(w.arrow_ipc) for w in stats_clients[0].writes)
        assert total_rows == 10
    finally:
        client.close()


def test_table_close_drains_queue(tracked_clients):
    client = LogClient.connect("http://h:1")
    table = client.get_table("iris.worker", WorkerStat)
    table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=1) for _ in range(5)])
    table.close()
    _, stats_clients = tracked_clients
    total = sum(_decode_ipc_row_count(w.arrow_ipc) for w in stats_clients[0].writes)
    assert total == 5
    client.close()


def test_table_close_drains_queue_when_thread_starts_late(monkeypatch):
    sent: list[list[object]] = []
    thread_targets = []

    class DeferredThread:
        def __init__(self, *, target, name, daemon):
            self._target = target
            self.name = name
            self.daemon = daemon
            thread_targets.append(target)

        def start(self):
            pass

        def join(self, timeout=None):
            self._target()

    monkeypatch.setattr(log_client_mod.threading, "Thread", DeferredThread)

    table = log_client_mod.Table(
        namespace="iris.worker",
        schema=Schema(columns=()),
        flusher=lambda _namespace, rows: sent.append(list(rows)),
    )
    table.write([{"worker_id": "w-1"}, {"worker_id": "w-2"}])
    table.close()

    assert len(thread_targets) == 1
    assert sent == [[{"worker_id": "w-1"}, {"worker_id": "w-2"}]]


def test_schema_from_proto_consistency():
    """schema_to_proto is the inverse of schema_from_proto for inferred schemas."""
    s = schema_from_dataclass(WorkerStat)
    proto = schema_to_proto(s)
    # Names round-trip in order.
    assert [c.name for c in proto.columns] == [c.name for c in s.columns]
