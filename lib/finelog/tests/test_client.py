# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from finelog.client import FlushResult, LogClient, RemoteLogHandler, StoragePolicy, schema_from_dataclass
from finelog.client import log_client as log_client_mod
from finelog.errors import (
    InvalidNamespaceError,
    QueryResultTooLargeError,
    SchemaValidationError,
)
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.schema import Column, Schema, schema_from_proto, schema_to_proto


class FakeLogClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.batches: list[tuple[str, list[logging_pb2.LogEntry]]] = []
        self._fail = fail

    def write_batch(self, key: str, messages: list[logging_pb2.LogEntry]) -> None:
        self.batches.append((key, list(messages)))
        if self._fail:
            raise ConnectionError("server unavailable")

    def flush(self, timeout: float | None = None) -> FlushResult:
        return FlushResult.SUCCEEDED

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


class _FakeStatsServiceClient:
    def __init__(self, address, **_kwargs):
        self.address = address
        self.registered: dict[str, stats_pb2.Schema] = {}
        self.registered_policies: dict[str, stats_pb2.StoragePolicy] = {}
        self.writes: list[stats_pb2.WriteRowsRequest] = []
        self.drops: list[str] = []
        self.queries: list[str] = []
        self.errors: list[Exception] = []
        self.query_handler = None

    def register_table(self, request):
        self.registered[request.namespace] = request.schema
        self.registered_policies[request.namespace] = request.storage_policy
        return stats_pb2.RegisterTableResponse(
            effective_schema=request.schema,
            effective_policy=request.storage_policy,
        )

    def write_rows(self, request):
        if self.errors:
            raise self.errors.pop(0)
        self.writes.append(request)
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


def _decode_ipc_table(blob: bytes) -> pa.Table:
    reader = paipc.open_stream(pa.BufferReader(blob))
    return reader.read_all()


@pytest.fixture
def tracked_clients(monkeypatch):
    """Patch the StatsService client class to record every constructed instance."""
    clients: list[_FakeStatsServiceClient] = []

    def stats_factory(address, timeout_ms=10_000, interceptors=(), **_kwargs):
        c = _FakeStatsServiceClient(address, timeout_ms=timeout_ms, interceptors=interceptors)
        clients.append(c)
        return c

    monkeypatch.setattr(log_client_mod, "StatsServiceClientSync", stats_factory)
    return clients


class _FakeLogServiceClient:
    def __init__(self, address, **_kwargs):
        self.address = address
        self.requests: list[logging_pb2.FetchLogsRequest] = []
        self.response: logging_pb2.FetchLogsResponse = logging_pb2.FetchLogsResponse()

    def fetch_logs(self, request):
        self.requests.append(request)
        return self.response

    def close(self):
        pass


@pytest.fixture
def tracked_log_service_clients(monkeypatch):
    """Patch LogServiceClientSync; expose request/response on the singleton fake."""
    fake = _FakeLogServiceClient(address=None)

    def factory(address, timeout_ms=10_000, interceptors=(), **_kwargs):
        fake.address = address
        return fake

    monkeypatch.setattr(log_client_mod, "LogServiceClientSync", factory)
    return fake


def test_connect_returns_usable_client(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        client.write_batch("key", [logging_pb2.LogEntry(source="t", data="hi")])
        assert client.flush(timeout=5.0) == FlushResult.SUCCEEDED
        assert tracked_clients and tracked_clients[0].writes[0].namespace == "log"
        decoded = _decode_ipc_table(tracked_clients[0].writes[0].arrow_ipc)
        assert decoded.column("key").to_pylist() == ["key"]
        assert decoded.column("data").to_pylist() == ["hi"]
    finally:
        client.close()


def test_close_is_idempotent(tracked_clients):
    client = LogClient.connect("http://h:1")
    client.close()
    client.close()
    with pytest.raises(RuntimeError):
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="x")])


def test_connect_accepts_host_port_tuple(tracked_clients):
    client = LogClient.connect(("h", 1234))
    try:
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="x")])
        assert client.flush(timeout=5.0) == FlushResult.SUCCEEDED
        assert tracked_clients[0].address == "http://h:1234"
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
        assert client.flush(timeout=5.0) == FlushResult.SUCCEEDED
    finally:
        client.close()
    assert resolver_calls == ["/system/log-server"]


def test_write_batch_not_blocked_by_in_progress_resolve(tracked_clients):
    """A blocked resolver must not wedge concurrent log writes.

    The background flush thread resolves the stats endpoint under the client
    lock, and in iris that resolver issues a blocking controller RPC. A
    foreground log emit (``write_batch`` -> ``_get_log_table``) must not wait on
    that same lock, or a shutdown-path ``logger.warning`` deadlocks teardown
    against a hung resolve (observed as an iris smoke teardown timeout).
    """
    resolving = threading.Event()
    release = threading.Event()

    def resolver(url: str) -> str:
        resolving.set()
        assert release.wait(timeout=10.0), "resolver was never released"
        return url

    wrote = threading.Event()

    def do_second_write():
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="second")])
        wrote.set()

    client = LogClient.connect("http://h:1", resolver=resolver)
    second_write = threading.Thread(target=do_second_write)
    try:
        # First write spins up the log Table; its flush thread enters the
        # resolver and (pre-fix) parks there holding the client lock.
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="first")])
        assert resolving.wait(timeout=5.0), "flush thread never reached the resolver"

        # A second write must not block on the lock held during the resolve.
        second_write.start()
        assert wrote.wait(timeout=2.0), "write_batch blocked behind an in-progress resolve"
    finally:
        release.set()
        second_write.join(timeout=5.0)
        client.close()


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
    client = LogClient.connect("http://h:1")
    try:
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="primer")])
        assert client.flush(timeout=5.0) == FlushResult.SUCCEEDED
        tracked_clients[0].errors.append(ConnectError(Code.UNAVAILABLE, "down"))
        client.write_batch("k", [logging_pb2.LogEntry(source="t", data="retry")])
        assert client.flush(timeout=5.0) == FlushResult.SUCCEEDED
        assert len(tracked_clients) >= 2, "expected re-resolution to construct a new client"

        def _retry_landed(req):
            decoded = _decode_ipc_table(req.arrow_ipc)
            return "retry" in decoded.column("data").to_pylist()

        assert any(_retry_landed(w) for w in tracked_clients[1].writes)
    finally:
        client.close()


def test_fetch_logs_round_trips(tracked_log_service_clients):
    client = LogClient.connect("http://h:1")
    try:
        request = logging_pb2.FetchLogsRequest(source="key", max_lines=10)
        canned = logging_pb2.FetchLogsResponse(
            entries=[logging_pb2.LogEntry(source="stdout", data="hi", level=2)],
            cursor=42,
        )
        canned.entries[0].timestamp.epoch_ms = 1700000000000
        tracked_log_service_clients.response = canned

        resp = client.fetch_logs(request)

        assert tracked_log_service_clients.requests == [request]
        assert resp.cursor == 42
        assert [e.data for e in resp.entries] == ["hi"]
    finally:
        client.close()


@dataclass
class WorkerStat:
    worker_id: str
    timestamp_ms: int
    mem_bytes: int
    note: str | None = None


def test_get_table_with_dataclass_round_trips(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        assert table.namespace == "iris.worker"
        assert tuple(c.name for c in table.schema.columns) == ("worker_id", "timestamp_ms", "mem_bytes", "note")
        table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=128, note="ok")])
        assert table.flush(timeout=5.0) == FlushResult.SUCCEEDED
        write_req = tracked_clients[0].writes[0]
        decoded = paipc.open_stream(pa.BufferReader(write_req.arrow_ipc)).read_all()
        assert decoded.num_rows == 1
        assert decoded.column_names == ["worker_id", "timestamp_ms", "mem_bytes", "note"]
        assert decoded.column("worker_id").to_pylist() == ["w-1"]
    finally:
        client.close()


def test_get_table_with_explicit_schema(tracked_clients):
    schema = Schema(
        columns=(
            Column(name="ts", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            Column(name="value", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=False),
        ),
        key_column="ts",
    )
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.metric", schema)
        assert table.schema.key_column == "ts"
        table.write([SimpleNamespace(ts=1, value=1.5)])
        assert table.flush(timeout=5.0) == FlushResult.SUCCEEDED
    finally:
        client.close()


def test_get_table_forwards_storage_policy(tracked_clients):
    """An explicit StoragePolicy on get_table is sent on the register_table request."""
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table(
            "iris.worker",
            WorkerStat,
            storage_policy=StoragePolicy(max_bytes=100, max_age_seconds=60),
        )
        # Registration is deferred to the flush thread; force it with a write+flush.
        table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=1)])
        assert table.flush(timeout=5.0) == FlushResult.SUCCEEDED
        policy = tracked_clients[0].registered_policies["iris.worker"]
        assert policy.max_bytes == 100
        assert policy.max_age_seconds == 60
        assert policy.max_segments == 0  # unset → proto3 zero
    finally:
        client.close()


def test_get_table_default_policy_is_empty(tracked_clients):
    """No policy argument sends an empty proto (all zeros = inherit defaults)."""
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        # Registration is deferred to the flush thread; force it with a write+flush.
        table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=1)])
        assert table.flush(timeout=5.0) == FlushResult.SUCCEEDED
        policy = tracked_clients[0].registered_policies["iris.worker"]
        assert policy.max_bytes == 0
        assert policy.max_age_seconds == 0
        assert policy.max_segments == 0
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
    client = LogClient.connect("http://h:1")
    try:
        client.get_table("iris.worker", WorkerStat)
        client.drop_table("iris.worker")
        assert tracked_clients[0].drops == ["iris.worker"]
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
    client = LogClient.connect("http://h:1")
    try:
        client.get_table("iris.worker", WorkerStat)
        client.drop_table("iris.worker")  # first drop lazily constructs the stats client

        def fail_drop(request):
            raise ConnectError(Code.NOT_FOUND, "namespace not registered")

        tracked_clients[0].drop_table = fail_drop  # type: ignore[method-assign]
        client.drop_table("iris.unknown")  # must not raise
    finally:
        client.close()


def test_get_table_registration_conflict_drops_batch(tracked_clients, monkeypatch):
    """A non-retryable registration failure is handled as a flush failure.

    Registration happens on the flush thread, so a schema conflict cannot
    propagate to the caller of get_table. The offending batch is dropped (the
    error is non-retryable) and the Table stays usable without crashing.
    """

    def conflict(self, request):
        raise ConnectError(Code.FAILED_PRECONDITION, "type mismatch")

    monkeypatch.setattr(_FakeStatsServiceClient, "register_table", conflict)
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=1)])
        # Non-retryable: the batch is dropped, the flush resolves, nothing raises.
        assert table.flush(timeout=5.0) == FlushResult.SUCCEEDED
        assert tracked_clients[0].writes == []
    finally:
        client.close()


def test_format_exc_summary_surfaces_connect_detail():
    """A ConnectError's server detail must survive into the log summary.

    A bare ``FAILED_PRECONDITION`` is undiagnosable; the schema-conflict detail
    it carries (which column, which mismatch) is the actionable part.
    """
    summary = log_client_mod._format_exc_summary(
        ConnectError(Code.FAILED_PRECONDITION, 'column "mem_bytes": type mismatch registered=int64 requested=float64')
    )
    assert "FAILED_PRECONDITION" in summary
    assert 'column "mem_bytes": type mismatch registered=int64 requested=float64' in summary


def test_get_table_retries_transient_registration_failure(tracked_clients, monkeypatch):
    """A retryable registration failure is retried on the flush thread.

    The first register attempt raises UNAVAILABLE; the flush thread backs off
    and retries, then the write lands once registration succeeds. ``get_table``
    itself never blocks or raises.
    """
    monkeypatch.setattr(log_client_mod, "_BACKOFF_INITIAL", 1e-9)
    monkeypatch.setattr(log_client_mod, "_BACKOFF_MAX", 1e-9)

    calls = {"n": 0}
    real_register = _FakeStatsServiceClient.register_table

    def flaky_register(self, request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectError(Code.UNAVAILABLE, "down")
        return real_register(self, request)

    monkeypatch.setattr(_FakeStatsServiceClient, "register_table", flaky_register)
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=128)])
        assert table.flush(timeout=5.0) == FlushResult.SUCCEEDED
        # First attempt failed and re-resolved the endpoint; the retry registered
        # and wrote against the freshly constructed client.
        assert calls["n"] >= 2
        landed = any(w.namespace == "iris.worker" for c in tracked_clients for w in c.writes)
        assert landed
    finally:
        client.close()


def test_table_query_round_trips(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)

        def handler(_sql: str) -> pa.Table:
            return pa.table({"worker_id": ["w-1", "w-2"], "mem_bytes": [10, 20]})

        table.query("SELECT 1")  # lazily construct the stats client
        tracked_clients[0].query_handler = handler
        result = table.query('SELECT worker_id, mem_bytes FROM "iris.worker"')
        assert result.column_names == ["worker_id", "mem_bytes"]
        assert result.column("worker_id").to_pylist() == ["w-1", "w-2"]
        assert tracked_clients[0].queries[-1] == 'SELECT worker_id, mem_bytes FROM "iris.worker"'
    finally:
        client.close()


def test_table_query_raises_on_too_large(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        table.query("SELECT 1")  # lazily construct the stats client
        tracked_clients[0].query_handler = lambda _sql: pa.table({"worker_id": ["w"] * 5})
        with pytest.raises(QueryResultTooLargeError):
            table.query('SELECT * FROM "iris.worker"', max_rows=2)
    finally:
        client.close()


def test_client_query_round_trips_without_table(tracked_clients):
    """LogClient.query lets the CLI run SQL without registering a Table first."""
    client = LogClient.connect("http://h:1")
    try:
        # First call lazily constructs the stats client; result is the fake's
        # empty default. Subsequent calls hit the handler.
        empty = client.query("SELECT 1 AS n")
        assert empty.num_rows == 0
        tracked_clients[0].query_handler = lambda _sql: pa.table({"n": [3]})
        result = client.query('SELECT COUNT(*) AS n FROM "iris.worker"')
        assert result.column("n").to_pylist() == [3]
        assert tracked_clients[0].queries[-1] == 'SELECT COUNT(*) AS n FROM "iris.worker"'
    finally:
        client.close()


def test_client_query_raises_on_too_large(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        client.query("SELECT 1")  # construct the stats client
        tracked_clients[0].query_handler = lambda _sql: pa.table({"x": list(range(5))})
        with pytest.raises(QueryResultTooLargeError):
            client.query('SELECT * FROM "iris.worker"', max_rows=2)
    finally:
        client.close()


def test_table_query_translates_invalid_argument(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        table.query("SELECT 1")  # lazily construct the stats client
        tracked_clients[0].errors.append(ConnectError(Code.INVALID_ARGUMENT, "syntax error"))
        with pytest.raises(SchemaValidationError):
            table.query("not valid sql")
    finally:
        client.close()


def test_close_drains_pending_log_rows(tracked_clients):
    client = LogClient.connect("http://h:1")
    entry = logging_pb2.LogEntry(source="t", data="line")
    client.write_batch("k", [entry, entry])
    client.close()
    assert tracked_clients[0].writes
    total = sum(_decode_ipc_table(w.arrow_ipc).num_rows for w in tracked_clients[0].writes)
    assert total == 2


def test_table_overflow_drops_oldest(tracked_clients, caplog):
    """Saturate beyond the row cap; oldest rows are dropped, no block.

    `_trim_oldest_locked` runs synchronously inside `Table.write()` once
    the queue exceeds the cap, so the warning is emitted on the calling
    thread. We stop the bg flush thread first so its cond-wake on every
    trim can't race the test into draining the queue mid-loop — the
    semantic under test is purely the synchronous trim path.
    """
    client = LogClient.connect("http://h:1")
    try:
        client.get_table("iris.worker", WorkerStat)
        table = client._tables["iris.worker"]
        # Stop the bg thread so trim_oldest's notify_all has no consumer.
        # Re-enable writes by clearing the flag once the thread has exited.
        with table._cond:
            table._closing = True
            table._cond.notify_all()
        table._thread.join(timeout=2.0)
        with table._cond:
            table._closing = False
        table._max_buffer_rows = 4
        table._max_buffer_bytes = 1024
        table._batch_rows = 1_000_000
        # Bypass the rate limiter so the very first overflow logs.
        table._overflow_log_limiter = log_client_mod.RateLimiter(interval_seconds=0)
        client_logger = logging.getLogger("finelog.client.log_client")
        client_logger.addHandler(caplog.handler)
        client_logger.setLevel(logging.WARNING)
        try:
            for i in range(20):
                table.write([WorkerStat(worker_id=f"w-{i}", timestamp_ms=i, mem_bytes=i)])
            assert any("buffer overflow" in r.message for r in caplog.records)
            with table._cond:
                surviving_ids = [item.payload.worker_id for item in table._queue]
            assert surviving_ids == ["w-16", "w-17", "w-18", "w-19"]
        finally:
            client_logger.removeHandler(caplog.handler)
    finally:
        client.close()


def test_schema_from_dataclass_all_columns_nullable():
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
    # Every column is nullable regardless of whether the field is Optional:
    # finelog adopts compacted segments as all-nullable, so a non-nullable
    # registration would conflict with its own adopted schema and wedge writes.
    assert all(c.nullable for c in s.columns)


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


def test_remote_log_handler_writes_via_log_client(tracked_clients):
    client = LogClient.connect("http://h:1")
    handler = RemoteLogHandler(client, key="proc")
    log = logging.getLogger("e2e_handler")
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    try:
        log.info("end-to-end")
        assert client.flush(timeout=5.0) == FlushResult.SUCCEEDED
        decoded = _decode_ipc_table(tracked_clients[0].writes[0].arrow_ipc)
        assert decoded.column("key").to_pylist() == ["proc"]
    finally:
        log.removeHandler(handler)
        handler.close()
        client.close()


def test_table_flush_waits_for_in_flight(tracked_clients):
    client = LogClient.connect("http://h:1")
    try:
        table = client.get_table("iris.worker", WorkerStat)
        table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=1) for _ in range(10)])
        assert table.flush(timeout=5.0) == FlushResult.SUCCEEDED
        total_rows = sum(_decode_ipc_row_count(w.arrow_ipc) for w in tracked_clients[0].writes)
        assert total_rows == 10
    finally:
        client.close()


def test_table_close_drains_queue(tracked_clients):
    client = LogClient.connect("http://h:1")
    table = client.get_table("iris.worker", WorkerStat)
    table.write([WorkerStat(worker_id="w-1", timestamp_ms=1, mem_bytes=1) for _ in range(5)])
    table.close()
    total = sum(_decode_ipc_row_count(w.arrow_ipc) for w in tracked_clients[0].writes)
    assert total == 5
    client.close()


def test_table_close_drains_queue_when_thread_starts_late(monkeypatch):
    sent: list[pa.RecordBatch] = []
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

    schema = Schema(
        columns=(Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),),
    )
    table = log_client_mod.Table(
        namespace="iris.worker",
        schema=schema,
        flusher=lambda _namespace, batch: sent.append(batch),
    )
    table.write([SimpleNamespace(worker_id="w-1"), SimpleNamespace(worker_id="w-2")])
    table.close()

    assert len(thread_targets) == 1
    assert len(sent) == 1
    assert sent[0].column("worker_id").to_pylist() == ["w-1", "w-2"]


def test_schema_from_proto_consistency():
    s = schema_from_dataclass(WorkerStat)
    proto = schema_to_proto(s)
    assert len(proto.columns) == len(s.columns)
    for proto_col, src_col in zip(proto.columns, s.columns, strict=True):
        assert proto_col.name == src_col.name
        assert proto_col.type == src_col.type
        assert proto_col.nullable == src_col.nullable
        assert proto_col.index.trigram == src_col.trigram_index


def test_trigram_index_round_trips_through_proto():
    s = Schema(
        columns=(
            Column(name="data", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False, trigram_index=True),
            Column(name="level", type=stats_pb2.COLUMN_TYPE_INT32, nullable=False),
            Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ),
    )
    back = schema_from_proto(schema_to_proto(s))
    assert {c.name: c.trigram_index for c in back.columns} == {
        "data": True,
        "level": False,
        "timestamp_ms": False,
    }
