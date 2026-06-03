# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-3 Query RPC parity tests.

Drive ``StatsService.Query`` over real HTTP/RPC against both the Python and Rust
servers via the client fixtures, asserting on the decoded Arrow table. These
never import store internals — the seam is the RPC socket.

Re-expresses the behavioral cases from ``tests/test_query.py`` as RPC parity:
write -> query visibility (no manual seal, because WriteRows acks only after the
rows are on a sealed L0 segment), typed-empty results, WHERE/ORDER BY/JOIN, and
the unknown-namespace error. Phase 3.7 extends this file with query-after-drop
and 'log'-namespace consistency.
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync
from finelog.rpc.logging_connect import LogServiceClientSync

from tests.parity.conftest import Backend, _worker_arrow_schema, worker_schema

pytestmark = pytest.mark.timeout(60)


# ---------------------------------------------------------------------------
# Wire helpers.
# ---------------------------------------------------------------------------


def _stats_client(url: str) -> StatsServiceClientSync:
    return StatsServiceClientSync(address=url)


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _decode(resp: stats_pb2.QueryResponse) -> pa.Table:
    return paipc.open_stream(io.BytesIO(resp.arrow_ipc)).read_all()


def _worker_batch(worker_ids: list[str], mem_bytes: list[int], ts: list[int]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {"worker_id": worker_ids, "mem_bytes": mem_bytes, "timestamp_ms": ts},
        schema=_worker_arrow_schema(),
    )


def _register(client: StatsServiceClientSync, namespace: str, schema: stats_pb2.Schema) -> None:
    client.register_table(stats_pb2.RegisterTableRequest(namespace=namespace, schema=schema))


def _write(client: StatsServiceClientSync, namespace: str, batch: pa.RecordBatch) -> None:
    client.write_rows(stats_pb2.WriteRowsRequest(namespace=namespace, arrow_ipc=_ipc_bytes(batch)))


def _query(client: StatsServiceClientSync, sql: str) -> pa.Table:
    return _decode(client.query(stats_pb2.QueryRequest(sql=sql)))


# ---------------------------------------------------------------------------
# Write -> query visibility (no manual seal needed; WriteRows seals before ack).
# ---------------------------------------------------------------------------


def test_query_round_trip_via_write(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    _write(client, "iris.worker", _worker_batch(["w-1", "w-2"], [100, 200], [1, 2]))

    table = _query(client, 'SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY worker_id')
    assert table.column_names == ["worker_id", "mem_bytes"]
    assert table.column("worker_id").to_pylist() == ["w-1", "w-2"]
    assert table.column("mem_bytes").to_pylist() == [100, 200]


def test_query_empty_namespace_typed_empty(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    # No writes: a registered-but-empty namespace is a typed empty result, not
    # an error. The column set includes the implicit `seq`.
    table = _query(client, 'SELECT * FROM "iris.worker"')
    assert table.num_rows == 0
    assert set(table.column_names) == {"seq", "worker_id", "mem_bytes", "timestamp_ms"}


def test_query_where_filter(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    _write(
        client,
        "iris.worker",
        _worker_batch(["w-1", "w-2", "w-3"], [100, 200, 300], [1, 2, 3]),
    )
    table = _query(
        client,
        'SELECT worker_id FROM "iris.worker" WHERE mem_bytes >= 200 ORDER BY worker_id',
    )
    assert table.column("worker_id").to_pylist() == ["w-2", "w-3"]


def test_query_multi_namespace_join(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    _write(client, "iris.worker", _worker_batch(["w-1", "w-2"], [100, 200], [1, 2]))

    task_schema = stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="task_count", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ],
        key_column="",
    )
    _register(client, "iris.task", task_schema)
    task_batch = pa.RecordBatch.from_pydict(
        {"worker_id": ["w-1", "w-2"], "task_count": [10, 20], "timestamp_ms": [1, 2]},
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("task_count", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )
    _write(client, "iris.task", task_batch)

    table = _query(
        client,
        'SELECT w.mem_bytes, t.task_count FROM "iris.worker" w '
        'JOIN "iris.task" t USING (worker_id) ORDER BY w.mem_bytes',
    )
    assert table.num_rows == 2
    assert table.column("mem_bytes").to_pylist() == [100, 200]
    assert table.column("task_count").to_pylist() == [10, 20]


def test_query_unknown_namespace_raises(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    with pytest.raises(ConnectError) as exc:
        client.query(stats_pb2.QueryRequest(sql='SELECT * FROM "nope.unknown"'))
    assert exc.value.code == Code.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# Phase 3.7: query-after-drop + the reserved 'log' namespace.
# ---------------------------------------------------------------------------


def test_query_after_drop_raises(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    _write(client, "iris.worker", _worker_batch(["w-1"], [100], [1]))
    # Visible before the drop.
    assert _query(client, 'SELECT * FROM "iris.worker"').num_rows == 1

    client.drop_table(stats_pb2.DropTableRequest(namespace="iris.worker"))
    # After drop the table no longer resolves: the unknown-table error (the
    # DuckDB CatalogException analog).
    with pytest.raises(ConnectError) as exc:
        client.query(stats_pb2.QueryRequest(sql='SELECT * FROM "iris.worker"'))
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_log_namespace_is_queryable(finelog_url: str, server_backend: Backend) -> None:
    # The reserved 'log' namespace is registered, so it must resolve in a FROM
    # clause (it is registered like every other live namespace).
    log = LogServiceClientSync(address=finelog_url)
    stats = _stats_client(finelog_url)
    entries = [
        logging_pb2.LogEntry(
            source="stdout",
            data=f"line {i}",
            timestamp=logging_pb2.Timestamp(epoch_ms=1000 + i),
            level=logging_pb2.LOG_LEVEL_INFO,
        )
        for i in range(4)
    ]
    log.push_logs(logging_pb2.PushLogsRequest(key="/job/a:0", entries=entries))

    table = _query(stats, 'SELECT count(*) AS n FROM "log"')
    assert table.column("n").to_pylist() == [4]


def test_fetch_logs_and_query_agree_on_log_rowcount(finelog_url: str, server_backend: Backend) -> None:
    # The two read surfaces (Query SQL over "log" and FetchLogs) must agree on
    # the row count for the same seeded sealed data.
    log = LogServiceClientSync(address=finelog_url)
    stats = _stats_client(finelog_url)
    key = "/job/a:0"
    entries = [
        logging_pb2.LogEntry(
            source="stdout",
            data=f"line {i}",
            timestamp=logging_pb2.Timestamp(epoch_ms=1000 + i),
            level=logging_pb2.LOG_LEVEL_INFO,
        )
        for i in range(7)
    ]
    log.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries))

    fetched = log.fetch_logs(logging_pb2.FetchLogsRequest(source=key, match_scope=logging_pb2.MATCH_SCOPE_EXACT))
    query_count = _query(stats, 'SELECT count(*) AS n FROM "log"').column("n").to_pylist()[0]
    assert len(fetched.entries) == 7
    assert query_count == 7
