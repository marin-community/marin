# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-2 write-path RPC parity tests.

Drive ``StatsService.WriteRows`` + ``LogService.PushLogs`` over real HTTP/RPC
against both the Python and Rust servers, observing results via
``StatsService.ListNamespaces`` stats (NOT Query, which is Phase 3). These never
import store internals — the seam is the RPC socket.

Re-expresses the direct-store tests ``test_write_rows`` / ``test_durable_writes``
/ ``test_ram_buffers`` / ``test_concurrency`` as RPC-level parity, asserting on
STRUCTURED output (NamespaceInfo fields, ConnectError codes), never on log text.

The durability contract (WriteRows/PushLogs return only after the rows are on an
L0 parquet segment) is verified by RESTART: a second server process boots over
the same ``--log-dir`` and must see the just-acked rows and seq.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync
from finelog.rpc.logging_connect import LogServiceClientSync

from tests.parity.conftest import Backend, RestartableServer

# Restart cases boot the backend twice; give parity tests extra room.
pytestmark = pytest.mark.timeout(60)


# ---------------------------------------------------------------------------
# Wire helpers (the exact shapes from tests/conftest.py, re-stated locally so
# the parity suite stays decoupled from store-level fixtures).
# ---------------------------------------------------------------------------


def _stats_client(url: str) -> StatsServiceClientSync:
    return StatsServiceClientSync(address=url)


def _log_client(url: str) -> LogServiceClientSync:
    return LogServiceClientSync(address=url)


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _worker_schema() -> stats_pb2.Schema:
    return stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ],
        key_column="",
    )


def _worker_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("worker_id", pa.string(), nullable=False),
            pa.field("mem_bytes", pa.int64(), nullable=False),
            pa.field("timestamp_ms", pa.int64(), nullable=False),
        ]
    )


def _worker_batch(worker_ids: list[str], mem_bytes: list[int], ts: list[int]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {"worker_id": worker_ids, "mem_bytes": mem_bytes, "timestamp_ms": ts},
        schema=_worker_arrow_schema(),
    )


def _register_worker(client: StatsServiceClientSync, namespace: str, schema: stats_pb2.Schema | None = None) -> None:
    client.register_table(stats_pb2.RegisterTableRequest(namespace=namespace, schema=schema or _worker_schema()))


def _write(client: StatsServiceClientSync, namespace: str, batch: pa.RecordBatch) -> int:
    resp = client.write_rows(stats_pb2.WriteRowsRequest(namespace=namespace, arrow_ipc=_ipc_bytes(batch)))
    return resp.rows_written


def _ns_info(client: StatsServiceClientSync, namespace: str) -> stats_pb2.NamespaceInfo:
    listed = {n.namespace: n for n in client.list_namespaces(stats_pb2.ListNamespacesRequest()).namespaces}
    return listed[namespace]


# ---------------------------------------------------------------------------
# WriteRows round-trip + stats.
# ---------------------------------------------------------------------------


def test_write_rows_round_trip_stats(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register_worker(client, "iris.worker")
    n = _write(client, "iris.worker", _worker_batch(["w1", "w2"], [10, 20], [100, 200]))
    assert n == 2

    info = _ns_info(client, "iris.worker")
    assert info.row_count == 2
    assert info.min_seq == 1
    assert info.max_seq == 2
    assert info.byte_size > 0
    # RAM-only vs flushed both acceptable for the count, but durability-before-ack
    # means the rows are sealed by the time WriteRows returns.
    assert info.segment_count in (0, 1)


def test_write_rows_missing_nullable_filled(finelog_url: str, server_backend: Backend) -> None:
    schema = stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            stats_pb2.Column(name="note", type=stats_pb2.COLUMN_TYPE_STRING, nullable=True),
        ],
        key_column="",
    )
    client = _stats_client(finelog_url)
    client.register_table(stats_pb2.RegisterTableRequest(namespace="iris.worker", schema=schema))
    # Omit the nullable `note` column -> NULL-filled at append, not rejected.
    batch = pa.RecordBatch.from_pydict(
        {"worker_id": ["w1"], "timestamp_ms": [100]},
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )
    resp = client.write_rows(stats_pb2.WriteRowsRequest(namespace="iris.worker", arrow_ipc=_ipc_bytes(batch)))
    assert resp.rows_written == 1
    assert _ns_info(client, "iris.worker").row_count == 1


def test_write_rows_dictionary_decoded(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register_worker(client, "iris.worker")
    # Dictionary-encode worker_id; the server must accept + decode to Utf8.
    dict_ids = pa.array(["a", "b", "a"]).dictionary_encode()
    batch = pa.RecordBatch.from_arrays(
        [dict_ids, pa.array([1, 2, 3], pa.int64()), pa.array([10, 20, 30], pa.int64())],
        schema=pa.schema(
            [
                pa.field("worker_id", dict_ids.type, nullable=False),
                pa.field("mem_bytes", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )
    resp = client.write_rows(stats_pb2.WriteRowsRequest(namespace="iris.worker", arrow_ipc=_ipc_bytes(batch)))
    assert resp.rows_written == 3
    assert _ns_info(client, "iris.worker").row_count == 3


# ---------------------------------------------------------------------------
# Schema-violation codes.
# ---------------------------------------------------------------------------


def test_write_rows_unknown_namespace_not_found(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    with pytest.raises(ConnectError) as exc:
        _write(client, "never.registered", _worker_batch(["w1"], [1], [1]))
    assert exc.value.code == Code.NOT_FOUND


def _missing_non_nullable_batch() -> pa.RecordBatch:
    # omit non-nullable mem_bytes.
    return pa.RecordBatch.from_pydict(
        {"worker_id": ["w1"], "timestamp_ms": [100]},
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )


def _unknown_column_batch() -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {"worker_id": ["w1"], "mem_bytes": [1], "timestamp_ms": [1], "bogus": [9]},
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("mem_bytes", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
                pa.field("bogus", pa.int64(), nullable=True),
            ]
        ),
    )


def _type_mismatch_batch() -> pa.RecordBatch:
    # mem_bytes declared int64; send float64.
    return pa.RecordBatch.from_pydict(
        {"worker_id": ["w1"], "mem_bytes": [1.5], "timestamp_ms": [1]},
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("mem_bytes", pa.float64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )


def _nested_type_batch() -> pa.RecordBatch:
    # worker_id arrives as a list -> nested type rejected.
    return pa.RecordBatch.from_pydict(
        {"worker_id": [[1, 2]], "mem_bytes": [1], "timestamp_ms": [1]},
        schema=pa.schema(
            [
                pa.field("worker_id", pa.list_(pa.int64()), nullable=False),
                pa.field("mem_bytes", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )


@pytest.mark.parametrize(
    "make_batch",
    [
        _missing_non_nullable_batch,
        _unknown_column_batch,
        _type_mismatch_batch,
        _nested_type_batch,
    ],
)
def test_write_rows_schema_violations_invalid_argument(finelog_url: str, server_backend: Backend, make_batch) -> None:
    client = _stats_client(finelog_url)
    _register_worker(client, "iris.worker")
    with pytest.raises(ConnectError) as exc:
        client.write_rows(stats_pb2.WriteRowsRequest(namespace="iris.worker", arrow_ipc=_ipc_bytes(make_batch())))
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_write_rows_too_many_rows_rejected(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register_worker(client, "iris.worker")
    # MAX_WRITE_ROWS_ROWS == 1_000_000 (NOT 1<<20). Use 1_000_001 so the probe
    # sits just above the cap on BOTH backends — a Rust cap of 1<<20 would have
    # wrongly accepted this, diverging from Python; this row count catches that.
    n = 1_000_001
    batch = _worker_batch(["w"] * n, [0] * n, [0] * n)
    with pytest.raises(ConnectError) as exc:
        client.write_rows(stats_pb2.WriteRowsRequest(namespace="iris.worker", arrow_ipc=_ipc_bytes(batch)))
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_write_rows_oversize_body_rejected(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register_worker(client, "iris.worker")
    # Build a batch whose IPC body exceeds MAX_WRITE_ROWS_BYTES (16 MiB) but stays
    # under the row cap: ~20k rows of ~1 KiB strings.
    big = "x" * 1024
    rows = 20_000
    batch = _worker_batch([big] * rows, [0] * rows, [0] * rows)
    body = _ipc_bytes(batch)
    assert len(body) > 16 * 1024 * 1024
    with pytest.raises(ConnectError) as exc:
        client.write_rows(stats_pb2.WriteRowsRequest(namespace="iris.worker", arrow_ipc=body))
    # The handler's own 16 MiB guard rejects with INVALID_ARGUMENT (the body is
    # under the 64 MiB transport limit, so it reaches the handler).
    assert exc.value.code == Code.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# RAM-buffer stats seq-window (two sequential writes).
# ---------------------------------------------------------------------------


def test_ram_buffer_stats_seq_window(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register_worker(client, "iris.worker")
    _write(client, "iris.worker", _worker_batch(["a", "b", "c"], [1, 2, 3], [1, 2, 3]))
    _write(client, "iris.worker", _worker_batch(["d", "e"], [4, 5], [4, 5]))
    info = _ns_info(client, "iris.worker")
    assert info.row_count == 5
    assert info.min_seq == 1
    assert info.max_seq == 5
    assert info.byte_size > 0


# ---------------------------------------------------------------------------
# 8-way concurrent writes -> distinct monotonic seq 1..8.
# ---------------------------------------------------------------------------


def test_concurrent_writes_all_durable(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register_worker(client, "iris.worker")

    def one_write(i: int) -> int:
        # Each thread needs its own client (httpx sync clients are not shared).
        c = _stats_client(finelog_url)
        batch = _worker_batch([f"w{i}"], [i], [i])
        return c.write_rows(
            stats_pb2.WriteRowsRequest(namespace="iris.worker", arrow_ipc=_ipc_bytes(batch))
        ).rows_written

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one_write, i) for i in range(8)]
        results = [f.result() for f in as_completed(futures)]
    assert results == [1] * 8

    info = _ns_info(client, "iris.worker")
    assert info.row_count == 8
    # Seqs are distinct + monotonic 1..8 under the insertion lock.
    assert info.min_seq == 1
    assert info.max_seq == 8


# ---------------------------------------------------------------------------
# Durable-before-ack via restart (the keystone gate).
# ---------------------------------------------------------------------------


def test_writerows_durable_before_ack_via_restart(restartable_server: RestartableServer) -> None:
    url = restartable_server.start()
    client = _stats_client(url)
    _register_worker(client, "iris.worker")
    n = _write(client, "iris.worker", _worker_batch(["w1", "w2", "w3"], [1, 2, 3], [10, 20, 30]))
    assert n == 3
    # WriteRows acked -> the rows are durable on disk. Stop and respawn over the
    # SAME log_dir; a fresh process must recover them from the parquet footers.
    restartable_server.stop()

    url2 = restartable_server.start()
    client2 = _stats_client(url2)
    info = _ns_info(client2, "iris.worker")
    assert info.row_count == 3
    assert info.min_seq == 1
    assert info.max_seq == 3
    assert info.segment_count >= 1  # an L0 segment survived the restart


def test_seq_monotonic_across_restart(restartable_server: RestartableServer) -> None:
    url = restartable_server.start()
    client = _stats_client(url)
    _register_worker(client, "iris.worker")
    _write(client, "iris.worker", _worker_batch(["w1", "w2"], [1, 2], [10, 20]))
    restartable_server.stop()

    # After restart, new writes continue the seq counter past the recovered max.
    url2 = restartable_server.start()
    client2 = _stats_client(url2)
    _register_worker(client2, "iris.worker")  # idempotent re-register
    _write(client2, "iris.worker", _worker_batch(["w3"], [3], [30]))
    info = _ns_info(client2, "iris.worker")
    assert info.row_count == 3
    assert info.min_seq == 1
    assert info.max_seq == 3


# ---------------------------------------------------------------------------
# PushLogs durability + empty.
# ---------------------------------------------------------------------------


def _entry(source: str, data: str, epoch_ms: int) -> logging_pb2.LogEntry:
    return logging_pb2.LogEntry(
        source=source,
        data=data,
        timestamp=logging_pb2.Timestamp(epoch_ms=epoch_ms),
        level=logging_pb2.LOG_LEVEL_INFO,
    )


def test_push_logs_durable_and_listed(finelog_url: str, server_backend: Backend) -> None:
    log = _log_client(finelog_url)
    stats = _stats_client(finelog_url)
    entries = [_entry("stdout", f"line {i}", 1000 + i) for i in range(4)]
    log.push_logs(logging_pb2.PushLogsRequest(key="/job/a", entries=entries))

    info = _ns_info(stats, "log")
    assert info.row_count >= 4
    assert info.max_seq >= 4


def test_push_logs_empty_is_noop(finelog_url: str, server_backend: Backend) -> None:
    log = _log_client(finelog_url)
    stats = _stats_client(finelog_url)
    before = _ns_info(stats, "log").row_count
    log.push_logs(logging_pb2.PushLogsRequest(key="/job/a", entries=[]))
    after = _ns_info(stats, "log").row_count
    assert after == before


def test_push_logs_durable_before_ack_via_restart(restartable_server: RestartableServer) -> None:
    url = restartable_server.start()
    log = _log_client(url)
    entries = [_entry("stdout", f"line {i}", 1000 + i) for i in range(5)]
    log.push_logs(logging_pb2.PushLogsRequest(key="/job/a", entries=entries))
    restartable_server.stop()

    url2 = restartable_server.start()
    stats2 = _stats_client(url2)
    info = _ns_info(stats2, "log")
    assert info.row_count >= 5
    assert info.max_seq >= 5
    assert info.segment_count >= 1
