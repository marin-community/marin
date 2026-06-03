# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-4 compaction parity tests.

Drive the native leveled compactor over real HTTP/RPC against BOTH the Python
and Rust servers. The frozen RPC contract cannot force a flush/compact cycle nor
read per-segment level+location, so these tests use the flag-gated, non-proto
``--debug-admin`` surface (``POST /debug/maintain`` + ``GET /debug/segments``)
threaded into both backends' spawn commands by ``conftest.py``. They never
import store internals — the seam is the (RPC + debug-admin) socket.

This file covers the 4c gate (force-compact L0 -> one L1 segment, still
queryable). The eviction / remote-sync cases (4d/4e) extend it later.
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

from tests.parity.conftest import Backend, maintain, segments

pytestmark = pytest.mark.timeout(60)


def _stats_client(url: str) -> StatsServiceClientSync:
    return StatsServiceClientSync(address=url)


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


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _write_one(client: StatsServiceClientSync, namespace: str, worker_id: str, mem: int, ts: int) -> None:
    batch = pa.RecordBatch.from_pydict(
        {"worker_id": [worker_id], "mem_bytes": [mem], "timestamp_ms": [ts]},
        schema=_worker_arrow_schema(),
    )
    # WriteRows acks only after the row is on a sealed L0 segment, so each call
    # produces exactly one L0 segment — no manual flush needed.
    client.write_rows(stats_pb2.WriteRowsRequest(namespace=namespace, arrow_ipc=_ipc_bytes(batch)))


def _query(client: StatsServiceClientSync, sql: str) -> pa.Table:
    resp = client.query(stats_pb2.QueryRequest(sql=sql))
    return paipc.open_stream(io.BytesIO(resp.arrow_ipc)).read_all()


def test_debug_maintain_promotes_l0(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register = client.register_table
    _register(stats_pb2.RegisterTableRequest(namespace="iris.worker", schema=_worker_schema()))

    # Three single-row writes -> three L0 segments (each WriteRows seals one).
    _write_one(client, "iris.worker", "w-1", 100, 30)
    _write_one(client, "iris.worker", "w-2", 200, 10)
    _write_one(client, "iris.worker", "w-3", 300, 20)

    before = segments(finelog_url, "iris.worker")
    assert len(before) == 3, "three L0 segments before compaction"
    assert all(s.level == 0 for s in before)

    # Force the L0 -> L1 merge synchronously.
    maintain(finelog_url, "iris.worker", force_compact_l0=True)

    after = segments(finelog_url, "iris.worker")
    assert len(after) == 1, "L0 segments merged into one L1 segment"
    seg = after[0]
    assert seg.level == 1
    assert seg.row_count == 3
    assert seg.min_seq == 1
    assert seg.max_seq == 3
    assert seg.location == "LOCAL"

    # The rows survive the swap and remain queryable. The merge sorts by
    # (key_column=timestamp_ms, seq); ordering by seq recovers write order.
    table = _query(client, 'SELECT worker_id, mem_bytes, "timestamp_ms" FROM "iris.worker" ORDER BY seq')
    assert table.column("worker_id").to_pylist() == ["w-1", "w-2", "w-3"]
    assert table.column("mem_bytes").to_pylist() == [100, 200, 300]

    # And ListNamespaces stats reflect the compacted single segment.
    listed = client.list_namespaces(stats_pb2.ListNamespacesRequest())
    info = next(n for n in listed.namespaces if n.namespace == "iris.worker")
    assert info.row_count == 3
    assert info.segment_count == 1
