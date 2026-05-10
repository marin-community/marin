# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import Column, Schema


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _ipc_to_table(payload: bytes) -> pa.Table:
    return paipc.open_stream(pa.BufferReader(payload)).read_all()


def _worker_schema() -> Schema:
    return Schema(
        columns=(
            Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ),
        key_column="",
    )


def _worker_batch(worker_ids: list[str], mem_bytes: list[int], ts: list[int]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {"worker_id": worker_ids, "mem_bytes": mem_bytes, "timestamp_ms": ts},
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("mem_bytes", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )


def _seal(store: DuckDBLogStore, namespace: str) -> None:
    """Run flush -> compact -> sync synchronously, mirroring one bg-loop tick."""
    ns = store._namespaces[namespace]
    ns._flush_step()
    ns._force_compact_l0()
    ns._sync_step()


@pytest.fixture()
def store(tmp_path):
    s = DuckDBLogStore(log_dir=tmp_path / "store")
    yield s
    s.close()
