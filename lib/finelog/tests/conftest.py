# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for finelog stats-service tests.

Pytest auto-discovers the ``store`` fixture. Non-fixture helpers
(``_ipc_bytes``, ``_worker_schema``, etc.) are imported directly from
this module by the test files that need them.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import Column, ColumnType, Schema


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    """Serialize a single RecordBatch as an Arrow IPC stream."""
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _ipc_to_table(payload: bytes) -> pa.Table:
    return paipc.open_stream(pa.BufferReader(payload)).read_all()


def _worker_schema() -> Schema:
    return Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING, nullable=False),
            Column(name="mem_bytes", type=ColumnType.INT64, nullable=False),
            Column(name="timestamp_ms", type=ColumnType.INT64, nullable=False),
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


def _make_store(tmp_path: Path, **kwargs) -> DuckDBLogStore:
    return DuckDBLogStore(log_dir=tmp_path / "data", **kwargs)


def _seal(store: DuckDBLogStore, namespace: str) -> None:
    """Force a flush + compaction so the namespace's data is queryable.

    The query path reads only sealed (``logs_*.parquet``) segments; a
    fresh ``_flush_step`` produces a ``tmp_*.parquet`` which is
    deliberately invisible to queries until compaction promotes it.
    """
    ns = store._namespaces[namespace]
    ns._flush_step()
    ns._compaction_step(compact_single=True)


@pytest.fixture()
def store(tmp_path: Path):
    s = _make_store(tmp_path)
    yield s
    s.close()
