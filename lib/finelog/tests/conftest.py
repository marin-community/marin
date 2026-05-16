# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import httpx
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
    """Run flush -> force-merge-L0 -> sync+evict synchronously.

    ``force_compact_l0`` is used (rather than ``compact``) because the
    default planner ``level_targets`` are huge so tiny test segments
    would never promote on their own. The trailing ``compact()`` runs
    sync + eviction so remote uploads land before any test assertion.
    """
    ns = store.catalog[namespace]
    ns.flush()
    ns.force_compact_l0()
    ns.compact()


def _post_and_request_persistance(
    store: DuckDBLogStore,
    namespace: str,
    post: Callable[[], httpx.Response],
) -> httpx.Response:
    """Run a persistence-blocking POST while requesting test-side flush."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(post)
        store.request_persistance(namespace, timeout=5.0)
        return future.result(timeout=5.0)


@pytest.fixture()
def store(tmp_path):
    s = DuckDBLogStore(log_dir=tmp_path / "store")
    yield s
    s.close()
