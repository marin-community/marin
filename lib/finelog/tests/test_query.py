# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

import duckdb
import pyarrow as pa
import pytest
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.log_namespace import _is_tmp_path
from finelog.store.schema import Column, Schema
from finelog.store.sql_escape import quote_ident, quote_literal

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema


def test_quote_ident_doubles_embedded_quotes():
    assert quote_ident('a"b') == '"a""b"'


def test_quote_literal_doubles_single_quotes():
    assert quote_literal("o'brien") == "'o''brien'"


def test_query_round_trip_via_sealed_segment(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    store.write_rows(
        "iris.worker",
        _ipc_bytes(_worker_batch(["w-1", "w-2"], [100, 200], [1, 2])),
    )
    _seal(store, "iris.worker")

    table = store.query('SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY worker_id')
    assert table.column_names == ["worker_id", "mem_bytes"]
    assert table.column("worker_id").to_pylist() == ["w-1", "w-2"]
    assert table.column("mem_bytes").to_pylist() == [100, 200]


def test_query_sees_unflushed_rows(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    store.write_rows(
        "iris.worker",
        _ipc_bytes(_worker_batch(["w-1", "w-2"], [100, 200], [1, 2])),
    )

    table = store.query('SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY worker_id')
    assert table.column("worker_id").to_pylist() == ["w-1", "w-2"]
    assert table.column("mem_bytes").to_pylist() == [100, 200]


def test_query_against_namespace_with_zero_sealed_segments_returns_empty(store: DuckDBLogStore):
    # Empty namespace must yield a typed empty view, not a DuckDB error.
    store.register_table("iris.worker", _worker_schema())
    table = store.query('SELECT * FROM "iris.worker"')
    assert table.num_rows == 0
    assert set(table.column_names) == {"seq", "worker_id", "mem_bytes", "timestamp_ms"}


def test_query_with_where_filter(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    store.write_rows(
        "iris.worker",
        _ipc_bytes(_worker_batch(["w-1", "w-2", "w-3"], [100, 200, 300], [1, 2, 3])),
    )
    _seal(store, "iris.worker")
    table = store.query('SELECT worker_id FROM "iris.worker" WHERE mem_bytes >= 200 ORDER BY worker_id')
    assert table.column("worker_id").to_pylist() == ["w-2", "w-3"]


def test_query_multi_namespace_join(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    task_schema = Schema(
        columns=(
            Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            Column(name="task_count", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ),
    )
    store.register_table("iris.task", task_schema)

    store.write_rows(
        "iris.worker",
        _ipc_bytes(_worker_batch(["w-1", "w-2"], [100, 200], [1, 2])),
    )
    store.write_rows(
        "iris.task",
        _ipc_bytes(
            pa.RecordBatch.from_pydict(
                {"worker_id": ["w-1", "w-2"], "task_count": [10, 20], "timestamp_ms": [1, 2]},
                schema=pa.schema(
                    [
                        pa.field("worker_id", pa.string(), nullable=False),
                        pa.field("task_count", pa.int64(), nullable=False),
                        pa.field("timestamp_ms", pa.int64(), nullable=False),
                    ]
                ),
            )
        ),
    )
    _seal(store, "iris.worker")
    _seal(store, "iris.task")

    table = store.query(
        "SELECT w.worker_id, w.mem_bytes, t.task_count "
        'FROM "iris.worker" w JOIN "iris.task" t USING (worker_id) ORDER BY w.worker_id'
    )
    assert table.num_rows == 2
    assert table.column("mem_bytes").to_pylist() == [100, 200]
    assert table.column("task_count").to_pylist() == [10, 20]


def test_query_unknown_namespace_in_sql_raises(store: DuckDBLogStore):
    with pytest.raises(duckdb.CatalogException):
        store.query('SELECT * FROM "nope.unknown"')


def test_query_blocks_concurrent_compaction_commit(store: DuckDBLogStore):
    """A reader holding the query-visibility lock blocks compaction's commit."""
    store.register_table("iris.worker", _worker_schema())
    ns = store._namespaces["iris.worker"]
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["a"], [1], [1])))
    ns._flush_step()
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["b"], [2], [2])))
    ns._flush_step()

    # Hold the read lock; poll _pending_writers to confirm the compaction
    # thread has queued without relying on sleep.
    rwlock = store._query_visibility_lock
    rwlock.read_acquire()
    try:
        compaction_done = threading.Event()

        def run_compaction():
            ns._compaction_step()
            compaction_done.set()

        t = threading.Thread(target=run_compaction, daemon=True)
        t.start()

        with rwlock._cond:
            deadline = time.monotonic() + 5.0
            while rwlock._pending_writers == 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError("compaction thread never queued for the write lock")
                rwlock._cond.wait(timeout=remaining)

        assert not compaction_done.is_set()
    finally:
        rwlock.read_release()

    t.join(timeout=5.0)
    assert compaction_done.is_set()
    ns = store._namespaces["iris.worker"]
    sealed = ns.sealed_segments()
    assert len(sealed) == 1
    all_segments = ns.all_segments_unlocked()
    assert not any(_is_tmp_path(s.path) for s in all_segments)


def test_query_completes_on_snapshot_during_compaction(store: DuckDBLogStore):
    """Read path is independent of which physical files back the namespace."""
    store.register_table("iris.worker", _worker_schema())
    ns = store._namespaces["iris.worker"]
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["a"], [1], [1])))
    ns._flush_step()
    ns._compaction_step(compact_single=True)
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["b"], [2], [2])))
    ns._flush_step()
    ns._compaction_step(compact_single=True)

    table = store.query('SELECT worker_id FROM "iris.worker" ORDER BY worker_id')
    assert table.column("worker_id").to_pylist() == ["a", "b"]

    ns._compaction_step()
    table2 = store.query('SELECT worker_id FROM "iris.worker" ORDER BY worker_id')
    assert table2.column("worker_id").to_pylist() == ["a", "b"]
