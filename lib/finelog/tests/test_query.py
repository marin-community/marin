# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import duckdb
import pyarrow as pa
import pytest
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.store.duckdb_store import DuckDBLogStore
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


def test_compaction_commit_waits_for_active_readers(store: DuckDBLogStore):
    """Commit (rename + catalog swap + unlink) must drain readers first.

    DuckDB opens parquet files lazily, so a reader holds the visibility
    read lock for the entire query and may dereference any path it
    snapshotted. Unlinking those paths under an active reader surfaces
    as ``IOException: No files found``. The commit therefore acquires
    the write side, which blocks until in-flight readers release.
    """
    store.register_table("iris.worker", _worker_schema())
    ns = store._namespaces["iris.worker"]
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["a"], [1], [1])))
    ns._flush_step()
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["b"], [2], [2])))
    ns._flush_step()

    rwlock = store._query_visibility_lock
    rwlock.read_acquire()
    compaction_done = threading.Event()
    try:

        def run_compaction():
            ns._force_compact_l0()
            compaction_done.set()

        t = threading.Thread(target=run_compaction, daemon=True)
        t.start()

        # Compaction must NOT finish while the read lock is held.
        assert not compaction_done.wait(
            timeout=0.5
        ), "compaction proceeded with active reader; concurrent unlink would race a lazy DuckDB scan"
    finally:
        rwlock.read_release()

    # Releasing the read lock must let the queued writer proceed promptly.
    assert compaction_done.wait(timeout=5.0), "compaction did not resume after reader released"
    t.join(timeout=5.0)
    all_segments = ns.all_segments_unlocked()
    assert len(all_segments) == 1
    assert all(s.level >= 1 for s in all_segments)


def test_query_completes_on_snapshot_during_compaction(store: DuckDBLogStore):
    """Read path is independent of which physical files back the namespace."""
    store.register_table("iris.worker", _worker_schema())
    ns = store._namespaces["iris.worker"]
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["a"], [1], [1])))
    ns._flush_step()
    ns._force_compact_l0()
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["b"], [2], [2])))
    ns._flush_step()
    ns._force_compact_l0()

    table = store.query('SELECT worker_id FROM "iris.worker" ORDER BY worker_id')
    assert table.column("worker_id").to_pylist() == ["a", "b"]

    ns._compaction_step()
    table2 = store.query('SELECT worker_id FROM "iris.worker" ORDER BY worker_id')
    assert table2.column("worker_id").to_pylist() == ["a", "b"]


def test_writes_proceed_during_compaction_copy(store: DuckDBLogStore, monkeypatch):
    """Compaction's parquet COPY must not block concurrent appends.

    Stubs ``compaction_conn.execute`` to gate on an event so the COPY hangs
    deterministically. A concurrent ``write_rows`` must finish before we
    release the gate — proving the insertion lock is not held across the COPY.
    """
    store.register_table("iris.worker", _worker_schema())
    ns = store._namespaces["iris.worker"]

    # Two L0 segments so ``_compaction_step`` plans an actual merge.
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["a"], [1], [1])))
    ns._flush_step()
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["b"], [2], [2])))
    ns._flush_step()

    copy_entered = threading.Event()
    release_copy = threading.Event()
    real_conn = ns._compaction_conn

    class _GatedConn:
        def execute(self, sql, *args, **kwargs):
            copy_entered.set()
            if not release_copy.wait(timeout=5.0):
                raise AssertionError("test forgot to release the COPY gate")
            return real_conn.execute(sql, *args, **kwargs)

    monkeypatch.setattr(ns, "_compaction_conn", _GatedConn())

    compaction_done = threading.Event()

    def run_compaction():
        ns._force_compact_l0()
        compaction_done.set()

    compactor = threading.Thread(target=run_compaction, daemon=True)
    compactor.start()
    assert copy_entered.wait(timeout=5.0), "compaction never entered the COPY"

    write_done = threading.Event()
    write_error: list[BaseException] = []

    def run_write():
        try:
            store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["c"], [3], [3])))
        except BaseException as exc:  # surface to the test thread
            write_error.append(exc)
        finally:
            write_done.set()

    writer = threading.Thread(target=run_write, daemon=True)
    writer.start()

    # Append must complete while the COPY is still gated.
    assert write_done.wait(timeout=2.0), "append blocked behind compaction COPY"
    assert not write_error, write_error
    assert not compaction_done.is_set()

    release_copy.set()
    compactor.join(timeout=5.0)
    assert compaction_done.is_set()
    writer.join(timeout=1.0)
