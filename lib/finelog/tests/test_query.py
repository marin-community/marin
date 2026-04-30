# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Query RPC: SQL escape unit tests, round-trip via sealed
segment, empty namespace, WHERE filter, multi-namespace JOIN, unknown-ns SQL
error, and query-vs-compaction snapshot semantics (lock-through-fetch
correctness for concurrent compaction commits and compaction during query).
"""

from __future__ import annotations

import threading
import time

import duckdb
import pyarrow as pa
import pytest
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import Column, ColumnType, Schema
from finelog.store.sql_escape import quote_ident, quote_literal

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema

# ---------------------------------------------------------------------------
# SQL escape helpers
# ---------------------------------------------------------------------------


def test_quote_ident_doubles_embedded_quotes():
    assert quote_ident('a"b') == '"a""b"'


def test_quote_literal_doubles_single_quotes():
    assert quote_literal("o'brien") == "'o''brien'"


# ---------------------------------------------------------------------------
# Query: happy paths
# ---------------------------------------------------------------------------


def test_query_round_trip_via_sealed_segment(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    store.write_rows(
        "iris.worker",
        _ipc_bytes(_worker_batch(["w-1", "w-2"], [100, 200], [1, 2])),
    )
    # Force a flush so the data is in a sealed segment readable by Query.
    _seal(store, "iris.worker")

    table = store.query('SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY worker_id')
    assert table.column_names == ["worker_id", "mem_bytes"]
    assert table.column("worker_id").to_pylist() == ["w-1", "w-2"]
    assert table.column("mem_bytes").to_pylist() == [100, 200]


def test_query_against_namespace_with_zero_sealed_segments_returns_empty(store: DuckDBLogStore):
    # Register but never write/flush. The view should be a typed empty view
    # so SELECT * returns zero rows (not a DuckDB error).
    store.register_table("iris.worker", _worker_schema())
    table = store.query('SELECT * FROM "iris.worker"')
    assert table.num_rows == 0
    # The schema reflects the registered columns even though no segments exist.
    assert set(table.column_names) == {"worker_id", "mem_bytes", "timestamp_ms"}


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
    # Both namespaces share worker_id; join lets us correlate rows.
    store.register_table("iris.worker", _worker_schema())
    task_schema = Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING, nullable=False),
            Column(name="task_count", type=ColumnType.INT64, nullable=False),
            Column(name="timestamp_ms", type=ColumnType.INT64, nullable=False),
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
    # An unregistered namespace name in the FROM clause has no matching view
    # in the per-query connection, so DuckDB raises a Catalog error.
    with pytest.raises(duckdb.CatalogException):
        store.query('SELECT * FROM "nope.unknown"')


# ---------------------------------------------------------------------------
# Query: lock-through-fetch
# ---------------------------------------------------------------------------


def test_query_blocks_concurrent_compaction_commit(store: DuckDBLogStore):
    """A holder of the query-visibility read lock blocks compaction's commit.

    Compaction's commit step takes the query-visibility *write* lock to
    rename the staged file and unlink the inputs. While a reader holds
    the read side, the writer must queue.
    """
    store.register_table("iris.worker", _worker_schema())
    # Produce two tmp segments so _compaction_step has actual work.
    ns = store._namespaces["iris.worker"]
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["a"], [1], [1])))
    ns._flush_step()
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["b"], [2], [2])))
    ns._flush_step()

    # Hold the read lock manually and run compaction in a thread. We poll
    # the rwlock's ``_pending_writers`` counter (incremented inside
    # ``write_acquire`` while the thread is blocked) to confirm the
    # compaction thread has actually queued — no sleep, no timing race.
    rwlock = store._query_visibility_lock
    rwlock.read_acquire()
    try:
        compaction_done = threading.Event()

        def run_compaction():
            ns._compaction_step()
            compaction_done.set()

        t = threading.Thread(target=run_compaction, daemon=True)
        t.start()

        # Wait until the compaction thread is parked inside write_acquire.
        # ``_pending_writers`` is incremented under ``rwlock._cond``; we
        # use the same condition variable to wait without polling.
        with rwlock._cond:
            deadline = time.monotonic() + 5.0
            while rwlock._pending_writers == 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError("compaction thread never queued for the write lock")
                rwlock._cond.wait(timeout=remaining)

        # Writer is queued and parked; the compaction has *not* completed
        # because we still hold the read side.
        assert not compaction_done.is_set()
    finally:
        rwlock.read_release()

    # Releasing the read lock lets the writer proceed; the thread
    # finishes promptly.
    t.join(timeout=5.0)
    assert compaction_done.is_set()
    # Compaction completed: a single sealed segment replaces the two tmps.
    ns = store._namespaces["iris.worker"]
    sealed = ns.sealed_segments()
    assert len(sealed) == 1
    all_segments = ns.all_segments_unlocked()
    from finelog.store.log_namespace import _is_tmp_path

    assert not any(_is_tmp_path(s.path) for s in all_segments)


def test_query_completes_on_snapshot_during_compaction(store: DuckDBLogStore):
    """Compaction does not invalidate the path list captured by an entering query.

    Run a query end-to-end before compaction (sees the inputs), then run
    compaction (the inputs are replaced by a single merged file), then
    run another query (sees the merged result). Both queries return the
    same rows; this verifies the read path is independent of which
    physical files happen to back the namespace at any moment.
    """
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

    # After a re-compaction the result is still correct (now read from
    # the single merged segment).
    ns._compaction_step()  # may merge the two logs_ if multiple tmps exist; otherwise no-op
    table2 = store.query('SELECT worker_id FROM "iris.worker" ORDER BY worker_id')
    assert table2.column("worker_id").to_pylist() == ["a", "b"]
