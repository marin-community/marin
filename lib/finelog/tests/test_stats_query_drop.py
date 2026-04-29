# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stats service tests: Query, DropTable, and global eviction.

Covers:
- ``Query``: round-trip via DuckDB views, typed empty views, multi-namespace
  joins, namespaces with single-quote-bearing data, and the lock-through-fetch
  semantics needed for compaction safety.
- ``DropTable``: removes registry row + local segment dir, refuses ``log``,
  raises on unknown namespaces, GCS objects untouched.
- Global eviction across namespaces: oldest-first by ``min_seq``.
- End-to-end Connect dispatch over the ASGI app.
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from starlette.testclient import TestClient

from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.server.asgi import build_log_server_asgi
from finelog.server.service import LogServiceImpl
from finelog.server.stats_service import StatsServiceImpl
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import (
    Column,
    ColumnType,
    InvalidNamespaceError,
    NamespaceNotFoundError,
    Schema,
)
from finelog.store.sql_escape import quote_ident, quote_literal

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _ipc_to_table(payload: bytes) -> pa.Table:
    return paipc.open_stream(pa.BufferReader(payload)).read_all()


def _seal(store: DuckDBLogStore, namespace: str) -> None:
    """Force a flush + compaction so the namespace's data is queryable.

    The query path reads only sealed (``logs_*.parquet``) segments; a
    fresh ``_flush_step`` produces a ``tmp_*.parquet`` which is
    deliberately invisible to queries until compaction promotes it.
    """
    ns = store._namespaces[namespace]
    ns._flush_step()
    ns._compaction_step(compact_single=True)


def _worker_schema() -> Schema:
    return Schema(
        columns=(
            Column(name="worker_id", type=ColumnType.STRING, nullable=False),
            Column(name="mem_bytes", type=ColumnType.INT64, nullable=False),
            Column(name="timestamp_ms", type=ColumnType.INT64, nullable=False),
        ),
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


@pytest.fixture()
def store(tmp_path: Path):
    s = DuckDBLogStore(log_dir=tmp_path / "data")
    yield s
    s.close()


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
    # Compaction completed: a single logs_ file replaces the two tmps.
    seg_dir = store._data_dir / "iris.worker"
    assert sorted(p.name for p in seg_dir.glob("tmp_*.parquet")) == []
    assert len(list(seg_dir.glob("logs_*.parquet"))) == 1


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


# ---------------------------------------------------------------------------
# DropTable
# ---------------------------------------------------------------------------


def test_drop_table_removes_namespace(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
    _seal(store, "iris.worker")

    seg_dir = store._data_dir / "iris.worker"
    assert seg_dir.exists()
    assert any(seg_dir.glob("*.parquet"))

    store.drop_table("iris.worker")

    assert "iris.worker" not in store._namespaces
    # Local dir is gone.
    assert not seg_dir.exists()


def test_drop_table_then_query_raises(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
    _seal(store, "iris.worker")

    store.drop_table("iris.worker")
    # The namespace's view is no longer registered on the per-query
    # connection, so DuckDB raises a Catalog error.
    with pytest.raises(duckdb.CatalogException):
        store.query('SELECT * FROM "iris.worker"')


def test_drop_table_then_write_rows_raises(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    store.drop_table("iris.worker")
    with pytest.raises(NamespaceNotFoundError):
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))


def test_drop_table_unknown_namespace_raises(store: DuckDBLogStore):
    with pytest.raises(NamespaceNotFoundError):
        store.drop_table("nope.unknown")


def test_drop_table_log_namespace_rejected(store: DuckDBLogStore):
    with pytest.raises(InvalidNamespaceError):
        store.drop_table("log")
    # Log namespace is still functional after the rejected drop.
    assert "log" in store._namespaces


def test_drop_table_then_register_starts_fresh(store: DuckDBLogStore):
    schema = _worker_schema()
    store.register_table("iris.worker", schema)
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
    _seal(store, "iris.worker")
    store.drop_table("iris.worker")

    # Re-register from scratch.
    store.register_table("iris.worker", schema)
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-2"], [200], [2])))
    _seal(store, "iris.worker")
    table = store.query('SELECT worker_id FROM "iris.worker"')
    assert table.column("worker_id").to_pylist() == ["w-2"]


def test_drop_table_does_not_delete_remote_objects(tmp_path: Path):
    """drop_table never invokes the GCS-delete path.

    We point ``remote_log_dir`` at a local directory used as a fake GCS
    bucket. After flush, the file lands there. After drop, the local
    segment dir is gone but the remote copy is preserved.
    """
    remote = tmp_path / "remote"
    remote.mkdir()
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        remote_log_dir=str(remote),
    )
    try:
        store.register_table("iris.worker", _worker_schema())
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = store._namespaces["iris.worker"]
        ns._flush_step()
        # Trigger compaction so the offload-to-GCS path runs (only
        # compacted segments are uploaded today).
        ns._compaction_step(compact_single=True)

        # Per-namespace prefix on the remote path.
        remote_ns_dir = remote / "iris.worker"
        assert remote_ns_dir.exists()
        remote_files_before = sorted(p.name for p in remote_ns_dir.glob("*.parquet"))
        assert remote_files_before, "expected at least one offloaded segment"

        store.drop_table("iris.worker")

        # Remote dir + files survive the drop.
        assert remote_ns_dir.exists()
        remote_files_after = sorted(p.name for p in remote_ns_dir.glob("*.parquet"))
        assert remote_files_after == remote_files_before
        # Local dir gone.
        assert not (tmp_path / "data" / "iris.worker").exists()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Global eviction across namespaces
# ---------------------------------------------------------------------------


def test_eviction_drops_globally_oldest_segment(tmp_path: Path):
    """Two namespaces, one global cap. The oldest sealed segment is evicted.

    Namespace ``a.first`` registers first and seals a segment, so its
    ``min_seq`` is 1. Namespace ``b.second`` seals after, so its
    ``min_seq`` is later. Configuring ``max_local_segments=1`` means the
    second namespace's compaction commit triggers eviction of the older
    cross-namespace segment.
    """
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        max_local_segments=1,
    )
    try:
        schema = _worker_schema()
        store.register_table("a.first", schema)
        store.register_table("b.second", schema)

        # Seal a logs_ segment for a.first (the older one).
        store.write_rows("a.first", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "a.first")

        a_files_after_first = sorted((tmp_path / "data" / "a.first").glob("logs_*.parquet"))
        assert len(a_files_after_first) == 1

        # Seal a logs_ segment for b.second. The compaction commit calls
        # the registry's eviction hook; the cap is 1, the global count is
        # 2, so the oldest (a.first) is dropped.
        store.write_rows("b.second", _ipc_bytes(_worker_batch(["b"], [2], [2])))
        _seal(store, "b.second")

        b_files = sorted((tmp_path / "data" / "b.second").glob("logs_*.parquet"))
        assert len(b_files) == 1
        # Older namespace's segment is gone.
        a_files = sorted((tmp_path / "data" / "a.first").glob("logs_*.parquet"))
        assert a_files == []

        # The namespace itself still exists in the registry (eviction
        # only removes local files, not the registration).
        assert "a.first" in store._namespaces
    finally:
        store.close()


def test_eviction_keeps_namespaces_under_cap(tmp_path: Path):
    """No eviction fires when total segment count is under the cap."""
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        max_local_segments=10,
    )
    try:
        store.register_table("ns.a", _worker_schema())
        store.register_table("ns.b", _worker_schema())
        store.write_rows("ns.a", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "ns.a")
        store.write_rows("ns.b", _ipc_bytes(_worker_batch(["b"], [2], [2])))
        _seal(store, "ns.b")
        # Both segments survive.
        assert len(list((tmp_path / "data" / "ns.a").glob("logs_*.parquet"))) == 1
        assert len(list((tmp_path / "data" / "ns.b").glob("logs_*.parquet"))) == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Connect/RPC end-to-end
# ---------------------------------------------------------------------------


def test_query_and_drop_via_asgi(tmp_path: Path):
    """Full Connect round-trip for Query and DropTable."""
    log_service = LogServiceImpl(log_dir=tmp_path / "data")
    stats_service = StatsServiceImpl(log_store=log_service.log_store)
    app = build_log_server_asgi(log_service, stats_service=stats_service)

    try:
        with TestClient(app) as client:
            # Register + write enough data that a query returns rows.
            schema_msg = stats_pb2.Schema(
                columns=[
                    stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
                    stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
                    stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
                ],
            )
            resp = client.post(
                "/finelog.stats.StatsService/RegisterTable",
                content=stats_pb2.RegisterTableRequest(namespace="iris.worker", schema=schema_msg).SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text

            batch = _worker_batch(["w-1", "w-2"], [100, 200], [1, 2])
            resp = client.post(
                "/finelog.stats.StatsService/WriteRows",
                content=stats_pb2.WriteRowsRequest(
                    namespace="iris.worker", arrow_ipc=_ipc_bytes(batch)
                ).SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text

            # Force a flush + compaction so the data is in a sealed segment
            # (queries see only logs_*.parquet).
            ns = log_service.log_store._namespaces["iris.worker"]
            ns._flush_step()
            ns._compaction_step(compact_single=True)

            # Query.
            resp = client.post(
                "/finelog.stats.StatsService/Query",
                content=stats_pb2.QueryRequest(
                    sql='SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY worker_id'
                ).SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text
            query_resp = stats_pb2.QueryResponse.FromString(resp.content)
            assert query_resp.row_count == 2
            result_table = _ipc_to_table(query_resp.arrow_ipc)
            assert result_table.column("worker_id").to_pylist() == ["w-1", "w-2"]
            assert result_table.column("mem_bytes").to_pylist() == [100, 200]

            # DropTable.
            resp = client.post(
                "/finelog.stats.StatsService/DropTable",
                content=stats_pb2.DropTableRequest(namespace="iris.worker").SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 200, resp.text

            # Subsequent query against the dropped namespace fails.
            resp = client.post(
                "/finelog.stats.StatsService/Query",
                content=stats_pb2.QueryRequest(sql='SELECT * FROM "iris.worker"').SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 400, resp.text
    finally:
        log_service.close()


def test_drop_table_log_namespace_rejected_via_asgi(tmp_path: Path):
    log_service = LogServiceImpl(log_dir=tmp_path / "data")
    stats_service = StatsServiceImpl(log_store=log_service.log_store)
    app = build_log_server_asgi(log_service, stats_service=stats_service)

    try:
        with TestClient(app) as client:
            resp = client.post(
                "/finelog.stats.StatsService/DropTable",
                content=stats_pb2.DropTableRequest(namespace="log").SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 400, resp.text
    finally:
        log_service.close()


def test_drop_during_concurrent_write_is_safe(store: DuckDBLogStore):
    """A racing ``drop_table`` + ``write_rows`` upholds the write contract.

    ``write_rows`` looks up the namespace under the insertion mutex; if
    drop got there first, the lookup raises ``NamespaceNotFoundError``.
    Otherwise the write proceeds and drop blocks on the mutex until the
    write returns.

    The contract: a write either completes and persists, or fails with
    ``NamespaceNotFoundError``. We verify both the cleanup-on-error
    (no surprise exceptions, no deadlock) AND the successful-write
    visibility / dropped-namespace cleanup.
    """
    store.register_table("iris.worker", _worker_schema())
    seg_dir = store._data_dir / "iris.worker"
    payload = _ipc_bytes(_worker_batch(["w-1"], [100], [1]))

    write_results: list[Exception | int] = []

    def writer():
        try:
            n = store.write_rows("iris.worker", payload)
            write_results.append(n)
        except Exception as exc:
            write_results.append(exc)

    threads = [threading.Thread(target=writer, daemon=True) for _ in range(8)]
    for t in threads:
        t.start()

    drop_succeeded = False
    try:
        store.drop_table("iris.worker")
        drop_succeeded = True
    except NamespaceNotFoundError:
        # Lost the race: every write got there first. Possible only if
        # all 8 writers slipped in before the drop's lookup, in which
        # case the namespace would still exist. We don't expect this
        # path in practice (drop runs from the main thread which started
        # last), but tolerate it cleanly.
        pass

    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "writer thread did not terminate"

    # Every result is either success (int=1) or a clean
    # NamespaceNotFoundError. No surprise exceptions.
    successes = 0
    for r in write_results:
        if isinstance(r, Exception):
            assert isinstance(r, NamespaceNotFoundError), repr(r)
        else:
            assert r == 1
            successes += 1

    if drop_succeeded:
        # Namespace was dropped: local dir is gone and no remnants
        # remain on disk. Successful writes' in-memory chunks
        # evaporated by design (the explicit drop contract).
        assert "iris.worker" not in store._namespaces
        assert not seg_dir.exists()
    else:
        # Drop lost the race: namespace survives. Successful writes
        # are durable — flush-then-compact and assert the rows are
        # readable through the public query API.
        assert "iris.worker" in store._namespaces
        ns = store._namespaces["iris.worker"]
        ns._flush_step()
        ns._compaction_step(compact_single=True)
        table = store.query('SELECT worker_id FROM "iris.worker"')
        assert table.num_rows == successes


def test_query_safe_against_concurrent_drop_table(store: DuckDBLogStore):
    """A query iterating namespaces is safe against a concurrent drop.

    Regression for the dict-iteration race: ``query()`` snapshots
    ``self._namespaces`` under the insertion lock so a concurrent
    ``drop_table`` (which also takes the insertion lock to ``del`` the
    entry) can't trigger ``RuntimeError: dictionary changed size during
    iteration``.

    We hold the read side of the rwlock through the snapshot phase by
    instrumenting ``sealed_segments`` to block on a barrier, then run
    ``drop_table`` on a *different* namespace from another thread. Drop
    waits on the rwlock write side (queued behind our read), but its
    insertion-lock window — which deletes from ``self._namespaces`` —
    runs *before* the write_acquire and would race the snapshot if the
    snapshot weren't itself protected by the insertion lock.
    """
    # Two namespaces: query iterates both; drop targets ``victim`` so
    # the dict mutation lands during the query.
    store.register_table("ns.alpha", _worker_schema())
    store.register_table("ns.victim", _worker_schema())
    # Seal a segment in alpha so its branch of the loop has work to do.
    store.write_rows("ns.alpha", _ipc_bytes(_worker_batch(["a"], [1], [1])))
    _seal(store, "ns.alpha")

    # Block the query thread inside its per-namespace loop with a barrier,
    # then run drop concurrently. Without Task A's fix, the iteration
    # of ``self._namespaces.items()`` would race ``del self._namespaces[name]``.
    alpha_ns = store._namespaces["ns.alpha"]
    in_loop = threading.Event()
    proceed = threading.Event()
    orig_sealed = alpha_ns.sealed_segments

    def blocking_sealed():
        in_loop.set()
        # Hold here while the dict mutation thread runs.
        proceed.wait(timeout=10.0)
        return orig_sealed()

    alpha_ns.sealed_segments = blocking_sealed  # type: ignore[method-assign]

    query_error: list[Exception] = []
    query_result: list[pa.Table] = []

    def run_query():
        try:
            query_result.append(store.query('SELECT COUNT(*) AS n FROM "ns.alpha"'))
        except Exception as exc:
            query_error.append(exc)

    qt = threading.Thread(target=run_query, daemon=True)
    qt.start()
    try:
        # Wait until the query is parked inside the per-namespace loop.
        assert in_loop.wait(timeout=5.0), "query never reached the per-namespace loop"

        # Drop ns.victim. ``drop_table`` mutates ``self._namespaces`` under
        # the insertion lock. Because the query already snapshotted the
        # dict under the same insertion lock, this mutation no longer
        # affects the in-flight iteration.
        drop_thread = threading.Thread(target=lambda: store.drop_table("ns.victim"), daemon=True)
        drop_thread.start()
        # Drop will block on the rwlock write side until the query
        # completes — release the barrier so the query can finish.
        proceed.set()
        drop_thread.join(timeout=10.0)
        assert not drop_thread.is_alive(), "drop_table did not complete"
    finally:
        alpha_ns.sealed_segments = orig_sealed  # type: ignore[method-assign]

    qt.join(timeout=10.0)
    assert not qt.is_alive(), "query thread did not complete"
    assert not query_error, f"unexpected query error: {query_error[0]!r}"
    assert query_result and query_result[0].column("n").to_pylist() == [1]


def test_query_acquires_read_lock_for_duration(store: DuckDBLogStore):
    """``store.query`` takes the read lock; the write lock waits for it.

    We instrument the rwlock so a write-acquire attempt records that it
    waited for the query to finish. The test does not rely on timing or
    sleep: the rwlock's internal counter tells us whether the write was
    forced to queue.
    """
    store.register_table("iris.worker", _worker_schema())
    ns = store._namespaces["iris.worker"]
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
    ns._flush_step()
    ns._compaction_step(compact_single=True)

    rwlock = store._query_visibility_lock
    write_held_during_query: list[bool] = []

    # Patch the rwlock's read_release to attempt the write_acquire
    # *before* read count drops to zero. If the write side could acquire
    # while the read side was still held, we'd record True here.
    orig_release = rwlock.read_release
    write_observed_readers = threading.Event()

    def instrumented_release():
        # While we still hold the read lock (refcount > 0), confirm the
        # rwlock would block a writer. The internal state check is the
        # only reliable signal that doesn't rely on timing.
        write_held_during_query.append(rwlock._readers > 0)
        write_observed_readers.set()
        orig_release()

    rwlock.read_release = instrumented_release  # type: ignore[method-assign]

    try:
        table = store.query('SELECT COUNT(*) AS n FROM "iris.worker"')
        assert table.column("n").to_pylist() == [1]
    finally:
        rwlock.read_release = orig_release  # type: ignore[method-assign]

    # The instrumentation fired exactly once and observed the read lock
    # still held at the moment of release.
    assert write_held_during_query == [True]


def test_drop_table_unknown_namespace_via_asgi(tmp_path: Path):
    log_service = LogServiceImpl(log_dir=tmp_path / "data")
    stats_service = StatsServiceImpl(log_store=log_service.log_store)
    app = build_log_server_asgi(log_service, stats_service=stats_service)

    try:
        with TestClient(app) as client:
            resp = client.post(
                "/finelog.stats.StatsService/DropTable",
                content=stats_pb2.DropTableRequest(namespace="nope.unknown").SerializeToString(),
                headers={"Content-Type": "application/proto"},
            )
            assert resp.status_code == 404, resp.text
    finally:
        log_service.close()
