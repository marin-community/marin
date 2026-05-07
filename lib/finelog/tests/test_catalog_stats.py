# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-segment catalog and the namespace stats it powers.

The segment catalog (``segments`` table in ``_finelog_registry.duckdb``) is
maintained in lockstep with each namespace's in-memory ``_local_segments``
deque. These tests pin that contract: row counts and seq ranges surfaced by
``DuckDBLogStore.list_namespaces_with_stats`` must match what's actually on
disk and in RAM after every lifecycle event (write → flush → compact →
evict → drop → restart).
"""

from __future__ import annotations

import pytest
from finelog.rpc import logging_pb2
from finelog.store.catalog import Catalog, NamespaceStats
from finelog.store.duckdb_store import DuckDBLogStore

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema


def _stats(store: DuckDBLogStore, namespace: str) -> NamespaceStats:
    for name, _schema, stats in store.list_namespaces_with_stats():
        if name == namespace:
            return stats
    raise KeyError(namespace)


def _segments(store: DuckDBLogStore, namespace: str):
    # DuckDB conns aren't thread-safe; catalog reads serialize on the
    # store's insertion lock, same contract as the production code.
    with store._insertion_lock:
        return store._catalog.list_segments(namespace)


# ---------------------------------------------------------------------------
# stats() tracks rows across the write → flush → compact lifecycle
# ---------------------------------------------------------------------------


def test_stats_reflect_ram_buffer_before_flush(store, tmp_path):
    store.register_table("iris.worker", _worker_schema())
    batch = _worker_batch(["w-0", "w-1", "w-2"], [10, 20, 30], [1, 2, 3])
    store.write_rows("iris.worker", _ipc_bytes(batch))

    stats = _stats(store, "iris.worker")
    assert stats.row_count == 3
    assert stats.byte_size > 0
    assert stats.min_seq == 1
    assert stats.max_seq == 3
    # No segment has been flushed yet.
    assert stats.segment_count == 0
    assert _segments(store, "iris.worker") == []


def test_stats_after_flush_match_segments_table(store):
    store.register_table("iris.worker", _worker_schema())
    batch = _worker_batch(["w-0", "w-1"], [1, 2], [10, 20])
    store.write_rows("iris.worker", _ipc_bytes(batch))

    store._namespaces["iris.worker"]._flush_step()

    stats = _stats(store, "iris.worker")
    assert stats.row_count == 2
    assert stats.segment_count == 1
    assert stats.min_seq == 1
    assert stats.max_seq == 2

    segs = _segments(store, "iris.worker")
    assert len(segs) == 1
    assert segs[0].level == 0
    assert segs[0].row_count == 2
    assert segs[0].min_seq == 1
    assert segs[0].max_seq == 2


def test_compaction_replaces_l0_rows_atomically(store):
    store.register_table("iris.worker", _worker_schema())
    for i in range(3):
        batch = _worker_batch([f"w-{i}"], [i], [i])
        store.write_rows("iris.worker", _ipc_bytes(batch))
        store._namespaces["iris.worker"]._flush_step()

    # Three L0 segments before compaction.
    pre = _segments(store, "iris.worker")
    assert len(pre) == 3
    assert all(s.level == 0 for s in pre)

    store._namespaces["iris.worker"]._force_compact_l0()

    post = _segments(store, "iris.worker")
    assert len(post) == 1
    assert post[0].level == 1
    assert post[0].row_count == 3
    assert post[0].min_seq == 1
    assert post[0].max_seq == 3

    stats = _stats(store, "iris.worker")
    assert stats.row_count == 3
    assert stats.segment_count == 1


def test_eviction_removes_segment_row(store):
    store.register_table("iris.worker", _worker_schema())
    batch = _worker_batch(["w-0"], [1], [1])
    store.write_rows("iris.worker", _ipc_bytes(batch))
    _seal(store, "iris.worker")

    segs = _segments(store, "iris.worker")
    assert len(segs) == 1
    path = segs[0].path

    store._namespaces["iris.worker"].evict_segment(path)

    assert _segments(store, "iris.worker") == []
    stats = _stats(store, "iris.worker")
    assert stats.row_count == 0
    assert stats.segment_count == 0


def test_drop_table_clears_namespace_segments(store):
    store.register_table("iris.worker", _worker_schema())
    batch = _worker_batch(["w-0"], [1], [1])
    store.write_rows("iris.worker", _ipc_bytes(batch))
    _seal(store, "iris.worker")
    assert _segments(store, "iris.worker") != []

    store.drop_table("iris.worker")

    # Catalog rows for the namespace go away with the namespace itself.
    assert _segments(store, "iris.worker") == []


# ---------------------------------------------------------------------------
# Reconciliation: catalog re-derives from on-disk parquet at startup
# ---------------------------------------------------------------------------


def test_segments_catalog_survives_restart(tmp_path):
    log_dir = tmp_path / "store"
    s1 = DuckDBLogStore(log_dir=log_dir)
    s1.register_table("iris.worker", _worker_schema())
    for i in range(2):
        batch = _worker_batch([f"w-{i}"], [i], [i])
        s1.write_rows("iris.worker", _ipc_bytes(batch))
    _seal(s1, "iris.worker")
    pre_segs = _segments(s1, "iris.worker")
    pre_stats = _stats(s1, "iris.worker")
    s1.close()

    s2 = DuckDBLogStore(log_dir=log_dir)
    try:
        post_segs = _segments(s2, "iris.worker")
        post_stats = _stats(s2, "iris.worker")
        # File set is identical; reconciliation rewrites the rows but the
        # post-rewrite content must match.
        assert {s.path for s in post_segs} == {s.path for s in pre_segs}
        assert sum(s.row_count for s in post_segs) == sum(s.row_count for s in pre_segs)
        assert post_stats.row_count == pre_stats.row_count
        assert post_stats.segment_count == pre_stats.segment_count
        assert post_stats.min_seq == pre_stats.min_seq
        assert post_stats.max_seq == pre_stats.max_seq
    finally:
        s2.close()


def test_reconcile_preserves_copied_at_ms(tmp_path):
    """A successful upload's stamp survives the next-boot reconcile pass."""
    log_dir = tmp_path / "store"
    s1 = DuckDBLogStore(log_dir=log_dir)
    s1.register_table("iris.worker", _worker_schema())
    s1.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-0"], [1], [1])))
    _seal(s1, "iris.worker")
    seg = _segments(s1, "iris.worker")[0]
    # Simulate a successful copy: stamp the catalog row.
    s1._namespaces["iris.worker"].mark_copied(seg.path, copied_at_ms=12345)
    assert _segments(s1, "iris.worker")[0].copied_at_ms == 12345
    s1.close()

    s2 = DuckDBLogStore(log_dir=log_dir)
    try:
        rows = _segments(s2, "iris.worker")
        assert len(rows) == 1
        assert rows[0].path == seg.path
        # Reconcile rewrites the row but keeps the copy stamp.
        assert rows[0].copied_at_ms == 12345
    finally:
        s2.close()


def test_reconcile_drops_rows_for_missing_files(tmp_path):
    """An out-of-band parquet deletion is reflected after a restart."""
    log_dir = tmp_path / "store"
    s1 = DuckDBLogStore(log_dir=log_dir)
    s1.register_table("iris.worker", _worker_schema())
    s1.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-0"], [1], [1])))
    _seal(s1, "iris.worker")
    seg_path = _segments(s1, "iris.worker")[0].path
    s1.close()

    # Delete the parquet behind the catalog's back.
    import os

    os.unlink(seg_path)

    s2 = DuckDBLogStore(log_dir=log_dir)
    try:
        # Reconciliation rewrites the segment list to match the disk truth.
        assert _segments(s2, "iris.worker") == []
        stats = _stats(s2, "iris.worker")
        assert stats.segment_count == 0
        assert stats.row_count == 0
    finally:
        s2.close()


# ---------------------------------------------------------------------------
# stats are also surfaced for the privileged "log" namespace
# ---------------------------------------------------------------------------


def test_log_namespace_stats_count_pushed_logs(store):
    entries = [
        logging_pb2.LogEntry(
            timestamp=logging_pb2.Timestamp(epoch_ms=ts),
            source="stdout",
            data=f"hello-{ts}",
        )
        for ts in (10, 20, 30)
    ]
    store.append("/system/test", entries)

    stats = _stats(store, "log")
    assert stats.row_count == 3
    assert stats.min_seq == 1
    assert stats.max_seq == 3


# ---------------------------------------------------------------------------
# key_column value bounds (catalog 0002)
# ---------------------------------------------------------------------------


def _flush_one(store: DuckDBLogStore, namespace: str) -> None:
    store._namespaces[namespace]._flush_step()


def test_key_value_bounds_populated_for_log_namespace(store):
    entries = [
        logging_pb2.LogEntry(
            timestamp=logging_pb2.Timestamp(epoch_ms=ts),
            source="stdout",
            data=f"hello-{ts}",
        )
        for ts in (10, 20, 30)
    ]
    store.append("/system/aaa", entries)
    store.append("/system/zzz", entries)
    _flush_one(store, "log")

    segs = _segments(store, "log")
    assert len(segs) == 1
    # ``key`` is the schema's key_column; bounds should bracket the
    # smallest and largest source keys we just appended.
    assert segs[0].min_key_value == "/system/aaa"
    assert segs[0].max_key_value == "/system/zzz"


def test_key_value_bounds_survive_compaction(store):
    store.append(
        "/system/aaa",
        [logging_pb2.LogEntry(timestamp=logging_pb2.Timestamp(epoch_ms=1), source="s", data="x")],
    )
    _flush_one(store, "log")
    store.append(
        "/system/zzz",
        [logging_pb2.LogEntry(timestamp=logging_pb2.Timestamp(epoch_ms=2), source="s", data="x")],
    )
    _flush_one(store, "log")

    pre = _segments(store, "log")
    assert {s.min_key_value for s in pre} == {"/system/aaa", "/system/zzz"}

    store._namespaces["log"]._force_compact_l0()

    post = _segments(store, "log")
    assert len(post) == 1
    # min-of-mins / max-of-maxes across the two tmp inputs.
    assert post[0].min_key_value == "/system/aaa"
    assert post[0].max_key_value == "/system/zzz"


def test_key_value_bounds_null_when_schema_has_no_key_column(store):
    store.register_table("iris.worker", _worker_schema())
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-0"], [1], [1])))
    _seal(store, "iris.worker")

    segs = _segments(store, "iris.worker")
    assert len(segs) == 1
    # _worker_schema() declares key_column=""; bounds remain unset.
    assert segs[0].min_key_value is None
    assert segs[0].max_key_value is None


def test_key_value_bounds_repopulated_on_reconcile(tmp_path):
    log_dir = tmp_path / "store"
    s1 = DuckDBLogStore(log_dir=log_dir)
    s1.append(
        "/system/mid",
        [logging_pb2.LogEntry(timestamp=logging_pb2.Timestamp(epoch_ms=1), source="s", data="x")],
    )
    _seal(s1, "log")
    s1.close()

    s2 = DuckDBLogStore(log_dir=log_dir)
    try:
        # Reconciliation re-derived the bounds from the parquet footer.
        segs = _segments(s2, "log")
        assert len(segs) == 1
        assert segs[0].min_key_value == "/system/mid"
        assert segs[0].max_key_value == "/system/mid"
    finally:
        s2.close()


# ---------------------------------------------------------------------------
# Catalog unit checks
# ---------------------------------------------------------------------------


def test_registry_replace_segments_is_atomic(tmp_path):
    """A failing upsert mid-replace must roll the whole swap back."""
    db = Catalog(tmp_path)
    db.upsert("ns", _worker_schema())
    from finelog.store.catalog import SegmentRow

    initial = SegmentRow(
        namespace="ns",
        path="/old.parquet",
        level=0,
        min_seq=1,
        max_seq=10,
        row_count=10,
        byte_size=100,
        created_at_ms=0,
    )
    db.upsert_segment(initial)

    # Inject a row whose primary key collides with the row we keep — this
    # would survive ON CONFLICT logic, but feed a row with a constraint
    # violation: NULL in NOT NULL column. Easiest is to monkey-patch.
    bad = SegmentRow(
        namespace="ns",
        path="/new.parquet",
        level=1,
        min_seq=1,
        max_seq=10,
        row_count=10,
        byte_size=200,
        created_at_ms=0,
    )

    real_upsert = db.upsert_segment

    def boom(seg):
        if seg.path == "/new.parquet":
            raise RuntimeError("simulated insert failure")
        real_upsert(seg)

    db.upsert_segment = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        db.replace_segments("ns", removed_paths=["/old.parquet"], added=[bad])

    # Rollback restored the original row.
    rows = db.list_segments("ns")
    assert [r.path for r in rows] == ["/old.parquet"]
    db.close()
