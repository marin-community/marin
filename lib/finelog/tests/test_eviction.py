# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from finelog.store.catalog import SegmentLocation
from finelog.store.compactor import CompactionConfig
from finelog.store.duckdb_store import DuckDBLogStore

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema


def _list_segments_locked(store: DuckDBLogStore, namespace: str):
    """Catalog read serialized against the bg thread.

    DuckDB connections aren't thread-safe and the catalog docstring is
    explicit that callers hold ``_insertion_lock``. Tests that read directly
    must do the same.
    """
    with store._insertion_lock:
        return store._catalog.list_segments(namespace)


def test_eviction_drops_oldest_segment_when_cap_exceeded(tmp_path: Path):
    """Per-namespace caps: oldest L>=1 copied segment is dropped first."""
    config = CompactionConfig(max_segments_per_namespace=1, level_targets=(1,))
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        remote_log_dir=str(tmp_path / "remote"),
        compaction_config=config,
    )
    try:
        schema = _worker_schema()
        store.register_table("ns", schema)

        store.write_rows("ns", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "ns")
        first_l1 = sorted((tmp_path / "data" / "ns").glob("seg_L1_*.parquet"))
        assert len(first_l1) == 1
        first_path = first_l1[0]

        store.write_rows("ns", _ipc_bytes(_worker_batch(["b"], [2], [2])))
        _seal(store, "ns")
        # Drive the eviction tick (compaction tail invokes _eviction_step).
        store._namespaces["ns"]._eviction_step()

        remaining = sorted((tmp_path / "data" / "ns").glob("seg_L1_*.parquet"))
        assert len(remaining) == 1
        assert remaining[0] != first_path
    finally:
        store.close()


def test_eviction_skips_segments_not_yet_copied(tmp_path: Path):
    """A freshly-promoted L1 segment is not evicted until the upload completes."""
    config = CompactionConfig(max_segments_per_namespace=1, level_targets=(1,))
    # No remote_log_dir → no upload happens → location stays LOCAL.
    store = DuckDBLogStore(log_dir=tmp_path / "data", compaction_config=config)
    try:
        store.register_table("ns", _worker_schema())
        store.write_rows("ns", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "ns")
        store.write_rows("ns", _ipc_bytes(_worker_batch(["b"], [2], [2])))
        _seal(store, "ns")

        # Two L1s on disk, neither marked copied; eviction must skip both.
        store._namespaces["ns"]._eviction_step()
        all_files = sorted((tmp_path / "data" / "ns").glob("seg_L1_*.parquet"))
        assert len(all_files) == 2
    finally:
        store.close()


def test_eviction_keeps_namespaces_under_cap(tmp_path: Path):
    """Below cap: nothing is evicted."""
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        compaction_config=CompactionConfig(max_segments_per_namespace=10, level_targets=(1,)),
    )
    try:
        store.register_table("ns.a", _worker_schema())
        store.register_table("ns.b", _worker_schema())
        store.write_rows("ns.a", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "ns.a")
        store.write_rows("ns.b", _ipc_bytes(_worker_batch(["b"], [2], [2])))
        _seal(store, "ns.b")
        assert len(list((tmp_path / "data" / "ns.a").glob("seg_L1_*.parquet"))) == 1
        assert len(list((tmp_path / "data" / "ns.b").glob("seg_L1_*.parquet"))) == 1
    finally:
        store.close()


def test_fifo_eviction_across_mixed_levels(tmp_path: Path):
    """Eviction picks the oldest L>=1 segment regardless of level — never the
    middle of the seq range."""
    config = CompactionConfig(
        max_segments_per_namespace=2,
        level_targets=(1, 2),  # L0 -> L1 -> L2 promotions on tiny byte budgets
    )
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        remote_log_dir=str(tmp_path / "remote"),
        compaction_config=config,
    )
    try:
        store.register_table("ns", _worker_schema())
        ns = store._namespaces["ns"]

        # Three flush+compact cycles drive promotions: with level_targets=(1, 2)
        # each L0 hits the size threshold and promotes to L1 then L2.
        for i in range(3):
            store.write_rows("ns", _ipc_bytes(_worker_batch([f"w-{i}"], [i], [i])))
            ns._flush_step()
            while ns._compaction_step():
                pass
            ns._sync_step()

        # Eviction should now have run and brought us to <=2 local segments,
        # popping oldest first. Evicted rows stay in the catalog with
        # location=REMOTE so the bucket archive is preserved.
        ns._eviction_step()
        all_rows = _list_segments_locked(store, "ns")
        local_rows = [r for r in all_rows if r.location in {SegmentLocation.LOCAL, SegmentLocation.BOTH}]
        remote_rows = [r for r in all_rows if r.location is SegmentLocation.REMOTE]
        assert len(local_rows) <= 2
        # Whatever's left locally, the smallest min_seq is strictly greater
        # than the smallest seq we ever wrote (0) — i.e. eviction came from
        # the front.
        assert min(r.min_seq for r in local_rows) > 0
        # The evicted rows show up as REMOTE archive pointers and their
        # remote files survive.
        assert remote_rows
        for r in remote_rows:
            assert (tmp_path / "remote" / "ns" / Path(r.path).name).exists()
    finally:
        store.close()
