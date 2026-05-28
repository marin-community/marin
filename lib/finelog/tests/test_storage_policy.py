# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-namespace StoragePolicy: persistence, override semantics, and eviction."""

from __future__ import annotations

import time
from pathlib import Path

from finelog.store.catalog import Catalog
from finelog.store.compactor import CompactionConfig
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.policy import StoragePolicy, policy_from_proto, policy_to_proto

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema


def test_policy_proto_round_trip():
    """Zero in proto maps to None in dataclass and vice versa."""
    src = StoragePolicy(max_segments=5, max_bytes=100, max_age_seconds=3600)
    assert policy_from_proto(policy_to_proto(src)) == src

    empty = StoragePolicy()
    assert empty.is_empty()
    assert policy_from_proto(policy_to_proto(empty)) == empty


def test_catalog_persists_policy(tmp_path: Path):
    cat = Catalog(tmp_path)
    try:
        policy = StoragePolicy(max_segments=3, max_bytes=1024, max_age_seconds=60)
        cat.upsert_policy("ns.a", policy)
        assert cat.get_policy("ns.a") == policy
        # Empty policy clears the row.
        cat.upsert_policy("ns.a", StoragePolicy())
        assert cat.get_policy("ns.a").is_empty()
    finally:
        cat.close()


def test_register_table_persists_policy_across_reopens(tmp_path: Path):
    policy = StoragePolicy(max_bytes=1024, max_age_seconds=42)
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        store.register_table("ns", _worker_schema(), policy)
        assert store.catalog.get_policy("ns") == policy
    finally:
        store.close()

    # Reopen; policy survives.
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        assert store.catalog.get_policy("ns") == policy
    finally:
        store.close()


def test_register_table_updates_policy_on_re_register(tmp_path: Path):
    """A non-empty re-register policy overrides; an empty one keeps the existing.

    The empty-policy carve-out is what protects newer-client policies from
    being wiped by older clients that don't know to send one.
    """
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        store.register_table("ns", _worker_schema(), StoragePolicy(max_bytes=1024))
        assert store.catalog.get_policy("ns") == StoragePolicy(max_bytes=1024)

        # Re-register with a non-empty policy overrides whole-record.
        store.register_table("ns", _worker_schema(), StoragePolicy(max_age_seconds=10))
        assert store.catalog.get_policy("ns") == StoragePolicy(max_age_seconds=10)

        # Re-register with an empty policy preserves the existing one
        # (old-client-safe).
        store.register_table("ns", _worker_schema(), StoragePolicy())
        assert store.catalog.get_policy("ns") == StoragePolicy(max_age_seconds=10)
    finally:
        store.close()


def test_drop_table_clears_policy(tmp_path: Path):
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        store.register_table("ns", _worker_schema(), StoragePolicy(max_bytes=1024))
        store.drop_table("ns")
        assert store.catalog.get_policy("ns").is_empty()
    finally:
        store.close()


def test_per_namespace_policy_overrides_global_segment_cap(tmp_path: Path):
    """A tight per-namespace max_segments evicts before the global cap would."""
    # Global cap is loose (10); per-namespace cap is 1.
    config = CompactionConfig(max_segments_per_namespace=10, level_targets=(1,))
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        remote_log_dir=str(tmp_path / "remote"),
        compaction_config=config,
    )
    try:
        store.register_table("ns", _worker_schema(), StoragePolicy(max_segments=1))

        store.write_rows("ns", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "ns")
        first = sorted((tmp_path / "data" / "ns").glob("seg_L1_*.parquet"))
        assert len(first) == 1
        first_path = first[0]

        store.write_rows("ns", _ipc_bytes(_worker_batch(["b"], [2], [2])))
        _seal(store, "ns")
        store.catalog["ns"].compact()

        remaining = sorted((tmp_path / "data" / "ns").glob("seg_L1_*.parquet"))
        assert len(remaining) == 1
        assert remaining[0] != first_path
    finally:
        store.close()


def test_age_eviction_drops_old_segments(tmp_path: Path):
    """``max_age_seconds`` evicts any L>=1 BOTH segment older than the cutoff."""
    config = CompactionConfig(max_segments_per_namespace=100, level_targets=(1,))
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        remote_log_dir=str(tmp_path / "remote"),
        compaction_config=config,
    )
    try:
        store.register_table("ns", _worker_schema(), StoragePolicy(max_age_seconds=1))

        store.write_rows("ns", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "ns")
        before = sorted((tmp_path / "data" / "ns").glob("seg_L1_*.parquet"))
        assert len(before) == 1
        old_path = before[0]

        # Backdate the segment's catalog created_at_ms to 1 hour ago so
        # it lands strictly outside the 1s retention window, without
        # paying the wall-clock cost.
        old_ms = int(time.time() * 1000) - 3600 * 1000
        store.catalog._conn.execute(
            "UPDATE segments SET created_at_ms = ? WHERE namespace = 'ns'",
            [old_ms],
        )

        store.catalog["ns"].compact()
        remaining = sorted((tmp_path / "data" / "ns").glob("seg_L1_*.parquet"))
        assert remaining == []
        # Aged-out segment is preserved remotely (BOTH -> REMOTE in the catalog).
        assert (tmp_path / "remote" / "ns" / old_path.name).exists()
    finally:
        store.close()


def test_age_eviction_scans_by_created_at_not_min_seq(tmp_path: Path):
    """Age trim picks the oldest-by-time segment even when it sits at higher min_seq.

    A compaction output inherits the smallest input's ``min_seq`` but
    gets a fresh ``created_at_ms``. A naive min_seq-ordered eviction
    would see that fresh segment first, decide it's young enough, and
    short-circuit — missing an older sibling at a higher min_seq.
    """
    config = CompactionConfig(max_segments_per_namespace=100, level_targets=(1,))
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        remote_log_dir=str(tmp_path / "remote"),
        compaction_config=config,
    )
    try:
        store.register_table("ns", _worker_schema(), StoragePolicy(max_age_seconds=1))

        # Two distinct segments. We'll forge the (min_seq, created_at_ms)
        # combination directly in the catalog rows: the low-min_seq one
        # is fresh, the higher-min_seq one is old.
        store.write_rows("ns", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "ns")
        store.write_rows("ns", _ipc_bytes(_worker_batch(["b"], [2], [2])))
        _seal(store, "ns")
        segs = sorted((tmp_path / "data" / "ns").glob("seg_L1_*.parquet"))
        assert len(segs) == 2

        now_ms = int(time.time() * 1000)
        rows = store.catalog.list_segments("ns")
        # Low min_seq → freshly compacted (cr=now); high min_seq → old (cr=now-1h).
        low = min(rows, key=lambda r: r.min_seq)
        high = max(rows, key=lambda r: r.min_seq)
        store.catalog._conn.execute(
            "UPDATE segments SET created_at_ms = ? WHERE namespace = 'ns' AND path = ?",
            [now_ms, low.path],
        )
        store.catalog._conn.execute(
            "UPDATE segments SET created_at_ms = ? WHERE namespace = 'ns' AND path = ?",
            [now_ms - 3600 * 1000, high.path],
        )

        store.catalog["ns"].compact()

        remaining = {r.path for r in store.catalog.list_segments("ns") if r.location.value != "REMOTE"}
        # The old (high min_seq) segment is gone; the fresh (low min_seq) one stays.
        assert high.path not in remaining
        assert low.path in remaining
    finally:
        store.close()


def test_age_eviction_keeps_recent_segments(tmp_path: Path):
    """Recent segments are not aged out even when the policy is tight."""
    config = CompactionConfig(max_segments_per_namespace=100, level_targets=(1,))
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        remote_log_dir=str(tmp_path / "remote"),
        compaction_config=config,
    )
    try:
        # 1-hour retention: a just-written segment must survive.
        store.register_table("ns", _worker_schema(), StoragePolicy(max_age_seconds=3600))

        store.write_rows("ns", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "ns")
        store.catalog["ns"].compact()

        remaining = sorted((tmp_path / "data" / "ns").glob("seg_L1_*.parquet"))
        assert len(remaining) == 1
    finally:
        store.close()
