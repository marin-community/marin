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
    """Last-write-wins: the most recent register_table policy overrides."""
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        store.register_table("ns", _worker_schema(), StoragePolicy(max_bytes=1024))
        assert store.catalog.get_policy("ns") == StoragePolicy(max_bytes=1024)

        # Re-register with a different policy.
        store.register_table("ns", _worker_schema(), StoragePolicy(max_age_seconds=10))
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
