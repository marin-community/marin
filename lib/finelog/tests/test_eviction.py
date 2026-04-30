# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for global segment eviction across namespaces: oldest-first ordering
by ``min_seq``, cap enforcement (eviction fires when over cap), and no-op
behavior when the total segment count stays under the cap.
"""

from __future__ import annotations

from pathlib import Path

from finelog.store.duckdb_store import DuckDBLogStore

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema

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
