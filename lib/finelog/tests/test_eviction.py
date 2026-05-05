# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from finelog.store.duckdb_store import DuckDBLogStore

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema


def test_eviction_drops_globally_oldest_segment(tmp_path: Path):
    """One global cap, two namespaces — oldest sealed segment evicted."""
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        max_local_segments=1,
    )
    try:
        schema = _worker_schema()
        store.register_table("a.first", schema)
        store.register_table("b.second", schema)

        store.write_rows("a.first", _ipc_bytes(_worker_batch(["a"], [1], [1])))
        _seal(store, "a.first")

        a_files_after_first = sorted((tmp_path / "data" / "a.first").glob("logs_*.parquet"))
        assert len(a_files_after_first) == 1

        store.write_rows("b.second", _ipc_bytes(_worker_batch(["b"], [2], [2])))
        _seal(store, "b.second")

        b_files = sorted((tmp_path / "data" / "b.second").glob("logs_*.parquet"))
        assert len(b_files) == 1
        a_files = sorted((tmp_path / "data" / "a.first").glob("logs_*.parquet"))
        assert a_files == []

        # Eviction removes local files, not the registration.
        assert "a.first" in store._namespaces
    finally:
        store.close()


def test_eviction_keeps_namespaces_under_cap(tmp_path: Path):
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
        assert len(list((tmp_path / "data" / "ns.a").glob("logs_*.parquet"))) == 1
        assert len(list((tmp_path / "data" / "ns.b").glob("logs_*.parquet"))) == 1
    finally:
        store.close()
