# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import InvalidNamespaceError, NamespaceNotFoundError

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema


def test_drop_table_removes_namespace(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
    _seal(store, "iris.worker")

    ns = store._namespaces["iris.worker"]
    assert any(s.level >= 1 for s in ns.all_segments_unlocked()), "expected an L>=1 segment after _seal"

    store.drop_table("iris.worker")

    assert "iris.worker" not in store._namespaces


def test_drop_table_then_query_raises(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
    _seal(store, "iris.worker")

    store.drop_table("iris.worker")
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
    assert "log" in store._namespaces


def test_drop_table_then_register_starts_fresh(store: DuckDBLogStore):
    schema = _worker_schema()
    store.register_table("iris.worker", schema)
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
    _seal(store, "iris.worker")
    store.drop_table("iris.worker")

    store.register_table("iris.worker", schema)
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-2"], [200], [2])))
    _seal(store, "iris.worker")
    table = store.query('SELECT worker_id FROM "iris.worker"')
    assert table.column("worker_id").to_pylist() == ["w-2"]


def test_drop_table_does_not_delete_remote_objects(tmp_path: Path):
    """drop_table never invokes the GCS-delete path."""
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
        # Only compacted segments are uploaded.
        ns._force_compact_l0()
        ns._sync_step()

        remote_ns_dir = remote / "iris.worker"
        assert remote_ns_dir.exists()
        remote_files_before = sorted(p.name for p in remote_ns_dir.glob("*.parquet"))
        assert remote_files_before, "expected at least one copied segment"

        store.drop_table("iris.worker")

        assert remote_ns_dir.exists()
        remote_files_after = sorted(p.name for p in remote_ns_dir.glob("*.parquet"))
        assert remote_files_after == remote_files_before
        assert not (tmp_path / "data" / "iris.worker").exists()
    finally:
        store.close()
