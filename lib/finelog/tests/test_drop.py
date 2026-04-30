# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DropTable RPC: removes namespace and local segment directory,
subsequent query/write raise appropriate errors, unknown-namespace raises,
log-namespace is protected, drop-then-register-fresh starts clean, and GCS
(remote) objects are not touched by a drop.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import InvalidNamespaceError, NamespaceNotFoundError

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema

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
