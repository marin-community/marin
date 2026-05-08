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


def test_drop_table_rejects_concurrent_register(tmp_path: Path):
    """register_table during a drop must fail rather than recreate the namespace.

    After ``drop_table`` removes the namespace from ``_namespaces`` and
    releases ``_insertion_lock``, the bg thread is still being joined and
    the late ``catalog.delete`` / ``remove_local_storage`` have not run
    yet. A concurrent ``register_table(name)`` for the same name would
    create a fresh namespace whose catalog row and on-disk directory
    would then be wiped by those cleanup steps. The fix reserves the name
    in ``_dropping`` while the drop runs; this test pins that contract.
    """
    remote = tmp_path / "remote"
    remote.mkdir()
    store = DuckDBLogStore(log_dir=tmp_path / "data", remote_log_dir=str(remote))
    try:
        schema = _worker_schema()
        store.register_table("iris.worker", schema)
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = store._namespaces["iris.worker"]
        ns._flush_step()
        ns._force_compact_l0()
        ns._sync_step()

        # Spy on ``stop_and_join`` to attempt a same-name ``register_table``
        # at the exact point of the prior race. ``_dropping`` should make
        # it fail.
        observed: dict[str, BaseException | None] = {"register_error": None}
        original_stop = ns.stop_and_join

        def spy() -> None:
            try:
                store.register_table("iris.worker", schema)
            except BaseException as exc:
                observed["register_error"] = exc
            original_stop()

        ns.stop_and_join = spy  # type: ignore[method-assign]

        store.drop_table("iris.worker")

        err = observed["register_error"]
        assert isinstance(err, InvalidNamespaceError), f"expected InvalidNamespaceError, got {err!r}"

        # Once the drop has fully completed the reservation is cleared
        # and the name is re-registrable.
        store.register_table("iris.worker", schema)
        assert "iris.worker" in store._namespaces
        assert "iris.worker" not in store._dropping
    finally:
        store.close()


def test_drop_table_stops_bg_thread_before_dropping_catalog_rows(tmp_path: Path):
    """drop_table joins the bg thread *before* dropping catalog rows.

    With the prior order (``catalog.delete → stop_and_join``), a bg
    ``_sync_step`` tick taken between the lock release and the join
    would see an empty catalog plus a populated bucket and ``fs.rm``
    every remote file. The fix is to stop the bg thread first. This
    test pins the new order.
    """
    remote = tmp_path / "remote"
    remote.mkdir()
    store = DuckDBLogStore(log_dir=tmp_path / "data", remote_log_dir=str(remote))
    try:
        store.register_table("iris.worker", _worker_schema())
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = store._namespaces["iris.worker"]
        ns._flush_step()
        ns._force_compact_l0()
        ns._sync_step()
        assert sorted((remote / "iris.worker").glob("*.parquet"))

        # Spy on stop_and_join to assert the catalog still has rows when
        # the bg thread joins; if the order ever regresses the catalog
        # would be empty here.
        observed = {"rows_at_stop": None}
        original_stop = ns.stop_and_join

        def spy() -> None:
            with store._insertion_lock:
                observed["rows_at_stop"] = list(store._catalog.list_segments("iris.worker"))
            original_stop()

        ns.stop_and_join = spy  # type: ignore[method-assign]

        store.drop_table("iris.worker")

        assert observed["rows_at_stop"], "bg thread joined after catalog rows were dropped"
    finally:
        store.close()
