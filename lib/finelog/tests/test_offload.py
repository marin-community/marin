# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-namespace remote sync: upload-on-promote and orphan cleanup."""

from __future__ import annotations

from pathlib import Path

from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.log_namespace import DiskLogNamespace

from tests.conftest import _ipc_bytes, _worker_batch, _worker_schema


def _remote_files(remote: Path, namespace: str) -> list[Path]:
    return sorted((remote / namespace).glob("*.parquet"))


def test_sync_uploads_compacted_segments(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()

    store = DuckDBLogStore(log_dir=tmp_path / "data", remote_log_dir=str(remote))
    try:
        store.register_table("iris.worker", _worker_schema())
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = store._namespaces["iris.worker"]
        assert isinstance(ns, DiskLogNamespace)

        ns._flush_step()
        ns._force_compact_l0()

        # Compaction itself does not upload — sync runs separately.
        assert _remote_files(remote, "iris.worker") == []

        ns._sync_step()
        assert len(_remote_files(remote, "iris.worker")) == 1
    finally:
        store.close()


def test_sync_deletes_orphaned_remote_segments(tmp_path: Path) -> None:
    """Remote objects with no matching catalog row are removed by sync.

    Models the pre-fix steady-state where compaction inputs lingered in
    the bucket after promotion to a higher tier.
    """
    remote = tmp_path / "remote"
    remote.mkdir()

    store = DuckDBLogStore(log_dir=tmp_path / "data", remote_log_dir=str(remote))
    try:
        store.register_table("iris.worker", _worker_schema())
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = store._namespaces["iris.worker"]
        assert isinstance(ns, DiskLogNamespace)
        ns._flush_step()
        ns._force_compact_l0()
        ns._sync_step()
        live_remote = _remote_files(remote, "iris.worker")
        assert len(live_remote) == 1

        live_names = {p.name for p in live_remote}
        orphan = remote / "iris.worker" / "seg_L1_9999999999999999999.parquet"
        assert orphan.name not in live_names
        orphan.write_bytes(b"orphan")

        ns._sync_step()
        remaining = _remote_files(remote, "iris.worker")
        assert remaining == live_remote
        assert not orphan.exists()
    finally:
        store.close()


def test_close_drains_pending_uploads(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()

    store = DuckDBLogStore(log_dir=tmp_path / "data", remote_log_dir=str(remote))
    try:
        store.register_table("iris.worker", _worker_schema())
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = store._namespaces["iris.worker"]
        assert isinstance(ns, DiskLogNamespace)
        ns._flush_step()
        ns._force_compact_l0()
        # Skip the manual sync — close() must run it.
    except Exception:
        store.close()
        raise

    store.close()
    assert _remote_files(remote, "iris.worker")
