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


def test_sync_does_not_delete_remote_after_eviction(tmp_path: Path) -> None:
    """Eviction makes the bucket the durable archive; sync must not delete it."""
    from finelog.store.compactor import CompactionConfig

    remote = tmp_path / "remote"
    remote.mkdir()

    config = CompactionConfig(max_segments_per_namespace=1, level_targets=(1,))
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        remote_log_dir=str(remote),
        compaction_config=config,
    )
    try:
        store.register_table("iris.worker", _worker_schema())
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = store._namespaces["iris.worker"]
        ns._flush_step()
        while ns._compaction_step():
            pass
        ns._sync_step()
        assert len(_remote_files(remote, "iris.worker")) == 1
        evicted_basename = _remote_files(remote, "iris.worker")[0].name

        # Trigger eviction: append another row, compact, evict (the cap is 1).
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-2"], [200], [2])))
        ns._flush_step()
        while ns._compaction_step():
            pass
        ns._sync_step()
        ns._eviction_step()

        # Several sync ticks: the durable archive must still be present.
        for _ in range(3):
            ns._sync_step()
        remote_names = {p.name for p in _remote_files(remote, "iris.worker")}
        assert evicted_basename in remote_names
    finally:
        store.close()


def test_orphan_delete_runs_only_after_replacement_uploaded(tmp_path: Path) -> None:
    """Compaction promotes inputs into a higher-level replacement: the
    input row drops at commit, the replacement is inserted at ``LOCAL``,
    and the bucket cleanup of the (now-orphan) input file happens only
    after phase 1 has uploaded the replacement in the same sync tick."""
    from finelog.store.compactor import CompactionConfig

    remote = tmp_path / "remote"
    remote.mkdir()

    # level_targets=(1, 2): every flush level-bumps L0 → L1 → L2. Each
    # promotion drops the input row and inserts a new row, exercising
    # the orphan-cleanup path on the input's remote file.
    config = CompactionConfig(level_targets=(1, 2))
    store = DuckDBLogStore(
        log_dir=tmp_path / "data",
        remote_log_dir=str(remote),
        compaction_config=config,
    )
    try:
        store.register_table("iris.worker", _worker_schema())
        ns = store._namespaces["iris.worker"]

        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns._flush_step()
        while ns._compaction_step():
            pass
        ns._sync_step()
        # After L0 → L1 → L2, only the L2 file should be in remote: the
        # L1 file was uploaded then immediately orphan-deleted by sync's
        # phase 2 once the L2 replacement was durable.
        post_remote = sorted(p.name for p in _remote_files(remote, "iris.worker"))
        assert post_remote
        assert all(name.startswith("seg_L2_") for name in post_remote), post_remote
        assert not any(name.startswith("seg_L1_") for name in post_remote), post_remote
    finally:
        store.close()


def test_wiped_catalog_recovers_from_remote(tmp_path: Path) -> None:
    """If the catalog DB is wiped while remote survives, the next boot
    must adopt remote files as ``REMOTE`` rows (not delete them as orphans)."""
    from finelog.store.catalog import CATALOG_DB_FILENAME, SegmentLocation

    remote = tmp_path / "remote"
    remote.mkdir()
    log_dir = tmp_path / "data"

    s1 = DuckDBLogStore(log_dir=log_dir, remote_log_dir=str(remote))
    try:
        s1.register_table("iris.worker", _worker_schema())
        s1.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = s1._namespaces["iris.worker"]
        ns._flush_step()
        ns._force_compact_l0()
        ns._sync_step()
        bucket_files_before = sorted(p.name for p in _remote_files(remote, "iris.worker"))
        assert bucket_files_before
    finally:
        s1.close()

    # Simulate a catalog wipe (e.g. PVC re-provision): blow away local
    # state but keep the bucket.
    (log_dir / CATALOG_DB_FILENAME).unlink()
    for p in (log_dir / "iris.worker").glob("*"):
        p.unlink()

    s2 = DuckDBLogStore(log_dir=log_dir, remote_log_dir=str(remote))
    try:
        s2.register_table("iris.worker", _worker_schema())
        ns = s2._namespaces["iris.worker"]
        rows = ns._catalog.list_segments("iris.worker")
        adopted = [r for r in rows if r.location is SegmentLocation.REMOTE]
        assert {Path(r.path).name for r in adopted} == set(bucket_files_before)
        # Sync runs without nuking the bucket.
        ns._sync_step()
        bucket_files_after = sorted(p.name for p in _remote_files(remote, "iris.worker"))
        assert bucket_files_after == bucket_files_before
    finally:
        s2.close()
