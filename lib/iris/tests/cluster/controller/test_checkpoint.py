# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller checkpoint: remote-only write and download-before-create restore."""

import pytest
from iris.cluster.controller.checkpoint import (
    download_checkpoint_to_local,
    list_checkpoints,
    prune_old_checkpoints,
    resolve_checkpoint_dir,
    write_checkpoint,
)
from iris.cluster.controller.db import ControllerDB
from rigging.timing import Duration


def test_write_checkpoint_uploads_compressed(tmp_path, make_controller):
    """write_checkpoint creates a timestamped directory with .zst files."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = make_controller(remote_state_dir=remote_dir)

    path, result = write_checkpoint(controller._db, remote_dir)

    assert path.startswith(f"file://{tmp_path}/remote/controller-state/")
    remote_state = tmp_path / "remote" / "controller-state"
    timestamped_dirs = [d for d in remote_state.iterdir() if d.is_dir()]
    assert len(timestamped_dirs) == 1
    assert (timestamped_dirs[0] / "controller.sqlite3.zst").exists()
    # Uncompressed file should NOT exist
    assert not (timestamped_dirs[0] / "controller.sqlite3").exists()

    assert result.job_count == 0
    assert result.task_count == 0
    assert result.worker_count == 0


def test_begin_checkpoint_returns_remote_path(tmp_path, make_controller):
    """begin_checkpoint returns a remote path string."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = make_controller(remote_state_dir=remote_dir)

    path, result = controller.begin_checkpoint()

    assert path.startswith(f"file://{tmp_path}/remote/controller-state/")
    assert result.job_count == 0


def test_atexit_checkpoint_writes_to_remote(tmp_path, make_controller):
    """_atexit_checkpoint writes directly to remote storage."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = make_controller(remote_state_dir=remote_dir)

    controller._atexit_checkpoint()

    remote_state = tmp_path / "remote" / "controller-state"
    timestamped_dirs = [d for d in remote_state.iterdir() if d.is_dir()]
    assert len(timestamped_dirs) >= 1
    assert (timestamped_dirs[0] / "controller.sqlite3.zst").exists()


def test_download_checkpoint_to_local(tmp_path):
    """download_checkpoint_to_local copies remote DB to local path."""
    remote_dir = f"file://{tmp_path}/remote"
    source_db = ControllerDB(db_dir=tmp_path / "source")
    write_checkpoint(source_db, remote_dir)
    source_db.close()

    local_db_dir = tmp_path / "local"
    result = download_checkpoint_to_local(remote_dir, local_db_dir)
    assert result is True
    assert (local_db_dir / "controller.sqlite3").exists()


def test_download_checkpoint_returns_false_when_missing(tmp_path):
    """Returns False when no remote checkpoint exists."""
    local_db_dir = tmp_path / "local"
    result = download_checkpoint_to_local(f"file://{tmp_path}/nonexistent", local_db_dir)
    assert result is False
    assert not (local_db_dir / "controller.sqlite3").exists()


def test_download_from_explicit_path(tmp_path):
    """download_checkpoint_to_local can restore from an explicit checkpoint directory."""
    remote_dir = f"file://{tmp_path}/remote"
    source_db = ControllerDB(db_dir=tmp_path / "source")
    path, _ = write_checkpoint(source_db, remote_dir)
    source_db.close()

    local_db_dir = tmp_path / "local"
    result = download_checkpoint_to_local(remote_dir, local_db_dir, checkpoint_dir=path)
    assert result is True
    assert (local_db_dir / "controller.sqlite3").exists()


def test_write_checkpoint_roundtrip(tmp_path, make_controller):
    """Write then download produces a valid DB."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = make_controller(remote_state_dir=remote_dir)
    write_checkpoint(controller._db, remote_dir)

    local_db_dir = tmp_path / "restored"
    download_checkpoint_to_local(remote_dir, local_db_dir)
    restored_db = ControllerDB(db_dir=local_db_dir)
    restored_db.close()


def test_write_checkpoint_cleans_up_temp_file(tmp_path, make_controller):
    """write_checkpoint does not leave temp files in the DB directory."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = make_controller(remote_state_dir=remote_dir)
    db_dir = controller._db.db_path.parent

    files_before = set(db_dir.iterdir())
    write_checkpoint(controller._db, remote_dir)
    files_after = set(db_dir.iterdir())

    new_files = files_after - files_before
    sqlite_temps = [f for f in new_files if ".sqlite3" in f.name and f.name != ControllerDB.DB_FILENAME]
    assert len(sqlite_temps) == 0


def test_local_db_exists_skips_remote_download(tmp_path):
    """When a local DB already exists, download_checkpoint_to_local should not be called.

    This simulates the Docker restart scenario: /var/cache/iris/controller/controller.sqlite3
    survives the restart, so we skip the remote fetch entirely.
    """
    remote_dir = f"file://{tmp_path}/remote"

    source_db = ControllerDB(db_dir=tmp_path / "source")
    write_checkpoint(source_db, remote_dir)
    source_db.close()

    local_db_dir = tmp_path / "local"
    existing_db = ControllerDB(db_dir=local_db_dir)
    existing_db.close()

    local_path = local_db_dir / "controller.sqlite3"
    assert local_path.exists()
    local_size_before = local_path.stat().st_size

    assert local_path.stat().st_size == local_size_before


def test_fresh_start_downloads_from_remote(tmp_path):
    """When no local DB exists, download_checkpoint_to_local fetches from remote."""
    remote_dir = f"file://{tmp_path}/remote"

    source_db = ControllerDB(db_dir=tmp_path / "source")
    write_checkpoint(source_db, remote_dir)
    source_db.close()

    local_db_dir = tmp_path / "local"
    assert not local_db_dir.exists() or not (local_db_dir / "controller.sqlite3").exists()

    restored = download_checkpoint_to_local(remote_dir, local_db_dir)
    assert restored is True
    assert (local_db_dir / "controller.sqlite3").exists()

    db = ControllerDB(db_dir=local_db_dir)
    db.close()


def test_download_from_explicit_path_pairs_auth_db(tmp_path):
    """When restoring from an explicit checkpoint path, the auth DB is derived as a sibling."""
    remote_dir = f"file://{tmp_path}/remote"
    source_db = ControllerDB(db_dir=tmp_path / "source")
    path, _ = write_checkpoint(source_db, remote_dir)
    source_db.close()

    local_db_dir = tmp_path / "local"
    result = download_checkpoint_to_local(remote_dir, local_db_dir, checkpoint_dir=path)
    assert result is True
    assert (local_db_dir / "controller.sqlite3").exists()

    assert (local_db_dir / "auth.sqlite3").exists(), "auth DB should be downloaded into local_db_dir"


def test_periodic_checkpoint_inline(tmp_path, make_controller):
    """Controller writes periodic checkpoints when limiter fires."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = make_controller(
        remote_state_dir=remote_dir,
        checkpoint_interval=Duration.from_seconds(0),
    )
    controller._periodic_checkpoint_limiter._last_run = 0

    if controller._periodic_checkpoint_limiter.should_run():
        write_checkpoint(controller._db, controller._config.remote_state_dir)

    remote_state = tmp_path / "remote" / "controller-state"
    timestamped_dirs = [d for d in remote_state.iterdir() if d.is_dir()]
    assert len(timestamped_dirs) >= 1
    assert (timestamped_dirs[0] / "controller.sqlite3.zst").exists()


def test_download_uncompressed_fallback(tmp_path):
    """download_checkpoint_to_local falls back to uncompressed files from old checkpoints."""
    remote_dir = f"file://{tmp_path}/remote"
    prefix = tmp_path / "remote" / "controller-state" / "1000000000000"
    prefix.mkdir(parents=True)

    # Write a plain (uncompressed) sqlite3 file to simulate an old checkpoint
    source_db = ControllerDB(db_dir=tmp_path / "source")
    source_db.backup_to(prefix / "controller.sqlite3")
    source_db.close()

    local_db_dir = tmp_path / "local"
    result = download_checkpoint_to_local(remote_dir, local_db_dir)
    assert result is True
    assert (local_db_dir / "controller.sqlite3").exists()

    db = ControllerDB(db_dir=local_db_dir)
    db.close()


def test_prune_old_checkpoints(tmp_path):
    """prune_old_checkpoints removes directories older than max_age."""
    remote_dir = f"file://{tmp_path}/remote"
    prefix = tmp_path / "remote" / "controller-state"
    prefix.mkdir(parents=True)

    # Create two "old" checkpoint directories (very old epoch_ms values)
    old_dir_1 = prefix / "1000"
    old_dir_1.mkdir()
    (old_dir_1 / "controller.sqlite3.zst").write_bytes(b"fake")

    old_dir_2 = prefix / "2000"
    old_dir_2.mkdir()
    (old_dir_2 / "controller.sqlite3.zst").write_bytes(b"fake")

    pruned = prune_old_checkpoints(remote_dir, max_age=Duration.from_seconds(1))
    assert pruned == 2
    assert not old_dir_1.exists()
    assert not old_dir_2.exists()


def test_prune_keeps_recent_checkpoints(tmp_path):
    """prune_old_checkpoints keeps checkpoints within max_age."""
    remote_dir = f"file://{tmp_path}/remote"
    source_db = ControllerDB(db_dir=tmp_path / "source")
    write_checkpoint(source_db, remote_dir)
    source_db.close()

    pruned = prune_old_checkpoints(remote_dir, max_age=Duration.from_hours(24))
    assert pruned == 0


def _make_checkpoint(prefix, epoch_ms: int, db_bytes: bytes) -> None:
    """Create a fake checkpoint dir with a compressed main DB of the given bytes."""
    ckpt = prefix / str(epoch_ms)
    ckpt.mkdir(parents=True)
    (ckpt / "controller.sqlite3.zst").write_bytes(db_bytes)


def test_list_checkpoints_newest_first_with_sizes(tmp_path):
    """list_checkpoints returns numeric checkpoint dirs newest-first with DB sizes."""
    remote_dir = f"file://{tmp_path}/remote"
    prefix = tmp_path / "remote" / "controller-state"
    _make_checkpoint(prefix, 1000, b"a")
    _make_checkpoint(prefix, 3000, b"ccc")
    _make_checkpoint(prefix, 2000, b"bb")
    # A non-numeric sibling must be ignored.
    (prefix / "scratch").mkdir()

    infos = list_checkpoints(remote_dir)

    assert [i.epoch_ms for i in infos] == [3000, 2000, 1000]
    assert [i.db_size_bytes for i in infos] == [3, 2, 1]
    assert infos[0].checkpoint_dir.endswith("/controller-state/3000")
    assert infos[0].has_db


def test_list_checkpoints_empty_when_missing(tmp_path):
    """No controller-state dir yields an empty list, not an error."""
    assert list_checkpoints(f"file://{tmp_path}/remote") == []


def test_resolve_checkpoint_dir_latest_and_explicit(tmp_path):
    """resolve_checkpoint_dir maps 'latest' and an epoch_ms to concrete dirs."""
    remote_dir = f"file://{tmp_path}/remote"
    prefix = tmp_path / "remote" / "controller-state"
    _make_checkpoint(prefix, 1000, b"a")
    _make_checkpoint(prefix, 3000, b"ccc")

    assert resolve_checkpoint_dir(remote_dir, "latest").endswith("/controller-state/3000")
    assert resolve_checkpoint_dir(remote_dir, "1000").endswith("/controller-state/1000")


def test_resolve_checkpoint_dir_rejects_unknown(tmp_path):
    """resolve_checkpoint_dir raises for a missing epoch_ms or bad selector."""
    remote_dir = f"file://{tmp_path}/remote"
    prefix = tmp_path / "remote" / "controller-state"
    _make_checkpoint(prefix, 1000, b"a")

    with pytest.raises(ValueError, match="not found"):
        resolve_checkpoint_dir(remote_dir, "9999")
    with pytest.raises(ValueError, match="epoch_ms"):
        resolve_checkpoint_dir(remote_dir, "notanumber")


def test_resolve_checkpoint_dir_no_checkpoints(tmp_path):
    """resolve_checkpoint_dir raises when no checkpoints exist at all."""
    with pytest.raises(ValueError, match="No controller checkpoints"):
        resolve_checkpoint_dir(f"file://{tmp_path}/remote", "latest")
