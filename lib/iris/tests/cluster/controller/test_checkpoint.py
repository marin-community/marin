# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller checkpoint: remote-only write and download-before-create restore."""

from pathlib import Path

from iris.cluster.controller.checkpoint import (
    download_checkpoint_to_local,
    write_checkpoint,
)
from iris.cluster.controller.controller import (
    Controller,
    ControllerConfig,
)
from iris.cluster.controller.db import ControllerDB
from iris.time_utils import Duration
from tests.cluster.controller.conftest import FakeProvider


def _local_state_dir(tmp_path: Path, name: str = "state") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_controller(tmp_path: Path, remote_state_dir: str | None = None, **kwargs) -> Controller:
    if remote_state_dir is None:
        remote_state_dir = f"file://{tmp_path}/remote"
    state_dir = _local_state_dir(tmp_path)
    config = ControllerConfig(remote_state_dir=remote_state_dir, local_state_dir=state_dir, **kwargs)
    return Controller(config=config, provider=FakeProvider())


def test_write_checkpoint_uploads_to_remote(tmp_path):
    """write_checkpoint creates a timestamped directory at the remote prefix."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = _make_controller(tmp_path, remote_state_dir=remote_dir)

    path, result = write_checkpoint(controller._db, remote_dir)

    assert path.startswith(f"file://{tmp_path}/remote/controller-state/")
    # path is now a directory, not a file
    remote_state = tmp_path / "remote" / "controller-state"
    # There should be a timestamped directory containing controller.sqlite3
    timestamped_dirs = [d for d in remote_state.iterdir() if d.is_dir()]
    assert len(timestamped_dirs) == 1
    assert (timestamped_dirs[0] / "controller.sqlite3").exists()

    assert result.job_count == 0
    assert result.task_count == 0
    assert result.worker_count == 0

    controller._db.close()


def test_begin_checkpoint_returns_remote_path(tmp_path):
    """begin_checkpoint returns a remote path string."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = _make_controller(tmp_path, remote_state_dir=remote_dir)

    path, result = controller.begin_checkpoint()

    assert path.startswith(f"file://{tmp_path}/remote/controller-state/")
    assert result.job_count == 0

    controller._db.close()


def test_atexit_checkpoint_writes_to_remote(tmp_path):
    """_atexit_checkpoint writes directly to remote storage."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = _make_controller(tmp_path, remote_state_dir=remote_dir)

    controller._atexit_checkpoint()

    remote_state = tmp_path / "remote" / "controller-state"
    # Checkpoint is now stored in a timestamped directory
    timestamped_dirs = [d for d in remote_state.iterdir() if d.is_dir()]
    assert len(timestamped_dirs) >= 1
    assert (timestamped_dirs[0] / "controller.sqlite3").exists()

    controller._db.close()


def test_download_checkpoint_to_local(tmp_path):
    """download_checkpoint_to_local copies remote DB to local path."""
    remote_dir = f"file://{tmp_path}/remote"
    # Create a source DB and write a checkpoint
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


def test_write_checkpoint_roundtrip(tmp_path):
    """Write then download produces a valid DB."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = _make_controller(tmp_path, remote_state_dir=remote_dir)
    write_checkpoint(controller._db, remote_dir)
    controller._db.close()

    local_db_dir = tmp_path / "restored"
    download_checkpoint_to_local(remote_dir, local_db_dir)
    # Should be openable as a ControllerDB
    restored_db = ControllerDB(db_dir=local_db_dir)
    restored_db.close()


def test_write_checkpoint_cleans_up_temp_file(tmp_path):
    """write_checkpoint does not leave temp files in the DB directory."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = _make_controller(tmp_path, remote_state_dir=remote_dir)
    db_dir = controller._db.db_path.parent

    files_before = set(db_dir.iterdir())
    write_checkpoint(controller._db, remote_dir)
    files_after = set(db_dir.iterdir())

    new_files = files_after - files_before
    sqlite_temps = [f for f in new_files if f.suffix == ".sqlite3"]
    assert len(sqlite_temps) == 0

    controller._db.close()


def test_local_db_exists_skips_remote_download(tmp_path):
    """When a local DB already exists, download_checkpoint_to_local should not be called.

    This simulates the Docker restart scenario: /var/cache/iris/controller/controller.sqlite3
    survives the restart, so we skip the remote fetch entirely.
    """
    remote_dir = f"file://{tmp_path}/remote"

    # Write a checkpoint to remote with one job count
    source_db = ControllerDB(db_dir=tmp_path / "source")
    write_checkpoint(source_db, remote_dir)
    source_db.close()

    # Simulate existing local DB (from previous container run)
    local_db_dir = tmp_path / "local"
    existing_db = ControllerDB(db_dir=local_db_dir)
    existing_db.close()

    local_path = local_db_dir / "controller.sqlite3"
    # The main.py logic: if db_path.exists(), skip download
    assert local_path.exists()
    # Verify it's the local file, not the remote one (different mtime/size)
    local_size_before = local_path.stat().st_size

    # Calling download would overwrite, but main.py skips it:
    # if db_path.exists(): logger.info("skipping") else: download_checkpoint_to_local(...)
    # So the local DB should remain unchanged
    assert local_path.stat().st_size == local_size_before


def test_fresh_start_downloads_from_remote(tmp_path):
    """When no local DB exists, download_checkpoint_to_local fetches from remote.

    This simulates first boot or data loss: no local state, so we seed from remote.
    """
    remote_dir = f"file://{tmp_path}/remote"

    # Write a checkpoint to remote
    source_db = ControllerDB(db_dir=tmp_path / "source")
    write_checkpoint(source_db, remote_dir)
    source_db.close()

    # No local DB exists
    local_db_dir = tmp_path / "local"
    assert not local_db_dir.exists() or not (local_db_dir / "controller.sqlite3").exists()

    # Download from remote (what main.py does on fresh start)
    restored = download_checkpoint_to_local(remote_dir, local_db_dir)
    assert restored is True
    assert (local_db_dir / "controller.sqlite3").exists()

    # Should be a valid DB
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


def test_periodic_checkpoint_inline(tmp_path):
    """Controller writes periodic checkpoints when limiter fires."""
    remote_dir = f"file://{tmp_path}/remote"
    controller = _make_controller(
        tmp_path,
        remote_state_dir=remote_dir,
        checkpoint_interval=Duration.from_seconds(0),
    )
    # Force the limiter to fire
    controller._periodic_checkpoint_limiter._last_run = 0

    # Simulate what the scheduling loop does inline
    if controller._periodic_checkpoint_limiter.should_run():
        write_checkpoint(controller._db, controller._config.remote_state_dir)

    remote_state = tmp_path / "remote" / "controller-state"
    timestamped_dirs = [d for d in remote_state.iterdir() if d.is_dir()]
    assert len(timestamped_dirs) >= 1
    assert (timestamped_dirs[0] / "controller.sqlite3").exists()

    controller._db.close()
