# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller checkpoint: SQLite backup to remote storage and restore.

This module handles only checkpoint I/O: writing timestamped SQLite copies
to remote storage and restoring the DB file from a remote checkpoint.

Checkpoint layout (remote):
    {remote_state_dir}/controller-state/{epoch_ms}/controller.sqlite3
    {remote_state_dir}/controller-state/{epoch_ms}/auth.sqlite3

Restore locates the most recent timestamped directory by listing, or
uses an explicit checkpoint directory path.  The "latest" alias convention
has been removed.

Autoscaler/scaling-group reconciliation lives in autoscaler.py and
scaling_group.py respectively.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fsspec.core

from iris.cluster.controller.db import JOBS, TASKS, WORKERS, ControllerDB
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointResult:
    """Metadata returned after a checkpoint DB copy is written."""

    created_at: Timestamp
    job_count: int
    task_count: int
    worker_count: int


# ---------------------------------------------------------------------------
# Checkpoint write + restore
# ---------------------------------------------------------------------------


def _fsspec_copy(src: str, dst: str) -> None:
    """Copy a file using fsspec so either path can be remote (e.g. GCS)."""
    with fsspec.core.open(src, "rb") as f_src, fsspec.core.open(dst, "wb") as f_dst:
        f_dst.write(f_src.read())


def _backup_sqlite_file(source: Path, dest: Path) -> None:
    """Hot-backup a standalone SQLite file using the backup API."""
    src_conn = sqlite3.connect(str(source))
    dst_conn = sqlite3.connect(str(dest))
    try:
        src_conn.backup(dst_conn)
        dst_conn.commit()
    finally:
        dst_conn.close()
        src_conn.close()


def write_checkpoint(
    db: ControllerDB,
    remote_state_dir: str,
) -> tuple[str, CheckpointResult]:
    """Write a timestamped SQLite checkpoint to a remote directory.

    Layout:
        {remote_state_dir}/controller-state/{epoch_ms}/controller.sqlite3
        {remote_state_dir}/controller-state/{epoch_ms}/auth.sqlite3

    Returns the remote directory path and a summary of checkpoint contents.
    """
    created_at = Timestamp.now()
    prefix = remote_state_dir.rstrip("/") + "/controller-state"
    checkpoint_dir = f"{prefix}/{created_at.epoch_ms()}"

    # Backup main DB
    main_remote = f"{checkpoint_dir}/{ControllerDB.DB_FILENAME}"
    tmp_dir = db.db_path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".sqlite3", dir=tmp_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        db.backup_to(tmp_path)
        _fsspec_copy(str(tmp_path), main_remote)
        logger.info("checkpoint main DB uploaded to %s", main_remote)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Backup auth DB
    auth_path = db.auth_db_path
    if auth_path.exists():
        auth_remote = f"{checkpoint_dir}/{ControllerDB.AUTH_DB_FILENAME}"
        fd2, tmp_name2 = tempfile.mkstemp(suffix=".sqlite3", dir=tmp_dir)
        os.close(fd2)
        tmp_path2 = Path(tmp_name2)
        try:
            _backup_sqlite_file(auth_path, tmp_path2)
            _fsspec_copy(str(tmp_path2), auth_remote)
            logger.info("checkpoint auth DB uploaded to %s", auth_remote)
        finally:
            tmp_path2.unlink(missing_ok=True)

    with db.snapshot() as snapshot:
        job_count = snapshot.count(JOBS)
        task_count = snapshot.count(TASKS)
        worker_count = snapshot.count(WORKERS)
    result = CheckpointResult(
        created_at=created_at,
        job_count=job_count,
        task_count=task_count,
        worker_count=worker_count,
    )
    return checkpoint_dir, result


def _find_latest_checkpoint_dir(remote_state_dir: str) -> str | None:
    """Find the most recent timestamped checkpoint directory.

    Lists {remote_state_dir}/controller-state/ for subdirectories with
    numeric names (epoch_ms), returns the path to the newest one.
    """
    prefix = remote_state_dir.rstrip("/") + "/controller-state"
    fs, fs_path = fsspec.core.url_to_fs(prefix)

    if not fs.exists(fs_path):
        return None

    # List immediate children — each is a timestamp directory
    try:
        entries = fs.ls(fs_path, detail=False)
    except FileNotFoundError:
        return None

    # Filter to numeric directory names (epoch_ms timestamps)
    timestamp_dirs: list[tuple[int, str]] = []
    for entry in entries:
        # entry may be "bucket/path/controller-state/1234567890" or similar
        basename = entry.rstrip("/").rsplit("/", 1)[-1]
        if basename.isdigit():
            timestamp_dirs.append((int(basename), entry))

    if not timestamp_dirs:
        return None

    # Return the most recent (highest timestamp)
    timestamp_dirs.sort(reverse=True)
    _, latest_path = timestamp_dirs[0]
    # Reconstruct as a proper URI using the original scheme
    scheme = remote_state_dir.split("://", 1)[0] if "://" in remote_state_dir else "file"
    return f"{scheme}://{latest_path.rstrip('/')}"


def download_checkpoint_to_local(
    remote_state_dir: str,
    local_db_dir: Path,
    checkpoint_dir: str | None = None,
) -> bool:
    """Download a remote checkpoint directory to a local db_dir.

    Looks for controller.sqlite3 and auth.sqlite3 in the checkpoint
    directory. If ``checkpoint_dir`` is not provided, finds the most
    recent timestamped checkpoint under ``remote_state_dir/controller-state/``.

    Returns True if a checkpoint was downloaded, False if none found.
    """
    if checkpoint_dir:
        source_dir = checkpoint_dir.rstrip("/")
    else:
        found = _find_latest_checkpoint_dir(remote_state_dir)
        if found is None:
            logger.info("No remote checkpoint found under %s, starting fresh", remote_state_dir)
            return False
        source_dir = found

    # Check that the main DB exists in the source directory
    main_source = f"{source_dir}/{ControllerDB.DB_FILENAME}"
    fs, fs_path = fsspec.core.url_to_fs(main_source)
    if not fs.exists(fs_path):
        logger.info("No remote checkpoint at %s, starting fresh", main_source)
        return False

    local_db_dir.mkdir(parents=True, exist_ok=True)

    # Download main DB
    local_main = local_db_dir / ControllerDB.DB_FILENAME
    tmp_path = local_main.with_suffix(".download.tmp")
    _fsspec_copy(main_source, str(tmp_path))
    tmp_path.rename(local_main)
    logger.info("Downloaded checkpoint from %s to %s", main_source, local_main)

    # Download auth DB if available
    auth_source = f"{source_dir}/{ControllerDB.AUTH_DB_FILENAME}"
    auth_fs, auth_fs_path = fsspec.core.url_to_fs(auth_source)
    if auth_fs.exists(auth_fs_path):
        local_auth = local_db_dir / ControllerDB.AUTH_DB_FILENAME
        auth_tmp = local_auth.with_suffix(".download.tmp")
        _fsspec_copy(auth_source, str(auth_tmp))
        auth_tmp.rename(local_auth)
        logger.info("Downloaded auth checkpoint from %s to %s", auth_source, local_auth)

    return True
