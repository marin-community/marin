# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller checkpoint: SQLite backup to remote storage and restore.

This module handles only checkpoint I/O: writing timestamped SQLite copies
to remote storage and restoring the DB file from a remote checkpoint.

Autoscaler/scaling-group reconciliation lives in autoscaler.py and
scaling_group.py respectively.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from collections.abc import Callable
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


def _upload_sqlite_backup(
    source_path: Path,
    prefix: str,
    label: str,
    created_at: Timestamp,
    *,
    backup_fn: Callable[[Path], None],
) -> None:
    """Backup a SQLite file via backup_fn and upload to remote storage."""
    remote_timestamped = f"{prefix}/{label}-{created_at.epoch_ms()}.sqlite3"
    remote_latest = f"{prefix}/{label}.sqlite3"

    tmp_dir = source_path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".sqlite3", dir=tmp_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        backup_fn(tmp_path)
        _fsspec_copy(str(tmp_path), remote_timestamped)
        _fsspec_copy(str(tmp_path), remote_latest)
        logger.info("%s uploaded to %s", label, remote_timestamped)
    finally:
        tmp_path.unlink(missing_ok=True)


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
    """Write a timestamped SQLite checkpoint copy directly to remote storage.

    Backs up the DB to a local temp file, uploads to remote, then deletes
    the local copy. Returns the remote path and a summary of checkpoint contents.
    Also backs up the auth DB.
    """
    created_at = Timestamp.now()
    prefix = remote_state_dir.rstrip("/") + "/controller-state"

    _upload_sqlite_backup(db.db_path, prefix, "checkpoint", created_at, backup_fn=db.backup_to)
    remote_timestamped = f"{prefix}/checkpoint-{created_at.epoch_ms()}.sqlite3"

    auth_path = db.auth_db_path
    if auth_path.exists():
        _upload_sqlite_backup(
            auth_path,
            prefix,
            "auth-checkpoint",
            created_at,
            backup_fn=lambda dest: _backup_sqlite_file(auth_path, dest),
        )

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
    return remote_timestamped, result


def download_checkpoint_to_local(
    remote_state_dir: str,
    local_db_path: Path,
    checkpoint_path: str | None = None,
) -> bool:
    """Download a remote checkpoint SQLite file to a local path.

    Used at startup to seed the local DB before ControllerDB is created.
    Also downloads the auth DB checkpoint if available.
    Returns True if a checkpoint was downloaded, False if none found.
    """
    if checkpoint_path:
        source = checkpoint_path
    else:
        source = remote_state_dir.rstrip("/") + "/controller-state/checkpoint.sqlite3"

    fs, fs_path = fsspec.core.url_to_fs(source)
    if not fs.exists(fs_path):
        logger.info("No remote checkpoint at %s, starting fresh", source)
        return False

    local_db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_db_path.with_suffix(".download.tmp")
    _fsspec_copy(source, str(tmp_path))
    tmp_path.rename(local_db_path)
    logger.info("Downloaded checkpoint from %s to %s", source, local_db_path)

    # Download auth DB checkpoint if available
    if checkpoint_path:
        auth_source = checkpoint_path.replace("checkpoint-", "auth-checkpoint-")
    else:
        auth_source = remote_state_dir.rstrip("/") + "/controller-state/auth-checkpoint.sqlite3"
    auth_fs, auth_fs_path = fsspec.core.url_to_fs(auth_source)
    if auth_fs.exists(auth_fs_path):
        auth_local_path = local_db_path.with_name("auth.sqlite3")
        auth_tmp = auth_local_path.with_suffix(".download.tmp")
        _fsspec_copy(auth_source, str(auth_tmp))
        auth_tmp.rename(auth_local_path)
        logger.info("Downloaded auth checkpoint from %s to %s", auth_source, auth_local_path)

    return True
