# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller checkpoint: SQLite backup to remote storage and restore.

This module handles only checkpoint I/O: writing timestamped SQLite copies
to remote storage and restoring the DB file from a remote checkpoint.

Checkpoint layout (remote):
    {remote_state_dir}/controller-state/{epoch_ms}/controller.sqlite3.zst
    {remote_state_dir}/controller-state/{epoch_ms}/auth.sqlite3.zst

Files are compressed with zstandard (level 3) before upload.  On download,
compressed (.zst) files are preferred; uncompressed files are accepted as
a fallback for checkpoints written before compression was added.

Restore locates the most recent timestamped directory by listing, or
uses an explicit checkpoint directory path.

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
import zstandard

from iris.cluster.controller.db import JOBS, TASKS, WORKERS, ControllerDB
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)

ZSTD_LEVEL = 3
DEFAULT_PRUNE_AGE = Duration.from_hours(3 * 24)  # 3 days


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------


def _compress_zstd(src: Path, dst: Path) -> None:
    """Compress *src* to *dst* using zstandard at ``ZSTD_LEVEL``."""
    cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)


def _decompress_zstd(src: Path, dst: Path) -> None:
    """Decompress a zstd-compressed *src* to *dst*."""
    dctx = zstandard.ZstdDecompressor()
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        dctx.copy_stream(f_in, f_out)


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
    """Hot-backup a standalone SQLite file using the backup API.

    After the backup, the destination is switched from WAL to DELETE
    journal mode so the result is a single self-contained file without
    -wal/-shm sidecars.  This prevents corruption when the file is
    later compressed and uploaded without its WAL/SHM companions.
    """
    src_conn = sqlite3.connect(str(source))
    dst_conn = sqlite3.connect(str(dest))
    try:
        src_conn.backup(dst_conn)
        dst_conn.execute("PRAGMA journal_mode = DELETE")
        dst_conn.commit()
    finally:
        dst_conn.close()
        src_conn.close()


def write_checkpoint(
    db: ControllerDB,
    remote_state_dir: str,
) -> tuple[str, CheckpointResult]:
    """Write a timestamped, zstd-compressed SQLite checkpoint to remote storage.

    Layout:
        {remote_state_dir}/controller-state/{epoch_ms}/controller.sqlite3.zst
        {remote_state_dir}/controller-state/{epoch_ms}/auth.sqlite3.zst

    Old checkpoints (> 3 days) are pruned best-effort after the write.
    Returns the remote directory path and a summary of checkpoint contents.
    """
    created_at = Timestamp.now()
    prefix = remote_state_dir.rstrip("/") + "/controller-state"
    checkpoint_dir = f"{prefix}/{created_at.epoch_ms()}"

    # Backup main DB (compressed)
    main_remote = f"{checkpoint_dir}/{ControllerDB.DB_FILENAME}.zst"
    tmp_dir = db.db_path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".sqlite3", dir=tmp_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_zst = tmp_path.with_suffix(".sqlite3.zst")
    try:
        db.backup_to(tmp_path)
        _compress_zstd(tmp_path, tmp_zst)
        _fsspec_copy(str(tmp_zst), main_remote)
        logger.info("checkpoint main DB uploaded to %s", main_remote)
    finally:
        tmp_path.unlink(missing_ok=True)
        tmp_zst.unlink(missing_ok=True)

    # Backup auth DB (compressed)
    auth_path = db.auth_db_path
    if auth_path.exists():
        auth_remote = f"{checkpoint_dir}/{ControllerDB.AUTH_DB_FILENAME}.zst"
        fd2, tmp_name2 = tempfile.mkstemp(suffix=".sqlite3", dir=tmp_dir)
        os.close(fd2)
        tmp_path2 = Path(tmp_name2)
        tmp_zst2 = tmp_path2.with_suffix(".sqlite3.zst")
        try:
            _backup_sqlite_file(auth_path, tmp_path2)
            _compress_zstd(tmp_path2, tmp_zst2)
            _fsspec_copy(str(tmp_zst2), auth_remote)
            logger.info("checkpoint auth DB uploaded to %s", auth_remote)
        finally:
            tmp_path2.unlink(missing_ok=True)
            tmp_zst2.unlink(missing_ok=True)

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

    # Best-effort pruning of old checkpoints
    try:
        pruned = prune_old_checkpoints(remote_state_dir)
        if pruned:
            logger.info("Pruned %d old checkpoint(s)", pruned)
    except Exception:
        logger.warning("Failed to prune old checkpoints", exc_info=True)

    return checkpoint_dir, result


def _reconstruct_uri(remote_state_dir: str, fs_path: str) -> str:
    """Reconstruct a full URI from a remote_state_dir (for its scheme) and an fs_path."""
    scheme = remote_state_dir.split("://", 1)[0] if "://" in remote_state_dir else "file"
    return f"{scheme}://{fs_path.rstrip('/')}"


def _list_checkpoint_entries(remote_state_dir: str) -> list[str] | None:
    """List immediate children of {remote_state_dir}/controller-state/.

    Returns None if the directory does not exist.
    """
    prefix = remote_state_dir.rstrip("/") + "/controller-state"
    fs, fs_path = fsspec.core.url_to_fs(prefix)

    if not fs.exists(fs_path):
        return None

    try:
        return fs.ls(fs_path, detail=False)
    except FileNotFoundError:
        return None


def _find_latest_checkpoint_dir(remote_state_dir: str) -> str | None:
    """Find the most recent timestamped checkpoint directory.

    Lists {remote_state_dir}/controller-state/ for subdirectories with
    numeric names (epoch_ms), returns the path to the newest one.
    """
    entries = _list_checkpoint_entries(remote_state_dir)
    if entries is None:
        return None

    # Filter to numeric directory names (epoch_ms timestamps)
    timestamp_dirs: list[tuple[int, str]] = []
    for entry in entries:
        basename = entry.rstrip("/").rsplit("/", 1)[-1]
        if basename.isdigit():
            timestamp_dirs.append((int(basename), entry))

    if not timestamp_dirs:
        return None

    # Return the most recent (highest timestamp)
    timestamp_dirs.sort(reverse=True)
    _, latest_path = timestamp_dirs[0]
    return _reconstruct_uri(remote_state_dir, latest_path)


def _pick_remote(zst_path: str, plain_path: str) -> tuple[str | None, bool]:
    """Return (remote_path, is_compressed) preferring the .zst variant."""
    fs, fs_path = fsspec.core.url_to_fs(zst_path)
    if fs.exists(fs_path):
        return zst_path, True
    fs2, fs_path2 = fsspec.core.url_to_fs(plain_path)
    if fs2.exists(fs_path2):
        return plain_path, False
    return None, False


def _download_one(remote: str, local: Path, *, compressed: bool) -> None:
    """Download a single file, decompressing if needed. Uses atomic rename."""
    if compressed:
        tmp_zst = local.with_suffix(".download.zst.tmp")
        _fsspec_copy(remote, str(tmp_zst))
        tmp_plain = local.with_suffix(".download.tmp")
        try:
            _decompress_zstd(tmp_zst, tmp_plain)
        finally:
            tmp_zst.unlink(missing_ok=True)
        tmp_plain.rename(local)
    else:
        tmp_path = local.with_suffix(".download.tmp")
        _fsspec_copy(remote, str(tmp_path))
        tmp_path.rename(local)


def prune_old_checkpoints(
    remote_state_dir: str,
    max_age: Duration = DEFAULT_PRUNE_AGE,
) -> int:
    """Delete checkpoint directories older than *max_age*.

    Returns the number of directories pruned.
    """
    entries = _list_checkpoint_entries(remote_state_dir)
    if entries is None:
        return 0

    cutoff_ms = Timestamp.now().add_ms(-max_age.to_ms()).epoch_ms()
    fs, _ = fsspec.core.url_to_fs(remote_state_dir)
    pruned = 0
    for entry in entries:
        basename = entry.rstrip("/").rsplit("/", 1)[-1]
        if not basename.isdigit():
            continue
        if int(basename) < cutoff_ms:
            try:
                fs.rm(entry, recursive=True)
                logger.info("Pruned old checkpoint: %s", entry)
                pruned += 1
            except Exception:
                logger.warning("Failed to prune checkpoint: %s", entry, exc_info=True)
    return pruned


def download_checkpoint_to_local(
    remote_state_dir: str,
    local_db_dir: Path,
    checkpoint_dir: str | None = None,
) -> bool:
    """Download a remote checkpoint directory to a local db_dir.

    Looks for controller.sqlite3(.zst) and auth.sqlite3(.zst) in the
    checkpoint directory. Compressed files are preferred; uncompressed
    files are accepted as a fallback.

    If ``checkpoint_dir`` is not provided, finds the most recent
    timestamped checkpoint under ``remote_state_dir/controller-state/``.

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

    # Prefer compressed (.zst), fall back to uncompressed for old checkpoints
    main_zst = f"{source_dir}/{ControllerDB.DB_FILENAME}.zst"
    main_plain = f"{source_dir}/{ControllerDB.DB_FILENAME}"
    main_source, compressed = _pick_remote(main_zst, main_plain)
    if main_source is None:
        logger.info("No remote checkpoint at %s, starting fresh", source_dir)
        return False

    local_db_dir.mkdir(parents=True, exist_ok=True)

    # Download main DB
    local_main = local_db_dir / ControllerDB.DB_FILENAME
    _download_one(main_source, local_main, compressed=compressed)
    logger.info("Downloaded checkpoint from %s to %s", main_source, local_main)

    # Download auth DB if available
    auth_zst = f"{source_dir}/{ControllerDB.AUTH_DB_FILENAME}.zst"
    auth_plain = f"{source_dir}/{ControllerDB.AUTH_DB_FILENAME}"
    auth_source, auth_compressed = _pick_remote(auth_zst, auth_plain)
    if auth_source is not None:
        local_auth = local_db_dir / ControllerDB.AUTH_DB_FILENAME
        _download_one(auth_source, local_auth, compressed=auth_compressed)
        logger.info("Downloaded auth checkpoint from %s to %s", auth_source, local_auth)

    return True
