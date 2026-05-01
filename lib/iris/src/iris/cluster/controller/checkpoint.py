# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller checkpoint: SQLite backup to remote storage and restore.

This module handles only checkpoint I/O: writing timestamped SQLite copies
to remote storage and restoring the DB file from a remote checkpoint.

Checkpoint layout (remote):
    {remote_state_dir}/controller-state/{epoch_ms}/controller.sqlite3.zst
    {remote_state_dir}/controller-state/{epoch_ms}/auth.sqlite3.zst
    {remote_state_dir}/controller-state/{epoch_ms}/profiles.sqlite3.zst

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
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import fsspec.core
import zstandard
from rigging.timing import Duration, Timestamp

from iris.cluster.controller.db import ControllerDB

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


@dataclass(frozen=True)
class DatabaseBackup:
    """Temporary local backup files produced by ``backup_databases``."""

    main_path: Path
    auth_path: Path | None
    profiles_path: Path | None
    created_at: Timestamp

    def cleanup(self) -> None:
        """Remove temporary backup files."""
        self.main_path.unlink(missing_ok=True)
        if self.auth_path is not None:
            self.auth_path.unlink(missing_ok=True)
        if self.profiles_path is not None:
            self.profiles_path.unlink(missing_ok=True)


@contextmanager
def _reserved_tmp_sqlite(tmp_dir: Path) -> Iterator[Path]:
    """Reserve a tempfile path under *tmp_dir* and guarantee cleanup on error.

    On normal exit the file is left in place -- callers adopt ownership
    (via ``ExitStack.pop_all``).  On exception the reserved file and its
    SQLite sidecars are removed so no orphans survive.  SQLite may create
    ``<name>-journal``/``-wal``/``-shm`` sidecars during backup even
    though we never open the file in WAL mode; we unlink all three
    defensively.

    We deliberately catch ``Exception`` rather than ``BaseException``:
    after ``pop_all()`` transfers our callback to a discarded stack, GC
    of the held generator raises ``GeneratorExit`` here, and we must
    treat that as a successful hand-off -- not unlink the file.
    """
    fd, tmp_name = tempfile.mkstemp(suffix=".sqlite3", dir=tmp_dir)
    os.close(fd)
    path = Path(tmp_name)
    try:
        yield path
    except Exception:
        path.unlink(missing_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
        raise


def backup_databases(db: ControllerDB) -> DatabaseBackup:
    """Create local SQLite backup copies of the main, auth and profiles DBs.

    Should be called while holding the write lock against the main DB -- it
    uses the SQLite backup API for a consistent snapshot.  The returned
    ``DatabaseBackup`` owns the temporary files and must be cleaned up by the
    caller (via ``DatabaseBackup.cleanup``) on the success path.  If any of
    the backup operations raise, all reserved temp files are unlinked before
    the exception propagates.
    """
    created_at = Timestamp.now()
    tmp_dir = db.db_path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        main_tmp = stack.enter_context(_reserved_tmp_sqlite(tmp_dir))
        db.backup_to(main_tmp)

        auth_tmp: Path | None = None
        if db.auth_db_path.exists():
            auth_tmp = stack.enter_context(_reserved_tmp_sqlite(tmp_dir))
            _backup_sqlite_file(db.auth_db_path, auth_tmp)

        profiles_tmp: Path | None = None
        if db.profiles_db_path.exists():
            profiles_tmp = stack.enter_context(_reserved_tmp_sqlite(tmp_dir))
            _backup_sqlite_file(db.profiles_db_path, profiles_tmp)

        # Success: transfer temp-file ownership to the returned DatabaseBackup,
        # whose .cleanup() is the caller's responsibility.
        stack.pop_all()
        return DatabaseBackup(
            main_path=main_tmp,
            auth_path=auth_tmp,
            profiles_path=profiles_tmp,
            created_at=created_at,
        )


def upload_checkpoint(
    db: ControllerDB,
    backup: DatabaseBackup,
    remote_state_dir: str,
) -> tuple[str, CheckpointResult]:
    """Compress, upload, count rows, and prune old checkpoints.

    This is the slow half of checkpointing (zstd compression + GCS upload)
    and does not need any write lock on the database.
    """
    prefix = remote_state_dir.rstrip("/") + "/controller-state"
    checkpoint_dir = f"{prefix}/{backup.created_at.epoch_ms()}"

    # Compress and upload main DB.  Backup files are owned by the caller
    # (via DatabaseBackup.cleanup); we only clean up the intermediate .zst.
    main_remote = f"{checkpoint_dir}/{ControllerDB.DB_FILENAME}.zst"
    tmp_zst = backup.main_path.with_suffix(".sqlite3.zst")
    try:
        _compress_zstd(backup.main_path, tmp_zst)
        _fsspec_copy(str(tmp_zst), main_remote)
        logger.info("checkpoint main DB uploaded to %s", main_remote)
    finally:
        tmp_zst.unlink(missing_ok=True)

    # Compress and upload auth DB.
    if backup.auth_path is not None:
        auth_remote = f"{checkpoint_dir}/{ControllerDB.AUTH_DB_FILENAME}.zst"
        tmp_zst2 = backup.auth_path.with_suffix(".sqlite3.zst")
        try:
            _compress_zstd(backup.auth_path, tmp_zst2)
            _fsspec_copy(str(tmp_zst2), auth_remote)
            logger.info("checkpoint auth DB uploaded to %s", auth_remote)
        finally:
            tmp_zst2.unlink(missing_ok=True)

    # Compress and upload profiles DB.
    if backup.profiles_path is not None:
        profiles_remote = f"{checkpoint_dir}/{ControllerDB.PROFILES_DB_FILENAME}.zst"
        tmp_zst3 = backup.profiles_path.with_suffix(".sqlite3.zst")
        try:
            _compress_zstd(backup.profiles_path, tmp_zst3)
            _fsspec_copy(str(tmp_zst3), profiles_remote)
            logger.info("checkpoint profiles DB uploaded to %s", profiles_remote)
        finally:
            tmp_zst3.unlink(missing_ok=True)

    # Row counts are read from the live DB (not the backup) for convenience.
    # They may diverge slightly from the backup contents if writes occurred
    # between backup and upload, but this is acceptable for checkpoint metadata.
    with db.read_snapshot() as snapshot:
        job_count = snapshot.fetchone("SELECT COUNT(*) FROM jobs")[0]  # type: ignore[index]
        task_count = snapshot.fetchone("SELECT COUNT(*) FROM tasks")[0]  # type: ignore[index]
        worker_count = snapshot.fetchone("SELECT COUNT(*) FROM workers")[0]  # type: ignore[index]
    result = CheckpointResult(
        created_at=backup.created_at,
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


def write_checkpoint(
    db: ControllerDB,
    remote_state_dir: str,
) -> tuple[str, CheckpointResult]:
    """Write a timestamped, zstd-compressed SQLite checkpoint to remote storage.

    Convenience wrapper that calls ``backup_databases`` then ``upload_checkpoint``.
    Callers that need fine-grained lock control (e.g. ``begin_checkpoint``)
    should call the two phases separately.
    """
    backup = backup_databases(db)
    try:
        return upload_checkpoint(db, backup, remote_state_dir)
    finally:
        backup.cleanup()


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


def _sync_dir(source_dir: str, local_dir: Path) -> None:
    """Mirror the immediate file contents of *source_dir* into *local_dir*.

    ``gs://`` sources use ``gcloud storage rsync`` when the CLI is available
    (parallel, skip-if-current transfers). Other schemes fall back to a
    per-file fsspec copy with atomic rename.
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    if source_dir.startswith("gs://") and shutil.which("gcloud") is not None:
        subprocess.check_call(
            ["gcloud", "storage", "rsync", source_dir.rstrip("/") + "/", str(local_dir) + "/"],
            stdout=subprocess.DEVNULL,
        )
        return

    fs, fs_path = fsspec.core.url_to_fs(source_dir)
    if not fs.exists(fs_path):
        return
    for entry in fs.ls(fs_path, detail=False):
        basename = entry.rstrip("/").rsplit("/", 1)[-1]
        dest = local_dir / basename
        tmp = dest.with_suffix(dest.suffix + ".download.tmp")
        _fsspec_copy(entry, str(tmp))
        tmp.rename(dest)


def download_checkpoint_to_local(
    remote_state_dir: str,
    local_db_dir: Path,
    checkpoint_dir: str | None = None,
) -> bool:
    """Sync a remote checkpoint directory to ``local_db_dir``.

    If ``checkpoint_dir`` is not provided, finds the most recent timestamped
    checkpoint under ``{remote_state_dir}/controller-state/``. After the sync,
    any ``.zst`` file is decompressed to its sibling so the resulting layout
    matches what ControllerDB expects.

    Returns True if the sync produced a usable ``controller.sqlite3``.
    """
    if checkpoint_dir:
        source_dir = checkpoint_dir.rstrip("/")
    else:
        found = _find_latest_checkpoint_dir(remote_state_dir)
        if found is None:
            logger.info("No remote checkpoint found under %s, starting fresh", remote_state_dir)
            return False
        source_dir = found

    _sync_dir(source_dir, local_db_dir)
    for zst_path in list(local_db_dir.glob("*.zst")):
        _decompress_zstd(zst_path, zst_path.with_suffix(""))
        zst_path.unlink()

    if not (local_db_dir / ControllerDB.DB_FILENAME).exists():
        logger.info("No checkpoint at %s, starting fresh", source_dir)
        return False

    logger.info("Synced checkpoint %s to %s", source_dir, local_db_dir)
    return True
