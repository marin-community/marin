# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller checkpoint: SQLite backup, upload, and snapshot data types.

This module handles only checkpoint I/O: writing timestamped SQLite copies,
uploading to remote storage, and restoring the DB file from a checkpoint.

Autoscaler/scaling-group reconciliation lives in autoscaler.py and
scaling_group.py respectively.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fsspec.core

from iris.cluster.controller.db import JOBS, TASKS, WORKERS, ControllerDB
from iris.time_utils import RateLimiter, Timestamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint data types (serializable snapshots for DB storage)
# ---------------------------------------------------------------------------


@dataclass
class SliceSnapshotData:
    """Serializable snapshot of a single slice."""

    slice_id: str
    scale_group: str
    lifecycle: str
    vm_addresses: list[str] = field(default_factory=list)
    created_at_ms: int = 0
    last_active_ms: int = 0
    error_message: str = ""


@dataclass
class ScalingGroupSnapshotData:
    """Serializable snapshot of a scaling group."""

    name: str
    slices: list[SliceSnapshotData] = field(default_factory=list)
    consecutive_failures: int = 0
    backoff_until_ms: int = 0
    last_scale_up_ms: int = 0
    last_scale_down_ms: int = 0
    quota_exceeded_until_ms: int = 0
    quota_reason: str = ""


@dataclass
class TrackedWorkerSnapshotData:
    """Serializable snapshot of a tracked worker."""

    worker_id: str
    slice_id: str
    scale_group: str
    internal_address: str


def serialize_scaling_group(data: ScalingGroupSnapshotData) -> bytes:
    """Serialize a ScalingGroupSnapshotData to bytes for DB storage."""
    return json.dumps(asdict(data)).encode()


def deserialize_scaling_group(raw: bytes) -> ScalingGroupSnapshotData:
    """Deserialize a ScalingGroupSnapshotData from bytes."""
    d = json.loads(raw)
    d["slices"] = [SliceSnapshotData(**s) for s in d.get("slices", [])]
    return ScalingGroupSnapshotData(**d)


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
# Checkpoint write + upload
# ---------------------------------------------------------------------------


def _fsspec_copy(src: str, dst: str) -> None:
    """Copy a file using fsspec so either path can be remote (e.g. GCS)."""
    with fsspec.core.open(src, "rb") as f_src, fsspec.core.open(dst, "wb") as f_dst:
        f_dst.write(f_src.read())


def is_remote_path(path: str) -> bool:
    """Return True if *path* uses a remote fsspec scheme (e.g. ``gs://``, ``s3://``)."""
    return "://" in path and not path.startswith("file://")


CHECKPOINT_DIR_NAME = "controller-checkpoints"


def remote_checkpoint_prefix(bundle_prefix: str) -> str | None:
    """Remote fsspec path for checkpoint uploads, or None for local-only."""
    if bundle_prefix and is_remote_path(bundle_prefix):
        return bundle_prefix.rstrip("/") + "/controller-state"
    return None


def upload_checkpoint_to_remote(local_path: Path, created_at: Timestamp, bundle_prefix: str) -> None:
    """Upload a local checkpoint file to remote storage via fsspec."""
    prefix = remote_checkpoint_prefix(bundle_prefix)
    if prefix is None:
        return
    try:
        remote_timestamped = f"{prefix}/checkpoint-{created_at.epoch_ms()}.sqlite3"
        remote_latest = f"{prefix}/latest.sqlite3"
        _fsspec_copy(str(local_path), remote_timestamped)
        _fsspec_copy(str(local_path), remote_latest)
        logger.info("Checkpoint uploaded to %s", remote_timestamped)
    except Exception:
        logger.exception("Failed to upload checkpoint to remote storage")


def write_checkpoint(
    db: ControllerDB,
    bundle_prefix: str,
) -> tuple[Path, CheckpointResult]:
    """Write a timestamped SQLite checkpoint copy with local + remote upload.

    Returns the local path and a summary of the checkpoint contents.
    """
    created_at = Timestamp.now()
    ckpt_dir = db.db_path.parent / CHECKPOINT_DIR_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint-{created_at.epoch_ms()}.sqlite3"
    db.backup_to(path)
    _fsspec_copy(str(path), str(ckpt_dir / "latest.sqlite3"))
    upload_checkpoint_to_remote(path, created_at, bundle_prefix)
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
    return path, result


def maybe_periodic_checkpoint(
    db: ControllerDB,
    bundle_prefix: str,
    limiter: RateLimiter | None,
    checkpoint_in_progress: bool,
) -> None:
    """Write a best-effort periodic checkpoint DB copy."""
    if limiter is None:
        return
    if checkpoint_in_progress:
        return
    if not limiter.should_run():
        return
    try:
        path, result = write_checkpoint(db, bundle_prefix)
        logger.info(
            "Periodic checkpoint written: %s (jobs=%d tasks=%d workers=%d)",
            path,
            result.job_count,
            result.task_count,
            result.worker_count,
        )
    except Exception:
        logger.exception("Periodic checkpoint failed")


def restore_db_from_checkpoint(
    db: ControllerDB,
    bundle_prefix: str,
    checkpoint_path: str | None = None,
) -> bool:
    """Restore the SQLite DB file from a checkpoint. Returns True if found."""
    source = (
        str(Path(checkpoint_path))
        if checkpoint_path
        else str(db.db_path.parent / CHECKPOINT_DIR_NAME / "latest.sqlite3")
    )
    fs, fs_path = fsspec.core.url_to_fs(source)
    if not fs.exists(fs_path):
        logger.info("No checkpoint DB found at %s, starting fresh", source)
        return False

    db.replace_from(source)
    logger.info("Restored checkpoint DB from %s", source)
    return True
