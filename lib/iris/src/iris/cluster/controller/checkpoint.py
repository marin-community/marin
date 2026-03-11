# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller checkpoint: types, write, upload, and restore.

All checkpoint logic lives here so that checkpoint names, paths, and
serialization formats are consistent across the write and restore surfaces.

The controller calls into this module for:
- Writing timestamped SQLite checkpoint copies (local + remote upload)
- Restoring controller + autoscaler state from a checkpoint DB
- Serializing/deserializing scaling group snapshots

Note: Several functions use local imports to break circular dependencies.
The data types at the top of this file are imported by autoscaler.py,
scaling_group.py, and transitions.py, which in turn are imported here
for restore/write operations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fsspec.core

from iris.cluster.platform.base import CloudWorkerState, CommandResult, Labels, Platform, SliceHandle, WorkerStatus
from iris.rpc import config_pb2
from iris.time_utils import Deadline, Duration, RateLimiter, Timestamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint data types (serializable snapshots for DB storage)
#
# These types must stay at the top with no imports from autoscaler,
# scaling_group, or transitions to avoid circular imports.
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


def checkpoint_dir(db_path: Path) -> Path:
    """Local directory for checkpoint copies, derived from the DB path."""
    return db_path.parent / "controller-checkpoints"


def latest_checkpoint_path(db_path: Path) -> Path:
    """Path to the ``latest.sqlite3`` symlink/copy."""
    return checkpoint_dir(db_path) / "latest.sqlite3"


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
    transitions: ControllerTransitions,
    db: ControllerDB,
    bundle_prefix: str,
) -> tuple[Path, CheckpointResult]:
    """Write a timestamped SQLite checkpoint copy with local + remote upload.

    Returns the local path and a summary of the checkpoint contents.
    All checkpoint callers funnel through this single method.
    """
    # Local import to break circular dependency.
    from iris.cluster.controller.db import JOBS, TASKS, WORKERS

    created_at = Timestamp.now()
    ckpt_dir = checkpoint_dir(transitions.db_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint-{created_at.epoch_ms()}.sqlite3"
    transitions.backup_to(path)
    _fsspec_copy(str(path), str(latest_checkpoint_path(transitions.db_path)))
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
    transitions: ControllerTransitions,
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
        path, result = write_checkpoint(transitions, db, bundle_prefix)
        logger.info(
            "Periodic checkpoint written: %s (jobs=%d tasks=%d workers=%d)",
            path,
            result.job_count,
            result.task_count,
            result.worker_count,
        )
    except Exception:
        logger.exception("Periodic checkpoint failed")


# ---------------------------------------------------------------------------
# Checkpoint restore: reconcile checkpointed state against live cloud
# ---------------------------------------------------------------------------


class _RestoredWorkerHandle:
    """Minimal handle placeholder used for restored tracked workers."""

    def __init__(self, worker_id: str, internal_address: str) -> None:
        self._worker_id = worker_id
        self._internal_address = internal_address

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def vm_id(self) -> str:
        return self._worker_id

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return None

    @property
    def bootstrap_log(self) -> str:
        return ""

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=CloudWorkerState.RUNNING)

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        raise NotImplementedError("RestoredWorkerHandle does not support run_command")

    def reboot(self) -> None:
        raise NotImplementedError("RestoredWorkerHandle does not support reboot")


@dataclass
class ScalingGroupRestoreResult:
    """Result of restoring a single scaling group from checkpoint metadata."""

    slices: dict[str, SliceState] = field(default_factory=dict)
    consecutive_failures: int = 0
    backoff_active: bool = False
    quota_exceeded_active: bool = False
    quota_reason: str = ""
    discarded_count: int = 0
    adopted_count: int = 0
    last_scale_up: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    last_scale_down: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    backoff_until: Deadline | None = None
    quota_exceeded_until: Deadline | None = None


def restore_tracked_worker(snap: TrackedWorkerSnapshotData) -> TrackedWorker:
    """Restore a single tracked worker from snapshot data."""
    # Local import to break circular dependency (autoscaler imports checkpoint types).
    from iris.cluster.controller.autoscaler import TrackedWorker

    handle = _RestoredWorkerHandle(worker_id=snap.worker_id, internal_address=snap.internal_address)
    return TrackedWorker(
        worker_id=snap.worker_id,
        slice_id=snap.slice_id,
        scale_group=snap.scale_group,
        handle=handle,
    )


def restore_tracked_workers(snapshots: list[TrackedWorkerSnapshotData]) -> dict[str, TrackedWorker]:
    """Restore tracked workers from checkpoint data."""
    # Local import to break circular dependency (autoscaler imports checkpoint types).
    from iris.cluster.controller.autoscaler import TrackedWorker

    workers: dict[str, TrackedWorker] = {}
    for snap in snapshots:
        tw = restore_tracked_worker(snap)
        workers[tw.worker_id] = tw
    return workers


def _wall_clock_to_deadline(wall_clock_ts: Timestamp) -> Deadline | None:
    """Convert a wall-clock timestamp from checkpoint into a deadline."""
    if wall_clock_ts.epoch_ms() == 0:
        return None
    return Deadline.after(wall_clock_ts, Duration.from_ms(0))


def restore_scaling_group(
    group_snapshot: ScalingGroupSnapshotData,
    platform: Platform,
    config: config_pb2.ScaleGroupConfig,
    label_prefix: str,
) -> ScalingGroupRestoreResult:
    """Reconcile checkpointed group slices against live cloud slices."""
    # Local import to break circular dependency (scaling_group imports checkpoint types).
    from iris.cluster.controller.scaling_group import SliceLifecycleState, SliceState, _zones_from_config

    labels = Labels(label_prefix)
    filter_labels = {labels.iris_scale_group: group_snapshot.name}

    zones = _zones_from_config(config)
    cloud_handles = platform.list_slices(zones=zones, labels=filter_labels)
    cloud_by_id: dict[str, SliceHandle] = {h.slice_id: h for h in cloud_handles}
    checkpoint_slices = {s.slice_id: s for s in group_snapshot.slices}

    result = ScalingGroupRestoreResult()
    result.consecutive_failures = group_snapshot.consecutive_failures

    for slice_id, slice_snap in checkpoint_slices.items():
        cloud_handle = cloud_by_id.get(slice_id)
        if cloud_handle is None:
            logger.info("Scaling group %s: discarding slice %s (missing from cloud)", group_snapshot.name, slice_id)
            result.discarded_count += 1
            continue

        try:
            lifecycle = SliceLifecycleState(slice_snap.lifecycle)
        except ValueError:
            logger.warning(
                "Scaling group %s: unknown lifecycle %r for slice %s, defaulting to BOOTING",
                group_snapshot.name,
                slice_snap.lifecycle,
                slice_id,
            )
            lifecycle = SliceLifecycleState.BOOTING

        result.slices[slice_id] = SliceState(
            handle=cloud_handle,
            lifecycle=lifecycle,
            vm_addresses=list(slice_snap.vm_addresses),
            last_active=Timestamp.from_ms(slice_snap.last_active_ms),
            error_message=slice_snap.error_message,
        )

    for slice_id, cloud_handle in cloud_by_id.items():
        if slice_id in checkpoint_slices:
            continue
        logger.info("Scaling group %s: adopting unknown cloud slice %s as BOOTING", group_snapshot.name, slice_id)
        result.slices[slice_id] = SliceState(handle=cloud_handle, lifecycle=SliceLifecycleState.BOOTING)
        result.adopted_count += 1

    backoff_ts = Timestamp.from_ms(group_snapshot.backoff_until_ms)
    if backoff_ts.epoch_ms() > 0:
        result.backoff_until = _wall_clock_to_deadline(backoff_ts)
        result.backoff_active = result.backoff_until is not None and not result.backoff_until.expired()

    quota_ts = Timestamp.from_ms(group_snapshot.quota_exceeded_until_ms)
    if quota_ts.epoch_ms() > 0:
        result.quota_exceeded_until = _wall_clock_to_deadline(quota_ts)
        result.quota_exceeded_active = (
            result.quota_exceeded_until is not None and not result.quota_exceeded_until.expired()
        )
        result.quota_reason = group_snapshot.quota_reason

    if group_snapshot.last_scale_up_ms > 0:
        result.last_scale_up = Timestamp.from_ms(group_snapshot.last_scale_up_ms)
    if group_snapshot.last_scale_down_ms > 0:
        result.last_scale_down = Timestamp.from_ms(group_snapshot.last_scale_down_ms)

    logger.info(
        "Restored scaling group %s: %d slices (%d discarded, %d adopted), consecutive_failures=%d, "
        "backoff_active=%s, quota_exceeded=%s",
        group_snapshot.name,
        len(result.slices),
        result.discarded_count,
        result.adopted_count,
        result.consecutive_failures,
        result.backoff_active,
        result.quota_exceeded_active,
    )
    return result


def restore_from_checkpoint(
    transitions: ControllerTransitions,
    db: ControllerDB,
    autoscaler: Autoscaler | None,
    bundle_prefix: str,
    checkpoint_path: str | None = None,
) -> bool:
    """Restore full controller state from a checkpoint SQLite copy.

    Returns True if a checkpoint was found and restored, False otherwise.
    """
    # Local import to break circular dependency.
    from iris.cluster.controller.db import SCALING_GROUPS, TRACKED_WORKERS

    source = (
        str(Path(checkpoint_path))
        if checkpoint_path
        else str(latest_checkpoint_path(transitions.db_path))
    )
    fs, fs_path = fsspec.core.url_to_fs(source)
    if not fs.exists(fs_path):
        logger.info("No checkpoint DB found at %s, starting fresh", source)
        return False

    transitions.restore_from(source)
    logger.info("Restored checkpoint DB from %s", source)

    with db.snapshot() as snapshot:
        scaling_rows = snapshot.select(
            SCALING_GROUPS,
            columns=(SCALING_GROUPS.c.name, SCALING_GROUPS.c.snapshot_proto),
        )
        tracked_rows = snapshot.select(
            TRACKED_WORKERS,
            columns=(
                TRACKED_WORKERS.c.worker_id,
                TRACKED_WORKERS.c.slice_id,
                TRACKED_WORKERS.c.scale_group,
                TRACKED_WORKERS.c.internal_address,
            ),
        )
    scaling_groups = {}
    for row in scaling_rows:
        scaling_groups[row.name] = deserialize_scaling_group(row.snapshot_proto)

    tracked_worker_snapshots: list[TrackedWorkerSnapshotData] = [
        TrackedWorkerSnapshotData(
            worker_id=row.worker_id,
            slice_id=row.slice_id,
            scale_group=row.scale_group,
            internal_address=row.internal_address,
        )
        for row in tracked_rows
    ]

    # Restore autoscaler scaling groups (parallelized — each calls platform.list_slices())
    if autoscaler is not None:
        groups_to_restore = []
        for group_snap in scaling_groups.values():
            group = autoscaler.groups.get(group_snap.name)
            if group is None:
                logger.warning(
                    "Checkpoint references scaling group %s which does not exist in config, skipping",
                    group_snap.name,
                )
                continue
            groups_to_restore.append((group_snap, group))

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(restore_scaling_group, gs, g.platform, g.config, g.label_prefix): (gs, g)
                for gs, g in groups_to_restore
            }
            for future in as_completed(futures):
                group_snap, group = futures[future]
                restore_result = future.result()
                group.restore_from_snapshot(
                    slices=restore_result.slices,
                    consecutive_failures=restore_result.consecutive_failures,
                    last_scale_up=restore_result.last_scale_up,
                    last_scale_down=restore_result.last_scale_down,
                    backoff_until=restore_result.backoff_until,
                    quota_exceeded_until=restore_result.quota_exceeded_until,
                    quota_reason=restore_result.quota_reason,
                )

        # Workers from discarded slices remain in ControllerTransitions as healthy.
        # They will naturally fail heartbeat checks and be pruned once
        # consecutive failures exceed the threshold. This is intentional:
        # the heartbeat failure path handles cleanup of stale workers
        # including task reassignment and resource release.

        # Restore tracked workers into the autoscaler.
        restored_workers = restore_tracked_workers(tracked_worker_snapshots)
        autoscaler.restore_tracked_workers(restored_workers)
        logger.info("Restored %d tracked workers", len(restored_workers))

    return True
