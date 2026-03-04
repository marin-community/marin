# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller checkpoint and restore.

Provides pure functions for creating and restoring controller snapshots.
The snapshot captures all non-transient state needed to resume controller
operation after a restart: jobs, tasks (with full attempt history), workers
(identity only — health resets to fresh), endpoints, and autoscaler state
(scaling groups with slice inventory, timing state, tracked workers).

The most critical restore logic is scaling group reconciliation: on restore,
we call platform.list_slices() to discover what slices actually exist in the
cloud and reconcile that against the checkpoint state. Slices missing from
the cloud are discarded; slices in the cloud but not in the checkpoint are
adopted as BOOTING.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import fsspec
from google.protobuf import json_format

from iris.cluster.controller.autoscaler import Autoscaler, TrackedWorker
from iris.cluster.controller.scaling_group import SliceLifecycleState, SliceState, _zones_from_config
from iris.cluster.controller.state import (
    ControllerEndpoint,
    ControllerJob,
    ControllerState,
    ControllerTask,
    ControllerTaskAttempt,
    ControllerWorker,
    ReservationClaim,
)
from iris.cluster.platform.base import CloudWorkerState, CommandResult, Labels, Platform, SliceHandle, WorkerStatus
from iris.cluster.types import AttributeValue, JobName, WorkerId
from iris.rpc import cluster_pb2, config_pb2, snapshot_pb2
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class _RestoredWorkerHandle:
    """Minimal RemoteWorkerHandle for tracked workers restored from a snapshot.

    Provides identity fields only. Infrastructure operations (run_command, reboot)
    raise NotImplementedError since the autoscaler will re-discover live workers
    during reconciliation and replace these stubs with real handles.
    """

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
        # Always returns RUNNING since we have no live infrastructure connection.
        # The autoscaler will replace this stub with a real handle during its
        # next reconciliation pass.
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


@dataclass(frozen=True)
class SnapshotResult:
    """Result of a snapshot operation."""

    proto: snapshot_pb2.ControllerSnapshot
    job_count: int
    task_count: int
    worker_count: int


@dataclass(frozen=True)
class RestoreResult:
    """Result of a restore operation."""

    job_count: int
    task_count: int
    worker_count: int
    endpoint_count: int


@dataclass
class ScalingGroupRestoreResult:
    """Result of restoring a single scaling group from a snapshot.

    Contains the reconciled slice inventory and timing state, plus counts
    of slices that were discarded (missing from cloud) or adopted (in cloud
    but not in checkpoint).
    """

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


# =============================================================================
# Snapshot creation
# =============================================================================


def _snapshot_attempt(attempt: ControllerTaskAttempt) -> snapshot_pb2.TaskAttemptSnapshot:
    snap = snapshot_pb2.TaskAttemptSnapshot(
        attempt_id=attempt.attempt_id,
        worker_id=str(attempt.worker_id) if attempt.worker_id else "",
        state=attempt.state,
        log_directory=attempt.log_directory or "",
        exit_code=attempt.exit_code or 0,
        error=attempt.error or "",
    )
    snap.created_at.CopyFrom(attempt.created_at.to_proto())
    if attempt.started_at:
        snap.started_at.CopyFrom(attempt.started_at.to_proto())
    if attempt.finished_at:
        snap.finished_at.CopyFrom(attempt.finished_at.to_proto())
    return snap


def _snapshot_task(task: ControllerTask) -> snapshot_pb2.TaskSnapshot:
    snap = snapshot_pb2.TaskSnapshot(
        task_id=str(task.task_id),
        job_id=str(task.job_id),
        state=task.state,
        error=task.error or "",
        exit_code=task.exit_code or 0,
        max_retries_failure=task.max_retries_failure,
        max_retries_preemption=task.max_retries_preemption,
        failure_count=task.failure_count,
        preemption_count=task.preemption_count,
    )
    snap.submitted_at.CopyFrom(task.submitted_at.to_proto())
    if task.started_at:
        snap.started_at.CopyFrom(task.started_at.to_proto())
    if task.finished_at:
        snap.finished_at.CopyFrom(task.finished_at.to_proto())
    for attempt in task.attempts:
        snap.attempts.append(_snapshot_attempt(attempt))
    return snap


def _snapshot_job(
    job: ControllerJob,
    tasks: list[ControllerTask],
    created_at: Timestamp,
) -> snapshot_pb2.JobSnapshot:
    snap = snapshot_pb2.JobSnapshot(
        job_id=str(job.job_id),
        state=job.state,
        error=job.error or "",
        exit_code=job.exit_code or 0,
        num_tasks=job.num_tasks,
        is_reservation_holder=job.is_reservation_holder,
    )
    snap.request.CopyFrom(job.request)
    snap.submitted_at.CopyFrom(job.submitted_at.to_proto())
    snap.root_submitted_at.CopyFrom(job.root_submitted_at.to_proto())
    if job.started_at:
        snap.started_at.CopyFrom(job.started_at.to_proto())
    if job.finished_at:
        snap.finished_at.CopyFrom(job.finished_at.to_proto())

    # Store scheduling deadline as wall-clock epoch_ms.
    # Deadlines are monotonic internally, so we convert the remaining time
    # to an absolute wall-clock timestamp for portability across restarts.
    if job.scheduling_deadline is not None:
        remaining_ms = job.scheduling_deadline.remaining_ms()
        snap.scheduling_deadline_epoch_ms = created_at.epoch_ms() + remaining_ms

    for task in tasks:
        snap.tasks.append(_snapshot_task(task))
    return snap


def _snapshot_worker(worker: ControllerWorker) -> snapshot_pb2.WorkerSnapshot:
    snap = snapshot_pb2.WorkerSnapshot(
        worker_id=str(worker.worker_id),
        address=worker.address,
    )
    snap.metadata.CopyFrom(worker.metadata)
    for key, attr in worker.attributes.items():
        snap.attributes[key].CopyFrom(attr.to_proto())
    return snap


def _snapshot_endpoint(
    endpoint: ControllerEndpoint,
    task_id: JobName | None = None,
) -> snapshot_pb2.EndpointSnapshot:
    snap = snapshot_pb2.EndpointSnapshot(
        endpoint_id=endpoint.endpoint_id,
        name=endpoint.name,
        address=endpoint.address,
        job_id=str(endpoint.job_id),
        task_id=str(task_id) if task_id else "",
    )
    for k, v in endpoint.metadata.items():
        snap.metadata[k] = v
    snap.registered_at.CopyFrom(endpoint.registered_at.to_proto())
    return snap


def _restore_tracked_worker(snap: snapshot_pb2.TrackedWorkerSnapshot) -> TrackedWorker:
    handle = _RestoredWorkerHandle(
        worker_id=snap.worker_id,
        internal_address=snap.internal_address,
    )
    return TrackedWorker(
        worker_id=snap.worker_id,
        slice_id=snap.slice_id,
        scale_group=snap.scale_group,
        handle=handle,
    )


def restore_tracked_workers(
    proto: snapshot_pb2.ControllerSnapshot,
) -> dict[str, TrackedWorker]:
    """Restore tracked workers from a snapshot proto."""
    workers: dict[str, TrackedWorker] = {}
    for tw_snap in proto.tracked_workers:
        tw = _restore_tracked_worker(tw_snap)
        workers[tw.worker_id] = tw
    return workers


def create_snapshot(
    state: ControllerState,
    autoscaler: Autoscaler | None = None,
    reservation_claims: dict[WorkerId, "ReservationClaim"] | None = None,
) -> SnapshotResult:
    """Create a snapshot from current controller state.

    Pure function: reads state under locks, returns a proto.
    Captures jobs, tasks, workers, endpoints, and autoscaler state
    (scaling groups with slice inventory, timing state, tracked workers).
    """
    created_at = Timestamp.now()
    proto = snapshot_pb2.ControllerSnapshot(schema_version=SCHEMA_VERSION)
    proto.created_at.CopyFrom(created_at.to_proto())

    total_tasks = 0

    jobs = state.list_all_jobs()
    for job in jobs:
        tasks = state.get_job_tasks(job.job_id)
        proto.jobs.append(_snapshot_job(job, tasks, created_at))
        total_tasks += len(tasks)

    for worker in state.list_all_workers():
        proto.workers.append(_snapshot_worker(worker))

    endpoint_task_map = state.get_endpoint_task_mapping()
    for endpoint in state.list_all_endpoints():
        task_id = endpoint_task_map.get(endpoint.endpoint_id)
        proto.endpoints.append(_snapshot_endpoint(endpoint, task_id=task_id))

    # Snapshot autoscaler state
    if autoscaler is not None:
        for group in autoscaler.groups.values():
            proto.scaling_groups.append(group.to_snapshot())
        for tw_snap in autoscaler.to_tracked_worker_snapshots():
            proto.tracked_workers.append(tw_snap)

    # Snapshot reservation claims
    if reservation_claims is not None:
        for worker_id, claim in reservation_claims.items():
            proto.reservation_claims.append(
                snapshot_pb2.ReservationClaimSnapshot(
                    worker_id=str(worker_id),
                    job_id=claim.job_id,
                    entry_idx=claim.entry_idx,
                )
            )

    return SnapshotResult(
        proto=proto,
        job_count=len(jobs),
        task_count=total_tasks,
        worker_count=len(proto.workers),
    )


# =============================================================================
# Snapshot restore
# =============================================================================

# Active task states where the task is bound to a worker.
_ACTIVE_TASK_STATES = frozenset(
    {
        cluster_pb2.TASK_STATE_ASSIGNED,
        cluster_pb2.TASK_STATE_BUILDING,
        cluster_pb2.TASK_STATE_RUNNING,
    }
)


def _restore_attempt(snap: snapshot_pb2.TaskAttemptSnapshot) -> ControllerTaskAttempt:
    return ControllerTaskAttempt(
        attempt_id=snap.attempt_id,
        worker_id=WorkerId(snap.worker_id) if snap.worker_id else None,
        state=snap.state,
        log_directory=snap.log_directory or None,
        created_at=Timestamp.from_proto(snap.created_at),
        started_at=Timestamp.from_proto(snap.started_at) if snap.started_at.epoch_ms else None,
        finished_at=Timestamp.from_proto(snap.finished_at) if snap.finished_at.epoch_ms else None,
        exit_code=snap.exit_code if snap.exit_code != 0 else None,
        error=snap.error if snap.error else None,
    )


def _restore_task(snap: snapshot_pb2.TaskSnapshot) -> ControllerTask:
    attempts = [_restore_attempt(a) for a in snap.attempts]
    return ControllerTask(
        task_id=JobName.from_string(snap.task_id),
        job_id=JobName.from_string(snap.job_id),
        state=snap.state,
        error=snap.error if snap.error else None,
        exit_code=snap.exit_code if snap.exit_code != 0 else None,
        started_at=Timestamp.from_proto(snap.started_at) if snap.started_at.epoch_ms else None,
        finished_at=Timestamp.from_proto(snap.finished_at) if snap.finished_at.epoch_ms else None,
        max_retries_failure=snap.max_retries_failure,
        max_retries_preemption=snap.max_retries_preemption,
        failure_count=snap.failure_count,
        preemption_count=snap.preemption_count,
        attempts=attempts,
        submitted_at=Timestamp.from_proto(snap.submitted_at),
    )


def _restore_job(
    snap: snapshot_pb2.JobSnapshot,
) -> tuple[ControllerJob, list[ControllerTask]]:
    """Restore a job and its tasks from a snapshot.

    The ``is_reservation_holder`` flag is persisted explicitly in the proto
    so restore is unambiguous and cannot misclassify normal jobs.
    """
    job_id = JobName.from_string(snap.job_id)
    tasks = [_restore_task(t) for t in snap.tasks]

    is_holder = snap.is_reservation_holder

    # Do NOT pre-populate task_state_counts here. state.add_job() will
    # iterate the tasks and increment counts itself. Building them here
    # would cause every task state to be double-counted.
    job = ControllerJob(
        job_id=job_id,
        request=snap.request,
        state=snap.state,
        submitted_at=Timestamp.from_proto(snap.submitted_at),
        root_submitted_at=Timestamp.from_proto(snap.root_submitted_at),
        started_at=Timestamp.from_proto(snap.started_at) if snap.started_at.epoch_ms else None,
        finished_at=Timestamp.from_proto(snap.finished_at) if snap.finished_at.epoch_ms else None,
        error=snap.error if snap.error else None,
        exit_code=snap.exit_code if snap.exit_code != 0 else None,
        num_tasks=snap.num_tasks,
        is_reservation_holder=is_holder,
    )

    # Restore scheduling deadline from wall-clock epoch_ms
    if snap.scheduling_deadline_epoch_ms > 0:
        remaining_ms = snap.scheduling_deadline_epoch_ms - Timestamp.now().epoch_ms()
        if remaining_ms > 0:
            job.scheduling_deadline = Deadline.from_now(Duration.from_ms(remaining_ms))
        else:
            job.scheduling_deadline = Deadline.from_now(Duration.from_ms(0))

    return job, tasks


def _restore_worker(snap: snapshot_pb2.WorkerSnapshot) -> ControllerWorker:
    """Restore a worker with fresh health state."""
    attributes = {k: AttributeValue.from_proto(v) for k, v in snap.attributes.items()}
    return ControllerWorker(
        worker_id=WorkerId(snap.worker_id),
        address=snap.address,
        metadata=snap.metadata,
        healthy=True,
        consecutive_failures=0,
        last_heartbeat=Timestamp.now(),
        attributes=attributes,
    )


def _restore_endpoint(snap: snapshot_pb2.EndpointSnapshot) -> tuple[ControllerEndpoint, JobName | None]:
    """Restore an endpoint and its task association from a snapshot."""
    task_id = JobName.from_string(snap.task_id) if snap.task_id else None
    endpoint = ControllerEndpoint(
        endpoint_id=snap.endpoint_id,
        name=snap.name,
        address=snap.address,
        job_id=JobName.from_string(snap.job_id),
        metadata=dict(snap.metadata),
        registered_at=Timestamp.from_proto(snap.registered_at),
    )
    return endpoint, task_id


def restore_snapshot(
    snapshot: snapshot_pb2.ControllerSnapshot,
    state: ControllerState,
) -> RestoreResult:
    """Restore controller state from a snapshot.

    Populates the (empty) state with checkpointed data:
    - Jobs and tasks with full state and attempt history
    - Workers with fresh health state (healthy=True, last_heartbeat=now())
    - Running task assignments rebuilt from task state
    - Derived indexes (_tasks_by_job, _task_queue, _endpoints_by_task) rebuilt

    Must be called before starting the scheduling/heartbeat loops.
    """
    if snapshot.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible snapshot schema version: got {snapshot.schema_version}, " f"expected {SCHEMA_VERSION}"
        )

    # Restore workers first so we can rebuild running_tasks from task state
    workers: dict[WorkerId, ControllerWorker] = {}
    for w_snap in snapshot.workers:
        worker = _restore_worker(w_snap)
        workers[worker.worker_id] = worker
        state.add_worker(worker)

    total_tasks = 0
    for job_snap in snapshot.jobs:
        job, tasks = _restore_job(job_snap)
        state.add_job(job, tasks)
        total_tasks += len(tasks)

        # Rebuild worker running_tasks and committed resources from active task state.
        # Holder tasks consume zero resources — only track them in running_tasks.
        for task in tasks:
            if task.state in _ACTIVE_TASK_STATES and task.worker_id:
                worker = workers.get(task.worker_id)
                if worker:
                    if job.is_reservation_holder:
                        worker.running_tasks.add(task.task_id)
                        worker.task_history.add(task.task_id)
                    else:
                        worker.assign_task(task.task_id, job.request.resources)

    # Restore endpoints with their task associations
    endpoint_count = 0
    for ep_snap in snapshot.endpoints:
        endpoint, task_id = _restore_endpoint(ep_snap)
        state.add_endpoint(endpoint, task_id=task_id)
        endpoint_count += 1

    logger.info(
        "Restored snapshot: jobs=%d tasks=%d workers=%d endpoints=%d schema_version=%d",
        len(snapshot.jobs),
        total_tasks,
        len(snapshot.workers),
        endpoint_count,
        snapshot.schema_version,
    )

    return RestoreResult(
        job_count=len(snapshot.jobs),
        task_count=total_tasks,
        worker_count=len(snapshot.workers),
        endpoint_count=endpoint_count,
    )


# =============================================================================
# Scaling group reconciliation
# =============================================================================


def _wall_clock_to_deadline(
    wall_clock_ts: Timestamp,
) -> Deadline | None:
    """Convert a wall-clock timestamp from a snapshot into a timestamp-mode Deadline.

    Returns None if the timestamp is zero (field was not set in the snapshot).
    Uses Deadline.after() so the result supports as_timestamp(), which is
    required by ScalingGroup.availability() and to_status().
    """
    if wall_clock_ts.epoch_ms() == 0:
        return None
    return Deadline.after(wall_clock_ts, Duration.from_ms(0))


def restore_scaling_group(
    group_snapshot: snapshot_pb2.ScalingGroupSnapshot,
    platform: Platform,
    config: config_pb2.ScaleGroupConfig,
    label_prefix: str,
) -> ScalingGroupRestoreResult:
    """Reconcile a checkpointed scaling group against live cloud state.

    Calls platform.list_slices() to discover what slices actually exist, then:
    - Slices in both checkpoint and cloud: keep, restore checkpoint lifecycle, attach cloud handle
    - Slices in checkpoint but NOT in cloud: discard (terminated during restart)
    - Slices in cloud but NOT in checkpoint: adopt as BOOTING (in-flight scale-up completed)

    Timing state (backoff, cooldowns) is restored with wall-clock-to-monotonic conversion.
    """
    labels = Labels(label_prefix)
    filter_labels = {labels.iris_scale_group: group_snapshot.name}

    # Discover live cloud slices for this group.
    # Use zones from config so GCP list_slices (which iterates per-zone) works correctly.
    zones = _zones_from_config(config)
    cloud_handles = platform.list_slices(zones=zones, labels=filter_labels)
    cloud_by_id: dict[str, SliceHandle] = {h.slice_id: h for h in cloud_handles}

    # Build checkpoint slice index
    checkpoint_slices = {s.slice_id: s for s in group_snapshot.slices}

    result = ScalingGroupRestoreResult()
    result.consecutive_failures = group_snapshot.consecutive_failures

    # Reconcile: checkpoint slices vs cloud
    for slice_id, slice_snap in checkpoint_slices.items():
        cloud_handle = cloud_by_id.get(slice_id)
        if cloud_handle is None:
            logger.info(
                "Scaling group %s: discarding slice %s (missing from cloud)",
                group_snapshot.name,
                slice_id,
            )
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
        state = SliceState(
            handle=cloud_handle,
            lifecycle=lifecycle,
            vm_addresses=list(slice_snap.vm_addresses),
            last_active=Timestamp.from_proto(slice_snap.last_active),
            error_message=slice_snap.error_message,
        )
        result.slices[slice_id] = state

    # Adopt cloud slices not in checkpoint
    for slice_id, cloud_handle in cloud_by_id.items():
        if slice_id in checkpoint_slices:
            continue
        logger.info(
            "Scaling group %s: adopting unknown cloud slice %s as BOOTING",
            group_snapshot.name,
            slice_id,
        )
        state = SliceState(
            handle=cloud_handle,
            lifecycle=SliceLifecycleState.BOOTING,
        )
        result.slices[slice_id] = state
        result.adopted_count += 1

    # Restore timing state
    if group_snapshot.HasField("backoff_until") and group_snapshot.backoff_until.epoch_ms > 0:
        backoff_ts = Timestamp.from_proto(group_snapshot.backoff_until)
        result.backoff_until = _wall_clock_to_deadline(backoff_ts)
        result.backoff_active = result.backoff_until is not None and not result.backoff_until.expired()
    else:
        result.backoff_active = False

    if group_snapshot.HasField("quota_exceeded_until") and group_snapshot.quota_exceeded_until.epoch_ms > 0:
        quota_ts = Timestamp.from_proto(group_snapshot.quota_exceeded_until)
        result.quota_exceeded_until = _wall_clock_to_deadline(quota_ts)
        result.quota_exceeded_active = (
            result.quota_exceeded_until is not None and not result.quota_exceeded_until.expired()
        )
        result.quota_reason = group_snapshot.quota_reason
    else:
        result.quota_exceeded_active = False

    if group_snapshot.HasField("last_scale_up") and group_snapshot.last_scale_up.epoch_ms > 0:
        result.last_scale_up = Timestamp.from_proto(group_snapshot.last_scale_up)
    if group_snapshot.HasField("last_scale_down") and group_snapshot.last_scale_down.epoch_ms > 0:
        result.last_scale_down = Timestamp.from_proto(group_snapshot.last_scale_down)

    logger.info(
        "Restored scaling group %s: %d slices (%d discarded, %d adopted), "
        "consecutive_failures=%d, backoff_active=%s, quota_exceeded=%s",
        group_snapshot.name,
        len(result.slices),
        result.discarded_count,
        result.adopted_count,
        result.consecutive_failures,
        result.backoff_active,
        result.quota_exceeded_active,
    )

    return result


# =============================================================================
# Storage I/O
# =============================================================================


def _snapshot_dir(storage_prefix: str) -> str:
    """Compute the snapshot directory from a storage prefix.

    Path: {storage_prefix}/controller-snapshots/
    """
    prefix = storage_prefix.rstrip("/")
    return f"{prefix}/controller-snapshots"


def write_snapshot(
    snapshot: snapshot_pb2.ControllerSnapshot,
    storage_prefix: str,
) -> str:
    """Write snapshot to storage as JSON. Returns the path of the timestamped file.

    Writes directly to the final path (no tmp+rename). GCS object writes are
    already atomic, and GCS rename is implemented as copy+delete which is
    strictly worse than a direct write.
    """

    snapshot_dir = _snapshot_dir(storage_prefix)
    timestamp_ms = snapshot.created_at.epoch_ms
    path = f"{snapshot_dir}/snapshot-{timestamp_ms}.json"
    latest_path = f"{snapshot_dir}/latest.json"

    json_bytes = json_format.MessageToJson(snapshot).encode("utf-8")

    fs, fs_path = fsspec.core.url_to_fs(path)
    fs.makedirs(fs.sep.join(fs_path.rsplit(fs.sep, 1)[:-1]), exist_ok=True)
    fs.pipe_file(fs_path, json_bytes)

    # Write latest.json
    _, latest_fs_path = fsspec.core.url_to_fs(latest_path)
    fs.pipe_file(latest_fs_path, json_bytes)

    logger.info("Wrote snapshot to %s (%d bytes)", path, len(json_bytes))
    return path


def read_snapshot_from_path(
    path: str,
) -> snapshot_pb2.ControllerSnapshot | None:
    """Read a snapshot from a specific file path, or None if not found."""

    fs, fs_path = fsspec.core.url_to_fs(path)
    if not fs.exists(fs_path):
        return None

    json_bytes = fs.cat_file(fs_path)
    proto = snapshot_pb2.ControllerSnapshot()
    json_format.Parse(json_bytes, proto, ignore_unknown_fields=True)
    return proto


def read_latest_snapshot(
    storage_prefix: str,
) -> snapshot_pb2.ControllerSnapshot | None:
    """Read the latest snapshot from storage, or None if not found."""

    snapshot_dir = _snapshot_dir(storage_prefix)
    latest_path = f"{snapshot_dir}/latest.json"

    fs, fs_path = fsspec.core.url_to_fs(latest_path)
    if not fs.exists(fs_path):
        return None

    json_bytes = fs.cat_file(fs_path)
    proto = snapshot_pb2.ControllerSnapshot()
    json_format.Parse(json_bytes, proto, ignore_unknown_fields=True)
    return proto
