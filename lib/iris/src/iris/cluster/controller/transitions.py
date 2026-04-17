# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller state machine: all DB-mutating transitions live here.

Read-only queries do NOT belong here — callers use db.read_snapshot() directly.
"""

import enum
import threading
import time
from collections import defaultdict
import json
import logging
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple

from iris.cluster.constraints import AttributeValue, Constraint, constraints_from_resources, merge_constraints
from iris.cluster.controller.budget import UserBudgetDefaults
from iris.cluster.controller.codec import (
    constraints_from_json,
    constraints_to_json,
    entrypoint_to_json,
    proto_from_json,
    proto_to_json,
    reservation_to_json,
    resource_spec_from_scalars,
)
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    EXECUTING_TASK_STATES,
    FAILURE_TASK_STATES,
    ControllerDB,
    TransactionCursor,
    task_row_can_be_scheduled,
    task_row_is_finished,
)
from iris.cluster.controller.schema import (
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
    TASK_DETAIL_PROJECTION,
    WORKER_DETAIL_PROJECTION,
    EndpointRow,
    JobDetailRow,
    WorkerDetailRow,
)
from iris.cluster.types import (
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    JobName,
    WorkerId,
    get_gpu_count,
    get_tpu_count,
)
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.time_proto import duration_from_proto
from rigging.timing import Duration, Timestamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReservationClaim:
    """A claim binding a worker to a specific reservation entry.

    The controller assigns unclaimed workers to unsatisfied reservation entries
    each scheduling cycle. Once every entry for a job is claimed, the
    reservation gate opens and the job's tasks can be scheduled.
    """

    job_id: str
    entry_idx: int


MAX_REPLICAS_PER_JOB = 10000
"""Maximum replicas allowed per job to prevent resource exhaustion."""

DEFAULT_MAX_RETRIES_PREEMPTION = 100
"""Default preemption retries. High because worker failures are typically transient."""

RESERVATION_HOLDER_JOB_NAME = ":reservation:"
"""Well-known name component for synthetic reservation holder child jobs.

Uses colons to clearly distinguish from user-created jobs and avoid
accidental collision with normal job names."""

HEARTBEAT_FAILURE_THRESHOLD = 10
"""Consecutive heartbeat failures before marking worker as failed."""

FAIL_HEARTBEATS_CHUNK_SIZE = 10
"""Number of worker failures processed per transaction in
``fail_heartbeats_batch``. Commits between chunks so the SQLite writer is
released and other RPCs (RegisterEndpoint, Register, LaunchJob,
apply_heartbeats_batch) can interleave. Keeps worst-case writer-hold
below ~1s even when a zone-wide failure removes hundreds of workers."""

HEARTBEAT_STALENESS_THRESHOLD = Duration.from_seconds(900)
"""If a worker's last successful heartbeat is older than this, it is failed
immediately. Catches workers restored from a checkpoint whose backing VMs
no longer exist — without this, the controller would need 10 consecutive
RPC failures (50s) per worker to notice, during which they appear healthy
in the dashboard and block scheduling capacity."""

WORKER_TASK_HISTORY_RETENTION = 500
"""Maximum worker_task_history rows retained per worker."""

WORKER_RESOURCE_HISTORY_RETENTION = 500
"""Maximum worker_resource_history rows retained per worker."""

TASK_RESOURCE_HISTORY_RETENTION = 50
"""Maximum task_resource_history rows retained per (task_id, attempt_id).
Logarithmic downsampling triggers at 2x this value."""

TASK_RESOURCE_HISTORY_TERMINAL_TTL = Duration.from_hours(1)
"""After a task reaches a terminal state, its resource history is fully
evicted this long after the finish timestamp. Dashboards surface peak
memory from tasks.peak_memory_mb once a task is done; retaining per-sample
rows forever bloats the DB (~85% of task_resource_history on prod is for
terminal tasks) and amplifies writer contention during heartbeat batches."""

TASK_RESOURCE_HISTORY_DELETE_CHUNK = 1000
"""Maximum task_ids per DELETE in prune_task_resource_history. Each chunk
is its own write transaction so the writer lock releases between chunks.
Sweep on a 1M-row prod checkpoint: chunk=1000 gives p95 ~400ms / max 1.3s
writer hold; chunk=5000 gives p95 1.6s / max 1.7s. The background loop
runs every 10 min so total wall-time is irrelevant — bounding worst-case
writer hold is what matters for concurrent RPCs."""

DIRECT_PROVIDER_PROMOTION_RATE = 128
"""Token bucket capacity for task promotion (pods per minute).

The direct provider relies on the Kubernetes scheduler (and the cloud
autoscaler) for placement and capacity management.  Pods that cannot be
scheduled immediately stay Pending — that signal drives node provisioning.
This rate limit exists only to bound API server pressure."""


@dataclass(frozen=True)
class PruneResult:
    """Counts of rows deleted by prune_old_data."""

    jobs_deleted: int = 0
    workers_deleted: int = 0
    txn_actions_deleted: int = 0
    profiles_deleted: int = 0

    @property
    def total(self) -> int:
        return self.jobs_deleted + self.workers_deleted + self.txn_actions_deleted + self.profiles_deleted


class HeartbeatAction(enum.Enum):
    """Result of processing a single heartbeat response."""

    OK = "ok"
    TRANSIENT_FAILURE = "transient_failure"
    WORKER_FAILED = "worker_failed"


@dataclass
class WorkerConfig:
    """Static worker configuration for v0.

    Args:
        worker_id: Unique worker identifier
        address: Worker RPC address (host:port)
        metadata: Worker environment metadata
    """

    worker_id: str
    address: str
    metadata: job_pb2.WorkerMetadata


@dataclass(frozen=True)
class TaskUpdate:
    """Single task state update applied in a batch."""

    task_id: JobName
    attempt_id: int
    new_state: int
    error: str | None = None
    exit_code: int | None = None
    resource_usage: job_pb2.ResourceUsage | None = None
    container_id: str | None = None


def task_updates_from_proto(entries) -> list[TaskUpdate]:
    """Convert worker-reported WorkerTaskStatus protos into TaskUpdates.

    Skips UNSPECIFIED/PENDING — the controller is only interested in
    transitions to BUILDING or beyond.
    """
    updates: list[TaskUpdate] = []
    for entry in entries:
        if entry.state in (job_pb2.TASK_STATE_UNSPECIFIED, job_pb2.TASK_STATE_PENDING):
            continue
        updates.append(
            TaskUpdate(
                task_id=JobName.from_wire(entry.task_id),
                attempt_id=entry.attempt_id,
                new_state=entry.state,
                error=entry.error or None,
                exit_code=entry.exit_code if entry.HasField("exit_code") else None,
                resource_usage=entry.resource_usage if entry.resource_usage.ByteSize() > 0 else None,
                container_id=entry.container_id or None,
            )
        )
    return updates


@dataclass(frozen=True)
class HeartbeatApplyRequest:
    """Batch of worker heartbeat updates applied atomically."""

    worker_id: WorkerId
    worker_resource_snapshot: job_pb2.WorkerResourceSnapshot | None
    updates: list[TaskUpdate]


@dataclass(frozen=True)
class Assignment:
    """Scheduler assignment decision."""

    task_id: JobName
    worker_id: WorkerId


@dataclass(frozen=True)
class TxResult:
    """Result payload from a state command transaction."""

    tasks_to_kill: set[JobName] = field(default_factory=set)
    task_kill_workers: dict[JobName, WorkerId] = field(default_factory=dict)
    has_real_dispatch: bool = False


@dataclass(frozen=True)
class AssignmentResult(TxResult):
    """Result of queue_assignments."""

    accepted: list[Assignment] = field(default_factory=list)
    rejected: list[Assignment] = field(default_factory=list)
    start_requests: list[tuple[WorkerId, str, job_pb2.RunTaskRequest]] = field(default_factory=list)


@dataclass(frozen=True)
class SubmitJobResult:
    job_id: JobName
    task_ids: list[JobName]


@dataclass(frozen=True)
class WorkerRegistrationResult:
    worker_id: WorkerId


@dataclass(frozen=True)
class HeartbeatApplyResult(TxResult):
    action: HeartbeatAction = HeartbeatAction.OK


@dataclass(frozen=True)
class HeartbeatFailureResult(TxResult):
    worker_removed: bool = False
    action: HeartbeatAction = HeartbeatAction.TRANSIENT_FAILURE
    consecutive_failures: int = 0
    failure_threshold: int = HEARTBEAT_FAILURE_THRESHOLD
    last_heartbeat_age_ms: int | None = None


@dataclass(frozen=True)
class WorkerFailureBatchResult(TxResult):
    """Result of applying a batch of worker failures."""

    removed_workers: list[tuple[WorkerId, str | None]] = field(default_factory=list)
    results: list[HeartbeatFailureResult] = field(default_factory=list)


class RunningTaskEntry(NamedTuple):
    """Task ID and attempt ID pair captured at snapshot time."""

    task_id: JobName
    attempt_id: int


@dataclass(frozen=True)
class DispatchBatch:
    """Drained worker dispatch plus running-task snapshot."""

    worker_id: WorkerId
    worker_address: str | None
    running_tasks: list[RunningTaskEntry]
    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    tasks_to_kill: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SchedulingEvent:
    """A scheduling event from the execution backend (e.g. k8s events)."""

    task_id: str
    attempt_id: int
    event_type: str
    reason: str
    message: str
    timestamp: Timestamp


@dataclass(frozen=True)
class ClusterCapacity:
    """Aggregate capacity reported by the execution backend."""

    schedulable_nodes: int
    total_cpu_millicores: int
    available_cpu_millicores: int
    total_memory_bytes: int
    available_memory_bytes: int


@dataclass(frozen=True)
class DirectProviderBatch:
    """Work batch for a KubernetesProvider sync cycle.

    Unlike DispatchBatch, there is no worker_id — tasks run without a registered
    worker daemon. task_attempts rows use NULL worker_id.
    """

    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    running_tasks: list[RunningTaskEntry] = field(default_factory=list)
    tasks_to_kill: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DirectProviderSyncResult:
    """Result from a KubernetesProvider sync cycle."""

    updates: list[TaskUpdate] = field(default_factory=list)
    scheduling_events: list[SchedulingEvent] = field(default_factory=list)
    capacity: ClusterCapacity | None = None


def _has_reservation_flag(request: controller_pb2.Controller.LaunchJobRequest) -> int:
    """Return 1 if the request carries reservation entries, else 0."""
    return 1 if request.HasField("reservation") and request.reservation.entries else 0


def delete_task_endpoints(cur: TransactionCursor, registry, task_id: str) -> None:
    """Remove all registered endpoints for a task through the endpoint registry."""
    registry.remove_by_task(cur, JobName.from_wire(task_id))


def enqueue_run_dispatch(
    cur: TransactionCursor,
    worker_id: str,
    payload_proto: bytes,
    now_ms: int,
) -> None:
    """Queue a 'run' dispatch entry for delivery on the next heartbeat."""
    cur.execute(
        "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
        "VALUES (?, 'run', ?, NULL, ?)",
        (worker_id, payload_proto, now_ms),
    )


def enqueue_kill_dispatch(
    cur: TransactionCursor,
    worker_id: str | None,
    task_id: str,
    now_ms: int,
) -> None:
    """Queue a 'kill' dispatch entry for delivery on the next heartbeat."""
    cur.execute(
        "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
        "VALUES (?, 'kill', NULL, ?, ?)",
        (worker_id, task_id, now_ms),
    )


def insert_task_attempt(
    cur: TransactionCursor,
    task_id: str,
    attempt_id: int,
    worker_id: str | None,
    state: int,
    now_ms: int,
) -> None:
    """Record a new task attempt row."""
    cur.execute(
        "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) " "VALUES (?, ?, ?, ?, ?)",
        (task_id, attempt_id, worker_id, state, now_ms),
    )


def _decommit_worker_resources(
    cur: TransactionCursor,
    worker_id: str,
    resources: "job_pb2.ResourceSpecProto",
) -> None:
    """Subtract a task's resource reservation from a worker, flooring at zero."""
    cur.execute(
        "UPDATE workers SET committed_cpu_millicores = MAX(0, committed_cpu_millicores - ?), "
        "committed_mem_bytes = MAX(0, committed_mem_bytes - ?), "
        "committed_gpu = MAX(0, committed_gpu - ?), committed_tpu = MAX(0, committed_tpu - ?) "
        "WHERE worker_id = ?",
        (
            int(resources.cpu_millicores),
            int(resources.memory_bytes),
            int(get_gpu_count(resources.device)),
            int(get_tpu_count(resources.device)),
            worker_id,
        ),
    )


def _remove_worker(cur: TransactionCursor, worker_id: str) -> None:
    """Remove a worker and sever all its foreign-key references.

    Must be called inside an existing transaction. The four statements
    enforce the multi-table invariant: no dangling worker_id references
    remain in task_attempts, tasks, or dispatch_queue after the worker
    row is deleted.
    """
    cur.execute("UPDATE task_attempts SET worker_id = NULL WHERE worker_id = ?", (worker_id,))
    cur.execute("UPDATE tasks SET current_worker_id = NULL WHERE current_worker_id = ?", (worker_id,))
    cur.execute("DELETE FROM dispatch_queue WHERE worker_id = ?", (worker_id,))
    cur.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))


def _assign_task(
    cur: TransactionCursor,
    task_id: str,
    worker_id: str | None,
    worker_address: str | None,
    attempt_id: int,
    now_ms: int,
) -> None:
    """Create an attempt and mark a task as ASSIGNED in one consistent step.

    worker_id may be None for direct-provider tasks that have no backing
    worker daemon.
    """
    insert_task_attempt(cur, task_id, attempt_id, worker_id, job_pb2.TASK_STATE_ASSIGNED, now_ms)
    if worker_id is not None:
        cur.execute(
            "UPDATE tasks SET state = ?, current_attempt_id = ?, "
            "current_worker_id = ?, current_worker_address = ?, "
            "started_at_ms = COALESCE(started_at_ms, ?) WHERE task_id = ?",
            (job_pb2.TASK_STATE_ASSIGNED, attempt_id, worker_id, worker_address, now_ms, task_id),
        )
    else:
        cur.execute(
            "UPDATE tasks SET state = ?, current_attempt_id = ?, "
            "started_at_ms = COALESCE(started_at_ms, ?) WHERE task_id = ?",
            (job_pb2.TASK_STATE_ASSIGNED, attempt_id, now_ms, task_id),
        )


def _terminate_task(
    cur: TransactionCursor,
    registry,
    task_id: str,
    attempt_id: int | None,
    state: int,
    error: str | None,
    now_ms: int,
    *,
    attempt_state: int | None = None,
    worker_id: str | None = None,
    resources: "job_pb2.ResourceSpecProto | None" = None,
    failure_count: int | None = None,
    preemption_count: int | None = None,
) -> None:
    """Move a task (and its current attempt) out of active state consistently.

    Enforces the multi-table invariant: attempt is marked terminal,
    task state/error/finished_at are updated, endpoints are deleted,
    and worker resources are released.

    ``attempt_state`` overrides the state written to the attempt row when it
    differs from the task state (e.g. attempt=WORKER_FAILED while task retries
    to PENDING). Defaults to ``state`` when not provided.

    attempt_id < 0 means no attempt exists; the attempt UPDATE is skipped.
    """
    finished_at_ms = None if state in ACTIVE_TASK_STATES or state == job_pb2.TASK_STATE_PENDING else now_ms
    effective_attempt_state = attempt_state if attempt_state is not None else state

    if attempt_id is not None and attempt_id >= 0:
        cur.execute(
            "UPDATE task_attempts SET state = ?, "
            "finished_at_ms = COALESCE(finished_at_ms, ?), error = ? "
            "WHERE task_id = ? AND attempt_id = ?",
            (effective_attempt_state, now_ms, error, task_id, attempt_id),
        )

    # Build the UPDATE tasks statement dynamically based on optional counters.
    # Use COALESCE for finished_at_ms when non-NULL to preserve any existing
    # timestamp (defensive against double-termination). When NULL (retrying to
    # PENDING), assign directly so the column is cleared.
    if finished_at_ms is not None:
        set_clauses = ["state = ?", "error = ?", "finished_at_ms = COALESCE(finished_at_ms, ?)"]
    else:
        set_clauses = ["state = ?", "error = ?", "finished_at_ms = ?"]
    params: list[object] = [state, error, finished_at_ms]

    if failure_count is not None:
        set_clauses.append("failure_count = ?")
        params.append(failure_count)
    if preemption_count is not None:
        set_clauses.append("preemption_count = ?")
        params.append(preemption_count)

    # Always clear worker columns when leaving active state.
    if state not in ACTIVE_TASK_STATES:
        set_clauses.append("current_worker_id = NULL")
        set_clauses.append("current_worker_address = NULL")

    params.append(task_id)
    cur.execute(
        f"UPDATE tasks SET {', '.join(set_clauses)} WHERE task_id = ?",
        tuple(params),
    )

    delete_task_endpoints(cur, registry, task_id)

    if worker_id is not None and resources is not None:
        _decommit_worker_resources(cur, worker_id, resources)


def _kill_non_terminal_tasks(
    cur: Any,
    registry,
    job_id_wire: str,
    reason: str,
    now_ms: int,
) -> tuple[set[JobName], dict[JobName, WorkerId]]:
    """Kill all non-terminal tasks for a single job, decommit resources, and delete endpoints."""
    terminal_states = tuple(sorted(TERMINAL_TASK_STATES))
    placeholders = ",".join("?" * len(terminal_states))
    rows = cur.execute(
        "SELECT t.task_id, t.current_attempt_id, t.current_worker_id, "
        "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json "
        "FROM tasks t "
        "JOIN jobs j ON j.job_id = t.job_id "
        f"{JOB_CONFIG_JOIN} "
        f"WHERE t.job_id = ? AND t.state NOT IN ({placeholders})",
        (job_id_wire, *terminal_states),
    ).fetchall()
    tasks_to_kill: set[JobName] = set()
    task_kill_workers: dict[JobName, WorkerId] = {}
    for row in rows:
        task_id = str(row["task_id"])
        worker_id = row["current_worker_id"]
        task_name = JobName.from_wire(task_id)
        resources = None
        if worker_id is not None:
            resources = resource_spec_from_scalars(
                int(row["res_cpu_millicores"]),
                int(row["res_memory_bytes"]),
                int(row["res_disk_bytes"]),
                row["res_device_json"],
            )
            task_kill_workers[task_name] = WorkerId(str(worker_id))
        _terminate_task(
            cur,
            registry,
            task_id,
            int(row["current_attempt_id"]),
            job_pb2.TASK_STATE_KILLED,
            reason,
            now_ms,
            worker_id=str(worker_id) if worker_id is not None else None,
            resources=resources,
        )
        tasks_to_kill.add(task_name)
    return tasks_to_kill, task_kill_workers


def _cascade_children(
    cur: Any,
    registry,
    job_id: JobName,
    now_ms: int,
    reason: str,
    exclude_reservation_holders: bool = False,
) -> tuple[set[JobName], dict[JobName, WorkerId]]:
    """Kill descendant jobs (not the job itself) when a parent reaches terminal state or is preempted.

    When exclude_reservation_holders is True, reservation holder jobs and their
    descendants are left alive. This is used during preemption retry: the parent
    goes back to PENDING and needs its reservation to survive so the scheduler
    can re-satisfy it.
    """
    tasks_to_kill: set[JobName] = set()
    task_kill_workers: dict[JobName, WorkerId] = {}

    if exclude_reservation_holders:
        # Skip reservation holder jobs and anything below them.
        descendants = cur.execute(
            "WITH RECURSIVE subtree(job_id) AS ("
            "  SELECT job_id FROM jobs WHERE parent_job_id = ? AND is_reservation_holder = 0 "
            "  UNION ALL "
            "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
            "   WHERE j.is_reservation_holder = 0"
            ") SELECT job_id FROM subtree",
            (job_id.to_wire(),),
        ).fetchall()
    else:
        descendants = cur.execute(
            "WITH RECURSIVE subtree(job_id) AS ("
            "  SELECT job_id FROM jobs WHERE parent_job_id = ? "
            "  UNION ALL "
            "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
            ") SELECT job_id FROM subtree",
            (job_id.to_wire(),),
        ).fetchall()
    for child_row in descendants:
        child_job_id = str(child_row["job_id"])
        child_tasks_to_kill, child_task_kill_workers = _kill_non_terminal_tasks(
            cur, registry, child_job_id, reason, now_ms
        )
        tasks_to_kill.update(child_tasks_to_kill)
        task_kill_workers.update(child_task_kill_workers)
        terminal_placeholders = ",".join("?" for _ in TERMINAL_JOB_STATES)
        cur.execute(
            "UPDATE jobs SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
            f"WHERE job_id = ? AND state NOT IN ({terminal_placeholders})",
            (
                job_pb2.JOB_STATE_KILLED,
                reason,
                now_ms,
                child_job_id,
                *TERMINAL_JOB_STATES,
            ),
        )
    return tasks_to_kill, task_kill_workers


def _cascade_terminal_job(
    cur: Any,
    registry,
    job_id: JobName,
    now_ms: int,
    reason: str,
) -> tuple[set[JobName], dict[JobName, WorkerId]]:
    """Kill remaining tasks and descendant jobs when a job reaches a terminal state."""
    tasks_to_kill, task_kill_workers = _kill_non_terminal_tasks(cur, registry, job_id.to_wire(), reason, now_ms)
    child_tasks_to_kill, child_task_kill_workers = _cascade_children(cur, registry, job_id, now_ms, reason)
    tasks_to_kill.update(child_tasks_to_kill)
    task_kill_workers.update(child_task_kill_workers)
    return tasks_to_kill, task_kill_workers


@dataclass(frozen=True, slots=True)
class _CoscheduledSibling:
    task_id: str  # wire format
    attempt_id: int
    max_retries_preemption: int
    worker_id: str | None


def _find_coscheduled_siblings(
    cur: Any,
    job_id: JobName,
    exclude_task_id: JobName,
    has_coscheduling: bool,
) -> list[_CoscheduledSibling]:
    """Find active siblings in a coscheduled job (read-only)."""
    if not has_coscheduling:
        return []
    rows = cur.execute(
        "SELECT t.task_id, t.current_attempt_id, t.max_retries_preemption, "
        "t.current_worker_id AS worker_id "
        "FROM tasks t "
        "WHERE t.job_id = ? AND t.task_id != ? AND t.state IN (?, ?, ?)",
        (
            job_id.to_wire(),
            exclude_task_id.to_wire(),
            job_pb2.TASK_STATE_ASSIGNED,
            job_pb2.TASK_STATE_BUILDING,
            job_pb2.TASK_STATE_RUNNING,
        ),
    ).fetchall()
    return [
        _CoscheduledSibling(
            task_id=str(r["task_id"]),
            attempt_id=int(r["current_attempt_id"]),
            max_retries_preemption=int(r["max_retries_preemption"]),
            worker_id=str(r["worker_id"]) if r["worker_id"] is not None else None,
        )
        for r in rows
    ]


def _terminate_coscheduled_siblings(
    cur: Any,
    registry,
    siblings: Iterable[_CoscheduledSibling],
    failed_task_id: JobName,
    resources: "job_pb2.ResourceSpecProto",
    now_ms: int,
) -> tuple[set[JobName], dict[JobName, WorkerId]]:
    """Terminate coscheduled siblings and decommit their resources.

    Each sibling is marked WORKER_FAILED with exhausted preemption count so it
    will not be retried.
    """
    tasks_to_kill: set[JobName] = set()
    task_kill_workers: dict[JobName, WorkerId] = {}
    error = f"Coscheduled sibling {failed_task_id.to_wire()} failed"

    for sib in siblings:
        _terminate_task(
            cur,
            registry,
            sib.task_id,
            sib.attempt_id,
            job_pb2.TASK_STATE_WORKER_FAILED,
            error,
            now_ms,
            worker_id=sib.worker_id,
            resources=resources if sib.worker_id is not None else None,
            preemption_count=sib.max_retries_preemption + 1,
        )
        if sib.worker_id is not None:
            task_kill_workers[JobName.from_wire(sib.task_id)] = WorkerId(sib.worker_id)
        tasks_to_kill.add(JobName.from_wire(sib.task_id))

    return tasks_to_kill, task_kill_workers


def _resolve_preemption_policy(cur: Any, job_id: JobName) -> int:
    """Resolve the effective preemption policy for a job.

    Defaults: single-task jobs → TERMINATE_CHILDREN, multi-task → PRESERVE_CHILDREN.
    """
    row = cur.execute(
        f"SELECT jc.preemption_policy, j.num_tasks FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
        (job_id.to_wire(),),
    ).fetchone()
    if row is None:
        return job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    policy = int(row["preemption_policy"])
    if policy != job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED:
        return policy
    if int(row["num_tasks"]) <= 1:
        return job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    return job_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN


_TERMINAL_STATE_REASONS: dict[int, str] = {
    job_pb2.JOB_STATE_FAILED: "Job exceeded max_task_failures",
    job_pb2.JOB_STATE_KILLED: "Job was terminated.",
    job_pb2.JOB_STATE_UNSCHEDULABLE: "Job could not be scheduled.",
    job_pb2.JOB_STATE_WORKER_FAILED: "Worker failed",
}


def _finalize_terminal_job(
    cur: Any,
    registry,
    job_id: JobName,
    terminal_state: int,
    now_ms: int,
) -> tuple[set[JobName], dict[JobName, WorkerId]]:
    """Kill remaining tasks and optionally cascade to children when a job goes terminal.

    Called after _recompute_job_state determines a job has reached a terminal
    state. Kills the job's own non-terminal tasks and, depending on preemption
    policy, cascades to descendant jobs.

    Succeeded jobs always cascade (children are no longer needed).
    Non-succeeded jobs cascade only if the preemption policy is TERMINATE_CHILDREN.
    """
    reason = _TERMINAL_STATE_REASONS.get(terminal_state, "Job finalized")
    tasks_to_kill, task_kill_workers = _kill_non_terminal_tasks(cur, registry, job_id.to_wire(), reason, now_ms)
    should_cascade = True
    if terminal_state != job_pb2.JOB_STATE_SUCCEEDED:
        policy = _resolve_preemption_policy(cur, job_id)
        should_cascade = policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    if should_cascade:
        child_tasks_to_kill, child_task_kill_workers = _cascade_children(cur, registry, job_id, now_ms, reason)
        tasks_to_kill.update(child_tasks_to_kill)
        task_kill_workers.update(child_task_kill_workers)
    return tasks_to_kill, task_kill_workers


def _resolve_task_failure_state(
    prior_state: int,
    preemption_count: int,
    max_preemptions: int,
    terminal_state: int,
) -> tuple[int, int]:
    """Determine new task state after a worker failure or preemption.

    Assigned tasks always retry. Executing tasks retry if preemption budget remains,
    otherwise go to the given terminal state.

    Returns (new_task_state, updated_preemption_count).
    """
    if prior_state == job_pb2.TASK_STATE_ASSIGNED:
        return job_pb2.TASK_STATE_PENDING, preemption_count
    if prior_state in EXECUTING_TASK_STATES:
        preemption_count += 1
        if preemption_count <= max_preemptions:
            return job_pb2.TASK_STATE_PENDING, preemption_count
    return terminal_state, preemption_count


# =============================================================================
# Batch helpers for apply_heartbeats_batch
# =============================================================================


def _batch_worker_health(
    cur: TransactionCursor,
    requests: list["HeartbeatApplyRequest"],
    now_ms: int,
) -> set[str]:
    """Batch-update worker health, resource snapshots, and history.

    Returns the set of worker IDs that actually exist in the DB so callers
    can skip updates from stale/removed workers.
    """
    worker_ids = [str(req.worker_id) for req in requests]
    if not worker_ids:
        return set()

    placeholders = ",".join("?" * len(worker_ids))
    rows = cur.execute(
        f"SELECT worker_id FROM workers WHERE worker_id IN ({placeholders})",
        tuple(worker_ids),
    ).fetchall()
    existing = {str(r["worker_id"]) for r in rows}

    health_params_no_snap = []
    health_params_with_snap = []
    history_params = []
    for req in requests:
        wid = str(req.worker_id)
        if wid not in existing:
            continue
        snap = req.worker_resource_snapshot
        if snap is not None:
            snap_fields = (
                snap.host_cpu_percent,
                snap.memory_used_bytes,
                snap.memory_total_bytes,
                snap.disk_used_bytes,
                snap.disk_total_bytes,
                snap.running_task_count,
                snap.total_process_count,
                snap.net_recv_bps,
                snap.net_sent_bps,
            )
            health_params_with_snap.append((now_ms, *snap_fields, wid))
            history_params.append((wid, *snap_fields, now_ms))
        else:
            health_params_no_snap.append((now_ms, wid))

    if health_params_no_snap:
        cur.executemany(
            "UPDATE workers SET healthy = 1, active = 1, consecutive_failures = 0, "
            "last_heartbeat_ms = ? WHERE worker_id = ?",
            health_params_no_snap,
        )
    if health_params_with_snap:
        cur.executemany(
            "UPDATE workers SET healthy = 1, active = 1, consecutive_failures = 0, "
            "last_heartbeat_ms = ?, "
            "snapshot_host_cpu_percent = ?, snapshot_memory_used_bytes = ?, "
            "snapshot_memory_total_bytes = ?, snapshot_disk_used_bytes = ?, "
            "snapshot_disk_total_bytes = ?, snapshot_running_task_count = ?, "
            "snapshot_total_process_count = ?, snapshot_net_recv_bps = ?, "
            "snapshot_net_sent_bps = ? WHERE worker_id = ?",
            health_params_with_snap,
        )
    if history_params:
        cur.executemany(
            "INSERT INTO worker_resource_history("
            "worker_id, snapshot_host_cpu_percent, snapshot_memory_used_bytes, "
            "snapshot_memory_total_bytes, snapshot_disk_used_bytes, snapshot_disk_total_bytes, "
            "snapshot_running_task_count, snapshot_total_process_count, "
            "snapshot_net_recv_bps, snapshot_net_sent_bps, timestamp_ms"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            history_params,
        )
    return existing


def _bulk_fetch_tasks(cur: TransactionCursor, task_ids: list[str]) -> dict[str, Any]:
    """Fetch task rows for all given IDs in chunked IN queries."""
    result: dict[str, Any] = {}
    for chunk_start in range(0, len(task_ids), 900):
        chunk = task_ids[chunk_start : chunk_start + 900]
        ph = ",".join("?" * len(chunk))
        rows = cur.execute(
            f"SELECT * FROM tasks WHERE task_id IN ({ph})",
            tuple(chunk),
        ).fetchall()
        for r in rows:
            result[str(r["task_id"])] = r
    return result


# =============================================================================
# Controller Transitions
# =============================================================================


class ControllerTransitions:
    """State machine for controller entities.

    All methods that mutate DB state live here. Each is a single atomic
    transaction. Read-only queries do NOT belong here — callers use
    db.read_snapshot() directly.

    SQLite is the sole source of truth. Any in-memory values are transient
    helpers and must never be required for correctness across restarts.
    """

    def __init__(
        self,
        db: ControllerDB,
        heartbeat_failure_threshold: int = HEARTBEAT_FAILURE_THRESHOLD,
        user_budget_defaults: UserBudgetDefaults | None = None,
    ):
        self._db = db
        self._heartbeat_failure_threshold = heartbeat_failure_threshold
        self._user_budget_defaults = user_budget_defaults or UserBudgetDefaults()

    def _record_transaction(
        self,
        cur: Any,
        kind: str,
        actions: list[tuple[str, str, dict[str, object]]],
        *,
        payload: dict[str, object] | None = None,
    ) -> None:
        created_ms = Timestamp.now().epoch_ms()
        cur.execute(
            "INSERT INTO txn_log(kind, payload_json, created_at_ms) VALUES (?, ?, ?)",
            (kind, json.dumps(payload or {}), created_ms),
        )
        txn_id = int(cur.lastrowid)
        for action, entity_id, details in actions:
            cur.execute(
                "INSERT INTO txn_actions(txn_id, action, entity_id, details_json, created_at_ms) VALUES (?, ?, ?, ?, ?)",
                (txn_id, action, entity_id, json.dumps(details), created_ms),
            )

    def _recompute_job_state(self, cur: Any, job_id: JobName) -> int | None:
        row = cur.execute(
            f"SELECT j.state, j.started_at_ms, jc.max_task_failures "
            f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
            (job_id.to_wire(),),
        ).fetchone()
        if row is None:
            return None
        current_state = int(row["state"])
        if current_state in TERMINAL_JOB_STATES:
            return current_state
        max_task_failures = int(row["max_task_failures"])
        counts_rows = cur.execute(
            "SELECT state, COUNT(*) AS c FROM tasks WHERE job_id = ? GROUP BY state",
            (job_id.to_wire(),),
        ).fetchall()
        counts = {int(r["state"]): int(r["c"]) for r in counts_rows}
        total = sum(counts.values())
        new_state = current_state
        now_ms = Timestamp.now().epoch_ms()
        if total > 0 and counts.get(job_pb2.TASK_STATE_SUCCEEDED, 0) == total:
            new_state = job_pb2.JOB_STATE_SUCCEEDED
        elif counts.get(job_pb2.TASK_STATE_FAILED, 0) > max_task_failures:
            new_state = job_pb2.JOB_STATE_FAILED
        elif counts.get(job_pb2.TASK_STATE_UNSCHEDULABLE, 0) > 0:
            new_state = job_pb2.JOB_STATE_UNSCHEDULABLE
        elif counts.get(job_pb2.TASK_STATE_KILLED, 0) > 0:
            new_state = job_pb2.JOB_STATE_KILLED
        elif (
            total > 0
            and (counts.get(job_pb2.TASK_STATE_WORKER_FAILED, 0) + counts.get(job_pb2.TASK_STATE_PREEMPTED, 0)) > 0
            and all(s in TERMINAL_TASK_STATES for s in counts)
        ):
            new_state = job_pb2.JOB_STATE_WORKER_FAILED
        elif (
            counts.get(job_pb2.TASK_STATE_ASSIGNED, 0) > 0
            or counts.get(job_pb2.TASK_STATE_BUILDING, 0) > 0
            or counts.get(job_pb2.TASK_STATE_RUNNING, 0) > 0
        ):
            new_state = job_pb2.JOB_STATE_RUNNING
        elif row["started_at_ms"] is not None:
            # Retries put tasks back into PENDING; keep job running once it has started.
            new_state = job_pb2.JOB_STATE_RUNNING
        elif total > 0:
            new_state = job_pb2.JOB_STATE_PENDING
        if new_state == current_state:
            return new_state
        terminal_placeholders = ",".join("?" for _ in TERMINAL_JOB_STATES)
        error_row = cur.execute(
            "SELECT error FROM tasks WHERE job_id = ? AND error IS NOT NULL ORDER BY task_index LIMIT 1",
            (job_id.to_wire(),),
        ).fetchone()
        error = str(error_row["error"]) if error_row is not None else None
        cur.execute(
            "UPDATE jobs SET state = ?, "
            "started_at_ms = CASE WHEN ? = ? THEN COALESCE(started_at_ms, ?) ELSE started_at_ms END, "
            f"finished_at_ms = CASE WHEN ? IN ({terminal_placeholders}) THEN ? ELSE finished_at_ms END, "
            "error = CASE WHEN ? IN (?, ?, ?, ?) THEN ? ELSE error END "
            "WHERE job_id = ?",
            (
                new_state,
                new_state,
                job_pb2.JOB_STATE_RUNNING,
                now_ms,
                new_state,
                *TERMINAL_JOB_STATES,
                now_ms,
                new_state,
                job_pb2.JOB_STATE_FAILED,
                job_pb2.JOB_STATE_KILLED,
                job_pb2.JOB_STATE_UNSCHEDULABLE,
                job_pb2.JOB_STATE_WORKER_FAILED,
                error,
                job_id.to_wire(),
            ),
        )
        return new_state

    def replace_reservation_claims(self, claims: dict[WorkerId, ReservationClaim]) -> None:
        """Replace all reservation claims atomically."""
        with self._db.transaction() as cur:
            cur.execute("DELETE FROM reservation_claims")
            for worker_id, claim in claims.items():
                cur.execute(
                    "INSERT INTO reservation_claims(worker_id, job_id, entry_idx) VALUES (?, ?, ?)",
                    (str(worker_id), claim.job_id, claim.entry_idx),
                )

    # =========================================================================
    # Command API
    # =========================================================================

    def submit_job(
        self,
        job_id: JobName,
        request: controller_pb2.Controller.LaunchJobRequest,
        ts: Timestamp,
    ) -> SubmitJobResult:
        """Submit a job and expand its tasks in one DB transaction."""
        submitted_ms = ts.epoch_ms()
        actions: list[tuple[str, str, dict[str, object]]] = []
        created_task_ids: list[JobName] = []

        with self._db.transaction() as cur:
            row = cur.execute("SELECT value FROM meta WHERE key = 'last_submission_ms'").fetchone()
            last_submission_ms = int(row["value"]) if row is not None else 0
            effective_submission_ms = max(submitted_ms, last_submission_ms + 1)
            if row is None:
                cur.execute("INSERT INTO meta(key, value) VALUES ('last_submission_ms', ?)", (effective_submission_ms,))
            else:
                cur.execute("UPDATE meta SET value = ? WHERE key = 'last_submission_ms'", (effective_submission_ms,))

            parent_job_id = job_id.parent.to_wire() if job_id.parent is not None else None
            if parent_job_id is not None:
                parent_exists = cur.execute("SELECT 1 FROM jobs WHERE job_id = ?", (parent_job_id,)).fetchone()
                if parent_exists is None:
                    parent_job_id = None
            root_submitted_ms = effective_submission_ms
            if parent_job_id is not None:
                parent = cur.execute(
                    "SELECT root_submitted_at_ms FROM jobs WHERE job_id = ?",
                    (parent_job_id,),
                ).fetchone()
                if parent is not None:
                    root_submitted_ms = int(parent["root_submitted_at_ms"])

            deadline_epoch_ms: int | None = None
            if request.HasField("scheduling_timeout") and request.scheduling_timeout.milliseconds > 0:
                deadline_epoch_ms = (
                    Timestamp.from_ms(effective_submission_ms)
                    .add(duration_from_proto(request.scheduling_timeout))
                    .epoch_ms()
                )

            cur.execute(
                "INSERT OR IGNORE INTO users(user_id, created_at_ms) VALUES (?, ?)",
                (job_id.user, effective_submission_ms),
            )
            # Create default user budget row alongside user creation.
            budget_defaults = self._user_budget_defaults
            cur.execute(
                "INSERT OR IGNORE INTO user_budgets(user_id, budget_limit, max_band, updated_at_ms) "
                "VALUES (?, ?, ?, ?)",
                (
                    job_id.user,
                    budget_defaults.budget_limit,
                    budget_defaults.max_band,
                    effective_submission_ms,
                ),
            )

            # Resolve priority band: use explicit request value, inherit from parent, or default to INTERACTIVE.
            requested_band = int(request.priority_band)
            if requested_band != job_pb2.PRIORITY_BAND_UNSPECIFIED:
                band_sort_key = requested_band
            elif parent_job_id is not None:
                parent_band_row = cur.execute(
                    "SELECT priority_band FROM tasks WHERE job_id = ? LIMIT 1",
                    (parent_job_id,),
                ).fetchone()
                if parent_band_row is not None:
                    band_sort_key = parent_band_row["priority_band"]
                else:
                    band_sort_key = job_pb2.PRIORITY_BAND_INTERACTIVE
            else:
                band_sort_key = job_pb2.PRIORITY_BAND_INTERACTIVE

            replicas = int(request.replicas)
            validation_error: str | None = None
            if replicas < 1:
                validation_error = f"Job {job_id} has invalid replicas={replicas}; must be >= 1"
                replicas = 0
            elif replicas > MAX_REPLICAS_PER_JOB:
                validation_error = f"Job {job_id} replicas={replicas} exceeds max {MAX_REPLICAS_PER_JOB}"
                replicas = 0

            state = job_pb2.JOB_STATE_PENDING if validation_error is None else job_pb2.JOB_STATE_FAILED
            finished_ms = None if validation_error is None else effective_submission_ms
            has_reservation = _has_reservation_flag(request)

            # Scheduling fields extracted from request proto.
            res = request.resources if request.HasField("resources") else None
            res_cpu = int(res.cpu_millicores) if res else 0
            res_mem = int(res.memory_bytes) if res else 0
            res_disk = int(res.disk_bytes) if res else 0
            res_device = proto_to_json(res.device) if res else None
            constraints_json = constraints_to_json(request.constraints)
            has_cosched = 1 if request.HasField("coscheduling") else 0
            cosched_group = request.coscheduling.group_by if has_cosched else ""
            sched_timeout: int | None = (
                int(request.scheduling_timeout.milliseconds)
                if request.HasField("scheduling_timeout") and request.scheduling_timeout.milliseconds > 0
                else None
            )
            max_failures = int(request.max_task_failures)
            # Serialize dispatch config fields for job_config.
            entrypoint_json = entrypoint_to_json(request.entrypoint)
            environment_json = proto_to_json(request.environment)
            ports_json = json.dumps(list(request.ports)) if request.ports else "[]"
            reservation_json = reservation_to_json(request)
            timeout_ms: int | None = int(request.timeout.milliseconds) if request.timeout.milliseconds > 0 else None

            job_name_lower = request.name.lower()
            cur.execute(
                "INSERT INTO jobs("
                "job_id, user_id, parent_job_id, root_job_id, depth, state, submitted_at_ms, "
                "root_submitted_at_ms, started_at_ms, finished_at_ms, scheduling_deadline_epoch_ms, "
                "error, exit_code, num_tasks, is_reservation_holder, name, has_reservation"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?, 0, ?, ?)",
                (
                    job_id.to_wire(),
                    job_id.user,
                    parent_job_id,
                    job_id.root_job.to_wire(),
                    job_id.depth,
                    state,
                    effective_submission_ms,
                    root_submitted_ms,
                    finished_ms,
                    deadline_epoch_ms,
                    validation_error,
                    replicas,
                    job_name_lower,
                    has_reservation,
                ),
            )
            cur.execute(
                "INSERT INTO job_config("
                "job_id, name, has_reservation, "
                "res_cpu_millicores, res_memory_bytes, res_disk_bytes, res_device_json, "
                "constraints_json, has_coscheduling, coscheduling_group_by, "
                "scheduling_timeout_ms, max_task_failures, "
                "entrypoint_json, environment_json, bundle_id, ports_json, "
                "max_retries_failure, max_retries_preemption, timeout_ms, "
                "preemption_policy, existing_job_policy, priority_band, "
                "task_image, submit_argv_json, reservation_json, fail_if_exists"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id.to_wire(),
                    job_name_lower,
                    has_reservation,
                    res_cpu,
                    res_mem,
                    res_disk,
                    res_device,
                    constraints_json,
                    has_cosched,
                    cosched_group,
                    sched_timeout,
                    max_failures,
                    entrypoint_json,
                    environment_json,
                    request.bundle_id,
                    ports_json,
                    int(request.max_retries_failure),
                    int(request.max_retries_preemption),
                    timeout_ms,
                    int(request.preemption_policy),
                    int(request.existing_job_policy),
                    int(request.priority_band),
                    request.task_image,
                    json.dumps(list(request.submit_argv)),
                    reservation_json,
                    1 if request.fail_if_exists else 0,
                ),
            )

            # Store workdir files in separate table.
            if request.entrypoint.workdir_files:
                for filename, data in request.entrypoint.workdir_files.items():
                    cur.execute(
                        "INSERT INTO job_workdir_files(job_id, filename, data) VALUES (?, ?, ?)",
                        (job_id.to_wire(), filename, data),
                    )

            if validation_error is None:
                insertion_base = self._db.next_sequence("task_priority_insertion", cur=cur)
                for idx in range(replicas):
                    task_id = job_id.task(idx).to_wire()
                    created_task_ids.append(JobName.from_wire(task_id))
                    cur.execute(
                        "INSERT INTO tasks("
                        "task_id, job_id, task_index, state, error, exit_code, submitted_at_ms, started_at_ms, "
                        "finished_at_ms, max_retries_failure, max_retries_preemption, failure_count, preemption_count, "
                        "current_attempt_id, priority_neg_depth, priority_root_submitted_ms, "
                        "priority_insertion, priority_band"
                        ") VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?, 0, 0, -1, ?, ?, ?, ?)",
                        (
                            task_id,
                            job_id.to_wire(),
                            idx,
                            job_pb2.TASK_STATE_PENDING,
                            effective_submission_ms,
                            int(request.max_retries_failure),
                            int(request.max_retries_preemption),
                            -job_id.depth,
                            root_submitted_ms,
                            insertion_base + idx,
                            band_sort_key,
                        ),
                    )
                if request.HasField("reservation") and request.reservation.entries:
                    holder_id = job_id.child(RESERVATION_HOLDER_JOB_NAME)
                    entry = request.reservation.entries[0]
                    holder_request = controller_pb2.Controller.LaunchJobRequest(
                        name=holder_id.to_wire(),
                        entrypoint=request.entrypoint,
                        resources=entry.resources,
                        environment=request.environment,
                        replicas=len(request.reservation.entries),
                        max_retries_preemption=DEFAULT_MAX_RETRIES_PREEMPTION,
                    )
                    merged = merge_constraints(
                        constraints_from_resources(entry.resources),
                        [Constraint.from_proto(c) for c in entry.constraints or request.constraints],
                    )
                    for constraint in merged:
                        holder_request.constraints.append(constraint.to_proto())
                    holder_res = holder_request.resources if holder_request.HasField("resources") else None
                    holder_res_cpu = int(holder_res.cpu_millicores) if holder_res else 0
                    holder_res_mem = int(holder_res.memory_bytes) if holder_res else 0
                    holder_res_disk = int(holder_res.disk_bytes) if holder_res else 0
                    holder_res_device = proto_to_json(holder_res.device) if holder_res else None
                    holder_constraints_json = constraints_to_json(holder_request.constraints)
                    holder_name_lower = holder_request.name.lower()
                    cur.execute(
                        "INSERT INTO jobs("
                        "job_id, user_id, parent_job_id, root_job_id, depth, state, submitted_at_ms, "
                        "root_submitted_at_ms, started_at_ms, finished_at_ms, scheduling_deadline_epoch_ms, "
                        "error, exit_code, num_tasks, is_reservation_holder, name, has_reservation"
                        ") VALUES ("
                        "?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, 1, ?, 0"
                        ")",
                        (
                            holder_id.to_wire(),
                            holder_id.user,
                            job_id.to_wire(),
                            holder_id.root_job.to_wire(),
                            holder_id.depth,
                            job_pb2.JOB_STATE_PENDING,
                            effective_submission_ms,
                            root_submitted_ms,
                            len(request.reservation.entries),
                            holder_name_lower,
                        ),
                    )
                    holder_entrypoint_json = entrypoint_to_json(holder_request.entrypoint)
                    holder_environment_json = proto_to_json(holder_request.environment)
                    cur.execute(
                        "INSERT INTO job_config("
                        "job_id, name, has_reservation, "
                        "res_cpu_millicores, res_memory_bytes, res_disk_bytes, res_device_json, "
                        "constraints_json, has_coscheduling, coscheduling_group_by, "
                        "scheduling_timeout_ms, max_task_failures, "
                        "entrypoint_json, environment_json, bundle_id, ports_json, "
                        "max_retries_failure, max_retries_preemption, timeout_ms, "
                        "preemption_policy, existing_job_policy, priority_band, "
                        "task_image, reservation_json"
                        ") VALUES ("
                        "?, ?, 0, ?, ?, ?, ?, ?, 0, '', NULL, 0, "
                        "?, ?, '', '[]', 0, ?, NULL, 0, 0, 0, '', NULL"
                        ")",
                        (
                            holder_id.to_wire(),
                            holder_name_lower,
                            holder_res_cpu,
                            holder_res_mem,
                            holder_res_disk,
                            holder_res_device,
                            holder_constraints_json,
                            holder_entrypoint_json,
                            holder_environment_json,
                            DEFAULT_MAX_RETRIES_PREEMPTION,
                        ),
                    )
                    holder_base = self._db.next_sequence("task_priority_insertion", cur=cur)
                    for idx in range(len(request.reservation.entries)):
                        created_task_ids.append(holder_id.task(idx))
                        cur.execute(
                            "INSERT INTO tasks("
                            "task_id, job_id, task_index, state, error, exit_code, submitted_at_ms, started_at_ms, "
                            "finished_at_ms, max_retries_failure, max_retries_preemption, "
                            "failure_count, preemption_count, "
                            "current_attempt_id, priority_neg_depth, priority_root_submitted_ms, "
                            "priority_insertion, priority_band"
                            ") VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?, 0, 0, -1, ?, ?, ?, ?)",
                            (
                                holder_id.task(idx).to_wire(),
                                holder_id.to_wire(),
                                idx,
                                job_pb2.TASK_STATE_PENDING,
                                effective_submission_ms,
                                0,
                                DEFAULT_MAX_RETRIES_PREEMPTION,
                                -holder_id.depth,
                                root_submitted_ms,
                                holder_base + idx,
                                band_sort_key,
                            ),
                        )

            actions.append(("job_submitted", job_id.to_wire(), {"num_tasks": replicas, "error": validation_error}))
            self._record_transaction(cur, "submit_job", actions)
        return SubmitJobResult(job_id=job_id, task_ids=created_task_ids)

    def cancel_job(self, job_id: JobName, reason: str) -> TxResult:
        """Cancel a job tree and return tasks that need kill RPCs."""
        with self._db.transaction() as cur:
            subtree = cur.execute(
                "WITH RECURSIVE subtree(job_id) AS ("
                "  SELECT job_id FROM jobs WHERE job_id = ? "
                "  UNION ALL "
                "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
                ") SELECT job_id FROM subtree",
                (job_id.to_wire(),),
            ).fetchall()
            if not subtree:
                return TxResult()
            subtree_ids = [str(row["job_id"]) for row in subtree]
            placeholders = ",".join("?" for _ in subtree_ids)
            running_rows = cur.execute(
                f"SELECT t.task_id, t.current_worker_id AS worker_id, "
                f"j.is_reservation_holder, "
                f"jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json "
                f"FROM tasks t "
                f"JOIN jobs j ON j.job_id = t.job_id "
                f"{JOB_CONFIG_JOIN} "
                f"WHERE t.job_id IN ({placeholders}) "
                "AND t.state IN (?, ?, ?)",
                (
                    *subtree_ids,
                    job_pb2.TASK_STATE_ASSIGNED,
                    job_pb2.TASK_STATE_BUILDING,
                    job_pb2.TASK_STATE_RUNNING,
                ),
            ).fetchall()
            tasks_to_kill = {JobName.from_wire(str(row["task_id"])) for row in running_rows}
            task_kill_workers = {
                JobName.from_wire(str(row["task_id"])): WorkerId(str(row["worker_id"]))
                for row in running_rows
                if row["worker_id"] is not None
            }
            # Decommit resources for each active task on its assigned worker.
            # cancel_job marks tasks as KILLED, but apply_heartbeat skips
            # already-finished tasks (is_finished() check), so the normal
            # heartbeat decommit path never fires for cancelled tasks.
            # Direct-provider tasks have NULL worker_id — skip decommit for them.
            for row in running_rows:
                if row["worker_id"] is not None and not int(row["is_reservation_holder"]):
                    resources = resource_spec_from_scalars(
                        int(row["res_cpu_millicores"]),
                        int(row["res_memory_bytes"]),
                        int(row["res_disk_bytes"]),
                        row["res_device_json"],
                    )
                    _decommit_worker_resources(cur, str(row["worker_id"]), resources)
            now_ms = Timestamp.now().epoch_ms()
            task_terminal_placeholders = ",".join("?" for _ in TERMINAL_TASK_STATES)
            cur.execute(
                f"UPDATE tasks SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?), "
                f"current_worker_id = NULL, current_worker_address = NULL "
                f"WHERE job_id IN ({placeholders}) AND state NOT IN ({task_terminal_placeholders})",
                (
                    job_pb2.TASK_STATE_KILLED,
                    reason,
                    now_ms,
                    *subtree_ids,
                    *TERMINAL_TASK_STATES,
                ),
            )
            # Deliberately excludes JOB_STATE_WORKER_FAILED from the guard set:
            # worker-failed jobs should still be cancellable (transitioned to KILLED).
            cancel_guard_states = TERMINAL_JOB_STATES - {job_pb2.JOB_STATE_WORKER_FAILED}
            cancel_guard_placeholders = ",".join("?" for _ in cancel_guard_states)
            cur.execute(
                f"UPDATE jobs SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
                f"WHERE job_id IN ({placeholders}) AND state NOT IN ({cancel_guard_placeholders})",
                (
                    job_pb2.JOB_STATE_KILLED,
                    reason,
                    now_ms,
                    *subtree_ids,
                    *cancel_guard_states,
                ),
            )
            self._db.endpoints.remove_by_job_ids(cur, [JobName.from_wire(jid) for jid in subtree_ids])
            self._record_transaction(cur, "cancel_job", [("job_cancelled", job_id.to_wire(), {"reason": reason})])
            return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def register_or_refresh_worker(
        self,
        worker_id: WorkerId,
        address: str,
        metadata: job_pb2.WorkerMetadata,
        ts: Timestamp,
        slice_id: str = "",
        scale_group: str = "",
    ) -> TxResult:
        """Register a new worker or refresh an existing one."""
        attrs: list[tuple[str, str, str | None, int | None, float | None]] = []
        for key, proto in metadata.attributes.items():
            value = AttributeValue.from_proto(proto).value
            if isinstance(value, int):
                attrs.append((key, "int", None, int(value), None))
            elif isinstance(value, float):
                attrs.append((key, "float", None, None, float(value)))
            else:
                attrs.append((key, "str", str(value), None, None))
        now_ms = ts.epoch_ms()
        gpu_count = get_gpu_count(metadata.device)
        tpu_count = get_tpu_count(metadata.device)
        if metadata.device.HasField("gpu"):
            device_type = "gpu"
            device_variant = metadata.device.gpu.variant
        elif metadata.device.HasField("tpu"):
            device_type = "tpu"
            device_variant = metadata.device.tpu.variant
        else:
            device_type = ""
            device_variant = ""
        with self._db.transaction() as cur:
            md_device_json = proto_to_json(metadata.device)
            cur.execute(
                "INSERT INTO workers("
                "worker_id, address, healthy, active, consecutive_failures, last_heartbeat_ms, "
                "committed_cpu_millicores, committed_mem_bytes, committed_gpu, committed_tpu, "
                "total_cpu_millicores, total_memory_bytes, total_gpu_count, total_tpu_count, "
                "device_type, device_variant, slice_id, scale_group, "
                "md_hostname, md_ip_address, md_cpu_count, md_memory_bytes, md_disk_bytes, "
                "md_tpu_name, md_tpu_worker_hostnames, md_tpu_worker_id, md_tpu_chips_per_host_bounds, "
                "md_gpu_count, md_gpu_name, md_gpu_memory_mb, "
                "md_gce_instance_name, md_gce_zone, md_git_hash, md_device_json"
                ") VALUES (?, ?, 1, 1, 0, ?, 0, 0, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(worker_id) DO UPDATE SET "
                "address=excluded.address, healthy=1, active=1, "
                "consecutive_failures=0, last_heartbeat_ms=excluded.last_heartbeat_ms, "
                "total_cpu_millicores=excluded.total_cpu_millicores, total_memory_bytes=excluded.total_memory_bytes, "
                "total_gpu_count=excluded.total_gpu_count, total_tpu_count=excluded.total_tpu_count, "
                "device_type=excluded.device_type, device_variant=excluded.device_variant, "
                "slice_id=excluded.slice_id, scale_group=excluded.scale_group, "
                "md_hostname=excluded.md_hostname, md_ip_address=excluded.md_ip_address, "
                "md_cpu_count=excluded.md_cpu_count, md_memory_bytes=excluded.md_memory_bytes, "
                "md_disk_bytes=excluded.md_disk_bytes, md_tpu_name=excluded.md_tpu_name, "
                "md_tpu_worker_hostnames=excluded.md_tpu_worker_hostnames, "
                "md_tpu_worker_id=excluded.md_tpu_worker_id, "
                "md_tpu_chips_per_host_bounds=excluded.md_tpu_chips_per_host_bounds, "
                "md_gpu_count=excluded.md_gpu_count, md_gpu_name=excluded.md_gpu_name, "
                "md_gpu_memory_mb=excluded.md_gpu_memory_mb, "
                "md_gce_instance_name=excluded.md_gce_instance_name, md_gce_zone=excluded.md_gce_zone, "
                "md_git_hash=excluded.md_git_hash, md_device_json=excluded.md_device_json",
                (
                    str(worker_id),
                    address,
                    now_ms,
                    metadata.cpu_count * 1000,
                    metadata.memory_bytes,
                    gpu_count,
                    tpu_count,
                    device_type,
                    device_variant,
                    slice_id,
                    scale_group,
                    metadata.hostname,
                    metadata.ip_address,
                    metadata.cpu_count,
                    metadata.memory_bytes,
                    metadata.disk_bytes,
                    metadata.tpu_name,
                    metadata.tpu_worker_hostnames,
                    metadata.tpu_worker_id,
                    metadata.tpu_chips_per_host_bounds,
                    metadata.gpu_count,
                    metadata.gpu_name,
                    metadata.gpu_memory_mb,
                    metadata.gce_instance_name,
                    metadata.gce_zone,
                    metadata.git_hash,
                    md_device_json,
                ),
            )
            cur.execute("DELETE FROM worker_attributes WHERE worker_id = ?", (str(worker_id),))
            for key, value_type, str_value, int_value, float_value in attrs:
                cur.execute(
                    "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(worker_id), key, value_type, str_value, int_value, float_value),
                )
            self._record_transaction(
                cur, "register_worker", [("worker_registered", str(worker_id), {"address": address})]
            )
        # Update in-memory attribute cache so scheduling sees the new worker immediately.
        attr_dict: dict[str, AttributeValue] = {}
        for key, value_type, str_value, int_value, float_value in attrs:
            if value_type == "int":
                attr_dict[key] = AttributeValue(int(int_value))
            elif value_type == "float":
                attr_dict[key] = AttributeValue(float(float_value))
            else:
                attr_dict[key] = AttributeValue(str(str_value or ""))
        self._db.set_worker_attributes(worker_id, attr_dict)
        return TxResult()

    def register_worker(
        self,
        worker_id: WorkerId,
        address: str,
        metadata: job_pb2.WorkerMetadata,
        ts: Timestamp,
        slice_id: str = "",
        scale_group: str = "",
    ) -> WorkerRegistrationResult:
        self.register_or_refresh_worker(
            worker_id=worker_id,
            address=address,
            metadata=metadata,
            ts=ts,
            slice_id=slice_id,
            scale_group=scale_group,
        )
        return WorkerRegistrationResult(worker_id=worker_id)

    def queue_assignments(self, assignments: list[Assignment], *, direct_dispatch: bool = False) -> AssignmentResult:
        """Commit assignments and enqueue dispatches in one transaction.

        When direct_dispatch=True, collects (worker_id, address, RunTaskRequest)
        tuples in start_requests instead of writing to the dispatch_queue table.
        The caller is responsible for sending StartTasks RPCs.
        """
        accepted: list[Assignment] = []
        rejected: list[Assignment] = []
        start_requests: list[tuple[WorkerId, str, job_pb2.RunTaskRequest]] = []
        has_real_dispatch = False
        with self._db.transaction() as cur:
            now_ms = Timestamp.now().epoch_ms()
            job_cache: dict[str, JobDetailRow] = {}
            jobs_to_update: set[str] = set()
            for assignment in assignments:
                task_row = cur.execute(
                    f"SELECT {TASK_DETAIL_PROJECTION.select_clause()} " "FROM tasks t WHERE t.task_id = ?",
                    (assignment.task_id.to_wire(),),
                ).fetchone()
                worker_row = cur.execute(
                    "SELECT worker_id, address, active, healthy "
                    "FROM workers WHERE worker_id = ? AND active = 1 AND healthy = 1",
                    (str(assignment.worker_id),),
                ).fetchone()
                if task_row is None or worker_row is None:
                    rejected.append(assignment)
                    continue
                task = TASK_DETAIL_PROJECTION.decode_one([task_row])
                if not task_row_can_be_scheduled(task):
                    rejected.append(assignment)
                    continue
                job_id_wire = task.job_id.to_wire()
                if job_id_wire not in job_cache:
                    job_row = cur.execute(
                        f"SELECT {JOB_DETAIL_PROJECTION.select_clause()} "
                        f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
                        (job_id_wire,),
                    ).fetchone()
                    if job_row is None:
                        rejected.append(assignment)
                        continue
                    decoded_job = JOB_DETAIL_PROJECTION.decode_one([job_row])
                    if decoded_job is None:
                        rejected.append(assignment)
                        continue
                    job_cache[job_id_wire] = decoded_job
                job = job_cache[job_id_wire]
                attempt_id = int(task_row["current_attempt_id"]) + 1
                _assign_task(
                    cur,
                    assignment.task_id.to_wire(),
                    str(assignment.worker_id),
                    str(worker_row["address"]),
                    attempt_id,
                    now_ms,
                )
                if not job.is_reservation_holder:
                    resources = resource_spec_from_scalars(
                        job.res_cpu_millicores,
                        job.res_memory_bytes,
                        job.res_disk_bytes,
                        job.res_device_json,
                    )
                    cur.execute(
                        "UPDATE workers SET committed_cpu_millicores = committed_cpu_millicores + ?, "
                        "committed_mem_bytes = committed_mem_bytes + ?, committed_gpu = committed_gpu + ?, "
                        "committed_tpu = committed_tpu + ? WHERE worker_id = ?",
                        (
                            int(resources.cpu_millicores),
                            int(resources.memory_bytes),
                            int(get_gpu_count(resources.device)),
                            int(get_tpu_count(resources.device)),
                            str(assignment.worker_id),
                        ),
                    )
                    entrypoint = proto_from_json(job.entrypoint_json, job_pb2.RuntimeEntrypoint)
                    # Load inline workdir files from the job_workdir_files table.
                    wf_rows = cur.execute(
                        "SELECT filename, data FROM job_workdir_files WHERE job_id = ?",
                        (job_id_wire,),
                    ).fetchall()
                    for wf_row in wf_rows:
                        entrypoint.workdir_files[wf_row["filename"]] = bytes(wf_row["data"])
                    run_request = job_pb2.RunTaskRequest(
                        task_id=assignment.task_id.to_wire(),
                        num_tasks=job.num_tasks,
                        entrypoint=entrypoint,
                        environment=proto_from_json(job.environment_json, job_pb2.EnvironmentConfig),
                        bundle_id=job.bundle_id,
                        resources=resources,
                        ports=json.loads(job.ports_json),
                        attempt_id=attempt_id,
                        constraints=[c.to_proto() for c in constraints_from_json(job.constraints_json)],
                        task_image=job.task_image,
                    )
                    if direct_dispatch:
                        start_requests.append((assignment.worker_id, str(worker_row["address"]), run_request))
                    else:
                        enqueue_run_dispatch(cur, str(assignment.worker_id), run_request.SerializeToString(), now_ms)
                    has_real_dispatch = True
                cur.execute(
                    "INSERT INTO worker_task_history(worker_id, task_id, assigned_at_ms) VALUES (?, ?, ?)",
                    (str(assignment.worker_id), assignment.task_id.to_wire(), now_ms),
                )
                jobs_to_update.add(job_id_wire)
                accepted.append(assignment)
            for job_id_wire in jobs_to_update:
                cur.execute(
                    "UPDATE jobs SET state = CASE WHEN state = ? THEN ? ELSE state END, "
                    "started_at_ms = COALESCE(started_at_ms, ?) WHERE job_id = ?",
                    (job_pb2.JOB_STATE_PENDING, job_pb2.JOB_STATE_RUNNING, now_ms, job_id_wire),
                )
            if accepted or rejected:
                actions = [("assignment_queued", a.task_id.to_wire(), {"worker_id": str(a.worker_id)}) for a in accepted]
                self._record_transaction(cur, "queue_assignments", actions)
        return AssignmentResult(
            tasks_to_kill=set(),
            has_real_dispatch=has_real_dispatch,
            accepted=accepted,
            rejected=rejected,
            start_requests=start_requests,
        )

    def _update_worker_health(self, cur: TransactionCursor, req: HeartbeatApplyRequest, now_ms: int) -> bool:
        """Update worker health, resource snapshot, and history.

        Returns False if the worker doesn't exist (caller should bail).
        """
        existing = _batch_worker_health(cur, [req], now_ms)
        return str(req.worker_id) in existing

    def _apply_task_transitions(
        self,
        cur: TransactionCursor,
        req: HeartbeatApplyRequest,
        now_ms: int,
    ) -> TxResult:
        """Apply task state updates for one worker within an existing transaction.

        Handles the full state machine: state transitions, retry logic,
        coscheduled cascade, resource decommit, endpoint cleanup, and
        deduplicated job recompute.
        """
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}
        cascaded_jobs: set[JobName] = set()
        jobs_to_recompute: set[JobName] = set()
        # Cache job_config rows keyed by job_id wire format.
        job_config_cache: dict[str, dict | None] = {}

        for update in req.updates:
            task_row = cur.execute("SELECT * FROM tasks WHERE task_id = ?", (update.task_id.to_wire(),)).fetchone()
            if task_row is None:
                continue
            task = TASK_DETAIL_PROJECTION.decode_one([task_row])
            if task_row_is_finished(task) or update.new_state in (
                job_pb2.TASK_STATE_UNSPECIFIED,
                job_pb2.TASK_STATE_PENDING,
            ):
                continue
            if update.attempt_id != int(task_row["current_attempt_id"]):
                stale = cur.execute(
                    "SELECT state FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
                    (update.task_id.to_wire(), update.attempt_id),
                ).fetchone()
                if stale is not None and int(stale["state"]) not in TERMINAL_TASK_STATES:
                    logger.error(
                        "Stale attempt precondition violation: task=%s reported=%d current=%d stale_state=%s",
                        update.task_id,
                        update.attempt_id,
                        int(task_row["current_attempt_id"]),
                        int(stale["state"]),
                    )
                continue

            prior_state = int(task_row["state"])

            # Fast path: task already in the reported state with no new data to apply.
            has_new_data = update.error is not None or update.exit_code is not None or update.resource_usage is not None
            if update.new_state == prior_state and not has_new_data:
                continue

            attempt_row = cur.execute(
                "SELECT * FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
                (update.task_id.to_wire(), update.attempt_id),
            ).fetchone()
            if attempt_row is None:
                continue
            # The attempt is already terminal (e.g. preempted, killed) but the task has
            # been rolled back to PENDING for retry and current_attempt_id still points
            # at the dead attempt. Reviving it would produce an inconsistent row where
            # state contradicts finished_at_ms/error.
            if int(attempt_row["state"]) in TERMINAL_TASK_STATES:
                logger.debug(
                    "Dropping late update for terminal attempt: task=%s attempt=%d attempt_state=%d reported=%d",
                    update.task_id,
                    update.attempt_id,
                    int(attempt_row["state"]),
                    int(update.new_state),
                )
                continue
            worker_id = attempt_row["worker_id"]
            if update.resource_usage is not None:
                ru = update.resource_usage
                cur.execute(
                    "INSERT INTO task_resource_history"
                    "(task_id, attempt_id, cpu_millicores, memory_mb, disk_mb, memory_peak_mb, timestamp_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        update.task_id.to_wire(),
                        update.attempt_id,
                        ru.cpu_millicores,
                        ru.memory_mb,
                        ru.disk_mb,
                        ru.memory_peak_mb,
                        now_ms,
                    ),
                )
            terminal_ms: int | None = None
            started_ms: int | None = None
            task_state = prior_state
            task_error = update.error
            task_exit = update.exit_code
            failure_count = int(task_row["failure_count"])
            preemption_count = int(task_row["preemption_count"])

            if update.new_state == job_pb2.TASK_STATE_RUNNING:
                started_ms = now_ms
                task_state = job_pb2.TASK_STATE_RUNNING
            elif update.new_state == job_pb2.TASK_STATE_BUILDING:
                task_state = job_pb2.TASK_STATE_BUILDING
            elif update.new_state in (
                job_pb2.TASK_STATE_FAILED,
                job_pb2.TASK_STATE_WORKER_FAILED,
                job_pb2.TASK_STATE_KILLED,
                job_pb2.TASK_STATE_UNSCHEDULABLE,
                job_pb2.TASK_STATE_SUCCEEDED,
            ):
                terminal_ms = now_ms
                task_state = int(update.new_state)
                if update.new_state == job_pb2.TASK_STATE_SUCCEEDED and task_exit is None:
                    task_exit = 0
                if update.new_state == job_pb2.TASK_STATE_UNSCHEDULABLE and task_error is None:
                    task_error = "Scheduling timeout exceeded"
                if update.new_state == job_pb2.TASK_STATE_FAILED:
                    failure_count += 1
                if update.new_state == job_pb2.TASK_STATE_WORKER_FAILED and prior_state in EXECUTING_TASK_STATES:
                    preemption_count += 1
                if update.new_state == job_pb2.TASK_STATE_WORKER_FAILED and prior_state == job_pb2.TASK_STATE_ASSIGNED:
                    task_state = job_pb2.TASK_STATE_PENDING
                    terminal_ms = None
                if update.new_state == job_pb2.TASK_STATE_FAILED and failure_count <= int(
                    task_row["max_retries_failure"]
                ):
                    task_state = job_pb2.TASK_STATE_PENDING
                    terminal_ms = None
                if (
                    update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
                    and preemption_count <= int(task_row["max_retries_preemption"])
                    and prior_state in EXECUTING_TASK_STATES
                ):
                    task_state = job_pb2.TASK_STATE_PENDING
                    terminal_ms = None

            # An attempt is terminal whenever the update itself is terminal, even
            # if the TASK rolls back to PENDING for a retry. terminal_ms above
            # tracks the task's finished_at_ms; the attempt needs its own stamp.
            attempt_terminal_ms = now_ms if int(update.new_state) in TERMINAL_TASK_STATES else None

            cur.execute(
                "UPDATE task_attempts SET state = ?, started_at_ms = COALESCE(started_at_ms, ?), "
                "finished_at_ms = COALESCE(finished_at_ms, ?), exit_code = COALESCE(?, exit_code), "
                "error = COALESCE(?, error) WHERE task_id = ? AND attempt_id = ?",
                (
                    int(update.new_state),
                    started_ms,
                    attempt_terminal_ms,
                    task_exit,
                    update.error,
                    update.task_id.to_wire(),
                    update.attempt_id,
                ),
            )
            # Clear denormalized worker columns when task leaves active state.
            if task_state in ACTIVE_TASK_STATES:
                cur.execute(
                    "UPDATE tasks SET state = ?, error = COALESCE(?, error), exit_code = COALESCE(?, exit_code), "
                    "started_at_ms = COALESCE(started_at_ms, ?), finished_at_ms = ?, "
                    "failure_count = ?, preemption_count = ? "
                    "WHERE task_id = ?",
                    (
                        task_state,
                        task_error,
                        task_exit,
                        started_ms,
                        terminal_ms,
                        failure_count,
                        preemption_count,
                        update.task_id.to_wire(),
                    ),
                )
            else:
                cur.execute(
                    "UPDATE tasks SET state = ?, error = COALESCE(?, error), exit_code = COALESCE(?, exit_code), "
                    "started_at_ms = COALESCE(started_at_ms, ?), finished_at_ms = ?, "
                    "failure_count = ?, preemption_count = ?, "
                    "current_worker_id = NULL, current_worker_address = NULL "
                    "WHERE task_id = ?",
                    (
                        task_state,
                        task_error,
                        task_exit,
                        started_ms,
                        terminal_ms,
                        failure_count,
                        preemption_count,
                        update.task_id.to_wire(),
                    ),
                )

            # Fetch and cache job_config row (avoids re-querying per task in same job).
            job_id_wire = task.job_id.to_wire()
            if job_id_wire not in job_config_cache:
                jc_row = cur.execute("SELECT * FROM job_config WHERE job_id = ?", (job_id_wire,)).fetchone()
                job_config_cache[job_id_wire] = dict(jc_row) if jc_row is not None else None
            jc = job_config_cache[job_id_wire]

            if worker_id is not None and task_state not in ACTIVE_TASK_STATES:
                if jc is not None:
                    resources = resource_spec_from_scalars(
                        int(jc["res_cpu_millicores"]),
                        int(jc["res_memory_bytes"]),
                        int(jc["res_disk_bytes"]),
                        jc["res_device_json"],
                    )
                    _decommit_worker_resources(cur, str(worker_id), resources)

            if update.new_state in TERMINAL_TASK_STATES:
                delete_task_endpoints(cur, self._db.endpoints, update.task_id.to_wire())

            # Coscheduled jobs: a terminal host failure should cascade to siblings.
            if jc is not None and task_state in FAILURE_TASK_STATES:
                has_cosched = bool(int(jc["has_coscheduling"]))
                siblings = _find_coscheduled_siblings(cur, task.job_id, update.task_id, has_cosched)
                resources = resource_spec_from_scalars(
                    int(jc["res_cpu_millicores"]),
                    int(jc["res_memory_bytes"]),
                    int(jc["res_disk_bytes"]),
                    jc["res_device_json"],
                )
                cascade_kill, cascade_workers = _terminate_coscheduled_siblings(
                    cur, self._db.endpoints, siblings, update.task_id, resources, now_ms
                )
                tasks_to_kill.update(cascade_kill)
                task_kill_workers.update(cascade_workers)

            # Mark job for recomputation (deduplicated, done after the task loop).
            if task_state != prior_state:
                jobs_to_recompute.add(task.job_id)

        # Recompute job states once per job instead of once per task.
        for job_id in jobs_to_recompute:
            if job_id in cascaded_jobs:
                continue
            new_job_state = self._recompute_job_state(cur, job_id)
            if new_job_state in TERMINAL_JOB_STATES:
                final_tasks_to_kill, final_task_kill_workers = _finalize_terminal_job(
                    cur, self._db.endpoints, job_id, new_job_state, now_ms
                )
                tasks_to_kill.update(final_tasks_to_kill)
                task_kill_workers.update(final_task_kill_workers)
                cascaded_jobs.add(job_id)
        if tasks_to_kill or cascaded_jobs:
            actions: list[tuple[str, str, dict[str, object]]] = [("heartbeat_applied", str(req.worker_id), {})]
            for job_id in cascaded_jobs:
                actions.append(("job_terminated", job_id.to_wire(), {}))
            self._record_transaction(cur, "apply_task_updates", actions)

        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def apply_task_updates(self, req: HeartbeatApplyRequest) -> TxResult:
        """Apply a batch of worker task updates atomically."""
        with self._db.transaction() as cur:
            now_ms = Timestamp.now().epoch_ms()
            if not self._update_worker_health(cur, req, now_ms):
                return TxResult()
            result = self._apply_task_transitions(cur, req, now_ms)

        return result

    def apply_heartbeats_batch(self, requests: list[HeartbeatApplyRequest]) -> list[HeartbeatApplyResult]:
        """Apply multiple heartbeats in a single transaction.

        Two-pass architecture to minimise SQL round-trips:

        1. Bulk-fetch all referenced task rows, classify each update as
           *steady-state* (same state, no error/exit_code) or *transition*.
        2a. Batch steady-state resource_usage writes via ``executemany``.
        2b. Feed only transitions through ``_apply_task_transitions``, which
            retains the full state machine (retry, cascade, decommit, etc.).

        Worker health updates are also batched via ``executemany``.
        """
        _empty = HeartbeatApplyResult(tasks_to_kill=set(), action=HeartbeatAction.OK)
        results: list[HeartbeatApplyResult] = [_empty] * len(requests)

        with self._db.transaction() as cur:
            now_ms = Timestamp.now().epoch_ms()

            # ── Batch worker health updates ───────────────────────────────
            existing_workers = _batch_worker_health(cur, requests, now_ms)

            # ── Bulk-fetch task rows for classification ───────────────────
            all_task_ids: list[str] = []
            for req in requests:
                if str(req.worker_id) not in existing_workers:
                    continue
                for update in req.updates:
                    if update.new_state not in (
                        job_pb2.TASK_STATE_UNSPECIFIED,
                        job_pb2.TASK_STATE_PENDING,
                    ):
                        all_task_ids.append(update.task_id.to_wire())

            task_row_map = _bulk_fetch_tasks(cur, all_task_ids)

            # ── Classify and split ────────────────────────────────────────
            task_history_params: list[tuple[str, int, int, int, int, int, int]] = []
            # (request_index, transition_request) pairs so results stay aligned.
            transition_entries: list[tuple[int, HeartbeatApplyRequest]] = []

            for req_idx, req in enumerate(requests):
                if str(req.worker_id) not in existing_workers:
                    continue

                transition_updates: list[TaskUpdate] = []
                for update in req.updates:
                    task_id_wire = update.task_id.to_wire()
                    task_row = task_row_map.get(task_id_wire)
                    if task_row is None:
                        continue

                    prior_state = int(task_row["state"])
                    is_state_change = update.new_state != prior_state
                    has_terminal_data = update.error is not None or update.exit_code is not None

                    if is_state_change or has_terminal_data:
                        transition_updates.append(update)
                    else:
                        # Steady-state: check finished / stale attempt before writing.
                        task = self._db.decode_task(task_row)
                        if task_row_is_finished(task):
                            continue
                        if update.attempt_id != int(task_row["current_attempt_id"]):
                            continue
                        if update.resource_usage is not None:
                            u = update.resource_usage
                            task_history_params.append(
                                (
                                    task_id_wire,
                                    update.attempt_id,
                                    u.cpu_millicores,
                                    u.memory_mb,
                                    u.disk_mb,
                                    u.memory_peak_mb,
                                    now_ms,
                                )
                            )

                if transition_updates:
                    transition_entries.append(
                        (
                            req_idx,
                            HeartbeatApplyRequest(
                                worker_id=req.worker_id,
                                worker_resource_snapshot=None,  # already handled above
                                updates=transition_updates,
                            ),
                        )
                    )

            # ── Pass 2a: batch task resource history writes ─────────────────
            if task_history_params:
                cur.executemany(
                    "INSERT INTO task_resource_history"
                    "(task_id, attempt_id, cpu_millicores, memory_mb, disk_mb, memory_peak_mb, timestamp_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    task_history_params,
                )

            # ── Pass 2b: transitions via existing state machine ───────────
            for req_idx, treq in transition_entries:
                tx_result = self._apply_task_transitions(cur, treq, now_ms)
                results[req_idx] = HeartbeatApplyResult(
                    tasks_to_kill=tx_result.tasks_to_kill,
                    action=HeartbeatAction.OK,
                )

        return results

    def apply_heartbeat(self, req: HeartbeatApplyRequest) -> HeartbeatApplyResult:
        result = self.apply_task_updates(req)
        return HeartbeatApplyResult(tasks_to_kill=result.tasks_to_kill, action=HeartbeatAction.OK)

    def _remove_failed_worker(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        error: str,
        *,
        now_ms: int,
    ) -> TxResult:
        """Remove a definitively failed worker and cascade its task state."""
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}
        task_rows = cur.execute(
            "SELECT t.task_id, t.current_attempt_id, t.state, t.preemption_count, t.max_retries_preemption, "
            "j.is_reservation_holder "
            "FROM tasks t "
            "JOIN jobs j ON j.job_id = t.job_id "
            "WHERE t.current_worker_id = ? AND t.state IN (?, ?, ?)",
            (str(worker_id), *ACTIVE_TASK_STATES),
        ).fetchall()
        for task_row in task_rows:
            tid = str(task_row["task_id"])
            prior_state = int(task_row["state"])
            is_reservation_holder = bool(int(task_row["is_reservation_holder"]))
            if is_reservation_holder:
                new_task_state = job_pb2.TASK_STATE_PENDING
                preemption_count = int(task_row["preemption_count"])
            else:
                new_task_state, preemption_count = _resolve_task_failure_state(
                    prior_state,
                    int(task_row["preemption_count"]),
                    int(task_row["max_retries_preemption"]),
                    job_pb2.TASK_STATE_WORKER_FAILED,
                )
            if is_reservation_holder:
                cur.execute(
                    "DELETE FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
                    (tid, int(task_row["current_attempt_id"])),
                )
                cur.execute(
                    "UPDATE tasks SET state = ?, current_attempt_id = -1, started_at_ms = NULL, "
                    "finished_at_ms = NULL, error = NULL, preemption_count = 0, "
                    "current_worker_id = NULL, current_worker_address = NULL WHERE task_id = ?",
                    (new_task_state, tid),
                )
            else:
                _terminate_task(
                    cur,
                    self._db.endpoints,
                    tid,
                    int(task_row["current_attempt_id"]),
                    new_task_state,
                    f"Worker {worker_id} failed: {error}",
                    now_ms,
                    attempt_state=job_pb2.TASK_STATE_WORKER_FAILED,
                    preemption_count=preemption_count,
                )
            task_id = JobName.from_wire(tid)
            parent_job_id, _ = task_id.require_task()
            new_job_state = self._recompute_job_state(cur, parent_job_id)
            if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
                cascaded_tasks_to_kill, cascaded_task_kill_workers = _cascade_terminal_job(
                    cur, self._db.endpoints, parent_job_id, now_ms, f"Worker {worker_id} failed"
                )
                tasks_to_kill.update(cascaded_tasks_to_kill)
                task_kill_workers.update(cascaded_task_kill_workers)
            elif new_task_state == job_pb2.TASK_STATE_PENDING:
                policy = _resolve_preemption_policy(cur, parent_job_id)
                if policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
                    child_tasks_to_kill, child_task_kill_workers = _cascade_children(
                        cur,
                        self._db.endpoints,
                        parent_job_id,
                        now_ms,
                        "Parent task preempted",
                        exclude_reservation_holders=True,
                    )
                    tasks_to_kill.update(child_tasks_to_kill)
                    task_kill_workers.update(child_task_kill_workers)
            if new_task_state == job_pb2.TASK_STATE_WORKER_FAILED:
                tasks_to_kill.add(task_id)
        _remove_worker(cur, str(worker_id))
        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def _record_heartbeat_failure(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        error: str,
        drained_dispatch: DispatchBatch,
        *,
        force_remove: bool = False,
        now_ms: int | None = None,
    ) -> HeartbeatFailureResult:
        """Apply a heartbeat failure inside an existing transaction."""
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}
        row = cur.execute(
            "SELECT consecutive_failures, last_heartbeat_ms FROM workers WHERE worker_id = ? AND active = 1",
            (str(worker_id),),
        ).fetchone()
        if row is None:
            return HeartbeatFailureResult(
                worker_removed=True,
                action=HeartbeatAction.WORKER_FAILED,
                failure_threshold=self._heartbeat_failure_threshold,
            )

        now_ms = now_ms or Timestamp.now().epoch_ms()
        last_heartbeat_ms = row["last_heartbeat_ms"]
        last_heartbeat_age_ms = None if last_heartbeat_ms is None else max(0, now_ms - int(last_heartbeat_ms))
        failures = int(row["consecutive_failures"]) + 1
        cur.execute(
            "UPDATE workers SET consecutive_failures = ?, healthy = CASE WHEN ? >= ? THEN 0 ELSE healthy END "
            "WHERE worker_id = ?",
            (failures, failures, self._heartbeat_failure_threshold, str(worker_id)),
        )
        should_remove = force_remove or failures >= self._heartbeat_failure_threshold
        if should_remove:
            removal = self._remove_failed_worker(cur, worker_id, error, now_ms=now_ms)
            tasks_to_kill.update(removal.tasks_to_kill)
            task_kill_workers.update(removal.task_kill_workers)
        else:
            for req in drained_dispatch.tasks_to_run:
                enqueue_run_dispatch(cur, str(worker_id), req.SerializeToString(), now_ms)
            for task_id in drained_dispatch.tasks_to_kill:
                enqueue_kill_dispatch(cur, str(worker_id), task_id, now_ms)
        action = HeartbeatAction.WORKER_FAILED if should_remove else HeartbeatAction.TRANSIENT_FAILURE
        return HeartbeatFailureResult(
            tasks_to_kill=tasks_to_kill,
            task_kill_workers=task_kill_workers,
            worker_removed=should_remove,
            action=action,
            consecutive_failures=failures,
            failure_threshold=self._heartbeat_failure_threshold,
            last_heartbeat_age_ms=last_heartbeat_age_ms,
        )

    def record_heartbeat_failure(
        self,
        worker_id: WorkerId,
        error: str,
        drained_dispatch: DispatchBatch,
        *,
        force_remove: bool = False,
    ) -> TxResult:
        """Record heartbeat failure and requeue/flush drained dispatches."""
        with self._db.transaction() as cur:
            result = self._record_heartbeat_failure(
                cur,
                worker_id,
                error,
                drained_dispatch,
                force_remove=force_remove,
            )
            self._record_transaction(
                cur,
                "heartbeat_failure",
                [("worker_heartbeat_failed", str(worker_id), {"error": error})],
            )
        if result.worker_removed:
            self._db.remove_worker_from_attr_cache(worker_id)
        return TxResult(tasks_to_kill=result.tasks_to_kill, task_kill_workers=result.task_kill_workers)

    def fail_heartbeat_for_worker(
        self,
        worker_id: WorkerId,
        error: str,
        snapshot: DispatchBatch,
        *,
        force_remove: bool = False,
    ) -> HeartbeatFailureResult:
        with self._db.transaction() as cur:
            result = self._record_heartbeat_failure(
                cur,
                worker_id,
                error,
                snapshot,
                force_remove=force_remove,
            )
            self._record_transaction(
                cur,
                "heartbeat_failure",
                [("worker_heartbeat_failed", str(worker_id), {"error": error})],
            )
        if result.worker_removed:
            self._db.remove_worker_from_attr_cache(worker_id)
        return result

    def fail_heartbeats_batch(
        self,
        failures: list[tuple[DispatchBatch, str]],
        *,
        force_remove: bool = False,
        chunk_size: int = FAIL_HEARTBEATS_CHUNK_SIZE,
    ) -> WorkerFailureBatchResult:
        """Apply heartbeat RPC failures in chunked transactions.

        Each chunk is its own write transaction so we release the SQLite
        writer between chunks and other RPCs (RegisterEndpoint, Register,
        LaunchJob, apply_heartbeats_batch) can interleave instead of
        stalling for the full batch. A single big transaction would starve
        them for seconds when a zone-wide failure knocks out hundreds of
        workers at once.

        Heartbeat failures are idempotent at the semantic level
        (``_record_heartbeat_failure`` guards on ``active = 1``), so
        partial progress on crash is safe. Downstream consumers do not
        rely on cross-worker atomicity.
        """
        if not failures:
            return WorkerFailureBatchResult()

        results: list[HeartbeatFailureResult] = []
        removed_workers: list[tuple[WorkerId, str | None]] = []
        all_tasks_to_kill: set[JobName] = set()
        all_task_kill_workers: dict[JobName, WorkerId] = {}

        for chunk_start in range(0, len(failures), chunk_size):
            chunk = failures[chunk_start : chunk_start + chunk_size]
            chunk_actions: list[tuple[str, str, dict[str, object]]] = []
            with self._db.transaction() as cur:
                now_ms = Timestamp.now().epoch_ms()
                for snapshot, error in chunk:
                    result = self._record_heartbeat_failure(
                        cur,
                        snapshot.worker_id,
                        error,
                        snapshot,
                        force_remove=force_remove,
                        now_ms=now_ms,
                    )
                    results.append(result)
                    chunk_actions.append(("worker_heartbeat_failed", str(snapshot.worker_id), {"error": error}))
                    all_tasks_to_kill.update(result.tasks_to_kill)
                    all_task_kill_workers.update(result.task_kill_workers)
                    if result.worker_removed:
                        removed_workers.append((snapshot.worker_id, snapshot.worker_address))
                self._record_transaction(
                    cur,
                    "heartbeat_failures_batch",
                    chunk_actions,
                    payload={"count": len(chunk_actions)},
                )

        for worker_id, _ in removed_workers:
            self._db.remove_worker_from_attr_cache(worker_id)
        return WorkerFailureBatchResult(
            tasks_to_kill=all_tasks_to_kill,
            task_kill_workers=all_task_kill_workers,
            removed_workers=removed_workers,
            results=results,
        )

    def mark_task_unschedulable(self, task_id: JobName, reason: str) -> TxResult:
        """Mark a task as unschedulable using the task transition engine."""
        with self._db.transaction() as cur:
            row = cur.execute("SELECT job_id FROM tasks WHERE task_id = ?", (task_id.to_wire(),)).fetchone()
            if row is None:
                return TxResult()
            now_ms = Timestamp.now().epoch_ms()
            _terminate_task(
                cur,
                self._db.endpoints,
                task_id.to_wire(),
                None,
                job_pb2.TASK_STATE_UNSCHEDULABLE,
                reason,
                now_ms,
            )
            self._recompute_job_state(cur, JobName.from_wire(str(row["job_id"])))
            self._record_transaction(
                cur, "mark_task_unschedulable", [("task_unschedulable", task_id.to_wire(), {"reason": reason})]
            )
        return TxResult()

    def preempt_task(self, task_id: JobName, reason: str) -> TxResult:
        """Preempt a running task, consuming from preemption retry budget.

        Marks the task as PREEMPTED (or retries as PENDING if budget remains),
        decommits its resources from the worker, and cascades to children if needed.
        """
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}
        with self._db.transaction() as cur:
            row = cur.execute(
                "SELECT t.task_id, t.job_id, t.state, t.current_attempt_id, "
                "t.preemption_count, t.max_retries_preemption, "
                "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json "
                f"FROM tasks t JOIN jobs j ON j.job_id = t.job_id {JOB_CONFIG_JOIN} "
                "WHERE t.task_id = ?",
                (task_id.to_wire(),),
            ).fetchone()
            if row is None:
                return TxResult()

            prior_state = int(row["state"])
            if prior_state not in ACTIVE_TASK_STATES:
                return TxResult()

            now_ms = Timestamp.now().epoch_ms()
            new_state, preemption_count = _resolve_task_failure_state(
                prior_state,
                int(row["preemption_count"]),
                int(row["max_retries_preemption"]),
                job_pb2.TASK_STATE_PREEMPTED,
            )
            # Fetch worker_id from the attempt for resource decommit.
            attempt_row = cur.execute(
                "SELECT worker_id FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
                (task_id.to_wire(), int(row["current_attempt_id"])),
            ).fetchone()
            attempt_worker_id = str(attempt_row["worker_id"]) if attempt_row and attempt_row["worker_id"] else None
            attempt_resources = None
            if attempt_worker_id is not None:
                attempt_resources = resource_spec_from_scalars(
                    int(row["res_cpu_millicores"]),
                    int(row["res_memory_bytes"]),
                    int(row["res_disk_bytes"]),
                    row["res_device_json"],
                )

            _terminate_task(
                cur,
                self._db.endpoints,
                task_id.to_wire(),
                int(row["current_attempt_id"]),
                new_state,
                reason,
                now_ms,
                attempt_state=job_pb2.TASK_STATE_PREEMPTED,
                worker_id=attempt_worker_id,
                resources=attempt_resources,
                preemption_count=preemption_count,
            )

            # Recompute job state and cascade if terminal
            job_id = JobName.from_wire(str(row["job_id"]))
            new_job_state = self._recompute_job_state(cur, job_id)
            if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
                cascade_kills, cascade_workers = _finalize_terminal_job(
                    cur, self._db.endpoints, job_id, new_job_state, now_ms
                )
                tasks_to_kill.update(cascade_kills)
                task_kill_workers.update(cascade_workers)
            elif new_state == job_pb2.TASK_STATE_PENDING:
                policy = _resolve_preemption_policy(cur, job_id)
                if policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
                    child_kills, child_workers = _cascade_children(
                        cur,
                        self._db.endpoints,
                        job_id,
                        now_ms,
                        reason,
                        exclude_reservation_holders=True,
                    )
                    tasks_to_kill.update(child_kills)
                    task_kill_workers.update(child_workers)

            if new_state == job_pb2.TASK_STATE_PREEMPTED:
                tasks_to_kill.add(task_id)

            self._record_transaction(cur, "preempt_task", [("task_preempted", task_id.to_wire(), {"reason": reason})])

        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def cancel_tasks_for_timeout(self, task_ids: set[JobName], reason: str) -> TxResult:
        """Mark executing tasks as FAILED due to execution timeout and return kill set.

        Each task is moved to TASK_STATE_FAILED with the given reason.
        Timeouts are hard failures — retry logic is intentionally bypassed.

        Two-phase design: all reads happen before any writes so that
        coscheduled siblings sharing a job are never double-processed from
        stale prefetched rows.
        """
        if not task_ids:
            return TxResult()
        with self._db.transaction() as cur:
            wires = [tid.to_wire() for tid in task_ids]
            placeholders = ",".join("?" for _ in wires)
            rows = cur.execute(
                f"SELECT t.task_id, t.job_id, t.current_worker_id AS worker_id, t.current_attempt_id, "
                f"t.failure_count, j.is_reservation_holder, "
                f"jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json, "
                f"jc.has_coscheduling "
                f"FROM tasks t JOIN jobs j ON j.job_id = t.job_id {JOB_CONFIG_JOIN} "
                f"WHERE t.task_id IN ({placeholders}) AND t.state IN (?, ?)",
                (*wires, *EXECUTING_TASK_STATES),
            ).fetchall()

            # -- Phase 1: read all state before any mutations. --
            now_ms = Timestamp.now().epoch_ms()
            job_row_cache: dict[str, dict] = {}
            # Collect directly-timed-out task wires for dedup against siblings.
            direct_task_wires: set[str] = set()
            # Per-job list of siblings to cascade (collected across all timed-out tasks).
            siblings_by_job: dict[str, list[_CoscheduledSibling]] = {}

            for row in rows:
                task_id_wire = str(row["task_id"])
                direct_task_wires.add(task_id_wire)
                job_id_wire = str(row["job_id"])
                if job_id_wire not in job_row_cache:
                    job_row_cache[job_id_wire] = dict(row)
                has_cosched = bool(int(row["has_coscheduling"]))
                tid = JobName.from_wire(task_id_wire)
                siblings = _find_coscheduled_siblings(cur, JobName.from_wire(job_id_wire), tid, has_cosched)
                if siblings:
                    existing = siblings_by_job.get(job_id_wire, [])
                    existing.extend(siblings)
                    siblings_by_job[job_id_wire] = existing

            # Deduplicate siblings: drop any that will already be terminated
            # directly as timed-out tasks, and deduplicate across multiple
            # trigger tasks within the same job.
            for job_id_wire, siblings in siblings_by_job.items():
                seen: set[str] = set()
                deduped: list[_CoscheduledSibling] = []
                for sib in siblings:
                    if sib.task_id not in direct_task_wires and sib.task_id not in seen:
                        seen.add(sib.task_id)
                        deduped.append(sib)
                siblings_by_job[job_id_wire] = deduped

            # -- Phase 2: apply all mutations. --
            tasks_to_kill: set[JobName] = set()
            task_kill_workers: dict[JobName, WorkerId] = {}
            jobs_to_update: set[str] = set()

            for row in rows:
                task_id_wire = str(row["task_id"])
                tid = JobName.from_wire(task_id_wire)
                job_id_wire = str(row["job_id"])
                worker_id_str = row["worker_id"]
                tasks_to_kill.add(tid)
                decommit_worker = None
                decommit_resources = None
                if worker_id_str is not None:
                    task_kill_workers[tid] = WorkerId(str(worker_id_str))
                    if not int(row["is_reservation_holder"]):
                        decommit_worker = str(worker_id_str)
                        decommit_resources = resource_spec_from_scalars(
                            int(row["res_cpu_millicores"]),
                            int(row["res_memory_bytes"]),
                            int(row["res_disk_bytes"]),
                            row["res_device_json"],
                        )
                attempt_id = row["current_attempt_id"]
                _terminate_task(
                    cur,
                    self._db.endpoints,
                    task_id_wire,
                    int(attempt_id) if attempt_id is not None else None,
                    job_pb2.TASK_STATE_FAILED,
                    reason,
                    now_ms,
                    worker_id=decommit_worker,
                    resources=decommit_resources,
                    failure_count=int(row["failure_count"]) + 1,
                )
                jobs_to_update.add(job_id_wire)

            # Terminate coscheduled siblings (deduplicated, all reads already done).
            for job_id_wire, siblings in siblings_by_job.items():
                if not siblings:
                    continue
                jc_row = job_row_cache[job_id_wire]
                job_resources = resource_spec_from_scalars(
                    int(jc_row["res_cpu_millicores"]),
                    int(jc_row["res_memory_bytes"]),
                    int(jc_row["res_disk_bytes"]),
                    jc_row["res_device_json"],
                )
                # Pick the first direct-timeout task in this job as the "cause" for the error message.
                cause_tid = next(JobName.from_wire(str(r["task_id"])) for r in rows if str(r["job_id"]) == job_id_wire)
                cascade_kill, cascade_workers = _terminate_coscheduled_siblings(
                    cur, self._db.endpoints, siblings, cause_tid, job_resources, now_ms
                )
                tasks_to_kill.update(cascade_kill)
                task_kill_workers.update(cascade_workers)
                jobs_to_update.add(job_id_wire)

            for job_wire in jobs_to_update:
                new_job_state = self._recompute_job_state(cur, JobName.from_wire(job_wire))
                if new_job_state in TERMINAL_JOB_STATES:
                    final_kill, final_workers = _finalize_terminal_job(
                        cur, self._db.endpoints, JobName.from_wire(job_wire), new_job_state, now_ms
                    )
                    tasks_to_kill.update(final_kill)
                    task_kill_workers.update(final_workers)
            self._record_transaction(
                cur,
                "cancel_tasks_for_timeout",
                [("task_timeout", tid.to_wire(), {"reason": reason}) for tid in tasks_to_kill],
            )
        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def drain_dispatch(self, worker_id: WorkerId) -> DispatchBatch | None:
        """Drain buffered dispatches and snapshot worker running tasks."""
        with self._db.transaction() as cur:
            worker_row = cur.execute(
                "SELECT worker_id, address FROM workers " "WHERE worker_id = ? AND active = 1 AND healthy = 1",
                (str(worker_id),),
            ).fetchone()
            if worker_row is None:
                return None
            dispatch_rows = cur.execute(
                "SELECT id, kind, payload_proto, task_id FROM dispatch_queue WHERE worker_id = ? ORDER BY id ASC",
                (str(worker_id),),
            ).fetchall()
            if dispatch_rows:
                cur.execute("DELETE FROM dispatch_queue WHERE worker_id = ?", (str(worker_id),))
            running_rows_raw = cur.execute(
                "SELECT t.task_id, t.current_attempt_id, t.job_id "
                "FROM tasks t "
                "WHERE t.current_worker_id = ? AND t.state IN (?, ?, ?) "
                "ORDER BY t.task_id ASC",
                (str(worker_id), *ACTIVE_TASK_STATES),
            ).fetchall()
            running_job_ids = {str(row["job_id"]) for row in running_rows_raw}
            if running_job_ids:
                holder_placeholders = ",".join("?" for _ in running_job_ids)
                holder_rows = cur.execute(
                    f"SELECT job_id FROM jobs WHERE job_id IN ({holder_placeholders}) AND is_reservation_holder = 1",
                    tuple(running_job_ids),
                ).fetchall()
                holder_ids = {str(r["job_id"]) for r in holder_rows}
            else:
                holder_ids = set()
            running_rows = [r for r in running_rows_raw if str(r["job_id"]) not in holder_ids]
            tasks_to_run: list[job_pb2.RunTaskRequest] = []
            tasks_to_kill: list[str] = []
            for row in dispatch_rows:
                if str(row["kind"]) == "run" and row["payload_proto"] is not None:
                    req = job_pb2.RunTaskRequest()
                    req.ParseFromString(bytes(row["payload_proto"]))
                    tasks_to_run.append(req)
                elif row["task_id"] is not None:
                    tasks_to_kill.append(str(row["task_id"]))
            return DispatchBatch(
                worker_id=WorkerId(str(worker_row["worker_id"])),
                worker_address=str(worker_row["address"]),
                running_tasks=[
                    RunningTaskEntry(
                        task_id=JobName.from_wire(str(row["task_id"])),
                        attempt_id=int(row["current_attempt_id"]),
                    )
                    for row in running_rows
                ],
                tasks_to_run=tasks_to_run,
                tasks_to_kill=tasks_to_kill,
            )

    def drain_dispatch_all(self) -> list[DispatchBatch]:
        """Drain buffered dispatches and snapshot running tasks for all healthy active workers.

        Reads (workers, running tasks, reservation filter) use a read snapshot
        to avoid holding the write lock. The write lock is only held for the
        dispatch_queue SELECT + DELETE.
        """
        # -- Phase 1: read-only queries (no write lock) --
        with self._db.read_snapshot() as snap:
            worker_rows = snap.fetchall("SELECT worker_id, address FROM workers WHERE active = 1 AND healthy = 1")
            if not worker_rows:
                return []

            worker_id_set = {str(row["worker_id"]) for row in worker_rows}

            running_rows = snap.fetchall(
                "SELECT t.current_worker_id AS worker_id, t.task_id, t.current_attempt_id, t.job_id "
                "FROM tasks t "
                "WHERE t.state IN (?, ?, ?) AND t.current_worker_id IS NOT NULL "
                "ORDER BY t.task_id ASC",
                tuple(ACTIVE_TASK_STATES),
            )

            # Batch-check reservation holders instead of joining the jobs table
            running_job_ids = {str(row["job_id"]) for row in running_rows}
            reservation_holder_ids: set[str] = set()
            if running_job_ids:
                job_placeholders = ",".join("?" for _ in running_job_ids)
                res_rows = snap.fetchall(
                    f"SELECT job_id FROM jobs WHERE job_id IN ({job_placeholders}) AND is_reservation_holder = 1",
                    tuple(running_job_ids),
                )
                reservation_holder_ids = {str(row["job_id"]) for row in res_rows}

        running_rows = [row for row in running_rows if str(row["job_id"]) not in reservation_holder_ids]

        # -- Phase 2: write lock only for dispatch_queue drain --
        placeholders = ",".join("?" for _ in worker_id_set)
        with self._db.transaction() as cur:
            dispatch_rows = cur.execute(
                f"SELECT worker_id, id, kind, payload_proto, task_id FROM dispatch_queue "
                f"WHERE worker_id IN ({placeholders}) ORDER BY id ASC",
                tuple(worker_id_set),
            ).fetchall()
            if dispatch_rows:
                cur.execute(
                    f"DELETE FROM dispatch_queue WHERE worker_id IN ({placeholders})",
                    tuple(worker_id_set),
                )

        # -- Phase 3: build results (pure Python, no lock) --
        dispatch_by_worker: dict[str, list[Any]] = defaultdict(list)
        for row in dispatch_rows:
            dispatch_by_worker[str(row["worker_id"])].append(row)

        running_by_worker: dict[str, list[Any]] = defaultdict(list)
        for row in running_rows:
            running_by_worker[str(row["worker_id"])].append(row)

        batches: list[DispatchBatch] = []
        for worker_row in worker_rows:
            wid = str(worker_row["worker_id"])
            w_dispatch = dispatch_by_worker.get(wid, [])
            w_running = running_by_worker.get(wid, [])

            tasks_to_run: list[job_pb2.RunTaskRequest] = []
            tasks_to_kill: list[str] = []
            for row in w_dispatch:
                if str(row["kind"]) == "run" and row["payload_proto"] is not None:
                    req = job_pb2.RunTaskRequest()
                    req.ParseFromString(bytes(row["payload_proto"]))
                    tasks_to_run.append(req)
                elif row["task_id"] is not None:
                    tasks_to_kill.append(str(row["task_id"]))

            batches.append(
                DispatchBatch(
                    worker_id=WorkerId(wid),
                    worker_address=str(worker_row["address"]),
                    running_tasks=[
                        RunningTaskEntry(
                            task_id=JobName.from_wire(str(row["task_id"])),
                            attempt_id=int(row["current_attempt_id"]),
                        )
                        for row in w_running
                    ],
                    tasks_to_run=tasks_to_run,
                    tasks_to_kill=tasks_to_kill,
                )
            )

        return batches

    def requeue_dispatch(self, batch: DispatchBatch) -> None:
        """Re-queue drained dispatch payloads for later delivery."""
        with self._db.transaction() as cur:
            now_ms = Timestamp.now().epoch_ms()
            for req in batch.tasks_to_run:
                enqueue_run_dispatch(cur, str(batch.worker_id), req.SerializeToString(), now_ms)
            for task_id in batch.tasks_to_kill:
                enqueue_kill_dispatch(cur, str(batch.worker_id), task_id, now_ms)

    def remove_finished_job(self, job_id: JobName) -> bool:
        """Remove a finished job and its tasks from state.

        Only removes jobs that are in a terminal state (SUCCEEDED, FAILED, KILLED,
        UNSCHEDULABLE). This allows job names to be reused after completion.

        Args:
            job_id: The job ID to remove

        Returns:
            True if the job was removed, False if it doesn't exist or is not finished
        """
        with self._db.transaction() as cur:
            row = cur.execute("SELECT state FROM jobs WHERE job_id = ?", (job_id.to_wire(),)).fetchone()
            if row is None:
                return False
            state = int(row["state"])
            if state not in (
                job_pb2.JOB_STATE_SUCCEEDED,
                job_pb2.JOB_STATE_FAILED,
                job_pb2.JOB_STATE_KILLED,
                job_pb2.JOB_STATE_UNSCHEDULABLE,
            ):
                return False
            cur.execute("DELETE FROM jobs WHERE job_id = ?", (job_id.to_wire(),))
            self._record_transaction(cur, "remove_finished_job", [("job_removed", job_id.to_wire(), {"state": state})])
            return True

    def remove_worker(self, worker_id: WorkerId) -> WorkerDetailRow | None:
        with self._db.transaction() as cur:
            row = cur.execute("SELECT * FROM workers WHERE worker_id = ?", (str(worker_id),)).fetchone()
            if row is None:
                return None
            _remove_worker(cur, str(worker_id))
            self._record_transaction(cur, "remove_worker", [("worker_removed", str(worker_id), {})])
        self._db.remove_worker_from_attr_cache(worker_id)
        return WORKER_DETAIL_PROJECTION.decode_one([row])

    def _prune_per_worker_history(
        self,
        table: str,
        retention: int,
        order_by: str = "id DESC",
    ) -> int:
        """Trim a per-worker history table to *retention* rows per worker.

        Used by the background prune thread. The NOT IN subquery per worker is
        expensive; batching all workers in a single transaction amortizes the
        overhead versus running it on every assign/heartbeat.
        """
        with self._db.transaction() as cur:
            rows = cur.execute(
                f"SELECT worker_id, COUNT(*) as cnt FROM {table} GROUP BY worker_id HAVING cnt > ?",
                (retention,),
            ).fetchall()
            total_deleted = 0
            for row in rows:
                wid = row["worker_id"]
                cur.execute(
                    f"DELETE FROM {table} "
                    "WHERE worker_id = ? "
                    f"AND id NOT IN ("
                    f"  SELECT id FROM {table} "
                    "  WHERE worker_id = ? "
                    f"  ORDER BY {order_by} LIMIT ?"
                    ")",
                    (wid, wid, retention),
                )
                total_deleted += cur.rowcount
        if total_deleted > 0:
            logger.info("Pruned %d %s rows", total_deleted, table)
        return total_deleted

    def prune_worker_task_history(self) -> int:
        """Trim worker_task_history to WORKER_TASK_HISTORY_RETENTION rows per worker."""
        return self._prune_per_worker_history(
            "worker_task_history",
            WORKER_TASK_HISTORY_RETENTION,
            order_by="assigned_at_ms DESC, id DESC",
        )

    def prune_worker_resource_history(self) -> int:
        """Trim worker_resource_history to WORKER_RESOURCE_HISTORY_RETENTION rows per worker."""
        return self._prune_per_worker_history(
            "worker_resource_history",
            WORKER_RESOURCE_HISTORY_RETENTION,
        )

    def prune_task_resource_history(self) -> int:
        """Two-pass prune:

        1. Evict all history for tasks that have been in a terminal state
           longer than TASK_RESOURCE_HISTORY_TERMINAL_TTL. Dashboards read
           peak memory from tasks.peak_memory_mb after termination; the
           per-sample rows are dead weight and are ~85% of the table on
           prod.
        2. Logarithmic downsampling for anything that remains: when a
           (task, attempt) exceeds 2*N rows, thin the older half by deleting
           every other row so older data grows exponentially sparser.

        Deletes are chunked so the writer lock releases between chunks.
        """
        now_ms = Timestamp.now().epoch_ms()
        ttl_cutoff_ms = now_ms - TASK_RESOURCE_HISTORY_TERMINAL_TTL.to_ms()
        terminal_placeholders = ",".join("?" for _ in TERMINAL_TASK_STATES)

        with self._db.read_snapshot() as snap:
            terminal_ids = [
                str(r["task_id"])
                for r in snap.fetchall(
                    f"SELECT task_id FROM tasks "
                    f"WHERE state IN ({terminal_placeholders}) "
                    f"AND finished_at_ms IS NOT NULL AND finished_at_ms < ?",
                    (*TERMINAL_TASK_STATES, ttl_cutoff_ms),
                )
            ]

        evicted_terminal = 0
        for chunk_start in range(0, len(terminal_ids), TASK_RESOURCE_HISTORY_DELETE_CHUNK):
            chunk = terminal_ids[chunk_start : chunk_start + TASK_RESOURCE_HISTORY_DELETE_CHUNK]
            ph = ",".join("?" * len(chunk))
            with self._db.transaction() as cur:
                cur.execute(f"DELETE FROM task_resource_history WHERE task_id IN ({ph})", tuple(chunk))
                evicted_terminal += cur.rowcount

        threshold = TASK_RESOURCE_HISTORY_RETENTION * 2
        with self._db.transaction() as cur:
            overflows = cur.execute(
                "SELECT task_id, attempt_id, COUNT(*) as cnt "
                "FROM task_resource_history "
                "GROUP BY task_id, attempt_id HAVING cnt > ?",
                (threshold,),
            ).fetchall()
            ids_to_delete: list[int] = []
            for row in overflows:
                tid, aid = row["task_id"], row["attempt_id"]
                # Load all IDs into Python for index-based thinning.
                # Bounded by 2*N + heartbeats-per-prune-cycle (~160 rows max at N=50).
                all_ids = [
                    r["id"]
                    for r in cur.execute(
                        "SELECT id FROM task_resource_history " "WHERE task_id = ? AND attempt_id = ? ORDER BY id ASC",
                        (tid, aid),
                    ).fetchall()
                ]
                # Keep the newest N rows untouched; thin the older portion by 2x.
                older = all_ids[: len(all_ids) - TASK_RESOURCE_HISTORY_RETENTION]
                ids_to_delete.extend(older[1::2])

            total_deleted = 0
            for chunk_start in range(0, len(ids_to_delete), 900):
                chunk = ids_to_delete[chunk_start : chunk_start + 900]
                ph = ",".join("?" * len(chunk))
                cur.execute(f"DELETE FROM task_resource_history WHERE id IN ({ph})", tuple(chunk))
                total_deleted += cur.rowcount
        if evicted_terminal > 0:
            logger.info("Evicted %d task_resource_history rows (terminal TTL)", evicted_terminal)
        if total_deleted > 0:
            logger.info("Pruned %d task_resource_history rows (log downsampling)", total_deleted)
        return evicted_terminal + total_deleted

    def _batch_delete(
        self,
        sql: str,
        params: tuple[object, ...],
        stopped: Callable[[], bool],
        pause_between_s: float,
    ) -> int:
        """Delete rows in batches, sleeping between transactions.

        Returns the total number of rows deleted.
        """
        total = 0
        while not stopped():
            with self._db.transaction() as cur:
                batch = cur.execute(sql, params).rowcount
            if batch == 0:
                break
            total += batch
            time.sleep(pause_between_s)
        return total

    def prune_old_data(
        self,
        *,
        job_retention: Duration,
        worker_retention: Duration,
        txn_action_retention: Duration,
        profile_retention: Duration,
        stop_event: threading.Event | None = None,
        pause_between_s: float = 1.0,
    ) -> PruneResult:
        """Incrementally delete old data, one row per transaction.

        Designed to run on a background thread. Each deletion holds the write
        lock for only one CASCADE delete (one job or one worker), then sleeps
        to let scheduling and heartbeats proceed.

        Args:
            job_retention: Delete terminal jobs whose finished_at is older than this.
            worker_retention: Delete inactive/unhealthy workers whose last heartbeat is older than this.
            txn_action_retention: Delete txn_actions older than this.
            profile_retention: Delete task_profiles older than this.
            stop_event: If set, abort early (e.g. during shutdown).
            pause_between_s: Sleep between individual deletes to reduce lock contention.
        """
        now_ms = Timestamp.now().epoch_ms()
        job_cutoff_ms = now_ms - job_retention.to_ms()
        worker_cutoff_ms = now_ms - worker_retention.to_ms()
        txn_cutoff_ms = now_ms - txn_action_retention.to_ms()

        terminal_states = tuple(TERMINAL_JOB_STATES)
        placeholders = ",".join("?" * len(terminal_states))

        def _stopped() -> bool:
            return stop_event is not None and stop_event.is_set()

        # 1. Jobs: one at a time (CASCADE to tasks → attempts, endpoints)
        jobs_deleted = 0
        while not _stopped():
            with self._db.read_snapshot() as snap:
                row = snap.fetchone(
                    f"SELECT job_id FROM jobs WHERE state IN ({placeholders})"
                    " AND finished_at_ms IS NOT NULL AND finished_at_ms < ? LIMIT 1",
                    (*terminal_states, job_cutoff_ms),
                )
            if row is None:
                break
            job_id = row["job_id"]
            with self._db.transaction() as cur:
                # Invalidate endpoint cache BEFORE the CASCADE so the registry
                # drops rows SQLite is about to delete for us.
                self._db.endpoints.remove_by_job_ids(cur, [JobName.from_wire(str(job_id))])
                cur.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                self._record_transaction(cur, "prune_old_data", [("job_pruned", str(job_id), {})])
            jobs_deleted += 1
            time.sleep(pause_between_s)

        # 2. Workers: one at a time (CASCADE to attributes, task_history, resource_history)
        workers_deleted = 0
        while not _stopped():
            with self._db.read_snapshot() as snap:
                row = snap.fetchone(
                    "SELECT worker_id FROM workers WHERE (active = 0 OR healthy = 0) AND last_heartbeat_ms < ? LIMIT 1",
                    (worker_cutoff_ms,),
                )
            if row is None:
                break
            worker_id = row["worker_id"]
            with self._db.transaction() as cur:
                _remove_worker(cur, str(worker_id))
                self._record_transaction(cur, "prune_old_data", [("worker_pruned", str(worker_id), {})])
            workers_deleted += 1
            time.sleep(pause_between_s)

        # 3. txn_actions: batch of 1000 per transaction (no CASCADE)
        txn_actions_deleted = self._batch_delete(
            "DELETE FROM txn_actions WHERE rowid IN "
            "(SELECT rowid FROM txn_actions WHERE created_at_ms < ? LIMIT 1000)",
            (txn_cutoff_ms,),
            _stopped,
            pause_between_s,
        )

        # 4. Task profiles: batch of 1000 per transaction
        profile_cutoff_ms = now_ms - profile_retention.to_ms()
        # 4a. Delete stale profiles by age.
        profiles_deleted = self._batch_delete(
            "DELETE FROM profiles.task_profiles WHERE rowid IN "
            "(SELECT rowid FROM profiles.task_profiles WHERE captured_at_ms < ? LIMIT 1000)",
            (profile_cutoff_ms,),
            _stopped,
            pause_between_s,
        )
        # 4b. Delete orphan profiles whose task no longer exists.
        profiles_deleted += self._batch_delete(
            "DELETE FROM profiles.task_profiles WHERE rowid IN "
            "(SELECT p.rowid FROM profiles.task_profiles p"
            " LEFT JOIN tasks t ON p.task_id = t.task_id"
            " WHERE t.task_id IS NULL LIMIT 1000)",
            (),
            _stopped,
            pause_between_s,
        )

        result = PruneResult(
            jobs_deleted=jobs_deleted,
            workers_deleted=workers_deleted,
            txn_actions_deleted=txn_actions_deleted,
            profiles_deleted=profiles_deleted,
        )
        if result.total > 0:
            logger.info(
                "Pruned old data: %d jobs, %d workers, %d txn_actions, %d profiles",
                result.jobs_deleted,
                result.workers_deleted,
                result.txn_actions_deleted,
                result.profiles_deleted,
            )
            self._db.optimize()

        return result

    # =========================================================================
    # Split Heartbeat Helpers
    # =========================================================================

    def touch_worker_liveness(self, worker_ids: Sequence[WorkerId]) -> None:
        """Cheap liveness bump: update last_heartbeat_ms without rewriting resources."""
        if not worker_ids:
            return
        now_ms = Timestamp.now().epoch_ms()
        with self._db.transaction() as cur:
            cur.executemany(
                "UPDATE workers SET last_heartbeat_ms = ? WHERE worker_id = ?",
                [(now_ms, str(wid)) for wid in worker_ids],
            )

    def update_worker_ping_success(self, worker_id: WorkerId, resource_snapshot: job_pb2.WorkerResourceSnapshot) -> None:
        """Update worker timestamp and resource snapshot from a successful ping.

        Does not reset consecutive_failures — the ping loop tracks failures in-memory.
        """
        snapshot_bytes = resource_snapshot.SerializeToString()
        now_ms = Timestamp.now().epoch_ms()
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE workers SET last_heartbeat_ms = ?, resource_snapshot_proto = ? WHERE worker_id = ?",
                (now_ms, snapshot_bytes, str(worker_id)),
            )
            cur.execute(
                "INSERT INTO worker_resource_history(worker_id, snapshot_proto, timestamp_ms) VALUES (?, ?, ?)",
                (str(worker_id), snapshot_bytes, now_ms),
            )

    def get_running_tasks_for_poll(
        self,
    ) -> tuple[dict[WorkerId, list[RunningTaskEntry]], dict[WorkerId, str]]:
        """Snapshot running tasks and worker addresses for PollTasks RPCs.

        Returns (running_by_worker, worker_addresses) where running_by_worker
        maps worker_id to its list of running task entries and worker_addresses
        maps worker_id to its RPC address.
        """
        with self._db.read_snapshot() as snap:
            worker_rows = snap.fetchall("SELECT worker_id, address FROM workers WHERE active = 1 AND healthy = 1")
            worker_addresses: dict[WorkerId, str] = {}
            worker_ids: list[str] = []
            for row in worker_rows:
                wid = WorkerId(str(row["worker_id"]))
                worker_addresses[wid] = str(row["address"])
                worker_ids.append(str(row["worker_id"]))

            if not worker_ids:
                return {}, {}

            placeholders = ",".join("?" for _ in worker_ids)
            task_rows = snap.fetchall(
                f"SELECT t.task_id, t.current_attempt_id, t.current_worker_id "
                f"FROM tasks t "
                f"WHERE t.current_worker_id IN ({placeholders}) AND t.state IN (?, ?, ?) "
                f"ORDER BY t.task_id ASC",
                (*worker_ids, *ACTIVE_TASK_STATES),
            )

        running: dict[WorkerId, list[RunningTaskEntry]] = {}
        for row in task_rows:
            wid = WorkerId(str(row["current_worker_id"]))
            entry = RunningTaskEntry(
                task_id=JobName.from_wire(str(row["task_id"])),
                attempt_id=int(row["current_attempt_id"]),
            )
            running.setdefault(wid, []).append(entry)
        return running, worker_addresses

    # =========================================================================
    # Heartbeat Dispatch API
    # =========================================================================

    def buffer_dispatch(self, worker_id: WorkerId, task_request: job_pb2.RunTaskRequest) -> None:
        """Buffer a task dispatch for the next heartbeat.

        Called by the scheduling thread after committing resources via TaskAssignedEvent.
        The dispatch will be delivered when begin_heartbeat() drains the buffer.
        """
        with self._db.transaction() as cur:
            enqueue_run_dispatch(cur, str(worker_id), task_request.SerializeToString(), Timestamp.now().epoch_ms())

    def buffer_kill(self, worker_id: WorkerId, task_id: str) -> None:
        """Buffer a task kill for the next heartbeat.

        Called when a task needs to be terminated on a worker. The kill will be
        delivered when begin_heartbeat() drains the buffer.
        """
        with self._db.transaction() as cur:
            enqueue_kill_dispatch(cur, str(worker_id), task_id, Timestamp.now().epoch_ms())

    def begin_heartbeat(self, worker_id: WorkerId) -> DispatchBatch | None:
        """Drain dispatch for a worker and snapshot expected running attempts."""
        return self.drain_dispatch(worker_id)

    def complete_heartbeat(
        self,
        snapshot: DispatchBatch,
        response: job_pb2.HeartbeatResponse,
    ) -> HeartbeatApplyResult:
        """Process successful heartbeat response (phase 3, success path).

        Preconditions:
            - snapshot was returned by begin_heartbeat for this worker
            - response is the worker's HeartbeatResponse
        Postconditions:
            - worker.healthy = True, consecutive_failures = 0
            - Task states updated from worker reports (BUILDING, RUNNING, terminal)
            - Terminal tasks trigger retry/cleanup via the normal state machine
            - Worker resource metrics updated

        Updates worker health state and processes task state changes from the response.
        Log entries are collected under the state lock but flushed to SQLite after
        the lock is released, so disk I/O does not block scheduling or RPCs.

        If the worker reports itself as unhealthy (worker_healthy=False), the worker
        is immediately failed and WORKER_FAILED is returned.
        """
        updates: list[TaskUpdate] = []
        for entry in response.tasks:
            if entry.state in (job_pb2.TASK_STATE_UNSPECIFIED, job_pb2.TASK_STATE_PENDING):
                continue
            updates.append(
                TaskUpdate(
                    task_id=JobName.from_wire(entry.task_id),
                    attempt_id=entry.attempt_id,
                    new_state=entry.state,
                    error=entry.error or None,
                    exit_code=entry.exit_code if entry.HasField("exit_code") else None,
                    resource_usage=entry.resource_usage if entry.resource_usage.ByteSize() > 0 else None,
                    container_id=entry.container_id or None,
                )
            )
        result = self.apply_heartbeat(
            HeartbeatApplyRequest(
                worker_id=snapshot.worker_id,
                worker_resource_snapshot=(
                    response.resource_snapshot if response.resource_snapshot.ByteSize() > 0 else None
                ),
                updates=updates,
            )
        )
        if result.action != HeartbeatAction.OK:
            return result
        # Check if the worker explicitly reported itself as unhealthy.
        # We intentionally use force_remove=False here: a self-reported unhealthy
        # worker still goes through the consecutive-failure threshold so that
        # transient health-check flaps don't cause immediate eviction.
        if not response.worker_healthy:
            health_error = response.health_error or "worker reported unhealthy"
            logger.warning("Worker %s reported unhealthy: %s", snapshot.worker_id, health_error)
            failure = self.fail_heartbeat_for_worker(
                worker_id=snapshot.worker_id,
                error=health_error,
                snapshot=DispatchBatch(
                    worker_id=snapshot.worker_id,
                    worker_address=snapshot.worker_address,
                    running_tasks=snapshot.running_tasks,
                ),
                force_remove=False,
            )
            return HeartbeatApplyResult(tasks_to_kill=failure.tasks_to_kill, action=failure.action)
        return result

    def fail_heartbeat(self, snapshot: DispatchBatch, error: str) -> HeartbeatAction:
        """Handle heartbeat RPC failure (phase 3, failure path).

        Preconditions:
            - snapshot was returned by begin_heartbeat for this worker
            - The heartbeat RPC failed (timeout, connection refused, etc.)
        Postconditions:
            - worker.consecutive_failures incremented
            - If threshold exceeded: worker pruned, ALL tasks cascade to WORKER_FAILED
            - If worker still healthy: buffered dispatches (tasks_to_run, tasks_to_kill)
              are re-queued for the next heartbeat. We cannot tell whether the worker
              received the previous heartbeat (RPC timeout ≠ delivery failure), so we
              re-send the same RunTaskRequests with the same attempt_ids. If the worker
              did receive them, it will reject re-sends as benign duplicates. If it
              did not, it will start them fresh.

        Note: we intentionally do NOT fire WORKER_FAILED for tasks_to_run here.
        The heartbeat may have timed out on the controller side but the worker may
        still be processing it. Firing WORKER_FAILED would bump the attempt_id; the
        next heartbeat would then carry a higher attempt_id, causing the worker to
        kill the running task and restart it unnecessarily.
        """
        result = self.fail_heartbeat_for_worker(snapshot.worker_id, error, snapshot)
        return result.action

    def fail_workers_batch(
        self,
        worker_ids: list[str],
        reason: str,
    ) -> WorkerFailureBatchResult:
        """Fail all active workers matching the given worker IDs in one transaction.

        Used for slice reaping: when one worker on a multi-VM slice fails, all
        sibling workers on that slice must be failed immediately rather than
        waiting for individual heartbeat timeouts.
        """
        if not worker_ids:
            return WorkerFailureBatchResult()
        target_set = sorted(set(worker_ids))
        placeholders = ",".join("?" for _ in target_set)
        with self._db.read_snapshot() as snap:
            rows = WORKER_DETAIL_PROJECTION.decode(
                snap.fetchall(
                    f"SELECT * FROM workers WHERE active = 1 AND worker_id IN ({placeholders})",
                    tuple(target_set),
                )
            )
        failures = [
            (
                DispatchBatch(
                    worker_id=row.worker_id,
                    worker_address=row.address,
                    running_tasks=[],
                ),
                reason,
            )
            for row in rows
        ]
        if not failures:
            return WorkerFailureBatchResult()
        results = self.fail_heartbeats_batch(failures, force_remove=True)
        return WorkerFailureBatchResult(
            tasks_to_kill=results.tasks_to_kill,
            task_kill_workers=results.task_kill_workers,
            removed_workers=[(wid, addr) for wid, addr in results.removed_workers if addr is not None],
            results=results.results,
        )

    def load_workers_from_config(self, configs: list[WorkerConfig]) -> None:
        """Load workers from static configuration."""
        now = Timestamp.now()
        for cfg in configs:
            self.register_or_refresh_worker(
                worker_id=WorkerId(cfg.worker_id),
                address=cfg.address,
                metadata=cfg.metadata,
                ts=now,
            )

    # --- Endpoint Management ---

    def add_endpoint(self, endpoint: EndpointRow) -> bool:
        """Add an endpoint row through the endpoint registry.

        Returns True if the endpoint was inserted, False if the task is already
        terminal (to prevent orphaned endpoints that would never be cleaned up).
        """
        with self._db.transaction() as cur:
            return self._db.endpoints.add(cur, endpoint)

    def remove_endpoint(self, endpoint_id: str) -> EndpointRow | None:
        with self._db.transaction() as cur:
            return self._db.endpoints.remove(cur, endpoint_id)

    # ---------------------------------------------------------------------
    # Test-only SQL mutation helpers
    # ---------------------------------------------------------------------

    def set_worker_health_for_test(self, worker_id: WorkerId, healthy: bool) -> None:
        """Test helper: set worker health in DB."""
        self._db.execute(
            "UPDATE workers SET healthy = ?, consecutive_failures = ? WHERE worker_id = ?",
            (1 if healthy else 0, 0 if healthy else 1, str(worker_id)),
        )

    def set_worker_attribute_for_test(self, worker_id: WorkerId, key: str, value: AttributeValue) -> None:
        """Test helper: upsert one worker attribute in DB."""
        str_value = int_value = float_value = None
        value_type = "str"
        if isinstance(value.value, int):
            value_type = "int"
            int_value = int(value.value)
        elif isinstance(value.value, float):
            value_type = "float"
            float_value = float(value.value)
        else:
            str_value = str(value.value)

        self._db.execute(
            "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(worker_id, key) DO UPDATE SET "
            "value_type=excluded.value_type, "
            "str_value=excluded.str_value, "
            "int_value=excluded.int_value, "
            "float_value=excluded.float_value",
            (str(worker_id), key, value_type, str_value, int_value, float_value),
        )

    # =========================================================================
    # Direct provider methods
    # =========================================================================

    def drain_for_direct_provider(
        self,
        max_promotions: int = DIRECT_PROVIDER_PROMOTION_RATE,
    ) -> DirectProviderBatch:
        """Drain pending tasks and snapshot running tasks for a direct provider sync cycle.

        Promotes up to ``max_promotions`` PENDING tasks to ASSIGNED (NULL
        worker_id), builds RunTaskRequest for each, and collects:
        - Newly promoted tasks -> tasks_to_run
        - Already ASSIGNED/BUILDING/RUNNING tasks with NULL worker_id -> running_tasks
        - Kill entries with NULL worker_id -> tasks_to_kill (deleted from queue)
        """
        with self._db.transaction() as cur:
            now_ms = Timestamp.now().epoch_ms()

            newly_promoted: set[str] = set()
            tasks_to_run: list[job_pb2.RunTaskRequest] = []

            if max_promotions <= 0:
                pending_rows = []
            else:
                pending_rows = cur.execute(
                    "SELECT t.task_id, t.job_id, t.current_attempt_id, j.num_tasks, j.is_reservation_holder, "
                    "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json, "
                    "jc.entrypoint_json, jc.environment_json, jc.bundle_id, jc.ports_json, "
                    "jc.constraints_json, jc.task_image, jc.timeout_ms "
                    f"FROM tasks t JOIN jobs j ON j.job_id = t.job_id {JOB_CONFIG_JOIN} "
                    "WHERE t.state = ? AND j.is_reservation_holder = 0 "
                    "LIMIT ?",
                    (job_pb2.TASK_STATE_PENDING, max_promotions),
                ).fetchall()

            for row in pending_rows:
                task_id = str(row["task_id"])
                attempt_id = int(row["current_attempt_id"]) + 1
                resources = resource_spec_from_scalars(
                    int(row["res_cpu_millicores"]),
                    int(row["res_memory_bytes"]),
                    int(row["res_disk_bytes"]),
                    row["res_device_json"],
                )

                _assign_task(cur, task_id, None, None, attempt_id, now_ms)

                entrypoint = proto_from_json(str(row["entrypoint_json"]), job_pb2.RuntimeEntrypoint)
                # Load inline workdir files from the job_workdir_files table.
                job_id_wire = str(row["job_id"])
                wf_rows = cur.execute(
                    "SELECT filename, data FROM job_workdir_files WHERE job_id = ?",
                    (job_id_wire,),
                ).fetchall()
                for wf_row in wf_rows:
                    entrypoint.workdir_files[wf_row["filename"]] = bytes(wf_row["data"])

                run_req = job_pb2.RunTaskRequest(
                    task_id=task_id,
                    num_tasks=int(row["num_tasks"]),
                    entrypoint=entrypoint,
                    environment=proto_from_json(str(row["environment_json"]), job_pb2.EnvironmentConfig),
                    bundle_id=str(row["bundle_id"]),
                    resources=resources,
                    ports=json.loads(str(row["ports_json"])),
                    attempt_id=attempt_id,
                    constraints=[c.to_proto() for c in constraints_from_json(row["constraints_json"])],
                    task_image=str(row["task_image"]),
                )
                # Propagate timeout for K8s activeDeadlineSeconds (Kubernetes-native enforcement).
                timeout_ms = row["timeout_ms"]
                if timeout_ms is not None and int(timeout_ms) > 0:
                    run_req.timeout.milliseconds = int(timeout_ms)
                tasks_to_run.append(run_req)
                newly_promoted.add(task_id)

            # Snapshot already-running tasks with NULL worker_id (excluding newly promoted).
            active_states = tuple(sorted(ACTIVE_TASK_STATES))
            placeholders = ",".join("?" * len(active_states))
            running_rows = cur.execute(
                "SELECT t.task_id, t.current_attempt_id "
                "FROM tasks t "
                f"WHERE t.current_worker_id IS NULL AND t.state IN ({placeholders}) "
                "ORDER BY t.task_id ASC",
                active_states,
            ).fetchall()
            running_tasks = [
                RunningTaskEntry(
                    task_id=JobName.from_wire(str(row["task_id"])),
                    attempt_id=int(row["current_attempt_id"]),
                )
                for row in running_rows
                if str(row["task_id"]) not in newly_promoted
            ]

            # Drain kill entries with NULL worker_id.
            kill_rows = cur.execute(
                "SELECT task_id FROM dispatch_queue WHERE worker_id IS NULL AND kind = 'kill'",
            ).fetchall()
            tasks_to_kill = [str(row["task_id"]) for row in kill_rows if row["task_id"] is not None]
            if kill_rows:
                cur.execute("DELETE FROM dispatch_queue WHERE worker_id IS NULL AND kind = 'kill'")

        return DirectProviderBatch(
            tasks_to_run=tasks_to_run,
            running_tasks=running_tasks,
            tasks_to_kill=tasks_to_kill,
        )

    def apply_direct_provider_updates(self, updates: list[TaskUpdate]) -> TxResult:
        """Apply a batch of task state updates from a KubernetesProvider.

        Same state machine as apply_task_updates but without worker lookup,
        health updates, or resource decommit (no committed resources tracked).
        """
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}

        with self._db.transaction() as cur:
            now_ms = Timestamp.now().epoch_ms()
            cascaded_jobs: set[JobName] = set()

            for update in updates:
                task_row = cur.execute("SELECT * FROM tasks WHERE task_id = ?", (update.task_id.to_wire(),)).fetchone()
                if task_row is None:
                    continue
                task = TASK_DETAIL_PROJECTION.decode_one([task_row])
                if task_row_is_finished(task) or update.new_state in (
                    job_pb2.TASK_STATE_UNSPECIFIED,
                    job_pb2.TASK_STATE_PENDING,
                ):
                    continue
                if update.attempt_id != int(task_row["current_attempt_id"]):
                    stale = cur.execute(
                        "SELECT state FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
                        (update.task_id.to_wire(), update.attempt_id),
                    ).fetchone()
                    if stale is not None and int(stale["state"]) not in TERMINAL_TASK_STATES:
                        logger.error(
                            "Stale attempt precondition violation: task=%s reported=%d current=%d stale_state=%s",
                            update.task_id,
                            update.attempt_id,
                            int(task_row["current_attempt_id"]),
                            int(stale["state"]),
                        )
                    continue
                attempt_row = cur.execute(
                    "SELECT * FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
                    (update.task_id.to_wire(), update.attempt_id),
                ).fetchone()
                if attempt_row is None:
                    continue
                # See _apply_task_transitions for rationale: the current attempt may
                # be terminal while the task is retrying in PENDING; late reports
                # must not revive it.
                if int(attempt_row["state"]) in TERMINAL_TASK_STATES:
                    logger.debug(
                        "Dropping late update for terminal attempt: task=%s attempt=%d attempt_state=%d reported=%d",
                        update.task_id,
                        update.attempt_id,
                        int(attempt_row["state"]),
                        int(update.new_state),
                    )
                    continue

                if update.resource_usage is not None:
                    ru = update.resource_usage
                    cur.execute(
                        "INSERT INTO task_resource_history"
                        "(task_id, attempt_id, cpu_millicores, memory_mb, disk_mb, memory_peak_mb, timestamp_ms) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            update.task_id.to_wire(),
                            update.attempt_id,
                            ru.cpu_millicores,
                            ru.memory_mb,
                            ru.disk_mb,
                            ru.memory_peak_mb,
                            now_ms,
                        ),
                    )
                if update.container_id is not None:
                    cur.execute(
                        "UPDATE tasks SET container_id = ? WHERE task_id = ?",
                        (update.container_id, update.task_id.to_wire()),
                    )

                terminal_ms: int | None = None
                started_ms: int | None = None
                task_state = int(task_row["state"])
                task_error = update.error
                task_exit = update.exit_code
                failure_count = int(task_row["failure_count"])
                preemption_count = int(task_row["preemption_count"])

                if update.new_state == job_pb2.TASK_STATE_RUNNING:
                    started_ms = now_ms
                    task_state = job_pb2.TASK_STATE_RUNNING
                elif update.new_state == job_pb2.TASK_STATE_BUILDING:
                    task_state = job_pb2.TASK_STATE_BUILDING
                elif update.new_state in (
                    job_pb2.TASK_STATE_FAILED,
                    job_pb2.TASK_STATE_WORKER_FAILED,
                    job_pb2.TASK_STATE_KILLED,
                    job_pb2.TASK_STATE_UNSCHEDULABLE,
                    job_pb2.TASK_STATE_SUCCEEDED,
                ):
                    terminal_ms = now_ms
                    task_state = int(update.new_state)
                    if update.new_state == job_pb2.TASK_STATE_SUCCEEDED and task_exit is None:
                        task_exit = 0
                    if update.new_state == job_pb2.TASK_STATE_UNSCHEDULABLE and task_error is None:
                        task_error = "Scheduling timeout exceeded"
                    if update.new_state == job_pb2.TASK_STATE_FAILED:
                        failure_count += 1
                    if (
                        update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
                        and int(task_row["state"]) in EXECUTING_TASK_STATES
                    ):
                        preemption_count += 1
                    # WORKER_FAILED while still ASSIGNED -> retry immediately as PENDING
                    if (
                        update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
                        and int(task_row["state"]) == job_pb2.TASK_STATE_ASSIGNED
                    ):
                        task_state = job_pb2.TASK_STATE_PENDING
                        terminal_ms = None
                    if update.new_state == job_pb2.TASK_STATE_FAILED and failure_count <= int(
                        task_row["max_retries_failure"]
                    ):
                        task_state = job_pb2.TASK_STATE_PENDING
                        terminal_ms = None
                    if (
                        update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
                        and preemption_count <= int(task_row["max_retries_preemption"])
                        and int(task_row["state"]) in EXECUTING_TASK_STATES
                    ):
                        task_state = job_pb2.TASK_STATE_PENDING
                        terminal_ms = None

                # An attempt is terminal whenever the update itself is terminal, even
                # if the TASK rolls back to PENDING for a retry.
                attempt_terminal_ms = now_ms if int(update.new_state) in TERMINAL_TASK_STATES else None

                cur.execute(
                    "UPDATE task_attempts SET state = ?, started_at_ms = COALESCE(started_at_ms, ?), "
                    "finished_at_ms = COALESCE(finished_at_ms, ?), exit_code = COALESCE(?, exit_code), "
                    "error = COALESCE(?, error) WHERE task_id = ? AND attempt_id = ?",
                    (
                        int(update.new_state),
                        started_ms,
                        attempt_terminal_ms,
                        task_exit,
                        update.error,
                        update.task_id.to_wire(),
                        update.attempt_id,
                    ),
                )
                if task_state in ACTIVE_TASK_STATES:
                    cur.execute(
                        "UPDATE tasks SET state = ?, error = COALESCE(?, error), exit_code = COALESCE(?, exit_code), "
                        "started_at_ms = COALESCE(started_at_ms, ?), finished_at_ms = ?, "
                        "failure_count = ?, preemption_count = ? "
                        "WHERE task_id = ?",
                        (
                            task_state,
                            task_error,
                            task_exit,
                            started_ms,
                            terminal_ms,
                            failure_count,
                            preemption_count,
                            update.task_id.to_wire(),
                        ),
                    )
                else:
                    cur.execute(
                        "UPDATE tasks SET state = ?, error = COALESCE(?, error), exit_code = COALESCE(?, exit_code), "
                        "started_at_ms = COALESCE(started_at_ms, ?), finished_at_ms = ?, "
                        "failure_count = ?, preemption_count = ?, "
                        "current_worker_id = NULL, current_worker_address = NULL "
                        "WHERE task_id = ?",
                        (
                            task_state,
                            task_error,
                            task_exit,
                            started_ms,
                            terminal_ms,
                            failure_count,
                            preemption_count,
                            update.task_id.to_wire(),
                        ),
                    )
                jc_row = cur.execute("SELECT * FROM job_config WHERE job_id = ?", (task.job_id.to_wire(),)).fetchone()

                if update.new_state in TERMINAL_TASK_STATES:
                    delete_task_endpoints(cur, self._db.endpoints, update.task_id.to_wire())

                # Coscheduled sibling cascade.
                if jc_row is not None and task_state in FAILURE_TASK_STATES:
                    has_cosched = bool(int(jc_row["has_coscheduling"]))
                    siblings = _find_coscheduled_siblings(cur, task.job_id, update.task_id, has_cosched)
                    job_resources = resource_spec_from_scalars(
                        int(jc_row["res_cpu_millicores"]),
                        int(jc_row["res_memory_bytes"]),
                        int(jc_row["res_disk_bytes"]),
                        jc_row["res_device_json"],
                    )
                    cascade_kill, cascade_workers = _terminate_coscheduled_siblings(
                        cur, self._db.endpoints, siblings, update.task_id, job_resources, now_ms
                    )
                    tasks_to_kill.update(cascade_kill)
                    task_kill_workers.update(cascade_workers)

                if task.job_id not in cascaded_jobs:
                    new_job_state = self._recompute_job_state(cur, task.job_id)
                    if new_job_state in TERMINAL_JOB_STATES:
                        final_tasks_to_kill, final_task_kill_workers = _finalize_terminal_job(
                            cur, self._db.endpoints, task.job_id, new_job_state, now_ms
                        )
                        tasks_to_kill.update(final_tasks_to_kill)
                        task_kill_workers.update(final_task_kill_workers)
                        cascaded_jobs.add(task.job_id)

            if tasks_to_kill or cascaded_jobs:
                actions: list[tuple[str, str, dict[str, object]]] = [("direct_provider_updates_applied", "direct", {})]
                for job_id in cascaded_jobs:
                    actions.append(("job_terminated", job_id.to_wire(), {}))
                self._record_transaction(cur, "apply_direct_provider_updates", actions)

        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def buffer_direct_kill(self, task_id: str) -> None:
        """Buffer a kill request for a direct-provider task.

        Inserts a kill entry into dispatch_queue with worker_id=NULL.
        Drained by drain_for_direct_provider().
        """
        with self._db.transaction() as cur:
            enqueue_kill_dispatch(cur, None, task_id, Timestamp.now().epoch_ms())

    # =========================================================================
    # Test helpers
    # =========================================================================

    def set_worker_consecutive_failures_for_test(self, worker_id: WorkerId, consecutive_failures: int) -> None:
        """Test helper: set worker consecutive failure count in DB."""
        self._db.execute(
            "UPDATE workers SET consecutive_failures = ? WHERE worker_id = ?",
            (consecutive_failures, str(worker_id)),
        )

    def set_task_state_for_test(
        self,
        task_id: JobName,
        state: int,
        *,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        """Test helper: set task state directly in DB."""
        if state in ACTIVE_TASK_STATES:
            self._db.execute(
                "UPDATE tasks SET state = ?, error = ?, exit_code = ? WHERE task_id = ?",
                (state, error, exit_code, task_id.to_wire()),
            )
        else:
            self._db.execute(
                "UPDATE tasks SET state = ?, error = ?, exit_code = ?, "
                "current_worker_id = NULL, current_worker_address = NULL WHERE task_id = ?",
                (state, error, exit_code, task_id.to_wire()),
            )

    def create_attempt_for_test(self, task_id: JobName, worker_id: WorkerId) -> int:
        """Test helper: append a new task_attempt without finalizing prior attempt."""
        task = self._db.fetchone("SELECT current_attempt_id FROM tasks WHERE task_id = ?", (task_id.to_wire(),))
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        worker_row = self._db.fetchone("SELECT address FROM workers WHERE worker_id = ?", (str(worker_id),))
        worker_address = str(worker_row["address"]) if worker_row is not None else str(worker_id)
        next_attempt_id = int(task["current_attempt_id"]) + 1
        now_ms = Timestamp.now().epoch_ms()
        with self._db.transaction() as cur:
            _assign_task(cur, task_id.to_wire(), str(worker_id), worker_address, next_attempt_id, now_ms)
        return next_attempt_id
