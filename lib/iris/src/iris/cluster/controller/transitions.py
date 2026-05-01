# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller state machine: all DB-mutating transitions live here.

Read-only queries do NOT belong here — callers use db.read_snapshot() directly.
"""

import json
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

from rigging.timing import Duration, Timestamp

from iris.cluster.constraints import AttributeValue, Constraint, constraints_from_resources, merge_constraints
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
    NON_TERMINAL_TASK_STATES,
    ControllerDB,
    QuerySnapshot,
    TransactionCursor,
    task_row_can_be_scheduled,
    task_row_is_finished,
)
from iris.cluster.controller.schema import (
    EndpointRow,
    JobDetailRow,
    WorkerDetailRow,
)
from iris.cluster.controller.stores import (
    ActiveTaskRow,
    ControllerStore,
    EndpointStore,
    JobConfigInsertParams,
    JobInsertParams,
    JobStore,
    TaskAttemptStore,
    TaskAttemptUpdateParams,
    TaskInsertParams,
    TaskScope,
    TaskStateUpdateParams,
    TaskStore,
    WorkerAttributeParams,
    WorkerStore,
    WorkerUpsertParams,
)
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import (
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    JobName,
    WorkerId,
    get_gpu_count,
    get_tpu_count,
)
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_from_proto

logger = logging.getLogger(__name__)


def log_event(
    action: str,
    entity_id: str,
    *,
    trigger: str | None = None,
    **details: object,
) -> None:
    """Emit a semi-structured audit line for a controller state transition.

    Each call produces one ``logger.info`` line of the shape::

        event=<action> entity=<entity_id> trigger=<trigger> k=v ...

    ``trigger`` names the upstream event when this is derived (e.g.
    ``trigger=heartbeat_applied`` on cascaded job terminations); callers omit
    it for externally-caused events and the line renders ``trigger=-``.

    These lines are captured by the Iris log server and queried via the normal
    ``iris process logs`` / log-store DuckDB interface — there is no SQLite
    audit table.
    """
    extras = " ".join(f"{k}={v}" for k, v in details.items() if v is not None)
    logger.info(
        "event=%s entity=%s trigger=%s %s",
        action,
        entity_id,
        trigger or "-",
        extras,
    )


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


FAIL_WORKERS_CHUNK_SIZE = 10
"""Number of worker failures processed per transaction in ``fail_workers``.
Commits between chunks so the SQLite writer is released and other RPCs
(RegisterEndpoint, Register, LaunchJob, apply_heartbeats_batch) can
interleave. Keeps worst-case writer-hold below ~1s even when a zone-wide
failure removes hundreds of workers."""

HEARTBEAT_STALENESS_THRESHOLD = Duration.from_seconds(900)
"""If a worker's last successful heartbeat is older than this, it is failed
immediately. Catches workers restored from a checkpoint whose backing VMs
no longer exist — without this, the controller would need 10 consecutive
RPC failures (50s) per worker to notice, during which they appear healthy
in the dashboard and block scheduling capacity."""

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
    profiles_deleted: int = 0

    @property
    def total(self) -> int:
        return self.jobs_deleted + self.workers_deleted + self.profiles_deleted


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
class WorkerFailureResult(TxResult):
    worker_removed: bool = False
    last_contact_age_ms: int | None = None


@dataclass(frozen=True)
class WorkerFailureBatchResult(TxResult):
    """Result of applying a batch of worker failures."""

    removed_workers: list[tuple[WorkerId, str | None]] = field(default_factory=list)
    results: list[WorkerFailureResult] = field(default_factory=list)


class RunningTaskEntry(NamedTuple):
    """Task ID and attempt ID pair captured at snapshot time."""

    task_id: JobName
    attempt_id: int


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

    No worker_id — tasks run without a registered worker daemon.
    task_attempts rows use NULL worker_id.
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


def delete_task_endpoints(cur: TransactionCursor, endpoints: EndpointStore, task_id: str) -> None:
    """Remove all registered endpoints for a task through the endpoint store."""
    endpoints.remove_by_task(cur, JobName.from_wire(task_id))


def _remove_worker(cur: TransactionCursor, workers: WorkerStore, worker_id: WorkerId) -> None:
    """Remove a worker and sever all its foreign-key references.

    Must be called inside an existing transaction. The four statements
    enforce the multi-table invariant: no dangling worker_id references
    remain in task_attempts, tasks, or dispatch_queue after the worker
    row is deleted.
    """
    workers.remove(cur, worker_id)


def _terminate_task(
    cur: TransactionCursor,
    attempts: TaskAttemptStore,
    tasks: TaskStore,
    workers: WorkerStore,
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
        attempts.mark_finished(
            cur,
            JobName.from_wire(task_id),
            attempt_id,
            effective_attempt_state,
            now_ms,
            error,
        )

    tasks.mark_terminal(
        cur,
        JobName.from_wire(task_id),
        state,
        error,
        finished_at_ms,
        failure_count=failure_count,
        preemption_count=preemption_count,
        active_states=ACTIVE_TASK_STATES,
    )

    delete_task_endpoints(cur, registry, task_id)

    if worker_id is not None and resources is not None:
        workers.decommit_resources(cur, WorkerId(worker_id), resources)


def _kill_non_terminal_tasks(
    cur: TransactionCursor,
    attempts: TaskAttemptStore,
    tasks: TaskStore,
    workers: WorkerStore,
    registry,
    job_id: JobName,
    reason: str,
    now_ms: int,
) -> tuple[set[JobName], dict[JobName, WorkerId]]:
    """Kill all non-terminal tasks for a single job, decommit resources, and delete endpoints."""
    rows = tasks.list_active(
        cur,
        TaskScope(job_id=job_id),
        states=NON_TERMINAL_TASK_STATES,
    )
    tasks_to_kill: set[JobName] = set()
    task_kill_workers: dict[JobName, WorkerId] = {}
    for row in rows:
        task_name = row.task_id
        task_id = task_name.to_wire()
        worker_id = row.current_worker_id
        is_reservation_holder = row.is_reservation_holder
        decommit_worker: str | None = None
        decommit_resources = None
        if worker_id is not None:
            task_kill_workers[task_name] = worker_id
            # Reservation holders never commit resources on assignment,
            # so they must not decommit on termination —
            # otherwise we subtract chips that were never added, which floors
            # committed_* below a co-tenant's legitimate reservation and lets
            # the scheduler double-book the worker.
            if not is_reservation_holder:
                decommit_worker = str(worker_id)
                decommit_resources = row.resources
        _terminate_task(
            cur,
            attempts,
            tasks,
            workers,
            registry,
            task_id,
            row.current_attempt_id,
            job_pb2.TASK_STATE_KILLED,
            reason,
            now_ms,
            worker_id=decommit_worker,
            resources=decommit_resources,
        )
        tasks_to_kill.add(task_name)
    return tasks_to_kill, task_kill_workers


def _cascade_children(
    cur: TransactionCursor,
    store: ControllerStore,
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

    descendants = store.jobs.list_descendants(
        cur,
        job_id,
        exclude_reservation_holders=exclude_reservation_holders,
    )
    for child_job_id in descendants:
        child_tasks_to_kill, child_task_kill_workers = _kill_non_terminal_tasks(
            cur,
            store.attempts,
            store.tasks,
            store.workers,
            store.endpoints,
            child_job_id,
            reason,
            now_ms,
        )
        tasks_to_kill.update(child_tasks_to_kill)
        task_kill_workers.update(child_task_kill_workers)
        store.jobs.update_state_if_not_terminal(cur, child_job_id, job_pb2.JOB_STATE_KILLED, reason, now_ms)
    return tasks_to_kill, task_kill_workers


def _cascade_terminal_job(
    cur: TransactionCursor,
    store: ControllerStore,
    job_id: JobName,
    now_ms: int,
    reason: str,
) -> tuple[set[JobName], dict[JobName, WorkerId]]:
    """Kill remaining tasks and descendant jobs when a job reaches a terminal state."""
    tasks_to_kill, task_kill_workers = _kill_non_terminal_tasks(
        cur,
        store.attempts,
        store.tasks,
        store.workers,
        store.endpoints,
        job_id,
        reason,
        now_ms,
    )
    child_tasks_to_kill, child_task_kill_workers = _cascade_children(cur, store, job_id, now_ms, reason)
    tasks_to_kill.update(child_tasks_to_kill)
    task_kill_workers.update(child_task_kill_workers)
    return tasks_to_kill, task_kill_workers


def _find_coscheduled_siblings(
    tx: "TransactionCursor",
    tasks: TaskStore,
    job_id: JobName,
    exclude_task_id: JobName,
    has_coscheduling: bool,
) -> list[ActiveTaskRow]:
    """Find active siblings in a coscheduled job (read-only)."""
    if not has_coscheduling:
        return []
    return tasks.list_active(
        tx,
        TaskScope(job_id=job_id),
        states=ACTIVE_TASK_STATES,
        exclude_task_id=exclude_task_id,
    )


def _terminate_coscheduled_siblings(
    cur: TransactionCursor,
    attempts: TaskAttemptStore,
    tasks: TaskStore,
    workers: WorkerStore,
    registry,
    siblings: Iterable[ActiveTaskRow],
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
        worker_id_str = str(sib.current_worker_id) if sib.current_worker_id is not None else None
        _terminate_task(
            cur,
            attempts,
            tasks,
            workers,
            registry,
            sib.task_id.to_wire(),
            sib.current_attempt_id,
            job_pb2.TASK_STATE_WORKER_FAILED,
            error,
            now_ms,
            worker_id=worker_id_str,
            resources=resources if sib.current_worker_id is not None else None,
            preemption_count=sib.max_retries_preemption + 1,
        )
        if sib.current_worker_id is not None:
            task_kill_workers[sib.task_id] = sib.current_worker_id
        tasks_to_kill.add(sib.task_id)

    return tasks_to_kill, task_kill_workers


def _requeue_coscheduled_siblings(
    cur: TransactionCursor,
    attempts: TaskAttemptStore,
    tasks: TaskStore,
    workers: WorkerStore,
    registry,
    siblings: Iterable[ActiveTaskRow],
    failed_task_id: JobName,
    resources: "job_pb2.ResourceSpecProto",
    now_ms: int,
) -> tuple[set[JobName], dict[JobName, WorkerId]]:
    """Bounce coscheduled siblings to PENDING so the job re-coschedules atomically.

    Used when one task of a coscheduled job hits a transient failure that will
    retry. Without this, only the failed task returns to PENDING and the
    scheduler may place its retry on a different slice from the still-RUNNING
    siblings, splitting the coscheduled job. Sibling preemption_count and
    failure_count are left untouched — they didn't actually fail, so only the
    originally-failing task pays its retry budget.

    Reservation-holder siblings are skipped; they never hold worker resources
    and don't participate in the slice.
    """
    tasks_to_kill: set[JobName] = set()
    task_kill_workers: dict[JobName, WorkerId] = {}
    error = f"Coscheduled sibling {failed_task_id.to_wire()} bounced for atomic re-scheduling"

    for sib in siblings:
        if sib.is_reservation_holder:
            continue
        worker_id_str = str(sib.current_worker_id) if sib.current_worker_id is not None else None
        _terminate_task(
            cur,
            attempts,
            tasks,
            workers,
            registry,
            sib.task_id.to_wire(),
            sib.current_attempt_id,
            job_pb2.TASK_STATE_PENDING,
            error,
            now_ms,
            attempt_state=job_pb2.TASK_STATE_PREEMPTED,
            worker_id=worker_id_str,
            resources=resources if sib.current_worker_id is not None else None,
        )
        if sib.current_worker_id is not None:
            task_kill_workers[sib.task_id] = sib.current_worker_id
            tasks_to_kill.add(sib.task_id)

    return tasks_to_kill, task_kill_workers


def _resolve_preemption_policy(jobs: JobStore, cur: TransactionCursor, job_id: JobName) -> int:
    """Resolve the effective preemption policy for a job.

    Defaults: single-task jobs → TERMINATE_CHILDREN, multi-task → PRESERVE_CHILDREN.
    """
    info = jobs.get_preemption_info(cur, job_id)
    if info is None:
        return job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    policy, num_tasks = info
    if policy != job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED:
        return policy
    if num_tasks <= 1:
        return job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    return job_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN


_TERMINAL_STATE_REASONS: dict[int, str] = {
    job_pb2.JOB_STATE_FAILED: "Job exceeded max_task_failures",
    job_pb2.JOB_STATE_KILLED: "Job was terminated.",
    job_pb2.JOB_STATE_UNSCHEDULABLE: "Job could not be scheduled.",
    job_pb2.JOB_STATE_WORKER_FAILED: "Worker failed",
}


def _finalize_terminal_job(
    cur: TransactionCursor,
    store: ControllerStore,
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
    tasks_to_kill, task_kill_workers = _kill_non_terminal_tasks(
        cur,
        store.attempts,
        store.tasks,
        store.workers,
        store.endpoints,
        job_id,
        reason,
        now_ms,
    )
    should_cascade = True
    if terminal_state != job_pb2.JOB_STATE_SUCCEEDED:
        policy = _resolve_preemption_policy(store.jobs, cur, job_id)
        should_cascade = policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    if should_cascade:
        child_tasks_to_kill, child_task_kill_workers = _cascade_children(cur, store, job_id, now_ms, reason)
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
        store: ControllerStore,
        health: WorkerHealthTracker | None = None,
    ):
        self._store = store
        self._health = health or WorkerHealthTracker()

    @property
    def _db(self) -> "ControllerDB":
        """Compatibility shim for tests that read directly via ``state._db``.

        Production code in this module never touches ``_db``; it routes
        through ``self._store`` (transactions, snapshots) or store
        sub-objects. This property exists only so existing tests can continue
        to call ``state._db.snapshot(...)`` for ad-hoc reads.
        """
        return self._store._db

    def _recompute_job_state(self, cur: TransactionCursor, job_id: JobName) -> int | None:
        basis = self._store.jobs.get_recompute_basis(cur, job_id)
        if basis is None:
            return None
        current_state = basis.state
        if current_state in TERMINAL_JOB_STATES:
            return current_state
        counts = self._store.tasks.state_counts_for_job(cur, job_id)
        total = sum(counts.values())
        new_state = current_state
        now_ms = Timestamp.now().epoch_ms()
        if total > 0 and counts.get(job_pb2.TASK_STATE_SUCCEEDED, 0) == total:
            new_state = job_pb2.JOB_STATE_SUCCEEDED
        elif counts.get(job_pb2.TASK_STATE_FAILED, 0) > basis.max_task_failures:
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
        elif basis.started_at_ms is not None:
            # Retries put tasks back into PENDING; keep job running once it has started.
            new_state = job_pb2.JOB_STATE_RUNNING
        elif total > 0:
            new_state = job_pb2.JOB_STATE_PENDING
        if new_state == current_state:
            return new_state
        error = self._store.tasks.first_error_for_job(cur, job_id)
        self._store.jobs.apply_recomputed_state(cur, job_id, new_state, now_ms, error)
        return new_state

    def replace_reservation_claims(self, cur: TransactionCursor, claims: dict[WorkerId, ReservationClaim]) -> None:
        """Replace all reservation claims."""
        self._store.reservations.replace_claims(
            cur,
            {worker_id: (claim.job_id, claim.entry_idx) for worker_id, claim in claims.items()},
        )

    # =========================================================================
    # Command API
    # =========================================================================

    def submit_job(
        self,
        cur: TransactionCursor,
        job_id: JobName,
        request: controller_pb2.Controller.LaunchJobRequest,
        ts: Timestamp,
    ) -> SubmitJobResult:
        """Insert the job row and expand its tasks. Caller owns the transaction."""
        submitted_ms = ts.epoch_ms()
        created_task_ids: list[JobName] = []

        effective_submission_ms = self._store.reservations.next_submission_ms(cur, submitted_ms)

        parent_job_id = job_id.parent.to_wire() if job_id.parent is not None else None
        root_submitted_ms = effective_submission_ms
        if job_id.parent is not None:
            # `launch_job` is responsible for rejecting submissions with a
            # missing parent; if we reach here the parent row must exist.
            parent_root = self._store.jobs.get_root_submitted_at_ms(cur, job_id.parent)
            if parent_root is None:
                raise ValueError(f"Cannot submit job {job_id}: parent {parent_job_id} is absent from the database")
            root_submitted_ms = parent_root

        deadline_epoch_ms: int | None = None
        if request.HasField("scheduling_timeout") and request.scheduling_timeout.milliseconds > 0:
            deadline_epoch_ms = (
                Timestamp.from_ms(effective_submission_ms)
                .add(duration_from_proto(request.scheduling_timeout))
                .epoch_ms()
            )

        self._store.jobs.ensure_user(cur, job_id.user, effective_submission_ms)
        # No user_budgets row is created here: absence means "apply
        # UserBudgetDefaults". Rows exist only for tier seeds from cluster
        # config (see reconcile_user_budget_tiers) and admin overrides via
        # set_user_budget.

        # Resolve priority band: use explicit request value, inherit from parent, or default to INTERACTIVE.
        requested_band = int(request.priority_band)
        if requested_band != job_pb2.PRIORITY_BAND_UNSPECIFIED:
            band_sort_key = requested_band
        elif job_id.parent is not None:
            parent_band = self._store.tasks.get_priority_band_for_job(cur, job_id.parent)
            if parent_band is not None:
                band_sort_key = parent_band
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
        self._store.jobs.insert(
            cur,
            JobInsertParams(
                job_id=job_id,
                user_id=job_id.user,
                parent_job_id=parent_job_id,
                root_job_id=job_id.root_job.to_wire(),
                depth=job_id.depth,
                state=state,
                submitted_at_ms=effective_submission_ms,
                root_submitted_at_ms=root_submitted_ms,
                started_at_ms=None,
                finished_at_ms=finished_ms,
                scheduling_deadline_epoch_ms=deadline_epoch_ms,
                error=validation_error,
                exit_code=None,
                num_tasks=replicas,
                is_reservation_holder=False,
                name=job_name_lower,
                has_reservation=bool(has_reservation),
            ),
        )
        self._store.jobs.insert_config(
            cur,
            JobConfigInsertParams(
                job_id=job_id,
                name=job_name_lower,
                has_reservation=bool(has_reservation),
                res_cpu_millicores=res_cpu,
                res_memory_bytes=res_mem,
                res_disk_bytes=res_disk,
                res_device_json=res_device,
                constraints_json=constraints_json,
                has_coscheduling=bool(has_cosched),
                coscheduling_group_by=cosched_group,
                scheduling_timeout_ms=sched_timeout,
                max_task_failures=max_failures,
                entrypoint_json=entrypoint_json,
                environment_json=environment_json,
                bundle_id=request.bundle_id,
                ports_json=ports_json,
                max_retries_failure=int(request.max_retries_failure),
                max_retries_preemption=int(request.max_retries_preemption),
                timeout_ms=timeout_ms,
                preemption_policy=int(request.preemption_policy),
                existing_job_policy=int(request.existing_job_policy),
                priority_band=int(request.priority_band),
                task_image=request.task_image,
                submit_argv_json=json.dumps(list(request.submit_argv)),
                reservation_json=reservation_json,
                fail_if_exists=bool(request.fail_if_exists),
            ),
        )

        # Store workdir files in separate table.
        self._store.jobs.insert_workdir_files(cur, job_id, dict(request.entrypoint.workdir_files))

        if validation_error is None:
            insertion_base = self._store.jobs.reserve_priority_insertion_base(cur)
            for idx in range(replicas):
                task_id = job_id.task(idx).to_wire()
                created_task_ids.append(JobName.from_wire(task_id))
                self._store.tasks.insert(
                    cur,
                    TaskInsertParams(
                        task_id=JobName.from_wire(task_id),
                        job_id=job_id,
                        task_index=idx,
                        state=job_pb2.TASK_STATE_PENDING,
                        submitted_at_ms=effective_submission_ms,
                        max_retries_failure=int(request.max_retries_failure),
                        max_retries_preemption=int(request.max_retries_preemption),
                        priority_neg_depth=-job_id.depth,
                        priority_root_submitted_ms=root_submitted_ms,
                        priority_insertion=insertion_base + idx,
                        priority_band=band_sort_key,
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
                self._store.jobs.insert(
                    cur,
                    JobInsertParams(
                        job_id=holder_id,
                        user_id=holder_id.user,
                        parent_job_id=job_id.to_wire(),
                        root_job_id=holder_id.root_job.to_wire(),
                        depth=holder_id.depth,
                        state=job_pb2.JOB_STATE_PENDING,
                        submitted_at_ms=effective_submission_ms,
                        root_submitted_at_ms=root_submitted_ms,
                        started_at_ms=None,
                        finished_at_ms=None,
                        scheduling_deadline_epoch_ms=None,
                        error=None,
                        exit_code=None,
                        num_tasks=len(request.reservation.entries),
                        is_reservation_holder=True,
                        name=holder_name_lower,
                        has_reservation=False,
                    ),
                )
                holder_entrypoint_json = entrypoint_to_json(holder_request.entrypoint)
                holder_environment_json = proto_to_json(holder_request.environment)
                self._store.jobs.insert_config(
                    cur,
                    JobConfigInsertParams(
                        job_id=holder_id,
                        name=holder_name_lower,
                        has_reservation=False,
                        res_cpu_millicores=holder_res_cpu,
                        res_memory_bytes=holder_res_mem,
                        res_disk_bytes=holder_res_disk,
                        res_device_json=holder_res_device,
                        constraints_json=holder_constraints_json,
                        has_coscheduling=False,
                        coscheduling_group_by="",
                        scheduling_timeout_ms=None,
                        max_task_failures=0,
                        entrypoint_json=holder_entrypoint_json,
                        environment_json=holder_environment_json,
                        bundle_id="",
                        ports_json="[]",
                        max_retries_failure=0,
                        max_retries_preemption=DEFAULT_MAX_RETRIES_PREEMPTION,
                        timeout_ms=None,
                        preemption_policy=0,
                        existing_job_policy=0,
                        priority_band=0,
                        task_image="",
                    ),
                )
                holder_base = self._store.jobs.reserve_priority_insertion_base(cur)
                for idx in range(len(request.reservation.entries)):
                    created_task_ids.append(holder_id.task(idx))
                    self._store.tasks.insert(
                        cur,
                        TaskInsertParams(
                            task_id=holder_id.task(idx),
                            job_id=holder_id,
                            task_index=idx,
                            state=job_pb2.TASK_STATE_PENDING,
                            submitted_at_ms=effective_submission_ms,
                            max_retries_failure=0,
                            max_retries_preemption=DEFAULT_MAX_RETRIES_PREEMPTION,
                            priority_neg_depth=-holder_id.depth,
                            priority_root_submitted_ms=root_submitted_ms,
                            priority_insertion=holder_base + idx,
                            priority_band=band_sort_key,
                        ),
                    )

        log_event("job_submitted", job_id.to_wire(), num_tasks=replicas, error=validation_error)
        return SubmitJobResult(job_id=job_id, task_ids=created_task_ids)

    def cancel_job(self, cur: TransactionCursor, job_id: JobName, reason: str) -> TxResult:
        """Cancel a job tree and return tasks that need kill RPCs. Caller owns the transaction."""
        subtree = self._store.jobs.list_subtree(cur, job_id)
        if not subtree:
            return TxResult()
        running_rows = self._store.tasks.list_active(
            cur,
            TaskScope(job_subtree=subtree),
            states=ACTIVE_TASK_STATES,
        )
        tasks_to_kill = {row.task_id for row in running_rows}
        task_kill_workers = {
            row.task_id: row.current_worker_id for row in running_rows if row.current_worker_id is not None
        }
        # Decommit resources for each active task on its assigned worker.
        # cancel_job marks tasks as KILLED, but apply_heartbeat skips
        # already-finished tasks (is_finished() check), so the normal
        # heartbeat decommit path never fires for cancelled tasks.
        # Direct-provider tasks have NULL worker_id — skip decommit for them.
        for row in running_rows:
            if row.current_worker_id is not None and not row.is_reservation_holder:
                self._store.workers.decommit_resources(cur, row.current_worker_id, row.resources)
        now_ms = Timestamp.now().epoch_ms()
        self._store.tasks.bulk_kill_non_terminal(cur, subtree, reason, now_ms, TERMINAL_TASK_STATES)
        # Without this, the current attempt row stays state=RUNNING forever
        # (apply_heartbeat skips terminal tasks so the per-attempt finalize
        # never fires). Dashboard queries that read task_attempts then report
        # the killed task as still running on its old worker.
        self._store.attempts.bulk_finalize_active(
            cur, subtree, job_pb2.TASK_STATE_KILLED, reason, now_ms, set(ACTIVE_TASK_STATES)
        )
        # Deliberately excludes JOB_STATE_WORKER_FAILED from the guard set:
        # worker-failed jobs should still be cancellable (transitioned to KILLED).
        cancel_guard_states = TERMINAL_JOB_STATES - {job_pb2.JOB_STATE_WORKER_FAILED}
        self._store.jobs.bulk_update_state(
            cur,
            subtree,
            job_pb2.JOB_STATE_KILLED,
            reason,
            now_ms,
            cancel_guard_states,
        )
        self._store.endpoints.remove_by_job_ids(cur, subtree)
        log_event("job_cancelled", job_id.to_wire(), reason=reason)
        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def register_or_refresh_worker(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        address: str,
        metadata: job_pb2.WorkerMetadata,
        ts: Timestamp,
        slice_id: str = "",
        scale_group: str = "",
    ) -> TxResult:
        """Register a new worker or refresh an existing one. Caller owns the transaction.

        The in-memory worker-attribute scheduling cache and the
        ``worker_registered`` audit line are scheduled as post-commit
        hooks on ``cur``; they fire only if the caller's transaction
        commits.
        """
        attrs: list[WorkerAttributeParams] = []
        for key, proto in metadata.attributes.items():
            value = AttributeValue.from_proto(proto).value
            if isinstance(value, int):
                attrs.append(WorkerAttributeParams(key, "int", None, int(value), None))
            elif isinstance(value, float):
                attrs.append(WorkerAttributeParams(key, "float", None, None, float(value)))
            else:
                attrs.append(WorkerAttributeParams(key, "str", str(value), None, None))
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
        self._store.workers.upsert(
            cur,
            WorkerUpsertParams(
                worker_id=worker_id,
                address=address,
                last_heartbeat_ms=now_ms,
                total_cpu_millicores=metadata.cpu_count * 1000,
                total_memory_bytes=metadata.memory_bytes,
                total_gpu_count=gpu_count,
                total_tpu_count=tpu_count,
                device_type=device_type,
                device_variant=device_variant,
                slice_id=slice_id,
                scale_group=scale_group,
                md_hostname=metadata.hostname,
                md_ip_address=metadata.ip_address,
                md_cpu_count=metadata.cpu_count,
                md_memory_bytes=metadata.memory_bytes,
                md_disk_bytes=metadata.disk_bytes,
                md_tpu_name=metadata.tpu_name,
                md_tpu_worker_hostnames=metadata.tpu_worker_hostnames,
                md_tpu_worker_id=metadata.tpu_worker_id,
                md_tpu_chips_per_host_bounds=metadata.tpu_chips_per_host_bounds,
                md_gpu_count=metadata.gpu_count,
                md_gpu_name=metadata.gpu_name,
                md_gpu_memory_mb=metadata.gpu_memory_mb,
                md_gce_instance_name=metadata.gce_instance_name,
                md_gce_zone=metadata.gce_zone,
                md_git_hash=metadata.git_hash,
                md_device_json=proto_to_json(metadata.device),
            ),
        )
        self._store.workers.replace_attributes(cur, worker_id, attrs)
        # Update in-memory attribute cache only after commit so a rolled-back tx
        # doesn't leave the scheduling cache ahead of the durable row.
        attr_dict: dict[str, AttributeValue] = {}
        for attr in attrs:
            if attr.value_type == "int":
                attr_dict[attr.key] = AttributeValue(int(attr.int_value))
            elif attr.value_type == "float":
                attr_dict[attr.key] = AttributeValue(float(attr.float_value))
            else:
                attr_dict[attr.key] = AttributeValue(str(attr.str_value or ""))
        cur.on_commit(lambda: self._store.workers.update_attr_cache(worker_id, attr_dict))
        cur.on_commit(lambda: log_event("worker_registered", str(worker_id), address=address))
        return TxResult()

    def register_worker(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        address: str,
        metadata: job_pb2.WorkerMetadata,
        ts: Timestamp,
        slice_id: str = "",
        scale_group: str = "",
    ) -> WorkerRegistrationResult:
        self.register_or_refresh_worker(
            cur,
            worker_id=worker_id,
            address=address,
            metadata=metadata,
            ts=ts,
            slice_id=slice_id,
            scale_group=scale_group,
        )
        return WorkerRegistrationResult(worker_id=worker_id)

    def queue_assignments(
        self,
        cur: TransactionCursor,
        assignments: list[Assignment],
        *,
        direct_dispatch: bool = False,
    ) -> AssignmentResult:
        """Commit assignments and enqueue dispatches in one transaction.

        When direct_dispatch=True, collects (worker_id, address, RunTaskRequest)
        tuples in start_requests instead of writing to the dispatch_queue table.
        The caller is responsible for sending StartTasks RPCs.
        """
        accepted: list[Assignment] = []
        rejected: list[Assignment] = []
        start_requests: list[tuple[WorkerId, str, job_pb2.RunTaskRequest]] = []
        has_real_dispatch = False
        now_ms = Timestamp.now().epoch_ms()
        job_cache: dict[str, JobDetailRow] = {}
        jobs_to_update: set[str] = set()
        for assignment in assignments:
            task = self._store.tasks.get_detail(cur, assignment.task_id)
            worker_address = self._store.workers.active_healthy_address(cur, assignment.worker_id)
            if task is None or worker_address is None:
                rejected.append(assignment)
                continue
            if not task_row_can_be_scheduled(task):
                rejected.append(assignment)
                continue
            job_id_wire = task.job_id.to_wire()
            if job_id_wire not in job_cache:
                decoded_job = self._store.jobs.get_detail(cur, task.job_id)
                if decoded_job is None:
                    rejected.append(assignment)
                    continue
                job_cache[job_id_wire] = decoded_job
            job = job_cache[job_id_wire]
            attempt_id = task.current_attempt_id + 1
            self._store.tasks.assign(
                cur,
                self._store.attempts,
                assignment.task_id,
                assignment.worker_id,
                worker_address,
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
                self._store.workers.add_committed_resources(cur, assignment.worker_id, resources)
                entrypoint = proto_from_json(job.entrypoint_json, job_pb2.RuntimeEntrypoint)
                # Load inline workdir files from the job_workdir_files table.
                for filename, data in self._store.jobs.get_workdir_files(cur, task.job_id).items():
                    entrypoint.workdir_files[filename] = data
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
                    start_requests.append((assignment.worker_id, worker_address, run_request))
                else:
                    self._store.dispatch.enqueue_run(cur, assignment.worker_id, run_request.SerializeToString(), now_ms)
                has_real_dispatch = True
            self._store.workers.record_task_assignment(cur, assignment.worker_id, assignment.task_id, now_ms)
            jobs_to_update.add(job_id_wire)
            accepted.append(assignment)
        for job_id_wire in jobs_to_update:
            self._store.jobs.mark_running_if_pending(cur, JobName.from_wire(job_id_wire), now_ms)
        for a in accepted:
            log_event("assignment_queued", a.task_id.to_wire(), worker=str(a.worker_id))
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
        existing = self._store.workers.filter_existing(cur, [req.worker_id])
        if str(req.worker_id) not in existing:
            return False
        self._store.workers.apply_snapshots(
            cur,
            [req.worker_id],
            now_ms,
            reset_health=True,
        )
        return True

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
            task = self._store.tasks.get_detail(cur, update.task_id)
            if task is None:
                continue
            if task_row_is_finished(task) or update.new_state in (
                job_pb2.TASK_STATE_UNSPECIFIED,
                job_pb2.TASK_STATE_PENDING,
            ):
                continue
            if update.attempt_id != task.current_attempt_id:
                stale_state = self._store.attempts.get_state(cur, update.task_id, update.attempt_id)
                if stale_state is not None and stale_state not in TERMINAL_TASK_STATES:
                    logger.error(
                        "Stale attempt precondition violation: task=%s reported=%d current=%d stale_state=%s",
                        update.task_id,
                        update.attempt_id,
                        task.current_attempt_id,
                        stale_state,
                    )
                continue

            prior_state = task.state

            # Fast path: task already in the reported state with no new data to apply.
            has_new_data = update.error is not None or update.exit_code is not None or update.resource_usage is not None
            if update.new_state == prior_state and not has_new_data:
                continue

            attempt = self._store.attempts.get(cur, update.task_id, update.attempt_id)
            if attempt is None:
                continue
            # The attempt is already terminal (e.g. preempted, killed) but the task has
            # been rolled back to PENDING for retry and current_attempt_id still points
            # at the dead attempt. Reviving it would produce an inconsistent row where
            # state contradicts finished_at_ms/error.
            if attempt.state in TERMINAL_TASK_STATES:
                logger.debug(
                    "Dropping late update for terminal attempt: task=%s attempt=%d attempt_state=%d reported=%d",
                    update.task_id,
                    update.attempt_id,
                    attempt.state,
                    int(update.new_state),
                )
                continue
            worker_id = attempt.worker_id
            terminal_ms: int | None = None
            started_ms: int | None = None
            task_state = prior_state
            task_error = update.error
            task_exit = update.exit_code
            failure_count = task.failure_count
            preemption_count = task.preemption_count

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
                    # A FAILED originating while the task was still BUILDING almost
                    # always means the worker couldn't pull the image or set up the
                    # runtime (disk full, DNS / registry unreachable, transient
                    # network hiccup). User-code failures only appear once the task
                    # has reached RUNNING. Treat the BUILDING -> FAILED transition
                    if prior_state == job_pb2.TASK_STATE_BUILDING and worker_id is not None:
                        self._health.build_failed(WorkerId(str(worker_id)))
                if update.new_state == job_pb2.TASK_STATE_WORKER_FAILED and prior_state in EXECUTING_TASK_STATES:
                    # A worker that truly died will also miss its next ping/heartbeat
                    # RPC, which bumps the tracker on the observer side. We don't
                    # double-count that signal here.
                    preemption_count += 1
                if update.new_state == job_pb2.TASK_STATE_WORKER_FAILED and prior_state == job_pb2.TASK_STATE_ASSIGNED:
                    task_state = job_pb2.TASK_STATE_PENDING
                    terminal_ms = None
                    # ASSIGNED -> WORKER_FAILED means the worker accepted the task but
                    # couldn't bring it up (e.g. TPU iommu/vfio already held by another
                    # process on the VM). Attribute the failure to the worker so a host
                    # that keeps failing launches gets reaped; otherwise the task loops
                    # forever without draining preemption budget.
                    if worker_id is not None:
                        self._health.build_failed(WorkerId(str(worker_id)))
                if update.new_state == job_pb2.TASK_STATE_FAILED and failure_count <= task.max_retries_failure:
                    task_state = job_pb2.TASK_STATE_PENDING
                    terminal_ms = None
                if (
                    update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
                    and preemption_count <= task.max_retries_preemption
                    and prior_state in EXECUTING_TASK_STATES
                ):
                    task_state = job_pb2.TASK_STATE_PENDING
                    terminal_ms = None

            # An attempt is terminal whenever the update itself is terminal, even
            # if the TASK rolls back to PENDING for a retry. terminal_ms above
            # tracks the task's finished_at_ms; the attempt needs its own stamp.
            attempt_terminal_ms = now_ms if int(update.new_state) in TERMINAL_TASK_STATES else None

            self._store.attempts.apply_update(
                cur,
                TaskAttemptUpdateParams(
                    task_id=update.task_id,
                    attempt_id=update.attempt_id,
                    state=int(update.new_state),
                    started_at_ms=started_ms,
                    finished_at_ms=attempt_terminal_ms,
                    exit_code=task_exit,
                    error=update.error,
                ),
            )
            self._store.tasks.apply_state_update(
                cur,
                TaskStateUpdateParams(
                    task_id=update.task_id,
                    state=task_state,
                    error=task_error,
                    exit_code=task_exit,
                    started_at_ms=started_ms,
                    finished_at_ms=terminal_ms,
                    failure_count=failure_count,
                    preemption_count=preemption_count,
                ),
                ACTIVE_TASK_STATES,
            )

            # Fetch and cache job_config row (avoids re-querying per task in same job).
            job_id_wire = task.job_id.to_wire()
            if job_id_wire not in job_config_cache:
                job_config_cache[job_id_wire] = self._store.jobs.get_config(cur, task.job_id)
            jc = job_config_cache[job_id_wire]

            if worker_id is not None and task_state not in ACTIVE_TASK_STATES:
                if jc is not None:
                    resources = resource_spec_from_scalars(
                        int(jc["res_cpu_millicores"]),
                        int(jc["res_memory_bytes"]),
                        int(jc["res_disk_bytes"]),
                        jc["res_device_json"],
                    )
                    self._store.workers.decommit_resources(cur, worker_id, resources)

            if update.new_state in TERMINAL_TASK_STATES:
                delete_task_endpoints(cur, self._store.endpoints, update.task_id.to_wire())

            # Coscheduled jobs: any failure of a sibling must clear the slice so
            # the job re-coschedules atomically. Branch on whether this task is
            # going terminal (kill siblings outright) or being downgraded to
            # PENDING for retry (requeue siblings, preserving their budgets).
            reported_failure = int(update.new_state) in FAILURE_TASK_STATES
            if jc is not None and bool(int(jc["has_coscheduling"])) and reported_failure:
                siblings = _find_coscheduled_siblings(cur, self._store.tasks, task.job_id, update.task_id, True)
                resources = resource_spec_from_scalars(
                    int(jc["res_cpu_millicores"]),
                    int(jc["res_memory_bytes"]),
                    int(jc["res_disk_bytes"]),
                    jc["res_device_json"],
                )
                if task_state in FAILURE_TASK_STATES:
                    cascade_kill, cascade_workers = _terminate_coscheduled_siblings(
                        cur,
                        self._store.attempts,
                        self._store.tasks,
                        self._store.workers,
                        self._store.endpoints,
                        siblings,
                        update.task_id,
                        resources,
                        now_ms,
                    )
                else:
                    cascade_kill, cascade_workers = _requeue_coscheduled_siblings(
                        cur,
                        self._store.attempts,
                        self._store.tasks,
                        self._store.workers,
                        self._store.endpoints,
                        siblings,
                        update.task_id,
                        resources,
                        now_ms,
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
                    cur, self._store, job_id, new_job_state, now_ms
                )
                tasks_to_kill.update(final_tasks_to_kill)
                task_kill_workers.update(final_task_kill_workers)
                cascaded_jobs.add(job_id)
        if tasks_to_kill or cascaded_jobs:
            log_event("heartbeat_applied", str(req.worker_id))
            for job_id in cascaded_jobs:
                log_event("job_terminated", job_id.to_wire(), trigger="heartbeat_applied")

        return TxResult(
            tasks_to_kill=tasks_to_kill,
            task_kill_workers=task_kill_workers,
        )

    def apply_task_updates(self, cur: TransactionCursor, req: HeartbeatApplyRequest) -> TxResult:
        """Apply a batch of worker task updates within the caller's transaction."""
        now_ms = Timestamp.now().epoch_ms()
        if not self._update_worker_health(cur, req, now_ms):
            return TxResult()
        result = self._apply_task_transitions(cur, req, now_ms)

        return result

    def apply_heartbeats_batch(self, cur: TransactionCursor, requests: list[HeartbeatApplyRequest]) -> list[TxResult]:
        """Apply multiple heartbeats in a single transaction.

        Two-pass architecture to minimise SQL round-trips:

        1. Bulk-fetch all referenced task rows, classify each update as
           *steady-state* (same state, no error/exit_code) or *transition*.
        2a. Batch steady-state resource_usage writes via ``executemany``.
        2b. Feed only transitions through ``_apply_task_transitions``, which
            retains the full state machine (retry, cascade, decommit, etc.).

        Worker health updates are also batched via ``executemany``.
        """
        _empty = TxResult(tasks_to_kill=set())
        results: list[TxResult] = [_empty] * len(requests)

        now_ms = Timestamp.now().epoch_ms()

        # ── Batch worker health updates ───────────────────────────────
        existing_workers = self._store.workers.filter_existing(cur, [req.worker_id for req in requests])
        self._store.workers.apply_snapshots(
            cur,
            [req.worker_id for req in requests if str(req.worker_id) in existing_workers],
            now_ms,
            reset_health=True,
        )

        # ── Bulk-fetch task rows for classification ───────────────────
        all_task_ids: list[JobName] = []
        for req in requests:
            if str(req.worker_id) not in existing_workers:
                continue
            for update in req.updates:
                if update.new_state not in (
                    job_pb2.TASK_STATE_UNSPECIFIED,
                    job_pb2.TASK_STATE_PENDING,
                ):
                    all_task_ids.append(update.task_id)

        task_map = self._store.tasks.bulk_get_detail(cur, all_task_ids)

        # ── Classify and split ────────────────────────────────────────
        # (request_index, transition_request) pairs so results stay aligned.
        transition_entries: list[tuple[int, HeartbeatApplyRequest]] = []

        for req_idx, req in enumerate(requests):
            if str(req.worker_id) not in existing_workers:
                continue

            transition_updates: list[TaskUpdate] = []
            for update in req.updates:
                task = task_map.get(update.task_id)
                if task is None:
                    continue

                prior_state = task.state
                is_state_change = update.new_state != prior_state
                has_terminal_data = update.error is not None or update.exit_code is not None

                if is_state_change or has_terminal_data:
                    transition_updates.append(update)
                # Steady-state resource usage no longer persisted; iris.worker /
                # iris.task stats namespaces own per-tick measurements.

            if transition_updates:
                transition_entries.append(
                    (
                        req_idx,
                        HeartbeatApplyRequest(
                            worker_id=req.worker_id,
                            updates=transition_updates,
                        ),
                    )
                )

        # ── Apply transitions via existing state machine ──────────────
        for req_idx, treq in transition_entries:
            results[req_idx] = self._apply_task_transitions(cur, treq, now_ms)

        return results

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
        task_rows = self._store.tasks.list_active(
            cur,
            TaskScope(worker_id=worker_id),
            states=ACTIVE_TASK_STATES,
        )
        for task_row in task_rows:
            prior_state = task_row.state
            is_reservation_holder = task_row.is_reservation_holder
            if is_reservation_holder:
                new_task_state = job_pb2.TASK_STATE_PENDING
                preemption_count = task_row.preemption_count
            else:
                new_task_state, preemption_count = _resolve_task_failure_state(
                    prior_state,
                    task_row.preemption_count,
                    task_row.max_retries_preemption,
                    job_pb2.TASK_STATE_WORKER_FAILED,
                )
            # Reservation holders retry to PENDING with preemption_count reset so the
            # reservation can re-acquire a worker without counting the failure as a preemption.
            holder_preemption_count = 0 if is_reservation_holder else preemption_count
            task_id = task_row.task_id
            _terminate_task(
                cur,
                self._store.attempts,
                self._store.tasks,
                self._store.workers,
                self._store.endpoints,
                task_id.to_wire(),
                task_row.current_attempt_id,
                new_task_state,
                f"Worker {worker_id} failed: {error}",
                now_ms,
                attempt_state=job_pb2.TASK_STATE_WORKER_FAILED,
                preemption_count=holder_preemption_count,
            )
            parent_job_id, _ = task_id.require_task()
            new_job_state = self._recompute_job_state(cur, parent_job_id)
            if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
                cascaded_tasks_to_kill, cascaded_task_kill_workers = _cascade_terminal_job(
                    cur, self._store, parent_job_id, now_ms, f"Worker {worker_id} failed"
                )
                tasks_to_kill.update(cascaded_tasks_to_kill)
                task_kill_workers.update(cascaded_task_kill_workers)
            elif new_task_state == job_pb2.TASK_STATE_PENDING:
                policy = _resolve_preemption_policy(self._store.jobs, cur, parent_job_id)
                if policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
                    child_tasks_to_kill, child_task_kill_workers = _cascade_children(
                        cur,
                        self._store,
                        parent_job_id,
                        now_ms,
                        "Parent task preempted",
                        exclude_reservation_holders=True,
                    )
                    tasks_to_kill.update(child_tasks_to_kill)
                    task_kill_workers.update(child_task_kill_workers)
            if new_task_state == job_pb2.TASK_STATE_WORKER_FAILED:
                tasks_to_kill.add(task_id)

            # Coscheduled siblings on surviving slice workers must clear so the
            # job re-coschedules atomically; otherwise the bounced task may
            # land on a different slice from its still-RUNNING siblings.
            if not is_reservation_holder and task_row.has_coscheduling:
                siblings = _find_coscheduled_siblings(cur, self._store.tasks, parent_job_id, task_id, True)
                if new_task_state in FAILURE_TASK_STATES:
                    sib_kill, sib_workers = _terminate_coscheduled_siblings(
                        cur,
                        self._store.attempts,
                        self._store.tasks,
                        self._store.workers,
                        self._store.endpoints,
                        siblings,
                        task_id,
                        task_row.resources,
                        now_ms,
                    )
                else:
                    sib_kill, sib_workers = _requeue_coscheduled_siblings(
                        cur,
                        self._store.attempts,
                        self._store.tasks,
                        self._store.workers,
                        self._store.endpoints,
                        siblings,
                        task_id,
                        task_row.resources,
                        now_ms,
                    )
                tasks_to_kill.update(sib_kill)
                task_kill_workers.update(sib_workers)
        _remove_worker(cur, self._store.workers, worker_id)
        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def _record_worker_failure(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        error: str,
        *,
        now_ms: int | None = None,
    ) -> WorkerFailureResult:
        """Remove a failed worker inside an existing transaction."""
        status = self._store.workers.get_active_status(cur, worker_id)
        if status is None:
            return WorkerFailureResult(worker_removed=True)

        now_ms = now_ms or Timestamp.now().epoch_ms()
        last_contact_age_ms = None if status.last_heartbeat_ms is None else max(0, now_ms - status.last_heartbeat_ms)
        self._store.workers.mark_unhealthy(cur, worker_id)
        removal = self._remove_failed_worker(cur, worker_id, error, now_ms=now_ms)
        return WorkerFailureResult(
            tasks_to_kill=removal.tasks_to_kill,
            task_kill_workers=removal.task_kill_workers,
            worker_removed=True,
            last_contact_age_ms=last_contact_age_ms,
        )

    def fail_workers(
        self,
        failures: list[tuple[WorkerId, str | None, str]],
        *,
        chunk_size: int = FAIL_WORKERS_CHUNK_SIZE,
    ) -> WorkerFailureBatchResult:
        """Remove the given active workers and cascade task state in chunks.

        Each ``(worker_id, worker_address, reason)`` tuple triggers a
        worker-removal transaction. Chunks commit between themselves so the
        SQLite writer is released and other RPCs (register, apply_heartbeats_batch,
        ...) can interleave instead of stalling behind a zone-wide failure.
        """
        if not failures:
            return WorkerFailureBatchResult()

        results: list[WorkerFailureResult] = []
        removed_workers: list[tuple[WorkerId, str | None]] = []
        all_tasks_to_kill: set[JobName] = set()
        all_task_kill_workers: dict[JobName, WorkerId] = {}

        for chunk_start in range(0, len(failures), chunk_size):
            chunk = failures[chunk_start : chunk_start + chunk_size]
            with self._store.transaction() as cur:
                now_ms = Timestamp.now().epoch_ms()
                for worker_id, worker_address, error in chunk:
                    result = self._record_worker_failure(
                        cur,
                        worker_id,
                        error,
                        now_ms=now_ms,
                    )
                    results.append(result)
                    log_event(
                        "worker_failed",
                        str(worker_id),
                        address=worker_address or "-",
                        last_contact_age_ms=result.last_contact_age_ms or "-",
                        error=error,
                    )
                    all_tasks_to_kill.update(result.tasks_to_kill)
                    all_task_kill_workers.update(result.task_kill_workers)
                    if result.worker_removed:
                        removed_workers.append((worker_id, worker_address))

        for worker_id, _ in removed_workers:
            self._store.workers.remove_from_attr_cache(worker_id)
        return WorkerFailureBatchResult(
            tasks_to_kill=all_tasks_to_kill,
            task_kill_workers=all_task_kill_workers,
            removed_workers=removed_workers,
            results=results,
        )

    def mark_task_unschedulable(self, cur: TransactionCursor, task_id: JobName, reason: str) -> TxResult:
        """Mark a task as unschedulable using the task transition engine."""
        job_id = self._store.tasks.get_job_id(cur, task_id)
        if job_id is None:
            return TxResult()
        now_ms = Timestamp.now().epoch_ms()
        _terminate_task(
            cur,
            self._store.attempts,
            self._store.tasks,
            self._store.workers,
            self._store.endpoints,
            task_id.to_wire(),
            None,
            job_pb2.TASK_STATE_UNSCHEDULABLE,
            reason,
            now_ms,
        )
        self._recompute_job_state(cur, job_id)
        log_event("task_unschedulable", task_id.to_wire(), reason=reason)
        return TxResult()

    def preempt_task(self, cur: TransactionCursor, task_id: JobName, reason: str) -> TxResult:
        """Preempt a running task, consuming from preemption retry budget.

        Marks the task as PREEMPTED (or retries as PENDING if budget remains),
        decommits its resources from the worker, and cascades to children if needed.
        """
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}
        row = self._store.tasks.get_with_resources(cur, task_id)
        if row is None:
            return TxResult()

        prior_state = row.state
        if prior_state not in ACTIVE_TASK_STATES:
            return TxResult()

        now_ms = Timestamp.now().epoch_ms()
        new_state, preemption_count = _resolve_task_failure_state(
            prior_state,
            row.preemption_count,
            row.max_retries_preemption,
            job_pb2.TASK_STATE_PREEMPTED,
        )
        # Fetch worker_id from the attempt for resource decommit.
        attempt_worker = self._store.attempts.get_worker_id(cur, task_id, row.current_attempt_id)
        attempt_worker_id = str(attempt_worker) if attempt_worker is not None else None
        attempt_resources = row.resources if attempt_worker_id is not None else None

        _terminate_task(
            cur,
            self._store.attempts,
            self._store.tasks,
            self._store.workers,
            self._store.endpoints,
            task_id.to_wire(),
            row.current_attempt_id,
            new_state,
            reason,
            now_ms,
            attempt_state=job_pb2.TASK_STATE_PREEMPTED,
            worker_id=attempt_worker_id,
            resources=attempt_resources,
            preemption_count=preemption_count,
        )

        # Coscheduled retry: bounce siblings back to PENDING so the job
        # re-coschedules atomically. Without this, the lone preempted task's
        # retry can land on a different slice from its still-RUNNING siblings,
        # splitting the SPMD mesh across pods.
        if new_state == job_pb2.TASK_STATE_PENDING and row.has_coscheduling:
            siblings = _find_coscheduled_siblings(cur, self._store.tasks, row.job_id, task_id, True)
            sibling_kills, sibling_workers = _requeue_coscheduled_siblings(
                cur,
                self._store.attempts,
                self._store.tasks,
                self._store.workers,
                self._store.endpoints,
                siblings,
                task_id,
                row.resources,
                now_ms,
            )
            tasks_to_kill.update(sibling_kills)
            task_kill_workers.update(sibling_workers)

        # Recompute job state and cascade if terminal
        job_id = row.job_id
        new_job_state = self._recompute_job_state(cur, job_id)
        if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
            cascade_kills, cascade_workers = _finalize_terminal_job(cur, self._store, job_id, new_job_state, now_ms)
            tasks_to_kill.update(cascade_kills)
            task_kill_workers.update(cascade_workers)
        elif new_state == job_pb2.TASK_STATE_PENDING:
            policy = _resolve_preemption_policy(self._store.jobs, cur, job_id)
            if policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
                child_kills, child_workers = _cascade_children(
                    cur,
                    self._store,
                    job_id,
                    now_ms,
                    reason,
                    exclude_reservation_holders=True,
                )
                tasks_to_kill.update(child_kills)
                task_kill_workers.update(child_workers)

        # Always send a StopTask RPC to the worker that was running this attempt:
        # _terminate_task decommits the worker's resources but does not stop the
        # remote process. Without the kill RPC the worker keeps running the old
        # code while the scheduler reuses the freed slot for a new task,
        # producing two concurrent processes on the same TPU.
        if attempt_worker_id is not None:
            tasks_to_kill.add(task_id)
            task_kill_workers[task_id] = WorkerId(attempt_worker_id)

        log_event("task_preempted", task_id.to_wire(), reason=reason)

        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def cancel_tasks_for_timeout(self, cur: TransactionCursor, task_ids: set[JobName], reason: str) -> TxResult:
        """Mark executing tasks as FAILED due to execution timeout and return kill set.

        Each task is moved to TASK_STATE_FAILED with the given reason.
        Timeouts are hard failures — retry logic is intentionally bypassed.

        Two-phase design: all reads happen before any writes so that
        coscheduled siblings sharing a job are never double-processed from
        stale prefetched rows.
        """
        if not task_ids:
            return TxResult()
        rows = self._store.tasks.list_active(
            cur,
            TaskScope(task_ids=sorted(task_ids, key=lambda tid: tid.to_wire())),
            states=EXECUTING_TASK_STATES,
        )

        # -- Phase 1: read all state before any mutations. --
        now_ms = Timestamp.now().epoch_ms()
        # Resources per job (all direct-timeout tasks in a job share job_config).
        job_resources_cache: dict[str, job_pb2.ResourceSpecProto] = {}
        # Collect directly-timed-out task wires for dedup against siblings.
        direct_task_wires: set[str] = set()
        # Per-job list of siblings to cascade (collected across all timed-out tasks).
        siblings_by_job: dict[str, list[ActiveTaskRow]] = {}

        for row in rows:
            task_id_wire = row.task_id.to_wire()
            direct_task_wires.add(task_id_wire)
            job_id_wire = row.job_id.to_wire()
            if job_id_wire not in job_resources_cache:
                job_resources_cache[job_id_wire] = row.resources
            siblings = _find_coscheduled_siblings(cur, self._store.tasks, row.job_id, row.task_id, row.has_coscheduling)
            if siblings:
                existing = siblings_by_job.get(job_id_wire, [])
                existing.extend(siblings)
                siblings_by_job[job_id_wire] = existing

        # Deduplicate siblings: drop any that will already be terminated
        # directly as timed-out tasks, and deduplicate across multiple
        # trigger tasks within the same job.
        for job_id_wire, siblings in siblings_by_job.items():
            seen: set[str] = set()
            deduped: list[ActiveTaskRow] = []
            for sib in siblings:
                sib_wire = sib.task_id.to_wire()
                if sib_wire not in direct_task_wires and sib_wire not in seen:
                    seen.add(sib_wire)
                    deduped.append(sib)
            siblings_by_job[job_id_wire] = deduped

        # -- Phase 2: apply all mutations. --
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}
        jobs_to_update: set[str] = set()

        for row in rows:
            tid = row.task_id
            tasks_to_kill.add(tid)
            decommit_worker = None
            decommit_resources = None
            if row.current_worker_id is not None:
                task_kill_workers[tid] = row.current_worker_id
                if not row.is_reservation_holder:
                    decommit_worker = str(row.current_worker_id)
                    decommit_resources = row.resources
            _terminate_task(
                cur,
                self._store.attempts,
                self._store.tasks,
                self._store.workers,
                self._store.endpoints,
                tid.to_wire(),
                row.current_attempt_id,
                job_pb2.TASK_STATE_FAILED,
                reason,
                now_ms,
                worker_id=decommit_worker,
                resources=decommit_resources,
                failure_count=row.failure_count + 1,
            )
            jobs_to_update.add(row.job_id.to_wire())

        # Terminate coscheduled siblings (deduplicated, all reads already done).
        for job_id_wire, siblings in siblings_by_job.items():
            if not siblings:
                continue
            job_resources = job_resources_cache[job_id_wire]
            # Pick the first direct-timeout task in this job as the "cause" for the error message.
            cause_tid = next(r.task_id for r in rows if r.job_id.to_wire() == job_id_wire)
            cascade_kill, cascade_workers = _terminate_coscheduled_siblings(
                cur,
                self._store.attempts,
                self._store.tasks,
                self._store.workers,
                self._store.endpoints,
                siblings,
                cause_tid,
                job_resources,
                now_ms,
            )
            tasks_to_kill.update(cascade_kill)
            task_kill_workers.update(cascade_workers)
            jobs_to_update.add(job_id_wire)

        for job_wire in jobs_to_update:
            new_job_state = self._recompute_job_state(cur, JobName.from_wire(job_wire))
            if new_job_state in TERMINAL_JOB_STATES:
                final_kill, final_workers = _finalize_terminal_job(
                    cur, self._store, JobName.from_wire(job_wire), new_job_state, now_ms
                )
                tasks_to_kill.update(final_kill)
                task_kill_workers.update(final_workers)
        for tid in tasks_to_kill:
            log_event("task_timeout", tid.to_wire(), reason=reason)
        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def remove_finished_job(self, cur: TransactionCursor, job_id: JobName) -> bool:
        """Remove a finished job and its tasks from state.

        Only removes jobs that are in a terminal state (SUCCEEDED, FAILED, KILLED,
        UNSCHEDULABLE). This allows job names to be reused after completion.

        Args:
            job_id: The job ID to remove

        Returns:
            True if the job was removed, False if it doesn't exist or is not finished
        """
        state = self._store.jobs.get_state(cur, job_id)
        if state is None:
            return False
        if state not in (
            job_pb2.JOB_STATE_SUCCEEDED,
            job_pb2.JOB_STATE_FAILED,
            job_pb2.JOB_STATE_KILLED,
            job_pb2.JOB_STATE_UNSCHEDULABLE,
        ):
            return False
        self._store.jobs.delete(cur, job_id)
        log_event("job_removed", job_id.to_wire(), state=state)
        return True

    def remove_worker(self, cur: TransactionCursor, worker_id: WorkerId) -> WorkerDetailRow | None:
        detail = self._store.workers.get_detail(cur, worker_id)
        if detail is None:
            return None
        _remove_worker(cur, self._store.workers, worker_id)
        cur.on_commit(lambda: log_event("worker_removed", str(worker_id)))
        cur.on_commit(lambda: self._store.workers.remove_from_attr_cache(worker_id))
        return detail

    def prune_old_data(
        self,
        *,
        job_retention: Duration,
        worker_retention: Duration,
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
            profile_retention: Delete task_profiles older than this.
            stop_event: If set, abort early (e.g. during shutdown).
            pause_between_s: Sleep between individual deletes to reduce lock contention.
        """
        now_ms = Timestamp.now().epoch_ms()
        job_cutoff_ms = now_ms - job_retention.to_ms()
        worker_cutoff_ms = now_ms - worker_retention.to_ms()

        def _stopped() -> bool:
            return stop_event is not None and stop_event.is_set()

        # 1. Jobs: one at a time (CASCADE to tasks → attempts, endpoints)
        jobs_deleted = 0
        while not _stopped():
            with self._store.read_snapshot() as snap:
                job_name = self._store.jobs.find_prunable(snap, job_cutoff_ms)
            if job_name is None:
                break
            with self._store.transaction() as cur:
                # Invalidate endpoint cache BEFORE the CASCADE so the cache
                # drops rows SQLite is about to delete for us.
                self._store.endpoints.remove_by_job_ids(cur, [job_name])
                self._store.tasks.remove_status_text_by_job_ids([job_name])
                self._store.jobs.delete(cur, job_name)
            log_event("job_pruned", job_name.to_wire())
            jobs_deleted += 1
            time.sleep(pause_between_s)

        # 2. Workers: one at a time (CASCADE to attributes, task_history, resource_history)
        workers_deleted = 0
        while not _stopped():
            with self._store.read_snapshot() as snap:
                worker_id = self._store.workers.find_prunable(snap, worker_cutoff_ms)
            if worker_id is None:
                break
            with self._store.transaction() as cur:
                _remove_worker(cur, self._store.workers, worker_id)
            log_event("worker_pruned", str(worker_id))
            workers_deleted += 1
            time.sleep(pause_between_s)

        # 3. Task profiles: batch of 1000 per transaction
        profile_cutoff_ms = now_ms - profile_retention.to_ms()
        profiles_deleted = self._store.tasks.prune_stale_profiles(
            cutoff_ms=profile_cutoff_ms,
            stopped=_stopped,
            pause_between_s=pause_between_s,
        )
        profiles_deleted += self._store.tasks.prune_orphan_profiles(
            stopped=_stopped,
            pause_between_s=pause_between_s,
        )

        result = PruneResult(
            jobs_deleted=jobs_deleted,
            workers_deleted=workers_deleted,
            profiles_deleted=profiles_deleted,
        )
        if result.total > 0:
            logger.info(
                "Pruned old data: %d jobs, %d workers, %d profiles",
                result.jobs_deleted,
                result.workers_deleted,
                result.profiles_deleted,
            )
            self._store.optimize()

        return result

    # =========================================================================
    # Split Heartbeat Helpers
    # =========================================================================

    def update_worker_pings(
        self,
        cur: TransactionCursor,
        worker_ids: Iterable[WorkerId],
    ) -> None:
        """Apply a batch of Ping RPC results within the caller's transaction.

        Bumps ``last_heartbeat_ms`` for each successfully-pinged worker.
        Per-tick host utilization is no longer persisted in the controller DB —
        workers emit it directly to the ``iris.worker`` stats namespace.
        Does not touch healthy/active/consecutive_failures — the ping loop
        tracks failures in-memory and uses ``fail_workers_batch`` to remove
        workers past threshold.
        """
        ids = list(worker_ids)
        if not ids:
            return
        now_ms = Timestamp.now().epoch_ms()
        self._store.workers.apply_snapshots(cur, ids, now_ms, reset_health=False)

    def get_running_tasks_for_poll(
        self,
        snap: QuerySnapshot | TransactionCursor,
    ) -> tuple[dict[WorkerId, list[RunningTaskEntry]], dict[WorkerId, str]]:
        """Snapshot running tasks and worker addresses for PollTasks RPCs.

        Caller passes an open read snapshot or transaction cursor.
        Returns (running_by_worker, worker_addresses) where running_by_worker
        maps worker_id to its list of running task entries and worker_addresses
        maps worker_id to its RPC address.
        """
        worker_addresses = self._store.workers.list_active_healthy(snap)
        if not worker_addresses:
            return {}, {}
        worker_ids = list(worker_addresses.keys())

        # Reservation holders are virtual — they live on ``current_worker_id``
        # only as a scheduling anchor and never get a RunTaskRequest. Sending
        # them in PollTasksRequest.expected_tasks makes the worker reconcile
        # against its _tasks dict, miss, and return WORKER_FAILED every cycle,
        # which drains the holder's preemption budget and (post the build-
        # failure health hook) reaps the claimed worker for a harmless miss.
        task_rows = self._store.tasks.list_active(
            snap,
            TaskScope(worker_ids=worker_ids),
            states=ACTIVE_TASK_STATES,
            exclude_reservation_holders=True,
            order_by_task_id=True,
        )

        running: dict[WorkerId, list[RunningTaskEntry]] = {}
        for row in task_rows:
            # list_active only returns rows where current_worker_id matched the IN filter,
            # so current_worker_id cannot be None here.
            assert row.current_worker_id is not None
            entry = RunningTaskEntry(
                task_id=row.task_id,
                attempt_id=row.current_attempt_id,
            )
            running.setdefault(row.current_worker_id, []).append(entry)
        return running, worker_addresses

    def fail_workers_batch(
        self,
        worker_ids: list[str],
        reason: str,
    ) -> WorkerFailureBatchResult:
        """Fail all active workers matching the given worker IDs.

        Used for slice reaping: when one worker on a multi-VM slice fails, all
        sibling workers on that slice must be failed immediately rather than
        waiting for individual ping timeouts.
        """
        if not worker_ids:
            return WorkerFailureBatchResult()
        with self._store.read_snapshot() as snap:
            rows = self._store.workers.list_active_by_ids(snap, worker_ids)
        failures = [(row.worker_id, row.address, reason) for row in rows]
        if not failures:
            return WorkerFailureBatchResult()
        results = self.fail_workers(failures)
        return WorkerFailureBatchResult(
            tasks_to_kill=results.tasks_to_kill,
            task_kill_workers=results.task_kill_workers,
            removed_workers=[(wid, addr) for wid, addr in results.removed_workers if addr is not None],
            results=results.results,
        )

    def load_workers_from_config(self, configs: list[WorkerConfig]) -> None:
        """Load workers from static configuration in a single transaction."""
        now = Timestamp.now()
        with self._store.transaction() as cur:
            for cfg in configs:
                self.register_or_refresh_worker(
                    cur,
                    worker_id=WorkerId(cfg.worker_id),
                    address=cfg.address,
                    metadata=cfg.metadata,
                    ts=now,
                )

    # --- Task Status Text ---

    def record_task_status_text(self, task_id: JobName, detail_md: str, summary_md: str) -> None:
        """Update the task's markdown status text for UI display (held in memory only)."""
        self._store.tasks.set_status_text(task_id.to_wire(), detail_md, summary_md)

    # --- Endpoint Management ---

    def add_endpoint(self, cur: TransactionCursor, endpoint: EndpointRow) -> bool:
        """Add an endpoint row through the store's endpoint cache.

        Returns True if the endpoint was inserted, False if the task is already
        terminal (to prevent orphaned endpoints that would never be cleaned up).
        """
        return self._store.endpoints.add(cur, endpoint)

    def remove_endpoint(self, cur: TransactionCursor, endpoint_id: str) -> EndpointRow | None:
        return self._store.endpoints.remove(cur, endpoint_id)

    # ---------------------------------------------------------------------
    # Test-only SQL mutation helpers
    # ---------------------------------------------------------------------

    def set_worker_health_for_test(self, worker_id: WorkerId, healthy: bool) -> None:
        """Test helper: set worker health in DB."""
        with self._store.transaction() as cur:
            self._store.workers.set_health_for_test(cur, worker_id, healthy)

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

        with self._store.transaction() as cur:
            self._store.workers.set_attribute_for_test(
                cur,
                worker_id,
                WorkerAttributeParams(key, value_type, str_value, int_value, float_value),
            )

    # =========================================================================
    # Direct provider methods
    # =========================================================================

    def drain_for_direct_provider(
        self,
        cur: TransactionCursor,
        max_promotions: int = DIRECT_PROVIDER_PROMOTION_RATE,
    ) -> DirectProviderBatch:
        """Drain pending tasks and snapshot running tasks for a direct provider sync cycle.

        Promotes up to ``max_promotions`` PENDING tasks to ASSIGNED (NULL
        worker_id), builds RunTaskRequest for each, and collects:
        - Newly promoted tasks -> tasks_to_run
        - Already ASSIGNED/BUILDING/RUNNING tasks with NULL worker_id -> running_tasks
        - Kill entries with NULL worker_id -> tasks_to_kill (deleted from queue)
        """
        now_ms = Timestamp.now().epoch_ms()

        newly_promoted: set[str] = set()
        tasks_to_run: list[job_pb2.RunTaskRequest] = []

        pending_rows = self._store.tasks.list_pending_for_direct_provider(cur, max_promotions)

        for row in pending_rows:
            attempt_id = row.current_attempt_id + 1

            self._store.tasks.assign(
                cur,
                self._store.attempts,
                row.task_id,
                None,
                None,
                attempt_id,
                now_ms,
            )

            entrypoint = proto_from_json(row.entrypoint_json, job_pb2.RuntimeEntrypoint)
            # Load inline workdir files from the job_workdir_files table.
            for filename, data in self._store.jobs.get_workdir_files(cur, row.job_id).items():
                entrypoint.workdir_files[filename] = data

            run_req = job_pb2.RunTaskRequest(
                task_id=row.task_id.to_wire(),
                num_tasks=row.num_tasks,
                entrypoint=entrypoint,
                environment=proto_from_json(row.environment_json, job_pb2.EnvironmentConfig),
                bundle_id=row.bundle_id,
                resources=row.resources,
                ports=json.loads(row.ports_json),
                attempt_id=attempt_id,
                constraints=[c.to_proto() for c in constraints_from_json(row.constraints_json)],
                task_image=row.task_image,
            )
            # Propagate timeout for K8s activeDeadlineSeconds (Kubernetes-native enforcement).
            if row.timeout_ms is not None and row.timeout_ms > 0:
                run_req.timeout.milliseconds = row.timeout_ms
            tasks_to_run.append(run_req)
            newly_promoted.add(row.task_id.to_wire())

        # Snapshot already-running tasks with NULL worker_id (excluding newly promoted).
        running_rows = self._store.tasks.list_active(
            cur,
            TaskScope(null_worker=True),
            states=ACTIVE_TASK_STATES,
            order_by_task_id=True,
        )
        running_tasks = [
            RunningTaskEntry(
                task_id=row.task_id,
                attempt_id=row.current_attempt_id,
            )
            for row in running_rows
            if row.task_id.to_wire() not in newly_promoted
        ]

        tasks_to_kill = self._store.dispatch.drain_direct_kills(cur)

        return DirectProviderBatch(
            tasks_to_run=tasks_to_run,
            running_tasks=running_tasks,
            tasks_to_kill=tasks_to_kill,
        )

    def apply_direct_provider_updates(self, cur: TransactionCursor, updates: list[TaskUpdate]) -> TxResult:
        """Apply a batch of task state updates from a KubernetesProvider.

        Same state machine as apply_task_updates but without worker lookup,
        health updates, or resource decommit (no committed resources tracked).
        """
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}

        now_ms = Timestamp.now().epoch_ms()
        cascaded_jobs: set[JobName] = set()

        for update in updates:
            task = self._store.tasks.get_detail(cur, update.task_id)
            if task is None:
                continue
            if task_row_is_finished(task) or update.new_state in (
                job_pb2.TASK_STATE_UNSPECIFIED,
                job_pb2.TASK_STATE_PENDING,
            ):
                continue
            if update.attempt_id != task.current_attempt_id:
                stale_state = self._store.attempts.get_state(cur, update.task_id, update.attempt_id)
                if stale_state is not None and stale_state not in TERMINAL_TASK_STATES:
                    logger.error(
                        "Stale attempt precondition violation: task=%s reported=%d current=%d stale_state=%s",
                        update.task_id,
                        update.attempt_id,
                        task.current_attempt_id,
                        stale_state,
                    )
                continue
            attempt = self._store.attempts.get(cur, update.task_id, update.attempt_id)
            if attempt is None:
                continue
            # See _apply_task_transitions for rationale: the current attempt may
            # be terminal while the task is retrying in PENDING; late reports
            # must not revive it.
            if attempt.state in TERMINAL_TASK_STATES:
                logger.debug(
                    "Dropping late update for terminal attempt: task=%s attempt=%d attempt_state=%d reported=%d",
                    update.task_id,
                    update.attempt_id,
                    attempt.state,
                    int(update.new_state),
                )
                continue

            if update.container_id is not None:
                self._store.tasks.update_container_id(cur, update.task_id, update.container_id)

            terminal_ms: int | None = None
            started_ms: int | None = None
            task_state = task.state
            task_error = update.error
            task_exit = update.exit_code
            failure_count = task.failure_count
            preemption_count = task.preemption_count

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
                if update.new_state == job_pb2.TASK_STATE_WORKER_FAILED and task.state in EXECUTING_TASK_STATES:
                    preemption_count += 1
                # WORKER_FAILED while still ASSIGNED -> retry immediately as PENDING
                if update.new_state == job_pb2.TASK_STATE_WORKER_FAILED and task.state == job_pb2.TASK_STATE_ASSIGNED:
                    task_state = job_pb2.TASK_STATE_PENDING
                    terminal_ms = None
                if update.new_state == job_pb2.TASK_STATE_FAILED and failure_count <= task.max_retries_failure:
                    task_state = job_pb2.TASK_STATE_PENDING
                    terminal_ms = None
                if (
                    update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
                    and preemption_count <= task.max_retries_preemption
                    and task.state in EXECUTING_TASK_STATES
                ):
                    task_state = job_pb2.TASK_STATE_PENDING
                    terminal_ms = None

            # An attempt is terminal whenever the update itself is terminal, even
            # if the TASK rolls back to PENDING for a retry.
            attempt_terminal_ms = now_ms if int(update.new_state) in TERMINAL_TASK_STATES else None

            self._store.attempts.apply_update(
                cur,
                TaskAttemptUpdateParams(
                    task_id=update.task_id,
                    attempt_id=update.attempt_id,
                    state=int(update.new_state),
                    started_at_ms=started_ms,
                    finished_at_ms=attempt_terminal_ms,
                    exit_code=task_exit,
                    error=update.error,
                ),
            )
            self._store.tasks.apply_state_update(
                cur,
                TaskStateUpdateParams(
                    task_id=update.task_id,
                    state=task_state,
                    error=task_error,
                    exit_code=task_exit,
                    started_at_ms=started_ms,
                    finished_at_ms=terminal_ms,
                    failure_count=failure_count,
                    preemption_count=preemption_count,
                ),
                ACTIVE_TASK_STATES,
            )
            jc_row = self._store.jobs.get_config(cur, task.job_id)

            if update.new_state in TERMINAL_TASK_STATES:
                delete_task_endpoints(cur, self._store.endpoints, update.task_id.to_wire())

            # Coscheduled sibling cascade. Branch on terminal vs transient (see
            # apply_state_updates above for the full rationale).
            reported_failure = int(update.new_state) in FAILURE_TASK_STATES
            if jc_row is not None and bool(int(jc_row["has_coscheduling"])) and reported_failure:
                siblings = _find_coscheduled_siblings(cur, self._store.tasks, task.job_id, update.task_id, True)
                job_resources = resource_spec_from_scalars(
                    int(jc_row["res_cpu_millicores"]),
                    int(jc_row["res_memory_bytes"]),
                    int(jc_row["res_disk_bytes"]),
                    jc_row["res_device_json"],
                )
                if task_state in FAILURE_TASK_STATES:
                    cascade_kill, cascade_workers = _terminate_coscheduled_siblings(
                        cur,
                        self._store.attempts,
                        self._store.tasks,
                        self._store.workers,
                        self._store.endpoints,
                        siblings,
                        update.task_id,
                        job_resources,
                        now_ms,
                    )
                else:
                    cascade_kill, cascade_workers = _requeue_coscheduled_siblings(
                        cur,
                        self._store.attempts,
                        self._store.tasks,
                        self._store.workers,
                        self._store.endpoints,
                        siblings,
                        update.task_id,
                        job_resources,
                        now_ms,
                    )
                tasks_to_kill.update(cascade_kill)
                task_kill_workers.update(cascade_workers)

            if task.job_id not in cascaded_jobs:
                new_job_state = self._recompute_job_state(cur, task.job_id)
                if new_job_state in TERMINAL_JOB_STATES:
                    final_tasks_to_kill, final_task_kill_workers = _finalize_terminal_job(
                        cur, self._store, task.job_id, new_job_state, now_ms
                    )
                    tasks_to_kill.update(final_tasks_to_kill)
                    task_kill_workers.update(final_task_kill_workers)
                    cascaded_jobs.add(task.job_id)

        if tasks_to_kill or cascaded_jobs:
            log_event("direct_provider_updates_applied", "direct")
            for job_id in cascaded_jobs:
                log_event("job_terminated", job_id.to_wire(), trigger="direct_provider_updates_applied")

        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def buffer_direct_kill(self, cur: TransactionCursor, task_id: str) -> None:
        """Buffer a kill request for a direct-provider task.

        Inserts a kill entry into dispatch_queue with worker_id=NULL.
        Drained by drain_for_direct_provider(). ``task_id`` is stored
        verbatim — callers in direct-provider land may pass IDs that
        aren't canonical ``JobName`` strings.
        """
        self._store.dispatch.enqueue_kill(cur, None, task_id, Timestamp.now().epoch_ms())

    # =========================================================================
    # Test helpers
    # =========================================================================

    def set_worker_consecutive_failures_for_test(self, worker_id: WorkerId, consecutive_failures: int) -> None:
        """Test helper: set worker consecutive failure count in DB."""
        with self._store.transaction() as cur:
            self._store.workers.set_consecutive_failures_for_test(cur, worker_id, consecutive_failures)

    def set_task_state_for_test(
        self,
        task_id: JobName,
        state: int,
        *,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        """Test helper: set task state directly in DB."""
        with self._store.transaction() as cur:
            self._store.tasks.set_state_for_test(cur, task_id, state, error=error, exit_code=exit_code)

    def create_attempt_for_test(self, task_id: JobName, worker_id: WorkerId) -> int:
        """Test helper: append a new task_attempt without finalizing prior attempt."""
        with self._store.read_snapshot() as snap:
            current_attempt_id = self._store.tasks.get_current_attempt_id(snap, task_id)
        if current_attempt_id is None:
            raise ValueError(f"unknown task: {task_id}")
        with self._store.read_snapshot() as snap:
            worker_address = self._store.workers.address(snap, worker_id) or str(worker_id)
        next_attempt_id = current_attempt_id + 1
        now_ms = Timestamp.now().epoch_ms()
        with self._store.transaction() as cur:
            self._store.tasks.assign(
                cur,
                self._store.attempts,
                task_id,
                worker_id,
                worker_address,
                next_attempt_id,
                now_ms,
            )
        return next_attempt_id
