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
    FAILURE_TASK_STATES,
    batch_delete,
)
from iris.cluster.controller.schema import (
    ACTIVE_TASK_STATES,
    EXECUTING_TASK_STATES,
    TASK_DETAIL_PROJECTION,
    WORKER_DETAIL_PROJECTION,
    EndpointRow,
    JobDetailRow,
    ResourceSpec,
    TaskDetailRow,
    WorkerDetailRow,
)
from iris.cluster.controller.store import (
    ActiveStateUpdate,
    AttemptFinalizer,
    ControllerStore,
    ControllerStores,
    DirectAssignment,
    HeartbeatApplyRequest,
    JobConfigInsert,
    JobInsert,
    KillResult,
    SiblingSnapshot,
    TaskFilter,
    TaskInsert,
    TaskProjection,
    TaskRetry,
    TaskTermination,
    TaskUpdate,
    WorkerAssignment,
    WorkerMetadata,
    WorkerUpsert,
    sql_placeholders,
    task_row_can_be_scheduled,
    task_row_is_finished,
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


_TERMINAL_STATE_REASONS: dict[int, str] = {
    job_pb2.JOB_STATE_FAILED: "Job exceeded max_task_failures",
    job_pb2.JOB_STATE_KILLED: "Job was terminated.",
    job_pb2.JOB_STATE_UNSCHEDULABLE: "Job could not be scheduled.",
    job_pb2.JOB_STATE_WORKER_FAILED: "Worker failed",
}


def _finalize_terminal_job(
    ctx: ControllerStore,
    job_id: JobName,
    terminal_state: int,
    now_ms: int,
) -> KillResult:
    """Kill remaining tasks and optionally cascade to children when a job goes terminal.

    Called after recompute_state determines a job has reached a terminal
    state. Kills the job's own non-terminal tasks and, depending on preemption
    policy, cascades to descendant jobs.

    Succeeded jobs always cascade (children are no longer needed).
    Non-succeeded jobs cascade only if the preemption policy is TERMINATE_CHILDREN.
    """
    reason = _TERMINAL_STATE_REASONS.get(terminal_state, "Job finalized")
    result = ctx.jobs.kill_non_terminal_tasks(ctx.cur, ctx.tasks, job_id.to_wire(), reason, now_ms)
    tasks_to_kill = set(result.tasks_to_kill)
    task_kill_workers = dict(result.task_kill_workers)
    should_cascade = True
    if terminal_state != job_pb2.JOB_STATE_SUCCEEDED:
        policy = ctx.jobs.get_preemption_policy(ctx.cur, job_id)
        should_cascade = policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    if should_cascade:
        child_result = ctx.jobs.cascade_children(ctx.cur, ctx.tasks, job_id, reason, now_ms)
        tasks_to_kill.update(child_result.tasks_to_kill)
        task_kill_workers.update(child_result.task_kill_workers)
    return KillResult(tasks_to_kill=frozenset(tasks_to_kill), task_kill_workers=task_kill_workers)


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
        stores: ControllerStores,
        heartbeat_failure_threshold: int = HEARTBEAT_FAILURE_THRESHOLD,
        user_budget_defaults: UserBudgetDefaults | None = None,
    ):
        self._stores = stores
        self._db = stores.db  # infra calls (read_snapshot, wal_checkpoint)
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

    def replace_reservation_claims(self, claims: dict[WorkerId, ReservationClaim]) -> None:
        """Replace all reservation claims atomically."""
        with self._stores.transact() as ctx:
            ctx.dispatch.replace_claims(ctx.cur, {wid: (claim.job_id, claim.entry_idx) for wid, claim in claims.items()})

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

        with self._stores.transact() as ctx:
            last_submission_ms = self._db.get_counter("last_submission_ms", ctx.cur)
            effective_submission_ms = max(submitted_ms, last_submission_ms + 1)
            self._db.set_counter("last_submission_ms", effective_submission_ms, ctx.cur)

            parent_job_id = job_id.parent.to_wire() if job_id.parent is not None else None
            if parent_job_id is not None:
                if not ctx.jobs.exists(ctx.cur, parent_job_id):
                    parent_job_id = None
            root_submitted_ms = effective_submission_ms
            if parent_job_id is not None:
                parent_root = ctx.jobs.get_root_submitted_ms(ctx.cur, parent_job_id)
                if parent_root is not None:
                    root_submitted_ms = parent_root

            deadline_epoch_ms: int | None = None
            if request.HasField("scheduling_timeout") and request.scheduling_timeout.milliseconds > 0:
                deadline_epoch_ms = (
                    Timestamp.from_ms(effective_submission_ms)
                    .add(duration_from_proto(request.scheduling_timeout))
                    .epoch_ms()
                )

            ctx.users.ensure_user_and_budget(ctx.cur, job_id.user, effective_submission_ms, self._user_budget_defaults)

            # Resolve priority band: use explicit request value, inherit from parent, or default to INTERACTIVE.
            requested_band = int(request.priority_band)
            if requested_band != job_pb2.PRIORITY_BAND_UNSPECIFIED:
                band_sort_key = requested_band
            elif parent_job_id is not None:
                parent_band = ctx.jobs.get_parent_band(ctx.cur, parent_job_id)
                band_sort_key = parent_band if parent_band is not None else job_pb2.PRIORITY_BAND_INTERACTIVE
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
            ctx.jobs.insert_job(
                ctx.cur,
                JobInsert(
                    job_id=job_id.to_wire(),
                    user_id=job_id.user,
                    parent_job_id=parent_job_id,
                    root_job_id=job_id.root_job.to_wire(),
                    depth=job_id.depth,
                    state=state,
                    submitted_at_ms=effective_submission_ms,
                    root_submitted_at_ms=root_submitted_ms,
                    finished_at_ms=finished_ms,
                    scheduling_deadline_epoch_ms=deadline_epoch_ms,
                    error=validation_error,
                    num_tasks=replicas,
                    is_reservation_holder=False,
                    name=job_name_lower,
                    has_reservation=has_reservation,
                ),
            )
            ctx.jobs.insert_job_config(
                ctx.cur,
                JobConfigInsert(
                    job_id=job_id.to_wire(),
                    name=job_name_lower,
                    has_reservation=has_reservation,
                    resources=ResourceSpec(
                        cpu_millicores=res_cpu,
                        memory_bytes=res_mem,
                        disk_bytes=res_disk,
                        device_json=res_device,
                    ),
                    constraints_json=constraints_json,
                    has_coscheduling=has_cosched,
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
                    fail_if_exists=1 if request.fail_if_exists else 0,
                ),
            )

            if request.entrypoint.workdir_files:
                ctx.jobs.insert_workdir_files(
                    ctx.cur,
                    job_id.to_wire(),
                    list(request.entrypoint.workdir_files.items()),
                )

            if validation_error is None:
                insertion_base = self._db.next_sequence("task_priority_insertion", cur=ctx.cur)
                for idx in range(replicas):
                    task_id = job_id.task(idx).to_wire()
                    created_task_ids.append(JobName.from_wire(task_id))
                    ctx.tasks.insert_task(
                        ctx.cur,
                        TaskInsert(
                            task_id=task_id,
                            job_id=job_id.to_wire(),
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
                    ctx.jobs.insert_job(
                        ctx.cur,
                        JobInsert(
                            job_id=holder_id.to_wire(),
                            user_id=holder_id.user,
                            parent_job_id=job_id.to_wire(),
                            root_job_id=holder_id.root_job.to_wire(),
                            depth=holder_id.depth,
                            state=job_pb2.JOB_STATE_PENDING,
                            submitted_at_ms=effective_submission_ms,
                            root_submitted_at_ms=root_submitted_ms,
                            finished_at_ms=None,
                            scheduling_deadline_epoch_ms=None,
                            error=None,
                            num_tasks=len(request.reservation.entries),
                            is_reservation_holder=True,
                            name=holder_name_lower,
                            has_reservation=False,
                        ),
                    )
                    holder_entrypoint_json = entrypoint_to_json(holder_request.entrypoint)
                    holder_environment_json = proto_to_json(holder_request.environment)
                    ctx.jobs.insert_job_config(
                        ctx.cur,
                        JobConfigInsert(
                            job_id=holder_id.to_wire(),
                            name=holder_name_lower,
                            has_reservation=False,
                            resources=ResourceSpec(
                                cpu_millicores=holder_res_cpu,
                                memory_bytes=holder_res_mem,
                                disk_bytes=holder_res_disk,
                                device_json=holder_res_device,
                            ),
                            constraints_json=holder_constraints_json,
                            has_coscheduling=0,
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
                    holder_base = self._db.next_sequence("task_priority_insertion", cur=ctx.cur)
                    for idx in range(len(request.reservation.entries)):
                        created_task_ids.append(holder_id.task(idx))
                        ctx.tasks.insert_task(
                            ctx.cur,
                            TaskInsert(
                                task_id=holder_id.task(idx).to_wire(),
                                job_id=holder_id.to_wire(),
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

            actions.append(("job_submitted", job_id.to_wire(), {"num_tasks": replicas, "error": validation_error}))
            self._record_transaction(ctx.cur, "submit_job", actions)
        return SubmitJobResult(job_id=job_id, task_ids=created_task_ids)

    def cancel_job(self, job_id: JobName, reason: str) -> TxResult:
        """Cancel a job tree and return tasks that need kill RPCs."""
        with self._stores.transact() as ctx:
            subtree_ids = ctx.jobs.get_subtree_ids(ctx.cur, job_id.to_wire())
            if not subtree_ids:
                return TxResult()
            running_rows = ctx.tasks.query(
                ctx.cur,
                TaskFilter(job_ids=tuple(subtree_ids), states=ACTIVE_TASK_STATES),
                projection=TaskProjection.WITH_JOB_CONFIG,
            )
            tasks_to_kill = {row.task_id for row in running_rows}
            task_kill_workers = {
                row.task_id: WorkerId(str(row.current_worker_id))
                for row in running_rows
                if row.current_worker_id is not None
            }
            # Decommit resources for each active task on its assigned worker.
            # cancel_job marks tasks as KILLED, but apply_heartbeat skips
            # already-finished tasks (is_finished() check), so the normal
            # heartbeat decommit path never fires for cancelled tasks.
            # Direct-provider tasks have NULL worker_id — skip decommit for them.
            for row in running_rows:
                if row.current_worker_id is not None and not row.is_reservation_holder:
                    resources = resource_spec_from_scalars(
                        row.resources.cpu_millicores,
                        row.resources.memory_bytes,
                        row.resources.disk_bytes,
                        row.resources.device_json,
                    )
                    ctx.workers.decommit_resources(ctx.cur, str(row.current_worker_id), resources)
            now_ms = Timestamp.now().epoch_ms()
            ctx.tasks.bulk_cancel(ctx.cur, subtree_ids, reason, now_ms)
            ctx.jobs.bulk_cancel(ctx.cur, subtree_ids, reason, now_ms)
            ctx.endpoints.remove_by_job_ids(ctx.cur, [JobName.from_wire(jid) for jid in subtree_ids])
            self._record_transaction(ctx.cur, "cancel_job", [("job_cancelled", job_id.to_wire(), {"reason": reason})])
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
        with self._stores.transact() as ctx:
            ctx.workers.upsert(
                ctx.cur,
                WorkerUpsert(
                    worker_id=str(worker_id),
                    address=address,
                    now_ms=now_ms,
                    total_cpu_millicores=metadata.cpu_count * 1000,
                    total_memory_bytes=metadata.memory_bytes,
                    total_gpu_count=gpu_count,
                    total_tpu_count=tpu_count,
                    device_type=device_type,
                    device_variant=device_variant,
                    slice_id=slice_id,
                    scale_group=scale_group,
                    metadata=WorkerMetadata(
                        hostname=metadata.hostname,
                        ip_address=metadata.ip_address,
                        cpu_count=metadata.cpu_count,
                        memory_bytes=metadata.memory_bytes,
                        disk_bytes=metadata.disk_bytes,
                        tpu_name=metadata.tpu_name,
                        tpu_worker_hostnames=metadata.tpu_worker_hostnames,
                        tpu_worker_id=metadata.tpu_worker_id,
                        tpu_chips_per_host_bounds=metadata.tpu_chips_per_host_bounds,
                        gpu_count=metadata.gpu_count,
                        gpu_name=metadata.gpu_name,
                        gpu_memory_mb=metadata.gpu_memory_mb,
                        gce_instance_name=metadata.gce_instance_name,
                        gce_zone=metadata.gce_zone,
                        git_hash=metadata.git_hash,
                        device_json=proto_to_json(metadata.device),
                    ),
                    attributes=attrs,
                ),
            )
            self._record_transaction(
                ctx.cur, "register_worker", [("worker_registered", str(worker_id), {"address": address})]
            )
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

    def queue_assignments(self, assignments: list[Assignment]) -> AssignmentResult:
        """Commit assignments and enqueue dispatches in one transaction."""
        accepted: list[Assignment] = []
        rejected: list[Assignment] = []
        has_real_dispatch = False
        with self._stores.transact() as ctx:
            now_ms = Timestamp.now().epoch_ms()
            job_cache: dict[str, JobDetailRow] = {}
            jobs_to_update: set[str] = set()
            for assignment in assignments:
                task_row = ctx.tasks.get_for_assignment(ctx.cur, assignment.task_id.to_wire())
                worker_row = ctx.workers.get_healthy_active(ctx.cur, str(assignment.worker_id))
                if task_row is None or worker_row is None:
                    rejected.append(assignment)
                    continue
                task = TASK_DETAIL_PROJECTION.decode_one([task_row])
                if not task_row_can_be_scheduled(task):
                    rejected.append(assignment)
                    continue
                job_id_wire = task.job_id.to_wire()
                if job_id_wire not in job_cache:
                    decoded_job = ctx.jobs.get_job_detail(ctx.cur, job_id_wire)
                    if decoded_job is None:
                        rejected.append(assignment)
                        continue
                    job_cache[job_id_wire] = decoded_job
                job = job_cache[job_id_wire]
                attempt_id = int(task_row["current_attempt_id"]) + 1
                ctx.tasks.assign_to_worker(
                    ctx.cur,
                    WorkerAssignment(
                        task_id=assignment.task_id.to_wire(),
                        attempt_id=attempt_id,
                        worker_id=str(assignment.worker_id),
                        worker_address=str(worker_row["address"]),
                        now_ms=now_ms,
                    ),
                )
                if not job.is_reservation_holder:
                    resources = resource_spec_from_scalars(
                        job.resources.cpu_millicores,
                        job.resources.memory_bytes,
                        job.resources.disk_bytes,
                        job.resources.device_json,
                    )
                    ctx.workers.commit_resources(ctx.cur, str(assignment.worker_id), resources)
                    entrypoint = proto_from_json(job.entrypoint_json, job_pb2.RuntimeEntrypoint)
                    for fn, data in ctx.jobs.get_workdir_files(ctx.cur, job_id_wire).items():
                        entrypoint.workdir_files[fn] = data
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
                    ctx.dispatch.enqueue_run(ctx.cur, str(assignment.worker_id), run_request.SerializeToString(), now_ms)
                    has_real_dispatch = True
                ctx.workers.record_worker_task_history(
                    ctx.cur, str(assignment.worker_id), assignment.task_id.to_wire(), now_ms
                )
                jobs_to_update.add(job_id_wire)
                accepted.append(assignment)
            for job_id_wire in jobs_to_update:
                ctx.jobs.start_if_pending(ctx.cur, job_id_wire, now_ms)
            if accepted or rejected:
                actions = [("assignment_queued", a.task_id.to_wire(), {"worker_id": str(a.worker_id)}) for a in accepted]
                self._record_transaction(ctx.cur, "queue_assignments", actions)
        return AssignmentResult(
            tasks_to_kill=set(), has_real_dispatch=has_real_dispatch, accepted=accepted, rejected=rejected
        )

    def _update_worker_health(self, ctx: ControllerStore, req: HeartbeatApplyRequest, now_ms: int) -> bool:
        """Update worker health, resource snapshot, and history.

        Returns False if the worker doesn't exist (caller should bail).
        """
        existing = ctx.workers.update_health_batch(ctx.cur, [req], now_ms)
        return str(req.worker_id) in existing

    def _apply_task_transitions(
        self,
        ctx: ControllerStore,
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

        for update in req.updates:
            snapshot = ctx.tasks.get_task(ctx.cur, update.task_id)
            if snapshot is None:
                continue
            if snapshot.state in TERMINAL_TASK_STATES or update.new_state in (
                job_pb2.TASK_STATE_UNSPECIFIED,
                job_pb2.TASK_STATE_PENDING,
            ):
                continue
            if update.attempt_id != snapshot.attempt_id:
                stale_state = ctx.tasks.get_attempt_state(ctx.cur, update.task_id.to_wire(), update.attempt_id)
                if stale_state is not None and stale_state not in TERMINAL_TASK_STATES:
                    logger.error(
                        "Stale attempt precondition violation: task=%s reported=%d current=%d stale_state=%s",
                        update.task_id,
                        update.attempt_id,
                        snapshot.attempt_id,
                        int(stale_state),
                    )
                continue

            prior_state = int(snapshot.state)

            # Fast path: task already in the reported state with no new data to apply.
            has_new_data = update.error is not None or update.exit_code is not None or update.resource_usage is not None
            if update.new_state == prior_state and not has_new_data:
                continue

            # The attempt is already terminal (e.g. preempted, killed) but the task has
            # been rolled back to PENDING for retry and current_attempt_id still points
            # at the dead attempt. Reviving it would produce an inconsistent row where
            # state contradicts finished_at_ms/error.
            if snapshot.attempt_state in TERMINAL_TASK_STATES:
                logger.debug(
                    "Dropping late update for terminal attempt: task=%s attempt=%d attempt_state=%d reported=%d",
                    update.task_id,
                    update.attempt_id,
                    int(snapshot.attempt_state),
                    int(update.new_state),
                )
                continue

            if update.resource_usage is not None:
                ctx.tasks.insert_resource_usage(
                    ctx.cur,
                    update.task_id.to_wire(),
                    update.attempt_id,
                    update.resource_usage,
                    now_ms,
                )

            if update.container_id is not None:
                ctx.tasks.update_container_id(ctx.cur, update.task_id.to_wire(), update.container_id)

            # --- Inline retry logic (PR 2 will extract to resolve_transition) ---
            task_state = prior_state
            task_error = update.error
            task_exit = update.exit_code
            failure_count = snapshot.failure_count
            preemption_count = snapshot.preemption_count
            started_ms: int | None = None
            is_terminal_update = False

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
                is_terminal_update = True
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
                if update.new_state == job_pb2.TASK_STATE_FAILED and failure_count <= snapshot.max_retries_failure:
                    task_state = job_pb2.TASK_STATE_PENDING
                if (
                    update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
                    and preemption_count <= snapshot.max_retries_preemption
                    and prior_state in EXECUTING_TASK_STATES
                ):
                    task_state = job_pb2.TASK_STATE_PENDING

            # --- Apply writes through store methods ---
            if task_state in ACTIVE_TASK_STATES:
                ctx.tasks.update_active(
                    ctx.cur,
                    ActiveStateUpdate(
                        task_id=update.task_id.to_wire(),
                        attempt_id=update.attempt_id,
                        state=task_state,
                        error=task_error,
                        exit_code=task_exit,
                        started_ms=started_ms,
                        failure_count=failure_count,
                        preemption_count=preemption_count,
                    ),
                )
            elif task_state == job_pb2.TASK_STATE_PENDING:
                ctx.tasks.requeue(
                    ctx.cur,
                    TaskRetry(
                        task_id=update.task_id.to_wire(),
                        finalize=AttemptFinalizer.build(
                            update.task_id.to_wire(), update.attempt_id, int(update.new_state), now_ms
                        ),
                        worker_id=snapshot.worker_id,
                        resources=snapshot.resources,
                        failure_count=failure_count,
                        preemption_count=preemption_count,
                    ),
                )
            elif is_terminal_update:
                ctx.tasks.terminate(
                    ctx.cur,
                    TaskTermination(
                        task_id=update.task_id.to_wire(),
                        state=task_state,
                        now_ms=now_ms,
                        error=task_error,
                        finalize=AttemptFinalizer(
                            task_id=update.task_id.to_wire(),
                            attempt_id=update.attempt_id,
                            attempt_state=task_state,
                            now_ms=now_ms,
                            error=task_error,
                            exit_code=task_exit,
                        ),
                        worker_id=snapshot.worker_id,
                        resources=snapshot.resources,
                        failure_count=failure_count,
                        preemption_count=preemption_count,
                    ),
                )

            # Coscheduled jobs: a terminal host failure should cascade to siblings.
            if task_state in FAILURE_TASK_STATES and snapshot.has_coscheduling:
                siblings = ctx.tasks.find_coscheduled_siblings(
                    ctx.cur, snapshot.job_id, update.task_id, snapshot.has_coscheduling
                )
                if siblings and snapshot.resources is not None:
                    kill_result = ctx.tasks.terminate_coscheduled_siblings(
                        ctx.cur, siblings, update.task_id, snapshot.resources, now_ms
                    )
                    tasks_to_kill.update(kill_result.tasks_to_kill)
                    task_kill_workers.update(kill_result.task_kill_workers)

            # Mark job for recomputation (deduplicated, done after the task loop).
            if task_state != prior_state:
                jobs_to_recompute.add(snapshot.job_id)

        # Recompute job states once per job instead of once per task.
        for job_id in jobs_to_recompute:
            if job_id in cascaded_jobs:
                continue
            new_job_state = ctx.jobs.recompute_state(ctx.cur, job_id)
            if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
                kill_result = _finalize_terminal_job(ctx, job_id, new_job_state, now_ms)
                tasks_to_kill.update(kill_result.tasks_to_kill)
                task_kill_workers.update(kill_result.task_kill_workers)
                cascaded_jobs.add(job_id)
        if tasks_to_kill or cascaded_jobs:
            actions: list[tuple[str, str, dict[str, object]]] = [("heartbeat_applied", str(req.worker_id), {})]
            for job_id in cascaded_jobs:
                actions.append(("job_terminated", job_id.to_wire(), {}))
            self._record_transaction(ctx.cur, "apply_task_updates", actions)

        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def apply_task_updates(self, req: HeartbeatApplyRequest) -> TxResult:
        """Apply a batch of worker task updates atomically."""
        with self._stores.transact() as ctx:
            now_ms = Timestamp.now().epoch_ms()
            if not self._update_worker_health(ctx, req, now_ms):
                return TxResult()
            result = self._apply_task_transitions(ctx, req, now_ms)

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

        with self._stores.transact() as ctx:
            now_ms = Timestamp.now().epoch_ms()

            # ── Batch worker health updates ───────────────────────────────
            existing_workers = ctx.workers.update_health_batch(ctx.cur, requests, now_ms)

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

            task_rows = ctx.tasks.query(ctx.cur, TaskFilter(task_ids=tuple(all_task_ids)))
            task_row_map = {t.task_id.to_wire(): t for t in task_rows}

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
                    task = task_row_map.get(task_id_wire)
                    if task is None:
                        continue

                    is_state_change = update.new_state != task.state
                    has_terminal_data = update.error is not None or update.exit_code is not None

                    if is_state_change or has_terminal_data:
                        transition_updates.append(update)
                    else:
                        # Steady-state: check finished / stale attempt before writing.
                        if task_row_is_finished(task):
                            continue
                        if update.attempt_id != task.current_attempt_id:
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
                ctx.tasks.insert_resource_usage_batch(ctx.cur, task_history_params)

            # ── Pass 2b: transitions via existing state machine ───────────
            for req_idx, treq in transition_entries:
                tx_result = self._apply_task_transitions(ctx, treq, now_ms)
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
        ctx: ControllerStore,
        worker_id: WorkerId,
        error: str,
        *,
        now_ms: int,
    ) -> TxResult:
        """Remove a definitively failed worker and cascade its task state."""
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}
        task_rows = ctx.tasks.query(
            ctx.cur,
            TaskFilter(worker_id=worker_id, states=ACTIVE_TASK_STATES),
            projection=TaskProjection.WITH_JOB,
        )
        for task_row in task_rows:
            tid = task_row.task_id.to_wire()
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
            if is_reservation_holder:
                ctx.tasks.delete_attempt(ctx.cur, tid, task_row.current_attempt_id)
                ctx.tasks.reset_reservation_holder(ctx.cur, tid, new_task_state)
            elif new_task_state == job_pb2.TASK_STATE_PENDING:
                ctx.tasks.requeue(
                    ctx.cur,
                    TaskRetry(
                        task_id=tid,
                        finalize=AttemptFinalizer.build(
                            tid, task_row.current_attempt_id, job_pb2.TASK_STATE_WORKER_FAILED, now_ms
                        ),
                        preemption_count=preemption_count,
                    ),
                )
            else:
                worker_fail_error = f"Worker {worker_id} failed: {error}"
                ctx.tasks.terminate(
                    ctx.cur,
                    TaskTermination(
                        task_id=tid,
                        state=new_task_state,
                        now_ms=now_ms,
                        error=worker_fail_error,
                        finalize=AttemptFinalizer.build(
                            tid,
                            task_row.current_attempt_id,
                            job_pb2.TASK_STATE_WORKER_FAILED,
                            now_ms,
                            error=worker_fail_error,
                        ),
                        preemption_count=preemption_count,
                    ),
                )
            task_id = task_row.task_id
            parent_job_id, _ = task_id.require_task()
            new_job_state = ctx.jobs.recompute_state(ctx.cur, parent_job_id)
            if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
                kill_result = _finalize_terminal_job(ctx, parent_job_id, new_job_state, now_ms)
                tasks_to_kill.update(kill_result.tasks_to_kill)
                task_kill_workers.update(kill_result.task_kill_workers)
            elif new_task_state == job_pb2.TASK_STATE_PENDING:
                policy = ctx.jobs.get_preemption_policy(ctx.cur, parent_job_id)
                if policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
                    child_result = ctx.jobs.cascade_children(
                        ctx.cur,
                        ctx.tasks,
                        parent_job_id,
                        "Parent task preempted",
                        now_ms,
                        exclude_reservation_holders=True,
                    )
                    tasks_to_kill.update(child_result.tasks_to_kill)
                    task_kill_workers.update(child_result.task_kill_workers)
            if new_task_state == job_pb2.TASK_STATE_WORKER_FAILED:
                tasks_to_kill.add(task_id)
        ctx.workers.remove(ctx.cur, str(worker_id))
        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def _record_heartbeat_failure(
        self,
        ctx: ControllerStore,
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
        row = ctx.workers.get_active_row(ctx.cur, str(worker_id))
        if row is None:
            return HeartbeatFailureResult(
                worker_removed=True,
                action=HeartbeatAction.WORKER_FAILED,
                failure_threshold=self._heartbeat_failure_threshold,
            )

        now_ms = now_ms or Timestamp.now().epoch_ms()
        last_heartbeat_ms = row.last_heartbeat_ms
        last_heartbeat_age_ms = None if last_heartbeat_ms is None else max(0, now_ms - last_heartbeat_ms)
        failures = row.consecutive_failures + 1
        ctx.workers.record_heartbeat_failure(ctx.cur, worker_id, failures, self._heartbeat_failure_threshold)
        should_remove = force_remove or failures >= self._heartbeat_failure_threshold
        if should_remove:
            removal = self._remove_failed_worker(ctx, worker_id, error, now_ms=now_ms)
            tasks_to_kill.update(removal.tasks_to_kill)
            task_kill_workers.update(removal.task_kill_workers)
        else:
            for req in drained_dispatch.tasks_to_run:
                ctx.dispatch.enqueue_run(ctx.cur, str(worker_id), req.SerializeToString(), now_ms)
            for task_id in drained_dispatch.tasks_to_kill:
                ctx.dispatch.enqueue_kill(ctx.cur, str(worker_id), task_id, now_ms)
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
        with self._stores.transact() as ctx:
            result = self._record_heartbeat_failure(
                ctx,
                worker_id,
                error,
                drained_dispatch,
                force_remove=force_remove,
            )
            self._record_transaction(
                ctx.cur,
                "heartbeat_failure",
                [("worker_heartbeat_failed", str(worker_id), {"error": error})],
            )
        return TxResult(tasks_to_kill=result.tasks_to_kill, task_kill_workers=result.task_kill_workers)

    def fail_heartbeat_for_worker(
        self,
        worker_id: WorkerId,
        error: str,
        snapshot: DispatchBatch,
        *,
        force_remove: bool = False,
    ) -> HeartbeatFailureResult:
        with self._stores.transact() as ctx:
            result = self._record_heartbeat_failure(
                ctx,
                worker_id,
                error,
                snapshot,
                force_remove=force_remove,
            )
            self._record_transaction(
                ctx.cur,
                "heartbeat_failure",
                [("worker_heartbeat_failed", str(worker_id), {"error": error})],
            )
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
            with self._stores.transact() as ctx:
                now_ms = Timestamp.now().epoch_ms()
                for snapshot, error in chunk:
                    result = self._record_heartbeat_failure(
                        ctx,
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
                    ctx.cur,
                    "heartbeat_failures_batch",
                    chunk_actions,
                    payload={"count": len(chunk_actions)},
                )

        return WorkerFailureBatchResult(
            tasks_to_kill=all_tasks_to_kill,
            task_kill_workers=all_task_kill_workers,
            removed_workers=removed_workers,
            results=results,
        )

    def mark_task_unschedulable(self, task_id: JobName, reason: str) -> TxResult:
        """Mark a task as unschedulable using the task transition engine."""
        with self._stores.transact() as ctx:
            job_id_wire = ctx.tasks.get_job_id(ctx.cur, task_id.to_wire())
            if job_id_wire is None:
                return TxResult()
            now_ms = Timestamp.now().epoch_ms()
            ctx.tasks.terminate(
                ctx.cur,
                TaskTermination(
                    task_id=task_id.to_wire(),
                    state=job_pb2.TASK_STATE_UNSCHEDULABLE,
                    now_ms=now_ms,
                    error=reason,
                ),
            )
            ctx.jobs.recompute_state(ctx.cur, JobName.from_wire(job_id_wire))
            self._record_transaction(
                ctx.cur, "mark_task_unschedulable", [("task_unschedulable", task_id.to_wire(), {"reason": reason})]
            )
        return TxResult()

    def preempt_task(self, task_id: JobName, reason: str) -> TxResult:
        """Preempt a running task, consuming from preemption retry budget.

        Marks the task as PREEMPTED (or retries as PENDING if budget remains),
        decommits its resources from the worker, and cascades to children if needed.
        """
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}
        with self._stores.transact() as ctx:
            preempt_rows = ctx.tasks.query(
                ctx.cur,
                TaskFilter(task_ids=(task_id.to_wire(),)),
                projection=TaskProjection.WITH_JOB_CONFIG,
            )
            if not preempt_rows:
                return TxResult()
            row = preempt_rows[0]

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
            attempt_worker_id = ctx.tasks.get_attempt_worker(ctx.cur, task_id.to_wire(), row.current_attempt_id)
            attempt_resources = None
            if attempt_worker_id is not None and row.resources is not None:
                attempt_resources = resource_spec_from_scalars(
                    row.resources.cpu_millicores,
                    row.resources.memory_bytes,
                    row.resources.disk_bytes,
                    row.resources.device_json,
                )

            ctx.tasks.terminate(
                ctx.cur,
                TaskTermination(
                    task_id=task_id.to_wire(),
                    state=new_state,
                    now_ms=now_ms,
                    error=reason,
                    finalize=AttemptFinalizer.build(
                        task_id.to_wire(),
                        row.current_attempt_id,
                        job_pb2.TASK_STATE_PREEMPTED,
                        now_ms,
                        error=reason,
                    ),
                    worker_id=attempt_worker_id,
                    resources=attempt_resources,
                    preemption_count=preemption_count,
                ),
            )

            # Recompute job state and cascade if terminal
            job_id = row.job_id
            new_job_state = ctx.jobs.recompute_state(ctx.cur, job_id)
            if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
                kill_result = _finalize_terminal_job(ctx, job_id, new_job_state, now_ms)
                tasks_to_kill.update(kill_result.tasks_to_kill)
                task_kill_workers.update(kill_result.task_kill_workers)
            elif new_state == job_pb2.TASK_STATE_PENDING:
                policy = ctx.jobs.get_preemption_policy(ctx.cur, job_id)
                if policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
                    child_result = ctx.jobs.cascade_children(
                        ctx.cur,
                        ctx.tasks,
                        job_id,
                        reason,
                        now_ms,
                        exclude_reservation_holders=True,
                    )
                    tasks_to_kill.update(child_result.tasks_to_kill)
                    task_kill_workers.update(child_result.task_kill_workers)

            if new_state == job_pb2.TASK_STATE_PREEMPTED:
                tasks_to_kill.add(task_id)

            self._record_transaction(
                ctx.cur, "preempt_task", [("task_preempted", task_id.to_wire(), {"reason": reason})]
            )

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
        with self._stores.transact() as ctx:
            wires = [tid.to_wire() for tid in task_ids]
            rows = ctx.tasks.query(
                ctx.cur,
                TaskFilter(task_ids=tuple(wires), states=EXECUTING_TASK_STATES),
                projection=TaskProjection.WITH_JOB_CONFIG,
            )

            # -- Phase 1: read all state before any mutations. --
            now_ms = Timestamp.now().epoch_ms()
            job_row_cache: dict[str, TaskDetailRow] = {}
            direct_task_wires: set[str] = set()
            siblings_by_job: dict[str, list[SiblingSnapshot]] = {}

            for row in rows:
                task_id_wire = row.task_id.to_wire()
                direct_task_wires.add(task_id_wire)
                job_id_wire = row.job_id.to_wire()
                if job_id_wire not in job_row_cache:
                    job_row_cache[job_id_wire] = row
                has_cosched = row.has_coscheduling
                tid = row.task_id
                siblings = ctx.tasks.find_coscheduled_siblings(ctx.cur, row.job_id, tid, has_cosched)
                if siblings:
                    existing = siblings_by_job.get(job_id_wire, [])
                    existing.extend(siblings)
                    siblings_by_job[job_id_wire] = existing

            # Deduplicate siblings: drop any that will already be terminated
            # directly as timed-out tasks, and deduplicate across multiple
            # trigger tasks within the same job.
            for job_id_wire, siblings in siblings_by_job.items():
                seen: set[str] = set()
                deduped: list[SiblingSnapshot] = []
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
                task_id_wire = row.task_id.to_wire()
                tid = row.task_id
                job_id_wire = row.job_id.to_wire()
                tasks_to_kill.add(tid)
                decommit_worker = None
                decommit_resources = None
                if row.current_worker_id is not None:
                    task_kill_workers[tid] = WorkerId(str(row.current_worker_id))
                    if not row.is_reservation_holder and row.resources is not None:
                        decommit_worker = str(row.current_worker_id)
                        decommit_resources = resource_spec_from_scalars(
                            row.resources.cpu_millicores,
                            row.resources.memory_bytes,
                            row.resources.disk_bytes,
                            row.resources.device_json,
                        )
                ctx.tasks.terminate(
                    ctx.cur,
                    TaskTermination(
                        task_id=task_id_wire,
                        state=job_pb2.TASK_STATE_FAILED,
                        now_ms=now_ms,
                        error=reason,
                        finalize=AttemptFinalizer.build(
                            task_id_wire,
                            row.current_attempt_id,
                            job_pb2.TASK_STATE_FAILED,
                            now_ms,
                            error=reason,
                        ),
                        worker_id=decommit_worker,
                        resources=decommit_resources,
                        failure_count=row.failure_count + 1,
                    ),
                )
                jobs_to_update.add(job_id_wire)

            # Terminate coscheduled siblings (deduplicated, all reads already done).
            for job_id_wire, siblings in siblings_by_job.items():
                if not siblings:
                    continue
                jc_row = job_row_cache[job_id_wire]
                assert jc_row.resources is not None
                job_resources = resource_spec_from_scalars(
                    jc_row.resources.cpu_millicores,
                    jc_row.resources.memory_bytes,
                    jc_row.resources.disk_bytes,
                    jc_row.resources.device_json,
                )
                cause_tid = next(r.task_id for r in rows if r.job_id.to_wire() == job_id_wire)
                cascade_result = ctx.tasks.terminate_coscheduled_siblings(
                    ctx.cur, siblings, cause_tid, job_resources, now_ms
                )
                tasks_to_kill.update(cascade_result.tasks_to_kill)
                task_kill_workers.update(cascade_result.task_kill_workers)
                jobs_to_update.add(job_id_wire)

            for job_wire in jobs_to_update:
                new_job_state = ctx.jobs.recompute_state(ctx.cur, JobName.from_wire(job_wire))
                if new_job_state in TERMINAL_JOB_STATES:
                    final_result = _finalize_terminal_job(ctx, JobName.from_wire(job_wire), new_job_state, now_ms)
                    tasks_to_kill.update(final_result.tasks_to_kill)
                    task_kill_workers.update(final_result.task_kill_workers)
            self._record_transaction(
                ctx.cur,
                "cancel_tasks_for_timeout",
                [("task_timeout", tid.to_wire(), {"reason": reason}) for tid in tasks_to_kill],
            )
        return TxResult(tasks_to_kill=tasks_to_kill, task_kill_workers=task_kill_workers)

    def drain_dispatch(self, worker_id: WorkerId) -> DispatchBatch | None:
        """Drain buffered dispatches and snapshot worker running tasks."""
        with self._stores.transact() as ctx:
            worker_row = ctx.workers.get_healthy_active(ctx.cur, str(worker_id))
            if worker_row is None:
                return None
            drained = ctx.dispatch.drain_for_worker(ctx.cur, str(worker_id))
            running_rows_raw = ctx.tasks.query(
                ctx.cur,
                TaskFilter(worker_id=worker_id, states=ACTIVE_TASK_STATES),
            )
            running_job_ids = {t.job_id.to_wire() for t in running_rows_raw}
            holder_ids = ctx.jobs.get_reservation_holder_ids(ctx.cur, running_job_ids)
            running_rows = [t for t in running_rows_raw if t.job_id.to_wire() not in holder_ids]
            tasks_to_run: list[job_pb2.RunTaskRequest] = []
            tasks_to_kill: list[str] = []
            for kind, payload_proto, task_id in drained:
                if kind == "run" and payload_proto is not None:
                    req = job_pb2.RunTaskRequest()
                    req.ParseFromString(bytes(payload_proto))
                    tasks_to_run.append(req)
                elif task_id is not None:
                    tasks_to_kill.append(str(task_id))
            return DispatchBatch(
                worker_id=WorkerId(str(worker_row["worker_id"])),
                worker_address=str(worker_row["address"]),
                running_tasks=[
                    RunningTaskEntry(
                        task_id=t.task_id,
                        attempt_id=t.current_attempt_id,
                    )
                    for t in running_rows
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
        with self._stores.transact() as ctx:
            dispatch_by_worker = ctx.dispatch.drain_for_workers(ctx.cur, list(worker_id_set))

        # -- Phase 3: build results (pure Python, no lock) --

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
            for kind, payload_proto, task_id in w_dispatch:
                if kind == "run" and payload_proto is not None:
                    req = job_pb2.RunTaskRequest()
                    req.ParseFromString(bytes(payload_proto))
                    tasks_to_run.append(req)
                elif task_id is not None:
                    tasks_to_kill.append(str(task_id))

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
        with self._stores.transact() as ctx:
            now_ms = Timestamp.now().epoch_ms()
            for req in batch.tasks_to_run:
                ctx.dispatch.enqueue_run(ctx.cur, str(batch.worker_id), req.SerializeToString(), now_ms)
            for task_id in batch.tasks_to_kill:
                ctx.dispatch.enqueue_kill(ctx.cur, str(batch.worker_id), task_id, now_ms)

    def remove_finished_job(self, job_id: JobName) -> bool:
        """Remove a finished job and its tasks from state.

        Only removes jobs that are in a terminal state (SUCCEEDED, FAILED, KILLED,
        UNSCHEDULABLE). This allows job names to be reused after completion.

        Args:
            job_id: The job ID to remove

        Returns:
            True if the job was removed, False if it doesn't exist or is not finished
        """
        with self._stores.transact() as ctx:
            state = ctx.jobs.get_state(ctx.cur, job_id)
            if state is None:
                return False
            if state not in (
                job_pb2.JOB_STATE_SUCCEEDED,
                job_pb2.JOB_STATE_FAILED,
                job_pb2.JOB_STATE_KILLED,
                job_pb2.JOB_STATE_UNSCHEDULABLE,
            ):
                return False
            ctx.jobs.delete_job(ctx.cur, job_id.to_wire())
            self._record_transaction(
                ctx.cur, "remove_finished_job", [("job_removed", job_id.to_wire(), {"state": state})]
            )
            return True

    def remove_worker(self, worker_id: WorkerId) -> WorkerDetailRow | None:
        with self._stores.transact() as ctx:
            row = ctx.workers.get_row(ctx.cur, str(worker_id))
            if row is None:
                return None
            ctx.workers.remove(ctx.cur, str(worker_id))
            self._record_transaction(ctx.cur, "remove_worker", [("worker_removed", str(worker_id), {})])
        return row

    def prune_worker_task_history(self) -> int:
        """Trim worker_task_history to WORKER_TASK_HISTORY_RETENTION rows per worker."""
        with self._stores.transact() as ctx:
            return self._stores.workers.prune_task_history(ctx.cur, WORKER_TASK_HISTORY_RETENTION)

    def prune_worker_resource_history(self) -> int:
        """Trim worker_resource_history to WORKER_RESOURCE_HISTORY_RETENTION rows per worker."""
        with self._stores.transact() as ctx:
            return self._stores.workers.prune_resource_history(ctx.cur, WORKER_RESOURCE_HISTORY_RETENTION)

    def prune_task_resource_history(self) -> int:
        now_ms = Timestamp.now().epoch_ms()
        ttl_cutoff_ms = now_ms - TASK_RESOURCE_HISTORY_TERMINAL_TTL.to_ms()
        terminal_placeholders = sql_placeholders(len(TERMINAL_TASK_STATES))

        with self._stores.read() as rctx:
            terminal_ids = [
                str(r["task_id"])
                for r in rctx.cur.execute(
                    f"SELECT task_id FROM tasks "
                    f"WHERE state IN ({terminal_placeholders}) "
                    f"AND finished_at_ms IS NOT NULL AND finished_at_ms < ?",
                    (*TERMINAL_TASK_STATES, ttl_cutoff_ms),
                ).fetchall()
            ]

        evicted_terminal = 0
        for chunk_start in range(0, len(terminal_ids), TASK_RESOURCE_HISTORY_DELETE_CHUNK):
            chunk = terminal_ids[chunk_start : chunk_start + TASK_RESOURCE_HISTORY_DELETE_CHUNK]
            ph = sql_placeholders(len(chunk))
            with self._stores.transact() as ctx:
                ctx.cur.execute(f"DELETE FROM task_resource_history WHERE task_id IN ({ph})", tuple(chunk))
                evicted_terminal += ctx.cur.rowcount

        with self._stores.transact() as ctx:
            total_deleted = self._stores.tasks.prune_task_resource_history(ctx.cur, TASK_RESOURCE_HISTORY_RETENTION)
        if evicted_terminal > 0:
            logger.info("Evicted %d task_resource_history rows (terminal TTL)", evicted_terminal)
        return evicted_terminal + total_deleted

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

        def _stopped() -> bool:
            return stop_event is not None and stop_event.is_set()

        # 1. Jobs: one at a time (CASCADE to tasks -> attempts, endpoints)
        jobs_deleted = 0
        while not _stopped():
            with self._stores.read() as read_ctx:
                job_ids = self._stores.jobs.get_finished_jobs_before(read_ctx.cur, job_cutoff_ms)
            if not job_ids:
                break
            job_id = job_ids[0]
            with self._stores.transact() as ctx:
                # Invalidate endpoint cache BEFORE the CASCADE so the registry
                # drops rows SQLite is about to delete for us.
                ctx.endpoints.remove_by_job_ids(ctx.cur, [JobName.from_wire(job_id)])
                ctx.jobs.delete_job(ctx.cur, job_id)
                self._record_transaction(ctx.cur, "prune_old_data", [("job_pruned", job_id, {})])
            jobs_deleted += 1
            time.sleep(pause_between_s)

        # 2. Workers: one at a time (CASCADE to attributes, task_history, resource_history)
        workers_deleted = 0
        while not _stopped():
            with self._stores.read() as read_ctx:
                worker_id = self._stores.workers.get_inactive_worker_before(read_ctx.cur, worker_cutoff_ms)
            if worker_id is None:
                break
            with self._stores.transact() as ctx:
                ctx.workers.remove(ctx.cur, worker_id)
                self._record_transaction(ctx.cur, "prune_old_data", [("worker_pruned", worker_id, {})])
            workers_deleted += 1
            time.sleep(pause_between_s)

        # 3. txn_actions: batch of 1000 per transaction (no CASCADE)
        txn_actions_deleted = batch_delete(
            self._db,
            "DELETE FROM txn_actions WHERE rowid IN "
            "(SELECT rowid FROM txn_actions WHERE created_at_ms < ? LIMIT 1000)",
            (txn_cutoff_ms,),
            _stopped,
            pause_between_s,
        )

        # 4. Task profiles: batch of 1000 per transaction
        profile_cutoff_ms = now_ms - profile_retention.to_ms()
        # 4a. Delete stale profiles by age.
        profiles_deleted = batch_delete(
            self._db,
            "DELETE FROM profiles.task_profiles WHERE rowid IN "
            "(SELECT rowid FROM profiles.task_profiles WHERE captured_at_ms < ? LIMIT 1000)",
            (profile_cutoff_ms,),
            _stopped,
            pause_between_s,
        )
        # 4b. Delete orphan profiles whose task no longer exists.
        profiles_deleted += batch_delete(
            self._db,
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
    # Heartbeat Dispatch API
    # =========================================================================

    def buffer_dispatch(self, worker_id: WorkerId, task_request: job_pb2.RunTaskRequest) -> None:
        """Buffer a task dispatch for the next heartbeat.

        Called by the scheduling thread after committing resources via TaskAssignedEvent.
        The dispatch will be delivered when begin_heartbeat() drains the buffer.
        """
        with self._stores.transact() as ctx:
            ctx.dispatch.enqueue_run(
                ctx.cur, str(worker_id), task_request.SerializeToString(), Timestamp.now().epoch_ms()
            )

    def buffer_kill(self, worker_id: WorkerId, task_id: str) -> None:
        """Buffer a task kill for the next heartbeat.

        Called when a task needs to be terminated on a worker. The kill will be
        delivered when begin_heartbeat() drains the buffer.
        """
        with self._stores.transact() as ctx:
            ctx.dispatch.enqueue_kill(ctx.cur, str(worker_id), task_id, Timestamp.now().epoch_ms())

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
        with self._stores.transact() as ctx:
            return ctx.endpoints.add(ctx.cur, endpoint)

    def remove_endpoint(self, endpoint_id: str) -> EndpointRow | None:
        with self._stores.transact() as ctx:
            return ctx.endpoints.remove(ctx.cur, endpoint_id)

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
        with self._stores.transact() as ctx:
            now_ms = Timestamp.now().epoch_ms()

            newly_promoted: set[str] = set()
            tasks_to_run: list[job_pb2.RunTaskRequest] = []

            if max_promotions <= 0:
                pending_rows = []
            else:
                pending_rows = ctx.tasks.get_pending_for_direct_provider(ctx.cur, max_promotions)

            for row in pending_rows:
                task_id = str(row["task_id"])
                attempt_id = int(row["current_attempt_id"]) + 1
                resources = resource_spec_from_scalars(
                    int(row["res_cpu_millicores"]),
                    int(row["res_memory_bytes"]),
                    int(row["res_disk_bytes"]),
                    row["res_device_json"],
                )

                ctx.tasks.assign_direct(
                    ctx.cur,
                    DirectAssignment(
                        task_id=task_id,
                        attempt_id=attempt_id,
                        now_ms=now_ms,
                    ),
                )

                entrypoint = proto_from_json(str(row["entrypoint_json"]), job_pb2.RuntimeEntrypoint)
                job_id_wire = str(row["job_id"])
                for fn, data in ctx.jobs.get_workdir_files(ctx.cur, job_id_wire).items():
                    entrypoint.workdir_files[fn] = data

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
            running_rows = ctx.tasks.query(
                ctx.cur,
                TaskFilter(worker_is_null=True, states=ACTIVE_TASK_STATES),
            )
            running_tasks = [
                RunningTaskEntry(
                    task_id=t.task_id,
                    attempt_id=t.current_attempt_id,
                )
                for t in running_rows
                if t.task_id.to_wire() not in newly_promoted
            ]

            # Drain kill entries with NULL worker_id.
            tasks_to_kill = ctx.dispatch.drain_direct_kills(ctx.cur)

        return DirectProviderBatch(
            tasks_to_run=tasks_to_run,
            running_tasks=running_tasks,
            tasks_to_kill=tasks_to_kill,
        )

    def apply_direct_provider_updates(self, updates: list[TaskUpdate]) -> TxResult:
        """Apply a batch of task state updates from a KubernetesProvider.

        Delegates to _apply_task_transitions with a synthetic HeartbeatApplyRequest.
        Direct-provider tasks have worker_id=None in their snapshots, so
        TaskStore.terminate/requeue correctly skip resource decommit.
        """
        if not updates:
            return TxResult()
        with self._stores.transact() as ctx:
            now_ms = Timestamp.now().epoch_ms()
            req = HeartbeatApplyRequest(
                worker_id=WorkerId("direct"),
                worker_resource_snapshot=None,
                updates=updates,
            )
            return self._apply_task_transitions(ctx, req, now_ms)

    def buffer_direct_kill(self, task_id: str) -> None:
        """Buffer a kill request for a direct-provider task.

        Inserts a kill entry into dispatch_queue with worker_id=NULL.
        Drained by drain_for_direct_provider().
        """
        with self._stores.transact() as ctx:
            ctx.dispatch.enqueue_kill(ctx.cur, None, task_id, Timestamp.now().epoch_ms())
