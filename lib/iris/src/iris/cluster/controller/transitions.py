# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller state machine: all DB-mutating transitions live here.

Read-only queries do NOT belong here — callers use db.snapshot() directly.
"""

import enum
import json
import logging
import tempfile
from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, NamedTuple

from iris.cluster.constraints import AttributeValue, Constraint, constraints_from_resources, merge_constraints
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    ENDPOINTS,
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    WORKERS,
    ControllerDB,
    Endpoint,
    EndpointQuery,
    Join,
    TransactionCursor,
    Worker,
    ENDPOINT_TASKS,
    endpoint_query_predicate,
)
from iris.cluster.controller.checkpoint import (
    ScalingGroupSnapshotData,
    TrackedWorkerSnapshotData,
    serialize_scaling_group,
)
from iris.cluster.log_store import LogStore, task_log_key
from iris.cluster.types import (
    JobName,
    WorkerId,
    get_gpu_count,
    get_tpu_count,
)
from iris.rpc import cluster_pb2, logging_pb2
from iris.time_utils import Duration, Timestamp

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

WORKER_TASK_HISTORY_RETENTION = 500
"""Maximum worker_task_history rows retained per worker."""

WORKER_RESOURCE_HISTORY_RETENTION = 500
"""Maximum worker_resource_history rows retained per worker."""


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
    metadata: cluster_pb2.WorkerMetadata


@dataclass(frozen=True)
class TaskUpdate:
    """Single task state update applied in a batch."""

    task_id: JobName
    attempt_id: int
    new_state: int
    error: str | None = None
    exit_code: int | None = None
    resource_usage: cluster_pb2.ResourceUsage | None = None
    log_entries: list[logging_pb2.LogEntry] = field(default_factory=list)


@dataclass(frozen=True)
class HeartbeatApplyRequest:
    """Batch of worker heartbeat updates applied atomically."""

    worker_id: WorkerId
    worker_resource_snapshot: cluster_pb2.WorkerResourceSnapshot | None
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


class RunningTaskEntry(NamedTuple):
    """Task ID and attempt ID pair captured at snapshot time."""

    task_id: JobName
    attempt_id: int


@dataclass(frozen=True)
class DispatchBatch:
    """Drained worker dispatch plus running-task snapshot."""

    worker_id: WorkerId
    worker_address: str
    vm_address: str
    running_tasks: list[RunningTaskEntry]
    tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest] = field(default_factory=list)
    tasks_to_kill: list[str] = field(default_factory=list)


def _decommit_worker_resources(
    cur: TransactionCursor,
    worker_id: str,
    resources: "cluster_pb2.ResourceSpecProto",
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


# =============================================================================
# Controller Transitions
# =============================================================================


class ControllerTransitions:
    """State machine for controller entities.

    All methods that mutate DB state live here. Each is a single atomic
    transaction. Read-only queries do NOT belong here — callers use
    db.snapshot() directly.

    SQLite is the sole source of truth. Any in-memory values are transient
    helpers and must never be required for correctness across restarts.
    """

    def __init__(
        self,
        heartbeat_failure_threshold: int = HEARTBEAT_FAILURE_THRESHOLD,
        log_dir: Path | None = None,
        db_path: Path | None = None,
        db: ControllerDB | None = None,
        log_store: LogStore | None = None,
    ):
        self._heartbeat_failure_threshold = heartbeat_failure_threshold
        if db is not None:
            self._db = db
        else:
            if db_path is None:
                if log_dir is not None:
                    db_path = log_dir / "controller.sqlite3"
                else:
                    tmp = Path(tempfile.mkdtemp(prefix="iris_controller_state_"))
                    db_path = tmp / "controller.sqlite3"
            self._db = ControllerDB(db_path=db_path)
        self._owns_log_store = log_store is None
        if log_store is None:
            log_store = LogStore(db_path=self._db.db_path)
        self._log_store: LogStore = log_store

    @property
    def log_store(self) -> LogStore:
        return self._log_store

    @property
    def db(self) -> ControllerDB:
        return self._db

    @property
    def db_path(self) -> Path:
        return self._db.db_path

    def close(self) -> None:
        if self._owns_log_store:
            self._log_store.close()
        self._db.close()

    def backup_to(self, destination: Path) -> None:
        self._db.backup_to(destination)

    def restore_from(self, source: str | Path) -> None:
        self._db.replace_from(source)

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
            "SELECT request_proto, state, started_at_ms FROM jobs WHERE job_id = ?",
            (job_id.to_wire(),),
        ).fetchone()
        if row is None:
            return None
        req = cluster_pb2.Controller.LaunchJobRequest()
        req.ParseFromString(row["request_proto"])
        counts_rows = cur.execute(
            "SELECT state, COUNT(*) AS c FROM tasks WHERE job_id = ? GROUP BY state",
            (job_id.to_wire(),),
        ).fetchall()
        counts = {int(r["state"]): int(r["c"]) for r in counts_rows}
        total = sum(counts.values())
        current_state = int(row["state"])
        new_state = current_state
        now_ms = Timestamp.now().epoch_ms()
        if total > 0 and counts.get(cluster_pb2.TASK_STATE_SUCCEEDED, 0) == total:
            new_state = cluster_pb2.JOB_STATE_SUCCEEDED
        elif counts.get(cluster_pb2.TASK_STATE_FAILED, 0) > int(req.max_task_failures):
            new_state = cluster_pb2.JOB_STATE_FAILED
        elif counts.get(cluster_pb2.TASK_STATE_UNSCHEDULABLE, 0) > 0:
            new_state = cluster_pb2.JOB_STATE_UNSCHEDULABLE
        elif counts.get(cluster_pb2.TASK_STATE_KILLED, 0) > 0:
            new_state = cluster_pb2.JOB_STATE_KILLED
        elif (
            counts.get(cluster_pb2.TASK_STATE_ASSIGNED, 0) > 0
            or counts.get(cluster_pb2.TASK_STATE_BUILDING, 0) > 0
            or counts.get(cluster_pb2.TASK_STATE_RUNNING, 0) > 0
        ):
            new_state = cluster_pb2.JOB_STATE_RUNNING
        elif row["started_at_ms"] is not None:
            # Retries put tasks back into PENDING; keep job running once it has started.
            new_state = cluster_pb2.JOB_STATE_RUNNING
        elif total > 0:
            new_state = cluster_pb2.JOB_STATE_PENDING
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
            "error = CASE WHEN ? IN (?, ?, ?) THEN ? ELSE error END "
            "WHERE job_id = ?",
            (
                new_state,
                new_state,
                cluster_pb2.JOB_STATE_RUNNING,
                now_ms,
                new_state,
                *TERMINAL_JOB_STATES,
                now_ms,
                new_state,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
                cluster_pb2.JOB_STATE_UNSCHEDULABLE,
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

    def persist_checkpoint_state(
        self,
        scaling_groups: list[ScalingGroupSnapshotData],
        tracked_workers: list[TrackedWorkerSnapshotData],
    ) -> None:
        """Persist checkpoint metadata rows into the controller DB."""

        with self._db.transaction() as cur:
            cur.execute("DELETE FROM scaling_groups")
            for group in scaling_groups:
                cur.execute(
                    "INSERT INTO scaling_groups(name, snapshot_proto, updated_at_ms) VALUES (?, ?, ?)",
                    (
                        group.name,
                        serialize_scaling_group(group),
                        Timestamp.now().epoch_ms(),
                    ),
                )

            cur.execute("DELETE FROM tracked_workers")
            for worker in tracked_workers:
                cur.execute(
                    "INSERT INTO tracked_workers(worker_id, slice_id, scale_group, internal_address) "
                    "VALUES (?, ?, ?, ?)",
                    (worker.worker_id, worker.slice_id, worker.scale_group, worker.internal_address),
                )

    # =========================================================================
    # Command API
    # =========================================================================

    def submit_job(
        self,
        job_id: JobName,
        request: cluster_pb2.Controller.LaunchJobRequest,
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
                    .add(Duration.from_proto(request.scheduling_timeout))
                    .epoch_ms()
                )

            cur.execute(
                "INSERT OR IGNORE INTO users(user_id, created_at_ms) VALUES (?, ?)",
                (job_id.user, effective_submission_ms),
            )

            replicas = int(request.replicas)
            validation_error: str | None = None
            if replicas < 1:
                validation_error = f"Job {job_id} has invalid replicas={replicas}; must be >= 1"
                replicas = 0
            elif replicas > MAX_REPLICAS_PER_JOB:
                validation_error = f"Job {job_id} replicas={replicas} exceeds max {MAX_REPLICAS_PER_JOB}"
                replicas = 0

            state = cluster_pb2.JOB_STATE_PENDING if validation_error is None else cluster_pb2.JOB_STATE_FAILED
            finished_ms = None if validation_error is None else effective_submission_ms
            cur.execute(
                "INSERT INTO jobs("
                "job_id, user_id, parent_job_id, root_job_id, depth, request_proto, state, submitted_at_ms, "
                "root_submitted_at_ms, started_at_ms, finished_at_ms, scheduling_deadline_epoch_ms, "
                "error, exit_code, num_tasks, is_reservation_holder"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?, 0)",
                (
                    job_id.to_wire(),
                    job_id.user,
                    parent_job_id,
                    job_id.root_job.to_wire(),
                    job_id.depth,
                    request.SerializeToString(),
                    state,
                    effective_submission_ms,
                    root_submitted_ms,
                    finished_ms,
                    deadline_epoch_ms,
                    validation_error,
                    replicas,
                ),
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
                        "resource_usage_proto, current_attempt_id, priority_neg_depth, priority_root_submitted_ms, "
                        "priority_insertion"
                        ") VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?, 0, 0, NULL, -1, ?, ?, ?)",
                        (
                            task_id,
                            job_id.to_wire(),
                            idx,
                            cluster_pb2.TASK_STATE_PENDING,
                            effective_submission_ms,
                            int(request.max_retries_failure),
                            int(request.max_retries_preemption),
                            -job_id.depth,
                            root_submitted_ms,
                            insertion_base + idx,
                        ),
                    )
                if request.HasField("reservation") and request.reservation.entries:
                    holder_id = job_id.child(RESERVATION_HOLDER_JOB_NAME)
                    entry = request.reservation.entries[0]
                    holder_request = cluster_pb2.Controller.LaunchJobRequest(
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
                    cur.execute(
                        "INSERT INTO jobs("
                        "job_id, user_id, parent_job_id, root_job_id, depth, request_proto, state, submitted_at_ms, "
                        "root_submitted_at_ms, started_at_ms, finished_at_ms, scheduling_deadline_epoch_ms, "
                        "error, exit_code, num_tasks, is_reservation_holder"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, 1)",
                        (
                            holder_id.to_wire(),
                            holder_id.user,
                            job_id.to_wire(),
                            holder_id.root_job.to_wire(),
                            holder_id.depth,
                            holder_request.SerializeToString(),
                            cluster_pb2.JOB_STATE_PENDING,
                            effective_submission_ms,
                            root_submitted_ms,
                            len(request.reservation.entries),
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
                            "resource_usage_proto, current_attempt_id, priority_neg_depth, priority_root_submitted_ms, "
                            "priority_insertion"
                            ") VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?, 0, 0, NULL, -1, ?, ?, ?)",
                            (
                                holder_id.task(idx).to_wire(),
                                holder_id.to_wire(),
                                idx,
                                cluster_pb2.TASK_STATE_PENDING,
                                effective_submission_ms,
                                0,
                                DEFAULT_MAX_RETRIES_PREEMPTION,
                                -holder_id.depth,
                                root_submitted_ms,
                                holder_base + idx,
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
                f"SELECT t.task_id FROM tasks t "
                f"JOIN task_attempts a ON a.task_id = t.task_id AND a.attempt_id = t.current_attempt_id "
                f"WHERE t.job_id IN ({placeholders}) "
                "AND t.state IN (?, ?, ?) "
                "AND a.worker_id IS NOT NULL",
                (
                    *subtree_ids,
                    cluster_pb2.TASK_STATE_ASSIGNED,
                    cluster_pb2.TASK_STATE_BUILDING,
                    cluster_pb2.TASK_STATE_RUNNING,
                ),
            ).fetchall()
            tasks_to_kill = {JobName.from_wire(str(row["task_id"])) for row in running_rows}
            now_ms = Timestamp.now().epoch_ms()
            task_terminal_placeholders = ",".join("?" for _ in TERMINAL_TASK_STATES)
            cur.execute(
                f"UPDATE tasks SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
                f"WHERE job_id IN ({placeholders}) AND state NOT IN ({task_terminal_placeholders})",
                (
                    cluster_pb2.TASK_STATE_KILLED,
                    reason,
                    now_ms,
                    *subtree_ids,
                    *TERMINAL_TASK_STATES,
                ),
            )
            # Deliberately excludes JOB_STATE_WORKER_FAILED from the guard set:
            # worker-failed jobs should still be cancellable (transitioned to KILLED).
            cancel_guard_states = TERMINAL_JOB_STATES - {cluster_pb2.JOB_STATE_WORKER_FAILED}
            cancel_guard_placeholders = ",".join("?" for _ in cancel_guard_states)
            cur.execute(
                f"UPDATE jobs SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
                f"WHERE job_id IN ({placeholders}) AND state NOT IN ({cancel_guard_placeholders})",
                (
                    cluster_pb2.JOB_STATE_KILLED,
                    reason,
                    now_ms,
                    *subtree_ids,
                    *cancel_guard_states,
                ),
            )
            self._record_transaction(cur, "cancel_job", [("job_cancelled", job_id.to_wire(), {"reason": reason})])
            return TxResult(tasks_to_kill=tasks_to_kill)

    def register_or_refresh_worker(
        self,
        worker_id: WorkerId,
        address: str,
        metadata: cluster_pb2.WorkerMetadata,
        ts: Timestamp,
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
        with self._db.transaction() as cur:
            cur.execute(
                "INSERT INTO workers("
                "worker_id, address, metadata_proto, healthy, active, consecutive_failures, last_heartbeat_ms, "
                "committed_cpu_millicores, committed_mem_bytes, committed_gpu, committed_tpu, resource_snapshot_proto"
                ") VALUES (?, ?, ?, 1, 1, 0, ?, 0, 0, 0, 0, NULL) "
                "ON CONFLICT(worker_id) DO UPDATE SET "
                "address=excluded.address, metadata_proto=excluded.metadata_proto, healthy=1, active=1, "
                "consecutive_failures=0, last_heartbeat_ms=excluded.last_heartbeat_ms",
                (str(worker_id), address, metadata.SerializeToString(), now_ms),
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
        return TxResult()

    def register_worker(
        self,
        worker_id: WorkerId,
        address: str,
        metadata: cluster_pb2.WorkerMetadata,
        ts: Timestamp,
    ) -> WorkerRegistrationResult:
        self.register_or_refresh_worker(worker_id=worker_id, address=address, metadata=metadata, ts=ts)
        return WorkerRegistrationResult(worker_id=worker_id)

    def queue_assignments(self, assignments: list[Assignment]) -> AssignmentResult:
        """Commit assignments and enqueue dispatches in one transaction."""
        accepted: list[Assignment] = []
        rejected: list[Assignment] = []
        has_real_dispatch = False
        with self._db.transaction() as cur:
            for assignment in assignments:
                task_row = cur.execute(
                    "SELECT * FROM tasks WHERE task_id = ?", (assignment.task_id.to_wire(),)
                ).fetchone()
                worker_row = cur.execute(
                    "SELECT * FROM workers WHERE worker_id = ? AND active = 1 AND healthy = 1",
                    (str(assignment.worker_id),),
                ).fetchone()
                if task_row is None or worker_row is None:
                    rejected.append(assignment)
                    continue
                task = self._db.decode_task(task_row)
                if not task.can_be_scheduled():
                    rejected.append(assignment)
                    continue
                job_row = cur.execute("SELECT * FROM jobs WHERE job_id = ?", (task.job_id.to_wire(),)).fetchone()
                if job_row is None:
                    rejected.append(assignment)
                    continue
                job = self._db.decode_job(job_row)
                attempt_id = int(task_row["current_attempt_id"]) + 1
                now_ms = Timestamp.now().epoch_ms()
                cur.execute(
                    "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        assignment.task_id.to_wire(),
                        attempt_id,
                        str(assignment.worker_id),
                        cluster_pb2.TASK_STATE_ASSIGNED,
                        now_ms,
                    ),
                )
                cur.execute(
                    "UPDATE tasks SET state = ?, current_attempt_id = ?, "
                    "started_at_ms = COALESCE(started_at_ms, ?) WHERE task_id = ?",
                    (cluster_pb2.TASK_STATE_ASSIGNED, attempt_id, now_ms, assignment.task_id.to_wire()),
                )
                if not job.is_reservation_holder:
                    resources = job.request.resources
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
                    run_request = cluster_pb2.Worker.RunTaskRequest(
                        task_id=assignment.task_id.to_wire(),
                        num_tasks=job.num_tasks,
                        entrypoint=job.request.entrypoint,
                        environment=job.request.environment,
                        bundle_id=job.request.bundle_id,
                        resources=resources,
                        ports=list(job.request.ports),
                        attempt_id=attempt_id,
                        constraints=list(job.request.constraints),
                    )
                    if job.request.timeout.milliseconds > 0:
                        run_request.timeout.CopyFrom(job.request.timeout)
                    cur.execute(
                        "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
                        "VALUES (?, 'run', ?, NULL, ?)",
                        (str(assignment.worker_id), run_request.SerializeToString(), now_ms),
                    )
                    has_real_dispatch = True
                cur.execute(
                    "INSERT INTO worker_task_history(worker_id, task_id, assigned_at_ms) VALUES (?, ?, ?)",
                    (str(assignment.worker_id), assignment.task_id.to_wire(), now_ms),
                )
                cur.execute(
                    "DELETE FROM worker_task_history "
                    "WHERE worker_id = ? "
                    "AND id NOT IN ("
                    "  SELECT id FROM worker_task_history "
                    "  WHERE worker_id = ? "
                    "  ORDER BY assigned_at_ms DESC, id DESC LIMIT ?"
                    ")",
                    (
                        str(assignment.worker_id),
                        str(assignment.worker_id),
                        WORKER_TASK_HISTORY_RETENTION,
                    ),
                )
                cur.execute(
                    "UPDATE jobs SET state = CASE WHEN state = ? THEN ? ELSE state END, "
                    "started_at_ms = COALESCE(started_at_ms, ?) WHERE job_id = ?",
                    (cluster_pb2.JOB_STATE_PENDING, cluster_pb2.JOB_STATE_RUNNING, now_ms, task.job_id.to_wire()),
                )
                accepted.append(assignment)
            if accepted or rejected:
                actions = [("assignment_queued", a.task_id.to_wire(), {"worker_id": str(a.worker_id)}) for a in accepted]
                self._record_transaction(cur, "queue_assignments", actions)
        return AssignmentResult(
            tasks_to_kill=set(), has_real_dispatch=has_real_dispatch, accepted=accepted, rejected=rejected
        )

    def apply_task_updates(self, req: HeartbeatApplyRequest) -> TxResult:
        """Apply a batch of worker task updates atomically."""
        pending_logs: list[tuple[str, list[logging_pb2.LogEntry]]] = []
        tasks_to_kill: set[JobName] = set()

        with self._db.transaction() as cur:
            worker = cur.execute("SELECT * FROM workers WHERE worker_id = ?", (str(req.worker_id),)).fetchone()
            if worker is None:
                return TxResult()
            now_ms = Timestamp.now().epoch_ms()
            snapshot_payload = (
                req.worker_resource_snapshot.SerializeToString() if req.worker_resource_snapshot is not None else None
            )
            cur.execute(
                "UPDATE workers SET healthy = 1, active = 1, consecutive_failures = 0, last_heartbeat_ms = ?, "
                "resource_snapshot_proto = COALESCE(?, resource_snapshot_proto) WHERE worker_id = ?",
                (now_ms, snapshot_payload, str(req.worker_id)),
            )
            if snapshot_payload is not None:
                cur.execute(
                    "INSERT INTO worker_resource_history(worker_id, snapshot_proto, timestamp_ms) VALUES (?, ?, ?)",
                    (str(req.worker_id), snapshot_payload, now_ms),
                )
                cur.execute(
                    "DELETE FROM worker_resource_history "
                    "WHERE worker_id = ? "
                    "AND id NOT IN ("
                    "  SELECT id FROM worker_resource_history "
                    "  WHERE worker_id = ? "
                    "  ORDER BY id DESC LIMIT ?"
                    ")",
                    (
                        str(req.worker_id),
                        str(req.worker_id),
                        WORKER_RESOURCE_HISTORY_RETENTION,
                    ),
                )

            for update in req.updates:
                task_row = cur.execute("SELECT * FROM tasks WHERE task_id = ?", (update.task_id.to_wire(),)).fetchone()
                if task_row is None:
                    continue
                task = self._db.decode_task(task_row)
                if task.is_finished() or update.new_state in (
                    cluster_pb2.TASK_STATE_UNSPECIFIED,
                    cluster_pb2.TASK_STATE_PENDING,
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
                worker_id = attempt_row["worker_id"]
                if update.log_entries and self._log_store is not None:
                    pending_logs.append((task_log_key(update.task_id, update.attempt_id), update.log_entries))
                usage_payload = update.resource_usage.SerializeToString() if update.resource_usage is not None else None
                if usage_payload is not None:
                    cur.execute(
                        "UPDATE tasks SET resource_usage_proto = ? WHERE task_id = ?",
                        (usage_payload, update.task_id.to_wire()),
                    )

                terminal_ms: int | None = None
                started_ms: int | None = None
                task_state = int(task_row["state"])
                task_error = update.error
                task_exit = update.exit_code
                failure_count = int(task_row["failure_count"])
                preemption_count = int(task_row["preemption_count"])

                if update.new_state == cluster_pb2.TASK_STATE_RUNNING:
                    started_ms = now_ms
                    task_state = cluster_pb2.TASK_STATE_RUNNING
                elif update.new_state == cluster_pb2.TASK_STATE_BUILDING:
                    task_state = cluster_pb2.TASK_STATE_BUILDING
                elif update.new_state in (
                    cluster_pb2.TASK_STATE_FAILED,
                    cluster_pb2.TASK_STATE_WORKER_FAILED,
                    cluster_pb2.TASK_STATE_KILLED,
                    cluster_pb2.TASK_STATE_UNSCHEDULABLE,
                    cluster_pb2.TASK_STATE_SUCCEEDED,
                ):
                    terminal_ms = now_ms
                    task_state = int(update.new_state)
                    if update.new_state == cluster_pb2.TASK_STATE_SUCCEEDED and task_exit is None:
                        task_exit = 0
                    if update.new_state == cluster_pb2.TASK_STATE_UNSCHEDULABLE and task_error is None:
                        task_error = "Scheduling timeout exceeded"
                    if update.new_state == cluster_pb2.TASK_STATE_FAILED:
                        failure_count += 1
                    if update.new_state == cluster_pb2.TASK_STATE_WORKER_FAILED and int(task_row["state"]) in (
                        cluster_pb2.TASK_STATE_BUILDING,
                        cluster_pb2.TASK_STATE_RUNNING,
                    ):
                        preemption_count += 1
                    if (
                        update.new_state == cluster_pb2.TASK_STATE_WORKER_FAILED
                        and int(task_row["state"]) == cluster_pb2.TASK_STATE_ASSIGNED
                    ):
                        task_state = cluster_pb2.TASK_STATE_PENDING
                        terminal_ms = None
                    if update.new_state == cluster_pb2.TASK_STATE_FAILED and failure_count <= int(
                        task_row["max_retries_failure"]
                    ):
                        task_state = cluster_pb2.TASK_STATE_PENDING
                        terminal_ms = None
                    if (
                        update.new_state == cluster_pb2.TASK_STATE_WORKER_FAILED
                        and preemption_count <= int(task_row["max_retries_preemption"])
                        and int(task_row["state"]) in (cluster_pb2.TASK_STATE_BUILDING, cluster_pb2.TASK_STATE_RUNNING)
                    ):
                        task_state = cluster_pb2.TASK_STATE_PENDING
                        terminal_ms = None

                cur.execute(
                    "UPDATE task_attempts SET state = ?, started_at_ms = COALESCE(started_at_ms, ?), "
                    "finished_at_ms = COALESCE(finished_at_ms, ?), exit_code = COALESCE(?, exit_code), "
                    "error = COALESCE(?, error) WHERE task_id = ? AND attempt_id = ?",
                    (
                        int(update.new_state),
                        started_ms,
                        terminal_ms,
                        task_exit,
                        update.error,
                        update.task_id.to_wire(),
                        update.attempt_id,
                    ),
                )
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
                if worker_id is not None and task_state in (
                    cluster_pb2.TASK_STATE_PENDING,
                    cluster_pb2.TASK_STATE_SUCCEEDED,
                    cluster_pb2.TASK_STATE_FAILED,
                    cluster_pb2.TASK_STATE_KILLED,
                    cluster_pb2.TASK_STATE_UNSCHEDULABLE,
                    cluster_pb2.TASK_STATE_WORKER_FAILED,
                ):
                    job_row = cur.execute(
                        "SELECT request_proto FROM jobs WHERE job_id = ?", (task.job_id.to_wire(),)
                    ).fetchone()
                    if job_row is not None:
                        job_req = cluster_pb2.Controller.LaunchJobRequest()
                        job_req.ParseFromString(job_row["request_proto"])
                        _decommit_worker_resources(cur, str(worker_id), job_req.resources)
                    cur.execute("DELETE FROM endpoints WHERE task_id = ?", (update.task_id.to_wire(),))

                job_row_full = cur.execute(
                    "SELECT request_proto FROM jobs WHERE job_id = ?",
                    (task.job_id.to_wire(),),
                ).fetchone()
                job_req = None
                if job_row_full is not None:
                    job_req = cluster_pb2.Controller.LaunchJobRequest()
                    job_req.ParseFromString(job_row_full["request_proto"])

                # Coscheduled jobs: a terminal host failure should cascade to siblings.
                if (
                    job_req is not None
                    and job_req.HasField("coscheduling")
                    and task_state in (cluster_pb2.TASK_STATE_FAILED, cluster_pb2.TASK_STATE_WORKER_FAILED)
                ):
                    sibling_rows = cur.execute(
                        "SELECT t.task_id, t.current_attempt_id, t.max_retries_preemption, a.worker_id "
                        "FROM tasks t LEFT JOIN task_attempts a "
                        "ON a.task_id = t.task_id AND a.attempt_id = t.current_attempt_id "
                        "WHERE t.job_id = ? AND t.task_id != ? AND t.state IN (?, ?, ?)",
                        (
                            task.job_id.to_wire(),
                            update.task_id.to_wire(),
                            cluster_pb2.TASK_STATE_ASSIGNED,
                            cluster_pb2.TASK_STATE_BUILDING,
                            cluster_pb2.TASK_STATE_RUNNING,
                        ),
                    ).fetchall()
                    for sibling in sibling_rows:
                        sibling_task_id = str(sibling["task_id"])
                        sibling_worker_id = sibling["worker_id"]
                        cur.execute(
                            "UPDATE task_attempts SET state = ?, "
                            "finished_at_ms = COALESCE(finished_at_ms, ?), error = ? "
                            "WHERE task_id = ? AND attempt_id = ?",
                            (
                                cluster_pb2.TASK_STATE_WORKER_FAILED,
                                now_ms,
                                f"Coscheduled sibling {update.task_id.to_wire()} failed",
                                sibling_task_id,
                                int(sibling["current_attempt_id"]),
                            ),
                        )
                        cur.execute(
                            "UPDATE tasks SET state = ?, finished_at_ms = ?, preemption_count = ?, error = ? "
                            "WHERE task_id = ?",
                            (
                                cluster_pb2.TASK_STATE_WORKER_FAILED,
                                now_ms,
                                int(sibling["max_retries_preemption"]) + 1,
                                f"Coscheduled sibling {update.task_id.to_wire()} failed",
                                sibling_task_id,
                            ),
                        )
                        if sibling_worker_id is not None:
                            _decommit_worker_resources(cur, str(sibling_worker_id), job_req.resources)
                        cur.execute("DELETE FROM endpoints WHERE task_id = ?", (sibling_task_id,))
                        tasks_to_kill.add(JobName.from_wire(sibling_task_id))

                new_job_state = self._recompute_job_state(cur, task.job_id)
                if new_job_state in (
                    cluster_pb2.JOB_STATE_FAILED,
                    cluster_pb2.JOB_STATE_KILLED,
                    cluster_pb2.JOB_STATE_UNSCHEDULABLE,
                    cluster_pb2.JOB_STATE_SUCCEEDED,
                ):
                    reason = "Job finalized"
                    if new_job_state == cluster_pb2.JOB_STATE_FAILED:
                        reason = "Job exceeded max_task_failures"
                    elif new_job_state == cluster_pb2.JOB_STATE_KILLED:
                        reason = "Job was terminated."
                    elif new_job_state == cluster_pb2.JOB_STATE_UNSCHEDULABLE:
                        reason = "Job could not be scheduled."
                    remaining = cur.execute(
                        "SELECT t.task_id, t.current_attempt_id, t.state, a.worker_id, j.request_proto "
                        "FROM tasks t "
                        "JOIN jobs j ON j.job_id = t.job_id "
                        "LEFT JOIN task_attempts a ON a.task_id = t.task_id AND a.attempt_id = t.current_attempt_id "
                        "WHERE t.job_id = ? AND t.state NOT IN (?, ?, ?, ?, ?)",
                        (
                            task.job_id.to_wire(),
                            cluster_pb2.TASK_STATE_SUCCEEDED,
                            cluster_pb2.TASK_STATE_FAILED,
                            cluster_pb2.TASK_STATE_KILLED,
                            cluster_pb2.TASK_STATE_UNSCHEDULABLE,
                            cluster_pb2.TASK_STATE_WORKER_FAILED,
                        ),
                    ).fetchall()
                    for rem in remaining:
                        rem_task_id = str(rem["task_id"])
                        rem_worker = rem["worker_id"]
                        cur.execute(
                            "UPDATE tasks SET state = ?, finished_at_ms = COALESCE(finished_at_ms, ?), "
                            "error = ? WHERE task_id = ?",
                            (cluster_pb2.TASK_STATE_KILLED, now_ms, reason, rem_task_id),
                        )
                        if int(rem["current_attempt_id"]) >= 0:
                            cur.execute(
                                "UPDATE task_attempts SET state = ?, "
                                "finished_at_ms = COALESCE(finished_at_ms, ?), error = ? "
                                "WHERE task_id = ? AND attempt_id = ?",
                                (
                                    cluster_pb2.TASK_STATE_KILLED,
                                    now_ms,
                                    reason,
                                    rem_task_id,
                                    int(rem["current_attempt_id"]),
                                ),
                            )
                        if rem_worker is not None:
                            rem_job_req = cluster_pb2.Controller.LaunchJobRequest()
                            rem_job_req.ParseFromString(rem["request_proto"])
                            _decommit_worker_resources(cur, str(rem_worker), rem_job_req.resources)
                            tasks_to_kill.add(JobName.from_wire(rem_task_id))
                        cur.execute("DELETE FROM endpoints WHERE task_id = ?", (rem_task_id,))

                    # Cascade terminal parent state to descendants (child jobs).
                    descendants = cur.execute(
                        "WITH RECURSIVE subtree(job_id) AS ("
                        "  SELECT job_id FROM jobs WHERE parent_job_id = ? "
                        "  UNION ALL "
                        "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
                        ") SELECT job_id FROM subtree",
                        (task.job_id.to_wire(),),
                    ).fetchall()
                    for child_row in descendants:
                        child_job_id = str(child_row["job_id"])
                        child_tasks = cur.execute(
                            "SELECT t.task_id, t.current_attempt_id, a.worker_id, j.request_proto "
                            "FROM tasks t "
                            "JOIN jobs j ON j.job_id = t.job_id "
                            "LEFT JOIN task_attempts a ON a.task_id = t.task_id AND a.attempt_id = t.current_attempt_id "
                            "WHERE t.job_id = ? AND t.state NOT IN (?, ?, ?, ?, ?)",
                            (
                                child_job_id,
                                cluster_pb2.TASK_STATE_SUCCEEDED,
                                cluster_pb2.TASK_STATE_FAILED,
                                cluster_pb2.TASK_STATE_KILLED,
                                cluster_pb2.TASK_STATE_UNSCHEDULABLE,
                                cluster_pb2.TASK_STATE_WORKER_FAILED,
                            ),
                        ).fetchall()
                        for child_task in child_tasks:
                            child_task_id = str(child_task["task_id"])
                            child_worker = child_task["worker_id"]
                            cur.execute(
                                "UPDATE tasks SET state = ?, finished_at_ms = COALESCE(finished_at_ms, ?), "
                                "error = ? WHERE task_id = ?",
                                (cluster_pb2.TASK_STATE_KILLED, now_ms, "Parent job terminated", child_task_id),
                            )
                            if int(child_task["current_attempt_id"]) >= 0:
                                cur.execute(
                                    "UPDATE task_attempts SET state = ?, "
                                    "finished_at_ms = COALESCE(finished_at_ms, ?), error = ? "
                                    "WHERE task_id = ? AND attempt_id = ?",
                                    (
                                        cluster_pb2.TASK_STATE_KILLED,
                                        now_ms,
                                        "Parent job terminated",
                                        child_task_id,
                                        int(child_task["current_attempt_id"]),
                                    ),
                                )
                            if child_worker is not None:
                                child_req = cluster_pb2.Controller.LaunchJobRequest()
                                child_req.ParseFromString(child_task["request_proto"])
                                _decommit_worker_resources(cur, str(child_worker), child_req.resources)
                                tasks_to_kill.add(JobName.from_wire(child_task_id))
                            cur.execute("DELETE FROM endpoints WHERE task_id = ?", (child_task_id,))
                        cur.execute(
                            "UPDATE jobs SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
                            "WHERE job_id = ? AND state NOT IN (?, ?, ?, ?)",
                            (
                                cluster_pb2.JOB_STATE_KILLED,
                                "Parent job terminated",
                                now_ms,
                                child_job_id,
                                cluster_pb2.JOB_STATE_SUCCEEDED,
                                cluster_pb2.JOB_STATE_FAILED,
                                cluster_pb2.JOB_STATE_KILLED,
                                cluster_pb2.JOB_STATE_UNSCHEDULABLE,
                            ),
                        )
            self._record_transaction(cur, "apply_task_updates", [("heartbeat_applied", str(req.worker_id), {})])

        if pending_logs and self._log_store is not None:
            self._log_store.append_batch(pending_logs)

        return TxResult(tasks_to_kill=tasks_to_kill)

    def apply_heartbeat(self, req: HeartbeatApplyRequest) -> HeartbeatApplyResult:
        result = self.apply_task_updates(req)
        return HeartbeatApplyResult(tasks_to_kill=result.tasks_to_kill, action=HeartbeatAction.OK)

    def record_heartbeat_failure(
        self,
        worker_id: WorkerId,
        error: str,
        drained_dispatch: DispatchBatch,
        *,
        force_remove: bool = False,
    ) -> TxResult:
        """Record heartbeat failure and requeue/flush drained dispatches.

        Args:
            force_remove: If True, skip the consecutive-failure threshold and
                immediately remove the worker. Used when the worker self-reports
                as unhealthy.
        """
        tasks_to_kill: set[JobName] = set()
        with self._db.transaction() as cur:
            row = cur.execute(
                "SELECT consecutive_failures FROM workers WHERE worker_id = ? AND active = 1",
                (str(worker_id),),
            ).fetchone()
            if row is None:
                return TxResult()
            failures = int(row["consecutive_failures"]) + 1
            cur.execute(
                "UPDATE workers SET consecutive_failures = ?, healthy = CASE WHEN ? >= ? THEN 0 ELSE healthy END "
                "WHERE worker_id = ?",
                (failures, failures, self._heartbeat_failure_threshold, str(worker_id)),
            )
            should_remove = force_remove or failures >= self._heartbeat_failure_threshold
            if should_remove:
                task_rows = cur.execute(
                    "SELECT t.task_id, t.current_attempt_id, t.state, t.preemption_count, t.max_retries_preemption, "
                    "j.is_reservation_holder "
                    "FROM tasks t "
                    "JOIN task_attempts ta ON t.task_id = ta.task_id AND t.current_attempt_id = ta.attempt_id "
                    "JOIN jobs j ON j.job_id = t.job_id "
                    "WHERE ta.worker_id = ? AND t.state IN (?, ?, ?)",
                    (str(worker_id), *ACTIVE_TASK_STATES),
                ).fetchall()
                now_ms = Timestamp.now().epoch_ms()
                for task_row in task_rows:
                    tid = str(task_row["task_id"])
                    prior_state = int(task_row["state"])
                    preemption_count = int(task_row["preemption_count"])
                    max_preemptions = int(task_row["max_retries_preemption"])
                    is_reservation_holder = bool(int(task_row["is_reservation_holder"]))
                    new_task_state = cluster_pb2.TASK_STATE_WORKER_FAILED
                    finished_ms: int | None = now_ms
                    if is_reservation_holder:
                        new_task_state = cluster_pb2.TASK_STATE_PENDING
                        finished_ms = None
                    elif prior_state == cluster_pb2.TASK_STATE_ASSIGNED:
                        new_task_state = cluster_pb2.TASK_STATE_PENDING
                        finished_ms = None
                    elif prior_state in (cluster_pb2.TASK_STATE_BUILDING, cluster_pb2.TASK_STATE_RUNNING):
                        preemption_count += 1
                        if preemption_count <= max_preemptions:
                            new_task_state = cluster_pb2.TASK_STATE_PENDING
                            finished_ms = None
                    if is_reservation_holder:
                        cur.execute(
                            "DELETE FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
                            (tid, int(task_row["current_attempt_id"])),
                        )
                        cur.execute(
                            "UPDATE tasks SET state = ?, current_attempt_id = -1, started_at_ms = NULL, "
                            "finished_at_ms = NULL, error = NULL, preemption_count = 0 WHERE task_id = ?",
                            (new_task_state, tid),
                        )
                    else:
                        cur.execute(
                            "UPDATE task_attempts SET state = ?, "
                            "finished_at_ms = COALESCE(finished_at_ms, ?), error = ? "
                            "WHERE task_id = ? AND attempt_id = ?",
                            (
                                cluster_pb2.TASK_STATE_WORKER_FAILED,
                                now_ms,
                                f"Worker {worker_id} failed: {error}",
                                tid,
                                int(task_row["current_attempt_id"]),
                            ),
                        )
                        cur.execute(
                            "UPDATE tasks SET state = ?, finished_at_ms = ?, error = ?, "
                            "preemption_count = ? WHERE task_id = ?",
                            (
                                new_task_state,
                                finished_ms,
                                f"Worker {worker_id} failed: {error}",
                                preemption_count,
                                tid,
                            ),
                        )
                    task_id = JobName.from_wire(tid)
                    parent_job_id, _ = task_id.require_task()
                    self._recompute_job_state(cur, parent_job_id)
                    if new_task_state == cluster_pb2.TASK_STATE_WORKER_FAILED:
                        tasks_to_kill.add(task_id)
                cur.execute(
                    "UPDATE task_attempts SET worker_id = NULL WHERE worker_id = ?",
                    (str(worker_id),),
                )
                cur.execute("DELETE FROM dispatch_queue WHERE worker_id = ?", (str(worker_id),))
                cur.execute("DELETE FROM workers WHERE worker_id = ?", (str(worker_id),))
            else:
                now_ms = Timestamp.now().epoch_ms()
                for req in drained_dispatch.tasks_to_run:
                    cur.execute(
                        "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
                        "VALUES (?, 'run', ?, NULL, ?)",
                        (str(worker_id), req.SerializeToString(), now_ms),
                    )
                for task_id in drained_dispatch.tasks_to_kill:
                    cur.execute(
                        "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
                        "VALUES (?, 'kill', NULL, ?, ?)",
                        (str(worker_id), task_id, now_ms),
                    )
            self._record_transaction(
                cur, "heartbeat_failure", [("worker_heartbeat_failed", str(worker_id), {"error": error})]
            )
        return TxResult(tasks_to_kill=tasks_to_kill)

    def fail_heartbeat_for_worker(
        self,
        worker_id: WorkerId,
        error: str,
        snapshot: DispatchBatch,
        *,
        force_remove: bool = False,
    ) -> HeartbeatFailureResult:
        result = self.record_heartbeat_failure(
            worker_id=worker_id,
            error=error,
            drained_dispatch=snapshot,
            force_remove=force_remove,
        )
        with self._db.snapshot() as snap:
            worker_removed = not snap.exists(WORKERS, where=WORKERS.c.worker_id == str(worker_id))
        action = HeartbeatAction.WORKER_FAILED if worker_removed else HeartbeatAction.TRANSIENT_FAILURE
        return HeartbeatFailureResult(tasks_to_kill=result.tasks_to_kill, worker_removed=worker_removed, action=action)

    def mark_task_unschedulable(self, task_id: JobName, reason: str) -> TxResult:
        """Mark a task as unschedulable using the task transition engine."""
        with self._db.transaction() as cur:
            row = cur.execute("SELECT job_id FROM tasks WHERE task_id = ?", (task_id.to_wire(),)).fetchone()
            if row is None:
                return TxResult()
            now_ms = Timestamp.now().epoch_ms()
            cur.execute(
                "UPDATE tasks SET state = ?, error = ?, finished_at_ms = ? WHERE task_id = ?",
                (cluster_pb2.TASK_STATE_UNSCHEDULABLE, reason, now_ms, task_id.to_wire()),
            )
            cur.execute("DELETE FROM endpoints WHERE task_id = ?", (task_id.to_wire(),))
            self._recompute_job_state(cur, JobName.from_wire(str(row["job_id"])))
            self._record_transaction(
                cur, "mark_task_unschedulable", [("task_unschedulable", task_id.to_wire(), {"reason": reason})]
            )
        return TxResult()

    def drain_dispatch(self, worker_id: WorkerId) -> DispatchBatch | None:
        """Drain buffered dispatches and snapshot worker running tasks."""
        with self._db.transaction() as cur:
            worker_row = cur.execute(
                "SELECT worker_id, address, metadata_proto FROM workers "
                "WHERE worker_id = ? AND active = 1 AND healthy = 1",
                (str(worker_id),),
            ).fetchone()
            if worker_row is None:
                return None
            metadata = cluster_pb2.WorkerMetadata()
            metadata.ParseFromString(worker_row["metadata_proto"])
            dispatch_rows = cur.execute(
                "SELECT id, kind, payload_proto, task_id FROM dispatch_queue WHERE worker_id = ? ORDER BY id ASC",
                (str(worker_id),),
            ).fetchall()
            if dispatch_rows:
                cur.execute("DELETE FROM dispatch_queue WHERE worker_id = ?", (str(worker_id),))
            running_rows = cur.execute(
                "SELECT t.task_id, t.current_attempt_id "
                "FROM tasks t "
                "JOIN task_attempts ta ON t.task_id = ta.task_id AND t.current_attempt_id = ta.attempt_id "
                "JOIN jobs j ON j.job_id = t.job_id "
                "WHERE ta.worker_id = ? AND t.state IN (?, ?, ?) AND j.is_reservation_holder = 0 "
                "ORDER BY t.task_id ASC",
                (str(worker_id), *ACTIVE_TASK_STATES),
            ).fetchall()
            tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest] = []
            tasks_to_kill: list[str] = []
            for row in dispatch_rows:
                if str(row["kind"]) == "run" and row["payload_proto"] is not None:
                    req = cluster_pb2.Worker.RunTaskRequest()
                    req.ParseFromString(bytes(row["payload_proto"]))
                    tasks_to_run.append(req)
                elif row["task_id"] is not None:
                    tasks_to_kill.append(str(row["task_id"]))
            return DispatchBatch(
                worker_id=WorkerId(str(worker_row["worker_id"])),
                worker_address=str(worker_row["address"]),
                vm_address=metadata.vm_address or "",
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

    def requeue_dispatch(self, batch: DispatchBatch) -> None:
        """Re-queue drained dispatch payloads for later delivery."""
        with self._db.transaction() as cur:
            now_ms = Timestamp.now().epoch_ms()
            for req in batch.tasks_to_run:
                cur.execute(
                    "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
                    "VALUES (?, 'run', ?, NULL, ?)",
                    (str(batch.worker_id), req.SerializeToString(), now_ms),
                )
            for task_id in batch.tasks_to_kill:
                cur.execute(
                    "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
                    "VALUES (?, 'kill', NULL, ?, ?)",
                    (str(batch.worker_id), task_id, now_ms),
                )

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
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
                cluster_pb2.JOB_STATE_UNSCHEDULABLE,
            ):
                return False
            cur.execute("DELETE FROM jobs WHERE job_id = ?", (job_id.to_wire(),))
            self._record_transaction(cur, "remove_finished_job", [("job_removed", job_id.to_wire(), {"state": state})])
            return True

    def remove_worker(self, worker_id: WorkerId) -> Worker | None:
        with self._db.transaction() as cur:
            row = cur.execute("SELECT * FROM workers WHERE worker_id = ?", (str(worker_id),)).fetchone()
            if row is None:
                return None
            cur.execute("DELETE FROM workers WHERE worker_id = ?", (str(worker_id),))
            self._record_transaction(cur, "remove_worker", [("worker_removed", str(worker_id), {})])
            return self._db.decode_worker(row)

    # =========================================================================
    # Heartbeat Dispatch API
    # =========================================================================

    def buffer_dispatch(self, worker_id: WorkerId, task_request: cluster_pb2.Worker.RunTaskRequest) -> None:
        """Buffer a task dispatch for the next heartbeat.

        Called by the scheduling thread after committing resources via TaskAssignedEvent.
        The dispatch will be delivered when begin_heartbeat() drains the buffer.
        """
        self._db.execute(
            "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
            "VALUES (?, 'run', ?, NULL, ?)",
            (str(worker_id), task_request.SerializeToString(), Timestamp.now().epoch_ms()),
        )

    def buffer_kill(self, worker_id: WorkerId, task_id: str) -> None:
        """Buffer a task kill for the next heartbeat.

        Called when a task needs to be terminated on a worker. The kill will be
        delivered when begin_heartbeat() drains the buffer.
        """
        self._db.execute(
            "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
            "VALUES (?, 'kill', NULL, ?, ?)",
            (str(worker_id), task_id, Timestamp.now().epoch_ms()),
        )

    def begin_heartbeat(self, worker_id: WorkerId) -> DispatchBatch | None:
        """Drain dispatch for a worker and snapshot expected running attempts."""
        return self.drain_dispatch(worker_id)

    def complete_heartbeat(
        self,
        snapshot: DispatchBatch,
        response: cluster_pb2.HeartbeatResponse,
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
            if entry.state in (cluster_pb2.TASK_STATE_UNSPECIFIED, cluster_pb2.TASK_STATE_PENDING):
                continue
            updates.append(
                TaskUpdate(
                    task_id=JobName.from_wire(entry.task_id),
                    attempt_id=entry.attempt_id,
                    new_state=entry.state,
                    error=entry.error or None,
                    exit_code=entry.exit_code if entry.HasField("exit_code") else None,
                    resource_usage=entry.resource_usage if entry.resource_usage.ByteSize() > 0 else None,
                    log_entries=list(entry.log_entries),
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
                    vm_address=snapshot.vm_address,
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

    def fail_workers_by_vm_addresses(
        self,
        vm_addresses: list[str],
        reason: str,
    ) -> list[tuple[WorkerId, str]]:
        """Fail all active workers matching the given VM addresses.

        Used for slice reaping: when one worker on a multi-VM slice fails, all
        sibling workers on that slice must be failed immediately rather than
        waiting for individual heartbeat timeouts.

        Returns list of (worker_id, worker_address) pairs for workers that were removed.
        """
        if not vm_addresses:
            return []
        target_set = set(vm_addresses)
        with self._db.snapshot() as snap:
            all_workers = snap.select(WORKERS, where=WORKERS.c.active == 1)
        candidates: list[tuple[WorkerId, str]] = []
        for w in all_workers:
            if w.metadata.vm_address in target_set:
                candidates.append((w.worker_id, w.address))
        removed: list[tuple[WorkerId, str]] = []
        for worker_id, address in candidates:
            result = self.fail_heartbeat_for_worker(
                worker_id=worker_id,
                error=reason,
                snapshot=DispatchBatch(
                    worker_id=worker_id,
                    worker_address=address,
                    vm_address="",
                    running_tasks=[],
                ),
                force_remove=True,
            )
            if result.worker_removed:
                removed.append((worker_id, address))
        return removed

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

    def add_endpoint(self, endpoint: Endpoint, task_id: JobName | None = None) -> None:
        """Add an endpoint row to the DB, optionally associated with a task."""
        self._db.execute(
            "INSERT OR REPLACE INTO endpoints("
            "endpoint_id, name, address, job_id, task_id, metadata_json, registered_at_ms"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                endpoint.endpoint_id,
                endpoint.name,
                endpoint.address,
                endpoint.job_id.to_wire(),
                task_id.to_wire() if task_id else None,
                json.dumps(endpoint.metadata),
                endpoint.registered_at.epoch_ms(),
            ),
        )

    def remove_endpoint(self, endpoint_id: str) -> Endpoint | None:
        return self._db.delete_endpoint(endpoint_id)

    def _remove_endpoints_for_task(self, task_id: JobName) -> list[Endpoint]:
        """Remove all endpoints associated with a task."""
        with self._db.snapshot() as snapshot:
            endpoints = snapshot.select(
                ENDPOINTS,
                joins=(Join(ENDPOINT_TASKS, ENDPOINTS.c.endpoint_id == ENDPOINT_TASKS.c.endpoint_id),),
                where=ENDPOINT_TASKS.c.task_id == task_id.to_wire(),
            )
        self._db.delete_endpoints([endpoint.endpoint_id for endpoint in endpoints])
        return endpoints

    def remove_endpoints_for_job(self, job_id: JobName) -> list[Endpoint]:
        """Remove all endpoints for a job by removing endpoints for all its tasks."""
        query = EndpointQuery(job_id=job_id, include_terminal_jobs=True)
        joins, where = endpoint_query_predicate(query)
        with self._db.snapshot() as q:
            endpoints = q.select(ENDPOINTS, where=where, joins=tuple(joins))
        self._db.delete_endpoints([endpoint.endpoint_id for endpoint in endpoints])
        return endpoints

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
        self._db.execute(
            "UPDATE tasks SET state = ?, error = ?, exit_code = ? WHERE task_id = ?",
            (state, error, exit_code, task_id.to_wire()),
        )

    def create_attempt_for_test(self, task_id: JobName, worker_id: WorkerId) -> int:
        """Test helper: append a new task_attempt without finalizing prior attempt."""
        task = self._db.fetchone("SELECT current_attempt_id FROM tasks WHERE task_id = ?", (task_id.to_wire(),))
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        next_attempt_id = int(task["current_attempt_id"]) + 1
        now_ms = Timestamp.now().epoch_ms()
        self._db.execute(
            "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) VALUES (?, ?, ?, ?, ?)",
            (
                task_id.to_wire(),
                next_attempt_id,
                str(worker_id),
                cluster_pb2.TASK_STATE_ASSIGNED,
                now_ms,
            ),
        )
        self._db.execute(
            "UPDATE tasks SET current_attempt_id = ?, state = ? WHERE task_id = ?",
            (next_attempt_id, cluster_pb2.TASK_STATE_ASSIGNED, task_id.to_wire()),
        )
        return next_attempt_id
