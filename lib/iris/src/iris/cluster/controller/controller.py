# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import atexit
import enum
import logging
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

import uvicorn

from iris.cluster.bundle import BundleStore
from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintOp,
    PlacementRequirements,
    WellKnownAttribute,
    constraints_from_resources,
    evaluate_constraint,
    extract_placement_requirements,
    merge_constraints,
    region_constraint as make_region_constraint,
)
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.codec import (
    constraints_from_json,
    reservation_entries_from_json,
    resource_spec_from_scalars,
)
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.checkpoint import (
    CheckpointResult,
    backup_databases,
    upload_checkpoint,
    write_checkpoint,
)
from iris.cluster.controller.db import (
    ControllerDB,
    healthy_active_workers_with_attributes,
    insert_task_profile,
    job_is_finished,
    job_scheduling_deadline,
    running_tasks_by_worker,
    task_row_can_be_scheduled,
    timed_out_executing_tasks,
)
from iris.cluster.controller.schema import (
    ATTEMPT_PROJECTION,
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
    JOB_SCHEDULING_PROJECTION,
    TASK_DETAIL_PROJECTION,
    TASK_ROW_PROJECTION,
    WORKER_DETAIL_PROJECTION,
    JobDetailRow,
    JobRow,
    JobSchedulingRow,
    TaskDetailRow,
    TaskRow,
    WorkerDetailRow,
    WorkerRow,
    proto_decoder,
    tasks_with_attempts,
)
from iris.cluster.controller.budget import (
    UserBudgetDefaults,
    UserTask,
    compute_effective_band,
    compute_user_spend,
    interleave_by_user,
    resource_value,
)
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.providers.k8s.tasks import K8sTaskProvider
from iris.cluster.controller.provider import TaskProvider
from iris.cluster.controller.worker_provider import WorkerProvider
from iris.cluster.controller.scheduler import (
    JobRequirements,
    Scheduler,
    SchedulingContext,
    WorkerCapacity,
    WorkerSnapshot,
)
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import (
    HEARTBEAT_FAILURE_THRESHOLD,
    HEARTBEAT_STALENESS_THRESHOLD,
    RESERVATION_HOLDER_JOB_NAME,
    Assignment,
    ClusterCapacity,
    ControllerTransitions,
    DIRECT_PROVIDER_PROMOTION_RATE,
    HeartbeatAction,
    ReservationClaim,
    SchedulingEvent,
)
from iris.cluster.log_store import CONTROLLER_LOG_KEY, LogStoreHandler
from iris.cluster.providers.types import find_free_port
from iris.log_server.client import LogPusher, RemoteLogHandler, RemoteLogService
from iris.log_server.main import build_log_server_asgi
from iris.log_server.server import LogServiceImpl
from iris.rpc.auth import AuthTokenInjector, NullAuthInterceptor, StaticTokenProvider
from iris.cluster.types import (
    JobName,
    WorkerStatus,
    WorkerStatusMap,
    WorkerId,
    get_gpu_count,
    get_tpu_count,
)
from rigging.log_setup import slow_log
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc.auth import TokenVerifier
from rigging.timing import Duration, ExponentialBackoff, RateLimiter, Timer, Timestamp, TokenBucket

logger = logging.getLogger(__name__)

_RESOURCE_SPEC_DECODER = proto_decoder(job_pb2.ResourceSpecProto)

# Sentinel for dry-run scheduling with per-worker limits disabled.
_UNLIMITED = sys.maxsize

_SLOW_HEARTBEAT_MS = 5000

# How often the prune loop trims worker_task_history (independent of the
# full data-prune interval, which is typically 1 hour).
_HISTORY_CLEANUP_INTERVAL_S = 60.0


class SchedulingOutcome(enum.Enum):
    """Result of a scheduling cycle, used to drive adaptive backoff."""

    NO_PENDING_TASKS = "no_pending_tasks"
    NO_ASSIGNMENTS = "no_assignments"
    ASSIGNMENTS_MADE = "assignments_made"


_HEALTH_SUMMARY_INTERVAL = RateLimiter(interval_seconds=30)

# Log a detailed per-phase scheduling trace every this many rounds.
_SCHEDULING_TRACE_INTERVAL = 50

# Taint attribute injected onto claimed workers to prevent non-reservation
# jobs from landing on them.  Non-reservation jobs get a NOT_EXISTS constraint
# for this key; reservation jobs do not, so they naturally prefer claimed
# workers (which appear first in the worker list).
RESERVATION_TAINT_KEY = "reservation-job"


@dataclass
class RunningTaskInfo:
    """Info about a running task used by the preemption pass."""

    task_id: JobName
    worker_id: WorkerId
    band_sort_key: int  # 1=production, 2=interactive, 3=batch
    resource_value: int
    is_coscheduled: bool
    resources: job_pb2.ResourceSpecProto
    already_preempted: bool = False


@dataclass(frozen=True)
class PreemptionCandidate:
    """An unscheduled task that may preempt running work."""

    job_name: JobName
    requirements: JobRequirements
    band: int  # proto PriorityBand value


@dataclass
class _SyncFailureAccumulator:
    """Mutable accumulator for tracking failures during provider sync."""

    fail_count: int = 0
    failed_workers: list[str] = field(default_factory=list)
    all_tasks_to_kill: set[JobName] = field(default_factory=set)
    all_task_kill_workers: dict[JobName, WorkerId] = field(default_factory=dict)


@dataclass(frozen=True)
class _SchedulingStateRead:
    """Snapshot of pending tasks and workers read at the start of a scheduling cycle."""

    pending_tasks: list[TaskRow]
    workers: list[WorkerRow]
    state_read_ms: int


@dataclass(frozen=True)
class _GatedCandidates:
    """Tasks that passed deadline, reservation, and per-job-cap gates."""

    schedulable_task_ids: list[JobName]
    jobs: dict[JobName, JobRequirements]
    has_reservation: set[JobName]
    has_direct_reservation: set[JobName]


@dataclass(frozen=True)
class _SchedulingOrder:
    """Priority-ordered task list with budget context for preemption."""

    ordered_task_ids: list[JobName]
    task_band_map: dict[JobName, int]
    user_spend: dict[str, int]
    user_budget_limits: dict[str, int]


def _resource_spec_from_row(job: JobRow | JobSchedulingRow) -> job_pb2.ResourceSpecProto:
    """Reconstruct a ResourceSpecProto from native job columns."""
    return resource_spec_from_scalars(
        job.res_cpu_millicores, job.res_memory_bytes, job.res_disk_bytes, job.res_device_json
    )


def job_requirements_from_job(job: JobSchedulingRow) -> JobRequirements:
    """Convert a job row to scheduler-compatible JobRequirements."""
    return JobRequirements(
        resources=_resource_spec_from_row(job),
        constraints=constraints_from_json(job.constraints_json),
        is_coscheduled=job.has_coscheduling,
        coscheduling_group_by=job.coscheduling_group_by if job.has_coscheduling else None,
    )


def compute_demand_entries(
    queries: ControllerDB,
    scheduler: Scheduler | None = None,
    workers: list[WorkerSnapshot] | None = None,
    reservation_claims: dict[WorkerId, ReservationClaim] | None = None,
) -> list[DemandEntry]:
    """Compute demand entries for the autoscaler from controller state.

    All pending tasks — both real and reservation holder — flow through a
    single unified path. Every task participates in the dry-run and generates
    demand through the same logic using its job's resource spec.

    Holder tasks consume zero resources on workers, so they won't be absorbed
    by the dry-run when workers have available capacity. This ensures they
    always generate demand, keeping reserved capacity alive via the
    autoscaler. The taint/constraint mechanism ensures only peer jobs can
    actually use the reserved workers.

    .. note::

        Demand from holder tasks and parent real tasks is additive. On a cold
        start with N reservation entries and M real tasks this reports N + M
        demand entries, which may overprovision. In practice reservations are
        used when the parent job does not request its own resources, so the
        additive behavior is correct. If that changes, a dedup path (e.g.
        ``max(real_pending, holders)``) should be added here.

    Args:
        queries: Controller DB read surface for pending tasks and jobs.
        scheduler: Scheduler for dry-run pass. If None, skips dry-run.
        workers: Available workers for dry-run. If None, skips dry-run.
        reservation_claims: Reservation claims to apply taint injection in the
            dry-run, matching the real scheduling path. If None, no taints applied.
    """
    demand_entries: list[DemandEntry] = []

    # Collect all schedulable pending tasks, grouped by job.
    tasks_by_job: dict[JobName, list[TaskRow]] = defaultdict(list)
    all_schedulable: list[TaskRow] = []
    pending = _schedulable_tasks(queries)
    job_rows = list(_jobs_by_id(queries, {task.job_id for task in pending}).values()) if pending else []
    jobs_by_id = {job.job_id: job for job in job_rows}
    for task in pending:
        if not task_row_can_be_scheduled(task):
            continue
        if task.job_id not in jobs_by_id:
            continue
        tasks_by_job[task.job_id].append(task)
        all_schedulable.append(task)

    # Build job requirements once, shared between dry-run and demand emission.
    # Also track which jobs have reservations so we can apply taint injection.
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    has_direct_reservation: set[JobName] = set()
    for task in all_schedulable:
        if task.job_id in jobs:
            continue
        job = jobs_by_id.get(task.job_id)
        if job is None:
            continue
        jobs[task.job_id] = job_requirements_from_job(job)
        if job.has_reservation:
            has_reservation.add(task.job_id)
            has_direct_reservation.add(task.job_id)
        elif _find_reservation_ancestor(queries, task.job_id) is not None:
            has_reservation.add(task.job_id)

    # Dry-run scheduling with building/assignment limits disabled.
    # All tasks participate — holders and real tasks alike.
    absorbed_task_ids: set[JobName] = set()
    if scheduler is not None and workers is not None and workers:
        building_counts = _building_counts(queries, workers)
        task_ids = [t.task_id for t in all_schedulable]
        claims = reservation_claims or {}
        dry_run_workers = _inject_reservation_taints(workers, claims)
        dry_run_jobs = _inject_taint_constraints(jobs, has_reservation, has_direct_reservation)

        context = scheduler.create_scheduling_context(
            dry_run_workers,
            building_counts=building_counts,
            pending_tasks=task_ids,
            jobs=dry_run_jobs,
            max_building_tasks=_UNLIMITED,
            max_assignments_per_worker=_UNLIMITED,
        )
        result = scheduler.find_assignments(context)
        for task_id, _ in result.assignments:
            absorbed_task_ids.add(task_id)

    # Emit demand for all unabsorbed tasks through a single path.
    for job_id, tasks in tasks_by_job.items():
        job = jobs_by_id.get(job_id)
        if not job:
            continue
        if job_is_finished(job.state):
            continue

        job_constraints = constraints_from_json(job.constraints_json)
        job_resources = _resource_spec_from_row(job)

        invalid_reason: str | None = None
        try:
            normalized = extract_placement_requirements(job_constraints)
        except ValueError as e:
            invalid_reason = f"invalid_constraints: {e}"
            normalized = PlacementRequirements(
                device_type=None,
                device_variants=None,
                preemptible=None,
                required_regions=None,
                required_zones=None,
            )

        if job.has_coscheduling:
            remaining_ids = []
            for t in tasks:
                if t.task_id in absorbed_task_ids:
                    continue
                remaining_ids.append(t.task_id.to_wire())
            if remaining_ids:
                demand_entries.append(
                    DemandEntry(
                        task_ids=remaining_ids,
                        coschedule_group_id=job.job_id.to_wire(),
                        normalized=normalized,
                        constraints=job_constraints,
                        resources=job_resources,
                        invalid_reason=invalid_reason,
                    )
                )
            continue

        for task in tasks:
            if task.task_id in absorbed_task_ids:
                continue
            demand_entries.append(
                DemandEntry(
                    task_ids=[task.task_id.to_wire()],
                    coschedule_group_id=None,
                    normalized=normalized,
                    constraints=job_constraints,
                    resources=job_resources,
                    invalid_reason=invalid_reason,
                )
            )

    return demand_entries


def _read_reservation_claims(db: ControllerDB) -> dict[WorkerId, ReservationClaim]:
    """Read reservation claims from the canonical DB table."""
    with db.read_snapshot() as snapshot:
        rows = snapshot.raw(
            "SELECT rc.worker_id, rc.job_id, rc.entry_idx FROM reservation_claims rc",
            decoders={"worker_id": WorkerId},
        )
    return {
        row.worker_id: ReservationClaim(
            job_id=row.job_id,
            entry_idx=row.entry_idx,
        )
        for row in rows
    }


def _jobs_by_id(queries: ControllerDB, job_ids: set[JobName]) -> dict[JobName, JobSchedulingRow]:
    if not job_ids:
        return {}
    wires = [job_id.to_wire() for job_id in job_ids]
    placeholders = ",".join("?" for _ in wires)
    with queries.read_snapshot() as snapshot:
        jobs = JOB_SCHEDULING_PROJECTION.decode(
            snapshot.fetchall(
                f"SELECT {JOB_SCHEDULING_PROJECTION.select_clause()} "
                f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id IN ({placeholders})",
                tuple(wires),
            ),
        )
    return {job.job_id: job for job in jobs}


def _jobs_with_reservations(queries: ControllerDB, states: tuple[int, ...]) -> list[JobDetailRow]:
    """Fetch only jobs that have reservations, filtering at the SQL level.

    Uses the has_reservation column on the jobs table to filter without a JOIN.
    """
    placeholders = ",".join("?" for _ in states)
    with queries.read_snapshot() as snapshot:
        rows = snapshot._fetchall(
            f"SELECT {JOB_DETAIL_PROJECTION.select_clause()} "
            f"FROM jobs j {JOB_CONFIG_JOIN} "
            f"WHERE j.state IN ({placeholders}) AND j.has_reservation = 1",
            list(states),
        )
    return JOB_DETAIL_PROJECTION.decode(rows)


def _get_running_tasks_with_band_and_value(
    db: ControllerDB,
    claimed_workers: set[WorkerId],
    user_spend: dict[str, int] | None = None,
    user_budget_limits: dict[str, int] | None = None,
) -> list[RunningTaskInfo]:
    """Query running tasks with band, worker, resource spec, and coscheduling status.

    Skips tasks on reservation-claimed workers since those workers are spoken for.
    When ``user_spend`` and ``user_budget_limits`` are provided, the effective band
    is computed so over-budget users' tasks are treated as BATCH for preemption.
    """
    with db.read_snapshot() as q:
        rows = q.raw(
            "SELECT t.task_id, t.priority_band, t.current_worker_id AS worker_id, "
            "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json, "
            "jc.has_coscheduling "
            "FROM tasks t "
            "JOIN job_config jc ON jc.job_id = t.job_id "
            "WHERE t.state = ? AND t.current_worker_id IS NOT NULL",
            (job_pb2.TASK_STATE_RUNNING,),
            decoders={
                "task_id": JobName.from_wire,
                "priority_band": int,
                "worker_id": WorkerId,
            },
        )
    _spend = user_spend or {}
    _limits = user_budget_limits or {}
    result: list[RunningTaskInfo] = []
    for row in rows:
        wid = row.worker_id
        if wid in claimed_workers:
            continue
        resources = resource_spec_from_scalars(
            row.res_cpu_millicores,
            row.res_memory_bytes,
            row.res_disk_bytes,
            row.res_device_json,
        )
        band = compute_effective_band(row.priority_band, row.task_id.user, _spend, _limits)
        result.append(
            RunningTaskInfo(
                task_id=row.task_id,
                worker_id=wid,
                band_sort_key=band,
                resource_value=resource_value(
                    resources.cpu_millicores,
                    resources.memory_bytes,
                    get_gpu_count(resources.device) + get_tpu_count(resources.device),
                ),
                is_coscheduled=bool(int(row.has_coscheduling)),
                resources=resources,
            )
        )
    return result


def _run_preemption_pass(
    unscheduled_tasks: list[PreemptionCandidate],
    running_tasks_info: list[RunningTaskInfo],
    context: SchedulingContext,
) -> list[tuple[JobName, JobName]]:
    """Find tasks to preempt for higher-priority unscheduled work.

    Rules:
    - PRODUCTION preempts INTERACTIVE and BATCH.
    - INTERACTIVE preempts BATCH only.
    - BATCH never preempts.
    - Within same band, no preemption (compete via scheduling order only).
    - Coscheduled tasks are not preemptible (v1).
    """
    preemptions: list[tuple[JobName, JobName]] = []

    # Sort victims: lowest priority first (highest band_sort_key), then cheapest
    victims = sorted(running_tasks_info, key=lambda t: (-t.band_sort_key, t.resource_value))

    for candidate in unscheduled_tasks:
        task_id, req, task_band_key = candidate.job_name, candidate.requirements, candidate.band
        # Batch never preempts
        if task_band_key >= job_pb2.PRIORITY_BAND_BATCH:
            continue

        for victim in victims:
            if victim.already_preempted:
                continue
            if victim.is_coscheduled:
                continue
            # Can only preempt strictly lower priority (higher band_sort_key)
            if victim.band_sort_key <= task_band_key:
                continue

            cap = context.capacities.get(victim.worker_id)
            if cap is None:
                continue

            if not cap.matches_constraints(req.constraints):
                continue

            # If current capacity already fits, no preemption needed
            if cap.can_fit(req) is None:
                continue

            # Check if freeing the victim's resources would create enough capacity
            hypothetical = WorkerCapacity(
                worker_id=cap.worker_id,
                available_cpu_millicores=cap.available_cpu_millicores + victim.resources.cpu_millicores,
                available_memory=cap.available_memory + victim.resources.memory_bytes,
                available_gpus=cap.available_gpus + get_gpu_count(victim.resources.device),
                available_tpus=cap.available_tpus + get_tpu_count(victim.resources.device),
                attributes=cap.attributes,
                building_task_count=max(0, cap.building_task_count - 1),
                max_building_tasks=cap.max_building_tasks,
            )
            if hypothetical.can_fit(req) is None:
                preemptions.append((task_id, victim.task_id))
                victim.already_preempted = True
                break

    return preemptions


def _schedulable_tasks(queries: ControllerDB) -> list[TaskRow]:
    # Only PENDING tasks can pass can_be_scheduled(); no need to fetch ASSIGNED/BUILDING/RUNNING.
    with queries.read_snapshot() as snapshot:
        tasks = TASK_ROW_PROJECTION.decode(
            snapshot.fetchall(
                f"SELECT {TASK_ROW_PROJECTION.select_clause()} FROM tasks t WHERE t.state = ? "
                "ORDER BY t.priority_band ASC, t.priority_neg_depth ASC, t.priority_root_submitted_ms ASC, "
                "t.submitted_at_ms ASC, t.priority_insertion ASC",
                (job_pb2.TASK_STATE_PENDING,),
            ),
        )
    return [task for task in tasks if task_row_can_be_scheduled(task)]


def _tasks_by_ids_with_attempts(queries: ControllerDB, task_ids: set[JobName]) -> dict[JobName, TaskDetailRow]:
    if not task_ids:
        return {}
    task_wires = [task_id.to_wire() for task_id in task_ids]
    placeholders = ",".join("?" for _ in task_wires)
    with queries.read_snapshot() as snapshot:
        tasks = TASK_DETAIL_PROJECTION.decode(
            snapshot.fetchall(
                f"SELECT {TASK_DETAIL_PROJECTION.select_clause()} "
                f"FROM tasks t WHERE t.task_id IN ({placeholders}) ORDER BY t.task_id ASC",
                tuple(task_wires),
            ),
        )
        attempts = ATTEMPT_PROJECTION.decode(
            snapshot.fetchall(
                f"SELECT {ATTEMPT_PROJECTION.select_clause()} FROM task_attempts ta "
                f"WHERE ta.task_id IN ({placeholders}) "
                "ORDER BY ta.task_id ASC, ta.attempt_id ASC",
                tuple(task_wires),
            ),
        )
    return {task.task_id: task for task in tasks_with_attempts(tasks, attempts)}


def _building_counts(queries: ControllerDB, workers: list[WorkerRow]) -> dict[WorkerId, int]:
    """Count tasks in BUILDING or ASSIGNED state per worker, excluding reservation-holder jobs."""
    if not workers:
        return {}
    worker_ids = [str(w.worker_id) for w in workers]
    placeholders = ",".join("?" for _ in worker_ids)
    sql = (
        "SELECT t.current_worker_id AS worker_id, COUNT(*) as cnt FROM tasks t "
        "JOIN jobs j ON t.job_id = j.job_id "
        f"WHERE t.current_worker_id IN ({placeholders}) "
        "AND t.state IN (?, ?) "
        "AND j.is_reservation_holder = 0 "
        "GROUP BY t.current_worker_id"
    )
    with queries.read_snapshot() as q:
        rows = q.raw(
            sql,
            (*worker_ids, job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_ASSIGNED),
            decoders={"worker_id": WorkerId, "cnt": int},
        )
    return {row.worker_id: row.cnt for row in rows}


def _workers_by_id(queries: ControllerDB, worker_ids: set[WorkerId]) -> dict[WorkerId, WorkerDetailRow]:
    if not worker_ids:
        return {}
    wires = [str(wid) for wid in worker_ids]
    placeholders = ",".join("?" for _ in wires)
    with queries.read_snapshot() as snapshot:
        workers = WORKER_DETAIL_PROJECTION.decode(
            snapshot.fetchall(
                f"SELECT {WORKER_DETAIL_PROJECTION.select_clause()} "
                f"FROM workers w WHERE w.worker_id IN ({placeholders})",
                tuple(wires),
            ),
        )
    return {worker.worker_id: worker for worker in workers}


def _task_worker_mapping(queries: ControllerDB, task_ids: set[JobName]) -> dict[JobName, WorkerId]:
    if not task_ids:
        return {}
    task_wires = [task_id.to_wire() for task_id in task_ids]
    placeholders = ",".join("?" for _ in task_wires)
    with queries.read_snapshot() as snapshot:
        rows = snapshot.raw(
            f"SELECT t.task_id, t.current_worker_id AS worker_id FROM tasks t "
            f"WHERE t.task_id IN ({placeholders}) AND t.current_worker_id IS NOT NULL",
            tuple(task_wires),
            decoders={"task_id": JobName.from_wire, "worker_id": WorkerId},
        )
    return {row.task_id: row.worker_id for row in rows}


def _worker_matches_reservation_entry(
    worker: WorkerRow,
    res_entry: job_pb2.ReservationEntry,
) -> bool:
    """Check if a worker is eligible for a reservation entry.

    Auto-injects device constraints from the reservation entry's resource spec
    and merges them with explicit constraints on the entry, then evaluates all
    constraints against the worker's attributes.
    """
    auto = constraints_from_resources(res_entry.resources)
    explicit = [Constraint.from_proto(c) for c in res_entry.constraints]
    merged = merge_constraints(auto, explicit)

    for constraint in merged:
        attr = worker.attributes.get(constraint.key)
        if not evaluate_constraint(attr, constraint):
            return False

    return True


def _inject_reservation_taints(
    workers: list[WorkerRow],
    claims: dict[WorkerId, ReservationClaim],
) -> list[WorkerRow]:
    """Create modified worker copies with reservation taints and prioritization.

    Claimed workers receive a ``reservation-job`` attribute set to the claiming
    job's ID.  The returned list is ordered with claimed workers first so that
    reservation jobs (which have no NOT_EXISTS constraint) naturally pick from
    their claimed workers before unclaimed ones.

    Workers are never mutated — ``dataclasses.replace`` produces shallow copies.
    """
    if not claims:
        return workers

    claimed: list[WorkerRow] = []
    unclaimed: list[WorkerRow] = []
    for worker in workers:
        claim = claims.get(worker.worker_id)
        if claim is not None:
            modified_attrs = dict(worker.attributes)
            modified_attrs[RESERVATION_TAINT_KEY] = AttributeValue(claim.job_id)
            claimed.append(replace(worker, attributes=modified_attrs))
        else:
            unclaimed.append(worker)
    return claimed + unclaimed


def _inject_taint_constraints(
    jobs: dict[JobName, JobRequirements],
    has_reservation: set[JobName],
    has_direct_reservation: set[JobName] | None = None,
) -> dict[JobName, JobRequirements]:
    """Add reservation taint constraints to jobs.

    Three-way logic:
    - Direct reservation jobs (has_direct_reservation): get an EQ constraint
      forcing them onto their claimed workers only.
    - Descendants of reservation jobs (has_reservation minus direct): no
      constraint — they can use both claimed and unclaimed workers.
    - Non-reservation jobs: get a NOT_EXISTS constraint blocking them from
      claimed workers.
    """
    if not has_reservation and not jobs:
        return jobs

    if has_direct_reservation is None:
        has_direct_reservation = set()

    taint_constraint = Constraint(key=RESERVATION_TAINT_KEY, op=ConstraintOp.NOT_EXISTS)

    modified: dict[JobName, JobRequirements] = {}
    for job_id, req in jobs.items():
        if job_id in has_direct_reservation:
            eq_constraint = Constraint.create(
                key=RESERVATION_TAINT_KEY,
                op=ConstraintOp.EQ,
                value=job_id.to_wire(),
            )
            modified[job_id] = replace(
                req,
                constraints=[*list(req.constraints), eq_constraint],
            )
        elif job_id in has_reservation:
            modified[job_id] = req
        else:
            modified[job_id] = replace(
                req,
                constraints=[*list(req.constraints), taint_constraint],
            )
    return modified


def _find_reservation_ancestor(queries: ControllerDB, job_id: JobName) -> JobName | None:
    """Walk up the job hierarchy to find the nearest ancestor with a reservation.

    Returns the ancestor's JobName, or None if no ancestor has a reservation.
    Uses the has_reservation column on the jobs table.
    """
    current = job_id.parent
    with queries.read_snapshot() as q:
        while current is not None:
            row = q.execute_sql(
                "SELECT has_reservation FROM jobs WHERE job_id = ?",
                (current.to_wire(),),
            ).fetchone()
            if row is not None and row[0]:
                return current
            current = current.parent
    return None


def _reservation_region_constraints(
    job_id_wire: str,
    claims: dict[WorkerId, ReservationClaim],
    queries: ControllerDB,
    existing_constraints: list[Constraint],
) -> list[Constraint]:
    """Derive region constraints from claimed reservation workers.

    When a reservation job has no explicit region constraint, this function
    extracts the region attributes of claimed workers and returns the existing
    constraints plus an injected region constraint.  If the job already has a
    region constraint, or if claimed workers lack region attributes, the
    existing constraints are returned unchanged.
    """
    if any(c.key == WellKnownAttribute.REGION for c in existing_constraints):
        return existing_constraints

    claimed_worker_ids = {worker_id for worker_id, claim in claims.items() if claim.job_id == job_id_wire}
    workers_by_id = {
        worker.worker_id: worker
        for worker in healthy_active_workers_with_attributes(queries)
        if worker.worker_id in claimed_worker_ids
    }
    regions: set[str] = set()
    for worker in workers_by_id.values():
        if worker is None:
            continue
        region_attr = worker.attributes.get(WellKnownAttribute.REGION)
        if region_attr is not None:
            regions.add(str(region_attr.value))

    if not regions:
        return existing_constraints

    return [*existing_constraints, make_region_constraint(sorted(regions))]


def _preference_pass(
    context: SchedulingContext,
    has_reservation: set[JobName],
    claims: dict[WorkerId, ReservationClaim],
) -> list[tuple[JobName, WorkerId]]:
    """Try to assign reservation-job tasks to their claimed workers first.

    Iterates reservation-job tasks and, for each, checks the (small) set of
    workers claimed for that job. If a claimed worker has capacity, the task
    is assigned immediately — deducting resources and marking the worker as
    scheduled in the shared context so the subsequent find_assignments pass
    sees the updated state.

    Coscheduled jobs are skipped because they require atomic all-or-nothing
    assignment across a worker group.

    Returns the list of (task_id, worker_id) assignments made.
    """
    if not has_reservation or not claims:
        return []

    # Reverse index: job_wire -> list of claimed worker IDs
    claimed_by_job: dict[str, list[WorkerId]] = defaultdict(list)
    for wid, claim in claims.items():
        claimed_by_job[claim.job_id].append(wid)

    assignments: list[tuple[JobName, WorkerId]] = []
    preference_scheduled: set[JobName] = set()

    for task_id in context.pending_tasks:
        job_id = task_id.parent
        if job_id is None or job_id not in has_reservation:
            continue

        req = context.jobs.get(job_id)
        if req is None or req.is_coscheduled:
            continue

        job_wire = job_id.to_wire()
        # Holder jobs are children of the reservation job — look up claims
        # under the parent's wire ID.
        claim_key = job_wire
        if RESERVATION_HOLDER_JOB_NAME in job_wire:
            parent = job_id.parent
            if parent is not None:
                claim_key = parent.to_wire()
        for wid in claimed_by_job.get(claim_key, ()):
            if context.assignment_counts.get(wid, 0) >= context.max_assignments_per_worker:
                continue
            capacity = context.capacities.get(wid)
            if capacity is None:
                continue
            if capacity.can_fit(req) is not None:
                continue
            capacity.deduct(req)
            context.assignment_counts[wid] = context.assignment_counts.get(wid, 0) + 1
            assignments.append((task_id, wid))
            preference_scheduled.add(task_id)
            break

    # Remove preference-assigned tasks from pending so find_assignments skips them.
    if preference_scheduled:
        context.pending_tasks = [t for t in context.pending_tasks if t not in preference_scheduled]

    return assignments


@dataclass
class ControllerConfig:
    """Controller configuration."""

    host: str = "127.0.0.1"
    """Host to bind the HTTP server to."""

    port: int = 0
    """Port to bind the HTTP server to. Use 0 for auto-assign."""

    remote_state_dir: str = ""
    """Remote URI for controller checkpoints and worker profiles (e.g. gs://bucket/iris/state)."""

    scheduler_min_interval: Duration = field(default_factory=lambda: Duration.from_seconds(1.0))
    """Minimum scheduling loop interval (used when cluster is active)."""

    scheduler_max_interval: Duration = field(default_factory=lambda: Duration.from_seconds(10.0))
    """Maximum scheduling loop interval (reached via exponential backoff when idle)."""

    heartbeat_interval: Duration = field(default_factory=lambda: Duration.from_seconds(5.0))
    """How often to send heartbeats to workers."""

    max_dispatch_parallelism: int = 32
    """Maximum number of concurrent RPC dispatch operations."""

    max_tasks_per_job_per_cycle: int = 4
    """Maximum tasks from a single non-coscheduled job to consider per scheduling
    cycle. Bounds CPU time in the scheduler when many tasks are pending, preventing
    GIL starvation of the heartbeat thread. Coscheduled jobs are exempt (they need
    all tasks for atomic assignment). Set to 0 for unlimited."""

    heartbeat_failure_threshold: int = HEARTBEAT_FAILURE_THRESHOLD
    """Consecutive heartbeat failures before marking worker as dead."""

    autoscaler_enabled: bool = False
    worker_access_address: str = ""

    checkpoint_interval: Duration | None = None
    """If set, take a periodic best-effort snapshot this often.
    Runs in the autoscaler loop thread; does not pause scheduling."""

    profile_interval: Duration = field(default_factory=lambda: Duration.from_seconds(600))
    """How often the controller captures CPU profiles for all running tasks."""

    profile_duration: int = 10
    """Duration in seconds for each profile capture (CPU and memory)."""

    prune_interval: Duration = field(default_factory=lambda: Duration.from_seconds(3600))
    """How often to run the data pruning sweep (default: 1 hour)."""

    job_retention: Duration = field(default_factory=lambda: Duration.from_seconds(7 * 86400))
    """Delete terminal jobs older than this (default: 7 days)."""

    worker_retention: Duration = field(default_factory=lambda: Duration.from_seconds(86400))
    """Delete inactive/unhealthy workers whose last heartbeat exceeds this (default: 24 hours)."""

    log_retention: Duration = field(default_factory=lambda: Duration.from_seconds(7 * 86400))
    """Delete controller logs older than this (default: 7 days)."""

    txn_action_retention: Duration = field(default_factory=lambda: Duration.from_seconds(3 * 86400))
    """Delete txn_actions older than this (default: 3 days)."""

    profile_retention: Duration = field(default_factory=lambda: Duration.from_seconds(86400))
    """Delete task_profiles older than this (default: 24 hours)."""

    profile_concurrency: int = 8
    """Maximum parallel profile RPCs to workers."""

    local_state_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="iris_controller_state_")))
    """Local directory for controller DB, logs, bundle cache."""

    auth_verifier: TokenVerifier | None = None
    """When set, all RPC calls require a valid bearer token verified by this verifier."""

    auth_provider: str | None = None
    """Name of the auth provider (e.g. "gcp", "static") for the dashboard UI."""

    auth: ControllerAuth | None = None
    """Full auth config passed to the service layer for login and API key management."""

    dry_run: bool = False
    """Start in dry-run mode: compute scheduling but suppress all side effects."""

    user_budget_defaults: UserBudgetDefaults = field(default_factory=UserBudgetDefaults)
    """Default budget settings applied when a new user is first seen."""

    log_service_address: str | None = None
    """Address of an externally-hosted log server (e.g. http://localhost:10001).
    When set, the controller assumes a separate process is already serving
    LogService at that URL and only acts as a client. When None, the
    Controller starts its own LogServiceImpl in a uvicorn thread on a free
    port and advertises that URL to workers via /system/log-server."""


def _log_client_interceptors(config: "ControllerConfig") -> tuple:
    """Return Connect interceptors for controller-originated LogService RPCs.

    When auth is configured, attach the worker JWT as a bearer token so the
    log server accepts PushLogs/FetchLogs. The worker token is signed with
    the same key the log server verifies against; no separate admin token
    is required for controller-initiated pushes.
    """
    token = config.auth.worker_token if config.auth and config.auth.worker_token else None
    if not token:
        return ()
    return (AuthTokenInjector(StaticTokenProvider(token)),)


class Controller:
    """Unified controller managing all components and lifecycle.

    Runs three background loops:
    - Scheduling loop: finds task assignments, checks worker timeouts
    - Provider loop: syncs task state with the execution backend via TaskProvider
    - Autoscaler loop: evaluates scaling decisions, manages slice lifecycle

    Each loop runs on its own thread so blocking operations in one don't
    stall the others.

    Example:
        ```python
        config = ControllerConfig(port=8080)
        controller = Controller(
            config=config,
            provider=WorkerProvider(stub_factory=RpcWorkerStubFactory()),
        )
        controller.start()
        try:
            job_id = controller.launch_job(request)
            status = controller.get_job_status(job_id)
        finally:
            controller.stop()
        ```

    Args:
        config: Controller configuration
        provider: TaskProvider for communicating with the execution backend
        autoscaler: Optional Autoscaler for managing VM slices. If provided,
                   the controller will run it in a background thread.
    """

    def __init__(
        self,
        config: ControllerConfig,
        provider: TaskProvider | K8sTaskProvider,
        autoscaler: "Autoscaler | None" = None,
        threads: ThreadContainer | None = None,
        db: ControllerDB | None = None,
    ):
        if not config.remote_state_dir:
            raise ValueError(
                "remote_state_dir is required. Set via ControllerConfig.remote_state_dir. "
                "Example: remote_state_dir='gs://my-bucket/iris/state'"
            )

        self._config = config
        self._provider: TaskProvider | K8sTaskProvider = provider
        self._provider_scheduling_events: list[SchedulingEvent] = []
        self._provider_capacity: ClusterCapacity | None = None
        self._promotion_bucket = TokenBucket(
            capacity=DIRECT_PROVIDER_PROMOTION_RATE,
            refill_period=Duration.from_minutes(1),
        )

        config.local_state_dir.mkdir(parents=True, exist_ok=True)
        if db is not None:
            self._db = db
        else:
            self._db = ControllerDB(db_dir=config.local_state_dir / "db")

        # --- Log service setup ---
        # Workers always push to a LogService reachable over RPC, advertised
        # via /system/log-server. Two hosting modes:
        #
        # 1. External (production): ``log_service_address`` points at a
        #    separate log server process. The Controller only acts as a
        #    client — pushing its own process logs and proxying FetchLogs.
        # 2. Local (default / tests): Controller owns a LogServiceImpl and
        #    serves it on ``_log_server_port`` via a uvicorn thread spawned
        #    in ``start()``. Controller-process logs short-circuit the RPC
        #    hop and write directly to the in-process LogStore.
        self._log_service: LogServiceImpl | None = None
        self._remote_log_service: RemoteLogService | None = None
        self._log_pusher: LogPusher | None = None
        self._log_server: uvicorn.Server | None = None
        self._log_server_port: int | None = None
        self._log_service_address: str | None = config.log_service_address

        if config.log_service_address:
            log_client_interceptors = _log_client_interceptors(config)
            self._remote_log_service = RemoteLogService(config.log_service_address, interceptors=log_client_interceptors)
            log_service_for_rpc: LogServiceImpl | RemoteLogService = self._remote_log_service

            # Providers push directly to the log server via RPC.
            provider_log_pusher = LogPusher(config.log_service_address, interceptors=log_client_interceptors)
            if isinstance(self._provider, (K8sTaskProvider, WorkerProvider)):
                self._provider.log_pusher = provider_log_pusher

            # Controller process logs ship to the log server via RemoteLogHandler.
            self._log_pusher = LogPusher(config.log_service_address, interceptors=log_client_interceptors)
            self._log_handler: RemoteLogHandler | LogStoreHandler = RemoteLogHandler(
                self._log_pusher, key=CONTROLLER_LOG_KEY
            )
        else:
            self._log_service = LogServiceImpl(
                log_dir=config.local_state_dir / "logs",
                remote_log_dir=f"{config.remote_state_dir.rstrip('/')}/logs",
            )
            log_service_for_rpc = self._log_service

            # Pre-allocate the log server's port so system endpoints and any
            # providers/clients constructed before start() see a stable URL.
            self._log_server_port = find_free_port()

            # Controller process logs bypass the loopback RPC and write
            # directly to the in-process LogStore.
            self._log_handler = LogStoreHandler(self._log_service.log_store, key=CONTROLLER_LOG_KEY)

        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger("iris").addHandler(self._log_handler)

        self._transitions = ControllerTransitions(
            db=self._db,
            heartbeat_failure_threshold=config.heartbeat_failure_threshold,
            user_budget_defaults=config.user_budget_defaults,
        )
        self._scheduler = Scheduler()

        self._bundle_store = BundleStore(storage_dir=f"{config.remote_state_dir.rstrip('/')}/bundles")

        self._service = ControllerServiceImpl(
            self._transitions,
            self._db,
            controller=self,
            bundle_store=self._bundle_store,
            log_service=log_service_for_rpc,
            auth=config.auth,
            system_endpoints={},
        )
        self._dashboard = ControllerDashboard(
            self._service,
            log_service=log_service_for_rpc,
            host=config.host,
            port=config.port,
            auth_verifier=config.auth_verifier,
            auth_provider=config.auth_provider,
            auth_optional=config.auth.optional if config.auth else False,
        )

        # Background loop state
        self._threads = threads if threads is not None else get_thread_container()
        self._wake_event = threading.Event()
        self._heartbeat_event = threading.Event()
        self._server: uvicorn.Server | None = None
        self._scheduling_thread: ManagedThread | None = None
        self._heartbeat_thread: ManagedThread | None = None
        self._autoscaler_thread: ManagedThread | None = None
        self._profile_thread: ManagedThread | None = None
        self._prune_thread: ManagedThread | None = None

        self._autoscaler: Autoscaler | None = autoscaler

        self._heartbeat_iteration = 0
        self._last_timeout_check_ms: int = 0

        # Cached scheduling diagnostics: populated each scheduling cycle for
        # pending jobs that could not be assigned.  Keyed by job wire ID.
        # RPC handlers read this dict instead of recomputing diagnostics,
        # avoiding expensive scheduler work on every CLI poll.
        self._scheduling_diagnostics: dict[str, str] = {}
        self._scheduling_round: int = 0

        # Set to True once start() is called. Used to gate operations that
        # are only valid before the controller loops begin (e.g. LoadCheckpoint).
        self._started = False

        # Checkpoint coordination: when set, scheduling and autoscaler loops
        # skip their work so the snapshot captures a quiescent state.
        # threading.Event (not a bare bool) for cross-thread memory ordering.
        self._checkpoint_paused = threading.Event()
        self._atexit_registered = False

        # Serializes heartbeat rounds against checkpoint snapshots so that
        # begin_checkpoint cannot fire while dispatches from begin_heartbeat()
        # are in flight (but not yet applied by complete_heartbeat).
        self._heartbeat_lock = threading.Lock()

        # Rate-limits periodic (best-effort) checkpoint writes.
        # None when checkpoint_interval is not configured.
        self._periodic_checkpoint_limiter: RateLimiter | None = (
            RateLimiter(interval_seconds=config.checkpoint_interval.to_seconds())
            if config.checkpoint_interval is not None
            else None
        )

    def wake(self) -> None:
        """Signal the scheduling loop to run immediately and reset backoff.

        Called on new job submission. Resets the adaptive backoff so the
        scheduler responds to new work within one cycle.
        """
        self._wake_event.set()

    @property
    def started(self) -> bool:
        """Whether the controller loops have been started."""
        return self._started

    def _start_local_log_server(self) -> None:
        """Spawn a uvicorn server hosting LogServiceImpl on a dedicated port.

        No-op when an external log server is configured.
        """
        if self._log_service is None:
            # External mode: address already populated in __init__.
            return

        assert self._log_server_port is not None, "log server port must be pre-allocated in __init__"

        # Wrap the verifier in NullAuthInterceptor so anonymous calls are
        # still accepted in null-auth/test mode, matching the dashboard's
        # behaviour. Real workers attach JWTs that the verifier validates.
        interceptors = (NullAuthInterceptor(verifier=self._config.auth_verifier),)
        app = build_log_server_asgi(self._log_service, interceptors=interceptors)
        log_server_config = uvicorn.Config(
            app,
            host=self._config.host,
            port=self._log_server_port,
            log_level="warning",
            log_config=None,
            timeout_keep_alive=120,
        )
        self._log_server = uvicorn.Server(log_server_config)
        self._threads.spawn_server(self._log_server, name="log-server")
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._log_server is not None and self._log_server.started,
            timeout=Duration.from_seconds(5.0),
        )

        self._log_service_address = f"http://{self.external_host}:{self._log_server_port}"
        logger.info("Local log server ready at %s", self._log_service_address)

        # Now that the address is live, wire up the push-side clients.
        log_client_interceptors = _log_client_interceptors(self._config)
        self._log_pusher = LogPusher(self._log_service_address, interceptors=log_client_interceptors)
        provider_log_pusher = LogPusher(self._log_service_address, interceptors=log_client_interceptors)
        if isinstance(self._provider, (K8sTaskProvider, WorkerProvider)):
            self._provider.log_pusher = provider_log_pusher

    def start(self) -> None:
        """Start main controller loop, dashboard server, and optionally autoscaler."""
        self._started = True
        if self._config.dry_run:
            logger.info("[DRY-RUN] Controller started in dry-run mode — all side effects suppressed")

        # Start the local log server first so the address is live before any
        # worker resolves /system/log-server or any provider tries to push.
        self._start_local_log_server()

        if isinstance(self._provider, K8sTaskProvider):
            self._heartbeat_thread = self._threads.spawn(self._run_direct_provider_loop, name="provider-loop")
        else:
            self._scheduling_thread = self._threads.spawn(self._run_scheduling_loop, name="scheduling-loop")
            self._heartbeat_thread = self._threads.spawn(self._run_provider_loop, name="provider-loop")
            if not self._config.dry_run:
                self._profile_thread = self._threads.spawn(self._run_profile_loop, name="profile-loop")
                self._prune_thread = self._threads.spawn(self._run_prune_loop, name="prune-loop")

        # Create and start uvicorn server via spawn_server, which bridges the
        # ManagedThread stop_event to server.should_exit automatically.
        # timeout_keep_alive: uvicorn defaults to 5s, which races with client polling
        # intervals of the same length, causing TCP resets on idle connections. Use 120s
        # to safely cover long polling gaps during job waits.
        server_config = uvicorn.Config(
            self._dashboard.app,
            host=self._config.host,
            port=self._config.port,
            log_level="warning",
            log_config=None,
            timeout_keep_alive=120,
        )
        self._server = uvicorn.Server(server_config)
        self._threads.spawn_server(self._server, name="controller-server")

        if self._autoscaler:
            logger.info("Autoscaler configured with %d scale groups", len(self._autoscaler.groups))
            self._autoscaler_thread = self._threads.spawn(self._run_autoscaler_loop, name="autoscaler-loop")

        # Register atexit hook to capture final state for post-mortem analysis.
        # Unregistered in stop() so it doesn't fire against a closed DB.
        self._atexit_registered = True
        atexit.register(self._atexit_checkpoint)

        # Wait for server startup with exponential backoff
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=Duration.from_seconds(5.0),
        )

        # Register system endpoints with the externally-reachable URL so
        # workers can resolve them via ListEndpoints. ``_log_service_address``
        # is always populated by now — either from config (external mode) or
        # from the uvicorn thread started above (local mode).
        assert self._log_service_address is not None, "log service address must be resolved before start() completes"
        self._service._system_endpoints["/system/log-server"] = self._log_service_address
        logger.info("Registered system endpoint /system/log-server -> %s", self._log_service_address)

    def stop(self) -> None:
        """Stop all background components gracefully.

        Shutdown ordering:
        1. Unregister atexit hook so it doesn't fire against a closed DB.
        2. Stop scheduling/heartbeat/autoscaler loops so no new work is triggered.
        3. Shut down the autoscaler (stops monitors, terminates VMs, stops platform).
        4. Stop remaining threads (server) and executors.
        """
        # Unregister atexit hook before closing DB connections.
        if self._atexit_registered:
            atexit.unregister(self._atexit_checkpoint)
            self._atexit_registered = False
        self._wake_event.set()
        self._heartbeat_event.set()
        join_timeout = Duration.from_seconds(5.0)
        if self._scheduling_thread:
            self._scheduling_thread.stop()
            self._scheduling_thread.join(timeout=join_timeout)
        if self._heartbeat_thread:
            self._heartbeat_thread.stop()
            self._heartbeat_thread.join(timeout=join_timeout)
        if self._prune_thread:
            self._prune_thread.stop()
            self._prune_thread.join(timeout=join_timeout)
        if self._autoscaler_thread:
            self._autoscaler_thread.stop()
            self._autoscaler_thread.join(timeout=join_timeout)

        if self._autoscaler:
            self._autoscaler.shutdown()

        self._threads.stop()
        self._provider.close()

        # Remove log handler before closing log resources to avoid errors
        # from late log records hitting a closed store or connection.
        logging.getLogger("iris").removeHandler(self._log_handler)
        self._log_handler.close()
        if self._log_pusher:
            self._log_pusher.close()
        if self._remote_log_service:
            self._remote_log_service.close()
        if self._log_service:
            self._log_service.close()
        self._db.close()
        self._bundle_store.close()

    def _atexit_checkpoint(self) -> None:
        """Best-effort checkpoint at interpreter shutdown for post-mortem analysis."""
        if self._config.dry_run:
            return
        try:
            path, _result = write_checkpoint(self._db, self._config.remote_state_dir)
            logger.info("atexit checkpoint written: %s", path)
        except Exception:
            logger.exception("atexit checkpoint failed")

    def _run_scheduling_loop(self, stop_event: threading.Event) -> None:
        """Scheduling loop with adaptive backoff.

        Backs off from min to max interval when idle (no pending tasks or no
        assignments possible). Resets to min interval when woken by a new job
        submission or when assignments are made.
        """
        backoff = ExponentialBackoff(
            initial=self._config.scheduler_min_interval.to_seconds(),
            maximum=self._config.scheduler_max_interval.to_seconds(),
            factor=2.0,
            jitter=0.1,
        )
        while not stop_event.is_set():
            interval = backoff.next_interval()
            woken = self._wake_event.wait(timeout=interval)
            self._wake_event.clear()

            if stop_event.is_set():
                break

            if self._checkpoint_paused.is_set():
                continue

            if woken:
                backoff.reset()

            self._enforce_execution_timeouts()

            outcome = self._run_scheduling()
            if outcome == SchedulingOutcome.ASSIGNMENTS_MADE:
                backoff.reset()

    def _run_prune_loop(self, stop_event: threading.Event) -> None:
        """Background pruning loop: history cleanup every 60s, full data prune on the configured interval."""
        last_full_prune = 0.0
        resource_history_limiter = RateLimiter(interval_seconds=600.0)
        full_prune_interval = self._config.prune_interval.to_seconds()

        while not stop_event.is_set():
            stop_event.wait(timeout=_HISTORY_CLEANUP_INTERVAL_S)
            if stop_event.is_set():
                break

            try:
                self._transitions.prune_worker_task_history()
            except Exception:
                logger.exception("Worker task history cleanup failed")

            if resource_history_limiter.should_run():
                try:
                    self._transitions.prune_worker_resource_history()
                except Exception:
                    logger.exception("Worker resource history cleanup failed")
                try:
                    self._transitions.prune_task_resource_history()
                except Exception:
                    logger.exception("Task resource history cleanup failed")

            now = time.monotonic()
            if now - last_full_prune >= full_prune_interval:
                last_full_prune = now
                try:
                    self._transitions.prune_old_data(
                        job_retention=self._config.job_retention,
                        worker_retention=self._config.worker_retention,
                        log_retention=self._config.log_retention,
                        txn_action_retention=self._config.txn_action_retention,
                        profile_retention=self._config.profile_retention,
                        stop_event=stop_event,
                    )
                except Exception:
                    logger.exception("Data pruning failed")

    def _run_autoscaler_loop(self, stop_event: threading.Event) -> None:
        """Autoscaler loop: runs on its own thread so blocking cloud API calls
        don't stall scheduling or heartbeats."""
        limiter = RateLimiter(interval_seconds=self._autoscaler.evaluation_interval.to_seconds())
        while not stop_event.is_set():
            if not limiter.wait(cancel=stop_event):
                break
            if self._checkpoint_paused.is_set():
                continue
            try:
                self._run_autoscaler_once()
            except Exception:
                logger.exception("Autoscaler loop iteration failed")

            if self._periodic_checkpoint_limiter is not None and self._periodic_checkpoint_limiter.should_run():
                if not self._config.dry_run:
                    try:
                        write_checkpoint(self._db, self._config.remote_state_dir)
                    except Exception:
                        logger.exception("Periodic checkpoint failed")

    def _run_provider_loop(self, stop_event: threading.Event) -> None:
        """Provider sync loop on its own thread so slow RPCs don't block scheduling."""
        limiter = RateLimiter(interval_seconds=self._config.heartbeat_interval.to_seconds())
        while not stop_event.is_set():
            self._heartbeat_event.wait(timeout=limiter.time_until_next())
            self._heartbeat_event.clear()
            limiter.mark_run()
            if stop_event.is_set():
                break
            if self._checkpoint_paused.is_set():
                continue
            try:
                with self._heartbeat_lock:
                    self._sync_all_execution_units()
            except Exception:
                logger.exception("Provider sync round failed, will retry next interval")

    def _run_direct_provider_loop(self, stop_event: threading.Event) -> None:
        """Provider sync loop for K8sTaskProvider: no scheduling, no workers."""
        limiter = RateLimiter(interval_seconds=self._config.heartbeat_interval.to_seconds())
        while not stop_event.is_set():
            self._heartbeat_event.wait(timeout=limiter.time_until_next())
            self._heartbeat_event.clear()
            limiter.mark_run()
            if stop_event.is_set():
                break
            if self._checkpoint_paused.is_set():
                continue
            try:
                self._sync_direct_provider()
            except Exception:
                logger.exception("Direct provider sync round failed, will retry next interval")

    def _sync_direct_provider(self) -> None:
        if self._config.dry_run:
            return
        assert isinstance(self._provider, K8sTaskProvider)
        provider = self._provider
        with self._heartbeat_lock:
            max_promotions = self._promotion_bucket.available
            batch = self._transitions.drain_for_direct_provider(
                max_promotions=max_promotions,
            )
            if batch.tasks_to_run:
                self._promotion_bucket.try_acquire(len(batch.tasks_to_run))
            result = provider.sync(batch)
            tx_result = self._transitions.apply_direct_provider_updates(result.updates)
            self._provider_scheduling_events = list(result.scheduling_events) if result.scheduling_events else []
            self._provider_capacity = result.capacity
            if tx_result.tasks_to_kill:
                self.kill_tasks_on_workers(tx_result.tasks_to_kill, tx_result.task_kill_workers)

    def _run_profile_loop(self, stop_event: threading.Event) -> None:
        """Periodically capture CPU and memory profiles for all running tasks.

        Runs on its own thread with a rate limiter. For each running task, sends
        ProfileTask RPCs (CPU then memory, sequentially) to the task's worker and
        stores the results in the controller DB.
        """
        limiter = RateLimiter(interval_seconds=self._config.profile_interval.to_seconds())
        while not stop_event.is_set():
            remaining = limiter.time_until_next()
            if remaining > 0:
                stop_event.wait(timeout=remaining)
            if stop_event.is_set():
                break
            limiter.mark_run()
            if self._checkpoint_paused.is_set():
                continue
            try:
                self._profile_all_running_tasks()
            except Exception:
                logger.exception("Profile loop iteration failed")

    def _profile_all_running_tasks(self) -> None:
        """Capture CPU and memory profiles for every running task and store in the DB."""
        workers = healthy_active_workers_with_attributes(self._db)
        if not workers:
            return
        workers_by_id = {w.worker_id: w for w in workers}
        tasks_by_worker = running_tasks_by_worker(self._db, set(workers_by_id.keys()))

        profile_targets: list[tuple[JobName, WorkerRow]] = []
        for worker_id, task_ids in tasks_by_worker.items():
            worker = workers_by_id[worker_id]
            for task_id in task_ids:
                profile_targets.append((task_id, worker))

        if not profile_targets:
            return

        cpu_profile_type = job_pb2.ProfileType(
            cpu=job_pb2.CpuProfile(format=job_pb2.CpuProfile.RAW),
        )
        self._dispatch_profiles(profile_targets, cpu_profile_type, "cpu", self._config.profile_duration)

        memory_profile_type = job_pb2.ProfileType(
            memory=job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.STATS),
        )
        self._dispatch_profiles(profile_targets, memory_profile_type, "memory", self._config.profile_duration)

        logger.info("Profile round (cpu+memory): captured for %d tasks", len(profile_targets))

    def _dispatch_profiles(
        self,
        targets: list[tuple[JobName, WorkerRow]],
        profile_type: job_pb2.ProfileType,
        profile_kind: str,
        duration: int,
    ) -> None:
        """Send profile RPCs for the given targets with bounded concurrency."""
        concurrency = min(self._config.profile_concurrency, len(targets))
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="profile") as pool:
            futures = [
                pool.submit(self._capture_one_profile, task_id, worker, profile_type, profile_kind, duration)
                for task_id, worker in targets
            ]
            for future in as_completed(futures):
                future.result()

    def _capture_one_profile(
        self,
        task_id: JobName,
        worker: WorkerRow,
        profile_type: job_pb2.ProfileType,
        profile_kind: str,
        duration: int,
    ) -> None:
        """Capture a single task profile via RPC and store it in the DB."""
        try:
            request = job_pb2.ProfileTaskRequest(
                target=task_id.to_wire(),
                duration_seconds=duration,
                profile_type=profile_type,
            )
            timeout_ms = duration * 1000 + 30000
            resp = self._provider.profile_task(worker.address, request, timeout_ms=timeout_ms)
            if resp.error:
                logger.debug("Profile (%s) failed for %s: %s", profile_kind, task_id, resp.error)
                return
            if not resp.profile_data:
                logger.debug("Empty %s profile for %s", profile_kind, task_id)
                return
            insert_task_profile(
                self._db,
                task_id=task_id.to_wire(),
                profile_data=resp.profile_data,
                captured_at=Timestamp.now(),
                profile_kind=profile_kind,
            )
            logger.debug("Stored %d byte %s profile for %s", len(resp.profile_data), profile_kind, task_id)
        except Exception:
            logger.debug("Profile capture (%s) failed for %s", profile_kind, task_id, exc_info=True)

    def _is_reservation_satisfied(
        self,
        job: JobSchedulingRow,
        claims: dict[WorkerId, ReservationClaim] | None = None,
    ) -> bool:
        """Check if a job's reservation is fully satisfied.

        Returns True if the job has no reservation or if enough workers
        have been claimed to cover every reservation entry.
        """
        if not job.has_reservation:
            return True

        claim_map = claims if claims is not None else _read_reservation_claims(self._db)
        claimed = self._count_reservation_claims(job.job_id.to_wire(), claim_map)
        entry_count = self._reservation_entry_count(job.job_id)
        return claimed >= entry_count

    def _count_reservation_claims(self, job_id_wire: str, claims: dict[WorkerId, ReservationClaim]) -> int:
        """Count workers claimed for the given job."""
        return sum(1 for c in claims.values() if c.job_id == job_id_wire)

    def _reservation_entry_count(self, job_id: JobName) -> int:
        """Get the number of reservation entries for a job from job_config.

        Only called for the rare jobs that have reservations.
        """
        with self._db.read_snapshot() as q:
            row = q.fetchone("SELECT reservation_json FROM job_config WHERE job_id = ?", (job_id.to_wire(),))
        if row is None or row[0] is None:
            return 0
        return len(reservation_entries_from_json(row[0]))

    def _cleanup_stale_claims(self, claims: dict[WorkerId, ReservationClaim] | None = None) -> bool:
        """Remove claims for workers that disappeared or jobs that finished."""
        persisted = False
        if claims is None:
            claims = _read_reservation_claims(self._db)
            persisted = True
        with self._db.read_snapshot() as snapshot:
            active_worker_ids = {
                WorkerId(str(row[0]))
                for row in snapshot.fetchall("SELECT w.worker_id FROM workers w WHERE w.active = 1")
            }
        claimed_job_ids = {JobName.from_wire(claim.job_id) for claim in claims.values()}
        claimed_jobs = list(_jobs_by_id(self._db, claimed_job_ids).values()) if claimed_job_ids else []
        jobs_by_id = {job.job_id.to_wire(): job for job in claimed_jobs}
        stale: list[WorkerId] = []
        for worker_id, claim in claims.items():
            if worker_id not in active_worker_ids:
                stale.append(worker_id)
                continue
            job = jobs_by_id.get(claim.job_id)
            if job is None or job_is_finished(job.state):
                stale.append(worker_id)
        for wid in stale:
            del claims[wid]
        if stale and persisted:
            self._transitions.replace_reservation_claims(claims)
        return bool(stale)

    def _claim_workers_for_reservations(self, claims: dict[WorkerId, ReservationClaim] | None = None) -> bool:
        """Assign unclaimed workers to unsatisfied reservation entries.

        Scans all non-finished jobs with reservations. For each unfulfilled
        entry, finds an eligible unclaimed worker and records the claim.
        """
        persisted = False
        if claims is None:
            claims = _read_reservation_claims(self._db)
            persisted = True
        claimed_entries: set[tuple[str, int]] = {(c.job_id, c.entry_idx) for c in claims.values()}
        claimed_worker_ids: set[WorkerId] = set(claims.keys())
        all_workers = healthy_active_workers_with_attributes(self._db)
        changed = False

        reservable_states = (
            job_pb2.JOB_STATE_PENDING,
            job_pb2.JOB_STATE_BUILDING,
            job_pb2.JOB_STATE_RUNNING,
        )
        reservation_jobs = _jobs_with_reservations(self._db, reservable_states)
        for job in reservation_jobs:
            job_wire = job.job_id.to_wire()
            for idx, res_entry in enumerate(reservation_entries_from_json(job.reservation_json)):
                if (job_wire, idx) in claimed_entries:
                    continue

                for worker in all_workers:
                    if worker.worker_id in claimed_worker_ids:
                        continue
                    if not worker.healthy:
                        continue
                    if not _worker_matches_reservation_entry(worker, res_entry):
                        continue

                    claims[worker.worker_id] = ReservationClaim(
                        job_id=job_wire,
                        entry_idx=idx,
                    )
                    claimed_worker_ids.add(worker.worker_id)
                    claimed_entries.add((job_wire, idx))
                    changed = True
                    break
        if changed and persisted:
            self._transitions.replace_reservation_claims(claims)
        return changed

    def _run_scheduling(self) -> SchedulingOutcome:
        """Run one scheduling cycle.

        Six-phase scheduling:
        1. Reservation claims: clean up stale claims and claim workers for
           reservation jobs.
        2. State reads: fetch pending tasks and workers, filter by deadlines,
           reservation gates, and per-job cap.
        3. Budget/band interleaving: compute user spend, map tasks to effective
           priority bands (down-weighting over-budget users), round-robin users
           within each band.
        4. Preference pass: steer reservation tasks toward their claimed workers
           (skips coscheduled jobs which need atomic assignment).
        5. Normal scheduling: run find_assignments for all remaining tasks.
        6. Preemption pass: evict lower-priority running tasks to free capacity
           for higher-priority unscheduled work.

        Phases 4-6 share a single SchedulingContext so capacity deductions
        are visible across passes.

        No lock is needed since only one scheduling thread exists. All state
        reads and writes go through ControllerTransitions, and every DB access
        is serialized by ControllerDB._lock with multi-statement mutations
        wrapped in BEGIN IMMEDIATE transactions.
        """
        self._scheduling_round += 1
        trace = self._scheduling_round % _SCHEDULING_TRACE_INTERVAL == 0

        claims = self._refresh_reservation_claims()

        timer = Timer()
        state = self._read_scheduling_state()

        if trace:
            logger.info(
                "[TRACE round=%d] Phase 0: %d pending tasks, %d workers, %d reservation claims",
                self._scheduling_round,
                len(state.pending_tasks),
                len(state.workers),
                len(claims),
            )

        if not state.pending_tasks:
            self._scheduling_diagnostics = {}
            return SchedulingOutcome.NO_PENDING_TASKS

        gated = self._apply_scheduling_gates(state.pending_tasks, claims, trace=trace)

        if not gated.schedulable_task_ids:
            self._scheduling_diagnostics = {}
            return SchedulingOutcome.NO_PENDING_TASKS

        order = self._compute_scheduling_order(
            gated.schedulable_task_ids,
            state.pending_tasks,
            gated.jobs,
            trace=trace,
        )

        all_assignments, context, tainted_jobs = self._run_scheduler_pass(
            order, gated, state, claims, timer, trace=trace
        )

        preemptions = self._apply_preemptions(order, tainted_jobs, all_assignments, claims, context)

        self._cache_scheduling_diagnostics(context, tainted_jobs, all_assignments, order.ordered_task_ids)

        if all_assignments or preemptions:
            return SchedulingOutcome.ASSIGNMENTS_MADE
        return SchedulingOutcome.NO_ASSIGNMENTS

    def _refresh_reservation_claims(self) -> dict[WorkerId, ReservationClaim]:
        """Read, clean up, and refresh reservation claims. Returns updated claims."""
        # Claims are read outside the scheduling transaction. This creates a
        # narrow race window where a worker could be removed between claim reads
        # and scheduling, but it's benign: queue_assignments() re-validates all
        # assignments transactionally, and stale claims are cleaned up next cycle.
        claims = _read_reservation_claims(self._db)
        claims_changed = self._cleanup_stale_claims(claims)
        claims_changed = self._claim_workers_for_reservations(claims) or claims_changed
        if claims_changed:
            if self._config.dry_run:
                logger.info("[DRY-RUN] Would update %d reservation claims", len(claims))
            else:
                self._transitions.replace_reservation_claims(claims)
        return claims

    def _read_scheduling_state(self) -> _SchedulingStateRead:
        """Fetch pending tasks and healthy workers from the DB."""
        timer = Timer()
        with slow_log(logger, "scheduling state reads", threshold_ms=50):
            pending_tasks = _schedulable_tasks(self._db)
            workers = healthy_active_workers_with_attributes(self._db)
        return _SchedulingStateRead(
            pending_tasks=pending_tasks,
            workers=workers,
            state_read_ms=timer.elapsed_ms(),
        )

    def _apply_scheduling_gates(
        self,
        pending_tasks: list[TaskRow],
        claims: dict[WorkerId, ReservationClaim],
        trace: bool = False,
    ) -> _GatedCandidates:
        """Filter tasks by deadline, reservation satisfaction, and per-job cap."""
        schedulable_task_ids: list[JobName] = []
        jobs: dict[JobName, JobRequirements] = {}
        has_reservation: set[JobName] = set()
        has_direct_reservation: set[JobName] = set()
        tasks_per_job: dict[JobName, int] = defaultdict(int)
        cap = self._config.max_tasks_per_job_per_cycle
        filter_counts: dict[str, int] = defaultdict(int)
        jobs_by_id = _jobs_by_id(self._db, {task.job_id for task in pending_tasks})
        for task in pending_tasks:
            if not task_row_can_be_scheduled(task):
                filter_counts["task_not_schedulable"] += 1
                continue
            job = jobs_by_id.get(task.job_id)
            if not job:
                filter_counts["job_not_found"] += 1
                continue
            deadline = job_scheduling_deadline(job.scheduling_deadline_epoch_ms)
            if deadline is not None and deadline.expired():
                filter_counts["deadline_expired"] += 1
                self._mark_task_unschedulable(task)
                continue
            # Gate: skip real tasks whose job has an unsatisfied reservation.
            # Holder tasks are always schedulable (they ARE the reservation).
            if not job.is_reservation_holder and not self._is_reservation_satisfied(job, claims):
                filter_counts["reservation_unsatisfied"] += 1
                continue
            if cap > 0 and not job.has_coscheduling and tasks_per_job[task.job_id] >= cap:
                filter_counts["per_job_cap"] += 1
                continue
            tasks_per_job[task.job_id] += 1
            schedulable_task_ids.append(task.task_id)
            if task.job_id not in jobs:
                jobs[task.job_id] = job_requirements_from_job(job)
                if job.has_reservation:
                    has_reservation.add(task.job_id)
                    has_direct_reservation.add(task.job_id)
                elif _find_reservation_ancestor(self._db, task.job_id) is not None:
                    has_reservation.add(task.job_id)
        if trace:
            logger.info(
                "[TRACE] Phase 2 gates: %d/%d tasks passed, %d distinct jobs; filtered: %s",
                len(schedulable_task_ids),
                len(pending_tasks),
                len(jobs),
                dict(filter_counts),
            )
        return _GatedCandidates(
            schedulable_task_ids=schedulable_task_ids,
            jobs=jobs,
            has_reservation=has_reservation,
            has_direct_reservation=has_direct_reservation,
        )

    def _compute_scheduling_order(
        self,
        schedulable_task_ids: list[JobName],
        pending_tasks: list[TaskRow],
        jobs: dict[JobName, JobRequirements],
        trace: bool = False,
    ) -> _SchedulingOrder:
        """Compute priority-band interleaving order.

        Maps tasks to effective bands (down-weighting over-budget users) and
        round-robins users within each band.
        """
        with self._db.read_snapshot() as budget_snapshot:
            user_spend = compute_user_spend(budget_snapshot)
        user_budget_limits = self._db.get_all_user_budget_limits()
        task_band_map: dict[JobName, int] = {
            task.task_id: compute_effective_band(task.priority_band, task.task_id.user, user_spend, user_budget_limits)
            for task in pending_tasks
        }
        tasks_by_band: dict[int, list[JobName]] = defaultdict(list)
        for task_id in schedulable_task_ids:
            band = task_band_map.get(task_id, job_pb2.PRIORITY_BAND_INTERACTIVE)
            tasks_by_band[band].append(task_id)

        interleaved: list[JobName] = []
        for band_key in sorted(tasks_by_band.keys()):
            band_tasks = tasks_by_band[band_key]
            user_tasks = [UserTask(user_id=tid.user, task=tid) for tid in band_tasks]
            interleaved.extend(interleave_by_user(user_tasks, user_spend))

        if trace:
            band_summary = {band: len(tids) for band, tids in tasks_by_band.items()}
            active_spend = {u: v for u, v in user_spend.items() if v > 0}
            logger.info(
                "[TRACE] Phase 3 order: %d tasks after interleaving+cap; bands=%s user_spend=%s budget_limits=%s",
                len(interleaved),
                band_summary,
                active_spend,
                user_budget_limits,
            )
        return _SchedulingOrder(
            ordered_task_ids=interleaved,
            task_band_map=task_band_map,
            user_spend=user_spend,
            user_budget_limits=user_budget_limits,
        )

    def _run_scheduler_pass(
        self,
        order: _SchedulingOrder,
        gated: _GatedCandidates,
        state: _SchedulingStateRead,
        claims: dict[WorkerId, ReservationClaim],
        timer: Timer,
        trace: bool = False,
    ) -> tuple[list[tuple[JobName, WorkerId]], SchedulingContext, dict[JobName, JobRequirements]]:
        """Run preference + normal assignment passes. Returns (assignments, context, taint-injected jobs)."""
        modified_workers = _inject_reservation_taints(state.workers, claims)
        modified_jobs = _inject_taint_constraints(gated.jobs, gated.has_reservation, gated.has_direct_reservation)

        with slow_log(logger, "building_counts", threshold_ms=50):
            building_counts = _building_counts(self._db, workers=state.workers)
        context = self._scheduler.create_scheduling_context(
            modified_workers,
            building_counts=building_counts,
            pending_tasks=order.ordered_task_ids,
            jobs=modified_jobs,
        )

        if trace:
            logger.info(
                "[TRACE] Phase 4 context: %d workers, %d pending tasks, %d jobs",
                len(context.capacities),
                len(context.pending_tasks),
                len(context.jobs),
            )

        # Soft preference — steer reservation tasks toward claimed workers.
        # Skips coscheduled jobs (they need atomic all-or-nothing via find_assignments).
        preference_assignments = _preference_pass(context, gated.has_reservation, claims)

        result = self._scheduler.find_assignments(context)

        all_assignments = preference_assignments + result.assignments
        if trace:
            logger.info(
                "[TRACE] Phase 5 assignments: %d total (%d preferred, %d normal)",
                len(all_assignments),
                len(preference_assignments),
                len(result.assignments),
            )
        if all_assignments:
            with slow_log(logger, "buffer_assignments", threshold_ms=200):
                self._buffer_assignments(all_assignments)
            logger.debug(
                "Scheduling cycle: %d assignments (%d preferred, %d normal), %dms (state read: %dms)",
                len(all_assignments),
                len(preference_assignments),
                len(result.assignments),
                timer.elapsed_ms(),
                state.state_read_ms,
            )
        return all_assignments, context, modified_jobs

    def _apply_preemptions(
        self,
        order: _SchedulingOrder,
        jobs: dict[JobName, JobRequirements],
        all_assignments: list[tuple[JobName, WorkerId]],
        claims: dict[WorkerId, ReservationClaim],
        context: SchedulingContext,
    ) -> list[tuple[JobName, JobName]]:
        """Evict lower-priority running tasks for higher-priority unscheduled work."""
        assigned_ids = {task_id for task_id, _ in all_assignments}
        unscheduled = [
            PreemptionCandidate(
                job_name=tid,
                requirements=jobs[tid.parent],
                band=order.task_band_map.get(tid, job_pb2.PRIORITY_BAND_INTERACTIVE),
            )
            for tid in order.ordered_task_ids
            if tid not in assigned_ids and tid.parent is not None and tid.parent in jobs
        ]
        preemptions: list[tuple[JobName, JobName]] = []
        if unscheduled:
            claimed_workers = set(claims.keys())
            running_info = _get_running_tasks_with_band_and_value(
                self._db, claimed_workers, user_spend=order.user_spend, user_budget_limits=order.user_budget_limits
            )
            preemptions = _run_preemption_pass(unscheduled, running_info, context)
            for preemptor_name, victim_id in preemptions:
                preempt_result = self._transitions.preempt_task(victim_id, reason=f"Preempted by {preemptor_name}")
                self.kill_tasks_on_workers(preempt_result.tasks_to_kill)
            if preemptions:
                logger.info("Preemption pass: %d tasks preempted", len(preemptions))
        return preemptions

    def _cache_scheduling_diagnostics(
        self,
        context: SchedulingContext,
        jobs: dict[JobName, JobRequirements],
        assignments: list[tuple[JobName, WorkerId]],
        schedulable_task_ids: list[JobName],
    ) -> None:
        """Compute and cache scheduling diagnostics for unassigned jobs."""
        assigned_task_ids = {task_id for task_id, _ in assignments}

        # Find unassigned jobs with a representative task
        unscheduled: dict[JobName, tuple[JobName, int]] = {}
        for task_id in schedulable_task_ids:
            if task_id in assigned_task_ids or task_id.parent is None:
                continue
            job_id = task_id.parent
            if job_id in unscheduled:
                _, count = unscheduled[job_id]
                unscheduled[job_id] = (unscheduled[job_id][0], count + 1)
            else:
                unscheduled[job_id] = (task_id, 1)

        diagnostics: dict[str, str] = {}
        for job_id, (representative_task, num_tasks) in unscheduled.items():
            req = jobs.get(job_id)
            if req is None:
                continue
            reason = self._scheduler.get_job_scheduling_diagnostics(
                req,
                context,
                representative_task,
                num_tasks=num_tasks,
            )
            diagnostics[job_id.to_wire()] = reason

        # Atomic replacement — safe for concurrent reads under the GIL.
        self._scheduling_diagnostics = diagnostics

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None:
        """Return cached scheduling diagnostic for a job, or None if unavailable."""
        return self._scheduling_diagnostics.get(job_wire_id)

    def _buffer_assignments(
        self,
        assignments: list[tuple[JobName, WorkerId]],
    ) -> None:
        """Commit assignments and enqueue worker dispatches in one state command."""
        if self._config.dry_run:
            for task_id, worker_id in assignments:
                logger.info("[DRY-RUN] Would assign task %s to worker %s", task_id, worker_id)
            return
        command = [Assignment(task_id=task_id, worker_id=worker_id) for task_id, worker_id in assignments]
        result = self._transitions.queue_assignments(command)
        if result.has_real_dispatch:
            self._heartbeat_event.set()

    _TIMEOUT_CHECK_INTERVAL_MS = 60_000  # Check at most once per minute.

    def _enforce_execution_timeouts(self) -> None:
        """Kill executing tasks that have exceeded their job's execution timeout.

        Throttled to run at most once per minute. Queries for tasks in
        BUILDING/RUNNING state whose started_at_ms + timeout is in the past,
        then cancels them via the same kill path used for job cancellation.
        """
        if self._config.dry_run:
            return
        now = Timestamp.now()
        now_ms = now.epoch_ms()
        if now_ms - self._last_timeout_check_ms < self._TIMEOUT_CHECK_INTERVAL_MS:
            return
        self._last_timeout_check_ms = now_ms
        timed_out = timed_out_executing_tasks(self._db, now)
        if not timed_out:
            return
        for task in timed_out:
            logger.warning("Task %s exceeded execution timeout, killing", task.task_id)
        task_ids = {t.task_id for t in timed_out}
        result = self._transitions.cancel_tasks_for_timeout(task_ids, reason="Execution timeout exceeded")
        if result.tasks_to_kill:
            self.kill_tasks_on_workers(result.tasks_to_kill, result.task_kill_workers)

    def _mark_task_unschedulable(self, task: TaskRow) -> None:
        """Mark a task as unschedulable due to timeout."""
        if self._config.dry_run:
            logger.info("[DRY-RUN] Would mark task %s as unschedulable", task.task_id)
            return
        job = _jobs_by_id(self._db, {task.job_id}).get(task.job_id)
        if job and job.scheduling_timeout_ms is not None:
            timeout = Duration.from_ms(job.scheduling_timeout_ms)
        else:
            timeout = None
        logger.warning(f"Task {task.task_id} exceeded scheduling timeout ({timeout}), marking as UNSCHEDULABLE")
        result = self._transitions.mark_task_unschedulable(
            task.task_id,
            reason=f"Scheduling timeout exceeded ({timeout})",
        )
        if result.tasks_to_kill:
            self.kill_tasks_on_workers(result.tasks_to_kill, result.task_kill_workers)

    def create_scheduling_context(self, workers: list[WorkerRow]) -> SchedulingContext:
        """Create a scheduling context for the given workers."""
        building_counts = _building_counts(self._db, workers)
        return self._scheduler.create_scheduling_context(
            workers,
            building_counts=building_counts,
        )

    def kill_tasks_on_workers(
        self,
        task_ids: set[JobName],
        task_kill_workers: dict[JobName, WorkerId] | None = None,
    ) -> None:
        """Buffer kill requests for delivery via next heartbeat.

        Called after state has marked tasks as killed. For each task that had
        a worker assigned, buffers the kill request for delivery via the next
        heartbeat to that worker. Tasks without a worker assignment are routed
        to the direct kill queue when a K8sTaskProvider is configured.
        """
        if self._config.dry_run:
            logger.info("[DRY-RUN] Would kill %d tasks on workers: %s", len(task_ids), list(task_ids)[:5])
            return
        any_buffered = False
        mapping = dict(task_kill_workers or {})
        unresolved = task_ids - set(mapping.keys())
        if unresolved:
            mapping.update(_task_worker_mapping(self._db, unresolved))
        workers = _workers_by_id(self._db, set(mapping.values()))
        for task_id, worker_id in mapping.items():
            worker = workers.get(worker_id)
            if worker is None:
                continue
            self._transitions.buffer_kill(worker_id, task_id.to_wire())
            any_buffered = True

        # Route kills for tasks without worker assignment to the direct kill queue.
        if isinstance(self._provider, K8sTaskProvider):
            for task_id in task_ids:
                if task_id not in mapping:
                    self._transitions.buffer_direct_kill(task_id.to_wire())
                    any_buffered = True

        # Wake heartbeat thread to deliver buffered kills immediately
        if any_buffered:
            self._heartbeat_event.set()

    def _reap_stale_workers(self) -> None:
        """Fail workers whose last heartbeat exceeds the staleness threshold.

        On controller restart from checkpoint, workers carry their old
        last_heartbeat_ms but consecutive_failures=0 and healthy=1.  If the
        backing VMs were preempted during the outage, we'd otherwise wait for
        10 RPC timeouts per worker before marking them dead.  This check
        short-circuits that by failing any worker that hasn't heartbeated in
        HEARTBEAT_STALENESS_THRESHOLD (15 minutes).
        """
        if isinstance(self._provider, K8sTaskProvider):
            return
        if self._config.dry_run:
            return
        threshold_ms = HEARTBEAT_STALENESS_THRESHOLD.to_ms()
        workers = healthy_active_workers_with_attributes(self._db)
        stale = [w for w in workers if w.last_heartbeat.age_ms() > threshold_ms]
        if not stale:
            return

        logger.warning(
            "Failing %d workers with stale heartbeats (threshold=%ds): %s",
            len(stale),
            HEARTBEAT_STALENESS_THRESHOLD.to_seconds(),
            [str(w.worker_id) for w in stale[:10]],
        )
        failure_result = self._transitions.fail_workers_batch(
            [str(w.worker_id) for w in stale],
            reason=f"heartbeat stale (>{int(HEARTBEAT_STALENESS_THRESHOLD.to_seconds())}s since last heartbeat)",
        )
        for wid, addr in failure_result.removed_workers:
            self._provider.on_worker_failed(wid, addr)
        if self._autoscaler:
            sibling_worker_ids = self._autoscaler.terminate_slices_for_workers(
                [str(wid) for wid, _ in failure_result.removed_workers]
            )
            sibling_failures = self._transitions.fail_workers_batch(
                sibling_worker_ids,
                reason="stale worker failed, slice terminated",
            )
            for wid, addr in sibling_failures.removed_workers:
                self._provider.on_worker_failed(wid, addr)
            failure_result.tasks_to_kill.update(sibling_failures.tasks_to_kill)
            failure_result.task_kill_workers.update(sibling_failures.task_kill_workers)
        if failure_result.tasks_to_kill:
            self.kill_tasks_on_workers(failure_result.tasks_to_kill, failure_result.task_kill_workers)

    def _sync_all_execution_units(self) -> None:
        if self._config.dry_run:
            return
        round_timer = Timer()

        self._reap_stale_workers()

        with slow_log(logger, "provider sync phase 1 (snapshot)", threshold_ms=100):
            batches = self._transitions.drain_dispatch_all()

        if not batches:
            return

        # Sync with the execution backend (ThreadPoolExecutor inside provider).
        results = self._provider.sync(batches)

        acc = _SyncFailureAccumulator()
        with slow_log(logger, "provider sync (apply results)", threshold_ms=500):
            success_reqs, failure_entries = self._separate_sync_results(results)
            self._apply_successful_heartbeats(success_reqs, acc)
            primary_failed_workers = self._handle_failed_heartbeats(failure_entries, acc)
            self._handle_sibling_worker_failures(primary_failed_workers, acc)

            if acc.all_tasks_to_kill:
                self.kill_tasks_on_workers(acc.all_tasks_to_kill, acc.all_task_kill_workers)

        self._log_sync_health_summary(
            batch_count=len(batches),
            fail_count=acc.fail_count,
            failed_workers=acc.failed_workers,
            elapsed_ms=round_timer.elapsed_ms(),
        )

    def _separate_sync_results(
        self,
        results: list,
    ) -> tuple[list, list[tuple]]:
        """Partition provider sync results into successes and failures."""
        success_reqs = []
        failure_entries = []
        for batch, apply_req, error in results:
            if apply_req is not None:
                success_reqs.append(apply_req)
            else:
                failure_entries.append((batch, error or "unknown error"))
        return success_reqs, failure_entries

    def _apply_successful_heartbeats(
        self,
        success_reqs: list,
        acc: _SyncFailureAccumulator,
    ) -> None:
        """Batch-apply successful heartbeat results, accumulating kill targets."""
        if not success_reqs:
            return
        batch_results = self._transitions.apply_heartbeats_batch(success_reqs)
        for result in batch_results:
            acc.all_tasks_to_kill.update(result.tasks_to_kill)
            acc.all_task_kill_workers.update(result.task_kill_workers)

    def _handle_failed_heartbeats(
        self,
        failure_entries: list[tuple],
        acc: _SyncFailureAccumulator,
    ) -> list[str]:
        """Process failed heartbeats: update accumulator and return primary failed worker IDs."""
        failure_result = self._transitions.fail_heartbeats_batch(failure_entries)
        acc.all_tasks_to_kill.update(failure_result.tasks_to_kill)
        acc.all_task_kill_workers.update(failure_result.task_kill_workers)

        primary_failed_workers: list[str] = []
        for (batch, error), result in zip(failure_entries, failure_result.results, strict=False):
            logger.debug("Sync error for %s: %s", batch.worker_id, error)
            if result.action == HeartbeatAction.WORKER_FAILED:
                acc.fail_count += 1
                acc.failed_workers.append(batch.worker_id)
                self._provider.on_worker_failed(batch.worker_id, batch.worker_address)
                primary_failed_workers.append(str(batch.worker_id))
            elif result.action == HeartbeatAction.TRANSIENT_FAILURE:
                acc.fail_count += 1
                acc.failed_workers.append(batch.worker_id)
        return primary_failed_workers

    def _handle_sibling_worker_failures(
        self,
        primary_failed_workers: list[str],
        acc: _SyncFailureAccumulator,
    ) -> None:
        """Terminate slices containing failed workers and fail their siblings."""
        if not self._autoscaler or not primary_failed_workers:
            return
        sibling_worker_ids = self._autoscaler.terminate_slices_for_workers(primary_failed_workers)
        # TODO(#3425): This prunes sibling workers before their in-flight
        # results are processed, causing apply_heartbeat() to
        # silently drop any logs/states those workers reported this round.
        sibling_failures = self._transitions.fail_workers_batch(
            sibling_worker_ids,
            reason="sibling worker failed, slice terminated",
        )
        acc.all_tasks_to_kill.update(sibling_failures.tasks_to_kill)
        acc.all_task_kill_workers.update(sibling_failures.task_kill_workers)
        for wid, addr in sibling_failures.removed_workers:
            self._provider.on_worker_failed(wid, addr)
        if sibling_failures.removed_workers:
            acc.fail_count += len(sibling_failures.removed_workers)
            acc.failed_workers.extend(wid for wid, _ in sibling_failures.removed_workers)
            logger.info(
                "Failed %d sibling workers from slices: %s",
                len(sibling_failures.removed_workers),
                [wid for wid, _ in sibling_failures.removed_workers],
            )

    def _log_sync_health_summary(
        self,
        batch_count: int,
        fail_count: int,
        failed_workers: list[str],
        elapsed_ms: int,
    ) -> None:
        """Log provider sync timing and periodic cluster health summary."""
        level = logging.WARNING if elapsed_ms > _SLOW_HEARTBEAT_MS else logging.DEBUG
        fmt = "Provider sync: %d workers, %d failed, %dms"
        args: list[object] = [batch_count, fail_count, elapsed_ms]
        if failed_workers:
            fmt += " failed=[%s]"
            args.append(", ".join(failed_workers))
        logger.log(level, fmt, *args)

        self._heartbeat_iteration += 1
        if _HEALTH_SUMMARY_INTERVAL.should_run():
            workers = healthy_active_workers_with_attributes(self._db)
            with self._db.read_snapshot() as snap:
                active = snap.fetchone("SELECT COUNT(*) FROM jobs j WHERE j.state = ?", (job_pb2.JOB_STATE_RUNNING,))[
                    0
                ]  # type: ignore[index]
            pending = len(_schedulable_tasks(self._db))
            logger.info(
                "Controller status: %d workers (%d failed), %d active jobs, %d pending tasks",
                len(workers),
                fail_count,
                active,
                pending,
            )

    def _run_autoscaler_once(self) -> None:
        """Run one autoscaler cycle: refresh (I/O) then update (CPU).

        Called from the autoscaler loop thread.
        """
        if not self._autoscaler:
            return

        if self._config.dry_run:
            logger.info("[DRY-RUN] Skipping autoscaler cycle (refresh + update)")
            return

        worker_status_map = self._build_worker_status_map()
        self._autoscaler.refresh(worker_status_map)
        workers = healthy_active_workers_with_attributes(self._db)
        demand_entries = compute_demand_entries(
            self._db,
            self._scheduler,
            workers,
            reservation_claims=_read_reservation_claims(self._db),
        )
        self._autoscaler.update(demand_entries)

    def _build_worker_status_map(self) -> WorkerStatusMap:
        """Build a map of worker_id to worker status for autoscaler idle tracking."""
        result: WorkerStatusMap = {}
        with self._db.read_snapshot() as snapshot:
            rows = snapshot.raw(
                "SELECT worker_id FROM workers WHERE active = 1",
                decoders={"worker_id": WorkerId},
            )
        worker_ids = {row.worker_id for row in rows}
        running_by_worker = running_tasks_by_worker(self._db, worker_ids)
        for wid in worker_ids:
            result[wid] = WorkerStatus(
                worker_id=wid,
                running_task_ids=frozenset(tid.to_wire() for tid in running_by_worker.get(wid, set())),
            )
        return result

    def begin_checkpoint(self) -> tuple[str, CheckpointResult]:
        """Pause loops and write a consistent SQLite checkpoint copy."""
        if self._config.dry_run:
            logger.info("[DRY-RUN] Skipping checkpoint write")
            return ("dry-run", CheckpointResult(created_at=Timestamp.now(), job_count=0, task_count=0, worker_count=0))
        self._checkpoint_paused.set()
        try:
            # Hold the heartbeat lock only for the SQLite backup (consistent
            # snapshot). Compression and GCS upload run outside the lock so
            # heartbeat processing is not blocked for 5-30s.
            with self._heartbeat_lock:
                backup = backup_databases(self._db)
            try:
                path, result = upload_checkpoint(self._db, backup, self._config.remote_state_dir)
            finally:
                backup.cleanup()
            logger.info(
                "Checkpoint written: %s (jobs=%d tasks=%d workers=%d)",
                path,
                result.job_count,
                result.task_count,
                result.worker_count,
            )
            return path, result
        finally:
            self._checkpoint_paused.clear()

    def launch_job(
        self,
        request: controller_pb2.Controller.LaunchJobRequest,
    ) -> controller_pb2.Controller.LaunchJobResponse:
        """Submit a job to the controller."""
        return self._service.launch_job(request, None)

    def get_job_status(
        self,
        job_id: str,
    ) -> controller_pb2.Controller.GetJobStatusResponse:
        """Get the status of a job."""
        request = controller_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        return self._service.get_job_status(request, None)

    def terminate_job(
        self,
        job_id: str,
    ) -> job_pb2.Empty:
        """Terminate a running job."""
        request = controller_pb2.Controller.TerminateJobRequest(job_id=job_id)
        return self._service.terminate_job(request, None)

    # Properties

    @property
    def state(self) -> ControllerTransitions:
        return self._transitions

    @property
    def provider(self) -> TaskProvider | K8sTaskProvider:
        return self._provider

    @property
    def has_direct_provider(self) -> bool:
        return isinstance(self._provider, K8sTaskProvider)

    @property
    def provider_scheduling_events(self) -> list[SchedulingEvent]:
        return self._provider_scheduling_events

    @property
    def provider_capacity(self) -> ClusterCapacity | None:
        return self._provider_capacity

    @property
    def port(self) -> int:
        """Actual bound port (may differ from config if port=0 was specified)."""
        if self._server and self._server.started:
            if self._server.servers and self._server.servers[0].sockets:
                return self._server.servers[0].sockets[0].getsockname()[1]
        return self._config.port

    @property
    def external_host(self) -> str:
        """Externally-reachable host address.

        When bound to 0.0.0.0, probes for the real network IP (same technique
        workers use in env_probe._get_ip_address).
        """
        host = self._config.host
        if host == "0.0.0.0":
            import socket

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    return s.getsockname()[0]
            except Exception:
                return "127.0.0.1"
        return host

    @property
    def url(self) -> str:
        return f"http://{self.external_host}:{self.port}"

    @property
    def reservation_claims(self) -> dict[WorkerId, ReservationClaim]:
        """Current reservation claims, keyed by worker ID."""
        return _read_reservation_claims(self._db)

    @property
    def autoscaler(self) -> "Autoscaler | None":
        """The autoscaler instance, if autoscaling is enabled."""
        return self._autoscaler
