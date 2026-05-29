# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduling, preemption, and reservation policy.

Free functions parameterized by their dependencies (``ControllerDB``,
``WorkerHealthTracker``, ``SchedulingContext``) rather than the ``Controller``
instance. DB I/O is concentrated in the context builders and the reservation-
claim lifecycle; the gate, ordering, and preemption passes are pure transforms
over an in-memory ``SchedulingContext``.
"""

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from rigging.log_setup import slow_log
from sqlalchemy import bindparam, select

from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintOp,
    PlacementRequirements,
    constraints_from_resources,
    evaluate_constraint,
    extract_placement_requirements,
    merge_constraints,
    split_hard_soft,
)
from iris.cluster.controller import db, reads, writes
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.budget import (
    UserTask,
    compute_effective_band,
    compute_user_spend,
    interleave_by_user,
    resource_value,
)
from iris.cluster.controller.codec import (
    constraints_from_json,
    device_counts_from_json,
    device_variant_from_json,
    reservation_entries_from_json,
    resource_spec_from_scalars,
)
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.reads import ReservationClaim, SchedulableWorker
from iris.cluster.controller.reconcile.policy import RESERVATION_HOLDER_JOB_NAME
from iris.cluster.controller.scheduler import (
    DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
    DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
    JobRequirements,
    Scheduler,
    SchedulingContext,
    WorkerCapacity,
    WorkerSnapshot,
    worker_snapshot_from_row,
)
from iris.cluster.controller.schema import (
    job_config_table,
    jobs_table,
    tasks_table,
)
from iris.cluster.controller.task_state import job_scheduling_deadline, task_row_can_be_scheduled
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import (
    JobName,
    PendingTask,
    UserBudgetDefaults,
    WorkerId,
    is_job_finished,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# Sentinel for dry-run scheduling with per-worker limits disabled.
_UNLIMITED = sys.maxsize


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
    cpu_millicores: int
    memory_bytes: int
    gpu_count: int
    tpu_count: int
    # Device variant (e.g. "v5p-64") the task is running on, derived from the
    # task's own resource spec. Used to gate preemption to same-variant victims
    # so a v5p-64 request can never reclaim a v5p-256 slice and vice versa.
    device_variant: str | None = None
    already_preempted: bool = False


@dataclass(frozen=True)
class PreemptionCandidate:
    """An unscheduled task that may preempt running work."""

    job_name: JobName
    requirements: JobRequirements
    band: int  # proto PriorityBand value


@dataclass(frozen=True)
class GatedCandidates:
    """Tasks that passed deadline, reservation, and per-job-cap gates."""

    schedulable_task_ids: list[JobName]
    jobs: dict[JobName, JobRequirements]
    has_reservation: set[JobName]
    has_direct_reservation: set[JobName]
    expired_tasks: list["PendingTask"]


@dataclass(frozen=True)
class SchedulingOrder:
    """Priority-ordered task list with budget context for preemption."""

    ordered_task_ids: list[JobName]
    task_band_map: dict[JobName, int]
    user_spend: dict[str, int]
    user_budget_limits: dict[str, int]


def job_requirements_from_job(job: PendingTask) -> JobRequirements:
    """Convert a job row to scheduler-compatible JobRequirements."""
    dc = device_counts_from_json(job.res_device_json)
    return JobRequirements(
        req_cpu_millicores=job.res_cpu_millicores,
        req_memory_bytes=job.res_memory_bytes,
        req_gpu_count=dc.gpu,
        req_tpu_count=dc.tpu,
        device_variant=device_variant_from_json(job.res_device_json),
        constraints=constraints_from_json(job.constraints_json),
        is_coscheduled=job.has_coscheduling,
        coscheduling_group_by=job.coscheduling_group_by if job.has_coscheduling else None,
    )


def compute_demand_entries(
    queries: ControllerDB,
    scheduler: Scheduler | None = None,
    workers: list[SchedulableWorker] | None = None,
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

    # Single combined query: each row carries task + job + job_config columns.
    # task_row_can_be_scheduled() is already applied inside _pending_tasks_with_jobs.
    tasks_by_job: dict[JobName, list[PendingTask]] = defaultdict(list)
    all_schedulable: list[PendingTask] = []
    with queries.read_snapshot() as tx:
        pending = _pending_tasks_with_jobs(tx)
    for task in pending:
        tasks_by_job[task.job_id].append(task)
        all_schedulable.append(task)

    # Build job requirements once, shared between dry-run and demand emission.
    # Also track which jobs have reservations so we can apply taint injection.
    # Pre-fetch the reserved-job set once so the per-task ancestor walk is
    # pure Python instead of one SQL round trip per unique pending job.
    reserved_jobs = _reserved_job_ids(queries)
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    has_direct_reservation: set[JobName] = set()
    for task in all_schedulable:
        if task.job_id in jobs:
            continue
        jobs[task.job_id] = job_requirements_from_job(task)
        if task.has_reservation:
            has_reservation.add(task.job_id)
            has_direct_reservation.add(task.job_id)
        elif _find_reservation_ancestor(reserved_jobs, task.job_id) is not None:
            has_reservation.add(task.job_id)

    # Dry-run scheduling with building/assignment limits disabled.
    # All tasks participate — holders and real tasks alike.
    absorbed_task_ids: set[JobName] = set()
    if scheduler is not None and workers is not None and workers:
        with queries.read_snapshot() as snap:
            building_counts = reads.building_counts(snap, [w.worker_id for w in workers])
            usage_by_worker = reads.resource_usage_by_worker(snap)
        snapshots = [worker_snapshot_from_row(w, usage_by_worker.get(w.worker_id)) for w in workers]
        task_ids = [t.task_id for t in all_schedulable]
        claims = reservation_claims or {}
        dry_run_workers = inject_reservation_taints(snapshots, claims)
        dry_run_jobs = inject_taint_constraints(jobs, has_reservation, has_direct_reservation)

        # Dry-run scheduling context — only the per-(task, worker) matching loop
        # consumes capacities/jobs/pending_tasks, so the raw-read fields stay
        # empty. Building/assignment limits are disabled so big workers can
        # absorb multiple tasks (prevents false demand on idle clusters).
        context = SchedulingContext(
            workers=dry_run_workers,
            building_counts=building_counts,
            max_building_tasks=_UNLIMITED,
            max_assignments_per_worker=_UNLIMITED,
            pending_tasks=task_ids,
            jobs=dry_run_jobs,
            pending_task_rows=[],
            user_spend={},
            user_budget_limits={},
            requested_bands={},
            reserved_job_ids=frozenset(),
            reservation_entry_counts={},
            user_budget_defaults=UserBudgetDefaults(),
        )
        result = scheduler.find_assignments(context)
        for task_id, _ in result.assignments:
            absorbed_task_ids.add(task_id)

    # Emit demand for all unabsorbed tasks through a single path.
    # Each task row carries job + job_config columns, so we read them from
    # any task in the group (all share the same job).
    for job_id, tasks in tasks_by_job.items():
        # Use the first task to source job-level columns.
        job_row = tasks[0]
        if is_job_finished(job_row.job_state):
            continue

        job_constraints = constraints_from_json(job_row.constraints_json)
        # Build the proto here — DemandEntry.resources is an autoscaler RPC field (legitimate boundary).
        job_resources = resource_spec_from_scalars(
            job_row.res_cpu_millicores, job_row.res_memory_bytes, job_row.res_disk_bytes, job_row.res_device_json
        )

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

        if job_row.has_coscheduling:
            remaining_ids = []
            for t in tasks:
                if t.task_id in absorbed_task_ids:
                    continue
                remaining_ids.append(t.task_id.to_wire())
            if remaining_ids:
                demand_entries.append(
                    DemandEntry(
                        task_ids=remaining_ids,
                        coschedule_group_id=job_id.to_wire(),
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


def read_reservation_claims(db: ControllerDB) -> dict[WorkerId, ReservationClaim]:
    """Read reservation claims from the canonical DB table."""
    with db.read_snapshot() as tx:
        return reads.list_claims(tx)


def _row_to_pending_task(row: Any) -> PendingTask:
    return PendingTask(
        task_id=row.task_id,
        job_id=row.job_id,
        state=int(row.state),
        current_attempt_id=int(row.current_attempt_id),
        failure_count=int(row.failure_count),
        preemption_count=int(row.preemption_count),
        max_retries_failure=int(row.max_retries_failure),
        max_retries_preemption=int(row.max_retries_preemption),
        submitted_at_ms=row.submitted_at_ms,
        priority_band=int(row.priority_band),
        priority_neg_depth=int(row.priority_neg_depth),
        priority_root_submitted_ms=int(row.priority_root_submitted_ms),
        priority_insertion=int(row.priority_insertion),
        job_state=int(row.job_state),
        scheduling_deadline_epoch_ms=row.scheduling_deadline_epoch_ms,
        is_reservation_holder=bool(row.is_reservation_holder),
        has_reservation=bool(row.has_reservation),
        scheduling_timeout_ms=row.scheduling_timeout_ms,
        has_coscheduling=bool(row.has_coscheduling),
        coscheduling_group_by=row.coscheduling_group_by,
        constraints_json=row.constraints_json,
        res_cpu_millicores=int(row.res_cpu_millicores),
        res_memory_bytes=int(row.res_memory_bytes),
        res_disk_bytes=int(row.res_disk_bytes),
        res_device_json=row.res_device_json,
    )


def _pending_tasks_with_jobs(tx: Tx) -> list[PendingTask]:
    """Return scheduling inputs for pending tasks, joining task + job + job_config in one query."""
    rows = tx.execute(
        select(
            tasks_table.c.task_id,
            tasks_table.c.job_id,
            tasks_table.c.state,
            tasks_table.c.current_attempt_id,
            tasks_table.c.failure_count,
            tasks_table.c.preemption_count,
            tasks_table.c.max_retries_failure,
            tasks_table.c.max_retries_preemption,
            tasks_table.c.submitted_at_ms,
            tasks_table.c.priority_band,
            tasks_table.c.priority_neg_depth,
            tasks_table.c.priority_root_submitted_ms,
            tasks_table.c.priority_insertion,
            # job columns (label job_state to avoid clash with tasks.state)
            jobs_table.c.state.label("job_state"),
            jobs_table.c.scheduling_deadline_epoch_ms,
            jobs_table.c.is_reservation_holder,
            jobs_table.c.has_reservation,
            # job_config columns
            job_config_table.c.scheduling_timeout_ms,
            job_config_table.c.has_coscheduling,
            job_config_table.c.coscheduling_group_by,
            job_config_table.c.constraints_json,
            job_config_table.c.res_cpu_millicores,
            job_config_table.c.res_memory_bytes,
            job_config_table.c.res_disk_bytes,
            job_config_table.c.res_device_json,
        )
        .select_from(
            tasks_table.join(jobs_table, jobs_table.c.job_id == tasks_table.c.job_id).join(
                job_config_table, job_config_table.c.job_id == tasks_table.c.job_id
            )
        )
        .where(tasks_table.c.state == bindparam("state"))
        .order_by(
            tasks_table.c.priority_neg_depth.asc(),
            tasks_table.c.priority_root_submitted_ms.asc(),
            tasks_table.c.submitted_at_ms.asc(),
            tasks_table.c.priority_insertion.asc(),
        ),
        {"state": job_pb2.TASK_STATE_PENDING},
    ).all()
    pending_tasks = [_row_to_pending_task(row) for row in rows]
    return [task for task in pending_tasks if task_row_can_be_scheduled(task)]


def _jobs_with_reservations(queries: ControllerDB, states: tuple[int, ...]) -> list:
    """Fetch (job_id, reservation_json) for jobs that hold a reservation.

    Per-tick hot path: only decode what the reservation-claim recomputation
    reads. Filters via ``jobs.has_reservation`` (no scan of ``job_config``)
    and joins ``job_config`` solely to pull ``reservation_json``.
    """
    with db.read_snapshot(queries.sa_read_engine) as tx:
        return tx.execute(
            select(jobs_table.c.job_id, job_config_table.c.reservation_json)
            .select_from(jobs_table.join(job_config_table, jobs_table.c.job_id == job_config_table.c.job_id))
            .where(
                jobs_table.c.state.in_(bindparam("states", expanding=True)),
                jobs_table.c.has_reservation == True,  # noqa: E712 — SQLAlchemy requires == not `is`
            ),
            {"states": list(states)},
        ).all()


def get_running_tasks_with_band_and_value(
    db: ControllerDB,
    claimed_workers: set[WorkerId],
) -> list[RunningTaskInfo]:
    """Query running tasks with band, worker, resource spec, and coscheduling status.

    Skips tasks on reservation-claimed workers since those workers are spoken for.

    The reported band is the value persisted in ``tasks.priority_band``, which
    is stamped at assignment time (see ``_commit_assignments`` and
    ``writes.tasks.assign_task``). The over-budget downgrade is applied at that
    stamping point, not on every scheduling tick, which prevents a running
    task from oscillating into BATCH and back as its own user crosses the
    budget cliff — the source of mutual same-band preemption between two
    users sitting at their limits.
    """
    with db.read_snapshot() as tx:
        rows = tx.execute(
            select(
                tasks_table.c.task_id,
                tasks_table.c.priority_band,
                tasks_table.c.current_worker_id.label("worker_id"),
                job_config_table.c.res_cpu_millicores,
                job_config_table.c.res_memory_bytes,
                job_config_table.c.res_disk_bytes,
                job_config_table.c.res_device_json,
                job_config_table.c.has_coscheduling,
            )
            .select_from(tasks_table.join(job_config_table, tasks_table.c.job_id == job_config_table.c.job_id))
            .where(
                tasks_table.c.state == bindparam("state"),
                tasks_table.c.current_worker_id.is_not(None),
            ),
            {"state": job_pb2.TASK_STATE_RUNNING},
        ).all()
    result: list[RunningTaskInfo] = []
    for row in rows:
        wid = row.worker_id
        if wid in claimed_workers:
            continue
        dc = device_counts_from_json(row.res_device_json)
        result.append(
            RunningTaskInfo(
                task_id=row.task_id,
                worker_id=wid,
                band_sort_key=row.priority_band,
                resource_value=resource_value(
                    row.res_cpu_millicores,
                    row.res_memory_bytes,
                    dc.gpu + dc.tpu,
                ),
                is_coscheduled=bool(int(row.has_coscheduling)),
                cpu_millicores=row.res_cpu_millicores,
                memory_bytes=row.res_memory_bytes,
                gpu_count=dc.gpu,
                tpu_count=dc.tpu,
                device_variant=device_variant_from_json(row.res_device_json),
            )
        )
    return result


def _preempt_solo(
    candidate: PreemptionCandidate,
    wanted_variant: str | None,
    solo_victims: list[RunningTaskInfo],
    context: SchedulingContext,
) -> tuple[JobName, JobName] | None:
    """Find a single solo victim whose eviction would free enough capacity for
    a non-coscheduled preemptor. Mutates the chosen victim's already_preempted
    flag so subsequent candidates skip it. Returns the (preemptor, victim) pair
    or None if no victim qualifies.

    The same-variant gate ensures the freed slot shape matches the preemptor;
    the hypothetical-capacity check covers partial-worker tenancy (e.g. a
    victim using only some of a worker's CPUs or TPUs).
    """
    req = candidate.requirements
    for victim in solo_victims:
        if victim.already_preempted:
            continue
        if victim.device_variant != wanted_variant:
            continue
        # Can only preempt strictly lower priority (higher band_sort_key).
        # `solo_victims` is sorted by descending band_sort_key, so once this
        # gate trips every later victim also fails — break, don't continue,
        # to avoid scanning the unpreemptible tail (issue #5888).
        if victim.band_sort_key <= candidate.band:
            break

        cap = context.capacities.get(victim.worker_id)
        if cap is None:
            continue
        if not cap.matches_constraints(req.constraints):
            continue
        # If current capacity already fits, no preemption needed
        if cap.can_fit(req) is None:
            continue

        # Would freeing this victim's resources create enough capacity?
        hypothetical = WorkerCapacity(
            worker_id=cap.worker_id,
            available_cpu_millicores=cap.available_cpu_millicores + victim.cpu_millicores,
            available_memory=cap.available_memory + victim.memory_bytes,
            available_gpus=cap.available_gpus + victim.gpu_count,
            available_tpus=cap.available_tpus + victim.tpu_count,
            attributes=cap.attributes,
            building_task_count=max(0, cap.building_task_count - 1),
            max_building_tasks=cap.max_building_tasks,
        )
        if hypothetical.can_fit(req) is None:
            victim.already_preempted = True
            return (candidate.job_name, victim.task_id)
    return None


def _preempt_coscheduled(
    candidate: PreemptionCandidate,
    wanted_variant: str | None,
    n_required: int,
    sorted_groups: list[tuple[JobName, list[RunningTaskInfo]]],
) -> list[tuple[JobName, JobName]]:
    """Find a victim slice (all running tasks of one coscheduled job) whose
    eviction satisfies a coscheduled preemptor. Returns one (preemptor, victim)
    pair per slice member, or [] if no slice qualifies. Mutates already_preempted
    on every member of the chosen slice.

    Coscheduled tasks own their workers whole, so once variant matches and the
    slice is at least as large as the preemptor, freeing it yields exactly the
    shape the preemptor needs — no per-worker capacity arithmetic required.
    """
    if wanted_variant is None:
        return []
    for _victim_job, members in sorted_groups:
        if any(m.already_preempted for m in members):
            continue
        if members[0].device_variant != wanted_variant:
            continue
        # Strict band: every sibling must be lower priority than the preemptor.
        if any(m.band_sort_key <= candidate.band for m in members):
            continue
        if len(members) < n_required:
            continue
        pairs = [(candidate.job_name, m.task_id) for m in members]
        for m in members:
            m.already_preempted = True
        return pairs
    return []


def run_preemption_pass(
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
    - Solo (non-coscheduled) preemptors only evict solo victims of the same
      device-variant.
    - Coscheduled preemptors evict an entire victim *slice* (all running tasks
      of one coscheduled job) of the same device-variant and at least the
      preemptor's task count. A non-coscheduled preemptor never tears down a
      slice. Same-variant + slice-shaped guarantees the freed capacity matches
      the request, which avoids large/small thrashing.
    """
    preemptions: list[tuple[JobName, JobName]] = []

    # Solo victims: existing per-worker preemption path (same-variant gated).
    solo_victims = sorted(
        (v for v in running_tasks_info if not v.is_coscheduled),
        key=lambda t: (-t.band_sort_key, t.resource_value),
    )

    # Lazy: only build coscheduled-victim slice index if some preemptor needs
    # one. The common case (no coscheduled preemptors) skips the bucketing.
    sorted_groups: list[tuple[JobName, list[RunningTaskInfo]]] = []
    if any(c.requirements.is_coscheduled for c in unscheduled_tasks):
        grouped: dict[JobName, list[RunningTaskInfo]] = {}
        for v in running_tasks_info:
            if not v.is_coscheduled or v.device_variant is None:
                continue
            vparent = v.task_id.parent
            if vparent is None:
                continue
            grouped.setdefault(vparent, []).append(v)
        sorted_groups = sorted(
            grouped.items(),
            key=lambda kv: (
                -max(t.band_sort_key for t in kv[1]),
                sum(t.resource_value for t in kv[1]),
            ),
        )

    # Preemptor jobs whose siblings have already been satisfied by a slice
    # eviction this pass; the remaining N-1 siblings short-circuit.
    satisfied_preemptor_jobs: set[JobName] = set()
    sibling_count: dict[JobName, int] = defaultdict(int)
    for c in unscheduled_tasks:
        if c.job_name.parent is not None:
            sibling_count[c.job_name.parent] += 1

    for candidate in unscheduled_tasks:
        # Batch never preempts
        if candidate.band >= job_pb2.PRIORITY_BAND_BATCH:
            continue

        parent = candidate.job_name.parent
        if parent is not None and parent in satisfied_preemptor_jobs:
            continue

        wanted_variant = candidate.requirements.device_variant

        if not candidate.requirements.is_coscheduled:
            pair = _preempt_solo(candidate, wanted_variant, solo_victims, context)
            if pair is not None:
                preemptions.append(pair)
            continue

        n_required = sibling_count.get(parent, 1) if parent is not None else 1
        pairs = _preempt_coscheduled(candidate, wanted_variant, n_required, sorted_groups)
        if pairs:
            preemptions.extend(pairs)
            if parent is not None:
                satisfied_preemptor_jobs.add(parent)

    return preemptions


def _job_state_by_id(queries: ControllerDB, job_ids: set[JobName]) -> dict[JobName, int]:
    """Fetch only ``jobs.state`` for the given job IDs.

    Intentionally narrow: only callers that need job state (not resources or
    config) should use this to avoid over-fetching.
    """
    if not job_ids:
        return {}
    with queries.read_snapshot() as tx:
        rows = tx.execute(
            select(jobs_table.c.job_id, jobs_table.c.state).where(
                jobs_table.c.job_id.in_(bindparam("job_ids", expanding=True))
            ),
            {"job_ids": list(job_ids)},
        ).all()
    return {row.job_id: int(row.state) for row in rows}


def _sort_pending_tasks_by_resolved_band(
    pending_tasks: list[PendingTask], requested_bands: dict[JobName, int]
) -> list[PendingTask]:
    """Order pending rows using immutable job_config priority bands."""
    if not pending_tasks:
        return []
    return sorted(
        pending_tasks,
        key=lambda task: (
            requested_bands.get(task.job_id, job_pb2.PRIORITY_BAND_INTERACTIVE),
            task.priority_neg_depth,
            task.priority_root_submitted_ms,
            task.submitted_at_ms.epoch_ms(),
            task.priority_insertion,
        ),
    )


def _worker_matches_reservation_entry(
    worker: SchedulableWorker,
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


def cleanup_stale_claims(
    claims: dict[WorkerId, ReservationClaim],
    db: ControllerDB,
    health: WorkerHealthTracker,
) -> bool:
    """Remove claims for workers that disappeared or jobs that finished.

    Mutates ``claims`` in place; returns whether anything was removed.
    """
    active_worker_ids = {wid for wid, lease in health.all().items() if lease.active}
    claimed_job_ids = {JobName.from_wire(claim.job_id) for claim in claims.values()}
    job_states = _job_state_by_id(db, claimed_job_ids)
    stale: list[WorkerId] = []
    for worker_id, claim in claims.items():
        if worker_id not in active_worker_ids:
            stale.append(worker_id)
            continue
        job_state = job_states.get(JobName.from_wire(claim.job_id))
        if job_state is None or is_job_finished(job_state):
            stale.append(worker_id)
    for wid in stale:
        del claims[wid]
    return bool(stale)


def claim_workers_for_reservations(
    claims: dict[WorkerId, ReservationClaim],
    db: ControllerDB,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsProjection,
) -> bool:
    """Assign unclaimed workers to unsatisfied reservation entries.

    Scans all non-finished jobs with reservations. For each unfulfilled entry,
    finds an eligible unclaimed worker and records the claim. Mutates ``claims``
    in place; returns whether any new claim was added.
    """
    claimed_entries: set[tuple[str, int]] = {(c.job_id, c.entry_idx) for c in claims.values()}
    claimed_worker_ids: set[WorkerId] = set(claims.keys())
    with db.read_snapshot() as tx:
        all_workers = reads.healthy_active_workers_with_attributes(tx, health, worker_attrs)
    changed = False

    reservable_states = (
        job_pb2.JOB_STATE_PENDING,
        job_pb2.JOB_STATE_BUILDING,
        job_pb2.JOB_STATE_RUNNING,
    )
    reservation_jobs = _jobs_with_reservations(db, reservable_states)
    for job in reservation_jobs:
        job_wire = job.job_id.to_wire()
        for idx, res_entry in enumerate(reservation_entries_from_json(job.reservation_json)):
            if (job_wire, idx) in claimed_entries:
                continue

            for worker in all_workers:
                if worker.worker_id in claimed_worker_ids:
                    continue
                if not _worker_matches_reservation_entry(worker, res_entry):
                    continue

                claims[worker.worker_id] = ReservationClaim(job_id=job_wire, entry_idx=idx)
                claimed_worker_ids.add(worker.worker_id)
                claimed_entries.add((job_wire, idx))
                changed = True
                break
    return changed


def refresh_reservation_claims(
    db: ControllerDB,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsProjection,
    *,
    persist: bool = True,
) -> dict[WorkerId, ReservationClaim]:
    """Read, clean up, and re-claim reservation workers; return updated claims.

    Claims are read outside any scheduling transaction. This creates a narrow
    race window where a worker could be removed between the claim read and
    scheduling, but it's benign: ``queue_assignments`` re-validates assignments
    transactionally and stale claims are cleaned up on the next cycle. Pass
    ``persist=False`` to compute the updated claims without writing them
    (dry-run).
    """
    claims = read_reservation_claims(db)
    changed = cleanup_stale_claims(claims, db, health)
    changed = claim_workers_for_reservations(claims, db, health, worker_attrs) or changed
    if changed and persist:
        with db.transaction() as cur:
            writes.replace_reservation_claims(cur, claims)
    return claims


def inject_reservation_taints(
    workers: list[WorkerSnapshot],
    claims: dict[WorkerId, ReservationClaim],
) -> list[WorkerSnapshot]:
    """Create modified worker snapshots with reservation taints and prioritization.

    Claimed workers receive a ``reservation-job`` attribute set to the claiming
    job's ID.  The returned list is ordered with claimed workers first so that
    reservation jobs (which have no NOT_EXISTS constraint) naturally pick from
    their claimed workers before unclaimed ones.

    Snapshots are never mutated — ``dataclasses.replace`` produces shallow copies.
    """
    if not claims:
        return workers

    claimed: list[WorkerSnapshot] = []
    unclaimed: list[WorkerSnapshot] = []
    for worker in workers:
        claim = claims.get(worker.worker_id)
        if claim is not None:
            modified_attrs = dict(worker.attributes)
            modified_attrs[RESERVATION_TAINT_KEY] = AttributeValue(claim.job_id)
            claimed.append(replace(worker, attributes=modified_attrs))
        else:
            unclaimed.append(worker)
    return claimed + unclaimed


def inject_taint_constraints(
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


def _reserved_job_ids(queries: ControllerDB) -> set[JobName]:
    """Return the set of job_ids with ``has_reservation = 1`` on the jobs table.

    Callers use this to drive :func:`_find_reservation_ancestor` purely in
    Python instead of issuing one SQL chain-walk per pending job.
    """
    with queries.read_snapshot() as tx:
        rows = tx.execute(select(jobs_table.c.job_id).where(jobs_table.c.has_reservation == 1)).all()
    return {row.job_id for row in rows}


def _find_reservation_ancestor(reserved_jobs: set[JobName], job_id: JobName) -> JobName | None:
    """Walk up the job hierarchy to find the nearest ancestor with a reservation.

    Pure Python walk against the pre-fetched ``reserved_jobs`` set. The old
    SQL-per-call form opened a fresh ``read_snapshot`` and issued 1-3 round
    trips per unique pending job, which dominated ``compute_demand_entries``
    once the SA Core machinery became the per-call floor.
    """
    current = job_id.parent
    while current is not None:
        if current in reserved_jobs:
            return current
        current = current.parent
    return None


def preference_pass(
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

        # Enforce the task's hard placement constraints against the claimed
        # worker, exactly as the normal scheduler does (compute_candidates).
        # ``can_fit`` checks resource counts only — not device type/variant,
        # region, or zone — so without this gate a reservation task could be
        # placed on a claimed worker that violates its own constraints and then
        # be stripped from ``pending_tasks`` before find_assignments can correct
        # it. ``split_hard_soft`` keeps soft constraints as preferences, not
        # filters. For direct-reservation jobs the injected EQ taint is itself a
        # hard constraint and is satisfied by the claimed worker's taint
        # attribute, so this does not reject a worker for its own claim.
        hard_constraints, _ = split_hard_soft(list(req.constraints))

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
            if not capacity.matches_constraints(hard_constraints):
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


def build_scheduling_context(
    queries: ControllerDB,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsProjection,
    defaults: UserBudgetDefaults,
    max_building_tasks: int = DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
) -> SchedulingContext:
    """Build a ``SchedulingContext`` from a single read snapshot.

    All scheduling-tick DB I/O lives here. The returned context carries
    un-tainted workers; ``_run_scheduler_pass`` applies reservation taints
    for the assignment pass only.
    """
    with slow_log(logger, "scheduling tick context", threshold_ms=50):
        with queries.read_snapshot() as snap:
            pending = _pending_tasks_with_jobs(snap)
            workers = reads.healthy_active_workers_with_attributes(snap, health, worker_attrs)
            usage_by_worker = reads.resource_usage_by_worker(snap)
            user_spend = compute_user_spend(snap)
            user_budget_limits = reads.get_all_user_budget_limits(snap)
            requested_bands = reads.get_priority_bands(snap, {t.job_id for t in pending})
            reserved_jobs = {
                row.job_id
                for row in snap.execute(select(jobs_table.c.job_id).where(jobs_table.c.has_reservation == 1)).all()
            }
            reservation_entry_counts = _reservation_entry_counts_for_pending(snap, pending)
            building_counts = reads.building_counts(snap, [w.worker_id for w in workers])

    snapshots = [worker_snapshot_from_row(w, usage_by_worker.get(w.worker_id)) for w in workers]
    sorted_pending = _sort_pending_tasks_by_resolved_band(pending, requested_bands)
    return SchedulingContext(
        workers=snapshots,
        building_counts=building_counts,
        max_building_tasks=max_building_tasks,
        max_assignments_per_worker=DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
        pending_tasks=[],
        jobs={},
        pending_task_rows=sorted_pending,
        user_spend=user_spend,
        user_budget_limits=user_budget_limits,
        requested_bands=requested_bands,
        reserved_job_ids=frozenset(reserved_jobs),
        reservation_entry_counts=reservation_entry_counts,
        user_budget_defaults=defaults,
    )


def _reservation_entry_counts_for_pending(tx: Tx, pending: list[PendingTask]) -> dict[JobName, int]:
    """Return reservation entry counts for pending jobs that hold a reservation."""
    job_ids = {t.job_id for t in pending if t.has_reservation}
    if not job_ids:
        return {}
    rows = tx.execute(
        select(job_config_table.c.job_id, job_config_table.c.reservation_json).where(
            job_config_table.c.job_id.in_(bindparam("job_ids", expanding=True))
        ),
        {"job_ids": list(job_ids)},
    ).all()
    counts: dict[JobName, int] = {}
    for row in rows:
        if row.reservation_json is None:
            continue
        counts[row.job_id] = len(reservation_entries_from_json(row.reservation_json))
    return counts


def apply_scheduling_gates(
    ctx: SchedulingContext,
    claims: dict[WorkerId, ReservationClaim],
    *,
    max_tasks_per_job_per_cycle: int,
    trace: bool = False,
) -> GatedCandidates:
    """Filter ``ctx.pending_task_rows`` by deadline, reservation, and per-job cap.

    Expired tasks are returned in ``GatedCandidates.expired_tasks`` for the
    caller to mark UNSCHEDULABLE; this function does no DB writes.
    """
    schedulable_task_ids: list[JobName] = []
    expired_tasks: list[PendingTask] = []
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    has_direct_reservation: set[JobName] = set()
    tasks_per_job: dict[JobName, int] = defaultdict(int)
    filter_counts: dict[str, int] = defaultdict(int)

    # Index claims by wire id so reservation-satisfaction is O(1) per check.
    claims_by_job: dict[str, int] = defaultdict(int)
    for claim in claims.values():
        claims_by_job[claim.job_id] += 1

    for task in ctx.pending_task_rows:
        if not task_row_can_be_scheduled(task):
            filter_counts["task_not_schedulable"] += 1
            continue
        deadline = job_scheduling_deadline(task.scheduling_deadline_epoch_ms)
        if deadline is not None and deadline.expired():
            filter_counts["deadline_expired"] += 1
            expired_tasks.append(task)
            continue
        # Gate: skip real tasks whose job has an unsatisfied reservation.
        # Holder tasks are always schedulable (they ARE the reservation).
        if not task.is_reservation_holder and task.has_reservation:
            wire_id = task.job_id.to_wire()
            required = ctx.reservation_entry_counts.get(task.job_id, 0)
            if claims_by_job.get(wire_id, 0) < required:
                filter_counts["reservation_unsatisfied"] += 1
                continue
        if (
            max_tasks_per_job_per_cycle > 0
            and not task.has_coscheduling
            and tasks_per_job[task.job_id] >= max_tasks_per_job_per_cycle
        ):
            filter_counts["per_job_cap"] += 1
            continue
        tasks_per_job[task.job_id] += 1
        schedulable_task_ids.append(task.task_id)
        if task.job_id not in jobs:
            jobs[task.job_id] = job_requirements_from_job(task)
            if task.has_reservation:
                has_reservation.add(task.job_id)
                has_direct_reservation.add(task.job_id)
            elif _find_reservation_ancestor(set(ctx.reserved_job_ids), task.job_id) is not None:
                has_reservation.add(task.job_id)
    if trace:
        logger.info(
            "[TRACE] Phase 2 gates: %d/%d tasks passed, %d distinct jobs; filtered: %s",
            len(schedulable_task_ids),
            len(ctx.pending_task_rows),
            len(jobs),
            dict(filter_counts),
        )
    return GatedCandidates(
        schedulable_task_ids=schedulable_task_ids,
        jobs=jobs,
        has_reservation=has_reservation,
        has_direct_reservation=has_direct_reservation,
        expired_tasks=expired_tasks,
    )


def compute_scheduling_order(
    ctx: SchedulingContext,
    gated: GatedCandidates,
    *,
    trace: bool = False,
) -> SchedulingOrder:
    """Compute priority-band interleaving order from gated candidates.

    Uses ``ctx.requested_bands`` (sourced from immutable job_config), not
    ``tasks.priority_band`` (overwritten at assign time), so a task demoted
    to BATCH while its user was over budget can recover after preemption.
    """
    user_spend = ctx.user_spend
    requested_bands = ctx.requested_bands
    user_budget_limits = ctx.user_budget_limits
    defaults = ctx.user_budget_defaults
    task_band_map: dict[JobName, int] = {
        task.task_id: compute_effective_band(
            requested_bands.get(task.job_id, task.priority_band),
            task.task_id.user,
            user_spend,
            user_budget_limits,
            defaults,
        )
        for task in ctx.pending_task_rows
    }
    tasks_by_band: dict[int, list[JobName]] = defaultdict(list)
    for task_id in gated.schedulable_task_ids:
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
    return SchedulingOrder(
        ordered_task_ids=interleaved,
        task_band_map=task_band_map,
        user_spend=user_spend,
        user_budget_limits=user_budget_limits,
    )
