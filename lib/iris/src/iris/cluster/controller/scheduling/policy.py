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
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace

from rigging.log_setup import slow_log

from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintOp,
    DeviceType,
    PlacementRequirements,
    constraints_from_resources,
    evaluate_constraint,
    extract_placement_requirements,
    merge_constraints,
    split_hard_soft,
)
from iris.cluster.controller import reads, writes
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
from iris.cluster.controller.reads import SchedulableWorker, WorkerAttrsSource
from iris.cluster.controller.reconcile.policy import RESERVATION_HOLDER_JOB_NAME
from iris.cluster.controller.scheduling.scheduler import (
    DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
    DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
    JobRequirements,
    RunningTaskInfo,
    Scheduler,
    SchedulingContext,
    WorkerCapacity,
    WorkerSnapshot,
    worker_snapshot_from_row,
)
from iris.cluster.controller.schema import ReservationClaim
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

# Reservation placements advance one claimed worker per cycle, independent of
# the (higher) non-reservation packing cap. Each claim is one reserved slot, so
# packing several reservation tasks onto the first claimed worker would anchor
# reserved capacity on one worker while the other claimed workers sit tainted
# but unused.
_MAX_RESERVATION_PLACEMENTS_PER_WORKER_PER_CYCLE = 1


# Taint attribute injected onto claimed workers to prevent non-reservation
# jobs from landing on them.  Non-reservation jobs get a NOT_EXISTS constraint
# for this key; reservation jobs do not, so they naturally prefer claimed
# workers (which appear first in the worker list).
RESERVATION_TAINT_KEY = "reservation-job"


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


def reservation_unsatisfied(
    task: PendingTask,
    claims_by_job: dict[str, int],
    reservation_entry_counts: dict[JobName, int],
    colocating_reservation_job_ids: AbstractSet[JobName],
) -> bool:
    """Whether a real task must wait for its job's reservation to be claimed.

    A non-holder task of a *co-locating* directly-reserved job is *unsatisfied*
    until the number of workers claimed for its reservation reaches the
    reservation's entry count. Such a task must neither be scheduled nor
    generate autoscaler demand: the reservation's holder task provisions the
    reserved capacity, and the real task runs only once that capacity exists.

    A directly-reserved job that does *not* co-locate (its own task targets a
    different device class than the reservation — e.g. a CPU orchestrator that
    reserves a TPU) is never gated here: it does not run on the reserved
    workers, so it schedules on its own resources independent of the
    reservation. Gating it would strand it whenever the reservation cannot be
    claimed (device stockout), and EQ-pinning it (see
    :func:`inject_taint_constraints`) would route phantom CPU demand the
    scheduler can never satisfy (the 2026-06-08 canary thrash).

    Both the scheduling gate (:func:`apply_scheduling_gates`) and the demand
    path (:func:`compute_demand_entries`) consult this predicate so they agree.
    If they disagree — demand emitting for a task the gate blocks — the
    autoscaler provisions generic capacity the task can never occupy and
    thrashes, booting workers that are reaped as idle every cycle.
    """
    if task.is_reservation_holder or not task.has_reservation:
        return False
    if task.job_id not in colocating_reservation_job_ids:
        return False
    required = reservation_entry_counts.get(task.job_id, 0)
    return claims_by_job.get(task.job_id.to_wire(), 0) < required


def _claims_by_job(claims: dict[WorkerId, ReservationClaim] | None) -> dict[str, int]:
    """Count reservation claims per claiming job (wire id) for the reservation gate."""
    counts: dict[str, int] = defaultdict(int)
    for claim in (claims or {}).values():
        counts[claim.job_id] += 1
    return counts


def _own_device_type(res_device_json: str | None) -> DeviceType:
    """Device class a task's *own* resource request targets (CPU when no accelerator)."""
    dc = device_counts_from_json(res_device_json)
    if dc.gpu > 0:
        return DeviceType.GPU
    if dc.tpu > 0:
        return DeviceType.TPU
    return DeviceType.CPU


def _colocating_reservation_job_ids(
    pending: list[PendingTask],
    entry_device_types: dict[JobName, frozenset[DeviceType]],
) -> frozenset[JobName]:
    """Directly-reserved jobs whose own task co-locates on their reserved workers.

    A ``--reserve`` job co-locates only when its own resource request targets the
    same device class the reservation reserves (e.g. ``--reserve v5p-8`` with a
    v5p-8 task). A reservation whose own task is CPU-only while it reserves an
    accelerator (the common orchestrator/``dev_tpu`` pattern) does NOT co-locate:
    the reserved workers exist for its holder and descendants, and the parent
    task itself schedules on a worker matching its own request. Only co-locating
    jobs are EQ-pinned to and gated on their claimed workers; see
    :func:`inject_taint_constraints` and :func:`reservation_unsatisfied`.
    """
    result: set[JobName] = set()
    for task in pending:
        if not task.has_reservation:
            continue
        entries = entry_device_types.get(task.job_id)
        if not entries:
            continue
        if _own_device_type(task.res_device_json) in entries:
            result.add(task.job_id)
    return frozenset(result)


def compute_demand_entries(
    ctx: SchedulingContext,
    scheduler: Scheduler,
    claims: dict[WorkerId, ReservationClaim],
    exclude_task_ids: AbstractSet[JobName] = frozenset(),
) -> list[DemandEntry]:
    """Compute the autoscaler demand entries for a scheduling snapshot.

    Returns the unmet demand the autoscaler must provision: every pending task
    the current fleet cannot already absorb, grouped into ``DemandEntry`` records.

    All pending tasks — both real and reservation holder — flow through a single
    unified path. A limits-free capacity-fit dry-run (per-worker building and
    assignment caps disabled) marks which tasks the existing fleet could already
    absorb; only the unabsorbed remainder emits demand. This is deliberate work
    distinct from the (limits-on) assignment pass: a task blocked only by the
    per-cycle assignment cap has capacity waiting for it and must not signal
    demand.

    Holder tasks request the reserved resources, so spare capacity does not
    absorb them and they keep reserved capacity alive via the autoscaler. The
    taint/constraint mechanism ensures only peer jobs can use the reserved
    workers.

    .. note::

        Demand from holder tasks and parent real tasks is additive. On a cold
        start with N reservation entries and M real tasks this reports N + M
        demand entries, which may overprovision. In practice reservations are
        used when the parent job does not request its own resources, so the
        additive behavior is correct. If that changes, a dedup path (e.g.
        ``max(real_pending, holders)``) should be added here.

    Args:
        ctx: The per-tick scheduling context (workers, pending task rows,
            reservation entry counts, reserved job ids, building counts).
        scheduler: The ``Scheduler`` used for the capacity-fit dry-run pass.
        claims: Reservation claims, applied as taint injection in the dry-run to
            match the real scheduling path.
        exclude_task_ids: Pending tasks to drop from both the dry-run and the
            emitted demand — used to skip tasks the caller is retiring this tick
            (e.g. deadline-expired tasks the scheduler marks UNSCHEDULABLE), so
            the autoscaler is never asked to provision for a failing job.
    """
    demand_entries: list[DemandEntry] = []

    # ``ctx.pending_task_rows`` is pre-sorted by *resolved band* for the
    # assignment pass; the demand residual is band-independent, and keying it on
    # band order could change which task ids fall out as residual under capacity
    # contention. Sort on the underlying jobs-query ORDER BY keys to keep the
    # residual stable.
    pending = sorted(
        (t for t in ctx.pending_task_rows if t.task_id not in exclude_task_ids),
        key=lambda t: (
            t.priority_neg_depth,
            t.priority_root_submitted_ms,
            t.submitted_at_ms.epoch_ms(),
            t.priority_insertion,
        ),
    )

    tasks_by_job: dict[JobName, list[PendingTask]] = defaultdict(list)
    for task in pending:
        tasks_by_job[task.job_id].append(task)

    reservation_entry_counts = ctx.reservation_entry_counts

    # Index reservation claims by job so the reservation gate below can compare
    # claimed workers against required entries, exactly as apply_scheduling_gates
    # does for the real scheduling pass.
    claims_by_job = _claims_by_job(claims)

    # Build job requirements once, shared between dry-run and demand emission.
    # Also track which jobs have reservations so we can apply taint injection.
    reserved_jobs = ctx.reserved_job_ids
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    has_direct_reservation: set[JobName] = set()
    colocating = ctx.colocating_reservation_job_ids
    for task in pending:
        if task.job_id in jobs:
            continue
        jobs[task.job_id] = job_requirements_from_job(task)
        if task.has_reservation and task.job_id in colocating:
            has_reservation.add(task.job_id)
            has_direct_reservation.add(task.job_id)
        elif _find_reservation_ancestor(reserved_jobs, task.job_id) is not None:
            has_reservation.add(task.job_id)

    # Dry-run scheduling with building/assignment limits disabled.
    # All tasks participate — holders and real tasks alike.
    absorbed_task_ids: set[JobName] = set()
    if ctx.workers:
        dry_run_workers = inject_reservation_taints(ctx.workers, claims)
        dry_run_jobs = inject_taint_constraints(jobs, has_reservation, has_direct_reservation)

        # Dry-run scheduling context — only the per-(task, worker) matching loop
        # consumes capacities/jobs/pending_tasks, so the raw-read fields stay
        # empty. Building/assignment limits are disabled so big workers can
        # absorb multiple tasks (prevents false demand on idle clusters).
        dry_run_context = SchedulingContext(
            workers=dry_run_workers,
            building_counts=ctx.building_counts,
            max_building_tasks=_UNLIMITED,
            max_assignments_per_worker=_UNLIMITED,
            pending_tasks=[t.task_id for t in pending],
            jobs=dry_run_jobs,
            pending_task_rows=[],
            user_spend={},
            user_budget_limits={},
            requested_bands={},
            reserved_job_ids=frozenset(),
            reservation_entry_counts={},
            user_budget_defaults=UserBudgetDefaults(),
        )
        result = scheduler.find_assignments(dry_run_context)
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

        # Reservation gate (mirrors apply_scheduling_gates): a real task of a
        # directly-reserved job emits no demand until its reservation is
        # claimed. The holder task (a separate job) provisions the reserved
        # capacity; emitting demand for the real task here would provision
        # generic capacity the scheduler gate will never let it occupy, so the
        # autoscaler would boot workers that idle out and get reaped forever.
        if reservation_unsatisfied(job_row, claims_by_job, reservation_entry_counts, ctx.colocating_reservation_job_ids):
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
                        task_ids=tuple(remaining_ids),
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
                    task_ids=(task.task_id.to_wire(),),
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
        return _running_tasks_with_band_and_value(tx, claimed_workers)


def _running_tasks_with_band_and_value(tx: Tx, claimed_workers: set[WorkerId]) -> list[RunningTaskInfo]:
    """Map the running-task band/resource rows into :class:`RunningTaskInfo`.

    Skips tasks on reservation-claimed workers, since those workers are spoken
    for and must not be considered as preemption victims.
    """
    rows = reads.running_task_band_rows(tx)
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
    # Gate on hard constraints only: soft constraints are placement preferences,
    # not requirements, so an unmet one must not veto an otherwise-valid eviction.
    hard_constraints, _ = split_hard_soft(req.constraints)
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
        if not cap.matches_constraints(hard_constraints):
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
    context: SchedulingContext,
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
    # Gate on hard constraints (mirrors _preempt_solo): the device variant alone
    # does not prove the preemptor can land on the freed slice — region/zone/
    # attribute constraints must hold on every member's worker too. Without this
    # we evict a slice the preemptor can never be placed on, wasting the eviction
    # and risking a preempt/reject thrash loop.
    hard_constraints, _ = split_hard_soft(candidate.requirements.constraints)
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
        if any(
            (cap := context.capacities.get(m.worker_id)) is None or not cap.matches_constraints(hard_constraints)
            for m in members
        ):
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
        pairs = _preempt_coscheduled(candidate, wanted_variant, n_required, sorted_groups, context)
        if pairs:
            preemptions.extend(pairs)
            if parent is not None:
                satisfied_preemptor_jobs.add(parent)

    return preemptions


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
    tx: Tx,
    health: WorkerHealthTracker,
) -> bool:
    """Remove claims for workers that disappeared or jobs that finished.

    Reads job state from the caller's snapshot ``tx``. Mutates ``claims`` in
    place; returns whether anything was removed.
    """
    active_worker_ids = {wid for wid, lease in health.all().items() if lease.active}
    claimed_job_ids = {JobName.from_wire(claim.job_id) for claim in claims.values()}
    job_states = reads.job_states_by_id(tx, claimed_job_ids)
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
    tx: Tx,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsProjection,
) -> bool:
    """Assign unclaimed workers to unsatisfied reservation entries.

    Scans all non-finished jobs with reservations from the caller's snapshot
    ``tx``. For each unfulfilled entry, finds an eligible unclaimed worker and
    records the claim. Mutates ``claims`` in place; returns whether any new
    claim was added.
    """
    claimed_entries: set[tuple[str, int]] = {(c.job_id, c.entry_idx) for c in claims.values()}
    claimed_worker_ids: set[WorkerId] = set(claims.keys())
    reservable_states = (
        job_pb2.JOB_STATE_PENDING,
        job_pb2.JOB_STATE_BUILDING,
        job_pb2.JOB_STATE_RUNNING,
    )
    all_workers = reads.healthy_active_workers_with_attributes(tx, health, worker_attrs)
    reservation_jobs = reads.jobs_with_reservations(tx, reservable_states)
    changed = False

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
    scheduling, but it's benign: ``assign`` re-validates assignments
    transactionally and stale claims are cleaned up on the next cycle. Pass
    ``persist=False`` to compute the updated claims without writing them
    (dry-run).
    """
    with db.read_snapshot() as tx:
        claims, changed = refresh_reservation_claims_in_tx(tx, health, worker_attrs)
    if changed and persist:
        with db.transaction() as cur:
            writes.replace_reservation_claims(cur, claims)
    return claims


def refresh_reservation_claims_in_tx(
    tx: Tx,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsProjection,
) -> tuple[dict[WorkerId, ReservationClaim], bool]:
    """Read claims and run the stale-sweep + re-claim passes over ``tx``.

    The DB-less core shared by :func:`refresh_reservation_claims` and the control
    tick's snapshot build. Returns the updated claims plus whether anything
    changed, so the caller persists them in its own write transaction.
    """
    claims = reads.list_claims(tx)
    changed = cleanup_stale_claims(claims, tx, health)
    changed = claim_workers_for_reservations(claims, tx, health, worker_attrs) or changed
    return claims, changed


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


def _find_reservation_ancestor(reserved_jobs: AbstractSet[JobName], job_id: JobName) -> JobName | None:
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
            if context.assignment_counts.get(wid, 0) >= _MAX_RESERVATION_PLACEMENTS_PER_WORKER_PER_CYCLE:
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
    snap: Tx,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsSource,
    defaults: UserBudgetDefaults,
    claims: dict[WorkerId, ReservationClaim],
    max_building_tasks: int = DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
) -> SchedulingContext:
    """Build a ``SchedulingContext`` from the caller's read snapshot ``snap``.

    All scheduling-tick DB reads live here. The returned context carries
    un-tainted workers; reservation taints are applied during the assignment
    pass only (in ``run_scheduling_decision``). Every read shares the caller's
    snapshot, so the control tick issues a single DB read for the whole tick.
    """
    with slow_log(logger, "scheduling tick context", threshold_ms=50):
        pending = reads.pending_tasks_with_jobs(snap)
        workers = reads.healthy_active_workers_with_attributes(snap, health, worker_attrs)
        usage_by_worker = reads.resource_usage_by_worker(snap)
        user_spend = compute_user_spend(snap)
        user_budget_limits = reads.get_all_user_budget_limits(snap)
        requested_bands = reads.get_priority_bands(snap, {t.job_id for t in pending})
        reserved_jobs = reads.reserved_job_ids(snap)
        reserved_pending_ids = {t.job_id for t in pending if t.has_reservation}
        reservation_entry_counts = reads.reservation_entry_counts(snap, reserved_pending_ids)
        entry_device_types = reads.reservation_entry_device_types(snap, reserved_pending_ids)
        building_counts = reads.building_counts(snap, [w.worker_id for w in workers])
        running = _running_tasks_with_band_and_value(snap, set(claims.keys()))

    snapshots = [worker_snapshot_from_row(w, usage_by_worker.get(w.worker_id)) for w in workers]
    sorted_pending = _sort_pending_tasks_by_resolved_band(pending, requested_bands)
    colocating_reservations = _colocating_reservation_job_ids(pending, entry_device_types)
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
        running_for_preemption=running,
        colocating_reservation_job_ids=colocating_reservations,
    )


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
    claims_by_job = _claims_by_job(claims)

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
        if reservation_unsatisfied(
            task, claims_by_job, ctx.reservation_entry_counts, ctx.colocating_reservation_job_ids
        ):
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
            if task.has_reservation and task.job_id in ctx.colocating_reservation_job_ids:
                has_reservation.add(task.job_id)
                has_direct_reservation.add(task.job_id)
            elif _find_reservation_ancestor(ctx.reserved_job_ids, task.job_id) is not None:
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
