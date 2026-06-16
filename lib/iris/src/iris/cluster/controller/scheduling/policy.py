# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduling and preemption policy.

Free functions parameterized by their dependencies (``ControllerDB``,
``WorkerHealthTracker``, ``SchedulingContext``) rather than the ``Controller``
instance. DB I/O is concentrated in the context builders; the gate, ordering,
and preemption passes are pure transforms over an in-memory ``SchedulingContext``.
"""

import logging
import sys
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace

from rigging.log_setup import slow_log

from iris.cluster.constraints import (
    AVAILABILITY_PREFIX,
    AttributeValue,
    PlacementRequirements,
    WellKnownAttribute,
    availability_key,
    extract_placement_requirements,
    is_availability_key,
    split_hard_soft,
)
from iris.cluster.controller import reads
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
    resource_spec_from_scalars,
)
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.reads import WorkerAttrsSource
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
from iris.cluster.controller.task_state import job_scheduling_deadline, task_row_can_be_scheduled
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import (
    JobName,
    PendingTask,
    UserBudgetDefaults,
    is_job_finished,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# Sentinel for dry-run scheduling with per-worker limits disabled.
_UNLIMITED = sys.maxsize


@dataclass(frozen=True)
class PreemptionCandidate:
    """An unscheduled task that may preempt running work."""

    job_name: JobName
    requirements: JobRequirements
    band: int  # proto PriorityBand value


@dataclass(frozen=True)
class GatedCandidates:
    """Tasks that passed deadline and per-job-cap gates."""

    schedulable_task_ids: list[JobName]
    jobs: dict[JobName, JobRequirements]
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


def enrich_workers_with_availability(
    workers: list[WorkerSnapshot],
    zone_capabilities: Mapping[str, frozenset[str]],
) -> list[WorkerSnapshot]:
    """Add each worker's zone ``availability:<variant>`` markers to its attributes.

    A worker inherits the accelerator variants its *zone* can provision (inferred
    from the autoscaler). A hard ``availability:<variant>`` job constraint then
    filters placement to workers in a capable zone, so a CPU orchestrator is placed
    only in the zone where its accelerator can be found (and waits otherwise). Keys
    on the existing ``zone`` attribute (the same one ``--zone``/``--region``
    matching uses); workers without a zone are passed through unchanged.
    """
    if not zone_capabilities:
        return workers
    enriched: list[WorkerSnapshot] = []
    for worker in workers:
        zone_attr = worker.attributes.get(WellKnownAttribute.ZONE)
        variants = zone_capabilities.get(str(zone_attr.value)) if zone_attr is not None else None
        if not variants:
            enriched.append(worker)
            continue
        attrs = dict(worker.attributes)
        for variant in variants:
            attrs[availability_key(variant)] = AttributeValue("true")
        enriched.append(replace(worker, attributes=attrs))
    return enriched


def demanded_availability_variants(pending_task_rows: list[PendingTask]) -> set[str]:
    """Accelerator variants some pending task constrains on via ``availability:<variant>``.

    Only these variants need injecting onto workers this tick: an availability
    marker matters solely when a pending job filters on it, and in practice a tiny
    set of variants (often just one, e.g. ``v5p-8``) is ever reserved. Restricting
    enrichment to this set confines the per-worker copy to the few workers in a
    zone that provisions a demanded variant. A substring pre-check skips JSON
    parsing for the common task that carries no availability constraint at all.

    Returned variants are lowercased to match the ``zone_capabilities`` map (both
    sides come through :func:`availability_key`).
    """
    variants: set[str] = set()
    for task in pending_task_rows:
        constraints_json = task.constraints_json
        if not constraints_json or AVAILABILITY_PREFIX not in constraints_json:
            continue
        for constraint in constraints_from_json(constraints_json):
            if is_availability_key(constraint.key):
                variants.add(constraint.key[len(AVAILABILITY_PREFIX) :])
    return variants


def compute_demand_entries(
    ctx: SchedulingContext,
    scheduler: Scheduler,
    exclude_task_ids: AbstractSet[JobName] = frozenset(),
) -> list[DemandEntry]:
    """Compute the autoscaler demand entries for a scheduling snapshot.

    Returns the unmet demand the autoscaler must provision: every pending task
    the current fleet cannot already absorb, grouped into ``DemandEntry`` records.

    A limits-free capacity-fit dry-run (per-worker building and assignment caps
    disabled) marks which tasks the existing fleet could already absorb; only the
    unabsorbed remainder emits demand. This is deliberate work distinct from the
    (limits-on) assignment pass: a task blocked only by the per-cycle assignment
    cap has capacity waiting for it and must not signal demand.

    Each entry carries the job's constraints, so a hard ``availability:<variant>``
    constraint rides along to the autoscaler, which restricts scaling groups to the
    zones that expose that capability (see ``route_demand``).

    Args:
        ctx: The per-tick scheduling context (workers, pending task rows,
            building counts).
        scheduler: The ``Scheduler`` used for the capacity-fit dry-run pass.
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

    # Build job requirements once, shared between dry-run and demand emission.
    jobs: dict[JobName, JobRequirements] = {}
    for task in pending:
        if task.job_id in jobs:
            continue
        jobs[task.job_id] = job_requirements_from_job(task)

    # Dry-run scheduling with building/assignment limits disabled: mark which
    # tasks the existing fleet can already absorb so only the remainder emits
    # demand. Building/assignment limits are disabled so big workers can absorb
    # multiple tasks (prevents false demand on idle clusters).
    absorbed_task_ids: set[JobName] = set()
    if ctx.workers:
        # Dry-run scheduling context — only the per-(task, worker) matching loop
        # consumes capacities/jobs/pending_tasks, so the raw-read fields stay
        # empty.
        dry_run_context = SchedulingContext(
            workers=ctx.workers,
            building_counts=ctx.building_counts,
            max_building_tasks=_UNLIMITED,
            max_assignments_per_worker=_UNLIMITED,
            pending_tasks=[t.task_id for t in pending],
            jobs=jobs,
            pending_task_rows=[],
            user_spend={},
            user_budget_limits={},
            requested_bands={},
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


def get_running_tasks_with_band_and_value(
    db: ControllerDB,
) -> list[RunningTaskInfo]:
    """Query running tasks with band, worker, resource spec, and coscheduling status.

    The reported band is the value persisted in ``tasks.priority_band``, which
    is stamped at assignment time (see ``_commit_assignments`` and
    ``writes.tasks.assign_task``). The over-budget downgrade is applied at that
    stamping point, not on every scheduling tick, which prevents a running
    task from oscillating into BATCH and back as its own user crosses the
    budget cliff — the source of mutual same-band preemption between two
    users sitting at their limits.
    """
    with db.read_snapshot() as tx:
        return _running_tasks_with_band_and_value(tx)


def _running_tasks_with_band_and_value(tx: Tx) -> list[RunningTaskInfo]:
    """Map the running-task band/resource rows into :class:`RunningTaskInfo`."""
    rows = reads.running_task_band_rows(tx)
    result: list[RunningTaskInfo] = []
    for row in rows:
        wid = row.worker_id
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


def build_scheduling_context(
    snap: Tx,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsSource,
    defaults: UserBudgetDefaults,
    max_building_tasks: int = DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
) -> SchedulingContext:
    """Build a ``SchedulingContext`` from the caller's read snapshot ``snap``.

    All scheduling-tick DB reads live here. Every read shares the caller's
    snapshot, so the control tick issues a single DB read for the whole tick.
    """
    with slow_log(logger, "scheduling tick context", threshold_ms=50):
        pending = reads.pending_tasks_with_jobs(snap)
        workers = reads.healthy_active_workers_with_attributes(snap, health, worker_attrs)
        usage_by_worker = reads.resource_usage_by_worker(snap)
        user_spend = compute_user_spend(snap)
        user_budget_limits = reads.get_all_user_budget_limits(snap)
        requested_bands = reads.get_priority_bands(snap, {t.job_id for t in pending})
        building_counts = reads.building_counts(snap, [w.worker_id for w in workers])
        running = _running_tasks_with_band_and_value(snap)

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
        user_budget_defaults=defaults,
        running_for_preemption=running,
    )


def apply_scheduling_gates(
    ctx: SchedulingContext,
    *,
    max_tasks_per_job_per_cycle: int,
    trace: bool = False,
) -> GatedCandidates:
    """Filter ``ctx.pending_task_rows`` by deadline and per-job cap.

    Expired tasks are returned in ``GatedCandidates.expired_tasks`` for the
    caller to mark UNSCHEDULABLE; this function does no DB writes.
    """
    schedulable_task_ids: list[JobName] = []
    expired_tasks: list[PendingTask] = []
    jobs: dict[JobName, JobRequirements] = {}
    tasks_per_job: dict[JobName, int] = defaultdict(int)
    filter_counts: dict[str, int] = defaultdict(int)

    for task in ctx.pending_task_rows:
        if not task_row_can_be_scheduled(task):
            filter_counts["task_not_schedulable"] += 1
            continue
        deadline = job_scheduling_deadline(task.scheduling_deadline_epoch_ms)
        if deadline is not None and deadline.expired():
            filter_counts["deadline_expired"] += 1
            expired_tasks.append(task)
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
