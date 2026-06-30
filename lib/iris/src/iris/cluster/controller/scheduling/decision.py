# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduling-decision passes shared by the Iris scheduling pipeline.

Pure scheduling-layer helpers: they consume only ``scheduling.policy`` /
``scheduling.scheduler`` types (plus protos) and return plain data, so they
carry no edge back to the controller's ops/reconcile/schema layer.
"""

from iris.cluster.controller.autoscaler.reserved_pool import ReservationLedger
from iris.cluster.controller.scheduling.policy import (
    PreemptionCandidate,
    SchedulingOrder,
    run_preemption_pass,
    run_reserved_pool_preemption,
)
from iris.cluster.controller.scheduling.scheduler import (
    JobRequirements,
    RunningTaskInfo,
    Scheduler,
    SchedulingContext,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2


def apply_preemptions(
    order: SchedulingOrder,
    jobs: dict[JobName, JobRequirements],
    all_assignments: list[tuple[JobName, WorkerId]],
    running_for_preemption: list[RunningTaskInfo],
    context: SchedulingContext,
    ledger: ReservationLedger | None = None,
) -> tuple[list[tuple[JobName, JobName]], set[WorkerId]]:
    """Decide which running tasks to evict for higher-priority unscheduled work.

    Runs the same-variant pass first, then a cross-variant pass over fungible
    reserved pools for the preemptors the same-variant pass could not satisfy.
    Returns the combined ``(preemptor, victim)`` pairs and the set of worker ids
    whose slices the autoscaler must drain (empty for same-variant evictions,
    which the worker-failure teardown already handles).
    """
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
    if not unscheduled:
        return [], set()

    pairs = run_preemption_pass(unscheduled, running_for_preemption, context)
    if ledger is None or ledger.is_empty():
        return pairs, set()

    # Only preemptors the same-variant pass did not already satisfy fall through
    # to the cross-variant reserved pass (the freed slot otherwise double-counts).
    # Key on the preemptor *job* (parent), matching the reserved pass's own dedup,
    # so a coscheduled preemptor satisfied via one sibling excludes all siblings.
    satisfied_jobs = {preemptor.parent or preemptor for preemptor, _ in pairs}
    remaining = [c for c in unscheduled if (c.job_name.parent or c.job_name) not in satisfied_jobs]
    reserved_pairs, drain_workers = run_reserved_pool_preemption(remaining, running_for_preemption, ledger)
    return pairs + reserved_pairs, drain_workers


def compute_diagnostics(
    scheduler: Scheduler,
    context: SchedulingContext,
    jobs: dict[JobName, JobRequirements],
    assignments: list[tuple[JobName, WorkerId]],
    schedulable_task_ids: list[JobName],
) -> dict[str, str]:
    """Compute per-job scheduling diagnostics for unassigned jobs."""
    assigned_task_ids = {task_id for task_id, _ in assignments}

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
        diagnostics[job_id.to_wire()] = scheduler.get_job_scheduling_diagnostics(
            req,
            context,
            representative_task,
            num_tasks=num_tasks,
        )
    return diagnostics
