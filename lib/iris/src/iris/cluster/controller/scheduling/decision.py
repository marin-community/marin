# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduling-decision passes shared by the Iris scheduling pipeline.

Pure scheduling-layer helpers: they consume only ``scheduling.policy`` /
``scheduling.scheduler`` types (plus protos) and return plain data, so they
carry no edge back to the controller's ops/reconcile/schema layer.
"""

from iris.cluster.controller.scheduling.policy import (
    PreemptionCandidate,
    PreemptionPlan,
    SchedulingOrder,
    run_preemption_pass,
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
) -> PreemptionPlan:
    """Decide which running tasks to evict for higher-priority unscheduled work,
    and bind each preemptor to the worker(s) its victims free."""
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
        return PreemptionPlan(evictions=[], placements=[])
    return run_preemption_pass(unscheduled, running_for_preemption, context)


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
