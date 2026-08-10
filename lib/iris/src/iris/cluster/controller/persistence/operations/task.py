# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped commands for tasks and attempts.

The glues here are small per-tick wrappers around the transition kernel: load
a closed snapshot covering the affected tasks, call the matching
``ReconcileState`` verb, return the effects. ``finalize`` wraps the kernel's
``finalize_tasks`` against the caller's write transaction and commits. ``assign``
is the only scheduler-driven write that doesn't go through the kernel — PENDING
→ ASSIGNED is a direct-write transition with no cascade semantics.

Backend observations enter the database-neutral transition kernel through
``controller.reconcile.apply``; this module owns only persistence commands.
"""

from rigging.timing import Timestamp

from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.persistence import reads, writes
from iris.cluster.controller.persistence.database import Tx
from iris.cluster.controller.persistence.reconcile.commit import commit_effects
from iris.cluster.controller.persistence.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.reconcile import ControllerEffects, ReconcileState, TerminalDecision
from iris.cluster.controller.scheduling.decision import Assignment
from iris.cluster.controller.task_state import task_row_can_be_scheduled
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.resources.names import (
    JobName,
)


def assign(
    cur: Tx,
    assignments: list[Assignment],
    *,
    health: WorkerHealthTracker,
) -> None:
    """Commit assignments to ``tasks.state = ASSIGNED`` + ``task_attempts``.

    Worker-bound dispatch is driven by the control tick's reconcile phase, which
    reads ``tasks.state = ASSIGNED`` rows from a snapshot and fans out Reconcile
    RPCs. This method does not enqueue or fan out anything; the next reconcile
    phase picks up the new ASSIGNED rows (a fresh assignment forces one).
    """
    accepted: list[Assignment] = []
    now_ms = Timestamp.now().epoch_ms()
    jobs_to_update: set[JobName] = set()

    task_map = reads.bulk_get_task_detail(cur, [a.task_id for a in assignments])

    liveness = health.all()
    healthy_worker_ids = [
        a.worker_id for a in assignments if (liv := liveness.get(a.worker_id)) is not None and liv.healthy and liv.active
    ]
    address_map = reads.bulk_get_worker_addresses(cur, healthy_worker_ids)

    potential_job_ids = {task.job_id for task in task_map.values()}
    job_config_map = reads.bulk_get_job_configs(cur, potential_job_ids)

    for assignment in assignments:
        task = task_map.get(assignment.task_id)
        worker_address: str | None = address_map.get(assignment.worker_id)
        if task is None or worker_address is None:
            continue
        if not task_row_can_be_scheduled(task):
            continue
        if task.job_id not in job_config_map:
            continue
        attempt_id = task.current_attempt_id + 1
        writes.assign_to_worker(
            cur,
            assignment.task_id,
            assignment.worker_id,
            worker_address,
            attempt_id,
            now_ms,
            assignment.priority_band,
        )
        jobs_to_update.add(task.job_id)
        accepted.append(assignment)
        task_wire = assignment.task_id.to_wire()
        worker_wire = str(assignment.worker_id)
        cur.register(
            lambda tw=task_wire, ww=worker_wire: log_event(
                "assignment_queued",
                tw,
                worker=ww,
            )
        )
    writes.mark_jobs_running(cur, jobs_to_update, now_ms)


def finalize(
    cur: Tx,
    decisions: list[TerminalDecision],
    *,
    now: Timestamp,
) -> ControllerEffects:
    """Load snapshot for a batch of terminal-state decisions, apply once.

    The snapshot's ``active_tasks_by_job`` already carries the per-victim
    ``ActiveTaskRow`` for PREEMPT/TIMEOUT, and ``bulk_get_attempts`` folds
    in the current attempt (with its ``worker_id``) for every requested
    task — both are derived from the snapshot inside the pure path.
    """
    if not decisions:
        return ControllerEffects()

    all_task_ids: list[JobName] = sorted({d.task_id for d in decisions}, key=lambda tid: tid.to_wire())
    snapshot = load_closed_snapshot(cur, now=now, seed_task_ids=all_task_ids)
    effects = ReconcileState.open(snapshot).finalize_tasks(decisions)
    commit_effects(cur, effects)
    return effects
