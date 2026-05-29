# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped commands for tasks and attempts.

The ``apply_*`` glues here are the small per-tick wrappers around the
transition kernel: load a closed snapshot covering the affected tasks via a
scoped loader, call the matching ``batches.apply_*_batch``, drain effects
through ``apply_effects``. ``queue_assignments`` is the only scheduler-driven
write that doesn't go through the kernel — PENDING → ASSIGNED is a
direct-write transition with no cascade semantics.

Worker-reported task states land through ``ops.worker.apply_reconcile_observations``
(the reconcile loop), not here.
"""

from dataclasses import dataclass

from rigging.timing import Timestamp
from sqlalchemy import case, func
from sqlalchemy import update as sa_update

from iris.cluster.controller import reads, writes
from iris.cluster.controller.audit import log_event
from iris.cluster.controller.db import Tx
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.reconcile.batches import (
    apply_direct_provider_updates_batch,
    apply_terminal_decisions_batch,
)
from iris.cluster.controller.reconcile.effects import ControllerEffects, apply_effects
from iris.cluster.controller.reconcile.loader import load_tasks_slice
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.task import TerminalDecision
from iris.cluster.controller.schema import jobs_table
from iris.cluster.controller.task_state import task_row_can_be_scheduled
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2


@dataclass(frozen=True)
class Assignment:
    """Scheduler assignment decision.

    ``priority_band`` is the effective band computed at scheduling time
    (after applying any over-budget downgrade). Stamped onto ``tasks.priority_band``
    when the row transitions to ASSIGNED so that the preemption pass uses a
    fixed, point-in-time band rather than re-evaluating against current spend
    on every tick. Re-evaluating caused mutual preemption between two
    same-band users sitting at the budget cliff. ``None`` leaves the column
    unchanged (used by call sites that do not run the budget computation,
    e.g. K8s direct-provider promotions and manual reassignment).
    """

    task_id: JobName
    worker_id: WorkerId
    priority_band: int | None = None


def queue_assignments(
    cur: Tx,
    assignments: list[Assignment],
    *,
    health: WorkerHealthTracker,
) -> None:
    """Commit assignments to ``tasks.state = ASSIGNED`` + ``task_attempts``.

    Worker-bound dispatch is driven by the polling reconcile loop, which
    reads ``tasks.state = ASSIGNED`` rows from a snapshot and fans out
    Reconcile RPCs. This method does not enqueue or fan out anything;
    callers are responsible for waking ``_polling_wake`` after commit so
    the reconcile loop sees the new ASSIGNED rows on its next tick.

    Reservation-holder assignments are admitted (they anchor the worker
    for taint-injection) but never produce a worker-bound RunTaskRequest.
    """
    accepted: list[Assignment] = []
    now_ms = Timestamp.now().epoch_ms()
    jobs_to_update: set[str] = set()

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
        jobs_to_update.add(task.job_id.to_wire())
        accepted.append(assignment)
    for job_id_wire in jobs_to_update:
        cur.execute(
            sa_update(jobs_table)
            .where(jobs_table.c.job_id == JobName.from_wire(job_id_wire))
            .values(
                state=case(
                    (jobs_table.c.state == job_pb2.JOB_STATE_PENDING, job_pb2.JOB_STATE_RUNNING),
                    else_=jobs_table.c.state,
                ),
                started_at_ms=func.coalesce(jobs_table.c.started_at_ms, now_ms),
            )
        )
    for a in accepted:
        task_wire = a.task_id.to_wire()
        worker_wire = str(a.worker_id)
        cur.register(
            lambda tw=task_wire, ww=worker_wire: log_event(
                "assignment_queued",
                tw,
                worker=ww,
            )
        )


def apply_provider_updates(
    cur: Tx,
    updates: list[TaskUpdate],
    *,
    health: WorkerHealthTracker,
    endpoints: EndpointsProjection,
    now: Timestamp,
) -> ControllerEffects:
    """Load snapshot for direct-provider updates, run state machine, apply effects."""
    relevant_task_ids = [
        update.task_id
        for update in updates
        if update.new_state not in (job_pb2.TASK_STATE_UNSPECIFIED, job_pb2.TASK_STATE_PENDING)
    ]
    attempt_keys = [(update.task_id, update.attempt_id) for update in updates]
    snapshot = load_tasks_slice(
        cur,
        relevant_task_ids,
        now=now,
        extra_attempt_keys=attempt_keys,
    )
    effects = apply_direct_provider_updates_batch(snapshot, updates)
    apply_effects(cur, effects, health=health, endpoints=endpoints, now=now)
    return effects


def apply_terminal_decisions(
    cur: Tx,
    decisions: list[TerminalDecision],
    *,
    health: WorkerHealthTracker,
    endpoints: EndpointsProjection,
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
    snapshot = load_tasks_slice(cur, all_task_ids, now=now)
    effects = apply_terminal_decisions_batch(snapshot, decisions)
    apply_effects(cur, effects, health=health, endpoints=endpoints, now=now)
    return effects
