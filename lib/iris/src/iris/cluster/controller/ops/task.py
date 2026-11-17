# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped commands for tasks and attempts.

The glues here are small per-tick wrappers around the transition kernel: load
a closed snapshot covering the affected tasks, call the matching
``ReconcileState`` verb, return the effects. ``finalize`` wraps the kernel's
``finalize_tasks`` against the caller's write transaction and commits; the
controller-facing ``apply_reconcile_updates`` validates exact backend
observations against a fresh read snapshot, wraps ``apply_updates``, and returns
effects uncommitted. ``assign`` is the only scheduler-driven
write that doesn't go through the kernel — PENDING → ASSIGNED is a direct-write
transition with no cascade semantics.

All backend-reported task states enter through ``apply_reconcile_updates``.
"""

import logging
from dataclasses import dataclass, replace

from rigging.timing import Timestamp

from iris.cluster.controller import reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.db import Tx
from iris.cluster.controller.reconcile import (
    ControllerEffects,
    ReconcileState,
    TaskUpdate,
    TerminalDecision,
)
from iris.cluster.controller.reconcile.commit import commit_effects
from iris.cluster.controller.reconcile.loader import TransitionReader, load_closed_snapshot
from iris.cluster.controller.reconcile.snapshot import TransitionSnapshot
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, task_row_can_be_scheduled
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


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


def apply_reconcile_updates(
    source: TransitionReader,
    observations: list[TaskUpdate],
    *,
    now: Timestamp,
) -> ControllerEffects:
    """Author effects for exact backend observations from a fresh snapshot.

    Observations from every backend use this path. Exact Attempt UIDs prevent a
    late report from targeting a replacement incarnation; optional ``worker_id``
    identifies observations that also participate in Iris worker liveness.
    """
    snapshot = _reconcile_snapshot(source, observations, now=now)
    updates = [
        _resolve_missing_observation(snapshot, update) for update in _exact_current_observations(snapshot, observations)
    ]
    return ReconcileState.open(snapshot).apply_updates(updates)


def _reconcile_snapshot(
    source: TransitionReader,
    observations: list[TaskUpdate],
    *,
    now: Timestamp,
) -> TransitionSnapshot:
    relevant_task_ids = [
        observation.task_id
        for observation in observations
        if observation.new_state not in (job_pb2.TASK_STATE_UNSPECIFIED, job_pb2.TASK_STATE_PENDING)
    ]
    attempt_keys = [(observation.task_id, observation.attempt_id) for observation in observations]
    return source.transition_snapshot(
        now=now,
        seed_task_ids=relevant_task_ids,
        extra_attempt_keys=attempt_keys,
        observation_uids=[observation.attempt_uid for observation in observations if observation.attempt_uid],
        seed_worker_ids=[observation.worker_id for observation in observations if observation.worker_id],
    )


def _exact_current_observations(
    snapshot: TransitionSnapshot,
    observations: list[TaskUpdate],
) -> list[TaskUpdate]:
    accepted: list[TaskUpdate] = []
    for observation in observations:
        if observation.attempt_uid is None:
            logger.warning(
                "Dropping backend observation without Attempt UID: task=%s attempt=%d",
                observation.task_id,
                observation.attempt_id,
            )
            continue
        resolved = snapshot.attempt_uid_index.get(observation.attempt_uid)
        if resolved != (observation.task_id, observation.attempt_id):
            logger.warning(
                "Dropping stale provider observation: task=%s attempt=%d uid=%s",
                observation.task_id,
                observation.attempt_id,
                observation.attempt_uid,
            )
            continue
        if observation.worker_id is not None and observation.worker_id not in snapshot.active_workers:
            logger.warning(
                "Dropping observation from inactive worker: task=%s attempt=%d worker=%s",
                observation.task_id,
                observation.attempt_id,
                observation.worker_id,
            )
            continue
        accepted.append(observation)
    return accepted


def _resolve_missing_observation(snapshot: TransitionSnapshot, observation: TaskUpdate) -> TaskUpdate:
    if observation.new_state != job_pb2.TASK_STATE_MISSING:
        return observation
    task = snapshot.tasks.get(observation.task_id)
    missing_state = (
        job_pb2.TASK_STATE_WORKER_FAILED
        if task is not None and task.state in ACTIVE_TASK_STATES
        else job_pb2.TASK_STATE_FAILED
    )
    return replace(observation, new_state=missing_state, error="worker_lost_spec")


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
