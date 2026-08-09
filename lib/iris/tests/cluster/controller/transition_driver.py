# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test driver for landing task-state updates through the production path.

The live controller lands worker-reported task states through the reconcile
loop (``ops.worker.apply_reconcile``). To keep tests exercising
the same code the controller runs, ``apply_task_observations`` rebuilds a
per-worker batch of :class:`WorkerTaskUpdates` into native reconcile
``AttemptObservation`` records and applies them through that production verb.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from iris.cluster.controller.persistence.database import Tx
from iris.cluster.controller.persistence.operations.task import apply_dispatch_updates
from iris.cluster.controller.persistence.operations.worker import apply_reconcile
from iris.cluster.controller.persistence.reconcile.commit import commit_effects
from iris.cluster.controller.persistence.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.persistence.schema import task_attempts_table
from iris.cluster.controller.reconcile.effects import ControllerEffects
from iris.cluster.controller.reconcile.snapshot import TaskUpdate, TransitionSnapshot
from iris.cluster.controller.reconcile.worker import (
    KeepAttempt,
    WorkerReconcilePlan,
    WorkerReconcileRequest,
    WorkerReconcileResult,
)
from iris.cluster.controller.worker_health import (
    WorkerHealthEvent,
    WorkerHealthEventKind,
    WorkerHealthTracker,
)
from iris.cluster.resources.attempt import AttemptObservation
from iris.cluster.resources.state import TaskState
from iris.cluster.types import AttemptUid, JobName, WorkerId
from rigging.timing import Timestamp
from sqlalchemy import select


@dataclass(frozen=True)
class CursorTransitionReader:
    """A ``TransitionReader`` backed by an open write transaction.

    Lets the test drivers author effects through the production ``apply_reconcile``
    / ``apply_dispatch_updates`` path while loading the snapshot from the very
    transaction they commit into — same ``cur``, same explicit ``now``, no extra
    ``Timestamp.now()`` and no second connection — so a frozen-clock replay
    scenario stays byte-identical to the pre-relocation commit-side load.
    """

    cur: Tx

    def transition_snapshot(
        self,
        *,
        now: Timestamp,
        seed_worker_ids: Iterable[WorkerId] = (),
        observation_uids: Iterable[AttemptUid] = (),
        seed_task_ids: Iterable[JobName] = (),
        extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
    ) -> TransitionSnapshot:
        return load_closed_snapshot(
            self.cur,
            now=now,
            seed_worker_ids=seed_worker_ids,
            observation_uids=observation_uids,
            seed_task_ids=seed_task_ids,
            extra_attempt_keys=extra_attempt_keys,
        )


@dataclass(frozen=True)
class WorkerTaskUpdates:
    """A worker reporting observed states for a batch of its attempts.

    A worker id plus the neutral per-task updates to land.
    """

    worker_id: WorkerId
    updates: list[TaskUpdate]


def commit_reconcile(
    cur: Tx,
    plan_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]],
    *,
    now: Timestamp,
) -> ControllerEffects:
    """Author + commit worker-reconcile effects against a write cursor (test glue).

    The two steps the controller now does apart (backend authors, controller
    commits), collapsed for tests that drive the kernel directly from a write
    transaction. Loads from ``cur`` so the snapshot reflects the same transaction
    the effects commit into.
    """
    effects = apply_reconcile(CursorTransitionReader(cur), plan_results, now=now)
    commit_effects(cur, effects)
    return effects


def commit_dispatch_updates(
    cur: Tx,
    updates: list[TaskUpdate],
    *,
    now: Timestamp,
) -> ControllerEffects:
    """Author + commit direct-provider effects against a write cursor (test glue)."""
    effects = apply_dispatch_updates(CursorTransitionReader(cur), updates, now=now)
    commit_effects(cur, effects)
    return effects


def _attempt_uid(cur: Tx, task_id: JobName, attempt_id: int) -> str:
    """Read the controller-minted attempt_uid that routes an observation."""
    row = cur.execute(
        select(task_attempts_table.c.attempt_uid).where(
            task_attempts_table.c.task_id == task_id,
            task_attempts_table.c.attempt_id == attempt_id,
        )
    ).first()
    assert row is not None, f"no task_attempts row for {task_id.to_wire()}/{attempt_id}"
    return row.attempt_uid


def _observation(uid: str, update: TaskUpdate) -> AttemptObservation:
    return AttemptObservation(
        attempt_uid=AttemptUid(uid),
        state=TaskState(update.new_state),
        exit_code=update.exit_code,
        error=update.error,
        container_id=update.container_id,
    )


def apply_task_observations(
    cur: Tx,
    requests: list[WorkerTaskUpdates],
    *,
    health: WorkerHealthTracker,
    now: Timestamp,
) -> ControllerEffects:
    """Land ``requests`` through the production reconcile-observation verb.

    Builds one ``(WorkerReconcilePlan, WorkerReconcileResult)`` pair per worker: the
    plan lists each touched attempt's uid as desired (so the production filter
    accepts the observation) and the result reports the observed state. The
    kernel-derived build failures ride back on the effects; this helper folds
    them into ``health`` the way ``Controller._fold_health`` does in production.
    """
    plan_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]] = []
    for req in requests:
        observations: list[AttemptObservation] = []
        desired: list[KeepAttempt] = []
        for update in req.updates:
            uid = _attempt_uid(cur, update.task_id, update.attempt_id)
            observations.append(_observation(uid, update))
            desired.append(KeepAttempt(AttemptUid(uid)))
        plan = WorkerReconcilePlan(
            worker_id=req.worker_id,
            request=WorkerReconcileRequest(worker_id=req.worker_id, desired=tuple(desired)),
        )
        result = WorkerReconcileResult(worker_id=req.worker_id, observations=observations, error=None)
        plan_results.append((plan, result))

    # Author the effects through the relocated (backend-side) reconcile glue,
    # reading from this write transaction, then commit them — the controller now
    # does these as two separate steps.
    effects = apply_reconcile(CursorTransitionReader(cur), plan_results, now=now)
    commit_effects(cur, effects)
    build_events = [WorkerHealthEvent(wid, WorkerHealthEventKind.BUILD_FAILED) for wid in effects.health.build_failed]
    if build_events:
        health.apply(build_events, now_ms=now.epoch_ms())
    return effects
