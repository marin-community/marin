# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drivers for landing test task observations through the controller path."""

from collections.abc import Iterable
from dataclasses import dataclass, replace

from rigging.timing import Timestamp
from sqlalchemy import select

from iris.cluster.controller.db import Tx
from iris.cluster.controller.ops.reconcile import apply_observation
from iris.cluster.controller.ops.task import apply_reconcile_updates
from iris.cluster.controller.reconcile.commit import commit_effects
from iris.cluster.controller.reconcile.effects import ControllerEffects
from iris.cluster.controller.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.reconcile.snapshot import TaskUpdate, TransitionSnapshot
from iris.cluster.controller.reconcile.worker import (
    WorkerReconcilePlan,
    WorkerReconcileResult,
    task_updates_from_result,
)
from iris.cluster.controller.schema import task_attempts_table
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import AttemptUid, JobName, WorkerId
from iris.rpc import job_pb2


@dataclass(frozen=True)
class CursorTransitionReader:
    """A ``TransitionReader`` backed by an open write transaction.

    Lets the test drivers author effects through ``apply_reconcile_updates``
    while loading the snapshot from the very
    transaction they commit into — same ``cur``, same explicit ``now``, no extra
    ``Timestamp.now()`` and no second connection. This keeps frozen-clock replay
    scenarios deterministic.
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
    """Normalize worker protocol fixtures and commit them through the controller path."""
    updates = [
        update for plan, result in plan_results for update in task_updates_from_result(plan, result, observed_at=now)
    ]
    effects = apply_reconcile_updates(CursorTransitionReader(cur), updates, now=now)
    commit_effects(cur, effects)
    return effects


def commit_dispatch_updates(
    cur: Tx,
    updates: list[TaskUpdate],
    *,
    now: Timestamp,
) -> ControllerEffects:
    """Author + commit direct-provider effects against a write cursor (test glue)."""
    observations = [
        replace(update, attempt_uid=AttemptUid(_attempt_uid(cur, update.task_id, update.attempt_id)))
        for update in updates
    ]
    return commit_observed_dispatch_updates(cur, observations, now=now)


def commit_observed_dispatch_updates(
    cur: Tx,
    observations: list[TaskUpdate],
    *,
    now: Timestamp,
) -> ControllerEffects:
    effects = apply_reconcile_updates(CursorTransitionReader(cur), observations, now=now)
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


def apply_task_observations(
    cur: Tx,
    requests: list[WorkerTaskUpdates],
    *,
    health: WorkerHealthTracker,
    now: Timestamp,
) -> ControllerEffects:
    """Land exact worker observations through the production controller verb."""
    observations: list[TaskUpdate] = []
    for req in requests:
        for update in req.updates:
            observations.append(
                replace(
                    update,
                    attempt_uid=AttemptUid(_attempt_uid(cur, update.task_id, update.attempt_id)),
                    worker_id=req.worker_id,
                    execution_started_at=now if update.new_state == job_pb2.TASK_STATE_BUILDING else None,
                )
            )

    application = apply_observation(
        CursorTransitionReader(cur),
        observations,
        [],
        worker_health=health,
        now=now,
    )
    commit_effects(cur, application.effects)
    return application.effects
