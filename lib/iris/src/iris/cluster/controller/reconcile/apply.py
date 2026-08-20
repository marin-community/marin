# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply backend observations to database-neutral transition snapshots."""

from rigging.timing import Timestamp

from iris.cluster.controller.reconcile import ControllerEffects, ReconcileState
from iris.cluster.controller.reconcile.reader import TransitionReader
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.worker import LaunchAttempt, WorkerReconcilePlan, WorkerReconcileResult
from iris.resources.names import AttemptUid, JobName, WorkerId
from iris.resources.state import TaskState


def apply_worker_reconcile(
    source: TransitionReader,
    plan_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]],
    *,
    now: Timestamp,
) -> ControllerEffects:
    """Author one atomic effect set from a worker reconcile batch."""
    task_ids: list[JobName] = []
    attempt_keys: list[tuple[JobName, int]] = []
    attempt_uids: list[AttemptUid] = []
    worker_ids: list[WorkerId] = []

    for plan, result in plan_results:
        worker_ids.append(plan.worker_id)
        if result.error is not None:
            for desired in plan.request.desired:
                if not isinstance(desired, LaunchAttempt):
                    continue
                launch = desired.launch
                task_ids.append(launch.task_id)
                attempt_keys.append((launch.task_id, launch.attempt_id))
            continue

        planned_uids = {desired.attempt_uid for desired in plan.request.desired if desired.attempt_uid}
        attempt_uids.extend(
            AttemptUid(observation.attempt_uid)
            for observation in result.observations
            if observation.attempt_uid and observation.attempt_uid in planned_uids
        )

    snapshot = source.transition_snapshot(
        now=now,
        seed_worker_ids=worker_ids,
        observation_uids=attempt_uids,
        seed_task_ids=task_ids,
        extra_attempt_keys=attempt_keys,
    )
    return ReconcileState.open(snapshot).reconcile(plan_results, now)


def apply_dispatch_updates(
    source: TransitionReader,
    updates: list[TaskUpdate],
    *,
    now: Timestamp,
) -> ControllerEffects:
    """Author one atomic effect set from direct-provider observations."""
    relevant_task_ids = [
        update.task_id for update in updates if update.new_state not in (TaskState.UNSPECIFIED, TaskState.PENDING)
    ]
    snapshot = source.transition_snapshot(
        now=now,
        seed_task_ids=relevant_task_ids,
        extra_attempt_keys=[(update.task_id, update.attempt_id) for update in updates],
    )
    return ReconcileState.open(snapshot).record_updates(updates)
