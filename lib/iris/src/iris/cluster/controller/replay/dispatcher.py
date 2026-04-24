# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch an :class:`IrisEvent` to the matching ``ControllerTransitions`` method.

This module targets ``main`` semantics: each transition method opens its
own ``with self._db.transaction()`` block and does not take a ``cur``
argument. The branch ``rjpower/iris-sql-store`` rewrites this single
file into the cur-passing form.
"""

from typing import Any

from iris.cluster.controller.replay.events import (
    AddEndpoint,
    ApplyDirectProviderUpdates,
    ApplyHeartbeatsBatch,
    ApplyTaskUpdates,
    BufferDirectKill,
    CancelJob,
    CancelTasksForTimeout,
    DrainForDirectProvider,
    IrisEvent,
    MarkTaskUnschedulable,
    PreemptTask,
    QueueAssignments,
    RegisterOrRefreshWorker,
    RemoveEndpoint,
    RemoveFinishedJob,
    RemoveWorker,
    ReplaceReservationClaims,
    SubmitJob,
    UpdateWorkerPings,
)
from iris.cluster.controller.transitions import ControllerTransitions


def apply_event(transitions: ControllerTransitions, event: IrisEvent) -> Any:
    """Dispatch ``event`` to the matching method on ``transitions``.

    Returns whatever the underlying method returns (``TxResult``,
    ``SubmitJobResult``, ``DirectProviderBatch``, etc.). The dispatcher
    is intentionally thin — its purpose is to keep scenarios free of
    branch-specific calling-convention details.
    """
    match event:
        case SubmitJob(job_id, request, ts):
            return transitions.submit_job(job_id, request, ts)
        case CancelJob(job_id, reason):
            return transitions.cancel_job(job_id, reason)
        case RegisterOrRefreshWorker(worker_id, address, metadata, ts, slice_id, scale_group):
            return transitions.register_or_refresh_worker(
                worker_id=worker_id,
                address=address,
                metadata=metadata,
                ts=ts,
                slice_id=slice_id,
                scale_group=scale_group,
            )
        case QueueAssignments(assignments, direct_dispatch):
            return transitions.queue_assignments(assignments, direct_dispatch=direct_dispatch)
        case ApplyTaskUpdates(request):
            return transitions.apply_task_updates(request)
        case ApplyHeartbeatsBatch(requests):
            return transitions.apply_heartbeats_batch(requests)
        case PreemptTask(task_id, reason):
            return transitions.preempt_task(task_id, reason)
        case CancelTasksForTimeout(task_ids, reason):
            return transitions.cancel_tasks_for_timeout(set(task_ids), reason)
        case MarkTaskUnschedulable(task_id, reason):
            return transitions.mark_task_unschedulable(task_id, reason)
        case RemoveFinishedJob(job_id):
            return transitions.remove_finished_job(job_id)
        case RemoveWorker(worker_id):
            return transitions.remove_worker(worker_id)
        case UpdateWorkerPings(snapshots):
            return transitions.update_worker_pings(snapshots)
        case DrainForDirectProvider(max_promotions):
            return transitions.drain_for_direct_provider(max_promotions)
        case ApplyDirectProviderUpdates(updates):
            return transitions.apply_direct_provider_updates(updates)
        case BufferDirectKill(task_id):
            return transitions.buffer_direct_kill(task_id)
        case AddEndpoint(endpoint):
            return transitions.add_endpoint(endpoint)
        case RemoveEndpoint(endpoint_id):
            return transitions.remove_endpoint(endpoint_id)
        case ReplaceReservationClaims(claims):
            return transitions.replace_reservation_claims(claims)
        case _:
            raise TypeError(f"unhandled IrisEvent variant: {type(event).__name__}")
