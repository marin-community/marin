# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``IrisEvent`` dataclass union + ``apply_event`` dispatcher.

Each variant captures the arguments of one public mutation method on
``ControllerTransitions``. ``apply_event`` opens a write transaction
and invokes the matching method.

Multi-transaction orchestrators (``fail_workers``, ``prune_old_data``)
and ``*_for_test`` helpers are intentionally excluded — scenarios call
those methods directly when needed.
"""

from dataclasses import dataclass
from typing import Any

from iris.cluster.controller.schema import EndpointRow
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    ReservationClaim,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp


@dataclass(frozen=True, slots=True)
class SubmitJob:
    job_id: JobName
    request: controller_pb2.Controller.LaunchJobRequest
    ts: Timestamp


@dataclass(frozen=True, slots=True)
class CancelJob:
    job_id: JobName
    reason: str


@dataclass(frozen=True, slots=True)
class RegisterOrRefreshWorker:
    worker_id: WorkerId
    address: str
    metadata: job_pb2.WorkerMetadata
    ts: Timestamp
    slice_id: str = ""
    scale_group: str = ""


@dataclass(frozen=True, slots=True)
class QueueAssignments:
    assignments: list[Assignment]
    direct_dispatch: bool = False


@dataclass(frozen=True, slots=True)
class ApplyTaskUpdates:
    request: HeartbeatApplyRequest


@dataclass(frozen=True, slots=True)
class ApplyHeartbeatsBatch:
    requests: list[HeartbeatApplyRequest]


@dataclass(frozen=True, slots=True)
class PreemptTask:
    task_id: JobName
    reason: str


@dataclass(frozen=True, slots=True)
class CancelTasksForTimeout:
    task_ids: frozenset[JobName]
    reason: str


@dataclass(frozen=True, slots=True)
class MarkTaskUnschedulable:
    task_id: JobName
    reason: str


@dataclass(frozen=True, slots=True)
class RemoveFinishedJob:
    job_id: JobName


@dataclass(frozen=True, slots=True)
class RemoveWorker:
    worker_id: WorkerId


@dataclass(frozen=True, slots=True)
class UpdateWorkerPings:
    worker_ids: list[WorkerId]


@dataclass(frozen=True, slots=True)
class DrainForDirectProvider:
    max_promotions: int = 16


@dataclass(frozen=True, slots=True)
class ApplyDirectProviderUpdates:
    updates: list[TaskUpdate]


@dataclass(frozen=True, slots=True)
class BufferDirectKill:
    task_id: str


@dataclass(frozen=True, slots=True)
class AddEndpoint:
    endpoint: EndpointRow


@dataclass(frozen=True, slots=True)
class RemoveEndpoint:
    endpoint_id: str


@dataclass(frozen=True, slots=True)
class ReplaceReservationClaims:
    claims: dict[WorkerId, ReservationClaim]


IrisEvent = (
    SubmitJob
    | CancelJob
    | RegisterOrRefreshWorker
    | QueueAssignments
    | ApplyTaskUpdates
    | ApplyHeartbeatsBatch
    | PreemptTask
    | CancelTasksForTimeout
    | MarkTaskUnschedulable
    | RemoveFinishedJob
    | RemoveWorker
    | UpdateWorkerPings
    | DrainForDirectProvider
    | ApplyDirectProviderUpdates
    | BufferDirectKill
    | AddEndpoint
    | RemoveEndpoint
    | ReplaceReservationClaims
)


def apply_event(transitions: ControllerTransitions, event: IrisEvent) -> Any:
    """Dispatch ``event`` to the matching method, opening one write transaction.

    On this branch ``ControllerTransitions`` methods take an explicit
    ``cur`` and the caller owns transaction scope. ``apply_event`` opens
    one transaction per event so scenarios stay branch-agnostic — same
    granularity as the main-flavor dispatcher that opens its own tx
    inside each method.
    """
    store = transitions._store
    with store.transaction() as cur:
        match event:
            case SubmitJob(job_id, request, ts):
                return transitions.submit_job(cur, job_id, request, ts)
            case CancelJob(job_id, reason):
                return transitions.cancel_job(cur, job_id, reason)
            case RegisterOrRefreshWorker(worker_id, address, metadata, ts, slice_id, scale_group):
                return transitions.register_or_refresh_worker(
                    cur,
                    worker_id=worker_id,
                    address=address,
                    metadata=metadata,
                    ts=ts,
                    slice_id=slice_id,
                    scale_group=scale_group,
                )
            case QueueAssignments(assignments, direct_dispatch):
                return transitions.queue_assignments(cur, assignments, direct_dispatch=direct_dispatch)
            case ApplyTaskUpdates(request):
                return transitions.apply_task_updates(cur, request)
            case ApplyHeartbeatsBatch(requests):
                return transitions.apply_heartbeats_batch(cur, requests)
            case PreemptTask(task_id, reason):
                return transitions.preempt_task(cur, task_id, reason)
            case CancelTasksForTimeout(task_ids, reason):
                return transitions.cancel_tasks_for_timeout(cur, set(task_ids), reason)
            case MarkTaskUnschedulable(task_id, reason):
                return transitions.mark_task_unschedulable(cur, task_id, reason)
            case RemoveFinishedJob(job_id):
                return transitions.remove_finished_job(cur, job_id)
            case RemoveWorker(worker_id):
                return transitions.remove_worker(cur, worker_id)
            case UpdateWorkerPings(worker_ids):
                return transitions.update_worker_pings(cur, worker_ids)
            case DrainForDirectProvider(max_promotions):
                return transitions.drain_for_direct_provider(cur, max_promotions)
            case ApplyDirectProviderUpdates(updates):
                return transitions.apply_direct_provider_updates(cur, updates)
            case BufferDirectKill(task_id):
                return transitions.buffer_direct_kill(cur, task_id)
            case AddEndpoint(endpoint):
                return transitions.add_endpoint(cur, endpoint)
            case RemoveEndpoint(endpoint_id):
                return transitions.remove_endpoint(cur, endpoint_id)
            case ReplaceReservationClaims(claims):
                return transitions.replace_reservation_claims(cur, claims)
            case _:
                raise TypeError(f"unhandled IrisEvent variant: {type(event).__name__}")
