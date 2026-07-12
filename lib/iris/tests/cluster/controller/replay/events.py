# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``IrisEvent`` dataclass union + ``apply_event`` dispatcher.

Each variant captures the arguments of one public mutation method on
``ControllerTestState``. ``apply_event`` opens a write transaction
and invokes the matching method.

Multi-transaction orchestrators (``ops.worker.fail``, ``prune_old_data``)
and ``*_for_test`` helpers are intentionally excluded — scenarios call
those methods directly when needed.
"""

from dataclasses import dataclass
from typing import Any

from iris.cluster.controller import ops
from iris.cluster.controller.ops.task import Assignment, apply_dispatch_updates, finalize
from iris.cluster.controller.projections.endpoints import EndpointRow
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.commit import commit_effects
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.transition_driver import (
    CursorTransitionReader,
    WorkerTaskUpdates,
    apply_task_observations,
)


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


@dataclass(frozen=True, slots=True)
class ApplyTaskUpdates:
    request: WorkerTaskUpdates


@dataclass(frozen=True, slots=True)
class PreemptTask:
    task_id: JobName
    reason: str


@dataclass(frozen=True, slots=True)
class CancelTasksForTimeout:
    task_ids: frozenset[JobName]
    reason: str


@dataclass(frozen=True, slots=True)
class DrainForDirectProvider:
    max_promotions: int = 16


@dataclass(frozen=True, slots=True)
class ApplyDirectProviderUpdates:
    updates: list[TaskUpdate]


@dataclass(frozen=True, slots=True)
class AddEndpoint:
    endpoint: EndpointRow


@dataclass(frozen=True, slots=True)
class RemoveEndpoint:
    endpoint_id: str


IrisEvent = (
    SubmitJob
    | CancelJob
    | RegisterOrRefreshWorker
    | QueueAssignments
    | ApplyTaskUpdates
    | PreemptTask
    | CancelTasksForTimeout
    | DrainForDirectProvider
    | ApplyDirectProviderUpdates
    | AddEndpoint
    | RemoveEndpoint
)


def apply_event(transitions: ControllerTestState, event: IrisEvent) -> Any:
    """Dispatch ``event`` to the matching method, opening one write transaction.

    On this branch ``ControllerTestState`` methods take an explicit
    ``cur`` and the caller owns transaction scope. ``apply_event`` opens
    one transaction per event so scenarios stay branch-agnostic — same
    granularity as the main-flavor dispatcher that opens its own tx
    inside each method.
    """
    with transitions._db.transaction() as cur:
        match event:
            case SubmitJob(job_id, request, ts):
                return ops.job.submit(
                    cur,
                    job_id=job_id,
                    request=request,
                    ts=ts,
                )
            case CancelJob(job_id, reason):
                return ops.job.cancel(cur, job_id=job_id, reason=reason)
            case RegisterOrRefreshWorker(worker_id, address, metadata, ts, slice_id, scale_group):
                return ops.worker.register(
                    cur,
                    worker_id=worker_id,
                    address=address,
                    metadata=metadata,
                    ts=ts,
                    health=transitions._health,
                    slice_id=slice_id,
                    scale_group=scale_group,
                )
            case QueueAssignments(assignments):
                return ops.task.assign(cur, assignments, health=transitions._health)
            case ApplyTaskUpdates(request):
                return apply_task_observations(
                    cur,
                    [request],
                    health=transitions._health,
                    now=Timestamp.now(),
                )
            case PreemptTask(task_id, reason):
                return finalize(
                    cur,
                    [TerminalDecision(TerminalKind.PREEMPT, task_id, reason)],
                    now=Timestamp.now(),
                )
            case CancelTasksForTimeout(task_ids, reason):
                return finalize(
                    cur,
                    [
                        TerminalDecision(TerminalKind.TIMEOUT, tid, reason)
                        for tid in sorted(task_ids, key=lambda t: t.to_wire())
                    ],
                    now=Timestamp.now(),
                )
            case DrainForDirectProvider(max_promotions):
                return dispatch.drain_for_dispatch(cur, max_promotions=max_promotions)
            case ApplyDirectProviderUpdates(updates):
                # Relocated glue: author the effects from this write transaction,
                # then commit them — the two steps the controller now does apart.
                effects = apply_dispatch_updates(CursorTransitionReader(cur), updates, now=Timestamp.now())
                commit_effects(cur, effects)
                return effects
            case AddEndpoint(endpoint):
                return transitions._endpoints.add(cur, endpoint)
            case RemoveEndpoint(endpoint_id):
                return transitions._endpoints.remove(cur, endpoint_id)
            case _:
                raise TypeError(f"unhandled IrisEvent variant: {type(event).__name__}")
