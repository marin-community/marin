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

from iris.cluster.controller import direct_provider, ops, reads, writes
from iris.cluster.controller.ops.task import Assignment, apply_terminal_decisions
from iris.cluster.controller.ops.task import apply_provider_updates as apply_direct_provider_updates
from iris.cluster.controller.projections.endpoints import EndpointRow
from iris.cluster.controller.reads import ReservationClaim
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.scheduling_policy import claim_workers_for_reservations, cleanup_stale_claims
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp

from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, apply_task_observations


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


@dataclass(frozen=True, slots=True)
class ReplaceReservationClaims:
    claims: dict[WorkerId, ReservationClaim]


@dataclass(frozen=True, slots=True)
class RunReservationClaimCycle:
    """Run the controller's reservation claim phase: clean up stale claims, then
    claim eligible workers for unsatisfied reservation entries, persisting the
    result. Drives the same ``scheduling_policy.refresh_reservation_claims`` path
    the controller runs each scheduling cycle.
    """


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
    | ReplaceReservationClaims
    | RunReservationClaimCycle
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
                    run_template_cache=transitions._run_template_cache,
                )
            case CancelJob(job_id, reason):
                return ops.job.cancel(
                    cur, job_id=job_id, reason=reason, endpoints=transitions._endpoints, health=transitions._health
                )
            case RegisterOrRefreshWorker(worker_id, address, metadata, ts, slice_id, scale_group):
                return ops.worker.register_or_refresh(
                    cur,
                    worker_id=worker_id,
                    address=address,
                    metadata=metadata,
                    ts=ts,
                    health=transitions._health,
                    worker_attrs=transitions._worker_attrs,
                    slice_id=slice_id,
                    scale_group=scale_group,
                )
            case QueueAssignments(assignments):
                return ops.task.queue_assignments(cur, assignments, health=transitions._health)
            case ApplyTaskUpdates(request):
                return apply_task_observations(
                    cur,
                    [request],
                    health=transitions._health,
                    endpoints=transitions._endpoints,
                    now=Timestamp.now(),
                )
            case PreemptTask(task_id, reason):
                return apply_terminal_decisions(
                    cur,
                    [TerminalDecision(TerminalKind.PREEMPT, task_id, reason)],
                    health=transitions._health,
                    endpoints=transitions._endpoints,
                    now=Timestamp.now(),
                )
            case CancelTasksForTimeout(task_ids, reason):
                return apply_terminal_decisions(
                    cur,
                    [
                        TerminalDecision(TerminalKind.TIMEOUT, tid, reason)
                        for tid in sorted(task_ids, key=lambda t: t.to_wire())
                    ],
                    health=transitions._health,
                    endpoints=transitions._endpoints,
                    now=Timestamp.now(),
                )
            case DrainForDirectProvider(max_promotions):
                return direct_provider.drain_for_direct_provider(
                    cur, cache=transitions._run_template_cache, max_promotions=max_promotions
                )
            case ApplyDirectProviderUpdates(updates):
                return apply_direct_provider_updates(
                    cur,
                    updates,
                    health=transitions._health,
                    endpoints=transitions._endpoints,
                    now=Timestamp.now(),
                )
            case AddEndpoint(endpoint):
                return transitions._endpoints.add(cur, endpoint)
            case RemoveEndpoint(endpoint_id):
                return transitions._endpoints.remove(cur, endpoint_id)
            case ReplaceReservationClaims(claims):
                return writes.replace_reservation_claims(cur, claims)
            case RunReservationClaimCycle():
                # Persist through the caller's ``cur`` rather than the standalone
                # ``refresh_reservation_claims`` write transaction, which would
                # nest inside this open one. Reads use the separate read engine.
                claims = reads.list_claims(cur)
                changed = cleanup_stale_claims(claims, transitions._db, transitions._health)
                changed = (
                    claim_workers_for_reservations(
                        claims, transitions._db, transitions._health, transitions._worker_attrs
                    )
                    or changed
                )
                if changed:
                    writes.replace_reservation_claims(cur, claims)
                return claims
            case _:
                raise TypeError(f"unhandled IrisEvent variant: {type(event).__name__}")
