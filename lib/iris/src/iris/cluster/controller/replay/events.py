# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``IrisEvent`` dataclass union — one variant per public mutation method on
``ControllerTransitions``.

Each variant captures the arguments of its corresponding method. The
intent is to be the *describe-what-happened* surface so the same
sequence can be replayed against fresh DBs on different branches. The
events are calling-convention agnostic — the caller-supplied
``cur: TransactionCursor`` is **not** an event field; the dispatcher
threads it in based on the branch's API shape.

Multi-transaction orchestrators (``fail_workers``, ``prune_old_data``)
and ``*_for_test`` helpers are intentionally excluded — scenarios call
those methods directly when needed.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from iris.cluster.controller.schema import EndpointRow
from iris.cluster.controller.transitions import (
    Assignment,
    HeartbeatApplyRequest,
    ReservationClaim,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp


@dataclass(frozen=True, slots=True)
class SubmitJob:
    """``ControllerTransitions.submit_job`` invocation."""

    job_id: JobName
    request: controller_pb2.Controller.LaunchJobRequest
    ts: Timestamp


@dataclass(frozen=True, slots=True)
class CancelJob:
    """``ControllerTransitions.cancel_job`` invocation."""

    job_id: JobName
    reason: str


@dataclass(frozen=True, slots=True)
class RegisterOrRefreshWorker:
    """``ControllerTransitions.register_or_refresh_worker`` invocation."""

    worker_id: WorkerId
    address: str
    metadata: job_pb2.WorkerMetadata
    ts: Timestamp
    slice_id: str = ""
    scale_group: str = ""


@dataclass(frozen=True, slots=True)
class QueueAssignments:
    """``ControllerTransitions.queue_assignments`` invocation."""

    assignments: list[Assignment]
    direct_dispatch: bool = False


@dataclass(frozen=True, slots=True)
class ApplyTaskUpdates:
    """``ControllerTransitions.apply_task_updates`` invocation."""

    request: HeartbeatApplyRequest


@dataclass(frozen=True, slots=True)
class ApplyHeartbeatsBatch:
    """``ControllerTransitions.apply_heartbeats_batch`` invocation."""

    requests: list[HeartbeatApplyRequest]


@dataclass(frozen=True, slots=True)
class PreemptTask:
    """``ControllerTransitions.preempt_task`` invocation."""

    task_id: JobName
    reason: str


@dataclass(frozen=True, slots=True)
class CancelTasksForTimeout:
    """``ControllerTransitions.cancel_tasks_for_timeout`` invocation."""

    task_ids: frozenset[JobName]
    reason: str


@dataclass(frozen=True, slots=True)
class MarkTaskUnschedulable:
    """``ControllerTransitions.mark_task_unschedulable`` invocation."""

    task_id: JobName
    reason: str


@dataclass(frozen=True, slots=True)
class RemoveFinishedJob:
    """``ControllerTransitions.remove_finished_job`` invocation."""

    job_id: JobName


@dataclass(frozen=True, slots=True)
class RemoveWorker:
    """``ControllerTransitions.remove_worker`` invocation."""

    worker_id: WorkerId


@dataclass(frozen=True, slots=True)
class UpdateWorkerPings:
    """``ControllerTransitions.update_worker_pings`` invocation."""

    snapshots: Mapping[WorkerId, job_pb2.WorkerResourceSnapshot | None]


@dataclass(frozen=True, slots=True)
class DrainForDirectProvider:
    """``ControllerTransitions.drain_for_direct_provider`` invocation."""

    max_promotions: int = 16


@dataclass(frozen=True, slots=True)
class ApplyDirectProviderUpdates:
    """``ControllerTransitions.apply_direct_provider_updates`` invocation."""

    updates: list[TaskUpdate]


@dataclass(frozen=True, slots=True)
class BufferDirectKill:
    """``ControllerTransitions.buffer_direct_kill`` invocation."""

    task_id: str


@dataclass(frozen=True, slots=True)
class AddEndpoint:
    """``ControllerTransitions.add_endpoint`` invocation."""

    endpoint: EndpointRow


@dataclass(frozen=True, slots=True)
class RemoveEndpoint:
    """``ControllerTransitions.remove_endpoint`` invocation."""

    endpoint_id: str


@dataclass(frozen=True, slots=True)
class ReplaceReservationClaims:
    """``ControllerTransitions.replace_reservation_claims`` invocation."""

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
"""Union of every event variant. ~18 in total."""
