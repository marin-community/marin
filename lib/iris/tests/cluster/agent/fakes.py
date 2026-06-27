# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test doubles for the remote-backend agent loop.

``FakeClusterBackend`` is a CLUSTER_VIEW :class:`TaskBackend` that advances each
desired attempt RUNNING then SUCCEEDED across reconcile rounds, recording the
``tasks_to_run`` it was asked to converge to. ``FakeTransport`` records each
``PollRequest`` and replies with scripted ``PollResponse``s, standing in for the
root over the network seam.
"""

from typing import ClassVar

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.backend import (
    AutoscaleResult,
    BackendCapability,
    ProviderUnsupportedError,
    ReconcileResult,
    ScheduleInput,
    ScheduleResult,
    TaskTarget,
)
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2, remote_agent_pb2, worker_pb2


def make_req(task_id: str, attempt_id: int, attempt_uid: str) -> job_pb2.RunTaskRequest:
    """Build a minimal ``RunTaskRequest`` with a runnable entrypoint."""
    req = job_pb2.RunTaskRequest()
    req.task_id = task_id
    req.attempt_id = attempt_id
    req.attempt_uid = attempt_uid
    req.entrypoint.run_command.argv.extend(["echo", "hi"])
    return req


def poll_response(
    reqs: list[job_pb2.RunTaskRequest],
    *,
    sync_id: int = 1,
    root_epoch: int = 1,
    acks: list[remote_agent_pb2.AckObservation] | None = None,
) -> remote_agent_pb2.PollResponse:
    """Build a full-snapshot ``PollResponse`` upserting every ``req`` in ``reqs``."""
    return remote_agent_pb2.PollResponse(
        root_epoch=root_epoch,
        new_sync_id=sync_id,
        snapshot=True,
        upserts=[
            remote_agent_pb2.DesiredAttempt(
                attempt_uid=req.attempt_uid,
                desired_generation=req.attempt_id,
                spec=worker_pb2.Worker.AttemptSpec(request=req),
                constraints=list(req.constraints),
            )
            for req in reqs
        ],
        acks=list(acks or []),
    )


class FakeTransport:
    """Records each ``PollRequest`` and replies with scripted ``PollResponse``s.

    Responses are consumed in order; once a single response remains it repeats,
    so a steady-state desired set persists across ticks.
    """

    def __init__(self, responses: list[remote_agent_pb2.PollResponse]) -> None:
        assert responses, "FakeTransport needs at least one scripted response"
        self._responses = list(responses)
        self.requests: list[remote_agent_pb2.PollRequest] = []

    def poll(self, request: remote_agent_pb2.PollRequest) -> remote_agent_pb2.PollResponse:
        self.requests.append(request)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class FakeClusterBackend:
    """CLUSTER_VIEW backend that walks each desired attempt RUNNING then SUCCEEDED.

    The first reconcile that sees an attempt only observes it; the second emits a
    RUNNING ``TaskUpdate`` and the third a SUCCEEDED one, mirroring how a pod
    backend reports progress across reconcile rounds.
    """

    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset({BackendCapability.CLUSTER_VIEW})

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.autoscaler: Autoscaler | None = None
        self._seen_count: dict[tuple[str, int], int] = {}
        self.last_seen_task_ids: set[str] = set()

    def schedule(self, snapshot: ScheduleInput) -> ScheduleResult:
        return ScheduleResult()

    def reconcile(self, snapshot: ControlSnapshot) -> ReconcileResult:
        self.last_seen_task_ids = {req.task_id for req in snapshot.tasks_to_run}
        updates: list[TaskUpdate] = []
        for req in snapshot.tasks_to_run:
            key = (req.task_id, req.attempt_id)
            count = self._seen_count.get(key, 0) + 1
            self._seen_count[key] = count
            if count == 2:
                updates.append(
                    TaskUpdate(
                        task_id=JobName.from_wire(req.task_id),
                        attempt_id=req.attempt_id,
                        new_state=job_pb2.TASK_STATE_RUNNING,
                    )
                )
            elif count == 3:
                updates.append(
                    TaskUpdate(
                        task_id=JobName.from_wire(req.task_id),
                        attempt_id=req.attempt_id,
                        new_state=job_pb2.TASK_STATE_SUCCEEDED,
                    )
                )
        return ReconcileResult(updates=updates)

    def autoscale(
        self,
        snapshot: ControlSnapshot,
        residual_demand: list[DemandEntry],
        dead_workers: list[WorkerId],
    ) -> AutoscaleResult:
        return AutoscaleResult()

    def attach_autoscaler(self, autoscaler: Autoscaler) -> None:
        raise ProviderUnsupportedError("FakeClusterBackend manages its own capacity")

    def get_process_status(
        self,
        target: TaskTarget,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        raise ProviderUnsupportedError("FakeClusterBackend has no interactive ops")

    def profile_task(
        self,
        target: TaskTarget,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        raise ProviderUnsupportedError("FakeClusterBackend has no interactive ops")

    def exec_in_container(
        self,
        target: TaskTarget,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        raise ProviderUnsupportedError("FakeClusterBackend has no interactive ops")

    def close(self) -> None:
        pass
