# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect transport adapter for the native worker-daemon client port."""

import asyncio
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from rigging.timing import Duration

from iris.chaos import chaos
from iris.cluster.controller.reconcile.worker import (
    AttemptStopReason,
    KeepAttempt,
    LaunchAttempt,
    StopAttempt,
    WorkerReconcilePlan,
    WorkerReconcileRequest,
    WorkerReconcileResult,
)
from iris.cluster.types import AttemptUid, WorkerId
from iris.resources.attempt import AttemptLaunch, AttemptObservation
from iris.resources.endpoint import ExecRequest, ExecResult, ProfileRequest, ProfileResult
from iris.resources.state import TaskState
from iris.resources.system import ProcessInfo
from iris.rpc import job_pb2, worker_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.legacy_job_codec import (
    constraint_to_proto,
    environment_to_proto,
    resource_spec_to_proto,
    runtime_entrypoint_to_proto,
)
from iris.rpc.profile_codec import profile_configuration_to_proto
from iris.rpc.worker_codec import process_info_from_proto
from iris.rpc.worker_connect import WorkerServiceClient
from iris.time_proto import duration_to_proto

DEFAULT_WORKER_RPC_TIMEOUT = Duration.from_seconds(10.0)
RECONCILE_RPC_TIMEOUT = Duration.from_seconds(3.0)
EXEC_IN_CONTAINER_MAX_TIMEOUT = Duration.from_seconds(900.0)

_DEFAULT_PROFILE_DURATION_SECONDS = 10
_PROFILE_RPC_TIMEOUT_MARGIN_MS = 30_000

_T = TypeVar("_T")
_R = TypeVar("_R")

_STOP_REASON_TO_WIRE = {
    AttemptStopReason.CANCELLED: worker_pb2.Worker.STOP_REASON_CANCELLED,
    AttemptStopReason.PREEMPTED: worker_pb2.Worker.STOP_REASON_PREEMPTED,
    AttemptStopReason.SUPERSEDED: worker_pb2.Worker.STOP_REASON_SUPERSEDED,
    AttemptStopReason.JOB_TERMINATED: worker_pb2.Worker.STOP_REASON_JOB_TERMINATED,
    AttemptStopReason.TASK_TIMEOUT: worker_pb2.Worker.STOP_REASON_TASK_TIMEOUT,
    AttemptStopReason.WORKER_DRAIN: worker_pb2.Worker.STOP_REASON_WORKER_DRAIN,
}


def _attempt_launch_to_proto(launch: AttemptLaunch) -> job_pb2.RunTaskRequest:
    template = launch.template
    result = job_pb2.RunTaskRequest(
        task_id=launch.task_id.to_wire(),
        attempt_id=launch.attempt_id,
        attempt_uid=launch.attempt_uid,
        num_tasks=template.num_tasks,
        entrypoint=runtime_entrypoint_to_proto(template.entrypoint),
        environment=environment_to_proto(template.environment),
        bundle_id=template.bundle_id,
        resources=resource_spec_to_proto(template.resources),
        ports=template.ports,
        constraints=[constraint_to_proto(constraint) for constraint in template.constraints],
        task_image=template.task_image,
        priority=int(template.priority_band),
        container_profile=int(template.container_profile),
    )
    if template.timeout is not None:
        result.timeout.CopyFrom(duration_to_proto(template.timeout))
    if template.coscheduling is not None:
        result.coscheduling.group_by = template.coscheduling.group_by
    return result


def _reconcile_request_to_proto(request: WorkerReconcileRequest) -> worker_pb2.Worker.ReconcileRequest:
    desired = []
    for item in request.desired:
        if isinstance(item, LaunchAttempt):
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=item.attempt_uid,
                    run=worker_pb2.Worker.AttemptSpec(request=_attempt_launch_to_proto(item.launch)),
                )
            )
        elif isinstance(item, KeepAttempt):
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=item.attempt_uid,
                    run=worker_pb2.Worker.AttemptSpec(),
                )
            )
        else:
            assert isinstance(item, StopAttempt)
            desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=item.attempt_uid,
                    stop=_STOP_REASON_TO_WIRE[item.reason],
                )
            )
    return worker_pb2.Worker.ReconcileRequest(worker_id=request.worker_id, desired=desired)


def _attempt_observation_from_proto(value: worker_pb2.Worker.AttemptObservation) -> AttemptObservation:
    return AttemptObservation(
        attempt_uid=AttemptUid(value.attempt_uid),
        state=TaskState(value.state),
        exit_code=value.exit_code or None,
        error=value.error or None,
        container_id=value.container_id or None,
    )


def _fan_out(
    items: Sequence[_T],
    parallelism: int,
    run_one: Callable[[asyncio.Semaphore, _T], Awaitable[_R]],
) -> list[_R]:
    if not items:
        return []

    async def _run() -> list[_R]:
        semaphore = asyncio.Semaphore(parallelism)
        return await asyncio.gather(*(run_one(semaphore, item) for item in items))

    return asyncio.run(_run())


class WorkerStubFactory(Protocol):
    """Factory for cached worker Connect clients."""

    def get_stub(self, address: str) -> WorkerServiceClient: ...
    def evict(self, address: str) -> None: ...
    def close(self) -> None: ...


class RpcWorkerStubFactory:
    """Caches one async Connect client per worker address."""

    def __init__(self, timeout: Duration = DEFAULT_WORKER_RPC_TIMEOUT) -> None:
        self._timeout = timeout
        self._stubs: dict[str, WorkerServiceClient] = {}
        self._lock = threading.Lock()

    def get_stub(self, address: str) -> WorkerServiceClient:
        with self._lock:
            stub = self._stubs.get(address)
            if stub is None:
                stub = WorkerServiceClient(
                    address=f"http://{address}",
                    timeout_ms=self._timeout.to_ms(),
                    accept_compression=IRIS_RPC_COMPRESSIONS,
                    send_compression=None,
                )
                self._stubs[address] = stub
            return stub

    def evict(self, address: str) -> None:
        with self._lock:
            self._stubs.pop(address, None)

    def close(self) -> None:
        with self._lock:
            self._stubs.clear()


@dataclass
class RpcWorkerClient:
    """Translate native worker operations to the legacy worker Connect wire."""

    stub_factory: WorkerStubFactory

    def reconcile(
        self,
        plans: list[WorkerReconcilePlan],
        addresses: Mapping[WorkerId, str],
        *,
        parallelism: int,
    ) -> list[WorkerReconcileResult]:
        async def reconcile_one(semaphore: asyncio.Semaphore, plan: WorkerReconcilePlan) -> WorkerReconcileResult:
            async with semaphore:
                try:
                    if rule := chaos("controller.reconcile"):
                        await asyncio.sleep(rule.delay_seconds)
                        raise ConnectionError("chaos: controller.reconcile")
                    response = await asyncio.wait_for(
                        self.stub_factory.get_stub(addresses[plan.worker_id]).reconcile(
                            _reconcile_request_to_proto(plan.request)
                        ),
                        timeout=RECONCILE_RPC_TIMEOUT.to_seconds(),
                    )
                    return WorkerReconcileResult(
                        worker_id=plan.worker_id,
                        observations=[_attempt_observation_from_proto(value) for value in response.observed],
                        error=None,
                        self_healthy=response.health.healthy,
                        responder_worker_id=response.worker_id or None,
                    )
                except Exception as error:
                    return WorkerReconcileResult(
                        worker_id=plan.worker_id,
                        observations=[],
                        error=str(error) or type(error).__name__,
                    )

        return _fan_out(plans, parallelism, reconcile_one)

    def evict(self, address: str) -> None:
        self.stub_factory.evict(address)

    def process_status(self, address: str) -> ProcessInfo:
        stub = self.stub_factory.get_stub(address)
        response = asyncio.run(stub.get_process_status(job_pb2.GetProcessStatusRequest(), timeout_ms=10_000))
        return process_info_from_proto(response.process_info)

    def profile(self, address: str, request: ProfileRequest) -> ProfileResult:
        stub = self.stub_factory.get_stub(address)
        duration_seconds = int(request.duration.to_seconds()) if request.duration is not None else 0
        wire_request = job_pb2.ProfileTaskRequest(
            target=(
                "/system/process"
                if request.attempt is None
                else f"{request.attempt.task.resource_id}:{request.attempt.attempt_number}"
            ),
            duration_seconds=duration_seconds,
            profile_type=profile_configuration_to_proto(request.profile),
        )
        timeout_ms = (duration_seconds or _DEFAULT_PROFILE_DURATION_SECONDS) * 1_000 + _PROFILE_RPC_TIMEOUT_MARGIN_MS
        response = asyncio.run(stub.profile_task(wire_request, timeout_ms=timeout_ms))
        return ProfileResult(response.profile_data, response.error)

    def exec(self, address: str, request: ExecRequest) -> ExecResult:
        stub = self.stub_factory.get_stub(address)
        timeout_seconds = int(request.timeout.to_seconds()) if request.timeout is not None else 0
        wire_request = worker_pb2.Worker.ExecInContainerRequest(
            task_id=request.attempt.task.resource_id,
            command=request.command,
            timeout_seconds=timeout_seconds,
        )
        rpc_timeout_ms = EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms() if timeout_seconds < 0 else (timeout_seconds + 5) * 1_000
        response = asyncio.run(stub.exec_in_container(wire_request, timeout_ms=rpc_timeout_ms))
        return ExecResult(response.exit_code, response.stdout, response.stderr, response.error)

    def close(self) -> None:
        self.stub_factory.close()
