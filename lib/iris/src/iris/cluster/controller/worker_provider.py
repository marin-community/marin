# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerProvider: TaskProvider backed by worker daemons via Connect RPC."""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Protocol

from rigging.timing import Duration

from iris.chaos import chaos
from iris.cluster.controller.provider import ProviderError
from iris.cluster.controller.reconcile import WorkerReconcilePlan
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.controller.transitions import (
    TaskUpdate,
    log_event,
    task_updates_from_proto,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2, worker_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.worker_connect import WorkerServiceClient

logger = logging.getLogger(__name__)

DEFAULT_WORKER_RPC_TIMEOUT = Duration.from_seconds(10.0)


@dataclass(frozen=True)
class PingResult:
    """Result of a Ping RPC to a single worker."""

    worker_id: WorkerId
    worker_address: str | None
    healthy: bool = True
    health_error: str = ""
    error: str | None = None


@dataclass(frozen=True)
class ReconcileResult:
    """Unified per-worker reconcile outcome.

    ``observations`` is the (possibly empty) list of proto observations the
    apply layer should consume. ``error`` is set when the reconcile RPC
    outright failed; ``observations`` is then empty.
    """

    worker_id: WorkerId
    observations: list[worker_pb2.Worker.AttemptObservation]
    error: str | None = None


@dataclass(frozen=True)
class _LegacyDispatch:
    """Legacy three-list wire payload derived from a reconcile plan."""

    worker_id: WorkerId
    address: str | None
    start_tasks: list[job_pb2.RunTaskRequest]
    expected_tasks: list[RunningTaskEntry]
    stop_tasks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _LegacyResult:
    """Per-worker StartTasks + PollTasks outcome (legacy wire only)."""

    worker_id: WorkerId
    start_response: worker_pb2.Worker.StartTasksResponse | None
    start_error: str | None
    poll_updates: list[TaskUpdate] | None
    poll_error: str | None


def _legacy_request_from_plan(plan: WorkerReconcilePlan, address: str | None) -> _LegacyDispatch:
    """Translate a reconcile plan into the legacy three-list dispatch payload."""
    start_tasks: list[job_pb2.RunTaskRequest] = []
    expected_tasks: list[RunningTaskEntry] = []
    stop_tasks: list[str] = []

    for desired in plan.request.desired:
        if desired.HasField("run"):
            expected_tasks.append(
                RunningTaskEntry(
                    task_id=JobName.from_wire(desired.task_id),
                    attempt_id=desired.attempt_id,
                )
            )
            if desired.run.HasField("request"):
                req = job_pb2.RunTaskRequest()
                req.CopyFrom(desired.run.request)
                req.task_id = desired.task_id
                req.attempt_id = desired.attempt_id
                start_tasks.append(req)
        elif desired.HasField("stop"):
            stop_tasks.append(desired.task_id)

    return _LegacyDispatch(
        worker_id=plan.worker_id,
        address=address,
        start_tasks=start_tasks,
        expected_tasks=expected_tasks,
        stop_tasks=stop_tasks,
    )


def _observations_from_legacy_result(
    plan: WorkerReconcilePlan,
    result: _LegacyResult,
) -> list[worker_pb2.Worker.AttemptObservation]:
    """Synthesize proto observations from a legacy StartTasks+PollTasks result.

    Forwards poll updates and emits a ``WORKER_FAILED`` observation for any
    task the worker explicitly rejected via a non-accepted ack.
    """
    observations: list[worker_pb2.Worker.AttemptObservation] = []

    if result.poll_error is None and result.poll_updates:
        for update in result.poll_updates:
            kwargs: dict = {
                "attempt_uid": "",
                "state": update.new_state,
                "task_id": update.task_id.to_wire(),
                "attempt_id": update.attempt_id,
            }
            if update.exit_code is not None:
                kwargs["exit_code"] = update.exit_code
            if update.error is not None:
                kwargs["error"] = update.error
            if update.container_id is not None:
                kwargs["container_id"] = update.container_id
            observations.append(worker_pb2.Worker.AttemptObservation(**kwargs))

    if result.start_response is not None:
        attempt_by_task: dict[str, int] = {d.task_id: d.attempt_id for d in plan.request.desired if d.HasField("run")}
        for ack in result.start_response.acks:
            if ack.accepted:
                continue
            log_event(
                "task_rejected",
                ack.task_id,
                trigger="start_tasks_ack",
                worker=str(result.worker_id),
                error=ack.error,
            )
            observations.append(
                worker_pb2.Worker.AttemptObservation(
                    attempt_uid="",
                    state=job_pb2.TASK_STATE_WORKER_FAILED,
                    error=f"Worker rejected task: {ack.error}",
                    task_id=ack.task_id,
                    attempt_id=attempt_by_task.get(ack.task_id, -1),
                )
            )

    return observations


class WorkerStubFactory(Protocol):
    """Factory for getting cached async worker RPC stubs."""

    def get_stub(self, address: str) -> WorkerServiceClient: ...
    def evict(self, address: str) -> None: ...
    def close(self) -> None: ...


class RpcWorkerStubFactory:
    """Caches async WorkerServiceClient stubs by address so each worker gets
    one persistent async HTTP client across RPCs."""

    def __init__(self, timeout: Duration = DEFAULT_WORKER_RPC_TIMEOUT) -> None:
        self._timeout = timeout
        self._stubs: dict[str, WorkerServiceClient] = {}
        self._lock = threading.Lock()

    @property
    def timeout_ms(self) -> int:
        return self._timeout.to_ms()

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
class WorkerProvider:
    """TaskProvider backed by worker daemons via async Connect RPCs.

    Each public method spins up an asyncio event loop and dispatches the
    relevant RPC to each worker concurrently via `asyncio.gather`, capped at
    `parallelism` in-flight requests by a local semaphore. Cached stubs in
    the factory keep their pyqwest connection pools across rounds.
    """

    stub_factory: WorkerStubFactory
    parallelism: int = 128

    def get_process_status(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        if not address:
            raise ProviderError(f"Worker {worker_id} has no address")
        stub = self.stub_factory.get_stub(address)
        # Forward with target cleared — the worker serves its own process status.
        forwarded = job_pb2.GetProcessStatusRequest(
            max_log_lines=request.max_log_lines,
            log_substring=request.log_substring,
            min_log_level=request.min_log_level,
        )
        return asyncio.run(stub.get_process_status(forwarded, timeout_ms=10000))

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        if address:
            self.stub_factory.evict(address)

    def profile_task(
        self,
        address: str,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        stub = self.stub_factory.get_stub(address)
        return asyncio.run(stub.profile_task(request, timeout_ms=timeout_ms))

    def exec_in_container(
        self,
        address: str,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        stub = self.stub_factory.get_stub(address)
        # Negative timeout means no limit; use a large RPC deadline (1 hour)
        if timeout_seconds < 0:
            rpc_timeout_ms = 3_600_000
        else:
            rpc_timeout_ms = (timeout_seconds + 5) * 1000
        return asyncio.run(stub.exec_in_container(request, timeout_ms=rpc_timeout_ms))

    def ping_workers(self, workers: list[tuple[WorkerId, str | None]]) -> list[PingResult]:
        """Send Ping RPCs to all workers concurrently. Returns per-worker results."""
        if not workers:
            return []

        async def _one(sem: asyncio.Semaphore, wid: WorkerId, addr: str | None) -> PingResult:
            async with sem:
                if not addr:
                    return PingResult(worker_id=wid, worker_address=addr, error=f"Worker {wid} has no address")
                try:
                    if rule := chaos("controller.ping"):
                        await asyncio.sleep(rule.delay_seconds)
                        raise ProviderError("chaos: controller.ping")
                    stub = self.stub_factory.get_stub(addr)
                    response = await stub.ping(worker_pb2.Worker.PingRequest())
                    if not response.healthy:
                        return PingResult(
                            worker_id=wid,
                            worker_address=addr,
                            error=f"worker {wid} reported unhealthy: {response.health_error}",
                        )
                    return PingResult(
                        worker_id=wid,
                        worker_address=addr,
                        healthy=response.healthy,
                        health_error=response.health_error,
                    )
                except Exception as e:
                    return PingResult(worker_id=wid, worker_address=addr, error=str(e))

        async def _run() -> list[PingResult]:
            sem = asyncio.Semaphore(self.parallelism)
            return await asyncio.gather(*(_one(sem, wid, addr) for wid, addr in workers))

        return asyncio.run(_run())

    async def _reconcile_one_legacy(
        self,
        sem: asyncio.Semaphore,
        dispatch: _LegacyDispatch,
    ) -> _LegacyResult:
        """Push StartTasks (if any) then PollTasks for one worker."""
        async with sem:
            if not dispatch.address:
                err = f"Worker {dispatch.worker_id} has no address"
                return _LegacyResult(
                    worker_id=dispatch.worker_id,
                    start_response=None,
                    start_error=err if dispatch.start_tasks else None,
                    poll_updates=None,
                    poll_error=err,
                )

            stub = self.stub_factory.get_stub(dispatch.address)
            start_response: worker_pb2.Worker.StartTasksResponse | None = None
            start_error: str | None = None
            if dispatch.start_tasks:
                try:
                    if rule := chaos("controller.start_tasks"):
                        await asyncio.sleep(rule.delay_seconds)
                        raise ProviderError("chaos: controller.start_tasks")
                    start_response = await stub.start_tasks(
                        worker_pb2.Worker.StartTasksRequest(tasks=dispatch.start_tasks)
                    )
                except Exception as e:
                    start_error = str(e)

            poll_updates: list[TaskUpdate] | None = None
            poll_error: str | None = None
            try:
                if rule := chaos("controller.poll_tasks"):
                    await asyncio.sleep(rule.delay_seconds)
                    raise ProviderError("chaos: controller.poll_tasks")
                expected = []
                for entry in dispatch.expected_tasks:
                    if iter_rule := chaos("controller.poll_iteration"):
                        await asyncio.sleep(iter_rule.delay_seconds)
                    expected.append(
                        job_pb2.WorkerTaskStatus(task_id=entry.task_id.to_wire(), attempt_id=entry.attempt_id)
                    )
                response = await stub.poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=expected))
                poll_updates = task_updates_from_proto(response.tasks)
            except Exception as e:
                poll_error = str(e)

            return _LegacyResult(
                worker_id=dispatch.worker_id,
                start_response=start_response,
                start_error=start_error,
                poll_updates=poll_updates,
                poll_error=poll_error,
            )

    async def _reconcile_one_via_reconcile(
        self,
        sem: asyncio.Semaphore,
        plan: WorkerReconcilePlan,
        address: str | None,
    ) -> ReconcileResult:
        """Issue a single Reconcile RPC to one worker under the shared semaphore."""
        async with sem:
            if not address:
                return ReconcileResult(
                    worker_id=plan.worker_id,
                    observations=[],
                    error=f"Worker {plan.worker_id} has no address",
                )
            try:
                if rule := chaos("controller.reconcile"):
                    await asyncio.sleep(rule.delay_seconds)
                    raise ProviderError("chaos: controller.reconcile")
                stub = self.stub_factory.get_stub(address)
                response = await stub.reconcile(plan.request)
                return ReconcileResult(
                    worker_id=plan.worker_id,
                    observations=list(response.observed),
                    error=None,
                )
            except Exception as e:
                return ReconcileResult(worker_id=plan.worker_id, observations=[], error=str(e))

    def reconcile_workers(
        self,
        plans: list[WorkerReconcilePlan],
        addresses: dict[WorkerId, str | None],
        *,
        use_reconcile_rpc: bool,
    ) -> list[ReconcileResult]:
        """Fan out one reconcile pass across many workers under a single event loop.

        Workers reconcile concurrently, capped at ``self.parallelism``.
        Branches internally on ``use_reconcile_rpc``: the new branch sends
        the ``Reconcile`` RPC; the legacy branch issues StartTasks+PollTasks
        and synthesizes proto observations from the results.
        """
        if not plans:
            return []

        if use_reconcile_rpc:

            async def _run_reconcile() -> list[ReconcileResult]:
                sem = asyncio.Semaphore(self.parallelism)
                return await asyncio.gather(
                    *(self._reconcile_one_via_reconcile(sem, p, addresses.get(p.worker_id)) for p in plans)
                )

            return asyncio.run(_run_reconcile())

        dispatches = [_legacy_request_from_plan(p, addresses.get(p.worker_id)) for p in plans]

        async def _run_legacy() -> list[_LegacyResult]:
            sem = asyncio.Semaphore(self.parallelism)
            return await asyncio.gather(*(self._reconcile_one_legacy(sem, d) for d in dispatches))

        legacy_results = asyncio.run(_run_legacy())

        plan_by_worker = {p.worker_id: p for p in plans}
        out: list[ReconcileResult] = []
        for legacy in legacy_results:
            plan = plan_by_worker[legacy.worker_id]
            if legacy.start_error is not None:
                out.append(
                    ReconcileResult(
                        worker_id=legacy.worker_id,
                        observations=[],
                        error=legacy.start_error,
                    )
                )
                continue
            if legacy.poll_error is not None:
                logger.debug("PollTasks failed for worker %s: %s", legacy.worker_id, legacy.poll_error)
            observations = _observations_from_legacy_result(plan, legacy)
            out.append(
                ReconcileResult(
                    worker_id=legacy.worker_id,
                    observations=observations,
                    error=None,
                )
            )
        return out

    def close(self) -> None:
        self.stub_factory.close()
