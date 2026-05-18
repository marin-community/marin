# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerProvider: TaskProvider backed by worker daemons via Connect RPC."""

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Protocol

from rigging.timing import Duration

from iris.chaos import chaos
from iris.cluster.controller.provider import ProviderError
from iris.cluster.controller.transitions import (
    RunningTaskEntry,
    TaskUpdate,
    task_updates_from_proto,
)
from iris.cluster.types import WorkerId
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
class WorkerReconcilePlan:
    """Per-worker reconcile work for one polling tick.

    ``start_tasks`` is the list of fresh ASSIGNED rows to dispatch (empty if
    none). ``expected_tasks`` is the full set the worker should currently
    have running — anything outside this set the worker auto-kills locally.
    """

    worker_id: WorkerId
    address: str | None
    start_tasks: list[job_pb2.RunTaskRequest]
    expected_tasks: list[RunningTaskEntry]


@dataclass(frozen=True)
class WorkerReconcileResult:
    """Combined StartTasks + PollTasks outcome for one worker.

    ``start_*`` fields are populated only when the plan included a
    StartTasks payload; ``poll_*`` are populated for every plan since every
    healthy worker is polled each tick.
    """

    worker_id: WorkerId
    start_response: worker_pb2.Worker.StartTasksResponse | None
    start_error: str | None
    poll_updates: list[TaskUpdate] | None
    poll_error: str | None


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

    async def _reconcile_one(
        self,
        sem: asyncio.Semaphore,
        plan: "WorkerReconcilePlan",
    ) -> "WorkerReconcileResult":
        """Per-worker reconcile: StartTasks (if any) followed by PollTasks.

        Order matters within a single worker — the worker's task table must
        be populated by the dispatch before PollTasks lands, otherwise the
        worker's ``_missing_task_status`` path would auto-kill freshly
        dispatched tasks. Different workers run independently under the
        shared semaphore.
        """
        async with sem:
            if not plan.address:
                err = f"Worker {plan.worker_id} has no address"
                return WorkerReconcileResult(
                    worker_id=plan.worker_id,
                    start_response=None,
                    start_error=err if plan.start_tasks else None,
                    poll_updates=None,
                    poll_error=err,
                )

            stub = self.stub_factory.get_stub(plan.address)
            start_response: worker_pb2.Worker.StartTasksResponse | None = None
            start_error: str | None = None
            if plan.start_tasks:
                try:
                    if rule := chaos("controller.start_tasks"):
                        await asyncio.sleep(rule.delay_seconds)
                        raise ProviderError("chaos: controller.start_tasks")
                    start_response = await stub.start_tasks(worker_pb2.Worker.StartTasksRequest(tasks=plan.start_tasks))
                except Exception as e:
                    start_error = str(e)

            poll_updates: list[TaskUpdate] | None = None
            poll_error: str | None = None
            try:
                if rule := chaos("controller.poll_tasks"):
                    await asyncio.sleep(rule.delay_seconds)
                    raise ProviderError("chaos: controller.poll_tasks")
                expected = []
                for entry in plan.expected_tasks:
                    if iter_rule := chaos("controller.poll_iteration"):
                        await asyncio.sleep(iter_rule.delay_seconds)
                    expected.append(
                        job_pb2.WorkerTaskStatus(task_id=entry.task_id.to_wire(), attempt_id=entry.attempt_id)
                    )
                response = await stub.poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=expected))
                poll_updates = task_updates_from_proto(response.tasks)
            except Exception as e:
                poll_error = str(e)

            return WorkerReconcileResult(
                worker_id=plan.worker_id,
                start_response=start_response,
                start_error=start_error,
                poll_updates=poll_updates,
                poll_error=poll_error,
            )

    def reconcile_workers(self, plans: list["WorkerReconcilePlan"]) -> list["WorkerReconcileResult"]:
        """Run one reconcile pass across many workers under a single event loop.

        Each plan describes the work for one worker (StartTasks payload,
        expected_tasks for the poll). Workers reconcile concurrently, capped
        at ``self.parallelism``; within one worker StartTasks completes
        before PollTasks (see ``_reconcile_one``).
        """
        if not plans:
            return []

        async def _run() -> list["WorkerReconcileResult"]:
            sem = asyncio.Semaphore(self.parallelism)
            return await asyncio.gather(*(self._reconcile_one(sem, p) for p in plans))

        return asyncio.run(_run())

    def close(self) -> None:
        self.stub_factory.close()
