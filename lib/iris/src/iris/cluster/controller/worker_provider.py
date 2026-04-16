# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerProvider: TaskProvider backed by worker daemons via heartbeat RPC."""

import asyncio
import logging
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from iris.chaos import chaos
from iris.cluster.controller.provider import ProviderError
from iris.cluster.controller.transitions import (
    DispatchBatch,
    HeartbeatApplyRequest,
    RunningTaskEntry,
    TaskUpdate,
    task_updates_from_proto,
)
from iris.cluster.types import WorkerId
from iris.rpc import job_pb2
from iris.rpc import worker_pb2
from iris.rpc.worker_connect import WorkerServiceClient
from rigging.timing import Duration

logger = logging.getLogger(__name__)

DEFAULT_WORKER_RPC_TIMEOUT = Duration.from_seconds(10.0)
_SLOW_HEARTBEAT_RPC_LOG_THRESHOLD_MS = 5_000


def _heartbeat_rpc_context(
    batch: DispatchBatch,
    *,
    elapsed_ms: int,
    timeout_ms: int | None,
) -> str:
    timeout_fragment = f" timeout_ms={timeout_ms}" if timeout_ms is not None else ""
    return (
        f"worker={batch.worker_id} address={batch.worker_address or '<missing>'}"
        f" elapsed_ms={elapsed_ms}{timeout_fragment}"
        f" expected={len(batch.running_tasks)} run={len(batch.tasks_to_run)} kill={len(batch.tasks_to_kill)}"
    )


@dataclass(frozen=True)
class PingResult:
    """Result of a Ping RPC to a single worker."""

    worker_id: WorkerId
    worker_address: str | None
    resource_snapshot: job_pb2.WorkerResourceSnapshot | None = None
    healthy: bool = True
    health_error: str = ""
    error: str | None = None


class WorkerStubFactory(Protocol):
    """Factory for getting cached async worker RPC stubs."""

    def get_stub(self, address: str) -> WorkerServiceClient: ...
    def evict(self, address: str) -> None: ...
    def close(self) -> None: ...


class RpcWorkerStubFactory:
    """Caches async WorkerServiceClient stubs by address so each worker gets
    one persistent async HTTP client instead of a new one per RPC."""

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
                )
                self._stubs[address] = stub
            return stub

    def evict(self, address: str) -> None:
        with self._lock:
            self._stubs.pop(address, None)

    def close(self) -> None:
        with self._lock:
            self._stubs.clear()


def _apply_request_from_response(
    worker_id: WorkerId,
    response: job_pb2.HeartbeatResponse,
) -> HeartbeatApplyRequest:
    """Convert a HeartbeatResponse proto to a HeartbeatApplyRequest."""
    return HeartbeatApplyRequest(
        worker_id=worker_id,
        worker_resource_snapshot=(response.resource_snapshot if response.resource_snapshot.ByteSize() > 0 else None),
        updates=task_updates_from_proto(response.tasks),
    )


@dataclass
class WorkerProvider:
    """TaskProvider backed by worker daemons via async heartbeat RPC.

    Per round, `sync()` spins up an asyncio event loop via `asyncio.run`
    and dispatches per-worker heartbeat RPCs concurrently via
    `asyncio.gather`, capped at `parallelism` in-flight requests by a
    local semaphore. Cached stubs in the factory keep their pyqwest
    connection pools across rounds independently of the Python loop.
    """

    stub_factory: WorkerStubFactory
    parallelism: int = 128

    def sync(
        self,
        batches: list[DispatchBatch],
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        if not batches:
            return []
        return asyncio.run(self._sync_all(batches))

    async def _sync_all(
        self,
        batches: list[DispatchBatch],
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        sem = asyncio.Semaphore(self.parallelism)
        return await asyncio.gather(*(self._heartbeat_one_safe(sem, b) for b in batches))

    async def _heartbeat_one_safe(
        self,
        sem: asyncio.Semaphore,
        batch: DispatchBatch,
    ) -> tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]:
        async with sem:
            try:
                apply_req = await self._heartbeat_one(batch)
                return (batch, apply_req, None)
            except Exception as e:
                return (batch, None, str(e))

    async def _heartbeat_one(self, batch: DispatchBatch) -> HeartbeatApplyRequest:
        """Send heartbeat RPC to one worker and return the apply request."""
        started = monotonic()
        timeout_ms = getattr(self.stub_factory, "timeout_ms", None)

        if rule := chaos("controller.heartbeat"):
            await asyncio.sleep(rule.delay_seconds)
            raise ProviderError("chaos: heartbeat unavailable")

        if not batch.worker_address:
            raise ProviderError(f"Worker {batch.worker_id} has no address for heartbeat")

        stub = self.stub_factory.get_stub(batch.worker_address)

        expected_tasks = []
        for entry in batch.running_tasks:
            if rule := chaos("controller.heartbeat.iteration"):
                await asyncio.sleep(rule.delay_seconds)
            expected_tasks.append(
                job_pb2.WorkerTaskStatus(
                    task_id=entry.task_id.to_wire(),
                    attempt_id=entry.attempt_id,
                )
            )
        request = job_pb2.HeartbeatRequest(
            tasks_to_run=batch.tasks_to_run,
            tasks_to_kill=batch.tasks_to_kill,
            expected_tasks=expected_tasks,
        )
        try:
            response = await stub.heartbeat(request)

            if not response.worker_healthy:
                health_error = response.health_error or "worker reported unhealthy"
                raise ProviderError(f"worker {batch.worker_id} reported unhealthy: {health_error}")

            elapsed_ms = int((monotonic() - started) * 1000)
            if elapsed_ms >= _SLOW_HEARTBEAT_RPC_LOG_THRESHOLD_MS:
                logger.warning(
                    "Slow heartbeat RPC succeeded: %s",
                    _heartbeat_rpc_context(batch, elapsed_ms=elapsed_ms, timeout_ms=timeout_ms),
                )
            return _apply_request_from_response(batch.worker_id, response)
        except Exception as e:
            elapsed_ms = int((monotonic() - started) * 1000)
            context = _heartbeat_rpc_context(batch, elapsed_ms=elapsed_ms, timeout_ms=timeout_ms)
            if isinstance(e, ProviderError):
                raise ProviderError(f"{e}; {context}") from e
            raise ProviderError(f"heartbeat RPC failed: {context}; error={e}") from e

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
        """Send Ping RPCs to all workers in parallel. Returns per-worker results."""
        if not workers:
            return []
        futures = {self._pool.submit(self._ping_one, wid, addr): (wid, addr) for wid, addr in workers}
        results = []
        for future in futures:
            wid, addr = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append(PingResult(worker_id=wid, worker_address=addr, error=str(e)))
        return results

    def _ping_one(self, worker_id: WorkerId, address: str | None) -> PingResult:
        if not address:
            raise ProviderError(f"Worker {worker_id} has no address")
        stub = self.stub_factory.get_stub(address)
        response = stub.ping(worker_pb2.Worker.PingRequest())
        if not response.healthy:
            raise ProviderError(f"worker {worker_id} reported unhealthy: {response.health_error}")
        return PingResult(
            worker_id=worker_id,
            worker_address=address,
            resource_snapshot=response.resource_snapshot if response.resource_snapshot.ByteSize() > 0 else None,
            healthy=response.healthy,
            health_error=response.health_error,
        )

    def start_tasks(
        self,
        worker_id: WorkerId,
        address: str,
        tasks: list[job_pb2.RunTaskRequest],
    ) -> worker_pb2.Worker.StartTasksResponse:
        """Send StartTasks RPC to a worker."""
        stub = self.stub_factory.get_stub(address)
        request = worker_pb2.Worker.StartTasksRequest(tasks=tasks)
        return stub.start_tasks(request)

    def stop_tasks(
        self,
        worker_id: WorkerId,
        address: str,
        task_ids: list[str],
    ) -> None:
        """Send StopTasks RPC to a worker."""
        stub = self.stub_factory.get_stub(address)
        request = worker_pb2.Worker.StopTasksRequest(task_ids=task_ids)
        stub.stop_tasks(request)

    def poll_workers(
        self,
        running: dict[WorkerId, list[RunningTaskEntry]],
        worker_addresses: dict[WorkerId, str],
    ) -> list[tuple[WorkerId, list[TaskUpdate] | None, str | None]]:
        """Poll all workers for task state via PollTasks RPC in parallel.

        Returns a list of (worker_id, updates_or_none, error_or_none).
        """
        results: list[tuple[WorkerId, list[TaskUpdate] | None, str | None]] = []

        def _poll_one(wid: WorkerId) -> tuple[WorkerId, list[TaskUpdate] | None, str | None]:
            address = worker_addresses.get(wid)
            if not address:
                return (wid, None, f"Worker {wid} has no address")
            entries = running[wid]
            expected = [
                job_pb2.WorkerTaskStatus(
                    task_id=e.task_id.to_wire(),
                    attempt_id=e.attempt_id,
                )
                for e in entries
            ]
            stub = self.stub_factory.get_stub(address)
            request = worker_pb2.Worker.PollTasksRequest(expected_tasks=expected)
            response = stub.poll_tasks(request)
            return (wid, task_updates_from_proto(response.tasks), None)

        futures = {self._pool.submit(_poll_one, wid): wid for wid in running}
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                wid = futures[future]
                results.append((wid, None, str(e)))
        return results

    def close(self) -> None:
        self.stub_factory.close()
