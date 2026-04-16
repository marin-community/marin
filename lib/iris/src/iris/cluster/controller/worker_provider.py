# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerProvider: TaskProvider backed by worker daemons via heartbeat RPC."""

import asyncio
import logging
import threading
from collections.abc import Coroutine
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Protocol, TypeVar

from iris.chaos import chaos
from iris.cluster.controller.provider import ProviderError
from iris.cluster.controller.transitions import (
    DispatchBatch,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from iris.rpc import worker_pb2
from iris.rpc.worker_connect import WorkerServiceClient
from rigging.timing import Duration

logger = logging.getLogger(__name__)

DEFAULT_WORKER_RPC_TIMEOUT = Duration.from_seconds(10.0)
_SLOW_HEARTBEAT_RPC_LOG_THRESHOLD_MS = 5_000

T = TypeVar("T")


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


class _AsyncLoopThread:
    """Runs an asyncio event loop on a dedicated daemon thread.

    Sync callers submit coroutines via `run()` / `submit()`; the loop hosts
    the long-lived async httpx clients so their connection pools survive
    across heartbeat rounds.
    """

    def __init__(self, name: str = "worker-provider-asyncio") -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self) -> None:
        if not self._loop.is_running():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)


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
    updates: list[TaskUpdate] = []
    for entry in response.tasks:
        if entry.state in (job_pb2.TASK_STATE_UNSPECIFIED, job_pb2.TASK_STATE_PENDING):
            continue
        updates.append(
            TaskUpdate(
                task_id=JobName.from_wire(entry.task_id),
                attempt_id=entry.attempt_id,
                new_state=entry.state,
                error=entry.error or None,
                exit_code=entry.exit_code if entry.HasField("exit_code") else None,
                resource_usage=entry.resource_usage if entry.resource_usage.ByteSize() > 0 else None,
                container_id=entry.container_id or None,
            )
        )
    return HeartbeatApplyRequest(
        worker_id=worker_id,
        worker_resource_snapshot=(response.resource_snapshot if response.resource_snapshot.ByteSize() > 0 else None),
        updates=updates,
    )


@dataclass
class WorkerProvider:
    """TaskProvider backed by worker daemons via async heartbeat RPC.

    Runs an asyncio event loop on a dedicated thread and dispatches
    per-worker heartbeat RPCs concurrently via `asyncio.gather`, capped at
    `parallelism` concurrent in-flight requests by a semaphore.
    """

    stub_factory: WorkerStubFactory
    parallelism: int = 128
    _loop_thread: _AsyncLoopThread = field(init=False)
    _semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self._loop_thread = _AsyncLoopThread()
        self._semaphore = self._loop_thread.run(self._make_semaphore())

    async def _make_semaphore(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(self.parallelism)

    def sync(
        self,
        batches: list[DispatchBatch],
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        if not batches:
            return []
        return self._loop_thread.run(self._sync_all(batches))

    async def _sync_all(
        self,
        batches: list[DispatchBatch],
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        coros = [self._heartbeat_one_safe(b) for b in batches]
        return await asyncio.gather(*coros)

    async def _heartbeat_one_safe(
        self,
        batch: DispatchBatch,
    ) -> tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]:
        try:
            apply_req = await self._heartbeat_one(batch)
            return (batch, apply_req, None)
        except Exception as e:
            return (batch, None, str(e))

    async def _heartbeat_one(self, batch: DispatchBatch) -> HeartbeatApplyRequest:
        """Send heartbeat RPC to one worker and return the apply request."""
        async with self._semaphore:
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
        return self._loop_thread.run(stub.get_process_status(forwarded, timeout_ms=10000))

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
        return self._loop_thread.run(stub.profile_task(request, timeout_ms=timeout_ms))

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
        return self._loop_thread.run(stub.exec_in_container(request, timeout_ms=rpc_timeout_ms))

    def close(self) -> None:
        self.stub_factory.close()
        self._loop_thread.close()
