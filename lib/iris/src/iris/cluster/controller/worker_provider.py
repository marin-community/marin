# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerProvider: TaskProvider backed by worker daemons via heartbeat RPC."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import sleep
from typing import Protocol

from iris.chaos import chaos
from iris.cluster.controller.provider import ProviderError
from iris.cluster.controller.transitions import (
    DispatchBatch,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.log_store import PROCESS_LOG_KEY, task_log_key
from iris.cluster.types import JobName, TaskAttempt, WorkerId
from iris.rpc import cluster_pb2, logging_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.time_utils import Duration

logger = logging.getLogger(__name__)


class WorkerStubFactory(Protocol):
    """Factory for getting cached worker RPC stubs."""

    def get_stub(self, address: str) -> WorkerServiceClientSync: ...
    def evict(self, address: str) -> None: ...
    def close(self) -> None: ...


class RpcWorkerStubFactory:
    """Caches WorkerServiceClientSync stubs by address so each worker gets
    one persistent httpx.Client instead of a new one per RPC."""

    def __init__(self, timeout: Duration = Duration.from_seconds(5.0)) -> None:
        self._timeout = timeout
        self._stubs: dict[str, WorkerServiceClientSync] = {}
        self._lock = threading.Lock()

    def get_stub(self, address: str) -> WorkerServiceClientSync:
        with self._lock:
            stub = self._stubs.get(address)
            if stub is None:
                stub = WorkerServiceClientSync(
                    address=f"http://{address}",
                    timeout_ms=self._timeout.to_ms(),
                )
                self._stubs[address] = stub
            return stub

    def evict(self, address: str) -> None:
        with self._lock:
            stub = self._stubs.pop(address, None)
        if stub is not None:
            stub.close()

    def close(self) -> None:
        with self._lock:
            stubs = list(self._stubs.values())
            self._stubs.clear()
        for stub in stubs:
            stub.close()


def _apply_request_from_response(
    worker_id: WorkerId,
    response: cluster_pb2.HeartbeatResponse,
) -> HeartbeatApplyRequest:
    """Convert a HeartbeatResponse proto to a HeartbeatApplyRequest."""
    updates: list[TaskUpdate] = []
    for entry in response.tasks:
        if entry.state in (cluster_pb2.TASK_STATE_UNSPECIFIED, cluster_pb2.TASK_STATE_PENDING):
            continue
        updates.append(
            TaskUpdate(
                task_id=JobName.from_wire(entry.task_id),
                attempt_id=entry.attempt_id,
                new_state=entry.state,
                error=entry.error or None,
                exit_code=entry.exit_code if entry.HasField("exit_code") else None,
                resource_usage=entry.resource_usage if entry.resource_usage.ByteSize() > 0 else None,
                log_entries=list(entry.log_entries),
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
    """TaskProvider backed by worker daemons via heartbeat RPC.

    Drop-in replacement for the controller's _do_heartbeat_rpc path. Uses a
    persistent ThreadPoolExecutor for parallel heartbeat dispatch.
    """

    stub_factory: WorkerStubFactory
    parallelism: int = 32
    _pool: ThreadPoolExecutor = field(init=False)

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=self.parallelism)

    def sync(
        self,
        batches: list[DispatchBatch],
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        if not batches:
            return []
        results: list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]] = []
        futures = {self._pool.submit(self._heartbeat_one, b): b for b in batches}
        for future in futures:
            batch = futures[future]
            try:
                apply_req = future.result()
                results.append((batch, apply_req, None))
            except Exception as e:
                results.append((batch, None, str(e)))
        return results

    def _heartbeat_one(self, batch: DispatchBatch) -> HeartbeatApplyRequest:
        """Send heartbeat RPC to one worker and return the apply request."""
        if rule := chaos("controller.heartbeat"):
            sleep(rule.delay_seconds)
            raise ProviderError("chaos: heartbeat unavailable")

        if not batch.worker_address:
            raise ProviderError(f"Worker {batch.worker_id} has no address for heartbeat")

        stub = self.stub_factory.get_stub(batch.worker_address)

        expected_tasks = []
        for entry in batch.running_tasks:
            if rule := chaos("controller.heartbeat.iteration"):
                sleep(rule.delay_seconds)
            expected_tasks.append(
                cluster_pb2.Controller.WorkerTaskStatus(
                    task_id=entry.task_id.to_wire(),
                    attempt_id=entry.attempt_id,
                )
            )
        request = cluster_pb2.HeartbeatRequest(
            tasks_to_run=batch.tasks_to_run,
            tasks_to_kill=batch.tasks_to_kill,
            expected_tasks=expected_tasks,
        )
        response = stub.heartbeat(request)

        if not response.worker_healthy:
            health_error = response.health_error or "worker reported unhealthy"
            raise ProviderError(f"worker {batch.worker_id} reported unhealthy: {health_error}")

        return _apply_request_from_response(batch.worker_id, response)

    def fetch_live_logs(
        self,
        worker_id: WorkerId,
        address: str | None,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        if not address:
            raise ProviderError(f"Worker {worker_id} has no address")
        stub = self.stub_factory.get_stub(address)
        resp = stub.fetch_logs(
            cluster_pb2.FetchLogsRequest(
                source=task_log_key(TaskAttempt(task_id=JobName.from_wire(task_id), attempt_id=attempt_id)),
                cursor=cursor,
                max_lines=max_lines,
            ),
            timeout_ms=10000,
        )
        return list(resp.entries), resp.cursor

    def fetch_process_logs(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: cluster_pb2.FetchLogsRequest,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        if not address:
            raise ProviderError(f"Worker {worker_id} has no address")
        stub = self.stub_factory.get_stub(address)
        # Forward the full request but override source to the worker's process log key.
        forwarded = cluster_pb2.FetchLogsRequest(
            source=PROCESS_LOG_KEY,
            cursor=request.cursor,
            max_lines=request.max_lines,
            since_ms=request.since_ms,
            substring=request.substring,
            tail=request.tail,
            min_level=request.min_level,
        )
        resp = stub.fetch_logs(forwarded, timeout_ms=10000)
        return list(resp.entries), resp.cursor

    def get_process_status(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: cluster_pb2.GetProcessStatusRequest,
    ) -> cluster_pb2.GetProcessStatusResponse:
        if not address:
            raise ProviderError(f"Worker {worker_id} has no address")
        stub = self.stub_factory.get_stub(address)
        # Forward with target cleared — the worker serves its own process status.
        forwarded = cluster_pb2.GetProcessStatusRequest(
            max_log_lines=request.max_log_lines,
            log_substring=request.log_substring,
            min_log_level=request.min_log_level,
        )
        return stub.get_process_status(forwarded, timeout_ms=10000)

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        if address:
            self.stub_factory.evict(address)

    def profile_task(
        self,
        address: str,
        request: cluster_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> cluster_pb2.ProfileTaskResponse:
        stub = self.stub_factory.get_stub(address)
        return stub.profile_task(request, timeout_ms=timeout_ms)

    def exec_in_container(
        self,
        address: str,
        request: cluster_pb2.Worker.ExecInContainerRequest,
    ) -> cluster_pb2.Worker.ExecInContainerResponse:
        stub = self.stub_factory.get_stub(address)
        return stub.exec_in_container(request, timeout_ms=65000)

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        self.stub_factory.close()
