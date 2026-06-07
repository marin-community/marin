# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RpcTaskBackend: a TaskBackend backed by worker daemons via Connect RPC.

The ``PlacementOwner.IRIS_CONTROLLER`` backend used by the GCP/TPU, CoreWeave-bare-metal,
manual, and local clusters. The Iris scheduler assigns task→worker; this
backend fans the per-worker Reconcile RPC out to the worker daemons and reports
the raw observations back to the controller.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, TypeVar

from finelog.client.log_client import Table
from finelog.types import LogWriterProtocol
from rigging.timing import Duration

from iris.chaos import chaos
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.backend import (
    BackendReconcileInput,
    BackendReconcileResult,
    CapacityInput,
    CapacityResult,
    PingResult,
    PlacementOwner,
    ProviderError,
    ScheduleInput,
    ScheduleResult,
    TaskTarget,
    WorkersFailedResult,
    run_scheduling_decision,
)
from iris.cluster.controller.reconcile.worker import ReconcileResult, WorkerReconcilePlan
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.types import WorkerId
from iris.rpc import job_pb2, worker_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.worker_connect import WorkerServiceClient

logger = logging.getLogger(__name__)

DEFAULT_WORKER_RPC_TIMEOUT = Duration.from_seconds(10.0)

_T = TypeVar("_T")
_R = TypeVar("_R")


def _fan_out(
    items: Sequence[_T],
    parallelism: int,
    run_one: Callable[[asyncio.Semaphore, _T], Awaitable[_R]],
) -> list[_R]:
    """Run ``run_one`` over every item concurrently, capped at ``parallelism``.

    Each coroutine receives the shared semaphore and is responsible for
    acquiring it; ``gather`` preserves input order in the returned list.
    """
    if not items:
        return []

    async def _run() -> list[_R]:
        sem = asyncio.Semaphore(parallelism)
        return await asyncio.gather(*(run_one(sem, item) for item in items))

    return asyncio.run(_run())


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
class RpcTaskBackend:
    """``PlacementOwner.IRIS_CONTROLLER`` :class:`~iris.cluster.controller.backend.TaskBackend`
    backed by worker daemons via async Connect RPCs.

    Each public method spins up an asyncio event loop and dispatches the
    relevant RPC to each worker concurrently via `asyncio.gather`, capped at
    `parallelism` in-flight requests by a local semaphore. Cached stubs in
    the factory keep their pyqwest connection pools across rounds.
    """

    stub_factory: WorkerStubFactory
    parallelism: int = 128
    name: str = "worker"
    # The Iris autoscaler that provisions capacity for this backend. Attached by
    # the controller's main() after construction (mirrors set_log_sink); None for
    # clusters with no scale groups, where capacity calls are no-ops.
    autoscaler: Autoscaler | None = None
    placement: ClassVar[PlacementOwner] = PlacementOwner.IRIS_CONTROLLER
    manages_capacity: ClassVar[bool] = False
    # Stateless: holds no per-tick state, so one shared instance is reused
    # across scheduling cycles (mirrors the autoscaler's own Scheduler).
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)

    def attach_autoscaler(self, autoscaler: Autoscaler) -> None:
        """Attach the Iris autoscaler that provisions capacity for this backend."""
        self.autoscaler = autoscaler

    def reconcile(self, batch: BackendReconcileInput) -> BackendReconcileResult:
        """Fan the Reconcile RPC out across all planned workers concurrently."""

        async def _one(sem: asyncio.Semaphore, plan: WorkerReconcilePlan) -> ReconcileResult:
            return await self._reconcile_one(sem, plan, batch.worker_addresses[plan.worker_id])

        results = _fan_out(batch.plans, self.parallelism, _one)
        return BackendReconcileResult(worker_results=results)

    def schedule(self, snapshot: ScheduleInput) -> ScheduleResult:
        """Run the Iris scheduling decision pipeline over the snapshot."""
        return run_scheduling_decision(self._scheduler, snapshot)

    def manage_capacity(self, snapshot: CapacityInput) -> CapacityResult:
        """Run one autoscaler cycle (refresh + probe_health + update) over the snapshot.

        The DB reads (worker status, demand) are done by the controller and
        handed in via ``snapshot``; this drives the in-memory autoscaler and
        returns its tracked state for the controller to persist.
        """
        if self.autoscaler is None:
            return CapacityResult()
        self.autoscaler.refresh(snapshot.worker_status_map)
        self.autoscaler.probe_health()
        self.autoscaler.update(snapshot.demand_entries)
        return CapacityResult(state=self.autoscaler.persistable_state())

    def on_workers_failed(self, worker_ids: list[WorkerId]) -> WorkersFailedResult:
        """Terminate the failed workers' slices and return their healthy siblings."""
        if self.autoscaler is None:
            return WorkersFailedResult()
        siblings = self.autoscaler.terminate_slices_for_workers([str(wid) for wid in worker_ids])
        return WorkersFailedResult(
            sibling_worker_ids=[WorkerId(wid) for wid in siblings],
            state=self.autoscaler.persistable_state(),
        )

    def get_process_status(
        self,
        target: TaskTarget,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        if not target.address:
            raise ProviderError(f"Worker {target.worker_id} has no address")
        stub = self.stub_factory.get_stub(target.address)
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

    def set_log_sink(
        self,
        log_client: LogWriterProtocol,
        task_stats_table: Table,
        profile_table: Table,
    ) -> None:
        """No-op: worker daemons write their own log/resource/profile rows."""

    def profile_task(
        self,
        target: TaskTarget,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        if not target.address:
            raise ProviderError(f"Worker {target.worker_id} has no address")
        stub = self.stub_factory.get_stub(target.address)
        return asyncio.run(stub.profile_task(request, timeout_ms=timeout_ms))

    def exec_in_container(
        self,
        target: TaskTarget,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        if not target.address:
            raise ProviderError(f"Worker {target.worker_id} has no address")
        stub = self.stub_factory.get_stub(target.address)
        # Negative timeout means no limit; use a large RPC deadline (1 hour)
        if timeout_seconds < 0:
            rpc_timeout_ms = 3_600_000
        else:
            rpc_timeout_ms = (timeout_seconds + 5) * 1000
        return asyncio.run(stub.exec_in_container(request, timeout_ms=rpc_timeout_ms))

    def ping_workers(self, workers: list[tuple[WorkerId, str | None]]) -> list[PingResult]:
        """Send Ping RPCs to all workers concurrently. Returns per-worker results."""

        async def _one(sem: asyncio.Semaphore, target: tuple[WorkerId, str | None]) -> PingResult:
            wid, addr = target
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

        return _fan_out(workers, self.parallelism, _one)

    async def _reconcile_one(
        self,
        sem: asyncio.Semaphore,
        plan: WorkerReconcilePlan,
        address: str,
    ) -> ReconcileResult:
        """Issue a single Reconcile RPC to one worker under the shared semaphore."""
        async with sem:
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

    def close(self) -> None:
        if self.autoscaler is not None:
            self.autoscaler.shutdown()
        self.stub_factory.close()
