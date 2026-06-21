# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RpcTaskBackend: a TaskBackend backed by worker daemons via Connect RPC.

The worker-daemon backend used by the GCP/TPU, CoreWeave-bare-metal, manual, and
local clusters. The Iris scheduler assigns task→worker; this backend fans the
per-worker Reconcile RPC out to the worker daemons, reports the raw observations
back to the controller, and surfaces the per-worker liveness it observed
(REACHED / UNREACHABLE) as health events the controller folds.
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
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.backend import (
    AutoscaleResult,
    BackendCapability,
    ProviderError,
    ReconcileResult,
    ScheduleInput,
    ScheduleResult,
    TaskTarget,
    plans_from_snapshot,
    run_scheduling_decision,
)
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.worker import WorkerReconcilePlan, WorkerReconcileResult
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerHealthEventKind
from iris.cluster.types import WorkerId
from iris.rpc import job_pb2, worker_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.worker_connect import WorkerServiceClient

logger = logging.getLogger(__name__)

# Per-worker RPC deadline for on-demand worker RPCs (profile_task, exec_in_container,
# get_process_status) and the cached stub's fallback timeout.
DEFAULT_WORKER_RPC_TIMEOUT = Duration.from_seconds(10.0)

# Tighter per-worker deadline for the reconcile fan-out: a hung worker can't gate
# the gather-joined round on the slow straggler, and a missed round never reaps a
# worker (the reconcile-failure threshold is dozens of rounds).
RECONCILE_RPC_TIMEOUT = Duration.from_seconds(3.0)

# Max concurrent in-flight per-worker RPCs in a fan-out (asyncio.Semaphore width).
# Kept >= fleet size so the whole fleet reconciles in one wave and a slow worker
# costs one RPC-timeout window per round, not one per wave.
RECONCILE_FANOUT_PARALLELISM = 512

# Generous deadline for an "unlimited" exec_in_container (negative timeout). Long
# enough for real interactive/debug commands, but not the old ~1-hour stall.
EXEC_IN_CONTAINER_MAX_TIMEOUT = Duration.from_seconds(900.0)

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
    """A worker-daemon :class:`~iris.cluster.controller.backend.TaskBackend`
    backed by async Connect RPCs.

    Each fan-out method spins up an asyncio event loop and dispatches the
    relevant RPC to each worker concurrently via `asyncio.gather`, capped at
    `parallelism` in-flight requests by a local semaphore. Cached stubs in
    the factory keep their pyqwest connection pools across rounds.
    """

    stub_factory: WorkerStubFactory
    parallelism: int = RECONCILE_FANOUT_PARALLELISM
    name: str = "worker"
    # The Iris autoscaler that provisions capacity for this backend. Attached by
    # the controller's main() after construction (mirrors set_log_sink); None for
    # clusters with no scale groups, where capacity calls are no-ops.
    autoscaler: Autoscaler | None = None
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset(
        {BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER}
    )
    # Stateless: holds no per-tick state, so one shared instance is reused
    # across scheduling cycles (mirrors the autoscaler's own Scheduler).
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)

    def attach_autoscaler(self, autoscaler: Autoscaler) -> None:
        """Attach the Iris autoscaler that provisions capacity for this backend."""
        self.autoscaler = autoscaler

    def schedule(self, snapshot: ScheduleInput) -> ScheduleResult:
        """Run the Iris scheduling decision pipeline over the snapshot.

        Reads the autoscaler's per-zone accelerator-capability map so the
        scheduler can inject ``availability:<variant>`` markers onto workers and
        confine a hard availability constraint to a capable zone. Clusters with no
        autoscaler pass an empty map: no worker gets an availability marker, so a
        job carrying an availability constraint there stays unschedulable (it has
        no zone that can satisfy it).
        """
        zone_capabilities = self.autoscaler.zone_capabilities() if self.autoscaler is not None else None
        reserved_view = self.autoscaler.reserved_pool_view() if self.autoscaler is not None else None
        return run_scheduling_decision(self._scheduler, snapshot, zone_capabilities, reserved_view)

    def reconcile(self, snapshot: ControlSnapshot) -> ReconcileResult:
        """Build per-worker plans, fan the Reconcile RPC out, observe liveness.

        Plans are built here (worker-daemon-specific protos) from the snapshot's
        reconcile rows + job specs. Each per-worker RPC carries the stub
        factory's deadline and the fan-out caps concurrency at ``parallelism``,
        so this returns in bounded time even when the whole fleet is hung. Each
        outcome yields a health event the controller folds:

        * a healthy response is REACHED;
        * an RPC error/timeout is UNREACHABLE, and the (likely broken) stub is
          evicted as I/O hygiene;
        * a response that self-reports unhealthy (e.g. failed disk) is also
          UNREACHABLE so the worker is eventually reaped, but the connection is
          fine so the stub is kept.

        The backend never decides a worker is dead — it only observes.
        """
        plans = plans_from_snapshot(snapshot)

        async def _one(sem: asyncio.Semaphore, plan: WorkerReconcilePlan) -> WorkerReconcileResult:
            return await self._reconcile_one(sem, plan, snapshot.worker_addresses[plan.worker_id])

        results = _fan_out(plans, self.parallelism, _one)

        worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]] = list(zip(plans, results, strict=True))
        health_events: list[WorkerHealthEvent] = []
        for plan, result in worker_results:
            address = snapshot.worker_addresses[plan.worker_id]
            if result.error is not None:
                health_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
                self.stub_factory.evict(address)
            elif result.responder_worker_id is not None and result.responder_worker_id != str(plan.worker_id):
                # Misrouted reconcile: a *different* live worker answered at this
                # address. GCP recycles a deleted worker's internal IP onto a new
                # VM, so the controller's stale address for the dead worker now
                # points at someone else. Counting the impostor's healthy reply as
                # REACHED would resurrect the dead worker (reset its failures to 0)
                # and keep it schedulable — a black hole that accepts and kills
                # every task. Treat it as UNREACHABLE so the stale worker accrues
                # failures and is reaped, and drop the impostor's stub.
                logger.warning(
                    "Reconcile for worker %s at %s was answered by %s (recycled address); marking unreachable",
                    plan.worker_id,
                    address,
                    result.responder_worker_id,
                )
                health_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
                self.stub_factory.evict(address)
            elif not result.self_healthy:
                health_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
            else:
                health_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.REACHED))
        return ReconcileResult(worker_results=worker_results, health_events=health_events)

    def autoscale(
        self,
        snapshot: ControlSnapshot,
        residual_demand: list[DemandEntry],
        dead_workers: list[WorkerId],
        drain_workers: Sequence[WorkerId] = (),
    ) -> AutoscaleResult:
        """Tear down dead/drained workers' slices, or run one provisioning cycle.

        With ``dead_workers`` set the autoscaler terminates their slices and
        returns the dead workers plus their healthy siblings as
        ``removed_workers`` (no provisioning this call); the cached stub for
        every torn-down worker is evicted, since none will be reconciled again.
        With only ``drain_workers`` set (cross-variant reserved-pool preemption),
        it drains those slices through the intentional-drain path — which does
        not feed the churn detector — and returns them the same way. Otherwise it
        runs a refresh + probe_health + update cycle against ``residual_demand``.
        The DB reads (worker status, demand) are done by the controller and
        handed in; this only drives the in-memory autoscaler.
        """
        if self.autoscaler is None:
            return AutoscaleResult()
        if dead_workers:
            siblings = self.autoscaler.terminate_slices_for_workers([str(wid) for wid in dead_workers])
            return self._evict_and_result(snapshot, list(dead_workers), siblings)
        if drain_workers:
            siblings = self.autoscaler.drain_slices_for_workers([str(wid) for wid in drain_workers])
            return self._evict_and_result(snapshot, list(drain_workers), siblings)
        self.autoscaler.refresh(snapshot.worker_status_map)
        self.autoscaler.probe_health()
        self.autoscaler.update(residual_demand)
        return AutoscaleResult(autoscaler_state=self.autoscaler.persistable_state())

    def _evict_and_result(
        self,
        snapshot: ControlSnapshot,
        primary_workers: list[WorkerId],
        sibling_ids: list[str],
    ) -> AutoscaleResult:
        """Evict torn-down workers' stubs and build the removal result.

        Shared by the dead-worker and intentional-drain teardown branches: both
        return the primary workers plus their healthy siblings as
        ``removed_workers`` and evict each cached stub, since none of these
        workers will be reconciled again.
        """
        assert self.autoscaler is not None
        removed = primary_workers + [WorkerId(wid) for wid in sibling_ids]
        for wid in removed:
            if address := snapshot.worker_addresses.get(wid):
                self.stub_factory.evict(address)
        return AutoscaleResult(removed_workers=removed, autoscaler_state=self.autoscaler.persistable_state())

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
        # Negative timeout means "no caller limit"; still bound the RPC deadline
        # with a generous cap so a hung exec can't pin the handler indefinitely.
        if timeout_seconds < 0:
            rpc_timeout_ms = EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms()
        else:
            rpc_timeout_ms = (timeout_seconds + 5) * 1000
        return asyncio.run(stub.exec_in_container(request, timeout_ms=rpc_timeout_ms))

    async def _reconcile_one(
        self,
        sem: asyncio.Semaphore,
        plan: WorkerReconcilePlan,
        address: str,
    ) -> WorkerReconcileResult:
        """Issue a single Reconcile RPC to one worker under the shared semaphore."""
        async with sem:
            try:
                if rule := chaos("controller.reconcile"):
                    await asyncio.sleep(rule.delay_seconds)
                    raise ProviderError("chaos: controller.reconcile")
                stub = self.stub_factory.get_stub(address)
                response = await asyncio.wait_for(
                    stub.reconcile(plan.request), timeout=RECONCILE_RPC_TIMEOUT.to_seconds()
                )
                return WorkerReconcileResult(
                    worker_id=plan.worker_id,
                    observations=list(response.observed),
                    error=None,
                    self_healthy=response.health.healthy,
                    responder_worker_id=response.worker_id or None,
                )
            except Exception as e:
                return WorkerReconcileResult(worker_id=plan.worker_id, observations=[], error=str(e) or type(e).__name__)

    def close(self) -> None:
        if self.autoscaler is not None:
            self.autoscaler.shutdown()
        self.stub_factory.close()
