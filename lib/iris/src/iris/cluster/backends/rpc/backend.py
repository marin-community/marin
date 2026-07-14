# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RpcTaskBackend: a TaskBackend backed by worker daemons via Connect RPC.

The worker-daemon backend used by the GCP/TPU, CoreWeave-bare-metal, manual, and
local clusters. The Iris scheduler assigns task→worker; this backend fans the
per-worker Reconcile RPC out to the worker daemons, resolves the observations
into task ``effects`` from its own read snapshot, and folds the per-worker
liveness it observed (REACHED / UNREACHABLE / kernel-derived BUILD_FAILED)
through the liveness tracker it constructs and owns (``self.health``, holding
only the workers in this backend's scale groups). The workers its fold reaps are
stashed and torn down by ``run_teardown`` after the controller commits the
effects, so no worker identity crosses the reconcile result boundary.
"""

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, TypeVar

from rigging.timing import Duration, Timestamp

from iris.chaos import chaos
from iris.cluster.constraints import DeviceType
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.status import overlay_worker_usability
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendCapability,
    BackendRuntime,
    DeviceCapacity,
    ProviderError,
    ReconcileRequest,
    ReconcileResult,
    ScheduleInput,
    ScheduleRequest,
    ScheduleResult,
    TaskTarget,
    assemble_scheduling_context,
    plans_from_snapshot,
    run_scheduling_decision,
)
from iris.cluster.controller.backend_store import BackendWorkerStore, DbBackendWorkerStore
from iris.cluster.controller.ops.worker import apply_reconcile
from iris.cluster.controller.reconcile.worker import WorkerReconcilePlan, WorkerReconcileResult
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.controller.worker_health import (
    DEFAULT_UNREACHABLE_GRACE,
    WorkerHealthEvent,
    WorkerHealthEventKind,
    WorkerHealthTracker,
)
from iris.cluster.types import WellKnownAttribute, WorkerId
from iris.rpc import controller_pb2, job_pb2, vm_pb2, worker_pb2
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

# Failure reason stamped on a worker the reconcile fold reaped (drained by
# ``run_teardown``).
WORKER_RECONCILE_TEARDOWN_REASON = "worker reconcile failure threshold exceeded"

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


@dataclass(frozen=True)
class FleetObservation:
    """One reconcile fan-out's raw outcome.

    The per-worker results paired with their plans, and the transport liveness
    each yielded (REACHED / UNREACHABLE). :meth:`RpcTaskBackend.reconcile` resolves
    the results into effects and folds the liveness.
    """

    worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]]
    transport_events: list[WorkerHealthEvent]


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
    # The id the controller assigned this backend, learned in ``bind_runtime``. The
    # backend stamps it onto the autoscaler groups it authors in ``status`` /
    # ``autoscaler_status``, so the controller reads those verbatim.
    backend_id: str = field(default="", init=False)
    # The Iris autoscaler that provisions capacity for this backend, passed by the
    # composer at construction after it builds the autoscaler from the provider
    # bundle; None for clusters with no scale groups, where capacity calls are no-ops.
    autoscaler: Autoscaler | None = None
    # This backend's worker store, built in ``bind_runtime`` from the controller-owned
    # ``BackendRuntime`` joined with this backend's own health tracker. The backend
    # reads its own workers and reaps its dead ones through this; the controller never
    # hands it a worker snapshot or a raw DB.
    _store: BackendWorkerStore | None = field(default=None, init=False, repr=False)
    # Wall-clock window a worker may stay continuously unreachable before this
    # backend's tracker reaps it; configures the WorkerHealthTracker built below.
    unreachable_grace: Duration = field(default_factory=lambda: DEFAULT_UNREACHABLE_GRACE)
    # Static routing metadata the meta-scheduler reads. ``advertised`` expands into
    # routing posting lists.
    advertised: dict[str, set[str]] = field(default_factory=dict)
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset(
        {BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER}
    )
    # This backend's liveness tracker, constructed and owned here, holding only the
    # workers in this backend's scale groups. The backend folds (reconcile) and
    # forgets (teardown) through it; the controller reads it for its
    # Fleet/exec/capacity/prune paths and routes a registering worker's liveness to
    # it by scale group.
    health: WorkerHealthTracker = field(init=False, repr=False)
    # One shared scheduler instance reused across cycles; per-tick worker state
    # comes from ``_store``.
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)
    # Workers this backend's reconcile fold reaped, awaiting teardown. ``reconcile``
    # appends; ``run_teardown`` drains post-commit. Kept off the reconcile result so
    # no worker identity crosses that boundary back to the controller.
    _pending_dead: list[WorkerId] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.health = WorkerHealthTracker(unreachable_grace=self.unreachable_grace)

    def bind_runtime(self, runtime: BackendRuntime) -> None:
        """Build this backend's worker store from runtime and the backend's own
        liveness tracker and autoscale callback."""
        self.backend_id = runtime.backend_id
        self._store = DbBackendWorkerStore(
            db=runtime.db,
            owns_scale_group=runtime.owns_scale_group,
            health=self.health,
            defaults=runtime.budget_defaults,
            autoscale=self.autoscale,
        )

    def seed_liveness(self) -> None:
        """Seed this backend's persisted workers as healthy so the scheduler sees them.

        Run at controller start and after a DB reopen (checkpoint restore): each
        owned worker is heartbeat-seeded so it comes up ACTIVE, then accrues
        failures through the reconcile fold and is reaped once over threshold.
        """
        assert self._store is not None, "RpcTaskBackend.seed_liveness called before worker store attached"
        worker_ids = self._store.owned_worker_ids()
        if worker_ids:
            self.health.heartbeat(worker_ids, Timestamp.now().epoch_ms())

    def advertised_attributes(self) -> dict[str, set[str]]:
        return self.advertised

    def configure_routing(self, advertised: dict[str, set[str]]) -> None:
        self.advertised = advertised

    def resource_capacity(self) -> dict[str, DeviceCapacity] | None:
        """Free and total GPU chips a peer could schedule onto, keyed by lowercased device-variant.

        Counts only capacity the scheduler would actually place onto — chips on
        live, schedulable workers — so the advertised numbers match what a handoff
        can use. v1 is GPU-only; TPU-slice availability is a documented follow-up.
        Always a dict (empty = authoritative "nothing free"), never ``None``: a
        worker-daemon backend always supplies the metric."""
        assert self._store is not None, "RpcTaskBackend.resource_capacity called before worker store attached"
        capacity: dict[str, DeviceCapacity] = {}
        for worker in self._store.scheduling_inputs().workers:
            device_type = worker.attributes.get(WellKnownAttribute.DEVICE_TYPE)
            variant = worker.attributes.get(WellKnownAttribute.DEVICE_VARIANT)
            if device_type is None or variant is None or str(device_type.value) != DeviceType.GPU.value:
                continue
            token = str(variant.value).strip().lower()
            prior = capacity.get(token, DeviceCapacity(free=0, total=0))
            capacity[token] = DeviceCapacity(
                free=prior.free + max(0, worker.total_gpu_count - worker.committed_gpu_count),
                total=prior.total + worker.total_gpu_count,
            )
        return capacity

    def autoscaler_status(self) -> vm_pb2.AutoscalerStatus:
        """Author this backend's autoscaler status from the state it owns.

        Tags each group with this backend's id, then overlays every VM with the
        usability/running-task/capacity verdict from this backend's own liveness
        tracker plus the running-task rows the store reads for those VMs.
        """
        assert self._store is not None, "RpcTaskBackend.autoscaler_status called before worker store attached"
        status = self.autoscaler.get_status() if self.autoscaler is not None else vm_pb2.AutoscalerStatus()
        for group in status.groups:
            group.backend_id = self.backend_id
        usability_by_id = {str(worker_id): live.usability for worker_id, live in self.health.all().items()}
        vm_ids = {WorkerId(vm.vm_id) for group in status.groups for s in group.slices for vm in s.vms if vm.vm_id}
        overlay_worker_usability(status, usability_by_id, self._store.running_tasks(vm_ids))
        return status

    def status(self) -> controller_pb2.Controller.BackendStatus:
        """Author the full ``worker`` status variant from this backend's own state:
        the health counts from its liveness tracker around its :meth:`autoscaler_status`.
        The controller reads the result verbatim and overlays nothing.
        """
        liveness = self.health.all()
        return controller_pb2.Controller.BackendStatus(
            worker=controller_pb2.Controller.WorkerFleetDetail(
                autoscaler=self.autoscaler_status(),
                total_worker_count=len(liveness),
                healthy_worker_count=sum(1 for live in liveness.values() if live.healthy),
            )
        )

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Assemble this backend's scheduling context and run the Iris pipeline.

        The routed pending tasks + budgets come from ``request``; the workers,
        building counts and running attempts come from this backend's own worker
        store. The autoscaler's per-zone accelerator-capability map injects
        ``availability:<variant>`` markers onto workers so a hard availability
        constraint is confined to a capable zone; clusters with no autoscaler pass
        an empty map, so a job carrying an availability constraint there stays
        unschedulable (no zone can satisfy it).
        """
        assert self._store is not None, "RpcTaskBackend.schedule called before worker store attached"
        context = assemble_scheduling_context(self._store.scheduling_inputs(), request)
        zone_capabilities = self.autoscaler.zone_capabilities() if self.autoscaler is not None else None
        return run_scheduling_decision(
            self._scheduler,
            ScheduleInput(
                context=context,
                max_tasks_per_job_per_cycle=request.max_tasks_per_job_per_cycle,
                trace=request.trace,
            ),
            zone_capabilities,
        )

    def _observe_fleet(self) -> "FleetObservation":
        """Source this backend's placement, fan the Reconcile RPC out, classify liveness.

        The reconcile snapshot (worker addresses + reconcile rows + job specs) comes
        from this backend's own worker store. Each per-worker RPC carries the stub
        factory's deadline and the fan-out caps concurrency at
        ``parallelism``, so this returns in bounded time even when the whole fleet
        is hung. Each outcome yields a transport liveness signal:

        * a healthy response is REACHED;
        * an RPC error/timeout is UNREACHABLE, and the (likely broken) stub is
          evicted as I/O hygiene;
        * a response that self-reports unhealthy (e.g. failed disk) is also
          UNREACHABLE so the worker is eventually reaped, but the connection is
          fine so the stub is kept.

        Pure observation — it never decides a worker dead; :meth:`reconcile` folds
        these signals into liveness.
        """
        assert self._store is not None, "RpcTaskBackend.reconcile called before worker store attached"
        snapshot = self._store.reconcile_snapshot()
        plans = plans_from_snapshot(snapshot)

        async def _one(sem: asyncio.Semaphore, plan: WorkerReconcilePlan) -> WorkerReconcileResult:
            return await self._reconcile_one(sem, plan, snapshot.worker_addresses[plan.worker_id])

        results = _fan_out(plans, self.parallelism, _one)

        worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]] = list(zip(plans, results, strict=True))
        transport_events: list[WorkerHealthEvent] = []
        for plan, result in worker_results:
            address = snapshot.worker_addresses[plan.worker_id]
            if result.error is not None:
                transport_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
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
                transport_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
                self.stub_factory.evict(address)
            elif not result.self_healthy:
                transport_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
            else:
                transport_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.REACHED))
        return FleetObservation(worker_results=worker_results, transport_events=transport_events)

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        """Observe the fleet, resolve its observations into effects, fold liveness.

        ``request`` (the cluster-view dispatch drain) is unused — a worker-daemon
        backend sources its own placement. :meth:`_observe_fleet` fans the Reconcile
        RPC out; this resolves the observations into task ``effects`` and folds the
        liveness it observed (transport signals plus kernel-derived BUILD_FAILED)
        through the shared ``WorkerHealthTracker``. The reaped workers are stashed
        for :meth:`run_teardown`; only the committable ``effects`` are returned.
        """
        assert self._store is not None, "RpcTaskBackend.reconcile called before worker store attached"
        observation = self._observe_fleet()

        # Fold transport events first, then the kernel's BUILD_FAILED; both go
        # through the SAME shared tracker reached via the worker store, so the
        # startup seed and reopen hook are preserved.
        now = Timestamp.now()
        effects = apply_reconcile(self._store, observation.worker_results, now=now)
        events = observation.transport_events + [
            WorkerHealthEvent(wid, WorkerHealthEventKind.BUILD_FAILED) for wid in effects.health.build_failed
        ]
        self._pending_dead.extend(self.health.apply(events, now_ms=now.epoch_ms()))
        return ReconcileResult(effects=effects)

    def run_teardown(self) -> None:
        """Tear down the workers this tick's reconcile fold reaped.

        Drains the stash and runs the same fail → slice-and-sibling teardown →
        forget sequence over a fresh snapshot. The controller calls this after it
        commits the reconcile effects, so a just-finalized attempt is already
        terminal and skipped. Empty between reaps, so most ticks are a no-op.
        """
        dead = self._pending_dead
        self._pending_dead = []
        self.teardown(dead, reason=WORKER_RECONCILE_TEARDOWN_REASON)

    def teardown(self, dead_workers: list[WorkerId], *, reason: str) -> None:
        """Fail ``dead_workers``, reap their slices and siblings, and forget them."""
        assert self._store is not None, "RpcTaskBackend.teardown called before worker store attached"
        self._store.reap_workers(dead_workers, reason=reason)

    def prune_dead_workers(self, *, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
        """Garbage-collect this backend's stale DEAD workers through its worker store."""
        assert self._store is not None, "RpcTaskBackend.prune_dead_workers called before worker store attached"
        return self._store.prune_dead_workers(cutoff_ms=cutoff_ms, stop_event=stop_event, pause=pause)

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        """Tear down dead workers' slices, or run one provisioning cycle.

        With ``request.dead_workers`` set the autoscaler terminates their slices
        and returns the dead workers plus their healthy siblings as
        ``removed_workers`` (no provisioning this call). Stubs for the removed
        workers are not evicted here: a dead worker's stub was already dropped as
        it accrued UNREACHABLE reconcile rounds, and a healthy sibling's stub
        self-evicts on the next reconcile RPC once its slice is gone. Otherwise it
        runs a refresh + probe_health + update cycle against
        ``request.residual_demand``, reading its own worker status.
        """
        if self.autoscaler is None:
            return AutoscaleResult()
        if request.dead_workers:
            siblings = self.autoscaler.terminate_slices_for_workers([str(wid) for wid in request.dead_workers])
            removed = list(request.dead_workers) + [WorkerId(wid) for wid in siblings]
            return AutoscaleResult(removed_workers=removed, autoscaler_state=self.autoscaler.persistable_state())
        assert self._store is not None, "RpcTaskBackend.autoscale called before worker store attached"
        self.autoscaler.refresh(self._store.worker_status())
        self.autoscaler.probe_health()
        self.autoscaler.update(request.residual_demand)
        return AutoscaleResult(autoscaler_state=self.autoscaler.persistable_state())

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
