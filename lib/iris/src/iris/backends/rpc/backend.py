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

import logging
import threading
from dataclasses import dataclass, field, replace
from typing import ClassVar

from rigging.timing import Duration, Timestamp

from iris.backends.protocol import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendCapability,
    BackendWorkerStore,
    DeviceCapacity,
    ProviderError,
    ReconcileRequest,
    ReconcileResult,
    ScheduleInput,
    ScheduleRequest,
    ScheduleResult,
    TaskTarget,
    WorkerClient,
    assemble_scheduling_context,
    plans_from_snapshot,
    run_scheduling_decision,
)
from iris.backends.status import AutoscalerStatus, BackendStatus, WorkerFleetStatus
from iris.cluster.constraints import DeviceType
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.status import overlay_worker_usability
from iris.cluster.controller.reconcile.apply import apply_worker_reconcile
from iris.cluster.controller.reconcile.worker import WorkerReconcilePlan, WorkerReconcileResult
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.controller.worker_health import (
    DEFAULT_UNREACHABLE_GRACE,
    WorkerHealthEvent,
    WorkerHealthEventKind,
    WorkerHealthTracker,
)
from iris.cluster.types import WellKnownAttribute
from iris.resources.endpoint import ExecRequest, ExecResult, ProfileRequest, ProfileResult
from iris.resources.names import WorkerId
from iris.resources.system import ProcessInfo

logger = logging.getLogger(__name__)

# Max concurrent in-flight per-worker RPCs in a fan-out (asyncio.Semaphore width).
# Kept >= fleet size so the whole fleet reconciles in one wave and a slow worker
# costs one RPC-timeout window per round, not one per wave.
RECONCILE_FANOUT_PARALLELISM = 512

# Failure reason stamped on a worker the reconcile fold reaped (drained by
# ``run_teardown``).
WORKER_RECONCILE_TEARDOWN_REASON = "worker reconcile failure threshold exceeded"


@dataclass(frozen=True)
class FleetObservation:
    """One reconcile fan-out's raw outcome.

    The per-worker results paired with their plans, and the transport liveness
    each yielded (REACHED / UNREACHABLE). :meth:`RpcTaskBackend.reconcile` resolves
    the results into effects and folds the liveness.
    """

    worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]]
    transport_events: list[WorkerHealthEvent]


@dataclass
class RpcTaskBackend:
    """A worker-daemon :class:`~iris.backends.protocol.TaskBackend`
    backed by async Connect RPCs.

    Each fan-out method spins up an asyncio event loop and dispatches the
    relevant RPC to each worker concurrently via `asyncio.gather`, capped at
    `parallelism` in-flight requests by a local semaphore. Cached stubs in
    the factory keep their pyqwest connection pools across rounds.
    """

    worker_client: WorkerClient
    parallelism: int = RECONCILE_FANOUT_PARALLELISM
    name: str = "worker"
    # The id the controller assigned this backend, learned with its worker store. The
    # backend stamps it onto the autoscaler groups it authors in ``status`` /
    # ``autoscaler_status``, so the controller reads those verbatim.
    backend_id: str = field(default="", init=False)
    # The Iris autoscaler that provisions capacity for this backend, passed by the
    # composer at construction after it builds the autoscaler from the provider
    # bundle; None for clusters with no scale groups, where capacity calls are no-ops.
    autoscaler: Autoscaler | None = None
    # This backend's typed worker store, composed and attached by the controller.
    # The backend reads its own workers and reaps its dead ones through this; it
    # never receives ControllerDB or a concrete persistence implementation.
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

    def attach_worker_store(self, backend_id: str, store: BackendWorkerStore) -> None:
        """Attach the controller-composed worker-state port."""
        self.backend_id = backend_id
        self._store = store

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

    def autoscaler_status(self) -> AutoscalerStatus:
        """Author this backend's autoscaler status from the state it owns.

        Tags each group with this backend's id, then overlays every VM with the
        usability/running-task/capacity verdict from this backend's own liveness
        tracker plus the running-task rows the store reads for those VMs.
        """
        assert self._store is not None, "RpcTaskBackend.autoscaler_status called before worker store attached"
        if self.autoscaler is None:
            return AutoscalerStatus()
        status = self.autoscaler.get_status()
        status = replace(
            status,
            groups=tuple(replace(group, backend_id=self.backend_id) for group in status.groups),
        )
        usability_by_id = {str(worker_id): live.usability for worker_id, live in self.health.all().items()}
        vm_ids = {WorkerId(vm.vm_id) for group in status.groups for s in group.slices for vm in s.vms if vm.vm_id}
        return overlay_worker_usability(status, usability_by_id, self._store.running_tasks(vm_ids))

    def status(self) -> BackendStatus:
        """Author the full ``worker`` status variant from this backend's own state:
        the health counts from its liveness tracker around its :meth:`autoscaler_status`.
        The controller reads the result verbatim and overlays nothing.
        """
        liveness = self.health.all()
        return BackendStatus(
            worker=WorkerFleetStatus(
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

        results = self.worker_client.reconcile(plans, snapshot.worker_addresses, parallelism=self.parallelism)

        worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]] = list(zip(plans, results, strict=True))
        transport_events: list[WorkerHealthEvent] = []
        for plan, result in worker_results:
            address = snapshot.worker_addresses[plan.worker_id]
            if result.error is not None:
                transport_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
                self.worker_client.evict(address)
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
                self.worker_client.evict(address)
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
        effects = apply_worker_reconcile(self._store, observation.worker_results, now=now)
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

    def get_process_status(self, target: TaskTarget) -> ProcessInfo:
        if not target.address:
            raise ProviderError(f"Worker {target.worker_id} has no address")
        return self.worker_client.process_status(target.address)

    def profile_task(
        self,
        target: TaskTarget,
        request: ProfileRequest,
    ) -> ProfileResult:
        if not target.address:
            raise ProviderError(f"Worker {target.worker_id} has no address")
        return self.worker_client.profile(target.address, request)

    def exec_in_container(
        self,
        target: TaskTarget,
        request: ExecRequest,
    ) -> ExecResult:
        if not target.address:
            raise ProviderError(f"Worker {target.worker_id} has no address")
        return self.worker_client.exec(target.address, request)

    def close(self) -> None:
        if self.autoscaler is not None:
            self.autoscaler.shutdown()
        self.worker_client.close()
