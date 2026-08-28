# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskBackend: the contract an Iris controller uses to drive its cluster.

Each controller owns one backend. Per tick the controller drives that backend
through three uniform methods — :meth:`TaskBackend.schedule`,
:meth:`TaskBackend.reconcile`, and :meth:`TaskBackend.autoscale` — passing
controller-owned inputs
(:class:`ScheduleRequest` / :class:`ReconcileRequest` / :class:`AutoscaleRequest`)
and getting back method-specific results (:class:`ScheduleResult` /
:class:`ReconcileObservation` / :class:`AutoscaleResult`). The controller supplies a
complete scheduling workspace, so scheduling never reads controller storage.
The worker backend's reconcile and autoscale phases use the controller database
through a bound worker store. Kubernetes binds no worker store and delegates
placement and capacity management to its substrate.

:attr:`TaskBackend.descriptor` declares the backend's immutable identity, kind,
and advertised capacity. The two backend kinds have deliberately different
mechanisms: a worker backend runs the Iris scheduler and worker RPCs, while a
Kubernetes backend hands placement to Kueue and reconciles Pods directly.
"""

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.autoscaler.state import AutoscalerState
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.snapshot import ObservedTaskUpdate
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.reconcile.worker import (
    ReconcileInputs,
    ReconcileRow,
    WorkerReconcilePlan,
    WorkerReconcileResult,
    build_reconcile_plans,
)
from iris.cluster.controller.scheduling.decision import apply_preemptions, compute_diagnostics
from iris.cluster.controller.scheduling.policy import (
    GatedCandidates,
    SchedulingOrder,
    apply_scheduling_gates,
    compute_demand_entries,
    compute_scheduling_order,
    demanded_availability_variants,
    enrich_workers_with_availability,
)
from iris.cluster.controller.scheduling.scheduler import JobRequirements, Scheduler, SchedulingContext
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerHealthTracker
from iris.cluster.types import JobName, PendingTask, WorkerId
from iris.rpc import controller_pb2, job_pb2, vm_pb2, worker_pb2

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Communication failure with the execution backend."""


class ProviderUnsupportedError(ProviderError):
    """Operation not supported by this backend implementation."""


class BackendKind(StrEnum):
    """The execution mechanism owned by a controller's backend."""

    WORKER = "worker-daemon"
    KUBERNETES = "kubernetes"


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    """Immutable identity and advertised metadata for one controller backend."""

    backend_id: str
    kind: BackendKind
    advertised_attributes: Mapping[str, frozenset[str]] = field(default_factory=dict)
    scale_groups: frozenset[str] = field(default_factory=frozenset)
    display_name: str | None = None

    def __post_init__(self) -> None:
        normalized = {key: frozenset(values) for key, values in self.advertised_attributes.items()}
        object.__setattr__(self, "advertised_attributes", MappingProxyType(normalized))
        object.__setattr__(self, "scale_groups", frozenset(self.scale_groups))


@dataclass(frozen=True, slots=True)
class DashboardBackendDescriptor:
    """Backend metadata exposed by the dashboard config endpoint."""

    name: str
    capabilities: list[str]


def dashboard_backend_descriptor(backend: "TaskBackend") -> DashboardBackendDescriptor:
    descriptor = backend.descriptor
    if descriptor.kind is BackendKind.KUBERNETES:
        capabilities = ["cluster"]
    else:
        capabilities = ["workers"]
        if backend.autoscaler is not None:
            capabilities.append("autoscaler")
    return DashboardBackendDescriptor(
        name=descriptor.display_name or descriptor.backend_id,
        capabilities=capabilities,
    )


def plans_from_snapshot(snapshot: ControlSnapshot) -> list[WorkerReconcilePlan]:
    """Group a snapshot's reconcile rows by worker and build per-worker plans.

    The worker-daemon reconcile prologue: every active worker gets a plan (idle
    workers an empty one), so a single reconcile pass reaches the whole fleet.
    """
    rows_by_worker: dict[WorkerId, list[ReconcileRow]] = {wid: [] for wid in snapshot.worker_addresses}
    for row in snapshot.reconcile_rows:
        rows_by_worker[row.worker_id].append(row)
    return build_reconcile_plans(
        ReconcileInputs(
            job_specs=snapshot.job_specs,
            worker_ids=list(snapshot.worker_addresses),
            rows_by_worker=rows_by_worker,
        )
    )


@dataclass(frozen=True)
class DeviceCapacity:
    """Free vs. total consumable capacity for one resource token, in the token's
    natural unit (accelerator variant → chips).

    ``held_by_band`` splits the non-free remainder by the ``PriorityBand`` holding
    it, so a federation parent can tell capacity it could reclaim by preemption
    (work it outranks) from capacity it cannot. Empty when the backend cannot
    attribute held capacity to a band; the parent then reclaims nothing.
    """

    free: int
    total: int
    held_by_band: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskTarget:
    """Addresses one task attempt for on-demand RPCs (status / profile / exec).

    Worker-daemon backends route by :attr:`address`; direct backends route by
    :attr:`task_id` / :attr:`attempt_id` / :attr:`attempt_uid`. Each backend reads
    the fields it needs; the controller fills them from the DB once at the RPC
    boundary. ``attempt_uid`` is the incarnation key the K8s backend needs to
    rebuild the pod name (which embeds it); empty for worker-daemon targets.
    """

    task_id: str
    attempt_id: int
    worker_id: WorkerId | None
    address: str | None
    attempt_uid: str = ""


@dataclass(frozen=True)
class ScheduleResult:
    """What :meth:`TaskBackend.schedule` decides for one scheduling tick.

    Pure decision, no I/O. A backend that doesn't run the Iris scheduler (e.g. a
    cluster backend where Kueue owns placement) returns an empty instance.
    """

    assignments: list[Assignment] = field(default_factory=list)
    """task→worker placements to commit (``ops.task.assign``)."""
    preemptions: list[TerminalDecision] = field(default_factory=list)
    """Victims to finalize PREEMPT (``ops.task.finalize``)."""
    unschedulable: list[PendingTask] = field(default_factory=list)
    """Expired/deadline pending rows to mark UNSCHEDULABLE."""
    residual_demand: list[DemandEntry] = field(default_factory=list)
    """Limits-free capacity-fit residual; cached for the autoscaler loop."""
    diagnostics: dict[str, str] = field(default_factory=dict)
    """Per-job scheduling diagnostics surfaced on the dashboard."""
    scheduling_context: SchedulingContext | None = None
    """Post-placement scheduling context cached for dashboard diagnostics."""


@dataclass(frozen=True)
class WorkerFleetObservation:
    """Raw worker RPC outcomes from one reconcile fan-out.

    The backend reports transport/runtime facts only. The controller reloads a
    fresh transition snapshot, resolves exact Attempt UIDs, applies Iris state
    policy, and folds liveness after this I/O returns.
    """

    worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]] = field(default_factory=list)
    transport_events: list[WorkerHealthEvent] = field(default_factory=list)


@dataclass(frozen=True)
class DirectTaskObservation:
    """Exact task-attempt observations from a direct execution provider."""

    updates: list[ObservedTaskUpdate] = field(default_factory=list)


type ReconcileObservation = WorkerFleetObservation | DirectTaskObservation


@dataclass(frozen=True)
class AutoscaleResult:
    """What :meth:`TaskBackend.autoscale` did this tick.

    A provisioning cycle returns the updated ``autoscaler_state`` to persist; a
    dead-worker teardown returns the full set of ``removed_workers`` (the dead
    workers plus their healthy slice siblings). A backend that owns its own
    capacity (e.g. Kubernetes) returns an empty instance.
    """

    removed_workers: list[WorkerId] = field(default_factory=list)
    """Workers torn down this tick — dead workers plus their healthy slice
    siblings. The controller serializes their removal and forgets them."""
    autoscaler_state: AutoscalerState | None = None
    """The autoscaler's tracked state for the controller to persist; None when
    the backend manages its own capacity or did not provision this tick."""


@dataclass(frozen=True)
class ScheduleRequest:
    """Complete single-use scheduling workspace for one backend and tick.

    The controller builds this from one read snapshot. The backend performs a
    pure decision over the supplied context and never reads controller storage
    while scheduling.
    """

    context: SchedulingContext
    max_tasks_per_job_per_cycle: int
    trace: bool = False


@dataclass(frozen=True)
class ReconcileRequest:
    """Controller-owned inputs for one backend's reconcile tick.

    A worker-daemon backend sources its own worker/placement snapshot and ignores
    this; a Kubernetes backend that owns placement receives the dispatch
    drain (the PENDING->ASSIGNED promotion the controller commits as a DB write)
    and applies it to its cluster.
    """

    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    running_tasks: list[RunningTaskEntry] = field(default_factory=list)


@dataclass(frozen=True)
class AutoscaleRequest:
    """Controller-owned inputs for one backend's autoscale tick.

    ``residual_demand`` is this tick's unmet demand (from the same backend's
    schedule). A non-empty ``dead_workers`` means "tear down these workers'
    slices and their healthy siblings" instead of provisioning; a backend tears
    down only the workers its own autoscaler tracks. The backend reads its own
    worker status for the provisioning refresh.
    """

    residual_demand: list[DemandEntry] = field(default_factory=list)
    dead_workers: list[WorkerId] = field(default_factory=list)


def run_scheduling_decision(
    scheduler: Scheduler,
    request: ScheduleRequest,
    zone_capabilities: Mapping[str, frozenset[str]] | None = None,
) -> ScheduleResult:
    """Run the full Iris scheduling decision pipeline over a DB-less snapshot.

    Stages: availability enrichment → gates → order → ``find_assignments`` →
    preemption pass. Returns the placement decisions plus the diagnostics/context
    the controller caches. Does no I/O — every input comes from ``snapshot``
    (plus the autoscaler-derived ``zone_capabilities`` snapshot) and every output
    is plain data.

    ``zone_capabilities`` (zone -> accelerator variants empirically available there)
    is folded onto worker attributes as ``availability:<variant>`` markers so a hard
    availability constraint confines a job to a zone where the accelerator has
    actually been obtained.
    """
    ctx = request.context
    trace = request.trace

    if zone_capabilities:
        # Inject only the availability markers some pending task actually constrains
        # on (typically a single variant, e.g. v5p-8). Pruning zone_capabilities to
        # the demanded variants confines the per-worker attribute copy to the handful
        # of workers in a zone that provisions one, instead of rebuilding every
        # worker's attributes every tick. No demand -> no enrichment, no index rebuild.
        demanded = demanded_availability_variants(ctx.pending_task_rows)
        relevant = {zone: kept for zone, variants in zone_capabilities.items() if (kept := variants & demanded)}
        if relevant:
            ctx = ctx.evolve_with_workers(
                workers=enrich_workers_with_availability(ctx.workers, relevant),
                jobs=ctx.jobs,
                building_counts=ctx.building_counts,
                max_building_tasks=ctx.max_building_tasks,
            )

    gated = apply_scheduling_gates(
        ctx,
        max_tasks_per_job_per_cycle=request.max_tasks_per_job_per_cycle,
        trace=trace,
    )

    # Residual demand is computed alongside the assignments from the same
    # snapshot and the same Scheduler instance: a limits-free capacity-fit over
    # the pending tasks. Tasks this tick retires as UNSCHEDULABLE (deadline
    # expired) are excluded so the autoscaler is never asked to provision for a
    # job the same tick is failing. ``apply_scheduling_gates`` above only reads
    # ``ctx``; this still runs before ``apply_placements`` mutates it.
    residual_demand = compute_demand_entries(ctx, scheduler, exclude_task_ids={t.task_id for t in gated.expired_tasks})

    if not gated.schedulable_task_ids:
        # No work to place. Expired tasks (if any) still flow back so the
        # controller can mark them UNSCHEDULABLE; the context is the diagnostics
        # snapshot for this tick.
        return ScheduleResult(
            unschedulable=list(gated.expired_tasks),
            scheduling_context=ctx,
            residual_demand=residual_demand,
        )

    order = compute_scheduling_order(ctx, gated, trace=trace)
    all_assignments, context, placed_jobs = apply_placements(scheduler, order, gated, ctx, trace=trace)
    preemption_plan = apply_preemptions(order, placed_jobs, all_assignments, ctx.running_for_preemption, context)

    # Commit each preemptor onto the worker its victim frees in the same tick as
    # the PREEMPT, so it is not re-competed for its own freed slot next tick. The
    # freed worker is not physically empty until the victim's attempt finalizes;
    # the reconcile dispatch gate holds the preemptor's run-intent until then.
    assignments = all_assignments + preemption_plan.placements
    diagnostics = compute_diagnostics(scheduler, context, placed_jobs, assignments, order.ordered_task_ids)

    return ScheduleResult(
        assignments=[
            Assignment(task_id=task_id, worker_id=worker_id, priority_band=order.task_band_map.get(task_id))
            for task_id, worker_id in assignments
        ],
        preemptions=[
            TerminalDecision(
                kind=TerminalKind.PREEMPT,
                task_id=victim_id,
                reason=f"Preempted by {preemptor_name}",
            )
            for preemptor_name, victim_id in preemption_plan.evictions
        ],
        unschedulable=list(gated.expired_tasks),
        diagnostics=diagnostics,
        scheduling_context=context,
        residual_demand=residual_demand,
    )


def apply_placements(
    scheduler: Scheduler,
    order: SchedulingOrder,
    gated: GatedCandidates,
    ctx: SchedulingContext,
    *,
    trace: bool,
) -> tuple[list[tuple[JobName, WorkerId]], SchedulingContext, dict[JobName, JobRequirements]]:
    """Run the assignment pass over the gated context in priority order."""
    ctx.pending_tasks = list(order.ordered_task_ids)
    ctx.jobs = gated.jobs
    context = ctx

    if trace:
        logger.info(
            "[TRACE] Phase 4 context: %d workers, %d pending tasks, %d jobs",
            len(context.capacities),
            len(context.pending_tasks),
            len(context.jobs),
        )

    result = scheduler.find_assignments(context)
    all_assignments = result.assignments
    if trace:
        logger.info("[TRACE] Phase 5 assignments: %d total", len(all_assignments))
    return all_assignments, context, gated.jobs


@dataclass(frozen=True)
class BackendRuntime:
    """Controller-owned values used to build a worker backend's store.

    Passed to :meth:`TaskBackend.bind_runtime` at startup.
    """

    db: ControllerDB
    """The controller database."""


class TaskBackend(Protocol):
    """Drives task execution + capacity reporting for a single cluster backend.

    The controller supplies a complete scheduling workspace and threads the
    per-user budget. Implementations dispatch backend-specific I/O and return
    plain data. Reconcile/autoscale use a controller-DB store; schedule is DB-less.
    """

    descriptor: BackendDescriptor
    """Stable identity, backend kind, and advertised capacity."""

    autoscaler: Autoscaler | None
    """The Iris :class:`Autoscaler` driving capacity, or None for backends that
    manage their own capacity or have no scale groups. Read-only handle the
    controller exposes for dashboard/status RPCs; capacity is driven through
    :meth:`autoscale`, never this attribute."""

    @property
    def health(self) -> WorkerHealthTracker | None:
        """The backend's worker tracker, or None for Kubernetes."""
        ...

    def resource_capacity(self) -> dict[str, DeviceCapacity] | None:
        """Free and total consumable capacity right now, per resource token.

        A federation parent advertises this to peers so a queued federated job can
        wait for a peer that actually has room (see ``federation.availability``);
        the dashboard renders the same numbers. v1 reports accelerator chips keyed
        by lowercased ``device-variant`` (e.g. ``{"h100": DeviceCapacity(8, 64)}``),
        computed from the same live-worker ``WorkerCapacity`` (``total - committed``)
        the scheduler uses.

        Returns ``None`` when this backend does not supply the metric (a placement-
        owning Kubernetes backend that does not track per-worker capacity); the
        controller then leaves ``BackendSummary.availability`` UNSET so a peer reading
        it falls back to shape-only federation. An empty dict is an authoritative
        "nothing free"."""
        ...

    def runtime_image(self, requested_image: str) -> str:
        """Resolve the container image used for a task request.

        Direct container backends can supply their configured default when the
        request leaves the image empty. Worker-daemon backends only know an
        explicitly requested image; their workers report build details.
        """
        ...

    def status(self) -> controller_pb2.Controller.BackendStatus:
        """Author this backend's expanded status for the dashboard Backends tab.

        Each backend authors the variant selected by :attr:`BackendDescriptor.kind`:
        Kubernetes fills ``kubernetes`` from its cached cluster-state snapshot;
        a worker backend fills ``worker`` in full from the state it owns
        — its liveness tracker (health counts + per-VM usability) and running-task
        rows, with its :meth:`autoscaler_status` embedded. The controller reads the
        result verbatim; it overlays nothing.
        """
        ...

    def autoscaler_status(self) -> vm_pb2.AutoscalerStatus:
        """This backend's autoscaler status, fully populated and self-contained.

        Every group is tagged with this backend's id and every VM carries its
        usability, running-task count, and capacity verdict. Empty for a backend
        with no autoscaler.
        """
        ...

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Decide task→worker placement for the routed tasks (pure decision).

        The request contains the controller-built scheduling context. Worker-daemon
        backends run the full Iris scheduling pipeline; cluster backends (Kueue,
        slurmctld) return an empty result because they place tasks themselves.
        """
        ...

    def reconcile(self, request: ReconcileRequest) -> ReconcileObservation:
        """Converge external execution and return neutral observations.

        Bounded I/O only. Worker-daemon backends return raw per-worker RPC
        outcomes and transport reachability. Kubernetes backends apply/poll Pods
        and return exact task-attempt observations. The controller owns snapshot
        reload, state-machine policy, liveness folding, and persistence.
        """
        ...

    def teardown(self, dead_workers: list[WorkerId], *, reason: str) -> None:
        """Tear down a specific set of this backend's workers now.

        The controller calls this after committing the state transitions caused
        by liveness reaping, and for the recycled-IP eviction queue. ``reason``
        is recorded on the worker failure. A backend that tracks no Iris workers
        is a no-op.
        """
        ...

    def prune_dead_workers(self, *, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
        """Garbage-collect this backend's DEAD workers whose heartbeat predates ``cutoff_ms``.

        Driven by the controller's background prune loop, not the control tick. The
        backend deletes its own dead worker rows (and their attributes) from its own
        tracker, one per transaction, sleeping ``pause`` between deletes and stopping
        early once ``stop_event`` is set. Returns the count removed. A backend that
        tracks no Iris workers returns 0.
        """
        ...

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        """Provision capacity for unmet demand, OR tear down dead workers.

        Bounded I/O. With ``request.dead_workers`` set, the backend terminates
        those workers' slices AND their healthy siblings and returns the full set
        as ``removed_workers`` (no provisioning this call). Otherwise it runs one
        scaling cycle against ``request.residual_demand``, reading its own worker
        status. Either way it returns its tracked ``autoscaler_state`` for the
        controller to persist. Backends that manage their own capacity (k8s)
        return an empty result.
        """
        ...

    def bind_runtime(self, runtime: BackendRuntime) -> None:
        """Build this backend's live-worker read surface from controller-owned deps.

        Called once by the controller for a worker-daemon backend. The backend joins
        ``runtime`` with its own liveness tracker to build the
        :class:`~iris.cluster.controller.backend_store.BackendWorkerStore` it reads
        through; Kubernetes backends track no Iris workers and no-op.
        """
        ...

    def seed_liveness(self) -> None:
        """Seed this backend's persisted workers as live so the scheduler sees them.

        Called by the controller at start and after a DB reopen (checkpoint
        restore), only on worker-daemon backends. The backend reads its own
        persisted workers and heartbeats them into the tracker it owns.
        Kubernetes backends track no liveness and no-op.
        """
        ...

    def get_process_status(
        self,
        target: TaskTarget,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        """Fetch full process status. Raises ProviderUnsupportedError if N/A."""
        ...

    def profile_task(
        self,
        target: TaskTarget,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        """Profile a task attempt. Raises ProviderUnsupportedError if N/A."""
        ...

    def exec_in_container(
        self,
        target: TaskTarget,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        """Exec a command in a task's container. Raises ProviderUnsupportedError if N/A."""
        ...

    def close(self) -> None:
        """Release backend-owned resources at controller shutdown.

        Called from Controller.stop(). Capacity-managing backends shut down
        their attached autoscaler here (terminating VMs, stopping the platform);
        others close any connections or collectors they own.
        """
        ...
