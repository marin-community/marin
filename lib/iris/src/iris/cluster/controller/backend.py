# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskBackend: the contract every Iris execution backend implements.

The controller owns task lifecycle and routing; each backend owns its live
workers. Per tick the controller routes pending *tasks* to backends (the
meta-scheduler), threads the per-user budget, and drives each backend through
three uniform methods — :meth:`TaskBackend.schedule`, :meth:`TaskBackend.reconcile`,
:meth:`TaskBackend.autoscale` — passing controller-owned inputs
(:class:`ScheduleRequest` / :class:`ReconcileRequest` / :class:`AutoscaleRequest`)
and getting back method-specific results (:class:`ScheduleResult` /
:class:`ReconcileResult` / :class:`AutoscaleResult`). Worker and placement state
never flow controller→backend: a backend sources its own workers through a
:class:`~iris.cluster.controller.backend_store.BackendWorkerStore` and decides
placement itself. Backends perform
backend-specific I/O (worker-daemon RPC fan-out, ``kubectl apply``) but never
read or write the controller database directly.

:attr:`TaskBackend.capabilities` is a pure descriptor for the dashboard tab list
and on-demand service-RPC routing (worker-daemon vs direct-pod exec). One narrow
exception gates the per-tick path: a ``CLUSTER_VIEW`` backend owns placement, so
the controller drains the dispatch queue (a DB write it owns) and hands the
promoted tasks to that backend's ``reconcile``. A worker-daemon backend instead
sources its own placement and folds the liveness it observed; a cluster backend
has no Iris workers, so its ``run_teardown`` is a no-op.
"""

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Protocol

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.autoscaler.state import AutoscalerState
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile import ControllerEffects
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.reconcile.worker import (
    ReconcileInputs,
    ReconcileRow,
    WorkerReconcilePlan,
    build_reconcile_plans,
)
from iris.cluster.controller.run_template import RunTemplateCache
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
from iris.cluster.controller.scheduling.scheduler import (
    DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
    DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
    JobRequirements,
    RunningTaskInfo,
    Scheduler,
    SchedulingContext,
    WorkerSnapshot,
)
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import JobName, PendingTask, UserBudgetDefaults, WorkerId
from iris.rpc import controller_pb2, job_pb2, vm_pb2, worker_pb2

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Communication failure with the execution backend."""


class ProviderUnsupportedError(ProviderError):
    """Operation not supported by this backend implementation."""


class BackendCapability(StrEnum):
    """A descriptor flag the dashboard and on-demand RPC routing key on.

    Mostly metadata: the controller calls ``schedule``/``reconcile``/``autoscale``
    uniformly regardless of these flags, and stores each backend's authored
    projection the same way. The one per-tick exception is ``CLUSTER_VIEW``, which
    tells the controller to drain the dispatch queue into the reconcile request (a
    DB write the placement-owning backend can't do).
    """

    WORKER_DAEMON = "workers"
    """Iris tracks worker daemons (Fleet tab; exec/profile route by worker)."""

    IRIS_AUTOSCALER = "autoscaler"
    """The Iris autoscaler provisions capacity for this backend (Autoscaler tab)."""

    CLUSTER_VIEW = "cluster"
    """The backend places tasks on its own cluster (Cluster tab; exec by task/pod)."""


@dataclass(frozen=True)
class BackendDescriptor:
    """Capabilities the dashboard uses to choose which panels to show.

    Derived from the live :class:`TaskBackend`; served (as JSON) by ``/auth/config``
    so the frontend tab list is data-driven.
    """

    name: str
    capabilities: list[str]


def backend_descriptor(backend: "TaskBackend") -> BackendDescriptor:
    """Build the dashboard capability descriptor from a live backend."""
    return BackendDescriptor(
        name=backend.name,
        capabilities=sorted(c.value for c in backend.capabilities),
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
class TaskTarget:
    """Addresses one task attempt for on-demand RPCs (status / profile / exec).

    Worker-daemon backends route by :attr:`address`; direct backends route by
    :attr:`task_id` / :attr:`attempt_id`. Each backend reads the fields it needs;
    the controller fills them from the DB once at the RPC boundary.
    """

    task_id: str
    attempt_id: int
    worker_id: WorkerId | None
    address: str | None


@dataclass(frozen=True)
class ScheduleInput:
    """The read-only state a :class:`TaskBackend` needs to make placement
    decisions for one scheduling tick."""

    context: SchedulingContext
    """Built by ``build_scheduling_context`` (workers + raw reads)."""
    max_tasks_per_job_per_cycle: int
    trace: bool = False
    """Whether to emit the per-phase scheduling trace logs this cycle."""


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
class ReconcileResult:
    """The committable projection :meth:`TaskBackend.reconcile` authored this tick.

    Carries only ``effects``: a backend that tracks Iris workers folds and tears
    down its own reaped workers, so no worker identity crosses this boundary.
    """

    effects: ControllerEffects = field(default_factory=ControllerEffects)
    """Task/attempt/job writes for the controller to commit (``commit_effects``)."""


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
class BackendSchedulingInputs:
    """The worker-side scheduling state a backend reads from its own source.

    The controller never supplies these — the backend assembles its
    :class:`SchedulingContext` by joining these (its live workers, their building
    counts, the attempts it is running) with the controller-owned
    :class:`ScheduleRequest` (the routed pending tasks + budgets).
    """

    workers: list[WorkerSnapshot]
    building_counts: dict[WorkerId, int]
    running_for_preemption: list[RunningTaskInfo]
    max_building_tasks: int = DEFAULT_MAX_BUILDING_TASKS_PER_WORKER
    max_assignments_per_worker: int = DEFAULT_MAX_ASSIGNMENTS_PER_WORKER


@dataclass(frozen=True)
class ScheduleRequest:
    """Controller-owned inputs for one backend's scheduling tick.

    Carries only the routed pending tasks and the per-user budget state — never
    worker data. The backend sources its own workers (a
    :class:`BackendSchedulingInputs`) and assembles the full scheduling context
    internally, so the controller does no worker partitioning. ``user_spend`` is
    threaded across backends in a fixed order so two backends cannot double-spend
    one user's budget in a single tick.
    """

    pending_task_rows: list[PendingTask]
    requested_bands: dict[JobName, int]
    user_spend: dict[str, int]
    user_budget_limits: dict[str, int]
    user_budget_defaults: UserBudgetDefaults
    max_tasks_per_job_per_cycle: int
    trace: bool = False


@dataclass(frozen=True)
class ReconcileRequest:
    """Controller-owned inputs for one backend's reconcile tick.

    A worker-daemon backend sources its own worker/placement snapshot and ignores
    this; a ``CLUSTER_VIEW`` backend that owns placement receives the dispatch
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


def user_admitted(allowed_users: frozenset[str], user: str) -> bool:
    """Whether an allow policy permits ``user`` (``"*"`` matches any user)."""
    return "*" in allowed_users or user in allowed_users


def assemble_scheduling_context(inputs: BackendSchedulingInputs, request: ScheduleRequest) -> SchedulingContext:
    """Join a backend's own worker-side inputs with the controller-owned request.

    The backend sources its live workers (``inputs``); the controller routes the
    pending tasks + budget state (``request``). Together they form the
    :class:`SchedulingContext` the Iris pipeline decides over.
    """
    return SchedulingContext(
        workers=inputs.workers,
        building_counts=inputs.building_counts,
        max_building_tasks=inputs.max_building_tasks,
        max_assignments_per_worker=inputs.max_assignments_per_worker,
        pending_tasks=[],
        jobs={},
        pending_task_rows=request.pending_task_rows,
        user_spend=request.user_spend,
        user_budget_limits=request.user_budget_limits,
        requested_bands=request.requested_bands,
        user_budget_defaults=request.user_budget_defaults,
        running_for_preemption=inputs.running_for_preemption,
    )


def run_scheduling_decision(
    scheduler: Scheduler,
    snapshot: ScheduleInput,
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
    ctx = snapshot.context
    trace = snapshot.trace

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
        max_tasks_per_job_per_cycle=snapshot.max_tasks_per_job_per_cycle,
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
    preemptions = apply_preemptions(order, placed_jobs, all_assignments, ctx.running_for_preemption, context)
    diagnostics = compute_diagnostics(scheduler, context, placed_jobs, all_assignments, order.ordered_task_ids)

    return ScheduleResult(
        assignments=[
            Assignment(task_id=task_id, worker_id=worker_id, priority_band=order.task_band_map.get(task_id))
            for task_id, worker_id in all_assignments
        ],
        preemptions=[
            TerminalDecision(
                kind=TerminalKind.PREEMPT,
                task_id=victim_id,
                reason=f"Preempted by {preemptor_name}",
            )
            for preemptor_name, victim_id in preemptions
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
    """The controller-owned values a worker-daemon backend builds its
    :class:`~iris.cluster.controller.backend_store.BackendWorkerStore` from.

    Passed to :meth:`TaskBackend.bind_runtime` at startup.
    """

    backend_id: str
    """The id the controller assigned this backend. The backend stamps it onto the
    autoscaler groups it authors, so the controller never has to tag them afterward."""
    db: ControllerDB
    """The controller database."""
    endpoints: EndpointsProjection
    """The worker-endpoint projection."""
    run_template_cache: RunTemplateCache
    """Per-job ``RunTaskRequest`` template cache."""
    owns_scale_group: Callable[[str], bool]
    """Whether a scale group belongs to this backend (the default backend also claims
    scale groups mapped to no backend)."""
    budget_defaults: UserBudgetDefaults
    """Per-user budget defaults."""


class TaskBackend(Protocol):
    """Drives task execution + capacity reporting for a single cluster backend.

    The controller routes pending *tasks* to a backend and threads the per-user
    budget; the backend sources its own *workers* and decides placement — worker
    and placement state never flow controller→backend. Implementations dispatch
    backend-specific I/O and return plain data; they never touch the controller
    database directly.
    """

    name: str
    """Stable identifier, e.g. ``"gcp"``, ``"coreweave"``, later ``"slurm-stanford"``."""

    capabilities: ClassVar[frozenset[BackendCapability]]
    """Descriptor for the dashboard + on-demand RPC routing. The controller calls
    ``schedule``/``reconcile``/``autoscale`` uniformly regardless; the sole
    per-tick exception is ``CLUSTER_VIEW`` (drives dispatch-drain into the
    reconcile snapshot)."""

    autoscaler: Autoscaler | None
    """The Iris :class:`Autoscaler` driving capacity, or None for backends that
    manage their own capacity or have no scale groups. Read-only handle the
    controller exposes for dashboard/status RPCs; capacity is driven through
    :meth:`autoscale`, never this attribute."""

    @property
    def health(self) -> WorkerHealthTracker | None:
        """The worker-liveness tracker this backend constructs and owns, holding only
        the workers in its scale groups, or None for a backend that tracks no Iris
        workers (k8s). The backend folds and reaps through it; the controller reaches
        worker liveness through it (routed by scale group) for its Fleet/exec/capacity/
        prune readers and to seed/register a worker into its owning backend."""
        ...

    @property
    def worker_attrs(self) -> WorkerAttrsProjection | None:
        """The worker-attributes projection this backend constructs and owns, holding
        only the workers in its scale groups, or None for a backend that tracks no
        Iris workers (k8s). The controller reaches it (routed by scale group) to
        register a worker's attributes into its owning backend."""
        ...

    allowed_users: frozenset[str]
    """The allow policy: user ids permitted to route here (``"*"`` matches any).
    Set by the composer via :meth:`configure_routing`; read for the dashboard's
    restricted / allowed-user-count summary."""

    def advertised_attributes(self) -> dict[str, set[str]]:
        """Backend-global attributes the meta-scheduler routes against.

        Each set-valued attribute (``device-variant: {"v5e-4", "v5p-8"}``) expands
        into routing posting lists. A backend advertising nothing is a catch-all
        that matches every job. Read once at startup (attributes are static)."""
        ...

    def admits(self, user: str) -> bool:
        """Whether this backend's allow policy permits ``user`` to route here."""
        ...

    def configure_routing(self, advertised: dict[str, set[str]], allowed_users: frozenset[str]) -> None:
        """Set the routing metadata the meta-scheduler reads.

        Called once by the composer from the backend's config. ``advertised`` is
        the (comma-expanded) attribute sets; ``allowed_users`` is the allow policy
        (``"*"`` matches any user)."""
        ...

    def status(self) -> controller_pb2.Controller.BackendStatus:
        """Author this backend's expanded status for the dashboard Backends tab.

        Each backend authors its own ``BackendStatus`` variant uniformly,
        selected by :attr:`capabilities`: a ``CLUSTER_VIEW`` backend fills
        ``kubernetes`` from its cached cluster-state snapshot; a
        ``WORKER_DAEMON`` backend fills ``worker`` in full from the state it owns
        — its liveness tracker (health counts + per-VM usability) and running-task
        rows, with its :meth:`autoscaler_status` embedded. The controller reads the
        result verbatim; it overlays nothing.
        """
        ...

    def autoscaler_status(self) -> vm_pb2.AutoscalerStatus:
        """This backend's autoscaler status, fully populated and self-contained.

        Every group is tagged with this backend's id and every VM carries its
        usability / running-task-count / capacity verdict, so the controller can
        merge statuses across backends without reaching into any backend's state.
        Empty for a backend with no autoscaler.
        """
        ...

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Decide task→worker placement for the routed tasks (pure decision).

        The request carries only the routed pending tasks + budget state; the
        backend sources its own workers (its worker store) and assembles the
        scheduling context internally. Worker-daemon backends run the full
        Iris scheduling pipeline; cluster backends (Kueue, slurmctld) return an
        empty result — they place tasks themselves.
        """
        ...

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        """Converge the backend toward the desired state and author the projection.

        Bounded I/O. Worker-daemon backends source their own worker/placement
        snapshot, fan the reconcile RPC out, resolve the observations into task
        ``effects``, and fold the per-worker liveness they observed — stashing the
        workers their fold reaped for the matching :meth:`run_teardown`; cluster
        backends apply/poll the pods in ``request`` and resolve those into
        ``effects`` (they track no Iris workers).
        """
        ...

    def run_teardown(self) -> None:
        """Tear down the workers this backend's reconcile fold reaped this tick.

        Bounded I/O. The controller calls this AFTER the tick's reconcile effects
        are committed, so the just-finalized terminal attempts read as terminal and
        are skipped. The backend drains its stash of reaped workers, fails them,
        terminates their slices and healthy siblings, and forgets them from its
        liveness tracker. A cluster backend tracks no Iris workers and no-ops.
        """
        ...

    def teardown(self, dead_workers: list[WorkerId], *, reason: str) -> None:
        """Tear down a specific set of this backend's workers now.

        The same fail → slice-and-sibling teardown → forget sequence
        :meth:`run_teardown` drains its stash into, but for an explicit set the
        controller resolved to this backend off the reconcile path — the
        recycled-IP eviction queue. ``reason`` is recorded on the worker failure.
        A backend that tracks no Iris workers is a no-op.
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

        Called once by the controller for worker-daemon backends. The backend joins
        ``runtime`` with its own liveness tracker to build the scale-group-scoped
        :class:`~iris.cluster.controller.backend_store.BackendWorkerStore` it reads
        through; capacity-managing backends (k8s) track no Iris workers and no-op.
        """
        ...

    def seed_liveness(self) -> None:
        """Seed this backend's persisted workers as live so the scheduler sees them.

        Called by the controller at start and after a DB reopen (checkpoint
        restore), only on worker-daemon backends. The backend reads its own
        scale-group-scoped workers and heartbeats them into the tracker it owns.
        Capacity-managing backends (k8s) track no liveness and no-op.
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
