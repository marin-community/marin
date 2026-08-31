# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete controller-to-backend phase contracts.

Each controller owns one backend and drives it through explicit phase methods.
Every request is assembled from controller-owned state before the call. Backend
implementations perform pure decisions or bounded provider I/O and return plain
results; they never open the controller database or mutate Iris resources.

:attr:`TaskBackend.descriptor` declares the backend's immutable identity, kind,
and advertised capacity. The two backend kinds have deliberately different
mechanisms: a worker backend runs the Iris scheduler and worker RPCs, while a
Kubernetes backend hands placement to Kueue and reconciles Pods directly.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from rigging.timing import Timestamp

from iris.cluster.constraints import Constraint
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.autoscaler.recovery import AutoscalerCheckpoint
from iris.cluster.controller.autoscaler.state import AutoscalerState
from iris.cluster.controller.autoscaler.status import PendingHint
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.reconcile.worker import (
    ReconcileInputs,
    ReconcileRow,
    WorkerReconcilePlan,
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
from iris.cluster.controller.scheduling.scheduler import (
    JobRequirements,
    Scheduler,
    SchedulingContext,
    WorkerSnapshot,
)
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerLiveness
from iris.cluster.types import AttemptUid, JobName, PendingTask, WorkerId, WorkerStatusMap
from iris.rpc import controller_pb2, job_pb2, worker_pb2

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Communication failure with the execution backend."""


class ProviderUnsupportedError(ProviderError):
    """Operation not supported by this backend implementation."""


class BackendKind(StrEnum):
    """The execution mechanism owned by a controller's backend."""

    WORKER = "worker-daemon"
    KUBERNETES = "kubernetes"


class BackendCapability(StrEnum):
    """Mechanisms a backend asks the controller to feed during each phase."""

    WORKER_FLEET = "worker-fleet"
    DIRECT_DISPATCH = "direct-dispatch"
    AUTOSCALER = "autoscaler"


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    """Immutable identity and advertised metadata for one controller backend."""

    backend_id: str
    kind: BackendKind
    advertised_attributes: Mapping[str, frozenset[str]] = field(default_factory=dict)
    scale_groups: frozenset[str] = field(default_factory=frozenset)
    capabilities: frozenset[BackendCapability] = field(default_factory=frozenset)
    display_name: str | None = None

    def __post_init__(self) -> None:
        normalized = {key: frozenset(values) for key, values in self.advertised_attributes.items()}
        object.__setattr__(self, "advertised_attributes", MappingProxyType(normalized))
        object.__setattr__(self, "scale_groups", frozenset(self.scale_groups))
        capabilities = self.capabilities
        if not capabilities:
            capabilities = frozenset(
                {BackendCapability.DIRECT_DISPATCH}
                if self.kind is BackendKind.KUBERNETES
                else {BackendCapability.WORKER_FLEET}
            )
        object.__setattr__(self, "capabilities", frozenset(capabilities))


@dataclass(frozen=True, slots=True)
class DashboardBackendDescriptor:
    """Backend metadata exposed by the dashboard config endpoint."""

    name: str
    capabilities: list[str]


def dashboard_backend_descriptor(backend: "TaskBackend") -> DashboardBackendDescriptor:
    descriptor = backend.descriptor
    capabilities = []
    if BackendCapability.DIRECT_DISPATCH in descriptor.capabilities:
        capabilities.append("cluster")
    if BackendCapability.WORKER_FLEET in descriptor.capabilities:
        capabilities.append("workers")
    if BackendCapability.AUTOSCALER in descriptor.capabilities:
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


@dataclass(frozen=True, slots=True)
class RuntimeReleaseTarget:
    """Exact external runtime that the backend must stop or prove absent."""

    task_id: JobName
    attempt_id: int
    attempt_uid: AttemptUid
    worker_id: WorkerId | None = None
    worker_address: str | None = None


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
    """Post-placement scheduling context; ``None`` when no context was built."""


@dataclass(frozen=True)
class ReconcileObservation:
    """Backend facts from one bounded reconciliation pass.

    Every backend reports exact task-attempt state through ``task_updates``.
    Backends that communicate with Iris workers may additionally report worker
    reachability through ``worker_health_events``. ``released_attempt_uids``
    confirms that exact terminal runtimes are stopped or absent. The controller
    folds all three without inspecting the backend implementation or kind.
    """

    task_updates: list[TaskUpdate] = field(default_factory=list)
    worker_health_events: list[WorkerHealthEvent] = field(default_factory=list)
    released_attempt_uids: frozenset[AttemptUid] = frozenset()
    """Exact runtimes observed as stopped or absent during this pass."""


@dataclass(frozen=True)
class AutoscaleResult:
    """What :meth:`TaskBackend.autoscale` did this tick.

    A provisioning cycle returns the updated ``autoscaler_state`` to persist.
    Physical removal is a separate phase so provisioning and teardown cannot be
    selected by an overloaded request field.
    """

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
class WorkerReconcileTarget:
    """One worker address and the exact desired plan to send there."""

    plan: WorkerReconcilePlan
    address: str


@dataclass(frozen=True)
class WorkerFleetReconcileRequest:
    """Complete worker-daemon fan-out and exact release work for one pass."""

    targets: list[WorkerReconcileTarget] = field(default_factory=list)
    release_targets: tuple[RuntimeReleaseTarget, ...] = ()


@dataclass(frozen=True)
class DirectReconcileRequest:
    """Complete desired execution and release view for a direct backend."""

    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    running_tasks: list[RunningTaskEntry] = field(default_factory=list)
    release_targets: tuple[RuntimeReleaseTarget, ...] = ()


ReconcileRequest = WorkerFleetReconcileRequest | DirectReconcileRequest


@dataclass(frozen=True)
class AutoscaleRequest:
    """Controller-owned inputs for one backend's autoscale tick.

    ``residual_demand`` is this tick's unmet demand. ``worker_status`` is the
    controller's complete liveness + workload snapshot for capacity refresh.
    """

    residual_demand: list[DemandEntry] = field(default_factory=list)
    worker_status: WorkerStatusMap = field(default_factory=dict)


@dataclass(frozen=True)
class RemoveCapacityRequest:
    """Workers already fenced from Iris state whose capacity may be removed."""

    worker_ids: list[WorkerId]


@dataclass(frozen=True)
class RemoveCapacityResult:
    """External capacity-removal result folded by the controller."""

    sibling_workers: list[WorkerId] = field(default_factory=list)
    autoscaler_state: AutoscalerState | None = None


@dataclass(frozen=True)
class BackendRecoveryRequest:
    """Controller checkpoint supplied before control loops start."""

    autoscaler_checkpoint: AutoscalerCheckpoint | None = None


@dataclass(frozen=True)
class BackendRecoveryResult:
    """Backend state worth mirroring after provider recovery."""

    autoscaler_state: AutoscalerState | None = None


@dataclass(frozen=True)
class BackendObservationRequest:
    """Controller facts needed to publish status and capacity without DB reads."""

    workers: list[WorkerSnapshot] = field(default_factory=list)
    liveness: Mapping[WorkerId, WorkerLiveness] = field(default_factory=dict)
    running_tasks: Mapping[WorkerId, set[JobName]] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendObservation:
    """Backend-authored provider view cached by the controller."""

    status: controller_pb2.Controller.BackendStatus = field(default_factory=controller_pb2.Controller.BackendStatus)
    resource_capacity: dict[str, DeviceCapacity] | None = None
    pending_hints: dict[str, PendingHint] = field(default_factory=dict)
    observed_at: Timestamp = field(default_factory=Timestamp.now)


@dataclass(frozen=True, slots=True)
class JobFeasibilityRequest:
    """Submitted workload shape for backend-specific capacity validation."""

    constraints: list[Constraint]
    replicas: int | None
    resources: job_pb2.ResourceSpecProto


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


class TaskBackend(Protocol):
    """Drives task execution + capacity reporting for a single cluster backend.

    The controller supplies a complete scheduling workspace and threads the
    per-user budget. Implementations dispatch backend-specific I/O and return
    plain data. No method reads or writes controller storage.
    """

    descriptor: BackendDescriptor
    """Stable identity, backend kind, and advertised capacity."""

    def initialize(self, request: BackendRecoveryRequest) -> BackendRecoveryResult:
        """Reconcile a controller checkpoint with the external provider."""
        ...

    def runtime_image(self, requested_image: str) -> str:
        """Resolve the container image used for a task request.

        Direct container backends can supply their configured default when the
        request leaves the image empty. Worker-daemon backends only know an
        explicitly requested image; their workers report build details.
        """
        ...

    def observe(self, request: BackendObservationRequest) -> BackendObservation:
        """Publish provider status from a complete controller fact snapshot."""
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

        Bounded I/O only. Every backend normalizes its execution mechanism into
        exact task-attempt updates and optional worker-health events. The
        controller owns snapshot reload, state-machine policy, liveness
        accounting, and persistence.
        """
        ...

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        """Provision capacity from controller-supplied demand and worker state."""
        ...

    def remove_capacity(self, request: RemoveCapacityRequest) -> RemoveCapacityResult:
        """Remove external capacity after the controller fences its workers."""
        ...

    def job_feasibility(self, request: JobFeasibilityRequest) -> str | None:
        """Return why a submitted shape can never run, or None if feasible."""
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
