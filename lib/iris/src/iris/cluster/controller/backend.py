# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskBackend: the contract every Iris execution backend implements.

The controller owns the database. Each tick it builds a placement-appropriate
:class:`BackendReconcileInput` from the DB, calls :meth:`TaskBackend.reconcile`,
and applies the returned :class:`BackendReconcileResult`. Backends perform
backend-specific I/O (worker-daemon RPC fan-out, ``kubectl apply``) but never
read or write the controller database — the interface is plain data in, plain
data out.

Two execution models exist, distinguished by :attr:`TaskBackend.placement`:

* ``PlacementOwner.IRIS_CONTROLLER`` — the Iris :class:`~iris.cluster.controller.scheduling.scheduler.Scheduler`
  assigns task→worker, then the backend fans the per-worker reconcile RPC out
  to worker daemons. The controller passes pre-built ``plans`` and applies the
  raw ``worker_results`` through ``ops.worker.apply_reconcile`` (which emits
  worker heartbeats and runs the ``WORKER_RECONCILE`` transition source).
* ``PlacementOwner.TASK_BACKEND`` — the backend (Kueue, and later slurmctld) owns
  placement. The controller passes the desired ``tasks_to_run`` plus the
  ``running_tasks`` snapshot; the backend converges its own resources and
  returns pre-computed ``updates`` applied through
  ``ops.task.apply_direct_provider_updates`` (``DIRECT_PROVIDER`` source).

The two apply paths are NOT interchangeable — they emit different effects — so
the controller selects between them on the declared ``placement`` capability
rather than on the concrete backend type. A new backend slots in by declaring
its capabilities, with no ``isinstance`` branches in the controller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Protocol

from finelog.client.log_client import Table
from finelog.types import LogWriterProtocol
from rigging.timing import Timestamp

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.autoscaler.state import AutoscalerState
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.reconcile.worker import ReconcileResult, WorkerReconcilePlan
from iris.cluster.controller.scheduling.decision import apply_preemptions, compute_diagnostics
from iris.cluster.controller.scheduling.policy import (
    GatedCandidates,
    RunningTaskInfo,
    SchedulingOrder,
    apply_scheduling_gates,
    compute_scheduling_order,
    inject_reservation_taints,
    inject_taint_constraints,
    preference_pass,
)
from iris.cluster.controller.scheduling.scheduler import JobRequirements, Scheduler, SchedulingContext
from iris.cluster.controller.schema import ReservationClaim
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.runtime.profile import IrisProfile
from iris.cluster.types import JobName, PendingTask, WorkerId, WorkerStatusMap
from iris.rpc import job_pb2, worker_pb2

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Communication failure with the execution backend."""


class ProviderUnsupportedError(ProviderError):
    """Operation not supported by this backend implementation."""


class PlacementOwner(StrEnum):
    """Who decides which node a task runs on."""

    IRIS_CONTROLLER = "iris_controller"
    """Iris schedules task→worker; the backend fans reconcile RPCs to daemons."""

    TASK_BACKEND = "task_backend"
    """The backend places tasks itself (Kueue, slurmctld); Iris does not schedule."""


@dataclass(frozen=True)
class BackendDescriptor:
    """Capabilities the dashboard uses to choose which panels to show.

    Derived from the live :class:`TaskBackend`; served (as JSON) by ``/auth/config``
    so the frontend tab list is data-driven rather than keyed on a provider-kind
    binary.
    """

    name: str
    placement: PlacementOwner
    manages_capacity: bool
    capabilities: list[str]


def backend_descriptor(backend: TaskBackend) -> BackendDescriptor:
    """Build the dashboard capability descriptor from a live backend.

    Capability strings drive which conditional dashboard tabs render:
    ``workers`` (Iris tracks worker daemons), ``autoscaler`` (Iris provisions
    capacity), ``cluster`` (the backend places tasks on its own cluster).
    """
    capabilities: list[str] = []
    if backend.placement is PlacementOwner.IRIS_CONTROLLER:
        capabilities.append("workers")  # Iris tracks worker daemons -> Workers (Fleet) tab
    if not backend.manages_capacity:
        capabilities.append("autoscaler")  # Iris autoscaler -> Autoscaler tab
    if backend.placement is PlacementOwner.TASK_BACKEND:
        capabilities.append("cluster")  # underlying cluster view -> Cluster tab
    return BackendDescriptor(
        name=backend.name,
        placement=backend.placement,
        manages_capacity=backend.manages_capacity,
        capabilities=capabilities,
    )


@dataclass(frozen=True)
class SchedulingEvent:
    """A scheduling event surfaced by the execution backend (e.g. k8s events)."""

    task_id: str
    attempt_id: int
    event_type: str
    reason: str
    message: str
    timestamp: Timestamp


@dataclass(frozen=True)
class ClusterCapacity:
    """Aggregate capacity reported by the execution backend."""

    schedulable_nodes: int
    total_cpu_millicores: int
    available_cpu_millicores: int
    total_memory_bytes: int
    available_memory_bytes: int


@dataclass(frozen=True)
class BackendReconcileInput:
    """Desired + observed task state handed to :meth:`TaskBackend.reconcile`.

    Each backend reads the subset matching its :attr:`TaskBackend.placement`;
    the controller leaves the other fields empty. No ``worker_id`` is attached
    to ``tasks_to_run`` (BACKEND placement chooses the node itself); ``plans``
    are already worker-bound (IRIS placement scheduled them).
    """

    # BACKEND placement: tasks to ensure running + the snapshot of running ones.
    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    running_tasks: list[RunningTaskEntry] = field(default_factory=list)
    # IRIS placement: pre-built per-worker reconcile plans + their addresses.
    plans: list[WorkerReconcilePlan] = field(default_factory=list)
    worker_addresses: dict[WorkerId, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendReconcileResult:
    """Observed outcome returned by :meth:`TaskBackend.reconcile`.

    BACKEND placement returns pre-computed ``updates`` (the backend mapped its
    own observations to task states). IRIS placement returns raw
    ``worker_results``; the controller converts them against its DB snapshot at
    apply time (uid resolution + worker-loss interpretation). ``scheduling_events``
    and ``capacity`` are shared and may be empty/None where a backend has none.
    """

    updates: list[TaskUpdate] = field(default_factory=list)
    worker_results: list[ReconcileResult] = field(default_factory=list)
    scheduling_events: list[SchedulingEvent] = field(default_factory=list)
    capacity: ClusterCapacity | None = None


@dataclass(frozen=True)
class PingResult:
    """Result of a liveness Ping to a single worker (IRIS placement)."""

    worker_id: WorkerId
    worker_address: str | None
    healthy: bool = True
    health_error: str = ""
    error: str | None = None


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
    """Built by ``build_scheduling_context`` (un-tainted workers + raw reads)."""
    claims: dict[WorkerId, ReservationClaim]
    """Reservation claims from ``refresh_reservation_claims``."""
    running_for_preemption: list[RunningTaskInfo]
    """Band/value of the currently-running tasks the preemption pass may evict."""
    max_tasks_per_job_per_cycle: int
    trace: bool = False
    """Whether to emit the per-phase scheduling trace logs this cycle."""


@dataclass(frozen=True)
class ScheduleResult:
    """PlacementOwner decisions returned by :meth:`TaskBackend.schedule`.

    Pure data: the controller commits ``assignments`` (``ops.task.assign``),
    ``preemptions`` (``finalize`` PREEMPT), and ``unschedulable`` (``finalize``
    UNSCHEDULABLE). ``diagnostics`` and ``post_taint_context`` carry the cached
    dashboard state the controller exposes via ``get_job_scheduling_diagnostics``
    / ``last_scheduling_context``.
    """

    assignments: list[Assignment] = field(default_factory=list)
    preemptions: list[TerminalDecision] = field(default_factory=list)
    # Expired/deadline-exceeded pending-task rows; the controller marks them
    # UNSCHEDULABLE (each row carries ``scheduling_timeout_ms``).
    unschedulable: list[PendingTask] = field(default_factory=list)
    diagnostics: dict[str, str] = field(default_factory=dict)
    # Post-taint context (or the un-tainted context when no claims were active),
    # surfaced for dashboard diagnostics. None only for the empty K8s result.
    post_taint_context: SchedulingContext | None = None


@dataclass(frozen=True)
class CapacityInput:
    """The read-only state a :class:`TaskBackend` needs to make capacity
    (autoscaling) decisions for one tick."""

    worker_status_map: WorkerStatusMap
    """Per-worker idle/running state (``_build_worker_status_map``)."""
    demand_entries: list[DemandEntry]
    """Unmet demand grouped by requirement (``compute_demand_entries``)."""


@dataclass(frozen=True)
class CapacityResult:
    """Outcome of :meth:`TaskBackend.manage_capacity`.

    Carries the autoscaler's current tracked state for the controller to mirror
    into the ``slices`` / ``scaling_groups`` tables. Empty for backends that
    provision their own capacity (k8s).
    """

    state: AutoscalerState = field(default_factory=AutoscalerState)


@dataclass(frozen=True)
class WorkersFailedResult:
    """Outcome of :meth:`TaskBackend.on_workers_failed`.

    The Iris autoscaler tears down the failed workers' slices, so their healthy
    siblings must be failed too. ``state`` carries the post-teardown autoscaler
    state for the controller to persist.
    """

    sibling_worker_ids: list[WorkerId] = field(default_factory=list)
    state: AutoscalerState = field(default_factory=AutoscalerState)


def run_scheduling_decision(scheduler: Scheduler, snapshot: ScheduleInput) -> ScheduleResult:
    """Run the full Iris scheduling decision pipeline over a DB-less snapshot.

    Stages: gates → order → reservation taints → preference pass →
    ``find_assignments`` → preemption pass. Returns the placement decisions plus
    the diagnostics/context the controller caches. Does no I/O — every input
    comes from ``snapshot`` and every output is plain data.
    """
    ctx = snapshot.context
    claims = snapshot.claims
    trace = snapshot.trace

    gated = apply_scheduling_gates(
        ctx,
        claims,
        max_tasks_per_job_per_cycle=snapshot.max_tasks_per_job_per_cycle,
        trace=trace,
    )
    if not gated.schedulable_task_ids:
        # No work to place. Expired tasks (if any) still flow back so the
        # controller can mark them UNSCHEDULABLE; the un-tainted context is the
        # diagnostics snapshot for this tick.
        return ScheduleResult(unschedulable=list(gated.expired_tasks), post_taint_context=ctx)

    order = compute_scheduling_order(ctx, gated, trace=trace)
    all_assignments, context, tainted_jobs = apply_placements(scheduler, order, gated, ctx, claims, trace=trace)
    preemptions = apply_preemptions(order, tainted_jobs, all_assignments, snapshot.running_for_preemption, context)
    diagnostics = compute_diagnostics(scheduler, context, tainted_jobs, all_assignments, order.ordered_task_ids)

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
        post_taint_context=context,
    )


def apply_placements(
    scheduler: Scheduler,
    order: SchedulingOrder,
    gated: GatedCandidates,
    ctx: SchedulingContext,
    claims: dict[WorkerId, ReservationClaim],
    *,
    trace: bool,
) -> tuple[list[tuple[JobName, WorkerId]], SchedulingContext, dict[JobName, JobRequirements]]:
    """Preference + normal assignment passes over a shared (post-taint) context.

    Reservation taints are injected here so gates/order/diagnostics saw the
    un-tainted workers. When there are no claims the un-tainted ``ctx`` is reused
    to avoid an index rebuild.
    """
    modified_jobs = inject_taint_constraints(gated.jobs, gated.has_reservation, gated.has_direct_reservation)

    if claims:
        modified_workers = inject_reservation_taints(list(ctx.workers), claims)
        building_counts = {wid: cap.building_task_count for wid, cap in ctx.capacities.items()}
        ctx.pending_tasks = list(order.ordered_task_ids)
        context = ctx.evolve_with_workers(
            workers=modified_workers,
            jobs=modified_jobs,
            building_counts=building_counts,
            max_building_tasks=scheduler.max_building_tasks_per_worker,
        )
    else:
        ctx.pending_tasks = list(order.ordered_task_ids)
        ctx.jobs = modified_jobs
        context = ctx

    if trace:
        logger.info(
            "[TRACE] Phase 4 context: %d workers, %d pending tasks, %d jobs",
            len(context.capacities),
            len(context.pending_tasks),
            len(context.jobs),
        )

    # Soft preference — steer reservation tasks toward claimed workers. Skips
    # coscheduled jobs (they need atomic all-or-nothing via find_assignments).
    preference_assignments = preference_pass(context, gated.has_reservation, claims)
    result = scheduler.find_assignments(context)
    all_assignments = preference_assignments + result.assignments
    if trace:
        logger.info(
            "[TRACE] Phase 5 assignments: %d total (%d preferred, %d normal)",
            len(all_assignments),
            len(preference_assignments),
            len(result.assignments),
        )
    return all_assignments, context, modified_jobs


class TaskBackend(Protocol):
    """Drives task execution + capacity reporting for a single cluster backend.

    Implementations dispatch backend-specific I/O and return plain data; they
    never touch the controller database.
    """

    name: str
    """Stable identifier, e.g. ``"gcp"``, ``"coreweave"``, later ``"slurm-stanford"``."""

    placement: ClassVar[PlacementOwner]
    """Who schedules task→node (selects the controller's input-build + apply path)."""

    manages_capacity: ClassVar[bool]
    """True when the backend provisions its own nodes (k8s cluster autoscaler);
    False when the Iris :class:`Autoscaler` provisions capacity for it."""

    autoscaler: Autoscaler | None
    """The Iris :class:`Autoscaler` driving capacity, or None for backends that
    manage their own (k8s) or have no scale groups. Read-only handle the
    controller exposes for dashboard/status RPCs; capacity is driven through
    :meth:`manage_capacity`, never this attribute."""

    def reconcile(self, batch: BackendReconcileInput) -> BackendReconcileResult:
        """Converge the backend toward ``batch`` and report observed state."""
        ...

    def schedule(self, snapshot: ScheduleInput) -> ScheduleResult:
        """Decide task→worker placement from a DB-less snapshot.

        IRIS placement runs the full Iris scheduling pipeline; BACKEND placement
        (Kueue, slurmctld) returns an empty result — the backend places tasks
        itself. No database access: snapshot in, decisions out.
        """
        ...

    def manage_capacity(self, snapshot: CapacityInput) -> CapacityResult:
        """Evaluate scaling decisions from a DB-less snapshot.

        IRIS placement drives the Iris :class:`Autoscaler`; BACKEND placement
        returns an empty result (the cluster autoscaler / Kueue handle capacity).
        Returns the autoscaler state for the controller to persist — no DB access.
        """
        ...

    def on_workers_failed(self, worker_ids: list[WorkerId]) -> WorkersFailedResult:
        """Tear down slices for definitively-failed workers and return siblings.

        IRIS placement terminates the failed workers' slices via the autoscaler
        and returns the sibling worker ids the controller must fail plus the
        post-teardown state to persist. BACKEND placement returns an empty result.
        """
        ...

    def attach_autoscaler(self, autoscaler: Autoscaler) -> None:
        """Attach the Iris autoscaler that provisions capacity for this backend.

        Called once by the controller's main() after construction (mirrors
        :meth:`set_log_sink`). Only invoked on backends with
        ``manages_capacity`` False; capacity-managing backends (k8s) never
        receive one.
        """
        ...

    def capacity(self) -> ClusterCapacity | None:
        """Latest aggregate capacity, or None if the backend does not report it."""
        ...

    def ping_workers(self, workers: list[tuple[WorkerId, str | None]]) -> list[PingResult]:
        """Liveness-probe worker daemons (IRIS placement). May be a no-op."""
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

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        """Evict cached connection state for a definitively-failed worker."""
        ...

    def set_log_sink(
        self,
        log_client: LogWriterProtocol,
        task_stats_table: Table,
        profile_table: Table[IrisProfile],
    ) -> None:
        """Inject the finelog handles the controller resolves after connecting.

        Backends without a worker daemon (BACKEND placement) collect logs and
        write resource/profile samples directly to finelog. Daemon-backed
        backends ignore these — the worker writes its own rows.
        """
        ...
