# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskBackend: the contract every Iris execution backend implements.

The controller owns the database. Each tick it composes a DB-less
:class:`~iris.cluster.controller.reads.ControlSnapshot` (and, for scheduling, a
:class:`ScheduleInput`) and drives the backend through three uniform methods —
:meth:`TaskBackend.schedule`, :meth:`TaskBackend.reconcile`,
:meth:`TaskBackend.autoscale` — each returning its own method-specific result
(:class:`ScheduleResult` / :class:`ReconcileResult` / :class:`AutoscaleResult`).
Backends perform backend-specific I/O (worker-daemon RPC fan-out, ``kubectl
apply``) but never read or write the controller database — plain data in, plain
data out.

The controller calls all three methods uniformly across backends and never
branches on backend type: a backend no-ops where a phase doesn't apply (the
worker-daemon backend's ``schedule`` runs the Iris scheduler while a cluster
backend returns empty; the cluster backend's ``reconcile`` reconciles pods while
it owns its own capacity, so its ``autoscale`` returns empty). Within a method,
the apply path dispatches on which result field is populated — e.g. a
worker-daemon ``reconcile`` returns ``worker_results`` + ``health_events`` while
a cluster ``reconcile`` returns ``updates``.

:attr:`TaskBackend.capabilities` is a pure descriptor for the dashboard tab list
and on-demand service-RPC routing (worker-daemon vs direct-pod exec). One narrow
exception gates the per-tick path: a ``CLUSTER_VIEW`` backend owns placement, so
the controller drains the dispatch queue (a DB write it owns) into the reconcile
snapshot for it. Worker liveness is OBSERVED by worker-daemon backends and
folded by the controller; cluster backends have no Iris workers, so they emit no
health events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Protocol

from finelog.client.log_client import Table
from finelog.types import LogWriterProtocol

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.autoscaler.state import AutoscalerState
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
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
    inject_reservation_taints,
    inject_taint_constraints,
    preference_pass,
)
from iris.cluster.controller.scheduling.scheduler import JobRequirements, Scheduler, SchedulingContext
from iris.cluster.controller.schema import ReservationClaim
from iris.cluster.controller.worker_health import WorkerHealthEvent
from iris.cluster.types import JobName, PendingTask, WorkerId
from iris.rpc import job_pb2, worker_pb2

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Communication failure with the execution backend."""


class ProviderUnsupportedError(ProviderError):
    """Operation not supported by this backend implementation."""


class BackendCapability(StrEnum):
    """A descriptor flag the dashboard and on-demand RPC routing key on.

    Mostly metadata: the controller calls ``schedule``/``reconcile``/``autoscale``
    uniformly regardless of these flags. The one per-tick exception is
    ``CLUSTER_VIEW``, which tells the controller to drain the dispatch queue into
    the reconcile snapshot (a DB write the placement-owning backend can't do).
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


def backend_descriptor(backend: TaskBackend) -> BackendDescriptor:
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
    """Built by ``build_scheduling_context`` (un-tainted workers + raw reads)."""
    claims: dict[WorkerId, ReservationClaim]
    """Reservation claims from ``refresh_reservation_claims``."""
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
    post_taint_context: SchedulingContext | None = None
    """Post-taint (or un-tainted) context cached for dashboard diagnostics."""


@dataclass(frozen=True)
class ReconcileResult:
    """What :meth:`TaskBackend.reconcile` observed this tick.

    The two carriers reflect the backend kind, not different methods: a
    worker-daemon backend populates ``worker_results`` + ``health_events``; a
    cluster (pod) backend populates ``updates``. The controller's reconcile apply
    path dispatches on which is non-empty.
    """

    worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]] = field(default_factory=list)
    """Worker-daemon reconcile outcomes, each paired with the plan that produced
    it. The pairing is required because resolving these to task state needs the
    DB/overlay (it is not a backend-side concern), so the backend cannot
    pre-convert them to neutral ``updates``."""
    updates: list[TaskUpdate] = field(default_factory=list)
    """Neutral task-state updates from a direct (e.g. Kubernetes) provider."""
    health_events: list[WorkerHealthEvent] = field(default_factory=list)
    """Per-worker liveness the backend OBSERVED (REACHED / UNREACHABLE). The
    controller folds these through the single ``WorkerHealthTracker.apply``."""


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

    # Residual demand is computed alongside the assignments from the same
    # snapshot and the same Scheduler instance: a limits-free capacity-fit over
    # the pending tasks. Tasks this tick retires as UNSCHEDULABLE (deadline
    # expired) are excluded so the autoscaler is never asked to provision for a
    # job the same tick is failing. ``apply_scheduling_gates`` above only reads
    # ``ctx``; this still runs before ``apply_placements`` mutates it.
    residual_demand = compute_demand_entries(
        ctx, scheduler, claims, exclude_task_ids={t.task_id for t in gated.expired_tasks}
    )

    if not gated.schedulable_task_ids:
        # No work to place. Expired tasks (if any) still flow back so the
        # controller can mark them UNSCHEDULABLE; the un-tainted context is the
        # diagnostics snapshot for this tick.
        return ScheduleResult(
            unschedulable=list(gated.expired_tasks),
            post_taint_context=ctx,
            residual_demand=residual_demand,
        )

    order = compute_scheduling_order(ctx, gated, trace=trace)
    all_assignments, context, tainted_jobs = apply_placements(scheduler, order, gated, ctx, claims, trace=trace)
    preemptions = apply_preemptions(order, tainted_jobs, all_assignments, ctx.running_for_preemption, context)
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
        residual_demand=residual_demand,
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
    modified_jobs = inject_taint_constraints(
        gated.jobs,
        gated.has_reservation,
        gated.has_direct_reservation,
        ctx.reservation_zones_by_job,
    )

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

    def schedule(self, snapshot: ScheduleInput) -> ScheduleResult:
        """Decide task→worker placement from a DB-less snapshot (pure decision).

        Worker-daemon backends run the full Iris scheduling pipeline; cluster
        backends (Kueue, slurmctld) return an empty result — they place tasks
        themselves. No database access: snapshot in, decisions out.
        """
        ...

    def reconcile(self, snapshot: ControlSnapshot) -> ReconcileResult:
        """Converge the backend toward the desired state and report observations.

        Bounded I/O. Worker-daemon backends fan the reconcile RPC out and return
        ``worker_results`` plus the per-worker liveness they OBSERVED
        (``health_events``); cluster backends apply/poll pods and return neutral
        ``updates``. The backend never decides a worker dead — the controller
        folds ``health_events`` through ``WorkerHealthTracker.apply``.
        """
        ...

    def autoscale(
        self,
        snapshot: ControlSnapshot,
        residual_demand: list[DemandEntry],
        dead_workers: list[WorkerId],
    ) -> AutoscaleResult:
        """Provision capacity for unmet demand, OR tear down dead workers.

        Bounded I/O. With ``dead_workers`` set, the backend terminates those
        workers' slices AND their healthy siblings and returns the full set as
        ``removed_workers`` (no provisioning this call). Otherwise it runs one
        scaling cycle against ``residual_demand``. Either way it returns its
        tracked ``autoscaler_state`` for the controller to persist. Backends that
        manage their own capacity (k8s) return an empty result.
        """
        ...

    def attach_autoscaler(self, autoscaler: Autoscaler) -> None:
        """Attach the Iris autoscaler that provisions capacity for this backend.

        Called once by the controller's main() after construction (mirrors
        :meth:`set_log_sink`). Only invoked on backends carrying
        :attr:`BackendCapability.IRIS_AUTOSCALER`; capacity-managing backends
        (k8s) never receive one.
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

    def set_log_sink(
        self,
        log_client: LogWriterProtocol,
        task_stats_table: Table,
        profile_table: Table,
    ) -> None:
        """Inject the finelog handles the controller resolves after connecting.

        Backends without a worker daemon collect logs and write resource/profile
        samples directly to finelog. Daemon-backed backends ignore these — the
        worker writes its own rows.
        """
        ...

    def close(self) -> None:
        """Release backend-owned resources at controller shutdown.

        Called from Controller.stop(). Capacity-managing backends shut down
        their attached autoscaler here (terminating VMs, stopping the platform);
        others close any connections or collectors they own.
        """
        ...
