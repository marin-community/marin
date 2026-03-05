# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import logging
import queue
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import sleep
from typing import Protocol

import uvicorn

from iris.chaos import chaos
from iris.cluster.controller.autoscaler import Autoscaler, DemandEntry
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.events import TaskAssignedEvent, TaskStateChangedEvent
from iris.cluster.controller.scheduler import (
    JobRequirements,
    Scheduler,
    SchedulingContext,
    WorkerSnapshot,
    _evaluate_constraint,
    device_compatible,
    device_variant_matches,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.state import (
    HEARTBEAT_FAILURE_THRESHOLD,
    ControllerJob,
    ControllerState,
    ControllerTask,
    ControllerWorker,
    HeartbeatSnapshot,
    ReservationClaim,
)
from iris.cluster.types import (
    REGION_ATTRIBUTE_KEY,
    AttributeValue,
    DeviceType,
    JobName,
    VmWorkerStatus,
    VmWorkerStatusMap,
    WorkerId,
    get_device_type,
    get_device_type_enum,
    get_device_variant,
    normalize_constraints,
)
from iris.cluster.controller.snapshot import (
    SnapshotResult,
    create_snapshot,
    read_latest_snapshot,
    restore_scaling_group,
    restore_snapshot,
    restore_tracked_workers,
    write_snapshot,
)
from iris.logging import get_global_buffer, slow_log
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import cluster_pb2, snapshot_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.time_utils import Duration, ExponentialBackoff, RateLimiter, Timer

logger = logging.getLogger(__name__)

# Sentinel for dry-run scheduling with per-worker limits disabled.
_UNLIMITED = sys.maxsize

_SLOW_HEARTBEAT_MS = 5000
_HEALTH_SUMMARY_INTERVAL = 6  # every ~30s at 5s heartbeat interval

# Taint attribute injected onto claimed workers to prevent non-reservation
# jobs from landing on them.  Non-reservation jobs get a NOT_EXISTS constraint
# for this key; reservation jobs do not, so they naturally prefer claimed
# workers (which appear first in the worker list).
RESERVATION_TAINT_KEY = "reservation-job"


def job_requirements_from_job(job: ControllerJob) -> JobRequirements:
    """Convert a ControllerJob to scheduler-compatible JobRequirements."""
    return JobRequirements(
        resources=job.request.resources,
        constraints=list(job.request.constraints),
        is_coscheduled=job.is_coscheduled,
        coscheduling_group_by=job.coscheduling_group_by,
    )


def compute_demand_entries(
    state: ControllerState,
    scheduler: Scheduler | None = None,
    workers: list[WorkerSnapshot] | None = None,
    reservation_claims: dict[WorkerId, ReservationClaim] | None = None,
) -> list[DemandEntry]:
    """Compute demand entries for the autoscaler from controller state.

    All pending tasks — both real and reservation holder — flow through a
    single unified path. Every task participates in the dry-run and generates
    demand through the same logic using its job's resource spec.

    Holder tasks consume zero resources on workers, so they won't be absorbed
    by the dry-run when workers have available capacity. This ensures they
    always generate demand, keeping reserved capacity alive via the
    autoscaler. The taint/constraint mechanism ensures only peer jobs can
    actually use the reserved workers.

    .. note::

        Demand from holder tasks and parent real tasks is additive. On a cold
        start with N reservation entries and M real tasks this reports N + M
        demand entries, which may overprovision. In practice reservations are
        used when the parent job does not request its own resources, so the
        additive behavior is correct. If that changes, a dedup path (e.g.
        ``max(real_pending, holders)``) should be added here.

    Args:
        state: Controller state to read pending tasks and jobs from.
        scheduler: Scheduler for dry-run pass. If None, skips dry-run.
        workers: Available workers for dry-run. If None, skips dry-run.
        reservation_claims: Reservation claims to apply taint injection in the
            dry-run, matching the real scheduling path. If None, no taints applied.
    """
    demand_entries: list[DemandEntry] = []

    # Collect all schedulable pending tasks, grouped by job.
    tasks_by_job: dict[JobName, list[ControllerTask]] = defaultdict(list)
    all_schedulable: list[ControllerTask] = []
    for task in state.peek_pending_tasks():
        if not task.can_be_scheduled():
            continue
        job = state.get_job(task.job_id)
        if not job:
            continue
        tasks_by_job[task.job_id].append(task)
        all_schedulable.append(task)

    # Build job requirements once, shared between dry-run and demand emission.
    # Also track which jobs have reservations so we can apply taint injection.
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    for task in all_schedulable:
        if task.job_id not in jobs:
            job = state.get_job(task.job_id)
            if job:
                jobs[task.job_id] = job_requirements_from_job(job)
                if job.request.HasField("reservation"):
                    has_reservation.add(task.job_id)
                elif _find_reservation_ancestor(state, task.job_id) is not None:
                    has_reservation.add(task.job_id)

    # Dry-run scheduling with building/assignment limits disabled.
    # All tasks participate — holders and real tasks alike.
    absorbed_task_ids: set[JobName] = set()
    if scheduler is not None and workers is not None and workers:
        building_counts = state.snapshot_building_counts()
        task_ids = [t.task_id for t in all_schedulable]
        claims = reservation_claims or {}
        dry_run_workers = _inject_reservation_taints(workers, claims)
        dry_run_jobs = _inject_taint_constraints(jobs, has_reservation)

        context = scheduler.create_scheduling_context(
            dry_run_workers,
            building_counts=building_counts,
            pending_tasks=task_ids,
            jobs=dry_run_jobs,
            max_building_tasks=_UNLIMITED,
            max_assignments_per_worker=_UNLIMITED,
        )
        result = scheduler.find_assignments(context)
        for task_id, _ in result.assignments:
            absorbed_task_ids.add(task_id)

    # Emit demand for all unabsorbed tasks through a single path.
    for job_id, tasks in tasks_by_job.items():
        job = state.get_job(job_id)
        if not job:
            continue
        if job.is_finished():
            continue

        device = job.request.resources.device
        device_type = get_device_type_enum(device)
        device_variant = get_device_variant(device) if device_type != DeviceType.CPU else None
        preemptible_pref: bool | None = None
        required_regions: frozenset[str] | None = None
        required_zones: frozenset[str] | None = None
        invalid_reason: str | None = None
        try:
            normalized = normalize_constraints(job.request.constraints)
            preemptible_pref = normalized.preemptible
            required_regions = normalized.required_regions
            required_zones = normalized.required_zones
        except ValueError as e:
            invalid_reason = f"invalid_constraints: {e}"

        if job.is_coscheduled:
            remaining_ids = []
            for t in tasks:
                if t.task_id in absorbed_task_ids:
                    continue
                remaining_ids.append(t.task_id.to_wire())
            if remaining_ids:
                demand_entries.append(
                    DemandEntry(
                        task_ids=remaining_ids,
                        coschedule_group_id=job.job_id.to_wire(),
                        device_type=device_type,
                        device_variant=device_variant,
                        constraints=list(job.request.constraints),
                        resources=job.request.resources,
                        preemptible=preemptible_pref,
                        required_regions=required_regions,
                        required_zones=required_zones,
                        invalid_reason=invalid_reason,
                    )
                )
            continue

        for task in tasks:
            if task.task_id in absorbed_task_ids:
                continue
            demand_entries.append(
                DemandEntry(
                    task_ids=[task.task_id.to_wire()],
                    coschedule_group_id=None,
                    device_type=device_type,
                    device_variant=device_variant,
                    constraints=list(job.request.constraints),
                    resources=job.request.resources,
                    preemptible=preemptible_pref,
                    required_regions=required_regions,
                    required_zones=required_zones,
                    invalid_reason=invalid_reason,
                )
            )

    return demand_entries


def _worker_matches_reservation_entry(
    worker: ControllerWorker,
    res_entry: cluster_pb2.ReservationEntry,
) -> bool:
    """Check if a worker is eligible for a reservation entry.

    Matches device type, device variant, and all constraints.
    """
    entry_device_type = get_device_type(res_entry.resources.device)
    if not device_compatible(entry_device_type, worker.device_type):
        return False

    entry_variant = get_device_variant(res_entry.resources.device)
    if entry_variant and entry_variant != "auto":
        if not device_variant_matches(entry_variant, worker.device_variant):
            return False

    for constraint in res_entry.constraints:
        attr = worker.attributes.get(constraint.key)
        if not _evaluate_constraint(attr, constraint):
            return False

    return True


def _inject_reservation_taints(
    workers: list[ControllerWorker],
    claims: dict[WorkerId, ReservationClaim],
) -> list[ControllerWorker]:
    """Create modified worker copies with reservation taints and prioritization.

    Claimed workers receive a ``reservation-job`` attribute set to the claiming
    job's ID.  The returned list is ordered with claimed workers first so that
    reservation jobs (which have no NOT_EXISTS constraint) naturally pick from
    their claimed workers before unclaimed ones.

    Workers are never mutated — ``dataclasses.replace`` produces shallow copies.
    """
    if not claims:
        return workers

    claimed: list[ControllerWorker] = []
    unclaimed: list[ControllerWorker] = []
    for worker in workers:
        claim = claims.get(worker.worker_id)
        if claim is not None:
            modified_attrs = dict(worker.attributes)
            modified_attrs[RESERVATION_TAINT_KEY] = AttributeValue(claim.job_id)
            claimed.append(replace(worker, attributes=modified_attrs))
        else:
            unclaimed.append(worker)
    return claimed + unclaimed


def _inject_taint_constraints(
    jobs: dict[JobName, JobRequirements],
    has_reservation: set[JobName],
) -> dict[JobName, JobRequirements]:
    """Add NOT_EXISTS reservation-job constraint to non-reservation jobs.

    This prevents normal jobs from being scheduled onto claimed workers.
    Reservation jobs are left unchanged — they can use both claimed and
    unclaimed workers (the reservation is a floor, not a ceiling).
    """
    if not has_reservation and not jobs:
        return jobs

    taint_constraint = cluster_pb2.Constraint(
        key=RESERVATION_TAINT_KEY,
        op=cluster_pb2.CONSTRAINT_OP_NOT_EXISTS,
    )

    modified: dict[JobName, JobRequirements] = {}
    for job_id, req in jobs.items():
        if job_id in has_reservation:
            modified[job_id] = req
        else:
            modified[job_id] = replace(
                req,
                constraints=[*list(req.constraints), taint_constraint],
            )
    return modified


def _find_reservation_ancestor(state: ControllerState, job_id: JobName) -> JobName | None:
    """Walk up the job hierarchy to find the nearest ancestor with a reservation.

    Returns the ancestor's JobName, or None if no ancestor has a reservation.
    """
    current = job_id.parent
    while current is not None:
        ancestor = state.get_job(current)
        if ancestor is not None and ancestor.request.HasField("reservation"):
            return current
        current = current.parent
    return None


def _reservation_region_constraints(
    job_id_wire: str,
    claims: dict[WorkerId, ReservationClaim],
    state: ControllerState,
    existing_constraints: list[cluster_pb2.Constraint],
) -> list[cluster_pb2.Constraint]:
    """Derive region constraints from claimed reservation workers.

    When a reservation job has no explicit region constraint, this function
    extracts the region attributes of claimed workers and returns the existing
    constraints plus an injected region constraint.  If the job already has a
    region constraint, or if claimed workers lack region attributes, the
    existing constraints are returned unchanged.
    """
    if any(c.key == REGION_ATTRIBUTE_KEY for c in existing_constraints):
        return existing_constraints

    regions: set[str] = set()
    for worker_id, claim in claims.items():
        if claim.job_id != job_id_wire:
            continue
        worker = state.get_worker(worker_id)
        if worker is None:
            continue
        region_attr = worker.attributes.get(REGION_ATTRIBUTE_KEY)
        if region_attr is not None:
            regions.add(str(region_attr.value))

    if not regions:
        return existing_constraints

    region_list = sorted(regions)
    if len(region_list) == 1:
        region_constraint = cluster_pb2.Constraint(
            key=REGION_ATTRIBUTE_KEY,
            op=cluster_pb2.CONSTRAINT_OP_EQ,
            value=cluster_pb2.AttributeValue(string_value=region_list[0]),
        )
    else:
        region_constraint = cluster_pb2.Constraint(
            key=REGION_ATTRIBUTE_KEY,
            op=cluster_pb2.CONSTRAINT_OP_IN,
            values=[cluster_pb2.AttributeValue(string_value=r) for r in region_list],
        )

    return [*existing_constraints, region_constraint]


def _preference_pass(
    context: SchedulingContext,
    has_reservation: set[JobName],
    claims: dict[WorkerId, ReservationClaim],
) -> list[tuple[JobName, WorkerId]]:
    """Try to assign reservation-job tasks to their claimed workers first.

    Iterates reservation-job tasks and, for each, checks the (small) set of
    workers claimed for that job. If a claimed worker has capacity, the task
    is assigned immediately — deducting resources and marking the worker as
    scheduled in the shared context so the subsequent find_assignments pass
    sees the updated state.

    Coscheduled jobs are skipped because they require atomic all-or-nothing
    assignment across a worker group.

    Returns the list of (task_id, worker_id) assignments made.
    """
    if not has_reservation or not claims:
        return []

    # Reverse index: job_wire -> list of claimed worker IDs
    claimed_by_job: dict[str, list[WorkerId]] = defaultdict(list)
    for wid, claim in claims.items():
        claimed_by_job[claim.job_id].append(wid)

    assignments: list[tuple[JobName, WorkerId]] = []
    preference_scheduled: set[JobName] = set()

    for task_id in context.pending_tasks:
        job_id = task_id.parent
        if job_id is None or job_id not in has_reservation:
            continue

        req = context.jobs.get(job_id)
        if req is None or req.is_coscheduled:
            continue

        job_wire = job_id.to_wire()
        for wid in claimed_by_job.get(job_wire, ()):
            if context.assignment_counts.get(wid, 0) >= context.max_assignments_per_worker:
                continue
            capacity = context.capacities.get(wid)
            if capacity is None:
                continue
            if capacity.can_fit(req) is not None:
                continue
            capacity.deduct(req)
            context.assignment_counts[wid] = context.assignment_counts.get(wid, 0) + 1
            assignments.append((task_id, wid))
            preference_scheduled.add(task_id)
            break

    # Remove preference-assigned tasks from pending so find_assignments skips them.
    if preference_scheduled:
        context.pending_tasks = [t for t in context.pending_tasks if t not in preference_scheduled]

    return assignments


class WorkerStubFactory(Protocol):
    """Factory for getting worker RPC stubs."""

    def get_stub(self, address: str) -> WorkerServiceClientSync: ...
    def evict(self, address: str) -> None: ...


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
            self._stubs.pop(address, None)


@dataclass
class ControllerConfig:
    """Controller configuration."""

    host: str = "127.0.0.1"
    """Host to bind the HTTP server to."""

    port: int = 0
    """Port to bind the HTTP server to. Use 0 for auto-assign."""

    bundle_prefix: str | None = None
    """URI prefix for storing job bundles (e.g., gs://bucket/path or file:///var/cache/iris/bundles).
    Uses fsspec for storage, so supports both GCS and local filesystems. For distributed deployments,
    use a GCS path so workers can download bundles."""

    scheduler_interval: Duration = field(default_factory=lambda: Duration.from_seconds(0.5))
    """How often to run the scheduling loop."""

    heartbeat_interval: Duration = field(default_factory=lambda: Duration.from_seconds(5.0))
    """How often to send heartbeats to workers."""

    max_dispatch_parallelism: int = 32
    """Maximum number of concurrent RPC dispatch operations."""

    heartbeat_failure_threshold: int = HEARTBEAT_FAILURE_THRESHOLD
    """Consecutive heartbeat failures before marking worker as dead."""

    autoscaler_enabled: bool = False
    worker_access_address: str = ""

    checkpoint_interval: Duration | None = None
    """If set, take a periodic best-effort snapshot this often.
    Runs in the autoscaler loop thread; does not pause scheduling."""

    log_dir: Path | None = None
    """Persistent directory for task log files. When None, uses a temp dir."""


class Controller:
    """Unified controller managing all components and lifecycle.

    Runs three background loops:
    - Scheduling loop: finds task assignments, checks worker timeouts
    - Heartbeat loop: sends heartbeat RPCs to workers, delivering buffered dispatches/kills
    - Autoscaler loop: evaluates scaling decisions, manages slice lifecycle

    Each loop runs on its own thread so blocking operations in one don't
    stall the others.

    Example:
        ```python
        config = ControllerConfig(port=8080)
        controller = Controller(
            config=config,
            worker_stub_factory=RpcWorkerStubFactory(),
        )
        controller.start()
        try:
            job_id = controller.launch_job(request)
            status = controller.get_job_status(job_id)
        finally:
            controller.stop()
        ```

    Args:
        config: Controller configuration
        worker_stub_factory: Factory for creating worker RPC stubs
        autoscaler: Optional Autoscaler for managing VM slices. If provided,
                   the controller will run it in a background thread.
    """

    def __init__(
        self,
        config: ControllerConfig,
        worker_stub_factory: WorkerStubFactory,
        autoscaler: "Autoscaler | None" = None,
        threads: ThreadContainer | None = None,
    ):
        if not config.bundle_prefix:
            raise ValueError(
                "bundle_prefix is required. Set via ControllerConfig.bundle_prefix. "
                "Example: bundle_prefix='gs://my-bucket/iris/bundles'"
            )

        self._config = config
        self.stub_factory = worker_stub_factory

        self._state = ControllerState(
            heartbeat_failure_threshold=config.heartbeat_failure_threshold,
            log_dir=config.log_dir,
        )
        self._scheduler = Scheduler()
        self._service = ControllerServiceImpl(
            self._state,
            self,
            bundle_prefix=config.bundle_prefix,
            log_buffer=get_global_buffer(),
        )
        self._dashboard = ControllerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        # Background loop state
        self._threads = threads if threads is not None else get_thread_container()
        self._wake_event = threading.Event()
        self._heartbeat_event = threading.Event()
        self._server: uvicorn.Server | None = None
        self._scheduling_thread: ManagedThread | None = None
        self._heartbeat_thread: ManagedThread | None = None
        self._autoscaler_thread: ManagedThread | None = None

        # Thread pool for parallel heartbeat dispatch, owned by the ThreadContainer
        # so it is shut down automatically during stop().
        self._dispatch_executor = self._threads.spawn_executor(
            max_workers=config.max_dispatch_parallelism,
            prefix="dispatch",
        )

        # Autoscaler (passed in, configured in start() if provided)
        self._autoscaler: Autoscaler | None = autoscaler

        # Reservation claims: worker_id -> ReservationClaim.
        # Populated each scheduling cycle by _claim_workers_for_reservations.
        self._reservation_claims: dict[WorkerId, ReservationClaim] = {}

        self._heartbeat_iteration = 0

        # Set to True once start() is called. Used to gate operations that
        # are only valid before the controller loops begin (e.g. LoadCheckpoint).
        self._started = False

        # Checkpoint coordination flag. When set, scheduling and autoscaler
        # loops skip their work so the snapshot captures a quiescent state.
        self._checkpoint_in_progress = False

        # Serializes heartbeat rounds against checkpoint snapshots so that
        # begin_checkpoint cannot fire while dispatches from begin_heartbeat()
        # are in flight (but not yet applied by complete_heartbeat).
        self._heartbeat_lock = threading.Lock()

        # Rate-limits periodic (best-effort) checkpoint writes.
        # None when checkpoint_interval is not configured.
        self._periodic_checkpoint_limiter: RateLimiter | None = (
            RateLimiter(interval_seconds=config.checkpoint_interval.to_seconds())
            if config.checkpoint_interval is not None
            else None
        )

    def wake(self) -> None:
        """Signal the controller loop to run immediately.

        Called when events occur that may make scheduling possible:
        - New job submitted
        - New worker registered
        - Task finished (freeing capacity)
        """
        self._wake_event.set()

    @property
    def started(self) -> bool:
        """Whether the controller loops have been started."""
        return self._started

    def start(self) -> None:
        """Start main controller loop, dashboard server, and optionally autoscaler."""
        self._started = True
        self._scheduling_thread = self._threads.spawn(self._run_scheduling_loop, name="scheduling-loop")
        self._heartbeat_thread = self._threads.spawn(self._run_heartbeat_loop, name="heartbeat-loop")

        # Create and start uvicorn server via spawn_server, which bridges the
        # ManagedThread stop_event to server.should_exit automatically.
        # timeout_keep_alive: uvicorn defaults to 5s, which races with client polling
        # intervals of the same length, causing TCP resets on idle connections. Use 120s
        # to safely cover long polling gaps during job waits.
        server_config = uvicorn.Config(
            self._dashboard.app,
            host=self._config.host,
            port=self._config.port,
            log_level="warning",
            timeout_keep_alive=120,
        )
        self._server = uvicorn.Server(server_config)
        self._threads.spawn_server(self._server, name="controller-server")

        if self._autoscaler:
            logger.info("Autoscaler configured with %d scale groups", len(self._autoscaler.groups))
            self._autoscaler_thread = self._threads.spawn(self._run_autoscaler_loop, name="autoscaler-loop")

        # Wait for server startup with exponential backoff
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=Duration.from_seconds(5.0),
        )

    def stop(self) -> None:
        """Stop all background components gracefully.

        Shutdown ordering:
        1. Stop scheduling/heartbeat/autoscaler loops so no new work is triggered.
        2. Shut down the autoscaler (stops monitors, terminates VMs, stops platform).
        3. Stop remaining threads (server) and executors.
        """
        self._wake_event.set()
        self._heartbeat_event.set()
        join_timeout = Duration.from_seconds(5.0)
        if self._scheduling_thread:
            self._scheduling_thread.stop()
            self._scheduling_thread.join(timeout=join_timeout)
        if self._heartbeat_thread:
            self._heartbeat_thread.stop()
            self._heartbeat_thread.join(timeout=join_timeout)
        if self._autoscaler_thread:
            self._autoscaler_thread.stop()
            self._autoscaler_thread.join(timeout=join_timeout)

        if self._autoscaler:
            self._autoscaler.shutdown()

        self._threads.stop()

    def _run_scheduling_loop(self, stop_event: threading.Event) -> None:
        """Scheduling loop: task assignment and worker timeout checks only."""
        limiter = RateLimiter(interval_seconds=self._config.scheduler_interval.to_seconds())
        while not stop_event.is_set():
            self._wake_event.wait(timeout=limiter.time_until_next())
            self._wake_event.clear()
            limiter.mark_run()

            if stop_event.is_set():
                break

            if self._checkpoint_in_progress:
                continue

            self._run_scheduling()

    def _run_autoscaler_loop(self, stop_event: threading.Event) -> None:
        """Autoscaler loop: runs on its own thread so blocking cloud API calls
        don't stall scheduling or heartbeats."""
        limiter = RateLimiter(interval_seconds=self._autoscaler.evaluation_interval.to_seconds())
        while not stop_event.is_set():
            if not limiter.wait(cancel=stop_event):
                break
            if self._checkpoint_in_progress:
                continue
            try:
                self._run_autoscaler_once()
            except Exception:
                logger.exception("Autoscaler loop iteration failed")

            self._maybe_periodic_checkpoint()

    def _run_heartbeat_loop(self, stop_event: threading.Event) -> None:
        """Heartbeat loop running on its own thread so slow RPCs don't block scheduling."""
        limiter = RateLimiter(interval_seconds=self._config.heartbeat_interval.to_seconds())
        while not stop_event.is_set():
            self._heartbeat_event.wait(timeout=limiter.time_until_next())
            self._heartbeat_event.clear()
            limiter.mark_run()
            if stop_event.is_set():
                break
            if self._checkpoint_in_progress:
                continue
            self._heartbeat_all_workers()

    def _is_reservation_satisfied(self, job: ControllerJob) -> bool:
        """Check if a job's reservation is fully satisfied.

        Returns True if the job has no reservation or if enough workers
        have been claimed to cover every reservation entry.
        """
        if not job.request.HasField("reservation"):
            return True

        claimed = self._count_reservation_claims(job.job_id.to_wire())
        return claimed >= len(job.request.reservation.entries)

    def _count_reservation_claims(self, job_id_wire: str) -> int:
        """Count workers claimed for the given job."""
        return sum(1 for c in self._reservation_claims.values() if c.job_id == job_id_wire)

    def _cleanup_stale_claims(self) -> None:
        """Remove claims for workers that disappeared or jobs that finished."""
        active_worker_ids = {w.worker_id for w in self._state.list_all_workers()}
        stale: list[WorkerId] = []
        for worker_id, claim in self._reservation_claims.items():
            if worker_id not in active_worker_ids:
                stale.append(worker_id)
                continue
            job = self._state.get_job(JobName.from_wire(claim.job_id))
            if job is None or job.is_finished():
                stale.append(worker_id)
        for wid in stale:
            del self._reservation_claims[wid]

    def _claim_workers_for_reservations(self) -> None:
        """Assign unclaimed workers to unsatisfied reservation entries.

        Scans all non-finished jobs with reservations. For each unfulfilled
        entry, finds an eligible unclaimed worker and records the claim.
        """
        claimed_entries: set[tuple[str, int]] = {(c.job_id, c.entry_idx) for c in self._reservation_claims.values()}
        claimed_worker_ids: set[WorkerId] = set(self._reservation_claims.keys())
        all_workers = self._state.list_all_workers()

        for job in self._state.list_all_jobs():
            if job.is_finished():
                continue
            if not job.request.HasField("reservation"):
                continue

            job_wire = job.job_id.to_wire()
            for idx, res_entry in enumerate(job.request.reservation.entries):
                if (job_wire, idx) in claimed_entries:
                    continue

                for worker in all_workers:
                    if worker.worker_id in claimed_worker_ids:
                        continue
                    if not worker.healthy:
                        continue
                    if not _worker_matches_reservation_entry(worker, res_entry):
                        continue

                    self._reservation_claims[worker.worker_id] = ReservationClaim(
                        job_id=job_wire,
                        entry_idx=idx,
                    )
                    claimed_worker_ids.add(worker.worker_id)
                    claimed_entries.add((job_wire, idx))
                    break

    def _run_scheduling(self) -> None:
        """Run one scheduling cycle.

        Three-phase scheduling:
        1. Preemption: kill running holder tasks on claimed workers when peer
           tasks are pending, freeing resources for real work.
        2. Preference pass: for reservation jobs, try claimed workers first.
           The claims set is small (≤ reservation entry count), so this is cheap.
        3. Normal pass: remaining tasks go through the standard scheduler.

        All passes share a single SchedulingContext so capacity deductions
        are visible across passes.

        No lock is needed since only one scheduling thread exists. All state
        reads and writes go through ControllerState which has its own lock.
        """
        self._cleanup_stale_claims()
        self._claim_workers_for_reservations()

        timer = Timer()
        with slow_log(logger, "scheduling state reads", threshold_ms=50):
            pending_tasks = self._state.peek_pending_tasks()
            workers = self._state.get_available_workers()
        state_read_ms = timer.elapsed_ms()

        if not pending_tasks:
            return

        # Handle timeouts and reservation gates before scheduling.
        # Holder tasks participate in scheduling like normal tasks.
        schedulable_task_ids: list[JobName] = []
        jobs: dict[JobName, JobRequirements] = {}
        has_reservation: set[JobName] = set()
        for task in pending_tasks:
            if not task.can_be_scheduled():
                continue
            job = self._state.get_job(task.job_id)
            if not job:
                continue
            if job.scheduling_deadline is not None and job.scheduling_deadline.expired():
                self._mark_task_unschedulable(task)
                continue
            # Gate: skip real tasks whose job has an unsatisfied reservation.
            # Holder tasks are always schedulable (they ARE the reservation).
            if not job.is_reservation_holder and not self._is_reservation_satisfied(job):
                continue
            schedulable_task_ids.append(task.task_id)
            if task.job_id not in jobs:
                jobs[task.job_id] = job_requirements_from_job(job)
                if job.request.HasField("reservation"):
                    has_reservation.add(task.job_id)
                elif _find_reservation_ancestor(self._state, task.job_id) is not None:
                    has_reservation.add(task.job_id)

        if not schedulable_task_ids:
            return

        # Inject reservation taints: claimed workers get a taint attribute,
        # non-reservation jobs get a NOT_EXISTS constraint for it.
        modified_workers = _inject_reservation_taints(workers, self._reservation_claims)
        jobs = _inject_taint_constraints(jobs, has_reservation)

        with slow_log(logger, "snapshot_building_counts", threshold_ms=50):
            building_counts = self._state.snapshot_building_counts()
        context = self._scheduler.create_scheduling_context(
            modified_workers,
            building_counts=building_counts,
            pending_tasks=schedulable_task_ids,
            jobs=jobs,
        )

        # Phase 1: soft preference — steer reservation tasks toward claimed workers.
        # Skips coscheduled jobs (they need atomic all-or-nothing via find_assignments).
        preference_assignments = _preference_pass(context, has_reservation, self._reservation_claims)

        # Phase 2: normal scheduler for all remaining tasks.
        result = self._scheduler.find_assignments(context)

        all_assignments = preference_assignments + result.assignments
        if all_assignments:
            with slow_log(logger, "buffer_assignments", threshold_ms=200):
                self._buffer_assignments(all_assignments)
            logger.debug(
                "Scheduling cycle: %d assignments (%d preferred, %d normal), %dms (state read: %dms)",
                len(all_assignments),
                len(preference_assignments),
                len(result.assignments),
                timer.elapsed_ms(),
                state_read_ms,
            )

    def _buffer_assignments(
        self,
        assignments: list[tuple[JobName, WorkerId]],
    ) -> None:
        """Commit resources and buffer task assignments for heartbeat delivery.

        Groups assignments by job, commits resources via TaskAssignedEvent, and
        buffers RunTaskRequest protos via state.buffer_dispatch().
        """
        # Group assignments by job for coscheduled handling
        by_job: dict[JobName, list[tuple[JobName, WorkerId]]] = defaultdict(list)
        for task_id, worker_id in assignments:
            job_id = task_id.parent
            if job_id is not None:
                by_job[job_id].append((task_id, worker_id))

        for job_id, job_assignments in by_job.items():
            job = self._state.get_job(job_id)
            if job is None:
                continue

            has_real_dispatch = False
            for task_id, worker_id in job_assignments:
                task = self._state.get_task(task_id)
                if task is None:
                    continue

                # Commit resources via event (handles synthetic vs real internally)
                self._state.handle_event(
                    TaskAssignedEvent(
                        task_id=task_id,
                        worker_id=worker_id,
                    )
                )

                # Holder job tasks are scheduled and assigned to workers
                # (committing resources to hold capacity), but never
                # dispatched — there is no entrypoint to run.
                if job.is_reservation_holder:
                    continue

                has_real_dispatch = True

                # Build the run request.
                # For reservation jobs, inject region constraints derived from
                # the claimed workers so child tasks inherit the region lock.
                task_constraints = (
                    _reservation_region_constraints(
                        job_id.to_wire(),
                        self._reservation_claims,
                        self._state,
                        list(job.request.constraints),
                    )
                    if job.request.HasField("reservation")
                    else list(job.request.constraints)
                )
                request = cluster_pb2.Worker.RunTaskRequest(
                    task_id=task_id.to_wire(),
                    num_tasks=job.num_tasks,
                    entrypoint=job.request.entrypoint,
                    environment=job.request.environment,
                    bundle_gcs_path=job.request.bundle_gcs_path,
                    resources=job.request.resources,
                    ports=list(job.request.ports),
                    attempt_id=task.current_attempt_id,
                    constraints=task_constraints,
                )
                # Copy timeout if set (check milliseconds field > 0)
                if job.request.timeout.milliseconds > 0:
                    request.timeout.CopyFrom(job.request.timeout)

                # Buffer dispatch (state handles the lock)
                self._state.buffer_dispatch(worker_id, request)

            # Wake heartbeat thread to deliver buffered dispatches immediately
            if has_real_dispatch:
                self._heartbeat_event.set()

    def _mark_task_unschedulable(self, task: ControllerTask) -> None:
        """Mark a task as unschedulable due to timeout."""
        job = self._state.get_job(task.job_id)
        if job and job.request.HasField("scheduling_timeout"):
            timeout = Duration.from_proto(job.request.scheduling_timeout)
        else:
            timeout = None
        logger.warning(f"Task {task.task_id} exceeded scheduling timeout ({timeout}), marking as UNSCHEDULABLE")
        txn = self._state.handle_event(
            TaskStateChangedEvent(
                task_id=task.task_id,
                new_state=cluster_pb2.TASK_STATE_UNSCHEDULABLE,
                attempt_id=task.current_attempt_id,
                error=f"Scheduling timeout exceeded ({timeout})",
            )
        )
        if txn.tasks_to_kill:
            self.kill_tasks_on_workers(txn.tasks_to_kill)

    def create_scheduling_context(self, workers: list[ControllerWorker]) -> SchedulingContext:
        """Create a scheduling context for the given workers."""
        building_counts = self._state.snapshot_building_counts()
        return self._scheduler.create_scheduling_context(
            workers,
            building_counts=building_counts,
        )

    def get_job_scheduling_diagnostics(self, job: ControllerJob, context: SchedulingContext) -> str:
        """Get detailed diagnostics for why a job cannot be scheduled."""
        req = job_requirements_from_job(job)
        tasks = self._state.get_job_tasks(job.job_id)
        schedulable_task_id = next((t.task_id for t in tasks if t.can_be_scheduled()), None)
        return self._scheduler.get_job_scheduling_diagnostics(req, context, schedulable_task_id, num_tasks=len(tasks))

    def kill_tasks_on_workers(self, task_ids: set[JobName]) -> None:
        """Buffer kill requests for delivery via next heartbeat.

        Called after state has marked tasks as killed. For each task that had
        a worker assigned, buffers the kill request for delivery via the next
        heartbeat to that worker.
        """
        any_buffered = False
        for task_id in task_ids:
            task = self._state.get_task(task_id)
            if not task or not task.worker_id:
                continue
            worker = self._state.get_worker(task.worker_id)
            if not worker:
                continue
            self._state.buffer_kill(worker.worker_id, task_id.to_wire())
            any_buffered = True

        # Wake heartbeat thread to deliver buffered kills immediately
        if any_buffered:
            self._heartbeat_event.set()

    def _heartbeat_all_workers(self) -> None:
        """Send heartbeats to all registered workers.

        Uses state-owned transitions: begin_heartbeat() atomically snapshots worker
        state and drains dispatch buffers, then RPCs proceed without locks, and
        complete_heartbeat()/fail_heartbeat() apply results.

        When fail_heartbeat causes a worker to exceed the failure threshold,
        _on_worker_failed prunes it from state. We detect this (worker no longer
        in state) and evict the cached stub + notify the autoscaler.

        Holds _heartbeat_lock for the entire round so begin_checkpoint() can
        wait for a complete quiescent state before snapshotting.
        """
        with self._heartbeat_lock:
            self._heartbeat_all_workers_inner()

    def _heartbeat_all_workers_inner(self) -> None:
        round_timer = Timer()

        # Phase 1: create snapshots for all healthy workers (lock-acquiring).
        with slow_log(logger, "heartbeat phase 1 (snapshot)", threshold_ms=100):
            snapshots: list[HeartbeatSnapshot] = []
            for w in self._state.get_available_workers():
                snapshot = self._state.begin_heartbeat(w.worker_id)
                if snapshot:
                    snapshots.append(snapshot)

        if not snapshots:
            return

        # Phase 2: stream heartbeats through a bounded worker queue.
        work_queue: queue.Queue[HeartbeatSnapshot] = queue.Queue()
        result_queue: queue.Queue[tuple[HeartbeatSnapshot, cluster_pb2.HeartbeatResponse | None, str | None]] = (
            queue.Queue()
        )
        for snapshot in snapshots:
            work_queue.put(snapshot)

        worker_count = min(self._config.max_dispatch_parallelism, len(snapshots))

        def _dispatch_worker() -> None:
            while True:
                try:
                    snapshot = work_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    response = self._do_heartbeat_rpc(snapshot)
                    result_queue.put((snapshot, response, None))
                except Exception as e:
                    result_queue.put((snapshot, None, str(e)))

        worker_futures = [self._dispatch_executor.submit(_dispatch_worker) for _ in range(worker_count)]

        # Phase 3: consume all responses; per-worker RPC timeout determines failures.
        fail_count = 0
        failed_workers: list[str] = []
        with slow_log(logger, "heartbeat phase 3 (process results)", threshold_ms=500):
            for _ in snapshots:
                snapshot, response, error = result_queue.get()
                if error is not None:
                    fail_count += 1
                    failed_workers.append(snapshot.worker_id)
                    logger.debug("Heartbeat error for %s: %s", snapshot.worker_id, error)
                    self._handle_heartbeat_failure(snapshot, error)
                    continue
                if response is not None:
                    self._state.complete_heartbeat(snapshot, response)

        for future in worker_futures:
            future.cancel()

        elapsed = round_timer.elapsed_ms()
        level = logging.WARNING if elapsed > _SLOW_HEARTBEAT_MS else logging.DEBUG
        fmt = "Heartbeat round: %d workers, %d failed, %dms"
        args: list[object] = [len(snapshots), fail_count, elapsed]
        if failed_workers:
            fmt += " failed=[%s]"
            args.append(", ".join(failed_workers))
        logger.log(level, fmt, *args)

        self._heartbeat_iteration += 1
        if self._heartbeat_iteration % _HEALTH_SUMMARY_INTERVAL == 0:
            workers = self._state.get_available_workers()
            jobs = self._state.list_all_jobs()
            active = sum(1 for j in jobs if j.state == cluster_pb2.JOB_STATE_RUNNING)
            pending = len(self._state.peek_pending_tasks())
            logger.info(
                "Controller status: %d workers (%d failed), %d active jobs, %d pending tasks",
                len(workers),
                fail_count,
                active,
                pending,
            )

    def _handle_heartbeat_failure(self, snapshot: HeartbeatSnapshot, error: str) -> None:
        """Process a heartbeat failure: update state, evict stub + notify autoscaler if worker died.

        After fail_heartbeat, if the worker was pruned from state (exceeded failure
        threshold), we evict the cached RPC stub and notify the autoscaler.
        """
        self._state.fail_heartbeat(snapshot, error)

        # fail_heartbeat -> _on_worker_heartbeat_failed -> _on_worker_failed prunes
        # the worker from state when consecutive failures exceed the threshold.
        if self._state.get_worker(snapshot.worker_id) is None:
            self.stub_factory.evict(snapshot.worker_address)
            if self._autoscaler and snapshot.vm_address:
                self._autoscaler.notify_worker_failed(snapshot.vm_address)

    def _do_heartbeat_rpc(
        self,
        snapshot: HeartbeatSnapshot,
    ) -> cluster_pb2.HeartbeatResponse:
        """Send a heartbeat RPC to a single worker.

        Raises:
            Exception on RPC failure (handled by caller via state.fail_heartbeat)
        """
        if rule := chaos("controller.heartbeat"):
            sleep(rule.delay_seconds)
            raise Exception("chaos: heartbeat unavailable")
        stub = self.stub_factory.get_stub(snapshot.worker_address)

        # Build expected_tasks from snapshot — no state lock needed.
        expected_tasks = []
        for entry in snapshot.running_tasks:
            if rule := chaos("controller.heartbeat.iteration"):
                sleep(rule.delay_seconds)
            expected_tasks.append(
                cluster_pb2.Controller.WorkerTaskStatus(
                    task_id=entry.task_id.to_wire(),
                    attempt_id=entry.attempt_id,
                )
            )
        request = cluster_pb2.HeartbeatRequest(
            tasks_to_run=snapshot.tasks_to_run,
            tasks_to_kill=snapshot.tasks_to_kill,
            expected_tasks=expected_tasks,
        )
        return stub.heartbeat(request)

    def _run_autoscaler_once(self) -> None:
        """Run one autoscaler cycle: refresh (I/O) then update (CPU).

        Called from the autoscaler loop thread.
        """
        if not self._autoscaler:
            return

        vm_status_map = self._build_vm_status_map()
        self._autoscaler.refresh(vm_status_map)
        workers = self._state.get_available_workers()
        demand_entries = compute_demand_entries(
            self._state,
            self._scheduler,
            workers,
            reservation_claims=self._reservation_claims,
        )
        self._autoscaler.update(demand_entries)

    def _build_vm_status_map(self) -> VmWorkerStatusMap:
        """Build a map of VM address to worker status for autoscaler.

        The autoscaler needs to look up worker status by VM address (not worker_id)
        because RemoteWorkerHandle only exposes the VM's IP address, not the worker's
        self-assigned ID. Workers self-discover their vm_address at startup via
        socket probe (env_probe.py).
        """
        result: VmWorkerStatusMap = {}
        for worker in self._state.list_all_workers():
            vm_addr = worker.metadata.vm_address
            if not vm_addr:
                logger.warning(
                    "Worker %s has no vm_address in metadata, skipping for autoscaler",
                    worker.worker_id,
                )
                continue

            result[vm_addr] = VmWorkerStatus(
                vm_address=vm_addr,
                # Snapshot the set to prevent concurrent modification errors
                running_task_ids=frozenset(tid.to_wire() for tid in list(worker.running_tasks)),
            )
        return result

    @property
    def _snapshot_storage_prefix(self) -> str:
        return self._config.bundle_prefix or ""

    def _maybe_periodic_checkpoint(self) -> None:
        """Write a best-effort periodic snapshot if the configured interval has elapsed.

        Unlike begin_checkpoint(), this does NOT set _checkpoint_in_progress — it
        captures a point-in-time snapshot without pausing scheduling or heartbeats.
        Suitable for crash recovery; not guaranteed to be fully quiescent.
        """
        if self._periodic_checkpoint_limiter is None:
            return
        if self._checkpoint_in_progress:
            return
        prefix = self._snapshot_storage_prefix
        if not prefix:
            return
        if not self._periodic_checkpoint_limiter.should_run():
            return
        try:
            result = create_snapshot(
                self._state,
                autoscaler=self._autoscaler,
                reservation_claims=self._reservation_claims,
            )
            path = write_snapshot(result.proto, prefix)
            logger.info(
                "Periodic checkpoint written: %s (jobs=%d tasks=%d workers=%d)",
                path,
                result.job_count,
                result.task_count,
                result.worker_count,
            )
        except Exception:
            logger.exception("Periodic checkpoint failed")

    def begin_checkpoint(self) -> tuple[str, SnapshotResult]:
        """Pause loops, snapshot state, write to storage. Returns (path, result).

        Sets _checkpoint_in_progress so the scheduling and autoscaler loops
        idle, waits for in-flight heartbeat dispatches to drain, then takes
        a consistent snapshot and writes it to remote storage.
        """
        prefix = self._snapshot_storage_prefix
        if not prefix:
            raise ValueError("Cannot checkpoint: no storage prefix configured (bundle_prefix is empty)")

        self._checkpoint_in_progress = True
        try:
            # Wait for any in-flight heartbeat round to complete.
            with self._heartbeat_lock:
                result = create_snapshot(
                    self._state,
                    autoscaler=self._autoscaler,
                    reservation_claims=self._reservation_claims,
                )
                path = write_snapshot(result.proto, prefix)
            logger.info(
                "Checkpoint written: %s (jobs=%d tasks=%d workers=%d)",
                path,
                result.job_count,
                result.task_count,
                result.worker_count,
            )
            return path, result
        finally:
            self._checkpoint_in_progress = False

    def restore_from_snapshot(self, proto: snapshot_pb2.ControllerSnapshot | None = None) -> bool:
        """Restore full controller state from a snapshot proto or from storage.

        When proto is None, reads the latest snapshot from storage.
        Called during startup, before background loops are started.
        Returns True if a snapshot was found and restored.
        """
        if proto is None:
            prefix = self._snapshot_storage_prefix
            if not prefix:
                return False
            proto = read_latest_snapshot(prefix)
            if proto is None:
                logger.info("No snapshot found at %s, starting fresh", prefix)
                return False

        result = restore_snapshot(proto, self._state)
        logger.info(
            "Restored snapshot: jobs=%d tasks=%d workers=%d endpoints=%d",
            result.job_count,
            result.task_count,
            result.worker_count,
            result.endpoint_count,
        )

        # Restore autoscaler scaling groups (parallelized — each calls platform.list_slices())
        if self._autoscaler is not None:
            groups_to_restore = []
            for group_snap in proto.scaling_groups:
                group = self._autoscaler.groups.get(group_snap.name)
                if group is None:
                    logger.warning(
                        "Snapshot references scaling group %s which does not exist in config, skipping",
                        group_snap.name,
                    )
                    continue
                groups_to_restore.append((group_snap, group))

            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {
                    executor.submit(restore_scaling_group, gs, g.platform, g.config, g.label_prefix): (gs, g)
                    for gs, g in groups_to_restore
                }
                for future in as_completed(futures):
                    group_snap, group = futures[future]
                    restore_result = future.result()
                    group.restore_from_snapshot(
                        slices=restore_result.slices,
                        consecutive_failures=restore_result.consecutive_failures,
                        last_scale_up=restore_result.last_scale_up,
                        last_scale_down=restore_result.last_scale_down,
                        backoff_until=restore_result.backoff_until,
                        quota_exceeded_until=restore_result.quota_exceeded_until,
                        quota_reason=restore_result.quota_reason,
                    )

            # Workers from discarded slices remain in ControllerState as healthy.
            # They will naturally fail heartbeat checks and be pruned once
            # consecutive failures exceed the threshold. This is intentional:
            # the heartbeat failure path handles cleanup of stale workers
            # including task reassignment and resource release.

            # Restore tracked workers into the autoscaler
            restored_workers = restore_tracked_workers(proto)
            self._autoscaler.restore_tracked_workers(restored_workers)
            logger.info("Restored %d tracked workers", len(restored_workers))

        # Restore reservation claims
        for claim_snap in proto.reservation_claims:
            worker_id = WorkerId(claim_snap.worker_id)
            self._reservation_claims[worker_id] = ReservationClaim(
                job_id=claim_snap.job_id,
                entry_idx=claim_snap.entry_idx,
            )

        return True

    def launch_job(
        self,
        request: cluster_pb2.Controller.LaunchJobRequest,
    ) -> cluster_pb2.Controller.LaunchJobResponse:
        """Submit a job to the controller."""
        return self._service.launch_job(request, None)

    def get_job_status(
        self,
        job_id: str,
    ) -> cluster_pb2.Controller.GetJobStatusResponse:
        """Get the status of a job."""
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        return self._service.get_job_status(request, None)

    def terminate_job(
        self,
        job_id: str,
    ) -> cluster_pb2.Empty:
        """Terminate a running job."""
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        return self._service.terminate_job(request, None)

    # Properties

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def port(self) -> int:
        """Actual bound port (may differ from config if port=0 was specified)."""
        if self._server and self._server.started:
            if self._server.servers and self._server.servers[0].sockets:
                return self._server.servers[0].sockets[0].getsockname()[1]
        return self._config.port

    @property
    def url(self) -> str:
        return f"http://{self._config.host}:{self.port}"

    @property
    def reservation_claims(self) -> dict[WorkerId, ReservationClaim]:
        """Current reservation claims, keyed by worker ID."""
        return self._reservation_claims

    @property
    def autoscaler(self) -> "Autoscaler | None":
        """The autoscaler instance, if autoscaling is enabled."""
        return self._autoscaler
