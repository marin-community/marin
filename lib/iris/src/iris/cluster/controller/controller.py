# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import logging
import queue
import shutil
import sys
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import sleep
from typing import Protocol

import uvicorn

from iris.chaos import chaos
from iris.cluster.bundle import BundleStore
from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    PlacementRequirements,
    WellKnownAttribute,
    constraints_from_resources,
    evaluate_constraint,
    extract_placement_requirements,
    merge_constraints,
)
from iris.cluster.controller.autoscaler import Autoscaler, DemandEntry
from iris.cluster.controller.db import (
    ATTEMPTS,
    JOBS,
    RESERVATION_CLAIMS,
    SCALING_GROUPS,
    TASKS,
    TERMINAL_TASK_STATES,
    TRACKED_WORKERS,
    WORKERS,
    ControllerDB,
    Join,
    Job,
    Task,
    Worker,
    _tasks_with_attempts,
    healthy_active_workers_with_attributes,
    running_tasks_by_worker,
    tasks_for_job_with_attempts,
)
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.scheduler import (
    JobRequirements,
    Scheduler,
    SchedulingContext,
    WorkerSnapshot,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.snapshot import restore_scaling_group, restore_tracked_workers
from iris.cluster.controller.transitions import (
    HEARTBEAT_FAILURE_THRESHOLD,
    RESERVATION_HOLDER_JOB_NAME,
    Assignment,
    ControllerTransitions,
    DispatchBatch,
    HeartbeatAction,
    ReservationClaim,
)
from iris.cluster.log_store import PROCESS_LOG_KEY, LogStoreHandler
from iris.cluster.types import (
    JobName,
    VmWorkerStatus,
    VmWorkerStatusMap,
    WorkerId,
)
from iris.logging import slow_log
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import cluster_pb2, snapshot_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.time_utils import Duration, ExponentialBackoff, RateLimiter, Timer, Timestamp

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


@dataclass(frozen=True)
class CheckpointResult:
    """Metadata returned after a checkpoint DB copy is written."""

    created_at: Timestamp
    job_count: int
    task_count: int
    worker_count: int


def job_requirements_from_job(job: Job) -> JobRequirements:
    """Convert a job row to scheduler-compatible JobRequirements."""
    return JobRequirements(
        resources=job.request.resources,
        constraints=list(job.request.constraints),
        is_coscheduled=job.is_coscheduled,
        coscheduling_group_by=job.coscheduling_group_by,
    )


def compute_demand_entries(
    queries: ControllerDB,
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
        queries: Controller DB read surface for pending tasks and jobs.
        scheduler: Scheduler for dry-run pass. If None, skips dry-run.
        workers: Available workers for dry-run. If None, skips dry-run.
        reservation_claims: Reservation claims to apply taint injection in the
            dry-run, matching the real scheduling path. If None, no taints applied.
    """
    demand_entries: list[DemandEntry] = []

    # Collect all schedulable pending tasks, grouped by job.
    tasks_by_job: dict[JobName, list[Task]] = defaultdict(list)
    all_schedulable: list[Task] = []
    pending = _schedulable_tasks(queries)
    job_rows = list(_jobs_by_id(queries, {task.job_id for task in pending}).values()) if pending else []
    jobs_by_id = {job.job_id: job for job in job_rows}
    for task in pending:
        if not task.can_be_scheduled():
            continue
        if task.job_id not in jobs_by_id:
            continue
        tasks_by_job[task.job_id].append(task)
        all_schedulable.append(task)

    # Build job requirements once, shared between dry-run and demand emission.
    # Also track which jobs have reservations so we can apply taint injection.
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    has_direct_reservation: set[JobName] = set()
    for task in all_schedulable:
        if task.job_id in jobs:
            continue
        job = jobs_by_id.get(task.job_id)
        if job is None:
            continue
        jobs[task.job_id] = job_requirements_from_job(job)
        if job.request.HasField("reservation"):
            has_reservation.add(task.job_id)
            has_direct_reservation.add(task.job_id)
        elif _find_reservation_ancestor(queries, task.job_id) is not None:
            has_reservation.add(task.job_id)

    # Dry-run scheduling with building/assignment limits disabled.
    # All tasks participate — holders and real tasks alike.
    absorbed_task_ids: set[JobName] = set()
    if scheduler is not None and workers is not None and workers:
        building_counts = _building_counts(queries)
        task_ids = [t.task_id for t in all_schedulable]
        claims = reservation_claims or {}
        dry_run_workers = _inject_reservation_taints(workers, claims)
        dry_run_jobs = _inject_taint_constraints(jobs, has_reservation, has_direct_reservation)

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
        job = jobs_by_id.get(job_id)
        if not job:
            continue
        if job.is_finished():
            continue

        invalid_reason: str | None = None
        try:
            normalized = extract_placement_requirements(job.request.constraints)
        except ValueError as e:
            invalid_reason = f"invalid_constraints: {e}"
            normalized = PlacementRequirements(
                device_type=None,
                device_variants=None,
                preemptible=None,
                required_regions=None,
                required_zones=None,
            )

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
                        normalized=normalized,
                        constraints=list(job.request.constraints),
                        resources=job.request.resources,
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
                    normalized=normalized,
                    constraints=list(job.request.constraints),
                    resources=job.request.resources,
                    invalid_reason=invalid_reason,
                )
            )

    return demand_entries


def _read_reservation_claims(db: ControllerDB) -> dict[WorkerId, ReservationClaim]:
    """Read reservation claims from the canonical DB table."""
    with db.snapshot() as snapshot:
        rows = snapshot.select(
            RESERVATION_CLAIMS,
            columns=(
                RESERVATION_CLAIMS.c.worker_id,
                RESERVATION_CLAIMS.c.job_id,
                RESERVATION_CLAIMS.c.entry_idx,
            ),
        )
    return {
        row.worker_id: ReservationClaim(
            job_id=row.job_id,
            entry_idx=row.entry_idx,
        )
        for row in rows
    }


def _jobs_by_id(queries: ControllerDB, job_ids: set[JobName]) -> dict[JobName, Job]:
    if not job_ids:
        return {}
    with queries.snapshot() as snapshot:
        jobs = snapshot.select(JOBS, where=JOBS.c.job_id.in_([job_id.to_wire() for job_id in job_ids]))
    return {job.job_id: job for job in jobs}


def _schedulable_tasks(queries: ControllerDB) -> list[Task]:
    with queries.snapshot() as snapshot:
        tasks = snapshot.select(
            TASKS,
            where=TASKS.c.state.not_null() & ~TASKS.c.state.in_(list(TERMINAL_TASK_STATES)),
            order_by=(
                TASKS.c.priority_neg_depth.asc(),
                TASKS.c.priority_root_submitted_ms.asc(),
                TASKS.c.submitted_at_ms.asc(),
                TASKS.c.task_id.asc(),
            ),
        )
    return [task for task in tasks if task.can_be_scheduled()]


def _tasks_by_ids_with_attempts(queries: ControllerDB, task_ids: set[JobName]) -> dict[JobName, Task]:
    if not task_ids:
        return {}
    task_wires = [task_id.to_wire() for task_id in task_ids]
    with queries.snapshot() as snapshot:
        tasks = snapshot.select(
            TASKS,
            where=TASKS.c.task_id.in_(task_wires),
            order_by=(TASKS.c.task_id.asc(),),
        )
        attempts = snapshot.select(
            ATTEMPTS,
            where=ATTEMPTS.c.task_id.in_(task_wires),
            order_by=(ATTEMPTS.c.task_id.asc(), ATTEMPTS.c.attempt_id.asc()),
        )
    return {task.task_id: task for task in _tasks_with_attempts(tasks, attempts)}


def _building_counts(queries: ControllerDB) -> dict[WorkerId, int]:
    workers = healthy_active_workers_with_attributes(queries)
    if not workers:
        return {}
    running_by_worker = running_tasks_by_worker(queries, {worker.worker_id for worker in workers})
    running_task_ids = {task_id for task_ids in running_by_worker.values() for task_id in task_ids}
    if not running_task_ids:
        return {}
    tasks = _tasks_by_ids_with_attempts(queries, running_task_ids)
    jobs = _jobs_by_id(queries, {task.job_id for task in tasks.values()})
    counts: dict[WorkerId, int] = {}
    for worker_id, task_ids in running_by_worker.items():
        count = 0
        for task_id in task_ids:
            task = tasks.get(task_id)
            if task is None:
                continue
            if task.state not in (cluster_pb2.TASK_STATE_BUILDING, cluster_pb2.TASK_STATE_ASSIGNED):
                continue
            job = jobs.get(task.job_id)
            if job is None or job.is_reservation_holder:
                continue
            count += 1
        if count > 0:
            counts[worker_id] = count
    return counts


def _workers_by_id(queries: ControllerDB, worker_ids: set[WorkerId]) -> dict[WorkerId, Worker]:
    if not worker_ids:
        return {}
    with queries.snapshot() as snapshot:
        workers = snapshot.select(
            WORKERS,
            where=WORKERS.c.worker_id.in_([str(worker_id) for worker_id in worker_ids]),
        )
    return {worker.worker_id: worker for worker in workers}


def _task_worker_mapping(queries: ControllerDB, task_ids: set[JobName]) -> dict[JobName, WorkerId]:
    if not task_ids:
        return {}
    with queries.snapshot() as snapshot:
        rows = snapshot.select(
            TASKS,
            columns=(TASKS.c.task_id, ATTEMPTS.c.worker_id),
            joins=(Join(table=ATTEMPTS, on=TASKS.c.task_id == ATTEMPTS.c.task_id),),
            where=TASKS.c.task_id.in_([task_id.to_wire() for task_id in task_ids])
            & (TASKS.c.current_attempt_id == ATTEMPTS.c.attempt_id)
            & ATTEMPTS.c.worker_id.not_null(),
        )
    return {row.task_id: row.worker_id for row in rows}


def _worker_matches_reservation_entry(
    worker: Worker,
    res_entry: cluster_pb2.ReservationEntry,
) -> bool:
    """Check if a worker is eligible for a reservation entry.

    Auto-injects device constraints from the reservation entry's resource spec
    and merges them with explicit constraints on the entry, then evaluates all
    constraints against the worker's attributes.
    """
    auto = constraints_from_resources(res_entry.resources)
    explicit = [Constraint.from_proto(c) for c in res_entry.constraints]
    merged = merge_constraints(auto, explicit)

    merged_protos = [c.to_proto() for c in merged]
    for constraint in merged_protos:
        attr = worker.attributes.get(constraint.key)
        if not evaluate_constraint(attr, constraint):
            return False

    return True


def _inject_reservation_taints(
    workers: list[Worker],
    claims: dict[WorkerId, ReservationClaim],
) -> list[Worker]:
    """Create modified worker copies with reservation taints and prioritization.

    Claimed workers receive a ``reservation-job`` attribute set to the claiming
    job's ID.  The returned list is ordered with claimed workers first so that
    reservation jobs (which have no NOT_EXISTS constraint) naturally pick from
    their claimed workers before unclaimed ones.

    Workers are never mutated — ``dataclasses.replace`` produces shallow copies.
    """
    if not claims:
        return workers

    claimed: list[Worker] = []
    unclaimed: list[Worker] = []
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
    has_direct_reservation: set[JobName] | None = None,
) -> dict[JobName, JobRequirements]:
    """Add reservation taint constraints to jobs.

    Three-way logic:
    - Direct reservation jobs (has_direct_reservation): get an EQ constraint
      forcing them onto their claimed workers only.
    - Descendants of reservation jobs (has_reservation minus direct): no
      constraint — they can use both claimed and unclaimed workers.
    - Non-reservation jobs: get a NOT_EXISTS constraint blocking them from
      claimed workers.
    """
    if not has_reservation and not jobs:
        return jobs

    if has_direct_reservation is None:
        has_direct_reservation = set()

    taint_constraint = cluster_pb2.Constraint(
        key=RESERVATION_TAINT_KEY,
        op=cluster_pb2.CONSTRAINT_OP_NOT_EXISTS,
    )

    modified: dict[JobName, JobRequirements] = {}
    for job_id, req in jobs.items():
        if job_id in has_direct_reservation:
            eq_constraint = cluster_pb2.Constraint(
                key=RESERVATION_TAINT_KEY,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value=job_id.to_wire()),
            )
            modified[job_id] = replace(
                req,
                constraints=[*list(req.constraints), eq_constraint],
            )
        elif job_id in has_reservation:
            modified[job_id] = req
        else:
            modified[job_id] = replace(
                req,
                constraints=[*list(req.constraints), taint_constraint],
            )
    return modified


def _find_reservation_ancestor(queries: ControllerDB, job_id: JobName) -> JobName | None:
    """Walk up the job hierarchy to find the nearest ancestor with a reservation.

    Returns the ancestor's JobName, or None if no ancestor has a reservation.
    """
    current = job_id.parent
    while current is not None:
        ancestor = _jobs_by_id(queries, {current}).get(current)
        if ancestor is not None and ancestor.request.HasField("reservation"):
            return current
        current = current.parent
    return None


def _reservation_region_constraints(
    job_id_wire: str,
    claims: dict[WorkerId, ReservationClaim],
    queries: ControllerDB,
    existing_constraints: list[cluster_pb2.Constraint],
) -> list[cluster_pb2.Constraint]:
    """Derive region constraints from claimed reservation workers.

    When a reservation job has no explicit region constraint, this function
    extracts the region attributes of claimed workers and returns the existing
    constraints plus an injected region constraint.  If the job already has a
    region constraint, or if claimed workers lack region attributes, the
    existing constraints are returned unchanged.
    """
    if any(c.key == WellKnownAttribute.REGION for c in existing_constraints):
        return existing_constraints

    claimed_worker_ids = {worker_id for worker_id, claim in claims.items() if claim.job_id == job_id_wire}
    workers_by_id = {
        worker.worker_id: worker
        for worker in healthy_active_workers_with_attributes(queries)
        if worker.worker_id in claimed_worker_ids
    }
    regions: set[str] = set()
    for worker in workers_by_id.values():
        if worker is None:
            continue
        region_attr = worker.attributes.get(WellKnownAttribute.REGION)
        if region_attr is not None:
            regions.add(str(region_attr.value))

    if not regions:
        return existing_constraints

    region_list = sorted(regions)
    if len(region_list) == 1:
        region_constraint = cluster_pb2.Constraint(
            key=WellKnownAttribute.REGION,
            op=cluster_pb2.CONSTRAINT_OP_EQ,
            value=cluster_pb2.AttributeValue(string_value=region_list[0]),
        )
    else:
        region_constraint = cluster_pb2.Constraint(
            key=WellKnownAttribute.REGION,
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
        # Holder jobs are children of the reservation job — look up claims
        # under the parent's wire ID.
        claim_key = job_wire
        if RESERVATION_HOLDER_JOB_NAME in job_wire:
            parent = job_id.parent
            if parent is not None:
                claim_key = parent.to_wire()
        for wid in claimed_by_job.get(claim_key, ()):
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
    def close(self) -> None: ...


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
            stub = self._stubs.pop(address, None)
        if stub is not None:
            stub.close()

    def close(self) -> None:
        with self._lock:
            stubs = list(self._stubs.values())
            self._stubs.clear()
        for stub in stubs:
            stub.close()


@dataclass
class ControllerConfig:
    """Controller configuration."""

    host: str = "127.0.0.1"
    """Host to bind the HTTP server to."""

    port: int = 0
    """Port to bind the HTTP server to. Use 0 for auto-assign."""

    bundle_prefix: str | None = None
    """Storage prefix for snapshots (e.g. gs://bucket/path, s3://bucket/path)."""

    scheduler_interval: Duration = field(default_factory=lambda: Duration.from_seconds(0.5))
    """How often to run the scheduling loop."""

    heartbeat_interval: Duration = field(default_factory=lambda: Duration.from_seconds(5.0))
    """How often to send heartbeats to workers."""

    max_dispatch_parallelism: int = 32
    """Maximum number of concurrent RPC dispatch operations."""

    max_tasks_per_job_per_cycle: int = 4
    """Maximum tasks from a single non-coscheduled job to consider per scheduling
    cycle. Bounds CPU time in the scheduler when many tasks are pending, preventing
    GIL starvation of the heartbeat thread. Coscheduled jobs are exempt (they need
    all tasks for atomic assignment). Set to 0 for unlimited."""

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

        if config.log_dir is not None:
            db_path = config.log_dir / "controller.sqlite3"
        else:
            tmp = Path(tempfile.mkdtemp(prefix="iris_controller_state_"))
            db_path = tmp / "controller.sqlite3"
        self._db = ControllerDB(db_path=db_path)
        self._transitions = ControllerTransitions(
            heartbeat_failure_threshold=config.heartbeat_failure_threshold,
            db=self._db,
        )
        self._scheduler = Scheduler()

        bundle_db_path = Path(tempfile.gettempdir()) / "iris-controller-bundles.sqlite3"
        self._bundle_store = BundleStore(db_path=bundle_db_path)

        self._service = ControllerServiceImpl(
            self._transitions,
            self._db,
            controller=self,
            bundle_store=self._bundle_store,
        )
        self._dashboard = ControllerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        # Ingest process logs into the LogStore so they are available via FetchLogs.
        self._log_store_handler = LogStoreHandler(self._transitions.log_store, key=PROCESS_LOG_KEY)
        self._log_store_handler.setLevel(logging.DEBUG)
        self._log_store_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger("iris").addHandler(self._log_store_handler)

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
            log_config=None,
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
        self.stub_factory.close()

        # Remove log handler before closing the log store to avoid
        # sqlite3.ProgrammingError spam from late log records.
        logging.getLogger("iris").removeHandler(self._log_store_handler)
        self._log_store_handler.close()
        self._transitions.close()
        self._bundle_store.close()

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
            try:
                self._heartbeat_all_workers()
            except Exception:
                logger.exception("Heartbeat round failed, will retry next interval")

    def _is_reservation_satisfied(
        self,
        job: Job,
        claims: dict[WorkerId, ReservationClaim] | None = None,
    ) -> bool:
        """Check if a job's reservation is fully satisfied.

        Returns True if the job has no reservation or if enough workers
        have been claimed to cover every reservation entry.
        """
        if not job.request.HasField("reservation"):
            return True

        claim_map = claims if claims is not None else _read_reservation_claims(self._db)
        claimed = self._count_reservation_claims(job.job_id.to_wire(), claim_map)
        return claimed >= len(job.request.reservation.entries)

    def _count_reservation_claims(self, job_id_wire: str, claims: dict[WorkerId, ReservationClaim]) -> int:
        """Count workers claimed for the given job."""
        return sum(1 for c in claims.values() if c.job_id == job_id_wire)

    def _cleanup_stale_claims(self, claims: dict[WorkerId, ReservationClaim] | None = None) -> bool:
        """Remove claims for workers that disappeared or jobs that finished."""
        persisted = False
        if claims is None:
            claims = _read_reservation_claims(self._db)
            persisted = True
        with self._db.snapshot() as snapshot:
            active_worker_ids = {
                row.worker_id
                for row in snapshot.select(
                    WORKERS,
                    columns=(WORKERS.c.worker_id,),
                    where=WORKERS.c.active == 1,
                )
            }
        claimed_job_ids = {JobName.from_wire(claim.job_id) for claim in claims.values()}
        claimed_jobs = list(_jobs_by_id(self._db, claimed_job_ids).values()) if claimed_job_ids else []
        jobs_by_id = {job.job_id.to_wire(): job for job in claimed_jobs}
        stale: list[WorkerId] = []
        for worker_id, claim in claims.items():
            if worker_id not in active_worker_ids:
                stale.append(worker_id)
                continue
            job = jobs_by_id.get(claim.job_id)
            if job is None or job.is_finished():
                stale.append(worker_id)
        for wid in stale:
            del claims[wid]
        if stale and persisted:
            self._transitions.replace_reservation_claims(claims)
        return bool(stale)

    def _claim_workers_for_reservations(self, claims: dict[WorkerId, ReservationClaim] | None = None) -> bool:
        """Assign unclaimed workers to unsatisfied reservation entries.

        Scans all non-finished jobs with reservations. For each unfulfilled
        entry, finds an eligible unclaimed worker and records the claim.
        """
        persisted = False
        if claims is None:
            claims = _read_reservation_claims(self._db)
            persisted = True
        claimed_entries: set[tuple[str, int]] = {(c.job_id, c.entry_idx) for c in claims.values()}
        claimed_worker_ids: set[WorkerId] = set(claims.keys())
        all_workers = healthy_active_workers_with_attributes(self._db)
        changed = False

        reservable_states = (
            cluster_pb2.JOB_STATE_PENDING,
            cluster_pb2.JOB_STATE_BUILDING,
            cluster_pb2.JOB_STATE_RUNNING,
        )
        with self._db.snapshot() as snapshot:
            reservable_jobs = snapshot.select(JOBS, where=JOBS.c.state.in_(list(reservable_states)))
        for job in reservable_jobs:
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

                    claims[worker.worker_id] = ReservationClaim(
                        job_id=job_wire,
                        entry_idx=idx,
                    )
                    claimed_worker_ids.add(worker.worker_id)
                    claimed_entries.add((job_wire, idx))
                    changed = True
                    break
        if changed and persisted:
            self._transitions.replace_reservation_claims(claims)
        return changed

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
        reads and writes go through ControllerTransitions which has its own lock.
        """
        claims = _read_reservation_claims(self._db)
        claims_changed = self._cleanup_stale_claims(claims)
        claims_changed = self._claim_workers_for_reservations(claims) or claims_changed
        if claims_changed:
            self._transitions.replace_reservation_claims(claims)

        timer = Timer()
        with slow_log(logger, "scheduling state reads", threshold_ms=50):
            pending_tasks = _schedulable_tasks(self._db)
            workers = healthy_active_workers_with_attributes(self._db)
        state_read_ms = timer.elapsed_ms()

        if not pending_tasks:
            return

        # Handle timeouts and reservation gates before scheduling.
        # Holder tasks participate in scheduling like normal tasks.
        # Cap non-coscheduled tasks per job to bound scheduling CPU time.
        schedulable_task_ids: list[JobName] = []
        jobs: dict[JobName, JobRequirements] = {}
        has_reservation: set[JobName] = set()
        has_direct_reservation: set[JobName] = set()
        tasks_per_job: dict[JobName, int] = defaultdict(int)
        cap = self._config.max_tasks_per_job_per_cycle
        jobs_by_id = _jobs_by_id(self._db, {task.job_id for task in pending_tasks})
        for task in pending_tasks:
            if not task.can_be_scheduled():
                continue
            job = jobs_by_id.get(task.job_id)
            if not job:
                continue
            if job.scheduling_deadline is not None and job.scheduling_deadline.expired():
                self._mark_task_unschedulable(task)
                continue
            # Gate: skip real tasks whose job has an unsatisfied reservation.
            # Holder tasks are always schedulable (they ARE the reservation).
            if not job.is_reservation_holder and not self._is_reservation_satisfied(job, claims):
                continue
            if cap > 0 and not job.is_coscheduled and tasks_per_job[task.job_id] >= cap:
                continue
            tasks_per_job[task.job_id] += 1
            schedulable_task_ids.append(task.task_id)
            if task.job_id not in jobs:
                jobs[task.job_id] = job_requirements_from_job(job)
                if job.request.HasField("reservation"):
                    has_reservation.add(task.job_id)
                    has_direct_reservation.add(task.job_id)
                elif _find_reservation_ancestor(self._db, task.job_id) is not None:
                    has_reservation.add(task.job_id)

        if not schedulable_task_ids:
            return

        # Inject reservation taints: claimed workers get a taint attribute,
        # non-reservation jobs get a NOT_EXISTS constraint for it.
        modified_workers = _inject_reservation_taints(workers, claims)
        jobs = _inject_taint_constraints(jobs, has_reservation, has_direct_reservation)

        with slow_log(logger, "building_counts", threshold_ms=50):
            building_counts = _building_counts(self._db)
        context = self._scheduler.create_scheduling_context(
            modified_workers,
            building_counts=building_counts,
            pending_tasks=schedulable_task_ids,
            jobs=jobs,
        )

        # Phase 1: soft preference — steer reservation tasks toward claimed workers.
        # Skips coscheduled jobs (they need atomic all-or-nothing via find_assignments).
        preference_assignments = _preference_pass(context, has_reservation, claims)

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
        """Commit assignments and enqueue worker dispatches in one state command."""
        command = [Assignment(task_id=task_id, worker_id=worker_id) for task_id, worker_id in assignments]
        result = self._transitions.queue_assignments(command)
        if result.has_real_dispatch:
            self._heartbeat_event.set()

    def _mark_task_unschedulable(self, task: Task) -> None:
        """Mark a task as unschedulable due to timeout."""
        job = _jobs_by_id(self._db, {task.job_id}).get(task.job_id)
        if job and job.request.HasField("scheduling_timeout"):
            timeout = Duration.from_proto(job.request.scheduling_timeout)
        else:
            timeout = None
        logger.warning(f"Task {task.task_id} exceeded scheduling timeout ({timeout}), marking as UNSCHEDULABLE")
        result = self._transitions.mark_task_unschedulable(
            task.task_id,
            reason=f"Scheduling timeout exceeded ({timeout})",
        )
        if result.tasks_to_kill:
            self.kill_tasks_on_workers(result.tasks_to_kill)

    def create_scheduling_context(self, workers: list[Worker]) -> SchedulingContext:
        """Create a scheduling context for the given workers."""
        building_counts = _building_counts(self._db)
        return self._scheduler.create_scheduling_context(
            workers,
            building_counts=building_counts,
        )

    def get_job_scheduling_diagnostics(self, job: Job, context: SchedulingContext) -> str:
        """Get detailed diagnostics for why a job cannot be scheduled."""
        req = job_requirements_from_job(job)
        schedulable_task_id = next(
            (t.task_id for t in _schedulable_tasks(self._db) if t.job_id == job.job_id),
            None,
        )
        num_tasks = len(tasks_for_job_with_attempts(self._db, job.job_id))
        return self._scheduler.get_job_scheduling_diagnostics(
            req,
            context,
            schedulable_task_id,
            num_tasks=num_tasks,
        )

    def kill_tasks_on_workers(self, task_ids: set[JobName]) -> None:
        """Buffer kill requests for delivery via next heartbeat.

        Called after state has marked tasks as killed. For each task that had
        a worker assigned, buffers the kill request for delivery via the next
        heartbeat to that worker.
        """
        any_buffered = False
        mapping = _task_worker_mapping(self._db, task_ids)
        workers = _workers_by_id(self._db, set(mapping.values()))
        for task_id, worker_id in mapping.items():
            worker = workers.get(worker_id)
            if worker is None:
                continue
            self._transitions.buffer_kill(worker_id, task_id.to_wire())
            any_buffered = True

        # Wake heartbeat thread to deliver buffered kills immediately
        if any_buffered:
            self._heartbeat_event.set()

    def _heartbeat_all_workers(self) -> None:
        """Send heartbeats to all registered workers.

        Uses state command boundaries: drain dispatch snapshot, execute RPC, then
        apply success/failure in one state command per worker.

        When heartbeat failure causes a worker to exceed the failure threshold,
        the state prunes it from active workers. We detect this and evict the
        cached stub + notify the autoscaler.

        Holds _heartbeat_lock for the entire round so begin_checkpoint() can
        wait for a complete quiescent state before snapshotting.
        """
        with self._heartbeat_lock:
            self._heartbeat_all_workers_inner()

    def _heartbeat_all_workers_inner(self) -> None:
        round_timer = Timer()

        # Phase 1: create snapshots for all healthy workers (lock-acquiring).
        with slow_log(logger, "heartbeat phase 1 (snapshot)", threshold_ms=100):
            snapshots: list[DispatchBatch] = []
            for w in healthy_active_workers_with_attributes(self._db):
                snapshot = self._transitions.drain_dispatch(w.worker_id)
                if snapshot:
                    snapshots.append(snapshot)

        if not snapshots:
            return

        # Phase 2: stream heartbeats through a bounded worker queue.
        work_queue: queue.Queue[DispatchBatch] = queue.Queue()
        result_queue: queue.Queue[tuple[DispatchBatch, cluster_pb2.HeartbeatResponse | None, str | None]] = queue.Queue()
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

        # Phase 3: consume all responses via complete_heartbeat / fail_heartbeat.
        # Each returns a HeartbeatAction: OK, TRANSIENT_FAILURE, or WORKER_FAILED.
        fail_count = 0
        failed_workers: list[str] = []
        with slow_log(logger, "heartbeat phase 3 (process results)", threshold_ms=500):
            for _ in snapshots:
                snapshot, response, error = result_queue.get()

                if response is not None:
                    result = self._transitions.complete_heartbeat(snapshot, response)
                    action = result.action
                    if result.tasks_to_kill:
                        self.kill_tasks_on_workers(result.tasks_to_kill)
                else:
                    logger.debug("Heartbeat error for %s: %s", snapshot.worker_id, error)
                    action = self._transitions.fail_heartbeat(snapshot, error or "unknown error")

                if action == HeartbeatAction.WORKER_FAILED:
                    fail_count += 1
                    failed_workers.append(snapshot.worker_id)
                    self.stub_factory.evict(snapshot.worker_address)
                    if self._autoscaler and snapshot.vm_address:
                        # Terminate the slice and get sibling VM addresses.
                        # All workers on the same slice must be failed immediately
                        # so their tasks (including reservation holders) are cascaded
                        # rather than waiting for heartbeat timeouts.
                        # TODO(#3425): This prunes sibling workers before their in-flight
                        # heartbeat results are processed, causing complete_heartbeat() to
                        # silently drop any logs/states those workers reported this round.
                        sibling_vms = self._autoscaler.notify_worker_failed(snapshot.vm_address)
                        if sibling_vms:
                            sibling_failed = self._transitions.fail_workers_by_vm_addresses(
                                sibling_vms,
                                reason=f"sibling worker at VM {snapshot.vm_address} failed, slice terminated",
                            )
                            for _wid, addr in sibling_failed:
                                self.stub_factory.evict(addr)
                            if sibling_failed:
                                fail_count += len(sibling_failed)
                                failed_workers.extend(wid for wid, _ in sibling_failed)
                                logger.info(
                                    "Failed %d sibling workers from slice: %s",
                                    len(sibling_failed),
                                    [wid for wid, _ in sibling_failed],
                                )
                elif action == HeartbeatAction.TRANSIENT_FAILURE:
                    fail_count += 1
                    failed_workers.append(snapshot.worker_id)

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
            workers = healthy_active_workers_with_attributes(self._db)
            with self._db.snapshot() as snapshot:
                active = snapshot.count(JOBS, where=JOBS.c.state == cluster_pb2.JOB_STATE_RUNNING)
            pending = len(_schedulable_tasks(self._db))
            logger.info(
                "Controller status: %d workers (%d failed), %d active jobs, %d pending tasks",
                len(workers),
                fail_count,
                active,
                pending,
            )

    def _do_heartbeat_rpc(
        self,
        snapshot: DispatchBatch,
    ) -> cluster_pb2.HeartbeatResponse:
        """Send a heartbeat RPC to a single worker.

        Raises:
            Exception on RPC failure (handled by caller via fail_heartbeat_for_worker)
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
        workers = healthy_active_workers_with_attributes(self._db)
        demand_entries = compute_demand_entries(
            self._db,
            self._scheduler,
            workers,
            reservation_claims=_read_reservation_claims(self._db),
        )
        self._autoscaler.update(demand_entries)
        self._persist_autoscaler_state()

    def _persist_autoscaler_state(self) -> None:
        """Persist autoscaler state as write-through DB metadata."""
        if self._autoscaler is None:
            return
        scaling_groups = [group.to_snapshot() for group in self._autoscaler.groups.values()]
        tracked_workers = self._autoscaler.to_tracked_worker_snapshots()
        self._transitions.persist_checkpoint_state(
            scaling_groups=scaling_groups,
            tracked_workers=tracked_workers,
        )

    def _build_vm_status_map(self) -> VmWorkerStatusMap:
        """Build a map of VM address to worker status for autoscaler.

        The autoscaler needs to look up worker status by VM address (not worker_id)
        because RemoteWorkerHandle only exposes the VM's IP address, not the worker's
        self-assigned ID. Workers self-discover their vm_address at startup via
        socket probe (env_probe.py).
        """
        result: VmWorkerStatusMap = {}
        with self._db.snapshot() as snapshot:
            workers = snapshot.select(WORKERS, where=WORKERS.c.active == 1)
        running_by_worker = running_tasks_by_worker(self._db, {worker.worker_id for worker in workers})
        for worker in workers:
            vm_addr = worker.metadata.vm_address
            if not vm_addr:
                logger.warning(
                    "Worker %s has no vm_address in metadata, skipping for autoscaler",
                    worker.worker_id,
                )
                continue

            result[vm_addr] = VmWorkerStatus(
                vm_address=vm_addr,
                running_task_ids=frozenset(tid.to_wire() for tid in running_by_worker.get(worker.worker_id, set())),
            )
        return result

    @property
    def _checkpoint_dir(self) -> Path:
        return self._transitions.db_path.parent / "controller-checkpoints"

    @property
    def _latest_checkpoint_path(self) -> Path:
        return self._checkpoint_dir / "latest.sqlite3"

    def _collect_checkpoint_result(self, created_at: Timestamp) -> CheckpointResult:
        with self._db.snapshot() as snapshot:
            job_count = snapshot.count(JOBS)
            task_count = snapshot.count(TASKS)
            worker_count = snapshot.count(WORKERS)
        return CheckpointResult(
            created_at=created_at,
            job_count=job_count,
            task_count=task_count,
            worker_count=worker_count,
        )

    def _maybe_periodic_checkpoint(self) -> None:
        """Write a best-effort periodic checkpoint DB copy."""
        if self._periodic_checkpoint_limiter is None:
            return
        if self._checkpoint_in_progress:
            return
        if not self._periodic_checkpoint_limiter.should_run():
            return
        try:
            created_at = Timestamp.now()
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self._checkpoint_dir / f"checkpoint-{created_at.epoch_ms()}.sqlite3"
            self._transitions.backup_to(path)
            shutil.copy2(path, self._latest_checkpoint_path)
            result = self._collect_checkpoint_result(created_at)
            logger.info(
                "Periodic checkpoint written: %s (jobs=%d tasks=%d workers=%d)",
                path,
                result.job_count,
                result.task_count,
                result.worker_count,
            )
        except Exception:
            logger.exception("Periodic checkpoint failed")

    def begin_checkpoint(self) -> tuple[str, CheckpointResult]:
        """Pause loops and write a consistent SQLite checkpoint copy."""
        self._checkpoint_in_progress = True
        try:
            # Wait for any in-flight heartbeat round to complete.
            with self._heartbeat_lock:
                created_at = Timestamp.now()
                self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
                path = self._checkpoint_dir / f"checkpoint-{created_at.epoch_ms()}.sqlite3"
                self._transitions.backup_to(path)
                shutil.copy2(path, self._latest_checkpoint_path)
                result = self._collect_checkpoint_result(created_at)
            logger.info(
                "Checkpoint written: %s (jobs=%d tasks=%d workers=%d)",
                path,
                result.job_count,
                result.task_count,
                result.worker_count,
            )
            return str(path), result
        finally:
            self._checkpoint_in_progress = False

    def restore_from_checkpoint(self, checkpoint_path: str | None = None) -> bool:
        """Restore full controller state from a checkpoint SQLite copy."""
        source = Path(checkpoint_path) if checkpoint_path else self._latest_checkpoint_path
        if not source.exists():
            logger.info("No checkpoint DB found at %s, starting fresh", source)
            return False

        self._transitions.restore_from(source)
        logger.info("Restored checkpoint DB from %s", source)

        with self._db.snapshot() as snapshot:
            scaling_rows = snapshot.select(
                SCALING_GROUPS,
                columns=(SCALING_GROUPS.c.name, SCALING_GROUPS.c.snapshot_proto),
            )
            tracked_rows = snapshot.select(
                TRACKED_WORKERS,
                columns=(
                    TRACKED_WORKERS.c.worker_id,
                    TRACKED_WORKERS.c.slice_id,
                    TRACKED_WORKERS.c.scale_group,
                    TRACKED_WORKERS.c.internal_address,
                ),
            )
        scaling_groups: dict[str, snapshot_pb2.ScalingGroupSnapshot] = {}
        for row in scaling_rows:
            snap = snapshot_pb2.ScalingGroupSnapshot()
            snap.ParseFromString(row.snapshot_proto)
            scaling_groups[row.name] = snap

        tracked_workers: dict[str, snapshot_pb2.TrackedWorkerSnapshot] = {}
        for row in tracked_rows:
            tracked_workers[row.worker_id] = snapshot_pb2.TrackedWorkerSnapshot(
                worker_id=row.worker_id,
                slice_id=row.slice_id,
                scale_group=row.scale_group,
                internal_address=row.internal_address,
            )

        # Restore autoscaler scaling groups (parallelized — each calls platform.list_slices())
        if self._autoscaler is not None:
            groups_to_restore = []
            for group_snap in scaling_groups.values():
                group = self._autoscaler.groups.get(group_snap.name)
                if group is None:
                    logger.warning(
                        "Checkpoint references scaling group %s which does not exist in config, skipping",
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

            # Workers from discarded slices remain in ControllerTransitions as healthy.
            # They will naturally fail heartbeat checks and be pruned once
            # consecutive failures exceed the threshold. This is intentional:
            # the heartbeat failure path handles cleanup of stale workers
            # including task reassignment and resource release.

            # Restore tracked workers into the autoscaler.
            tracked_proto = snapshot_pb2.ControllerSnapshot()
            tracked_proto.tracked_workers.extend(tracked_workers.values())
            restored_workers = restore_tracked_workers(tracked_proto)
            self._autoscaler.restore_tracked_workers(restored_workers)
            logger.info("Restored %d tracked workers", len(restored_workers))

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
    def state(self) -> ControllerTransitions:
        return self._transitions

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
        return _read_reservation_claims(self._db)

    @property
    def autoscaler(self) -> "Autoscaler | None":
        """The autoscaler instance, if autoscaling is enabled."""
        return self._autoscaler
