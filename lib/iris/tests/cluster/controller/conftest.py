# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for controller unit tests."""

import shutil
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace as _replace
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from finelog.client.log_client import Table
from finelog.rpc.logging_connect import LogServiceClientSync
from iris.cluster.backends.rpc.backend import (
    WORKER_RECONCILE_TEARDOWN_REASON,
    teardown_dead_workers,
)
from iris.cluster.backends.types import CloudSliceState
from iris.cluster.bundle import BundleStore
from iris.cluster.config import (
    AutoscalerConfig,
    GcpPlatformConfig,
    GcpSliceConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    SliceConfig,
    WorkerConfig,
)
from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintOp,
    DeviceType,
    PlacementRequirements,
    WellKnownAttribute,
    constraints_from_resources,
    device_variant_constraint,
    merge_constraints,
    preemptible_constraint,
    region_constraint,
    zone_constraint,
)
from iris.cluster.controller import ops, reads
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendCapability,
    ProviderUnsupportedError,
    ReconcileRequest,
    ReconcileResult,
    ScheduleInput,
    ScheduleRequest,
    ScheduleResult,
    TaskTarget,
    WorkerSource,
    assemble_scheduling_context,
    plans_from_snapshot,
    run_scheduling_decision,
)
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.log_stack import build_log_stack
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.ops.worker import apply_reconcile
from iris.cluster.controller.reads import SchedulableWorker
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.worker import WorkerReconcilePlan, WorkerReconcileResult
from iris.cluster.controller.run_template import RunTemplateCache
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.controller.schema import (
    task_attempts_table,
    tasks_table,
    worker_attributes_table,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, task_is_finished, task_row_can_be_scheduled
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerHealthEventKind, WorkerHealthTracker
from iris.cluster.platforms.gcp.fake import InMemoryGcpService
from iris.cluster.platforms.gcp.workers import GcpWorkerProvider
from iris.cluster.service_mode import ServiceMode
from iris.cluster.types import (
    DEFAULT_BACKEND_ID,
    TERMINAL_TASK_STATES,
    AcceleratorType,
    CapacityType,
    JobName,
    WorkerId,
    is_job_finished,
)
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration, RateLimiter, Timestamp
from sqlalchemy import func, select
from sqlalchemy import update as sa_update

from tests.cluster.backends.conftest import make_mock_platform
from tests.cluster.controller._test_support import ControllerTestState, set_task_state_for_test
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, apply_task_observations

check_task_can_be_scheduled = task_row_can_be_scheduled


def check_task_is_finished(task) -> bool:
    return task_is_finished(
        task.state,
        task.failure_count,
        task.max_retries_failure,
        task.preemption_count,
        task.max_retries_preemption,
    )


def check_is_job_finished(j) -> bool:
    """Whether a job row is in a terminal state."""
    return is_job_finished(j.state)


def run_worker_daemon_schedule(
    scheduler: Scheduler, worker_source: WorkerSource | None, request: ScheduleRequest
) -> ScheduleResult:
    """Assemble the scheduling context from the attached source and run the Iris
    pipeline — the worker-daemon fakes' shared mirror of ``RpcTaskBackend.schedule``."""
    assert worker_source is not None, "worker-daemon backend scheduled before worker source attached"
    context = assemble_scheduling_context(worker_source.scheduling_inputs(), request)
    return run_scheduling_decision(
        scheduler,
        ScheduleInput(
            context=context,
            max_tasks_per_job_per_cycle=request.max_tasks_per_job_per_cycle,
            trace=request.trace,
        ),
    )


def run_worker_daemon_reconcile(
    worker_source: WorkerSource | None,
    worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]],
    transport_events: list[WorkerHealthEvent],
) -> tuple[ReconcileResult, list[WorkerId]]:
    """Author reconcile effects from a fake's worker results and fold the observed
    liveness — the worker-daemon fakes' shared mirror of ``RpcTaskBackend.reconcile``'s
    tail (resolve observations into effects, then fold transport + BUILD_FAILED
    through the shared tracker reached via the source).

    Returns the committable result plus the workers the fold reaped; the caller
    stashes the latter for its ``run_teardown``, the way the real backend stashes
    on ``self._pending_dead``."""
    assert worker_source is not None, "worker-daemon backend reconciled before worker source attached"
    now = Timestamp.now()
    effects = apply_reconcile(worker_source, worker_results, now=now)
    events = transport_events + [
        WorkerHealthEvent(wid, WorkerHealthEventKind.BUILD_FAILED) for wid in effects.health.build_failed
    ]
    dead = worker_source.health.apply(events, now_ms=now.epoch_ms())
    return ReconcileResult(effects=effects), dead


def run_worker_daemon_teardown(
    worker_source: WorkerSource | None,
    dead_workers: list[WorkerId],
    autoscale: Callable[[AutoscaleRequest], AutoscaleResult],
    *,
    reason: str,
) -> None:
    """Tear down ``dead_workers`` through the production teardown — the fakes'
    shared mirror of ``RpcTaskBackend.teardown`` (source the failure-write
    collaborators from the worker source, drive the fake's own ``autoscale``)."""
    assert worker_source is not None, "worker-daemon backend teardown before worker source attached"
    teardown_dead_workers(
        dead_workers,
        db=worker_source.db,
        health=worker_source.health,
        endpoints=worker_source.endpoints,
        worker_attrs=worker_source.worker_attrs,
        autoscale=autoscale,
        reason=reason,
    )


class FakeProvider:
    """Minimal worker-daemon TaskBackend for tests exercising transitions, not RPCs."""

    name = "worker"
    capabilities = frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})
    autoscaler = None

    def __init__(self) -> None:
        # Real Iris scheduler: ``ctrl._run_scheduling`` routes the decision
        # through ``schedule`` now, so the fake must run the real pipeline for
        # scheduler/preemption tests to exercise placement.
        self._scheduler = Scheduler()
        # Attached by the controller, exactly as for RpcTaskBackend; the fake
        # sources its own workers through it rather than the controller slicing one.
        self.worker_source: WorkerSource | None = None
        self.advertised: dict[str, set[str]] = {}
        self.allowed_users: frozenset[str] = frozenset({"*"})
        # Workers this fake's reconcile fold reaped, awaiting run_teardown.
        self._pending_dead: list[WorkerId] = []

    def advertised_attributes(self) -> dict[str, set[str]]:
        return self.advertised

    def admits(self, user: str) -> bool:
        return "*" in self.allowed_users or user in self.allowed_users

    def configure_routing(self, advertised: dict[str, set[str]], allowed_users: frozenset[str]) -> None:
        self.advertised = advertised
        self.allowed_users = allowed_users

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        return run_worker_daemon_schedule(self._scheduler, self.worker_source, request)

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        # Mirror RpcTaskBackend: source the snapshot, build plans, report every
        # reached worker healthy with no observations (these tests drive task
        # transitions directly via the transition driver, not through RPCs), then
        # author effects + fold liveness exactly as the real backend does.
        assert self.worker_source is not None, "FakeProvider.reconcile called before worker source attached"
        snapshot = self.worker_source.reconcile_snapshot()
        plans = plans_from_snapshot(snapshot)
        worker_results = [(p, WorkerReconcileResult(worker_id=p.worker_id, observations=[], error=None)) for p in plans]
        events = [WorkerHealthEvent(p.worker_id, WorkerHealthEventKind.REACHED) for p in plans]
        result, dead = run_worker_daemon_reconcile(self.worker_source, worker_results, events)
        self._pending_dead.extend(dead)
        return result

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        return AutoscaleResult()

    def run_teardown(self) -> None:
        dead = self._pending_dead
        self._pending_dead = []
        run_worker_daemon_teardown(self.worker_source, dead, self.autoscale, reason=WORKER_RECONCILE_TEARDOWN_REASON)

    def teardown(self, dead_workers: list[WorkerId], *, reason: str) -> None:
        run_worker_daemon_teardown(self.worker_source, dead_workers, self.autoscale, reason=reason)

    def attach_autoscaler(self, autoscaler) -> None:
        self.autoscaler = autoscaler

    def attach_worker_source(self, source: WorkerSource) -> None:
        self.worker_source = source

    def attach_health(self, tracker: WorkerHealthTracker) -> None:
        self.health = tracker

    def get_process_status(
        self,
        target: TaskTarget,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        raise ProviderUnsupportedError("fake")

    def profile_task(
        self,
        target: TaskTarget,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        raise ProviderUnsupportedError("fake")

    def close(self) -> None:
        pass


@pytest.fixture
def state():
    """Create a fresh ControllerTestState with temp DB and log store."""
    with make_controller_state() as s:
        yield s


class MockController:
    """Mock that implements the ControllerProtocol surface used by ControllerServiceImpl."""

    def __init__(self):
        self.wake = Mock()
        self.request_worker_eviction = Mock()
        self.request_task_kicks = Mock()
        self.get_job_scheduling_diagnostics = Mock(return_value=None)
        self.last_scheduling_context = None
        self.provider = Mock()
        # A bare Mock would auto-create a truthy .autoscaler; the per-backend
        # feasibility/pending-hint paths read it, so pin it to "no autoscaler".
        self.provider.autoscaler = None
        self.capabilities = frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})
        self.run_template_cache: RunTemplateCache = RunTemplateCache(256)
        self.scale_group_to_backend: dict[str, str] = {}
        self.last_unroutable_jobs: dict[str, str] = {}
        self.backends: dict = {DEFAULT_BACKEND_ID: self.provider}

    def backend_id_for_scale_group(self, scale_group: str) -> str:
        return self.scale_group_to_backend.get(scale_group, DEFAULT_BACKEND_ID)


@pytest.fixture
def mock_controller() -> MockController:
    return MockController()


@pytest.fixture
def log_service(embedded_log_server) -> LogServiceClientSync:
    """A LogService RPC client against a fresh in-process finelog server.

    The native server makes pushed log entries immediately fetchable (RAM
    buffer), so push→fetch is synchronously visible within a test without any
    manual flush. The sync client exposes ``push_logs(request)`` /
    ``fetch_logs(request)``.
    """
    return LogServiceClientSync(address=embedded_log_server.address)


@pytest.fixture
def controller_service(state, log_client, mock_controller, tmp_path) -> ControllerServiceImpl:
    """ControllerServiceImpl with fresh DB, log service, and mock controller."""
    return ControllerServiceImpl(
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        health=state._health,
        endpoints=state._endpoints,
        worker_attrs=state._worker_attrs,
        endpoint_service=EndpointServiceImpl(db=state._db, endpoints=state._endpoints),
    )


# =============================================================================
# State factory helpers
# =============================================================================


@contextmanager
def make_controller_state(**kwargs):
    """Yield a ControllerTestState with a fresh temp DB, cleaning up on exit."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_test_"))
    try:
        db = ControllerDB(db_dir=tmp)
        yield ControllerTestState(db, **kwargs)
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def make_controller(tmp_path):
    """Factory for building ``Controller`` instances with automatic teardown.

    ``Controller.__init__`` attaches a ``RemoteLogHandler`` to the ``iris``
    logger and spawns a ``LogClient`` drain thread. Without ``stop()``, those
    leak across the test session and pull every ``iris.*`` log record into
    their internal queue — which can then be flushed into another test's
    monkeypatched ``LogServiceClientSync``. The factory tracks every
    constructed controller and ``stop()``s them at fixture teardown.

    Pass ``db=`` to inject a pre-built ``ControllerDB`` (otherwise the
    ``Controller`` opens one under ``config.local_state_dir``). Pass
    ``provider=`` to override the default ``FakeProvider``. Any remaining
    keyword arguments are forwarded to ``ControllerConfig``.

    Usage::

        def test_foo(make_controller, tmp_path):
            ctrl = make_controller(remote_state_dir="file:///tmp/iris-state")
            # Or inject an existing DB / provider:
            ctrl = make_controller(
                remote_state_dir="file:///tmp/iris-state",
                local_state_dir=tmp_path,
                db=my_db,
            )
    """
    created: list[Controller] = []

    def _factory(
        config: ControllerConfig | None = None,
        *,
        provider=None,
        backends: dict | None = None,
        backend_configs: dict | None = None,
        db: ControllerDB | None = None,
        **config_kwargs,
    ) -> Controller:
        if config is None:
            config_kwargs.setdefault("remote_state_dir", f"file://{tmp_path}/remote")
            config_kwargs.setdefault("local_state_dir", tmp_path / "local")
            config = ControllerConfig(**config_kwargs)
        elif config_kwargs:
            raise TypeError("make_controller: pass either a config or config kwargs, not both")
        log_stack = build_log_stack(
            log_service_address="",
            local_log_dir=config.local_state_dir / "log-server",
            host=config.host,
            worker_token=config.auth.worker_token if config.auth and config.auth.worker_token else None,
        )
        if backends is None:
            backends = {DEFAULT_BACKEND_ID: provider if provider is not None else FakeProvider()}
        controller = Controller(
            config=config,
            backends=backends,
            log_stack=log_stack,
            db=db,
            backend_configs=backend_configs,
        )
        created.append(controller)
        return controller

    yield _factory
    errors: list[BaseException] = []
    for controller in created:
        try:
            controller.stop()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def _spent_limiter() -> RateLimiter:
    """A ``RateLimiter`` whose ``should_run()`` returns False (already ran, long interval)."""
    limiter = RateLimiter(interval_seconds=1e9)
    limiter.mark_run()
    return limiter


def reconcile_once(ctrl: Controller) -> None:
    """Drive exactly one reconcile pass through the production control tick.

    Reconcile runs only as a phase of ``Controller._control_tick``, so this forces
    a reconcile-only tick: the reconcile phase fires while the schedule and
    autoscale phases are held off.
    """
    ctrl._force_reconcile = True
    ctrl._control_tick(
        woken=False,
        schedule_limiter=_spent_limiter(),
        reconcile_limiter=_spent_limiter(),
        autoscale_limiter=_spent_limiter(),
    )


def autoscale_once(ctrl: Controller) -> None:
    """Drive one autoscale pass through the production control tick.

    Autoscale runs only as a phase of ``Controller._control_tick``, always paired
    with a fresh schedule. In dry-run the tick short-circuits to the schedule-only
    path, so the autoscale backend call is suppressed.
    """
    ctrl._force_reconcile = False
    ctrl._control_tick(
        woken=False,
        schedule_limiter=_spent_limiter(),
        reconcile_limiter=_spent_limiter(),
        autoscale_limiter=RateLimiter(interval_seconds=0.0),
    )


def schedule_once(ctrl: Controller) -> None:
    """Drive exactly one schedule pass through the production control tick.

    A wake forces the schedule phase while reconcile and autoscale are held off,
    so only routing/placement (and its commit) runs this tick.
    """
    ctrl._force_reconcile = False
    ctrl._control_tick(
        woken=True,
        schedule_limiter=RateLimiter(interval_seconds=0.0),
        reconcile_limiter=_spent_limiter(),
        autoscale_limiter=_spent_limiter(),
    )


def make_test_entrypoint() -> job_pb2.RuntimeEntrypoint:
    entrypoint = job_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


def make_direct_job_request(
    name: str = "test-job",
    replicas: int = 1,
    task_image: str = "",
    coscheduling_group_by: str = "",
    priority_band: int = 0,
) -> controller_pb2.Controller.LaunchJobRequest:
    job_name = JobName.root("test-user", name)
    req = controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=replicas,
        task_image=task_image,
        priority_band=priority_band,
    )
    if coscheduling_group_by:
        req.coscheduling.group_by = coscheduling_group_by
    return req


def submit_direct_job(
    state: ControllerTestState,
    name: str,
    replicas: int = 1,
    task_image: str = "",
    coscheduling_group_by: str = "",
    priority_band: int = 0,
) -> list[JobName]:
    jid = JobName.root("test-user", name)
    req = make_direct_job_request(
        name,
        replicas,
        task_image=task_image,
        coscheduling_group_by=coscheduling_group_by,
        priority_band=priority_band,
    )
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=jid, request=req, ts=Timestamp.now(), run_template_cache=state._run_template_cache)
    with state._db.read_snapshot() as tx:
        rows = tx.execute(select(tasks_table.c.task_id).where(tasks_table.c.job_id == jid)).all()
    return [row.task_id for row in rows]


# =============================================================================
# DB query helpers (shared across test_scheduler, test_transitions, etc.)
# =============================================================================


def query_task(state: ControllerTestState, task_id: JobName):
    """Return the SA Row for ``task_id`` or None.

    Callers access ``row.state``, ``row.current_attempt_id``, etc. via attribute access.
    """
    with state._db.read_snapshot() as tx:
        return reads.get_task_detail(tx, task_id)


def query_attempt(state: ControllerTestState, task_id: JobName, attempt_id: int):
    """Return the SA Row for the given attempt or None."""
    with state._db.read_snapshot() as tx:
        return tx.execute(
            select(*reads.ATTEMPT_COLS).where(
                task_attempts_table.c.task_id == task_id,
                task_attempts_table.c.attempt_id == attempt_id,
            )
        ).first()


def query_job(state: ControllerTestState, job_id: JobName):
    """Return the SA Row for ``job_id`` joining jobs+job_config, or None."""
    with state._db.read_snapshot() as tx:
        return reads.get_job_detail(tx, job_id)


def query_job_row(state: ControllerTestState, job_id: JobName):
    """Return the SA Row for ``job_id`` (same as query_job; alias for scheduling projection tests)."""
    with state._db.read_snapshot() as tx:
        return reads.get_job_detail(tx, job_id)


@dataclass(frozen=True, slots=True)
class WorkerView:
    """Combined snapshot for tests that read DB row data + liveness in one call."""

    worker_id: WorkerId
    address: str
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    attributes: dict
    healthy: bool
    active: bool
    consecutive_failures: int
    last_heartbeat_ms: int


def _worker_view(row, liveness) -> WorkerView:
    return WorkerView(
        worker_id=row.worker_id,
        address=row.address,
        total_cpu_millicores=row.total_cpu_millicores,
        total_memory_bytes=row.total_memory_bytes,
        total_gpu_count=row.total_gpu_count,
        total_tpu_count=row.total_tpu_count,
        device_type=row.device_type,
        device_variant=row.device_variant,
        attributes=getattr(row, "attributes", {}),
        healthy=liveness.healthy,
        active=liveness.active,
        consecutive_failures=liveness.consecutive_failures,
        last_heartbeat_ms=liveness.last_heartbeat_ms,
    )


def query_worker(state: ControllerTestState, worker_id: WorkerId) -> WorkerView | None:
    with state._db.read_snapshot() as tx:
        row = reads.get_worker_detail(tx, worker_id)
    if row is None:
        return None
    return _worker_view(row, state._health.liveness(row.worker_id))


def query_tasks_for_job(state: ControllerTestState, job_id: JobName) -> list:
    """Return SA Rows for all tasks in ``job_id``."""
    with state._db.read_snapshot() as tx:
        return tx.execute(
            select(tasks_table).where(tasks_table.c.job_id == job_id).order_by(tasks_table.c.task_index)
        ).all()


def schedulable_tasks(state: ControllerTestState) -> list:
    """Return non-terminal task SA Rows eligible for scheduling, in priority order."""
    with state._db.read_snapshot() as tx:
        tasks = tx.execute(
            select(tasks_table)
            .where(tasks_table.c.state.not_in(list(TERMINAL_TASK_STATES)))
            .order_by(
                tasks_table.c.priority_neg_depth.asc(),
                tasks_table.c.priority_root_submitted_ms.asc(),
                tasks_table.c.submitted_at_ms.asc(),
                tasks_table.c.task_id.asc(),
            )
        ).all()
    return [t for t in tasks if check_task_can_be_scheduled(t)]


def building_counts(state: ControllerTestState) -> dict[WorkerId, int]:
    """Count tasks in BUILDING/ASSIGNED state per worker."""
    with state._db.read_snapshot() as tx:
        rows = tx.execute(
            select(task_attempts_table.c.worker_id, func.count().label("c"))
            .join(
                tasks_table,
                (tasks_table.c.task_id == task_attempts_table.c.task_id)
                & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
            )
            .where(tasks_table.c.state.in_([job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_ASSIGNED]))
            .group_by(task_attempts_table.c.worker_id)
            .order_by(task_attempts_table.c.worker_id.asc())
        ).all()
    return {row.worker_id: int(row.c) for row in rows}


def register_worker(
    state: ControllerTestState,
    worker_id: str,
    address: str,
    metadata: job_pb2.WorkerMetadata,
    healthy: bool = True,
    slice_id: str = "",
    scale_group: str = "",
) -> WorkerId:
    wid = WorkerId(worker_id)
    with state._db.transaction() as cur:
        ops.worker.register(
            cur,
            worker_id=wid,
            address=address,
            metadata=metadata,
            ts=Timestamp.now(),
            health=state._health,
            worker_attrs=state._worker_attrs,
            slice_id=slice_id,
            scale_group=scale_group,
        )
    if not healthy:
        state._health.set_health_for_test(wid, healthy=False)
    return wid


def inject_device_constraints(request: controller_pb2.Controller.LaunchJobRequest) -> None:
    """Auto-inject device constraints from the resource spec, mirroring service.py.

    In production, the service layer merges auto-generated device constraints
    into the request before storing the job. Tests bypass the service layer,
    so we replicate that logic here.
    """
    auto = constraints_from_resources(request.resources)
    if not auto:
        return
    user = [Constraint.from_proto(c) for c in request.constraints]
    merged = merge_constraints(auto, user)
    del request.constraints[:]
    for c in merged:
        request.constraints.append(c.to_proto())


def submit_job(
    state: ControllerTestState,
    job_id: str,
    request: controller_pb2.Controller.LaunchJobRequest,
    timestamp_ms: int | None = None,
) -> list:
    """Submit a job and return created task rows.

    Auto-injects resource-derived device constraints to mirror service-layer
    behavior, then submits and returns the created tasks.
    """
    inject_device_constraints(request)
    jid = JobName.from_string(job_id) if job_id.startswith("/") else JobName.root("test-user", job_id)
    request.name = jid.to_wire()
    with state._db.transaction() as cur:
        ops.job.submit(
            cur,
            job_id=jid,
            request=request,
            ts=Timestamp.from_ms(timestamp_ms) if timestamp_ms is not None else Timestamp.now(),
            run_template_cache=state._run_template_cache,
        )
    return query_tasks_for_job(state, jid)


# =============================================================================
# Shared test helpers (deduplicated from test_transitions, test_scheduler,
# test_service, test_dashboard)
# =============================================================================


@dataclass(frozen=True, slots=True)
class TaskWithAttempts:
    """SA Row for a task with its attempt rows attached under ``.attempts``."""

    _row: object
    attempts: list

    def __getattr__(self, name: str):
        return getattr(self._row, name)


def query_tasks_with_attempts(state: ControllerTestState, job_id: JobName) -> list[TaskWithAttempts]:
    """Return task rows with their attempt rows attached under ``.attempts``."""
    with state._db.read_snapshot() as tx:
        task_rows = tx.execute(
            select(tasks_table).where(tasks_table.c.job_id == job_id).order_by(tasks_table.c.task_index.asc())
        ).all()
        if not task_rows:
            return []
        task_ids = [t.task_id for t in task_rows]
        attempt_rows = tx.execute(
            select(task_attempts_table)
            .where(task_attempts_table.c.task_id.in_(task_ids))
            .order_by(task_attempts_table.c.task_id.asc(), task_attempts_table.c.attempt_id.asc())
        ).all()
    attempts_by_task: dict = {}
    for a in attempt_rows:
        attempts_by_task.setdefault(a.task_id, []).append(a)
    return [TaskWithAttempts(_row=t, attempts=attempts_by_task.get(t.task_id, [])) for t in task_rows]


def query_task_with_attempts(state: ControllerTestState, task_id: JobName) -> TaskWithAttempts | None:
    """Return the task row with attempts attached, or None."""
    with state._db.read_snapshot() as tx:
        task_row = tx.execute(select(tasks_table).where(tasks_table.c.task_id == task_id)).first()
        if task_row is None:
            return None
        attempt_rows = tx.execute(
            select(task_attempts_table)
            .where(task_attempts_table.c.task_id == task_id)
            .order_by(task_attempts_table.c.attempt_id.asc())
        ).all()
    return TaskWithAttempts(_row=task_row, attempts=list(attempt_rows))


def make_job_request(
    name: str = "test-job",
    cpu: int = 1,
    memory_bytes: int = 1024**3,
    replicas: int = 1,
    max_retries_failure: int = 0,
    max_retries_preemption: int = 0,
    max_task_failures: int = 0,
    scheduling_timeout_seconds: int = 0,
    priority_band: int = 0,
    task_image: str = "",
) -> controller_pb2.Controller.LaunchJobRequest:
    job_name = JobName.from_string(name) if name.startswith("/") else JobName.root("test-user", name)
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=cpu * 1000, memory_bytes=memory_bytes),
        environment=job_pb2.EnvironmentConfig(),
        max_retries_failure=max_retries_failure,
        max_retries_preemption=max_retries_preemption,
        max_task_failures=max_task_failures,
        replicas=replicas,
        priority_band=priority_band,
        task_image=task_image,
    )
    if scheduling_timeout_seconds > 0:
        request.scheduling_timeout.CopyFrom(duration_to_proto(Duration.from_seconds(scheduling_timeout_seconds)))
    return request


def make_worker_metadata(
    cpu: int = 10,
    memory_bytes: int = 10 * 1024**3,
    disk_bytes: int = 10 * 1024**3,
    gpu_count: int = 0,
    gpu_name: str = "",
    tpu_name: str = "",
) -> job_pb2.WorkerMetadata:
    """Build WorkerMetadata with device config and well-known attributes.

    Populates device-type and device-variant attributes so constraint-based
    scheduling works the same way as production.
    """
    device = job_pb2.DeviceConfig()
    if tpu_name:
        device.tpu.CopyFrom(job_pb2.TpuDevice(variant=tpu_name))
    elif gpu_count > 0:
        device.gpu.CopyFrom(job_pb2.GpuDevice(variant=gpu_name or "auto", count=gpu_count))
    else:
        device.cpu.CopyFrom(job_pb2.CpuDevice(variant="cpu"))

    meta = job_pb2.WorkerMetadata(
        hostname="test-worker",
        ip_address="127.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_bytes,
        disk_bytes=disk_bytes,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        tpu_name=tpu_name,
        device=device,
    )

    if tpu_name:
        meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "tpu"
        meta.attributes[WellKnownAttribute.DEVICE_VARIANT].string_value = tpu_name.lower()
    elif gpu_count > 0:
        meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "gpu"
        if gpu_name:
            meta.attributes[WellKnownAttribute.DEVICE_VARIANT].string_value = gpu_name.lower()
    else:
        meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "cpu"

    return meta


def worker_running_tasks(state: ControllerTestState, worker_id: WorkerId) -> frozenset[JobName]:
    with state._db.read_snapshot() as tx:
        rows = tx.execute(
            select(tasks_table.c.task_id)
            .join(
                task_attempts_table,
                (tasks_table.c.task_id == task_attempts_table.c.task_id)
                & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
            )
            .where(
                task_attempts_table.c.worker_id == worker_id,
                tasks_table.c.state.in_(list(ACTIVE_TASK_STATES)),
            )
        ).all()
    return frozenset(row.task_id for row in rows)


def _decode_attr_value(row) -> AttributeValue:
    """Decode a worker_attributes SA row value based on value_type."""
    if row.value_type == "int":
        return AttributeValue(int(row.int_value))
    if row.value_type == "float":
        return AttributeValue(float(row.float_value))
    return AttributeValue(str(row.str_value or ""))


def hydrate_worker_attributes(state: ControllerTestState, workers: list) -> list:
    if not workers:
        return workers
    worker_ids = [w.worker_id for w in workers]
    with state._db.read_snapshot() as tx:
        attr_rows = tx.execute(
            select(worker_attributes_table).where(worker_attributes_table.c.worker_id.in_(worker_ids))
        ).all()
    attrs_by_worker: dict = {}
    for row in attr_rows:
        attrs_by_worker.setdefault(row.worker_id, {})[row.key] = _decode_attr_value(row)
    return [_replace(w, attributes=attrs_by_worker.get(w.worker_id, {})) for w in workers]


def healthy_active_workers(state: ControllerTestState) -> list[SchedulableWorker]:
    with state._db.read_snapshot() as tx:
        return reads.healthy_active_workers_with_attributes(tx, state._health, state._worker_attrs)


def dispatch_task(state: ControllerTestState, task, worker_id: WorkerId) -> None:
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=query_task(state, task.task_id).current_attempt_id,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                        )
                    ],
                )
            ],
            health=state._health,
            endpoints=state._endpoints,
            now=Timestamp.now(),
        )


def transition_task(
    state: ControllerTestState,
    task_id: JobName,
    new_state: int,
    *,
    error: str | None = None,
    exit_code: int | None = None,
) -> object:
    task = query_task_with_attempts(state, task_id)
    assert task is not None
    if new_state == job_pb2.TASK_STATE_KILLED:
        with state._db.transaction() as cur:
            ops.job.cancel(
                cur, job_id=task.job_id, reason=error or "killed", endpoints=state._endpoints, health=state._health
            )
        return state
    # Compute worker_id: prefer current attempt's worker, fall back to current_worker_id.
    current_attempt = task.attempts[-1] if task.attempts else None
    worker_id = current_attempt.worker_id if current_attempt is not None else task.current_worker_id
    if worker_id is None:
        set_task_state_for_test(
            state,
            task_id,
            new_state,
            error=error,
            exit_code=exit_code,
        )
        return state
    with state._db.transaction() as cur:
        return apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task_id,
                            attempt_id=task.current_attempt_id,
                            new_state=new_state,
                            error=error,
                            exit_code=exit_code,
                        )
                    ],
                )
            ],
            health=state._health,
            endpoints=state._endpoints,
            now=Timestamp.now(),
        )


def fail_worker(state: ControllerTestState, worker_id: WorkerId, error: str) -> None:
    """Force-remove a worker via the explicit kill path used by the reaper thread."""
    ops.worker.fail(
        state._db,
        worker_ids=[str(worker_id)],
        reason=error,
        health=state._health,
        endpoints=state._endpoints,
        worker_attrs=state._worker_attrs,
    )


# =============================================================================
# ControllerTestHarness
# =============================================================================


class ControllerTestHarness:
    """Wraps ControllerTestState with ergonomic helpers for the common
    register-workers -> submit-jobs -> dispatch -> transition test pattern."""

    def __init__(self, state: ControllerTestState):
        self.state = state

    def add_worker(
        self,
        worker_id: str = "w1",
        address: str | None = None,
        *,
        cpu: int = 10,
        memory_bytes: int = 10 * 1024**3,
        disk_bytes: int = 10 * 1024**3,
        gpu_count: int = 0,
        gpu_name: str = "",
        tpu_name: str = "",
        healthy: bool = True,
    ) -> WorkerId:
        meta = make_worker_metadata(
            cpu=cpu,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            tpu_name=tpu_name,
        )
        return register_worker(self.state, worker_id, address or f"{worker_id}:8080", meta, healthy=healthy)

    def submit(self, name: str = "test-job", *, cpu: int = 1, replicas: int = 1, **kwargs) -> list:
        req = make_job_request(name=name, cpu=cpu, replicas=replicas, **kwargs)
        return submit_job(self.state, name, req)

    def dispatch(self, task, worker_id: WorkerId) -> None:
        dispatch_task(self.state, task, worker_id)

    def transition(self, task_id: JobName, new_state: int, **kwargs) -> None:
        transition_task(self.state, task_id, new_state, **kwargs)

    def query_task(self, task_id: JobName):
        return query_task(self.state, task_id)

    def query_job(self, job_id: JobName):
        return query_job(self.state, job_id)


@pytest.fixture
def harness(state) -> ControllerTestHarness:
    return ControllerTestHarness(state)


# =============================================================================
# Autoscaler helpers (shared by test_autoscaler, test_demand_routing,
# test_autoscaler_integration)
# =============================================================================


DEFAULT_RESOURCES = ScaleGroupResources(
    cpu_millicores=128000,
    memory_bytes=128 * 1024**3,
    disk_bytes=100 * 1024**3,
    device_type=AcceleratorType.TPU,
    device_variant="v5p-8",
    device_count=8,
)


def ensure_scale_group_resources(config: ScaleGroupConfig) -> ScaleGroupConfig:
    if config.resources is None:
        config.resources = DEFAULT_RESOURCES.model_copy(deep=True)
    if config.num_vms is None:
        config.num_vms = 1
    return config


def make_scale_group_config(
    *,
    accelerator_type: AcceleratorType = AcceleratorType.TPU,
    accelerator_variant: str = "v5p-8",
    runtime_version: str | None = None,
    zones: list[str] | None = None,
    capacity_type: CapacityType | None = None,
    **kwargs: object,
) -> ScaleGroupConfig:
    config = ensure_scale_group_resources(ScaleGroupConfig(**kwargs))
    config.resources.device_type = accelerator_type
    if accelerator_variant:
        config.resources.device_variant = accelerator_variant
    if capacity_type is not None:
        config.resources.capacity_type = capacity_type

    # Derive slice template fields from resources, matching what
    # ScaleGroupConfig._derive_slice_template() does in production config loading.
    # GcpWorkerProvider validates these fields on create_slice().
    if config.slice_template is None:
        config.slice_template = SliceConfig()
    template = config.slice_template
    template.accelerator_type = accelerator_type
    if accelerator_variant:
        template.accelerator_variant = accelerator_variant
    if capacity_type is not None:
        template.capacity_type = capacity_type
    if accelerator_type == AcceleratorType.GPU and config.resources.device_count > 0:
        template.gpu_count = config.resources.device_count

    if template.gcp is None:
        template.gcp = GcpSliceConfig()
    gcp = template.gcp
    if runtime_version:
        gcp.runtime_version = runtime_version
    elif accelerator_type == AcceleratorType.TPU and not gcp.runtime_version:
        gcp.runtime_version = "v2-alpha-tpuv5"
    if zones:
        gcp.zone = zones[0]
    elif not gcp.zone:
        gcp.zone = "us-central1-a"

    return config


def make_demand_entries(
    count: int,
    *,
    device_type: DeviceType = DeviceType.TPU,
    device_variant: str | None = "v5p-8",
    device_variants: frozenset[str] | None = None,
    capacity_type: CapacityType | None = None,
    required_regions: frozenset[str] | None = None,
    required_zones: frozenset[str] | None = None,
    task_prefix: str = "task",
) -> list[DemandEntry]:
    if count <= 0:
        return []
    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024)
    if device_type == DeviceType.TPU:
        resources.device.tpu.variant = device_variant or ""
    elif device_type == DeviceType.GPU:
        resources.device.gpu.variant = device_variant or ""
    elif device_type == DeviceType.CPU:
        resources.device.cpu.variant = ""
    effective_variants = device_variants
    if effective_variants is None and device_variant is not None:
        effective_variants = frozenset({device_variant})
    preemptible = (capacity_type == CapacityType.PREEMPTIBLE) if capacity_type is not None else None
    normalized = PlacementRequirements(
        device_type=device_type,
        device_variants=effective_variants,
        preemptible=preemptible,
        required_regions=required_regions,
        required_zones=required_zones,
    )

    constraint_list: list[Constraint] = []
    if device_type is not None:
        constraint_list.append(
            Constraint.create(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value=device_type.value)
        )
    if effective_variants:
        constraint_list.append(device_variant_constraint(sorted(effective_variants)))
    if capacity_type is not None:
        constraint_list.append(preemptible_constraint(capacity_type == CapacityType.PREEMPTIBLE))
    if required_regions:
        constraint_list.append(region_constraint(sorted(required_regions)))
    if required_zones:
        for z in sorted(required_zones):
            constraint_list.append(zone_constraint(z))
    return [
        DemandEntry(
            task_ids=(f"{task_prefix}-{i}",),
            coschedule_group_id=None,
            normalized=normalized,
            constraints=constraint_list,
            resources=resources,
        )
        for i in range(count)
    ]


def make_big_demand_entries(
    count: int,
    *,
    cpu_millicores: int = 32000,
    memory_bytes: int = 32 * 1024**3,
    disk_bytes: int = 0,
    device_type: DeviceType = DeviceType.CPU,
    device_variants: frozenset[str] | None = None,
    task_prefix: str = "task",
    coschedule_group_id: str | None = None,
) -> list[DemandEntry]:
    """Create demand entries with explicit resource sizes for packing tests."""
    resources = job_pb2.ResourceSpecProto(
        cpu_millicores=cpu_millicores,
        memory_bytes=memory_bytes,
        disk_bytes=disk_bytes,
    )
    normalized = PlacementRequirements(
        device_type=device_type,
        device_variants=device_variants,
        preemptible=None,
        required_regions=None,
        required_zones=None,
    )
    if coschedule_group_id:
        return [
            DemandEntry(
                task_ids=tuple(f"{task_prefix}-{i}" for i in range(count)),
                coschedule_group_id=coschedule_group_id,
                normalized=normalized,
                constraints=[],
                resources=resources,
            )
        ]
    return [
        DemandEntry(
            task_ids=(f"{task_prefix}-{i}",),
            coschedule_group_id=None,
            normalized=normalized,
            constraints=[],
            resources=resources,
        )
        for i in range(count)
    ]


def make_autoscaler(
    scale_groups: dict[str, ScalingGroup],
    config: AutoscalerConfig | None = None,
    platform: MagicMock | None = None,
    base_worker_config: WorkerConfig | None = None,
    provisioning_table: Table | None = None,
) -> Autoscaler:
    """Create an Autoscaler with the given groups."""
    mock_platform = platform or make_mock_platform()

    if config:
        return Autoscaler.from_config(
            scale_groups=scale_groups,
            config=config,
            platform=mock_platform,
            base_worker_config=base_worker_config,
            provisioning_table=provisioning_table,
        )
    else:
        return Autoscaler(
            scale_groups=scale_groups,
            evaluation_interval=Duration.from_seconds(0.1),
            platform=mock_platform,
            base_worker_config=base_worker_config,
            provisioning_table=provisioning_table,
        )


def mark_discovered_ready(group: ScalingGroup, handles: list[MagicMock], timestamp: Timestamp | None = None) -> None:
    """Mark discovered slices as READY with their worker IDs."""
    for handle in handles:
        worker_ids = [w.worker_id for w in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids, timestamp=timestamp)


def mark_all_slices_ready(group: ScalingGroup) -> None:
    """Mark all tracked slices as READY with their worker IDs.

    Used after advance_all_tpus() to simulate the controller detecting that
    slices have finished booting and are ready for work.
    """
    for handle in group.slice_handles():
        desc = handle.describe()
        if desc.state == CloudSliceState.READY:
            worker_ids = [w.worker_id for w in desc.workers]
            group.mark_slice_ready(handle.slice_id, worker_ids)


def make_gcp_provider(
    config: ScaleGroupConfig,
    zone: str = "us-central1-a",
) -> tuple[GcpWorkerProvider, InMemoryGcpService]:
    """Create a GcpWorkerProvider backed by InMemoryGcpService(DRY_RUN).

    Returns both the provider and the backing service so tests can inject
    failures and advance TPU state.
    """
    service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project", label_prefix="iris")
    gcp_config = GcpPlatformConfig(project_id="test-project", zones=[zone])
    provider = GcpWorkerProvider(gcp_config=gcp_config, label_prefix="iris", worker_port=10001, gcp_service=service)
    return provider, service


def advance_all_tpus(service: InMemoryGcpService, state: str = "READY") -> None:
    """Transition all TPUs in an InMemoryGcpService to the given state."""
    for name, zone in list(service._tpus.keys()):
        if service._tpus[(name, zone)].state != state:
            service.advance_tpu_state(name, zone, state)


def set_task_band(db: ControllerDB, task_id: JobName, band: int) -> None:
    """Directly set priority_band on a task row for testing.

    Prefer setting priority_band on the LaunchJobRequest for new submissions.
    This helper is still needed for tests that change a task's band mid-flight
    (e.g., simulating admin band overrides or budget-triggered demotions).
    """
    with db.transaction() as tx:
        tx.execute(sa_update(tasks_table).where(tasks_table.c.task_id == task_id).values(priority_band=band))
