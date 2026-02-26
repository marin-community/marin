# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import logging
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from time import sleep
from typing import Protocol

import uvicorn

from iris.chaos import chaos
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.events import TaskAssignedEvent, TaskStateChangedEvent
from iris.cluster.controller.scheduler import (
    JobRequirements,
    Scheduler,
    SchedulingContext,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.state import (
    HEARTBEAT_FAILURE_THRESHOLD,
    ControllerJob,
    ControllerState,
    ControllerTask,
    ControllerWorker,
    HeartbeatSnapshot,
)
from iris.cluster.types import (
    JobName,
    VmWorkerStatus,
    VmWorkerStatusMap,
    WorkerId,
    get_device_type_enum,
    get_device_variant,
    normalize_constraints,
)
from iris.logging import get_global_buffer
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.time_utils import Duration, ExponentialBackoff, Timer

logger = logging.getLogger(__name__)

_SLOW_HEARTBEAT_MS = 5000
_HEALTH_SUMMARY_INTERVAL = 6  # every ~30s at 5s heartbeat interval


def job_requirements_from_job(job: ControllerJob) -> JobRequirements:
    """Convert a ControllerJob to scheduler-compatible JobRequirements."""
    return JobRequirements(
        resources=job.request.resources,
        constraints=list(job.request.constraints),
        is_coscheduled=job.is_coscheduled,
        coscheduling_group_by=job.coscheduling_group_by,
    )


def compute_demand_entries(state: ControllerState) -> list:
    """Compute demand entries from controller state."""
    from iris.cluster.controller.autoscaler import DemandEntry
    from iris.cluster.types import DeviceType

    demand_entries: list[DemandEntry] = []

    tasks_by_job: dict[JobName, list[ControllerTask]] = defaultdict(list)
    for task in state.peek_pending_tasks():
        if not task.can_be_scheduled():
            continue
        tasks_by_job[task.job_id].append(task)

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
            task_ids = [t.task_id.to_wire() for t in tasks]
            entry = DemandEntry(
                task_ids=task_ids,
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
            demand_entries.append(entry)
            continue

        for task in tasks:
            entry = DemandEntry(
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
            demand_entries.append(entry)

    return demand_entries


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

        self._state = ControllerState(heartbeat_failure_threshold=config.heartbeat_failure_threshold)
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

        self._heartbeat_iteration = 0

    def wake(self) -> None:
        """Signal the controller loop to run immediately.

        Called when events occur that may make scheduling possible:
        - New job submitted
        - New worker registered
        - Task finished (freeing capacity)
        """
        self._wake_event.set()

    def start(self) -> None:
        """Start main controller loop, dashboard server, and optionally autoscaler."""
        self._scheduling_thread = self._threads.spawn(self._run_scheduling_loop, name="scheduling-loop")
        self._heartbeat_thread = self._threads.spawn(self._run_heartbeat_loop, name="heartbeat-loop")

        # Create and start uvicorn server via spawn_server, which bridges the
        # ManagedThread stop_event to server.should_exit automatically.
        # timeout_keep_alive: uvicorn defaults to 5s, which races with client polling
        # intervals of the same length, causing TCP resets on idle connections. Use 120s
        # to safely cover long polling gaps during job waits.
        server_config = uvicorn.Config(
            self._dashboard._app,
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
        while not stop_event.is_set():
            self._wake_event.wait(timeout=self._config.scheduler_interval.to_seconds())
            self._wake_event.clear()

            if stop_event.is_set():
                break

            self._run_scheduling()

    def _run_autoscaler_loop(self, stop_event: threading.Event) -> None:
        """Autoscaler loop: runs on its own thread so blocking cloud API calls
        don't stall scheduling or heartbeats."""
        while not stop_event.is_set():
            stop_event.wait(timeout=self._autoscaler.evaluation_interval.to_seconds())
            if stop_event.is_set():
                break
            try:
                self._run_autoscaler_once()
            except Exception:
                logger.exception("Autoscaler loop iteration failed")

    def _run_heartbeat_loop(self, stop_event: threading.Event) -> None:
        """Heartbeat loop running on its own thread so slow RPCs don't block scheduling."""
        while not stop_event.is_set():
            self._heartbeat_event.wait(timeout=self._config.heartbeat_interval.to_seconds())
            self._heartbeat_event.clear()
            if stop_event.is_set():
                break
            self._heartbeat_all_workers()

    def _run_scheduling(self) -> None:
        """Run one scheduling cycle.

        Computes task assignments and buffers them for heartbeat delivery.
        No direct dispatch RPCs - tasks are delivered via the next heartbeat cycle.

        No lock is needed since only one scheduling thread exists. All state
        reads and writes go through ControllerState which has its own lock.
        """
        timer = Timer()
        pending_tasks = self._state.peek_pending_tasks()
        workers = self._state.get_available_workers()
        state_read_ms = timer.elapsed_ms()

        if not pending_tasks:
            return

        # Handle timeouts before scheduling (scheduler doesn't know about deadlines)
        schedulable_task_ids: list[JobName] = []
        jobs: dict[JobName, JobRequirements] = {}
        for task in pending_tasks:
            if not task.can_be_scheduled():
                continue
            job = self._state.get_job(task.job_id)
            if not job:
                continue
            if job.scheduling_deadline is not None and job.scheduling_deadline.expired():
                self._mark_task_unschedulable(task)
                continue
            schedulable_task_ids.append(task.task_id)
            if task.job_id not in jobs:
                jobs[task.job_id] = job_requirements_from_job(job)

        if not schedulable_task_ids:
            return

        building_counts = self._state.snapshot_building_counts()
        context = self._scheduler.create_scheduling_context(
            workers,
            building_counts=building_counts,
            pending_tasks=schedulable_task_ids,
            jobs=jobs,
        )
        result = self._scheduler.find_assignments(context)

        # Buffer assignments for heartbeat delivery (commits resources via TaskAssignedEvent)
        if result.assignments:
            self._buffer_assignments(result.assignments)
            logger.debug(
                "Scheduling cycle: %d assignments, %dms (state read: %dms)",
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

            for task_id, worker_id in job_assignments:
                task = self._state.get_task(task_id)
                if task is None:
                    continue

                # Commit resources via event
                self._state.handle_event(
                    TaskAssignedEvent(
                        task_id=task_id,
                        worker_id=worker_id,
                    )
                )

                # Build the run request
                request = cluster_pb2.Worker.RunTaskRequest(
                    task_id=task_id.to_wire(),
                    num_tasks=len(self._state.get_job_tasks(job_id)),
                    entrypoint=job.request.entrypoint,
                    environment=job.request.environment,
                    bundle_gcs_path=job.request.bundle_gcs_path,
                    resources=job.request.resources,
                    ports=list(job.request.ports),
                    attempt_id=task.current_attempt_id,
                    constraints=list(job.request.constraints),
                )
                # Copy timeout if set (check milliseconds field > 0)
                if job.request.timeout.milliseconds > 0:
                    request.timeout.CopyFrom(job.request.timeout)

                # Buffer dispatch (state handles the lock)
                self._state.buffer_dispatch(worker_id, request)

            # Wake heartbeat thread to deliver buffered dispatches immediately
            if job_assignments:
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
        """
        round_timer = Timer()

        # Phase 1: create snapshots for all healthy workers (lock-acquiring).
        # Timing this phase separately gives a lock-contention signal.
        snapshot_timer = Timer()
        snapshots: list[HeartbeatSnapshot] = []
        for w in self._state.get_available_workers():
            snapshot = self._state.begin_heartbeat(w.worker_id)
            if snapshot:
                snapshots.append(snapshot)
        snapshot_ms = snapshot_timer.elapsed_ms()

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
        for _ in snapshots:
            snapshot, response, error = result_queue.get()
            if error is not None:
                fail_count += 1
                logger.warning("Heartbeat error for %s: %s", snapshot.worker_id, error)
                self._handle_heartbeat_failure(snapshot, error)
                continue
            if response is not None:
                self._state.complete_heartbeat(snapshot, response)

        for future in worker_futures:
            future.cancel()

        elapsed = round_timer.elapsed_ms()
        level = logging.WARNING if elapsed > _SLOW_HEARTBEAT_MS else logging.DEBUG
        logger.log(
            level,
            "Heartbeat round: %d workers, %d failed, %dms (snapshot: %dms)",
            len(snapshots),
            fail_count,
            elapsed,
            snapshot_ms,
        )

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
                cluster_pb2.Controller.RunningTaskEntry(
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
        demand_entries = compute_demand_entries(self._state)
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
        # uvicorn 0.40+ sets .servers dynamically during startup(), not in __init__.
        # Use getattr since this is an unstable internal API at a system boundary.
        if self._server and self._server.started:
            servers = getattr(self._server, "servers", [])
            if servers and servers[0].sockets:
                return servers[0].sockets[0].getsockname()[1]
        return self._config.port

    @property
    def url(self) -> str:
        return f"http://{self._config.host}:{self.port}"

    @property
    def autoscaler(self) -> "Autoscaler | None":
        """The autoscaler instance, if autoscaling is enabled."""
        return self._autoscaler
