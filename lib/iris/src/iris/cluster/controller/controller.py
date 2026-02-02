# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Iris Controller logic for connecting state, scheduler and managing workers."""

import logging
import threading
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import sleep
from typing import Protocol

import uvicorn

from iris.chaos import chaos
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.events import (
    TaskAssignedEvent,
    TaskStateChangedEvent,
    WorkerHeartbeatEvent,
    WorkerHeartbeatFailedEvent,
)
from iris.cluster.controller.scheduler import Scheduler, SchedulingContext, TaskScheduleResult
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.state import (
    ControllerState,
    ControllerTask,
    ControllerWorker,
    get_device_type_enum,
    get_device_variant,
)
from iris.cluster.types import JobId, TaskId, WorkerId, VmWorkerStatus, VmWorkerStatusMap, PREEMPTIBLE_ATTRIBUTE_KEY
from iris.cluster.vm.autoscaler import Autoscaler
from iris.logging import get_global_buffer
from iris.managed_thread import ThreadContainer
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.time_utils import Duration, ExponentialBackoff, Timestamp

logger = logging.getLogger(__name__)


@dataclass
class WorkerDispatch:
    """Transient per-worker dispatch state owned by the controller scheduling loop.

    Separate from ControllerWorker (state.py) which holds durable worker metadata.
    """

    dispatch_outbox: list[cluster_pb2.Worker.RunTaskRequest] = field(default_factory=list)
    kill_outbox: list[str] = field(default_factory=list)


def _extract_preemptible_preference(constraints: Sequence[cluster_pb2.Constraint]) -> bool | None:
    """Extract preemptible preference from job constraints.

    Returns True if the job requires preemptible workers, False if it requires
    non-preemptible workers, or None if no preference is expressed.
    """
    for c in constraints:
        if c.key == PREEMPTIBLE_ATTRIBUTE_KEY and c.op == cluster_pb2.CONSTRAINT_OP_EQ:
            if c.value.HasField("string_value"):
                return c.value.string_value == "true"
    return None


def compute_demand_entries(state: ControllerState) -> list:
    """Compute demand entries from controller state.

    Groups pending tasks by device type and variant, returning DemandEntry objects.
    CPU-only tasks get DeviceType.CPU with variant=None.

    For coscheduled jobs, all tasks in the job share a single slice, so we count
    the job once (not each task). For non-coscheduled jobs, each task counts as
    one slice of demand.

    Args:
        state: Controller state with pending tasks.

    Returns:
        List of DemandEntry objects with device_type, device_variant, and count.
    """
    from iris.cluster.vm.autoscaler import DemandEntry
    from iris.cluster.types import DeviceType

    @dataclass
    class DemandAccumulator:
        count: int = 0
        total_cpu: int = 0
        total_memory_bytes: int = 0

    demand_by_device: dict[tuple[DeviceType, str | None, bool | None], DemandAccumulator] = {}
    coscheduled_jobs_counted: set[str] = set()

    for task in state.peek_pending_tasks():
        job = state.get_job(task.job_id)
        if not job:
            continue

        # For coscheduled jobs, only count once per job (all tasks share one slice)
        if job.is_coscheduled:
            if job.job_id in coscheduled_jobs_counted:
                continue
            coscheduled_jobs_counted.add(job.job_id)

        device = job.request.resources.device
        device_type = get_device_type_enum(device)
        device_variant = get_device_variant(device) if device_type != DeviceType.CPU else None
        preemptible_pref = _extract_preemptible_preference(job.request.constraints)

        key = (device_type, device_variant, preemptible_pref)
        if key not in demand_by_device:
            demand_by_device[key] = DemandAccumulator()
        acc = demand_by_device[key]
        acc.count += 1
        acc.total_cpu += job.request.resources.cpu
        acc.total_memory_bytes += job.request.resources.memory_bytes

    return [
        DemandEntry(
            device_type=dt,
            device_variant=variant,
            count=acc.count,
            total_cpu=acc.total_cpu,
            total_memory_bytes=acc.total_memory_bytes,
            preemptible=preemptible,
        )
        for (dt, variant, preemptible), acc in demand_by_device.items()
    ]


class WorkerClient(Protocol):
    """Protocol for worker RPC client.

    Matches client-side WorkerServiceClientSync signature. The server Protocol has different signatures.
    """

    def get_task_status(
        self,
        request: cluster_pb2.Worker.GetTaskStatusRequest,
    ) -> cluster_pb2.TaskStatus: ...

    def list_tasks(
        self,
        request: cluster_pb2.Worker.ListTasksRequest,
    ) -> cluster_pb2.Worker.ListTasksResponse: ...

    def health_check(
        self,
        request: cluster_pb2.Empty,
    ) -> cluster_pb2.Worker.HealthResponse: ...

    def heartbeat(
        self,
        request: cluster_pb2.HeartbeatRequest,
    ) -> cluster_pb2.HeartbeatResponse: ...


class WorkerStubFactory(Protocol):
    """Factory for getting worker RPC stubs."""

    def get_stub(self, address: str) -> WorkerClient:
        """Get a worker stub for the given address.

        Args:
            address: Worker address in "host:port" format

        Returns:
            A WorkerClient stub for making RPC calls
        """
        ...


class RpcWorkerStubFactory:
    """Factory that creates real gRPC client stubs for worker communication."""

    def get_stub(self, address: str) -> WorkerClient:
        return WorkerServiceClientSync(
            address=f"http://{address}",
            timeout_ms=10000,
        )


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

    scheduler_interval_seconds: float = 0.5
    """How often to run the scheduling loop (in seconds)."""

    worker_timeout: Duration = field(default_factory=lambda: Duration.from_seconds(60.0))
    """How long without worker heartbeats before declaring a worker unavailable."""

    max_dispatch_parallelism: int = 32
    """Maximum number of concurrent RPC dispatch operations."""

    autoscaler_enabled: bool = False
    worker_access_address: str = ""


class Controller:
    """Unified controller managing all components and lifecycle.

    Runs two background loops:
    - Scheduling loop: finds task assignments, checks worker timeouts, runs autoscaler
    - Heartbeat loop: sends heartbeat RPCs to workers, delivering buffered dispatches/kills

    Separating these ensures slow heartbeat RPCs don't block scheduling and vice versa.

    When an autoscaler is provided, manages it in a background thread that
    provisions/terminates VM slices based on demand.

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
    ):
        if not config.bundle_prefix:
            raise ValueError(
                "bundle_prefix is required. Set via ControllerConfig.bundle_prefix. "
                "Example: bundle_prefix='gs://my-bucket/iris/bundles'"
            )

        self._config = config
        self._stub_factory = worker_stub_factory

        self._state = ControllerState()
        self._scheduler = Scheduler(self._state)
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

        # Thread management
        self._threads = ThreadContainer()
        self._stop = False
        self._wake_event = threading.Event()
        self._heartbeat_event = threading.Event()
        self._server: uvicorn.Server | None = None

        # Per-worker dispatch state (outboxes). Protected by _dispatch_lock since
        # the scheduling thread writes and the heartbeat thread reads+drains.
        self._dispatch: dict[WorkerId, WorkerDispatch] = {}
        self._dispatch_lock = threading.Lock()

        # Thread pool for parallel heartbeat dispatch
        self._dispatch_executor = ThreadPoolExecutor(
            max_workers=config.max_dispatch_parallelism,
            thread_name_prefix="dispatch",
        )

        # Autoscaler (passed in, configured in start() if provided)
        self._autoscaler: Autoscaler | None = autoscaler
        self._last_autoscaler_run: float = 0.0

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
        self._stop = False

        # Start main controller loop
        self._threads.spawn(
            target=self._run_scheduling_loop,
            name="controller-scheduling",
        )

        # Start heartbeat loop (separate from scheduling so slow RPCs don't block scheduling)
        self._threads.spawn(
            target=self._run_heartbeat_loop,
            name="controller-heartbeat",
        )

        # Start dashboard server in background thread
        self._threads.spawn(
            target=self._run_server,
            name="controller-server",
        )

        # Log autoscaler configuration if provided (runs from scheduling loop, not separate thread)
        if self._autoscaler:
            logger.info("Autoscaler configured with %d scale groups", len(self._autoscaler.groups))

        # Wait for server startup with exponential backoff
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=5.0,
        )

    def stop(self, timeout: float = 30.0) -> None:
        """Stop all background components gracefully.

        Args:
            timeout: Maximum time to wait for all threads to exit (default 30s).
                    If threads don't exit within this time, they will be left running.
        """
        self._stop = True
        self._wake_event.set()
        self._heartbeat_event.set()

        # Shutdown dispatch executor first so heartbeat thread can exit as_completed()
        # Don't cancel futures to avoid httpx logging to closed stdout/stderr
        self._dispatch_executor.shutdown(wait=True, cancel_futures=False)

        # Stop all managed threads (scheduling, heartbeat, server)
        self._threads.stop(timeout=timeout)

        # Shutdown autoscaler
        if self._autoscaler:
            self._autoscaler.shutdown()

    def _run_scheduling_loop(self, stop_event: threading.Event) -> None:
        """Main controller loop running scheduling, autoscaler, and worker timeout checks."""
        while not stop_event.is_set():
            self._wake_event.wait(timeout=self._config.scheduler_interval_seconds)
            self._wake_event.clear()

            if stop_event.is_set():
                break

            self._run_scheduling()
            self._check_worker_timeouts()

            if self._autoscaler:
                self._run_autoscaler_once()

    def _run_heartbeat_loop(self, stop_event: threading.Event) -> None:
        """Heartbeat loop running on its own thread so slow RPCs don't block scheduling."""
        while not stop_event.is_set():
            self._heartbeat_event.wait(timeout=self._config.scheduler_interval_seconds)
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
        pending_tasks = self._state.peek_pending_tasks()
        workers = self._state.get_available_workers()

        if not pending_tasks:
            return

        result = self._scheduler.find_assignments(pending_tasks, workers)

        # Buffer assignments for heartbeat delivery (commits resources via TaskAssignedEvent)
        if result.assignments:
            self._buffer_assignments(result.assignments)

        # Handle timed-out tasks
        for task in result.timed_out_tasks:
            self._mark_task_unschedulable(task)

    def _buffer_assignments(
        self,
        assignments: list[tuple[ControllerTask, ControllerWorker]],
    ) -> None:
        """Commit resources and buffer task assignments for heartbeat delivery.

        Groups assignments by job and buffers RunTaskRequest protos into per-worker outboxes.
        Resource commitment happens via TaskAssignedEvent.
        """
        # Group assignments by job for coscheduled handling
        by_job: dict[JobId, list[tuple[ControllerTask, ControllerWorker]]] = defaultdict(list)
        for task, worker in assignments:
            by_job[task.job_id].append((task, worker))

        for job_id, job_assignments in by_job.items():
            job = self._state.get_job(job_id)
            if job is None:
                continue

            for task, worker in job_assignments:
                # Commit resources and buffer dispatch atomically under _dispatch_lock.
                # This prevents the heartbeat thread from seeing the task in
                # worker.running_tasks (via TaskAssignedEvent) before the
                # RunTaskRequest is in the outbox.
                with self._dispatch_lock:
                    self._state.handle_event(
                        TaskAssignedEvent(
                            task_id=task.task_id,
                            worker_id=worker.worker_id,
                        )
                    )
                    request = cluster_pb2.Worker.RunTaskRequest(
                        job_id=str(task.job_id),
                        task_id=str(task.task_id),
                        task_index=task.task_index,
                        num_tasks=len(self._state.get_job_tasks(task.job_id)),
                        entrypoint=job.request.entrypoint,
                        environment=job.request.environment,
                        bundle_gcs_path=job.request.bundle_gcs_path,
                        resources=job.request.resources,
                        ports=list(job.request.ports),
                        attempt_id=task.current_attempt_id,
                    )
                    # Copy timeout if set (check milliseconds field > 0)
                    if job.request.timeout.milliseconds > 0:
                        request.timeout.CopyFrom(job.request.timeout)
                    wd = self._dispatch.setdefault(worker.worker_id, WorkerDispatch())
                    wd.dispatch_outbox.append(request)

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

    def task_schedule_status(self, task: ControllerTask, context: SchedulingContext) -> TaskScheduleResult:
        """Get the current scheduling status of a task (for dashboard display).

        Delegates to the internal scheduler.
        """
        return self._scheduler.task_schedule_status(task, context)

    def kill_tasks_on_workers(self, task_ids: set[TaskId]) -> None:
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
            with self._dispatch_lock:
                wd = self._dispatch.setdefault(worker.worker_id, WorkerDispatch())
                wd.kill_outbox.append(str(task_id))
                any_buffered = True

        # Wake heartbeat thread to deliver buffered kills immediately
        if any_buffered:
            self._heartbeat_event.set()

    def _heartbeat_all_workers(self) -> None:
        """Send heartbeats to all registered workers.

        Uses a lock-copy-unlock-send-lock-apply pattern: snapshots dispatch outboxes
        under lock, sends RPCs without holding the lock, then re-locks to apply results.
        """
        # Exit early if shutdown is in progress
        if self._stop:
            return

        # Phase 1: snapshot worker list (no lock), then drain outboxes (under lock)
        all_workers = self._state.get_available_workers()
        with self._dispatch_lock:
            worker_dispatches: dict[WorkerId, tuple[ControllerWorker, list, list]] = {}
            for w in all_workers:
                wd = self._dispatch.pop(w.worker_id, WorkerDispatch())
                worker_dispatches[w.worker_id] = (w, wd.dispatch_outbox, wd.kill_outbox)

        # Phase 2: send RPCs (no lock held)
        futures = {}
        for _worker_id, (worker, tasks_to_run, tasks_to_kill) in worker_dispatches.items():
            future = self._dispatch_executor.submit(self._heartbeat_worker, worker, tasks_to_run, tasks_to_kill)
            futures[future] = (worker, tasks_to_run, tasks_to_kill)

        # Phase 3: process results, requeue failures
        try:
            for future in as_completed(futures, timeout=10):
                # Exit if shutdown started while waiting for futures
                if self._stop:
                    return
                worker, tasks_to_run, tasks_to_kill = futures.pop(future)
                self._handle_heartbeat_future(future, worker, tasks_to_run, tasks_to_kill)
        except TimeoutError:
            # Process any futures that completed before timeout
            for future, (worker, tasks_to_run, tasks_to_kill) in futures.items():
                if future.done():
                    self._handle_heartbeat_future(future, worker, tasks_to_run, tasks_to_kill)
                else:
                    logger.warning(f"Heartbeat timed out for {worker.worker_id}")
                    self._requeue_and_fail(worker, tasks_to_run, tasks_to_kill)
                    future.cancel()

    def _handle_heartbeat_future(
        self,
        future: Future,
        worker: ControllerWorker,
        tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest],
        tasks_to_kill: list[str],
    ) -> None:
        """Process a completed heartbeat future."""
        try:
            response = future.result()
            self._process_heartbeat_response(worker, response)
        except Exception as e:
            logger.warning(f"Heartbeat error for {worker.worker_id}: {e}")
            self._requeue_and_fail(worker, tasks_to_run, tasks_to_kill)

    def _heartbeat_worker(
        self,
        worker: ControllerWorker,
        tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest],
        tasks_to_kill: list[str],
    ) -> cluster_pb2.HeartbeatResponse:
        """Send a heartbeat to a single worker.

        Raises:
            Exception on RPC failure (caught and handled by _handle_heartbeat_future)
        """
        if rule := chaos("controller.heartbeat"):
            sleep(rule.delay_seconds)
            raise Exception("chaos: heartbeat unavailable")
        stub = self._stub_factory.get_stub(worker.address)
        expected_tasks = [
            cluster_pb2.Controller.RunningTaskEntry(
                task_id=str(tid),
                attempt_id=self._state.get_task(tid).current_attempt_id if self._state.get_task(tid) else 0,
            )
            for tid in worker.running_tasks
        ]
        request = cluster_pb2.HeartbeatRequest(
            tasks_to_run=tasks_to_run,
            tasks_to_kill=tasks_to_kill,
            expected_tasks=expected_tasks,
        )
        return stub.heartbeat(request)

    def _process_heartbeat_response(
        self,
        worker: ControllerWorker,
        response: cluster_pb2.HeartbeatResponse,
    ) -> None:
        """Process a successful heartbeat response.

        Updates heartbeat timestamp and processes completed tasks. Reconciliation
        of running vs expected tasks is handled worker-side using expected_tasks.
        """
        self._state.handle_event(WorkerHeartbeatEvent(worker_id=worker.worker_id, timestamp=Timestamp.now()))

        for entry in response.completed_tasks:
            task_id = TaskId(entry.task_id)
            task = self._state.get_task(task_id)
            if task and not task.is_finished():
                self._state.handle_event(
                    TaskStateChangedEvent(
                        task_id=task_id,
                        new_state=entry.state,
                        error=entry.error or None,
                        exit_code=entry.exit_code,
                        attempt_id=entry.attempt_id,
                    )
                )

    def _requeue_and_fail(
        self,
        worker: ControllerWorker,
        tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest],
        tasks_to_kill: list[str],
    ) -> None:
        """Re-queue outbox entries and record a heartbeat failure for a worker."""
        with self._dispatch_lock:
            wd = self._dispatch.setdefault(worker.worker_id, WorkerDispatch())
            if tasks_to_run:
                wd.dispatch_outbox.extend(tasks_to_run)
            if tasks_to_kill:
                wd.kill_outbox.extend(tasks_to_kill)
        self._handle_heartbeat_failure(worker)

    def _handle_heartbeat_failure(self, worker: ControllerWorker) -> None:
        """Handle a failed heartbeat RPC via the state layer."""
        self._state.handle_event(
            WorkerHeartbeatFailedEvent(
                worker_id=worker.worker_id,
                error=f"Heartbeat failed for worker {worker.worker_id}",
            )
        )

    def _check_worker_timeouts(self) -> None:
        """Check for worker timeouts and send kill RPCs for affected tasks."""
        # State computes failed workers and marks them atomically under lock
        tasks_to_kill = self._state.check_worker_timeouts(self._config.worker_timeout)

        # Send kill RPCs outside lock
        if tasks_to_kill:
            self.kill_tasks_on_workers(tasks_to_kill)

    def _run_autoscaler_once(self) -> None:
        """Run one autoscaler evaluation cycle.

        Called from the scheduling loop every cycle. Computes demand from pending
        tasks and worker idle state, then runs the autoscaler.

        Rate-limits evaluations based on the autoscaler's configured evaluation_interval_seconds.
        """
        if not self._autoscaler:
            return

        from time import monotonic

        now = monotonic()
        interval = self._autoscaler.evaluation_interval_seconds
        if interval > 0 and now - self._last_autoscaler_run < interval:
            return
        self._last_autoscaler_run = now

        demand_entries = compute_demand_entries(self._state)
        vm_status_map = self._build_vm_status_map()
        self._autoscaler.run_once(demand_entries, vm_status_map=vm_status_map)

    def _build_vm_status_map(self) -> VmWorkerStatusMap:
        """Build a map of VM address to worker status for autoscaler.

        The autoscaler needs to look up worker status by VM address (not worker_id)
        because ManagedVm only knows the VM's IP address, not the worker's self-assigned ID.
        Workers include their vm_address (from IRIS_VM_ADDRESS env var) in metadata.
        """
        result: VmWorkerStatusMap = {}
        for worker in self._state.list_all_workers():
            vm_addr = worker.metadata.vm_address
            if not vm_addr:
                raise ValueError(
                    f"Worker {worker.worker_id} has no vm_address in metadata. "
                    "Workers must report IRIS_VM_ADDRESS in their metadata."
                )

            result[vm_addr] = VmWorkerStatus(
                vm_address=vm_addr,
                running_task_ids=frozenset(str(tid) for tid in worker.running_tasks),
            )
        return result

    def _run_server(self, stop_event: threading.Event) -> None:
        try:
            config = uvicorn.Config(
                self._dashboard._app,
                host=self._config.host,
                port=self._config.port,
                log_level="error",
            )
            self._server = uvicorn.Server(config)

            # Bridge stop_event to server shutdown
            def _watch_stop() -> None:
                stop_event.wait()
                if self._server:
                    self._server.should_exit = True

            watch_thread = threading.Thread(target=_watch_stop, daemon=False, name="server-stop-bridge")
            watch_thread.start()

            try:
                self._server.run()
            finally:
                # Ensure bridge thread exits (should be immediate since stop_event was set)
                watch_thread.join(timeout=1.0)
        except Exception as e:
            logger.exception("Controller server error: %s", e)

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
        if self._server and self._server.servers:
            # Get actual port from the first server socket
            sockets = self._server.servers[0].sockets
            if sockets:
                return sockets[0].getsockname()[1]
        return self._config.port

    @property
    def url(self) -> str:
        return f"http://{self._config.host}:{self.port}"

    @property
    def autoscaler(self) -> "Autoscaler | None":
        """The autoscaler instance, if autoscaling is enabled."""
        return self._autoscaler
