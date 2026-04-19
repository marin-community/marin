# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-RPC synthetic workers for the fleet-wide scenario.

Each successful `LoadtestGcpService.tpu_create` spawns a `SyntheticWorker`.
A synthetic worker is a *real* Connect/RPC server: it binds a free localhost
port and mounts `WorkerServiceWSGIApplication` the same way the production
worker does (see `iris.cluster.worker.dashboard`). It then registers with
the controller DB via `ControllerTransitions.register_worker` using its real
`host:port`, so any controller code that holds a `WorkerServiceClientSync`
against that address will hit the worker over a real socket.

Why real sockets: Stage-3 showed the harness plateaus at ~24 concurrent
scale-up threads while prod saw ~96. The suspected prod CPU channel is the
controller→worker RPC path (connection pooling, retries, timeouts); an
in-process shortcut leaves that channel uninstrumented. With a real server
on a real port, the harness's prober thread (see `controller_prober` in
`harness.py`) exercises that same channel — closing the gap.

Task lifecycle: the worker drives each assigned task through
`ASSIGNED → BUILDING → RUNNING → COMPLETED` on a background timer. Per-
transition delays are configurable via `HarnessConfig` so tests can compress
the lifecycle to <1 s while production-style runs keep the prod-realistic
~60 s per transition.

Preemption: when `tpu_delete` fires, `stop(abrupt=True)` sets
`server.should_exit` *and* closes the underlying socket immediately. The
controller's next RPC to this worker will fail with a connection error —
that's the observed failure path in prod. Graceful deregistration is
explicitly not modelled (prod preemption is ungraceful).
"""

from __future__ import annotations

import logging
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import Callable

import uvicorn
from connectrpc.request import RequestContext
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.routing import Mount

from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.types import WorkerId, get_tpu_topology
from iris.rpc import job_pb2, worker_pb2
from iris.rpc.worker_connect import WorkerServiceWSGIApplication
from rigging.timing import Timer, Timestamp

logger = logging.getLogger(__name__)


# Prod cadence (see ControllerConfig.heartbeat_interval). Kept manual to avoid
# a cross-module import for a constant.
HEARTBEAT_INTERVAL_SECONDS = 5.0

# Prod-realistic per-state delays. Compressible via SyntheticWorkerConfig.
DEFAULT_BUILDING_SECONDS = 60.0
DEFAULT_RUNNING_SECONDS = 60.0


@dataclass
class SyntheticTask:
    """Mutable task state driven by the lifecycle thread."""

    task_id: str
    attempt_id: int
    state: int  # job_pb2.TaskState
    state_started_at: float = field(default_factory=time.monotonic)


@dataclass
class SyntheticWorkerConfig:
    """Identity + sizing for one synthetic worker.

    ``tpu_worker_id`` identifies the worker's position within a multi-host TPU
    slice — all workers of the same slice share ``slice_id`` (acting as
    ``tpu-name``) but carry distinct ids 0..N-1. The scheduler co-schedules a
    multi-host job by finding ``num_tasks`` workers with the same ``tpu-name``.
    """

    worker_id: WorkerId
    slice_id: str
    scale_group: str
    zone: str
    device_variant: str
    tpu_worker_id: int = 0
    host: str = "127.0.0.1"


@dataclass
class LifecycleDelays:
    """Per-transition delays. Compressed for fast tests."""

    building_seconds: float = DEFAULT_BUILDING_SECONDS
    running_seconds: float = DEFAULT_RUNNING_SECONDS


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Bind-and-release to discover an ephemeral port.

    There's an inherent TOCTOU race between releasing the port here and
    rebinding it inside uvicorn. In a loopback-only test harness spawning
    100s of workers back-to-back the collision risk is low enough to be
    empirically negligible; if it ever fires, the uvicorn startup will raise
    and the caller retries.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class _SyntheticTaskProvider:
    """Minimal `WorkerService.TaskProvider` for a synthetic worker.

    Only the RPCs the controller actually dispatches in production are
    implemented meaningfully; the rest raise or return empty results. The
    goal is to satisfy the real RPC surface, not to emulate container
    execution.
    """

    def __init__(self, worker_id: WorkerId, delays: LifecycleDelays) -> None:
        self._worker_id = worker_id
        self._delays = delays
        self._tasks: dict[str, SyntheticTask] = {}
        self._lock = threading.Lock()
        self._timer = Timer()

    # ---- controller-initiated RPCs ----------------------------------------

    def ping(self, _req: worker_pb2.Worker.PingRequest) -> worker_pb2.Worker.PingResponse:
        resp = worker_pb2.Worker.PingResponse(healthy=True)
        return resp

    def heartbeat(self, request: job_pb2.HeartbeatRequest) -> job_pb2.HeartbeatResponse:
        """Apply controller-driven reconciliation.

        Controller heartbeats carry `tasks_to_run` and `tasks_to_kill`; we
        record the former into our in-memory state machine and drop the
        latter. The response echoes our current task state.
        """
        now = time.monotonic()
        with self._lock:
            for run in request.tasks_to_run:
                task = self._tasks.get(run.task_id)
                if task is None:
                    self._tasks[run.task_id] = SyntheticTask(
                        task_id=run.task_id,
                        attempt_id=run.attempt_id,
                        state=job_pb2.TASK_STATE_ASSIGNED,
                        state_started_at=now,
                    )
            for kill_id in request.tasks_to_kill:
                task = self._tasks.get(kill_id)
                if task is not None:
                    task.state = job_pb2.TASK_STATE_KILLED
                    task.state_started_at = now
            task_states = [
                job_pb2.WorkerTaskStatus(task_id=t.task_id, attempt_id=t.attempt_id, state=t.state)
                for t in self._tasks.values()
            ]
        resp = job_pb2.HeartbeatResponse(worker_healthy=True)
        resp.tasks.extend(task_states)
        return resp

    def poll_tasks(self, request: worker_pb2.Worker.PollTasksRequest) -> worker_pb2.Worker.PollTasksResponse:
        with self._lock:
            statuses = [
                job_pb2.WorkerTaskStatus(task_id=t.task_id, attempt_id=t.attempt_id, state=t.state)
                for t in self._tasks.values()
            ]
        return worker_pb2.Worker.PollTasksResponse(tasks=statuses)

    def health_check(self, _req: job_pb2.Empty) -> worker_pb2.Worker.HealthResponse:
        with self._lock:
            running = sum(1 for t in self._tasks.values() if t.state == job_pb2.TASK_STATE_RUNNING)
        resp = worker_pb2.Worker.HealthResponse(healthy=True, running_tasks=running)
        resp.uptime.milliseconds = self._timer.elapsed_ms()
        return resp

    def list_tasks(self, _req: worker_pb2.Worker.ListTasksRequest) -> worker_pb2.Worker.ListTasksResponse:
        with self._lock:
            tasks = [_task_status(t) for t in self._tasks.values()]
        return worker_pb2.Worker.ListTasksResponse(tasks=tasks)

    def get_task_status(
        self, request: worker_pb2.Worker.GetTaskStatusRequest, _ctx: RequestContext | None = None
    ) -> job_pb2.TaskStatus:
        with self._lock:
            task = self._tasks.get(request.task_id)
        if task is None:
            return job_pb2.TaskStatus(task_id=request.task_id, state=job_pb2.TASK_STATE_FAILED)
        return _task_status(task)

    def start_tasks(self, request: worker_pb2.Worker.StartTasksRequest) -> worker_pb2.Worker.StartTasksResponse:
        now = time.monotonic()
        acks: list[worker_pb2.Worker.TaskAck] = []
        with self._lock:
            for run in request.tasks:
                self._tasks[run.task_id] = SyntheticTask(
                    task_id=run.task_id,
                    attempt_id=run.attempt_id,
                    state=job_pb2.TASK_STATE_ASSIGNED,
                    state_started_at=now,
                )
                acks.append(worker_pb2.Worker.TaskAck(task_id=run.task_id, accepted=True))
        return worker_pb2.Worker.StartTasksResponse(acks=acks)

    def stop_tasks(self, request: worker_pb2.Worker.StopTasksRequest) -> worker_pb2.Worker.StopTasksResponse:
        with self._lock:
            for task_id in request.task_ids:
                task = self._tasks.get(task_id)
                if task is not None:
                    task.state = job_pb2.TASK_STATE_KILLED
        return worker_pb2.Worker.StopTasksResponse()

    # ---- internal progression ---------------------------------------------

    def advance(self, delays: LifecycleDelays) -> None:
        """Advance all non-terminal tasks one transition if their dwell
        time has elapsed."""
        now = time.monotonic()
        with self._lock:
            for task in self._tasks.values():
                elapsed = now - task.state_started_at
                if task.state == job_pb2.TASK_STATE_ASSIGNED:
                    # ASSIGNED -> BUILDING is immediate (matches prod).
                    task.state = job_pb2.TASK_STATE_BUILDING
                    task.state_started_at = now
                elif task.state == job_pb2.TASK_STATE_BUILDING and elapsed >= delays.building_seconds:
                    task.state = job_pb2.TASK_STATE_RUNNING
                    task.state_started_at = now
                elif task.state == job_pb2.TASK_STATE_RUNNING and elapsed >= delays.running_seconds:
                    task.state = job_pb2.TASK_STATE_SUCCEEDED
                    task.state_started_at = now


def _task_status(task: SyntheticTask) -> job_pb2.TaskStatus:
    return job_pb2.TaskStatus(task_id=task.task_id, state=task.state, attempt_id=task.attempt_id)


def _build_metadata(config: SyntheticWorkerConfig) -> job_pb2.WorkerMetadata:
    metadata = job_pb2.WorkerMetadata()
    metadata.hostname = f"synth-{config.worker_id}"
    metadata.ip_address = config.host
    # Match a TPU VM host: 200 vCPU / 600 GiB / ~2 TiB disk. TPU jobs request
    # 8 vCPU + 32 GiB each; CPU jobs pack into the leftover capacity on the
    # same host (prod has no dedicated CPU fleet).
    metadata.cpu_count = 200
    metadata.memory_bytes = 600 * 1024**3
    metadata.disk_bytes = 2 * 1024**4
    metadata.gce_instance_name = f"synth-{config.slice_id}"
    metadata.gce_zone = config.zone
    metadata.tpu_name = config.slice_id
    metadata.tpu_worker_id = str(config.tpu_worker_id)
    metadata.device.tpu.variant = config.device_variant
    metadata.device.tpu.count = 1

    # Scheduler matches on metadata.attributes, not the device_* columns. Mirror
    # what iris.cluster.worker.env_probe.build_worker_attributes writes on a
    # real worker so constraint-based scheduling can route to us.
    region = config.zone.rsplit("-", 1)[0]
    preemptible = "preemptible" in config.scale_group
    attrs = metadata.attributes
    attrs[WellKnownAttribute.DEVICE_TYPE].string_value = "tpu"
    attrs[WellKnownAttribute.DEVICE_VARIANT].string_value = config.device_variant.lower()
    attrs[WellKnownAttribute.PREEMPTIBLE].string_value = str(preemptible).lower()
    attrs[WellKnownAttribute.ZONE].string_value = config.zone
    attrs[WellKnownAttribute.REGION].string_value = region
    attrs[WellKnownAttribute.TPU_NAME].string_value = config.slice_id
    attrs[WellKnownAttribute.TPU_WORKER_ID].int_value = config.tpu_worker_id
    attrs[WellKnownAttribute.TPU_TOPOLOGY].string_value = config.device_variant
    try:
        topo = get_tpu_topology(config.device_variant)
        attrs[WellKnownAttribute.TPU_VM_COUNT].int_value = topo.vm_count
    except ValueError:
        pass
    return metadata


class _WorkerServiceAdapter:
    """Adapter exposing `WorkerServiceImpl`-compatible method signatures.

    The generated `WorkerServiceWSGIApplication` calls methods with
    `(request, ctx)` — we wrap `_SyntheticTaskProvider`'s single-arg methods
    accordingly so the provider keeps a clean domain API.
    """

    def __init__(self, provider: _SyntheticTaskProvider) -> None:
        self._provider = provider

    def get_task_status(self, request, _ctx):
        return self._provider.get_task_status(request)

    def list_tasks(self, request, _ctx):
        return self._provider.list_tasks(request)

    def health_check(self, request, _ctx):
        return self._provider.health_check(request)

    def heartbeat(self, request, _ctx):
        return self._provider.heartbeat(request)

    def profile_task(self, _request, _ctx):
        return job_pb2.ProfileTaskResponse(error="not supported on synthetic worker")

    def get_process_status(self, _request, _ctx):
        return job_pb2.GetProcessStatusResponse()

    def exec_in_container(self, _request, _ctx):
        return worker_pb2.Worker.ExecInContainerResponse(error="not supported on synthetic worker")

    def ping(self, request, _ctx):
        return self._provider.ping(request)

    def start_tasks(self, request, _ctx):
        return self._provider.start_tasks(request)

    def stop_tasks(self, request, _ctx):
        return self._provider.stop_tasks(request)

    def poll_tasks(self, request, _ctx):
        return self._provider.poll_tasks(request)


class SyntheticWorker:
    """Real Connect/RPC server on a bound localhost port."""

    def __init__(
        self,
        config: SyntheticWorkerConfig,
        *,
        transitions: ControllerTransitions,
        db: ControllerDB,
        delays: LifecycleDelays,
    ) -> None:
        self._config = config
        self._transitions = transitions
        self._db = db
        self._delays = delays
        self._provider = _SyntheticTaskProvider(config.worker_id, delays)
        self._adapter = _WorkerServiceAdapter(self._provider)

        self._port = _find_free_port(config.host)
        self._address = f"{config.host}:{self._port}"

        rpc_wsgi_app = WorkerServiceWSGIApplication(service=self._adapter)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)
        self._app = Starlette(routes=[Mount(rpc_wsgi_app.path, app=rpc_app)])

        self._server = uvicorn.Server(
            uvicorn.Config(
                self._app,
                host=config.host,
                port=self._port,
                log_level="error",
                log_config=None,
                timeout_keep_alive=30,
                # A single loop/thread per worker keeps per-worker overhead low.
                # Connect-WSGI runs synchronously inside the middleware anyway.
                loop="asyncio",
                # We don't use startup/shutdown events; leaving lifespan on
                # means every worker teardown logs a CancelledError traceback
                # from starlette's receive loop.
                lifespan="off",
            )
        )
        self._server_thread: threading.Thread | None = None
        self._lifecycle_thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def address(self) -> str:
        return self._address

    @property
    def worker_id(self) -> WorkerId:
        return self._config.worker_id

    def start(self) -> None:
        self._server_thread = threading.Thread(
            target=self._server.run, name=f"synth-rpc-{self._config.worker_id}", daemon=True
        )
        self._server_thread.start()
        # Wait briefly for uvicorn to accept. uvicorn sets `started` after the
        # socket is bound; 500 ms has been plenty even with many workers.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not getattr(self._server, "started", False):
            time.sleep(0.01)
        self._register()
        self._lifecycle_thread = threading.Thread(
            target=self._run_lifecycle, name=f"synth-life-{self._config.worker_id}", daemon=True
        )
        self._lifecycle_thread.start()

    def _register(self) -> None:
        metadata = _build_metadata(self._config)
        self._transitions.register_worker(
            worker_id=self._config.worker_id,
            address=self._address,
            metadata=metadata,
            ts=Timestamp.now(),
            slice_id=self._config.slice_id,
            scale_group=self._config.scale_group,
        )

    def _run_lifecycle(self) -> None:
        """Advance task states on a fixed cadence.

        Cadence is the min of `building_seconds` and `running_seconds`
        divided by 4 so each transition fires promptly in fast tests but
        doesn't burn CPU at full-scale delays.
        """
        cadence = max(0.1, min(self._delays.building_seconds, self._delays.running_seconds) / 4.0)
        while not self._stop.is_set():
            try:
                self._provider.advance(self._delays)
            except Exception:
                logger.exception("synthetic worker %s lifecycle advance failed", self._config.worker_id)
            if self._stop.wait(cadence):
                return

    def signal_stop(self, *, abrupt: bool = True) -> None:
        """Non-blocking: tell the worker to exit. Threads are daemons so a
        subsequent join is optional — callers that need deterministic teardown
        can call :meth:`join` afterwards (ideally in parallel across workers).
        """
        self._stop.set()
        if self._server is not None:
            self._server.should_exit = True
            if abrupt:
                self._server.force_exit = True

    def join(self, *, timeout: float = 2.0) -> None:
        if self._server_thread is not None:
            self._server_thread.join(timeout=timeout)
            self._server_thread = None
        if self._lifecycle_thread is not None:
            self._lifecycle_thread.join(timeout=timeout)
            self._lifecycle_thread = None

    def stop(self, *, abrupt: bool = True, timeout: float = 2.0) -> None:
        """Signal + join sequentially. Prefer ``signal_stop`` + parallel
        ``join`` when shutting down many workers at once; this combined form is
        kept for single-worker preemption where serial cost is negligible.
        """
        self.signal_stop(abrupt=abrupt)
        self.join(timeout=timeout)


class SyntheticWorkerPool:
    """Lifecycle manager for per-slice synthetic workers."""

    def __init__(
        self,
        transitions: ControllerTransitions,
        db: ControllerDB,
        *,
        delays: LifecycleDelays | None = None,
    ) -> None:
        self._transitions = transitions
        self._db = db
        self._delays = delays or LifecycleDelays()
        self._workers: dict[str, list[SyntheticWorker]] = {}
        self._lock = threading.Lock()

    def spawn_for_slice(
        self,
        *,
        slice_id: str,
        scale_group: str,
        zone: str,
        device_variant: str,
    ) -> list[SyntheticWorker]:
        """Spawn one synthetic worker per VM in the slice.

        Multi-host TPU jobs carry ``replicas = topology.vm_count`` and the
        scheduler co-schedules them by grouping workers on ``tpu-name``. We
        create ``vm_count`` workers sharing ``slice_id`` (→ ``tpu-name``) with
        distinct ``tpu_worker_id`` values 0..N-1, matching a real multi-host
        slice.
        """
        try:
            vm_count = get_tpu_topology(device_variant).vm_count
        except ValueError:
            vm_count = 1
        workers: list[SyntheticWorker] = []
        for worker_index in range(vm_count):
            config = SyntheticWorkerConfig(
                worker_id=WorkerId(f"synth-{uuid.uuid4().hex[:12]}"),
                slice_id=slice_id,
                scale_group=scale_group,
                zone=zone,
                device_variant=device_variant,
                tpu_worker_id=worker_index,
            )
            worker = SyntheticWorker(
                config,
                transitions=self._transitions,
                db=self._db,
                delays=self._delays,
            )
            worker.start()
            workers.append(worker)
        with self._lock:
            self._workers[slice_id] = workers
        return workers

    def _register_external(self, worker: SyntheticWorker) -> None:
        """Track a worker spawned outside of ``spawn_for_slice``.

        Used by the pre-load path: each pre-loaded worker is its own synthetic
        1-VM slice, so we append under the worker's ``slice_id`` list.
        """
        with self._lock:
            self._workers.setdefault(worker._config.slice_id, []).append(worker)

    def stop_for_slice(self, slice_id: str, *, timeout: float = 2.0) -> None:
        with self._lock:
            workers = self._workers.pop(slice_id, [])
        for worker in workers:
            worker.signal_stop(abrupt=True)
        deadline = time.monotonic() + timeout
        for worker in workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

    def stop_all(self, *, timeout: float = 2.0) -> None:
        """Signal every worker to exit, then wait up to ``timeout`` seconds
        *total* for them to drain — not per-worker. Lifecycle + uvicorn
        threads are daemons, so any that overshoot the deadline are left for
        process exit to reap. This turns a 600-worker teardown from ~20 min
        (worst case, serial 2 s joins) into a bounded one-shot wait.
        """
        with self._lock:
            workers = [w for slice_workers in self._workers.values() for w in slice_workers]
            self._workers.clear()
        for worker in workers:
            worker.signal_stop(abrupt=True)
        deadline = time.monotonic() + timeout
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            worker.join(timeout=remaining)

    def active_count(self) -> int:
        with self._lock:
            return sum(len(ws) for ws in self._workers.values())

    def addresses(self) -> list[tuple[WorkerId, str]]:
        """Snapshot of (worker_id, address) pairs across every slice — used by the prober."""
        with self._lock:
            return [(w.worker_id, w.address) for ws in self._workers.values() for w in ws]


def run_controller_prober(
    get_addresses: Callable[[], list[tuple[WorkerId, str]]],
    *,
    stop: threading.Event,
    interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
    timeout_ms: int = 2000,
) -> None:
    """Controller-side ping loop over real sockets.

    The Stage-6 harness doesn't boot the full Controller (only Autoscaler +
    DB). A real Controller would hit each registered worker's `Ping` RPC on
    a heartbeat loop via `WorkerServiceClientSync`. To exercise the same
    channel from the harness, this helper polls every known worker address
    at `interval_seconds`. Workers whose sockets have been torn down
    (preemption) raise ConnectError — which is exactly the failure we want
    to observe.

    Intentionally sequential per-tick: we're instrumenting the protocol, not
    maximising throughput. A thread pool would hide connection-pooling
    artefacts.
    """
    # Local import so the module is importable without connectrpc at tool time.
    from iris.rpc.worker_connect import WorkerServiceClientSync

    clients: dict[str, WorkerServiceClientSync] = {}
    while not stop.is_set():
        for worker_id, address in get_addresses():
            client = clients.get(address)
            if client is None:
                client = WorkerServiceClientSync(address=f"http://{address}", timeout_ms=timeout_ms)
                clients[address] = client
            try:
                client.ping(worker_pb2.Worker.PingRequest())
            except Exception:
                # Preempted workers are expected to fail here; log at DEBUG.
                logger.debug("prober ping to %s (%s) failed", worker_id, address, exc_info=True)
                clients.pop(address, None)
        if stop.wait(interval_seconds):
            return
