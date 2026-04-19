# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-RPC synthetic workers for the fleet-wide scenario.

Each successful `LoadtestGcpService.tpu_create` spawns a `SyntheticWorker`.
A synthetic worker is a *real* Connect/RPC server running in its own OS
subprocess — it binds a free localhost port and mounts
`WorkerServiceWSGIApplication` the same way the production worker does (see
`iris.cluster.worker.dashboard`). The parent registers the worker with the
controller DB via `ControllerTransitions.register_worker` using the child's
real `host:port`, so any controller code that holds a
`WorkerServiceClientSync` against that address will hit the worker over a
real socket.

Why an OS subprocess (not a thread): each worker's uvicorn event loop used
to share the harness interpreter's GIL with the Controller. That invalidates
ablation measurements of controller concurrency — the controller and its
"fleet" compete for a single interpreter. Splitting each worker into its own
process gives real parallel CPU isolation, matching prod where every worker
is on a separate VM.

Why real sockets: the suspected prod CPU channel is the controller→worker RPC
path (connection pooling, retries, timeouts); an in-process shortcut leaves
that channel uninstrumented. With a real server on a real port, the
Controller's own ping loop exercises that channel — closing the gap.

Task lifecycle: the child drives each assigned task through
`ASSIGNED → BUILDING → RUNNING → COMPLETED` on a background timer inside
the subprocess. Per-transition delays are configurable via `HarnessConfig`
so tests can compress the lifecycle to <1 s while production-style runs
keep the prod-realistic ~60 s per transition.

Preemption: ``signal_stop(abrupt=True)`` sends SIGKILL to the child's
process group; a subsequent ``join`` reaps it. The controller's next RPC
to the dead worker will fail with a connection error — that's the
observed failure path in prod. Graceful deregistration is explicitly not
modelled (prod preemption is ungraceful).
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from connectrpc.request import RequestContext

from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.types import WorkerId, get_tpu_topology
from iris.rpc import job_pb2, worker_pb2
from rigging.timing import Timer, Timestamp

logger = logging.getLogger(__name__)


# Prod cadence (see ControllerConfig.heartbeat_interval). Kept manual to avoid
# a cross-module import for a constant.
HEARTBEAT_INTERVAL_SECONDS = 5.0

# Prod-realistic per-state delays. Compressible via SyntheticWorkerConfig.
DEFAULT_BUILDING_SECONDS = 60.0
DEFAULT_RUNNING_SECONDS = 60.0

# Seconds to wait for a freshly-spawned child to emit its READY line.
# 10 s is conservative — empirically ~300-500 ms cold for a fresh Python
# process. Bumped on slower CI hardware.
READY_TIMEOUT_SECONDS = 10.0

# Parallelism for subprocess fan-out when starting many workers at once
# (preload / autoscaler burst). Bounded to keep fork storms away from the
# FD limit and syscall throughput.
STARTUP_PARALLELISM = 16

_READY_RE = re.compile(r"^READY port=(\d+)\s*$")


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
    """Real Connect/RPC server running in its own OS subprocess."""

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

        self._process: subprocess.Popen | None = None
        self._address: str | None = None
        self._reader_thread: threading.Thread | None = None

    @property
    def address(self) -> str:
        assert self._address is not None, "start() must be called before reading address"
        return self._address

    @property
    def worker_id(self) -> WorkerId:
        return self._config.worker_id

    # ---- startup ----------------------------------------------------------

    def start(self) -> None:
        """Spawn the child, block until READY, then register with the controller.

        ``sys.executable -m iris.loadtest.synthetic_worker_main`` is used so
        the child inherits the current venv without requiring ``uv run``
        re-entry. ``start_new_session=True`` puts the child in its own
        process group — ``os.killpg`` then cleanly reaps uvicorn workers /
        any asyncio helpers the event loop may spawn.
        """
        cmd = [
            sys.executable,
            "-m",
            "iris.loadtest.synthetic_worker_main",
            "--worker-id",
            str(self._config.worker_id),
            "--slice-id",
            self._config.slice_id,
            "--scale-group",
            self._config.scale_group,
            "--zone",
            self._config.zone,
            "--device-variant",
            self._config.device_variant,
            "--tpu-worker-id",
            str(self._config.tpu_worker_id),
            "--building-seconds",
            str(self._delays.building_seconds),
            "--running-seconds",
            str(self._delays.running_seconds),
            "--host",
            self._config.host,
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            bufsize=1,  # line-buffered so READY arrives promptly
        )
        port = self._await_ready()
        self._address = f"{self._config.host}:{port}"

        # Drain remaining child stdout to DEBUG. Daemon so it never blocks
        # parent exit; on EOF (child has exited) the thread returns.
        self._reader_thread = threading.Thread(
            target=self._drain_output,
            name=f"synth-log-{self._config.worker_id}",
            daemon=True,
        )
        self._reader_thread.start()

        self._register()

    def _await_ready(self) -> int:
        assert self._process is not None
        assert self._process.stdout is not None
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_unconditionally()
                raise RuntimeError(
                    f"synthetic worker {self._config.worker_id} did not emit READY "
                    f"within {READY_TIMEOUT_SECONDS:.1f}s"
                )
            line = self._process.stdout.readline()
            if line == "":
                # EOF — child died before ready.
                rc = self._process.poll()
                raise RuntimeError(f"synthetic worker {self._config.worker_id} exited (rc={rc}) before READY")
            match = _READY_RE.match(line)
            if match:
                return int(match.group(1))
            # Non-READY pre-ready log lines go to DEBUG.
            logger.debug("[worker %s pre-ready] %s", self._config.worker_id, line.rstrip())

    def _drain_output(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        try:
            for line in self._process.stdout:
                logger.debug("[worker %s] %s", self._config.worker_id, line.rstrip())
        except Exception:
            logger.exception("reader thread for worker %s failed", self._config.worker_id)

    def _register(self) -> None:
        assert self._address is not None
        metadata = _build_metadata(self._config)
        self._transitions.register_worker(
            worker_id=self._config.worker_id,
            address=self._address,
            metadata=metadata,
            ts=Timestamp.now(),
            slice_id=self._config.slice_id,
            scale_group=self._config.scale_group,
        )

    # ---- shutdown ---------------------------------------------------------

    def signal_stop(self, *, abrupt: bool = True) -> None:
        """Fire a signal at the child's process group.

        ``abrupt=True`` simulates prod preemption (SIGKILL, connection dies
        mid-flight). ``abrupt=False`` lets uvicorn drain in-flight RPCs
        first (SIGTERM → server.should_exit).
        """
        if self._process is None:
            return
        sig = signal.SIGKILL if abrupt else signal.SIGTERM
        try:
            os.killpg(os.getpgid(self._process.pid), sig)
        except ProcessLookupError:
            # Child already exited — nothing to kill.
            pass

    def join(self, *, timeout: float = 2.0) -> None:
        if self._process is None:
            return
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Deadline exceeded — escalate to SIGKILL and wait again.
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "synthetic worker %s did not exit after SIGKILL; leaking process pid=%s",
                    self._config.worker_id,
                    self._process.pid,
                )
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=timeout)
            self._reader_thread = None
        self._process = None

    def stop(self, *, abrupt: bool = True, timeout: float = 2.0) -> None:
        """Signal + join sequentially. Prefer ``signal_stop`` + parallel
        ``join`` when shutting down many workers at once; this combined form is
        kept for single-worker preemption where serial cost is negligible.
        """
        self.signal_stop(abrupt=abrupt)
        self.join(timeout=timeout)

    def _kill_unconditionally(self) -> None:
        """Best-effort teardown used when startup fails."""
        if self._process is None:
            return
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def _start_worker(worker: SyntheticWorker) -> SyntheticWorker:
    """Helper for ThreadPoolExecutor fan-out startup."""
    worker.start()
    return worker


class SyntheticWorkerPool:
    """Lifecycle manager for per-slice synthetic workers."""

    def __init__(
        self,
        transitions: ControllerTransitions,
        db: ControllerDB,
        *,
        delays: LifecycleDelays | None = None,
        startup_parallelism: int = STARTUP_PARALLELISM,
    ) -> None:
        self._transitions = transitions
        self._db = db
        self._delays = delays or LifecycleDelays()
        self._startup_parallelism = startup_parallelism
        self._workers: dict[str, list[SyntheticWorker]] = {}
        self._lock = threading.Lock()

    def start_workers_parallel(self, workers: list[SyntheticWorker]) -> None:
        """Start a batch of workers concurrently.

        Subprocess spawn + READY wait is ~300-500 ms per worker; serial
        startup of 600 preload workers would burn ~5 minutes. A bounded
        pool parallelises the Popen + readline block.
        """
        if not workers:
            return
        with ThreadPoolExecutor(
            max_workers=min(self._startup_parallelism, len(workers)),
            thread_name_prefix="synth-start",
        ) as pool:
            # list() forces all futures to resolve so exceptions propagate.
            list(pool.map(_start_worker, workers))

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
            workers.append(worker)
        self.start_workers_parallel(workers)
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
        """SIGKILL every worker's process group, then wait up to ``timeout``
        seconds *total* for them to exit. Subprocesses we fail to reap here
        become orphaned; ``start_new_session=True`` plus the explicit
        ``killpg`` should make that vanishingly rare.
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
