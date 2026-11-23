# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct-dispatch Iris backend that runs each task attempt in its own Daytona v2 sandbox.

The controller performs all provider I/O. It creates the sandbox, uploads the
code bundle, starts the task script as a sandbox process, follows the process
log into finelog and reads the exit status, so a sandbox never connects to the
controller. Tasks that call back into Iris (child jobs, endpoints, actors) still
need ``controller_address`` to be reachable from Daytona, and tasks cannot reach
each other: this backend runs independent CPU tasks.
"""

import base64
import hashlib
import io
import logging
import math
import os
import shlex
import signal
import tarfile
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from daytona import Daytona, DaytonaConfig, DaytonaError, DaytonaNotFoundError, DaytonaV2Client
from daytona.v2 import ProcessHandle, Resources, Sandbox, WaitResult
from daytona.v2 import _transport as daytona_transport
from finelog.client.log_client import LogClient
from finelog.rpc import logging_pb2
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration

from iris.cluster.bundle import BundleStore
from iris.cluster.config import DaytonaProviderConfig
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendDescriptor,
    BackendObservation,
    BackendObservationRequest,
    BackendRecoveryRequest,
    BackendRecoveryResult,
    DirectReconcileRequest,
    EgressAllowlistMode,
    JobFeasibilityRequest,
    ProviderUnsupportedError,
    ReconcileObservation,
    ReconcileRequest,
    RemoveCapacityRequest,
    RemoveCapacityResult,
    ScheduleRequest,
    ScheduleResult,
    TaskTarget,
)
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.log_keys import classify_log_level
from iris.cluster.runtime.env import build_common_iris_env, render_setup_steps
from iris.cluster.runtime.sandbox import task_isolation
from iris.cluster.types import AttemptUid, JobName
from iris.preemption import resolve_termination_grace
from iris.rpc import controller_pb2, job_pb2, worker_pb2

logger = logging.getLogger(__name__)

WORKDIR = "/app"
# Labels every sandbox this backend owns, so a restarted controller can reclaim them.
CLUSTER_LABEL = "iris-cluster"
ATTEMPT_LABEL = "iris-attempt-uid"
IMAGE_PREFIX = "iris-"
# One v2 core is two vCPUs; a sandbox needs at least half a core.
VCPUS_PER_CORE = 2
MIN_CORES = 0.5
DEFAULT_MEMORY_GIB = 1
CREATE_TIMEOUT_SECONDS = 600
DELETE_TIMEOUT_SECONDS = 60
LOG_BATCH_LINES = 200


def sandbox_resources(resources: job_pb2.ResourceSpecProto) -> Resources:
    """A fixed CPU and memory shape for the request; v2 otherwise boots a burstable sandbox."""
    vcpus = resources.cpu_millicores / 1000 if resources.cpu_millicores else 1
    cores = max(MIN_CORES, math.ceil(vcpus) / VCPUS_PER_CORE)
    memory = max(1, math.ceil(resources.memory_bytes / 2**30)) if resources.memory_bytes else DEFAULT_MEMORY_GIB
    disk = math.ceil(resources.disk_bytes / 2**30) if resources.disk_bytes else None
    return Resources(cpu=cores, cpu_max=cores, memory=memory, memory_max=memory, disk=disk)


def build_task_script(run_req: job_pb2.RunTaskRequest) -> str:
    """Run the setup steps, then the run command, in the bundle directory."""
    lines = ["set -e", "ulimit -c 0", f"cd {WORKDIR}"]
    lines.extend(render_setup_steps(run_req.entrypoint.setup_commands))
    lines.append('[ -f "$IRIS_VENV/bin/activate" ] && source "$IRIS_VENV/bin/activate"')
    if run_req.entrypoint.run_command.argv:
        lines.append("exec " + shlex.join(run_req.entrypoint.run_command.argv))
    return "\n".join(lines)


def task_environment(run_req: job_pb2.RunTaskRequest, controller_address: str) -> dict[str, str]:
    """The job's variables overlaid with the Iris system variables, as a pod receives them."""
    isolation = task_isolation(run_req.container_profile, run_req.egress_policy)
    iris_env = build_common_iris_env(
        task_id=run_req.task_id,
        attempt_id=run_req.attempt_id,
        attempt_uid=run_req.attempt_uid,
        num_tasks=run_req.num_tasks,
        bundle_id=run_req.bundle_id,
        controller_address=controller_address if controller_address and isolation.include_controller_address else None,
        task_token=run_req.task_token if isolation.include_task_token else None,
        environment=run_req.environment,
        constraints=run_req.constraints,
        ports=run_req.ports,
        resources=run_req.resources if run_req.HasField("resources") else None,
        termination_grace=resolve_termination_grace(Duration.from_ms(run_req.termination_grace.milliseconds)),
    )
    return {**dict(run_req.environment.env_vars), **iris_env}


def image_name(reference: str) -> str:
    return IMAGE_PREFIX + hashlib.sha256(reference.encode()).hexdigest()[:40]


def exit_status(result: WaitResult) -> int:
    """The process exit code, or ``128 + signal`` like a shell for a signaled process."""
    if result.exit_code is not None:
        return result.exit_code
    if result.signal is not None:
        return 128 + signal.Signals[f"SIG{result.signal}"].value
    raise RuntimeError(f"Daytona process ended without an exit status: {result}")


def decode_frame(data: str) -> bytes:
    """Decode a v2 base64 log frame, which is URL-safe and unpadded."""
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


@dataclass
class _Attempt:
    """One task attempt's sandbox and process, as far as they got."""

    run_req: job_pb2.RunTaskRequest
    launch: Future["_Launched"]
    reported_running: bool = False
    exit_code: int | None = None
    error: str | None = None
    finished: threading.Event = field(default_factory=threading.Event)

    @property
    def task_id(self) -> JobName:
        return JobName.from_wire(self.run_req.task_id)

    @property
    def uid(self) -> str:
        return self.run_req.attempt_uid


@dataclass(frozen=True)
class _Launched:
    sandbox: Sandbox
    process: ProcessHandle


class DaytonaTaskProvider:
    """A ``TaskBackend`` with direct dispatch onto Daytona v2 sandboxes."""

    def __init__(
        self,
        config: DaytonaProviderConfig,
        *,
        descriptor: BackendDescriptor,
        client: DaytonaV2Client,
        bundles: BundleStore,
        logs: LogClient,
        cluster_name: str,
    ):
        self.descriptor = descriptor
        self._config = config
        self._client = client
        self._bundles = bundles
        self._logs = logs
        self._cluster_name = cluster_name
        self._attempts: dict[str, _Attempt] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=config.launch_workers, thread_name_prefix="daytona-launch")

    # -- lifecycle -----------------------------------------------------------------

    def initialize(self, request: BackendRecoveryRequest) -> BackendRecoveryResult:
        """Delete sandboxes a previous controller left behind; their attempts are re-dispatched."""
        for sandbox in self._client.find_by_name(self._cluster_name):
            logger.info("Deleting Daytona sandbox %s left by an earlier controller", sandbox.id)
            self._delete(sandbox)
        return BackendRecoveryResult()

    def close(self) -> None:
        with self._lock:
            attempts = list(self._attempts.values())
            self._attempts.clear()
        for attempt in attempts:
            self._stop(attempt)
        self._pool.shutdown(wait=True)
        self._client.close()

    # -- reconcile -----------------------------------------------------------------

    def reconcile(self, request: ReconcileRequest) -> ReconcileObservation:
        assert isinstance(request, DirectReconcileRequest), "Daytona dispatches tasks directly"
        updates: list[TaskUpdate] = []
        desired = {run_req.attempt_uid for run_req in request.tasks_to_run}
        desired.update(str(entry.attempt_uid) for entry in request.running_tasks)
        with self._lock:
            for uid in [uid for uid in self._attempts if uid not in desired]:
                # Cancelled, timed out or finished: the attempt left the desired set.
                self._pool.submit(self._stop, self._attempts.pop(uid))
            for run_req in request.tasks_to_run:
                if run_req.attempt_uid not in self._attempts:
                    attempt = _Attempt(run_req=run_req, launch=Future())
                    attempt.launch = self._pool.submit(self._launch, attempt)
                    self._attempts[run_req.attempt_uid] = attempt
                updates.append(self._update(self._attempts[run_req.attempt_uid], job_pb2.TASK_STATE_BUILDING))
            for entry in request.running_tasks:
                attempt = self._attempts.get(str(entry.attempt_uid))
                if attempt is None:
                    updates.append(
                        TaskUpdate(
                            task_id=entry.task_id,
                            attempt_id=entry.attempt_id,
                            attempt_uid=entry.attempt_uid,
                            new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                            error="No Daytona sandbox runs this attempt (the controller restarted)",
                        )
                    )
                    continue
                update = self._observe(attempt)
                if update is not None:
                    updates.append(update)
        return ReconcileObservation(task_updates=updates)

    def _update(self, attempt: _Attempt, state: int, **fields) -> TaskUpdate:
        return TaskUpdate(
            task_id=attempt.task_id,
            attempt_id=attempt.run_req.attempt_id,
            attempt_uid=AttemptUid(attempt.uid),
            new_state=state,
            **fields,
        )

    def _observe(self, attempt: _Attempt) -> TaskUpdate | None:
        if not attempt.launch.done():
            return None
        error = attempt.launch.exception()
        if error is not None:
            # Nothing ran, so the attempt retries without charging the task's failure budget.
            return self._update(attempt, job_pb2.TASK_STATE_WORKER_FAILED, error=f"Daytona launch failed: {error}")
        if attempt.finished.is_set():
            if attempt.error is not None:
                return self._update(attempt, job_pb2.TASK_STATE_WORKER_FAILED, error=attempt.error)
            if attempt.exit_code == 0:
                return self._update(attempt, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
            return self._update(
                attempt,
                job_pb2.TASK_STATE_FAILED,
                exit_code=attempt.exit_code,
                error=f"Task exited with status {attempt.exit_code}",
            )
        if attempt.reported_running:
            return None
        attempt.reported_running = True
        return self._update(attempt, job_pb2.TASK_STATE_RUNNING, status_message="")

    # -- provider I/O (launch pool) ------------------------------------------------

    def _ensure_image(self, reference: str) -> str:
        name = image_name(reference)
        try:
            image = self._client.images.get(name)
        except DaytonaNotFoundError:
            image = self._client.images.pull(reference, name=name, wait=False)
        while image.status not in ("ready", "failed", "error"):
            image = self._client.images.get(image.id, wait="ready")
        if image.status != "ready":
            raise RuntimeError(f"Daytona image {reference} is {image.status}: {image.error}")
        return name

    def _launch(self, attempt: _Attempt) -> _Launched:
        run_req = attempt.run_req
        image = self._ensure_image(run_req.task_image or self._config.default_task_image)
        timeout_seconds = run_req.timeout.milliseconds // 1000 if run_req.timeout.milliseconds else 0
        ttl = self._config.sandbox_ttl_minutes * 60
        sandbox = self._client.create(
            image,
            # The cluster name is the search key that recovery deletes by; the label names the attempt.
            name=self._cluster_name,
            idempotency_key=f"{self._cluster_name}-{attempt.uid}",
            labels={CLUSTER_LABEL: self._cluster_name, ATTEMPT_LABEL: attempt.uid},
            resources=sandbox_resources(run_req.resources),
            user="root",
            lifecycle_policy={
                "timeout_seconds": min(ttl, timeout_seconds + 600) if timeout_seconds else ttl,
                "on_timeout": "delete",
            },
            wait_seconds=CREATE_TIMEOUT_SECONDS,
            create_timeout_seconds=CREATE_TIMEOUT_SECONDS,
        )
        try:
            if sandbox.state != "started":
                raise RuntimeError(f"Daytona sandbox {sandbox.id} is {sandbox.state}: {sandbox.error}")
            self._upload_workdir(sandbox, run_req)
            process = sandbox.process.start(
                argv=["bash", "-lc", build_task_script(run_req)],
                cwd=WORKDIR,
                env=task_environment(run_req, self._config.controller_address),
                tag=attempt.uid,
            )
        except BaseException:
            self._delete(sandbox)
            raise
        threading.Thread(
            target=self._follow, args=(attempt, process), name=f"daytona-log-{attempt.uid}", daemon=True
        ).start()
        return _Launched(sandbox, process)

    def _upload_workdir(self, sandbox: Sandbox, run_req: job_pb2.RunTaskRequest) -> None:
        """Unpack the bundle and the inline workdir files into ``WORKDIR`` through one tar upload."""
        with tempfile.TemporaryDirectory(prefix="iris-daytona-") as staging:
            root = Path(staging) / "app"
            if run_req.bundle_id:
                self._bundles.extract_bundle_to(run_req.bundle_id, root)
            root.mkdir(parents=True, exist_ok=True)
            for name, data in run_req.entrypoint.workdir_files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(data)
            for name, digest in run_req.entrypoint.workdir_file_refs.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(self._bundles.get(digest))
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                tar.add(root, arcname=".")
        remote = f"/tmp/iris-workdir-{run_req.attempt_uid}.tar.gz"
        sandbox.fs.write_bytes(remote, archive.getvalue())
        unpacked = sandbox.process.run(
            argv=["sh", "-c", f"mkdir -p {WORKDIR} && tar xzf {remote} -C {WORKDIR} && rm -f {remote}"]
        )
        if unpacked.exit_code != 0:
            raise RuntimeError(f"Cannot unpack the task workdir: {unpacked.output}")

    def _follow(self, attempt: _Attempt, process: ProcessHandle) -> None:
        """Copy the process log into finelog under the attempt's key and record its exit status."""
        key = f"{attempt.run_req.task_id}:{attempt.run_req.attempt_id}"
        pending: dict[str, bytes] = {"stdout": b"", "stderr": b""}
        batch: list[logging_pb2.LogEntry] = []

        def emit(source: str, line: bytes) -> None:
            text = line.decode(errors="replace")
            entry = logging_pb2.LogEntry(
                source=source,
                data=text,
                attempt_id=attempt.run_req.attempt_id,
                level=classify_log_level(source, text),
            )
            entry.timestamp.epoch_ms = int(time.time() * 1000)
            batch.append(entry)

        try:
            for event in process.stream_logs(encoding="base64"):
                if event.type == "log":
                    source = "stderr" if event.frame.channel == "stderr" else "stdout"
                    *lines, pending[source] = (pending[source] + decode_frame(event.frame.data)).split(b"\n")
                    for line in lines:
                        emit(source, line)
                elif event.type == "state":
                    attempt.exit_code = exit_status(event.result)
                if len(batch) >= LOG_BATCH_LINES:
                    self._logs.write_batch(key, batch)
                    batch = []
            for source, rest in pending.items():
                if rest:
                    emit(source, rest)
            if attempt.exit_code is None:
                attempt.error = "Daytona process log ended without an exit status"
        except DaytonaError as error:
            attempt.error = f"Lost the Daytona process: {error}"
        finally:
            self._logs.write_batch(key, batch)
            attempt.finished.set()

    def _stop(self, attempt: _Attempt) -> None:
        try:
            launched = attempt.launch.result()
        except BaseException:
            return
        self._delete(launched.sandbox)

    def _delete(self, sandbox: Sandbox) -> None:
        try:
            self._client.delete(sandbox, timeout=DELETE_TIMEOUT_SECONDS)
        except DaytonaNotFoundError:
            pass
        except DaytonaError as error:
            # The lifecycle policy deletes the sandbox at its TTL if this request is lost.
            logger.warning("Cannot delete Daytona sandbox %s: %s", sandbox.id, error)

    # -- on-demand RPCs ------------------------------------------------------------

    def exec_in_container(
        self,
        target: TaskTarget,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        with self._lock:
            attempt = self._attempts.get(target.attempt_uid)
        if attempt is None or not attempt.launch.done() or attempt.launch.exception() is not None:
            return worker_pb2.Worker.ExecInContainerResponse(error="The attempt has no running Daytona sandbox")
        # A negative timeout means none, as for the other backends.
        result = attempt.launch.result().sandbox.process.run(
            argv=list(request.command), cwd=WORKDIR, execution_timeout=timeout_seconds if timeout_seconds > 0 else None
        )
        if result.exit_code is None:
            return worker_pb2.Worker.ExecInContainerResponse(error=f"Command did not exit: {result.reason}")
        return worker_pb2.Worker.ExecInContainerResponse(
            exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr
        )

    def get_process_status(self, target: TaskTarget, request: job_pb2.GetProcessStatusRequest):
        raise ProviderUnsupportedError("Daytona does not report process status")

    def profile_task(
        self, target: TaskTarget, request: job_pb2.ProfileTaskRequest, timeout_ms: int
    ) -> job_pb2.ProfileTaskResponse:
        raise ProviderUnsupportedError("Daytona does not profile tasks")

    # -- admission -----------------------------------------------------------------

    def runtime_image(self, requested_image: str) -> str:
        return requested_image or self._config.default_task_image

    def validate_job(self, request: controller_pb2.Controller.LaunchJobRequest) -> str | None:
        if request.resources.HasField("device") and not request.resources.device.HasField("cpu"):
            return "Daytona runs CPU tasks only"
        if request.coscheduling.group_by:
            return "Daytona tasks cannot reach each other, so coscheduled jobs are unsupported"
        return None

    def job_feasibility(self, request: JobFeasibilityRequest) -> str | None:
        if request.resources.HasField("device") and not request.resources.device.HasField("cpu"):
            return "Daytona runs CPU tasks only"
        return None

    def egress_allowlist_mode(self, request: controller_pb2.Controller.LaunchJobRequest) -> EgressAllowlistMode:
        return EgressAllowlistMode.UNAVAILABLE

    # -- capacity (Daytona places sandboxes itself) --------------------------------

    def observe(self, request: BackendObservationRequest) -> BackendObservation:
        return BackendObservation()

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        return ScheduleResult()

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        return AutoscaleResult()

    def remove_capacity(self, request: RemoveCapacityRequest) -> RemoveCapacityResult:
        return RemoveCapacityResult()


def daytona_client(api_key_env: str) -> DaytonaV2Client:
    """A v2 client for the key in ``api_key_env``, never the SDK's ambient DAYTONA_API_KEY.

    Over HTTP/2 the sandbox proxy stalls requests that share a connection with a
    followed process log, and the SDK enables HTTP/2 whenever h2 is importable, so
    this turns it off for every Daytona client in the process.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Daytona provider requires the API key in ${api_key_env}")
    daytona_transport._HTTP2_AVAILABLE = False
    return Daytona(DaytonaConfig(api_key=api_key)).v2


def make_daytona_provider(
    config: DaytonaProviderConfig,
    *,
    descriptor: BackendDescriptor,
    remote_state_dir: str,
    logs: LogClient,
    cluster_name: str,
) -> DaytonaTaskProvider:
    return DaytonaTaskProvider(
        config,
        descriptor=descriptor,
        client=daytona_client(config.api_key_env),
        bundles=BundleStore(storage_dir=prefix_join(remote_state_dir, "bundles")),
        logs=logs,
        cluster_name=cluster_name,
    )
