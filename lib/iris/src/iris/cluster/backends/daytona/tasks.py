# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct-dispatch Iris backend that runs each task attempt in its own Daytona v2 sandbox.

The controller performs all provider I/O. It creates the sandbox, uploads the
code bundle, workdir files and secrets, starts the task script as a sandbox
process, follows the process log into finelog and reads the exit status, so a
sandbox never connects to the controller. Tasks that call back into Iris (child
jobs, endpoints, actors) still need ``controller_address`` to be reachable from
Daytona, and tasks cannot reach each other: this backend runs independent CPU
tasks.
"""

import base64
import hashlib
import io
import logging
import math
import os
import signal
import tarfile
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, sleep

from daytona import (
    Daytona,
    DaytonaConfig,
    DaytonaConnectionError,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaTimeoutError,
    DaytonaV2Client,
)
from daytona.v2 import ProcessHandle, Resources, Sandbox, WaitResult
from daytona.v2 import _transport as daytona_transport
from finelog.client.log_client import LogClient
from finelog.rpc import logging_pb2
from rigging.filesystem.storage_path import prefix_join
from rigging.secrets import SECRETS_DIR, SECRETS_DIR_ENV, secret_file_contents, secret_file_environment
from rigging.timing import Duration, ExponentialBackoff

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
from iris.cluster.runtime.env import build_common_iris_env, build_task_script, write_workdir_files
from iris.cluster.runtime.sandbox import task_isolation
from iris.cluster.secrets import task_secret_values
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
# A launch fails, charging the task, when its image is not ready within this time.
IMAGE_TIMEOUT_SECONDS = 600
# Seconds one create request waits for its sandbox to start.
CREATE_WAIT_SECONDS = 600
# Daytona states of a sandbox that is still being provisioned.
PROVISIONING_STATES = frozenset({"queued", "creating", "starting"})
DELETE_TIMEOUT_SECONDS = 60
# A deletion whose answer was lost is repeated this many times.
DELETE_RETRIES = 3
# A followed log that ends before its eof resumes from its last cursor this many times in a row.
LOG_STREAM_RESUMES = 5
# Seconds a sandbox outlives its task's own timeout before Daytona deletes it.
TASK_TIMEOUT_SLACK_SECONDS = 600


class AttemptCancelled(Exception):
    """The attempt left the desired set while it was being launched."""


class SandboxNotStarted(Exception):
    """Daytona did not start the sandbox within the create request's wait."""


def sandbox_resources(resources: job_pb2.ResourceSpecProto) -> Resources:
    """A fixed CPU and memory shape for the request; v2 otherwise boots a burstable sandbox."""
    vcpus = resources.cpu_millicores / 1000 if resources.cpu_millicores else 1
    cores = max(MIN_CORES, math.ceil(vcpus) / VCPUS_PER_CORE)
    memory = max(1, math.ceil(resources.memory_bytes / 2**30)) if resources.memory_bytes else DEFAULT_MEMORY_GIB
    disk = math.ceil(resources.disk_bytes / 2**30) if resources.disk_bytes else None
    return Resources(cpu=cores, cpu_max=cores, memory=memory, memory_max=memory, disk=disk)


def task_environment(
    run_req: job_pb2.RunTaskRequest, *, controller_address: str, cluster_env: dict[str, str]
) -> dict[str, str]:
    """The task's variables as a Kubernetes pod receives them: cluster, then job, then Iris system values."""
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
    combined = {
        **(cluster_env if isolation.include_cluster_env else {}),
        **dict(run_req.environment.env_vars),
        **iris_env,
    }
    secrets = task_secret_values(run_req.environment)
    if secrets:
        combined[SECRETS_DIR_ENV] = str(SECRETS_DIR)
        for name, value in secret_file_environment(secrets).items():
            combined.setdefault(name, value)
    return combined


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


def _is_ambiguous(error: DaytonaError) -> bool:
    """Whether Daytona may have done what a request asked even though its answer was lost."""
    return isinstance(error, DaytonaConnectionError | DaytonaTimeoutError) or error.status_code in (502, 503, 504)


def _is_capacity(error: BaseException) -> bool:
    """Whether a launch failed for lack of Daytona capacity or a lost answer, not because of the task."""
    return isinstance(error, DaytonaError) and (_is_ambiguous(error) or error.status_code == 429)


def launch_failure_state(error: BaseException) -> int:
    """A provider failure before the process starts retries for free (still ASSIGNED); others charge the task."""
    if _is_capacity(error) or isinstance(error, SandboxNotStarted):
        return job_pb2.TASK_STATE_WORKER_FAILED
    return job_pb2.TASK_STATE_FAILED


@dataclass
class _Attempt:
    """One task attempt's sandbox and process, as far as they got."""

    run_req: job_pb2.RunTaskRequest
    launch: Future["_Launched"] = field(default_factory=Future)
    cancelled: threading.Event = field(default_factory=threading.Event)
    released: bool = False
    # Set before the process start request: from then on user code may run.
    start_requested: bool = False
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

    def check(self) -> None:
        if self.cancelled.is_set():
            raise AttemptCancelled(self.uid)

    def pause(self, seconds: float) -> None:
        """Wait ``seconds``, or less if the attempt is cancelled meanwhile."""
        self.cancelled.wait(seconds)
        self.check()


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
        cluster_env: dict[str, str],
    ):
        self.descriptor = descriptor
        self._config = config
        self._client = client
        self._bundles = bundles
        self._logs = logs
        self._cluster_name = cluster_name
        self._cluster_env = cluster_env
        self._attempts: dict[str, _Attempt] = {}
        self._lock = threading.Lock()
        self._release_lock = threading.Lock()
        self._launches = ThreadPoolExecutor(max_workers=config.launch_workers, thread_name_prefix="daytona-launch")
        # Deletions never queue behind launches.
        self._deletions = ThreadPoolExecutor(max_workers=config.launch_workers, thread_name_prefix="daytona-delete")

    # -- lifecycle -----------------------------------------------------------------

    def initialize(self, request: BackendRecoveryRequest) -> BackendRecoveryResult:
        """Delete sandboxes a previous controller left behind; their attempts are retried."""
        for sandbox in self._client.find_by_name(self._cluster_name):
            logger.info("Deleting Daytona sandbox %s left by an earlier controller", sandbox.id)
            self._delete(sandbox)
        return BackendRecoveryResult()

    def close(self) -> None:
        with self._lock:
            attempts = list(self._attempts.values())
            self._attempts.clear()
        for attempt in attempts:
            self._cancel(attempt)
        self._launches.shutdown(wait=True)
        self._deletions.shutdown(wait=True)
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
                self._cancel(self._attempts.pop(uid))
            for run_req in request.tasks_to_run:
                if run_req.attempt_uid not in self._attempts:
                    attempt = _Attempt(run_req=run_req)
                    attempt.launch = self._launches.submit(self._launch, attempt)
                    # Whatever the launch's outcome, a cancelled attempt's sandbox is deleted.
                    attempt.launch.add_done_callback(lambda _future, attempt=attempt: self._release(attempt))
                    self._attempts[run_req.attempt_uid] = attempt
            observed: set[str] = set()
            for uid in [run_req.attempt_uid for run_req in request.tasks_to_run] + [
                str(entry.attempt_uid) for entry in request.running_tasks
            ]:
                if uid in observed or uid not in self._attempts:
                    continue
                observed.add(uid)
                update = self._observe(self._attempts[uid])
                if update is not None:
                    updates.append(update)
            for entry in request.running_tasks:
                if str(entry.attempt_uid) not in self._attempts:
                    updates.append(
                        TaskUpdate(
                            task_id=entry.task_id,
                            attempt_id=entry.attempt_id,
                            attempt_uid=entry.attempt_uid,
                            new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                            error="No Daytona sandbox runs this attempt (the controller restarted)",
                        )
                    )
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
        # The attempt stays ASSIGNED until its process may run, so a failed launch retries for free.
        if not attempt.launch.done():
            return None
        error = attempt.launch.exception()
        if attempt.start_requested and not attempt.reported_running:
            # Report RUNNING before any end, so that a failure charges the preemption budget.
            attempt.reported_running = True
            return self._update(attempt, job_pb2.TASK_STATE_RUNNING, status_message="")
        if error is not None:
            state = job_pb2.TASK_STATE_WORKER_FAILED if attempt.start_requested else launch_failure_state(error)
            return self._update(attempt, state, error=f"Daytona launch failed: {error}")
        if attempt.finished.is_set():
            if attempt.exit_code is None:
                return self._update(attempt, job_pb2.TASK_STATE_WORKER_FAILED, error=attempt.error)
            if attempt.exit_code == 0:
                return self._update(attempt, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
            return self._update(
                attempt,
                job_pb2.TASK_STATE_FAILED,
                exit_code=attempt.exit_code,
                error=f"Task exited with status {attempt.exit_code}",
            )
        return None

    def _cancel(self, attempt: _Attempt) -> None:
        """Stop the attempt's launch; its sandbox is deleted once the launch has settled."""
        attempt.cancelled.set()
        attempt.launch.cancel()
        if attempt.launch.done():
            self._release(attempt)

    def _release(self, attempt: _Attempt) -> None:
        """Delete a cancelled attempt's sandbox from the deletion pool, once its launch has settled.

        Both the launch's done callback and ``_cancel`` call this; whichever sees
        a settled, cancelled launch first deletes the sandbox.
        """
        with self._release_lock:
            if attempt.released or not attempt.cancelled.is_set() or not attempt.launch.done():
                return
            attempt.released = True
        if attempt.launch.cancelled() or attempt.launch.exception() is not None:
            # A launch that never ran, or failed, has deleted what it created.
            return
        self._deletions.submit(self._delete, attempt.launch.result().sandbox)

    # -- provider I/O (launch pool) ------------------------------------------------

    def _ensure_image(self, attempt: _Attempt, reference: str, deadline: float) -> str:
        name = image_name(reference)
        try:
            image = self._client.images.get(name)
        except DaytonaNotFoundError:
            image = self._client.images.pull(reference, name=name, wait=False)
        while image.status not in ("ready", "failed", "error"):
            attempt.check()
            if monotonic() > deadline:
                raise RuntimeError(f"Daytona image {reference} was not ready within {IMAGE_TIMEOUT_SECONDS} s")
            # One server long poll per request.
            image = self._client.images.get(image.id, wait="ready")
        if image.status != "ready":
            raise RuntimeError(f"Daytona image {reference} is {image.status}: {image.error}")
        return name

    def _create(self, attempt: _Attempt, image: str) -> Sandbox:
        """Create the attempt's sandbox, waiting for Daytona capacity as a pod waits for a node.

        A repeat with the same idempotency key returns the sandbox an earlier,
        unanswered request created. If the launch stops after such a request,
        the sandbox is searched for by its attempt label and deleted.
        """
        run_req = attempt.run_req
        timeout_seconds = run_req.timeout.milliseconds // 1000 if run_req.timeout.milliseconds else 0
        ttl = self._config.sandbox_ttl_minutes * 60
        backoff = ExponentialBackoff(initial=1, maximum=30, factor=2, jitter=0.5)
        unanswered = False
        try:
            while True:
                attempt.check()
                try:
                    return self._client.create(
                        image,
                        # The cluster name is the search key that recovery deletes by; the label names the attempt.
                        name=self._cluster_name,
                        idempotency_key=f"{self._cluster_name}-{attempt.uid}",
                        labels={CLUSTER_LABEL: self._cluster_name, ATTEMPT_LABEL: attempt.uid},
                        resources=sandbox_resources(run_req.resources),
                        user="root",
                        lifecycle_policy={
                            "timeout_seconds": (
                                min(ttl, timeout_seconds + TASK_TIMEOUT_SLACK_SECONDS) if timeout_seconds else ttl
                            ),
                            "on_timeout": "delete",
                        },
                        wait_seconds=CREATE_WAIT_SECONDS,
                        create_timeout_seconds=CREATE_WAIT_SECONDS,
                    )
                except DaytonaError as error:
                    unanswered = unanswered or _is_ambiguous(error)
                    if not _is_capacity(error):
                        raise
                    attempt.pause(max(backoff.next_interval(), error.retry_after_seconds or 0))
        except BaseException:
            if unanswered:
                self._delete_attempt_sandboxes(attempt.uid)
            raise

    def _delete_attempt_sandboxes(self, uid: str) -> None:
        # The search is eventually consistent; the lifecycle policy deletes a sandbox it misses.
        for sandbox in self._client.find_by_name(self._cluster_name):
            if (sandbox.labels or {}).get(ATTEMPT_LABEL) == uid:
                self._delete(sandbox)

    def _launch(self, attempt: _Attempt) -> _Launched:
        run_req = attempt.run_req
        deadline = monotonic() + IMAGE_TIMEOUT_SECONDS
        attempt.check()
        image = self._ensure_image(attempt, run_req.task_image or self._config.default_task_image, deadline)
        sandbox = self._create(attempt, image)
        try:
            if sandbox.state in PROVISIONING_STATES:
                raise SandboxNotStarted(f"Daytona sandbox {sandbox.id} is still {sandbox.state}")
            if sandbox.state != "started":
                raise RuntimeError(f"Daytona sandbox {sandbox.id} is {sandbox.state}: {sandbox.error}")
            attempt.check()
            self._upload_workdir(sandbox, run_req)
            attempt.check()
            attempt.start_requested = True
            process = sandbox.process.start(
                argv=["bash", "-lc", build_task_script(run_req)],
                cwd=WORKDIR,
                env=task_environment(
                    run_req, controller_address=self._config.controller_address, cluster_env=self._cluster_env
                ),
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
        """Unpack the bundle, workdir files and secret files through one tar upload."""
        with tempfile.TemporaryDirectory(prefix="iris-daytona-") as staging:
            root = Path(staging)
            app = root / WORKDIR.lstrip("/")
            if run_req.bundle_id:
                self._bundles.extract_bundle_to(run_req.bundle_id, app)
            app.mkdir(parents=True, exist_ok=True)
            # Paths are validated, so a name cannot escape the workdir.
            write_workdir_files(app, dict(run_req.entrypoint.workdir_files))
            write_workdir_files(
                app,
                {name: self._bundles.get(digest) for name, digest in run_req.entrypoint.workdir_file_refs.items()},
            )
            secrets = secret_file_contents(task_secret_values(run_req.environment))
            secrets_dir = root / str(SECRETS_DIR).lstrip("/")
            if secrets:
                secrets_dir.mkdir(parents=True, mode=0o700)
                for name, value in secrets.items():
                    (secrets_dir / name).write_text(value)
                    (secrets_dir / name).chmod(0o600)
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                # Name each tree, so that unpacking at / leaves the parent directories as they are.
                tar.add(app, arcname=str(app.relative_to(root)))
                if secrets:
                    tar.add(secrets_dir, arcname=str(secrets_dir.relative_to(root)))
        remote = f"/tmp/iris-workdir-{run_req.attempt_uid}.tar.gz"
        sandbox.fs.write_bytes(remote, archive.getvalue())
        unpacked = sandbox.process.run(argv=["sh", "-c", f"tar xzf {remote} -C / && rm -f {remote}"])
        if unpacked.exit_code != 0:
            raise RuntimeError(f"Cannot unpack the task workdir: {unpacked.output}")

    def _follow(self, attempt: _Attempt, process: ProcessHandle) -> None:
        """Copy the process log into finelog under the attempt's key and record its exit status."""
        key = f"{attempt.run_req.task_id}:{attempt.run_req.attempt_id}"
        pending: dict[str, bytes] = {"stdout": b"", "stderr": b""}

        def entry(source: str, line: bytes) -> logging_pb2.LogEntry:
            text = line.decode(errors="replace")
            record = logging_pb2.LogEntry(
                source=source, data=text, attempt_id=attempt.run_req.attempt_id, level=classify_log_level(source, text)
            )
            record.timestamp.epoch_ms = int(time.time() * 1000)
            return record

        cursor: str | None = None
        resumes = 0
        backoff = ExponentialBackoff(initial=0.5, maximum=4, factor=2, jitter=0.5)
        try:
            try:
                while True:
                    try:
                        for event in process.stream_logs(cursor=cursor, encoding="base64"):
                            if event.type == "log":
                                source = "stderr" if event.frame.channel == "stderr" else "stdout"
                                *lines, pending[source] = (pending[source] + decode_frame(event.frame.data)).split(b"\n")
                                if lines:
                                    # Flush each frame's complete lines, so a quiet task still streams.
                                    self._logs.write_batch(key, [entry(source, line) for line in lines])
                            elif event.type == "state":
                                attempt.exit_code = exit_status(event.result)
                            elif event.type == "warning":
                                logger.warning("Daytona dropped output of attempt %s: %s", attempt.uid, event.message)
                            cursor, resumes = event.cursor, 0
                        break
                    except DaytonaConnectionError:
                        resumes += 1
                        if resumes > LOG_STREAM_RESUMES or attempt.cancelled.is_set():
                            raise
                        sleep(backoff.next_interval())
            except DaytonaError as error:
                # The log is lost from here on; the exit status is read from the process record below.
                logger.warning("Lost the log of Daytona attempt %s: %s", attempt.uid, error)
            if any(pending.values()):
                self._logs.write_batch(key, [entry(source, rest) for source, rest in pending.items() if rest])
            while attempt.exit_code is None and not attempt.cancelled.is_set():
                result = process.wait(timeout=DELETE_TIMEOUT_SECONDS)
                if result.reason != "timed_out":
                    attempt.exit_code = exit_status(result)
        except (DaytonaError, RuntimeError) as error:
            if attempt.exit_code is None:
                attempt.error = f"Lost the Daytona process: {error}"
        finally:
            attempt.finished.set()

    def _delete(self, sandbox: Sandbox) -> None:
        backoff = ExponentialBackoff(initial=1, maximum=10, factor=2, jitter=0.5)
        for retry in range(DELETE_RETRIES + 1):
            try:
                self._client.delete(sandbox, timeout=DELETE_TIMEOUT_SECONDS)
                return
            except DaytonaNotFoundError:
                return
            except DaytonaError as error:
                if not _is_ambiguous(error) or retry == DELETE_RETRIES:
                    # The lifecycle policy deletes the sandbox at its TTL if this request never lands.
                    logger.warning("Cannot delete Daytona sandbox %s: %s", sandbox.id, error)
                    return
                sleep(backoff.next_interval())

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
        # A negative timeout means none, as for the other backends; Daytona also reads zero as none.
        result = attempt.launch.result().sandbox.process.run(
            argv=list(request.command), cwd=WORKDIR, execution_timeout=timeout_seconds if timeout_seconds >= 0 else None
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
    # Assigning a name the SDK no longer reads would silently re-enable HTTP/2.
    assert hasattr(daytona_transport, "_HTTP2_AVAILABLE"), "The Daytona SDK no longer has its HTTP/2 switch"
    daytona_transport._HTTP2_AVAILABLE = False
    return Daytona(DaytonaConfig(api_key=api_key)).v2


def make_daytona_provider(
    config: DaytonaProviderConfig,
    *,
    descriptor: BackendDescriptor,
    remote_state_dir: str,
    logs: LogClient,
    cluster_name: str,
    cluster_env: dict[str, str],
) -> DaytonaTaskProvider:
    return DaytonaTaskProvider(
        config,
        descriptor=descriptor,
        client=daytona_client(config.api_key_env),
        bundles=BundleStore(storage_dir=prefix_join(remote_state_dir, "bundles")),
        logs=logs,
        cluster_name=cluster_name,
        cluster_env=cluster_env,
    )
