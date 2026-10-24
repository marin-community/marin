# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a registry image as a gVisor Iris job and exec into its container."""

import asyncio
import base64
import math
import re
import shlex
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cli.connect import ControllerEndpoint, connect_controller
from iris.client import IrisClient, Job, Task
from iris.cluster.client.job_info import get_job_info
from iris.cluster.runtime.sandbox import (
    SANDBOX_SERVICE_GROUP_CAPABILITY,
    validate_sandbox_environment,
    validate_sandbox_services,
)
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, SandboxServiceSpec
from iris.resources.state import TaskState
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.errors import DEFAULT_RETRY_MAX_ATTEMPTS, DEFAULT_RETRY_MAX_ELAPSED
from rigging.secrets import SecretSpec, resolve_secret_spec
from rigging.timing import Duration, ExponentialBackoff, retry_with_backoff

from shellbox.image import RegistryImage
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    Backend,
    Command,
    CommandSemantics,
    ExecutionRequirements,
    ExitReason,
    MachineSpec,
    MachineTerminated,
    NetworkPolicy,
    Result,
    UnsupportedMachineSpec,
    check_execution_requirements,
)
from shellbox.service import ServiceGroup, ServiceGroupSpec

# The worker runs `docker exec` or `kubectl exec` with the command as argv, and Linux caps one
# argument at 128 KiB (MAX_ARG_STRLEN). An upload chunk travels base64-encoded inside one
# `sh -c` script, so 64 KiB of data (87,384 encoded bytes) leaves room for the target path.
TRANSFER_CHUNK_BYTES = 64 * 1024
DEFAULT_MEMORY_MB = 2048
DEFAULT_DISK_MB = 10240
DEFAULT_SCHEDULING_TIMEOUT = 600
IDLE_ENTRYPOINT = "trap 'exit 0' TERM INT; sleep infinity & wait"
"""Keeps the sandbox alive for exec while exiting promptly when Iris stops it."""
DEFAULT_JOB_TTL = 6 * 60 * 60
RPC_PADDING_SECONDS = 60
# How long ``close`` waits for admitted execs to end after it cancels the sandbox job.
CLOSE_DRAIN_SECONDS = 60
EXEC_SHED_BACKOFF = ExponentialBackoff(initial=0.5, maximum=10.0, factor=2.0)
# How much of each stuck exec's command ``close`` quotes when the drain times out.
STUCK_EXEC_PREVIEW_CHARS = 120

# ALLOW reaches public internet addresses only; neither mode reaches the cluster.
EGRESS_POLICIES = {
    NetworkPolicy.ALLOW: job_pb2.EGRESS_POLICY_INTERNET,
    NetworkPolicy.DENY: job_pb2.EGRESS_POLICY_NONE,
}


def _exec_was_shed(error: Exception) -> bool:
    """The controller refused the exec before running it because its exec pool was full.

    Only this refusal is retried: after any other error the command may already have run.
    """
    return isinstance(error, ConnectError) and error.code == Code.RESOURCE_EXHAUSTED


@dataclass(frozen=True)
class ExecOutput:
    """The completed command's exit code and text output from one Iris container exec."""

    exit_code: int
    stdout: str
    stderr: str


@dataclass
class IrisExecutionState:
    """Shared admission and shutdown state for every container in one Iris job."""

    closed: bool = False
    admission: threading.Condition = field(default_factory=threading.Condition)
    in_flight: dict[str, list[str]] = field(default_factory=dict)


class IrisMachine:
    """One Iris task container; file transfer uses bounded base64 exec calls."""

    def __init__(
        self,
        endpoint: ControllerEndpoint,
        client: IrisClient,
        rpc: ControllerServiceClientSync,
        job: Job,
        task: Task,
        spec: MachineSpec,
        *,
        container_name: str = "",
        execution: IrisExecutionState | None = None,
    ):
        self.endpoint = endpoint
        self.client = client
        self.rpc = rpc
        self.job = job
        self.task = task
        self.spec = spec
        self.container_name = container_name
        self.execution = execution if execution is not None else IrisExecutionState()
        self._container_user: tuple[str, str] | None = None

    def _exec_sync(self, argv: list[str], timeout: float | None = None) -> ExecOutput:
        seconds = math.ceil(timeout) if timeout is not None else -1
        request = controller_pb2.Controller.ExecInContainerRequest(
            task_id=self.task.task_id.to_wire(),
            command=argv,
            timeout_seconds=seconds,
            container_name=self.container_name,
        )
        timeout_ms = (seconds + RPC_PADDING_SECONDS) * 1000 if seconds >= 0 else DEFAULT_JOB_TTL * 1000

        def attempt() -> controller_pb2.Controller.ExecInContainerResponse:
            # A timed-out ``run`` closes the machine while this thread may still be retrying.
            with self.execution.admission:
                if self.execution.closed:
                    raise RuntimeError("Machine is closed")
                attempt_id = uuid.uuid4().hex
                self.execution.in_flight[attempt_id] = argv
            try:
                return self.rpc.exec_in_container(request, timeout_ms=timeout_ms)
            finally:
                with self.execution.admission:
                    del self.execution.in_flight[attempt_id]
                    self.execution.admission.notify_all()

        try:
            response = retry_with_backoff(
                attempt,
                retryable=_exec_was_shed,
                max_attempts=DEFAULT_RETRY_MAX_ATTEMPTS,
                max_elapsed=DEFAULT_RETRY_MAX_ELAPSED,
                backoff=EXEC_SHED_BACKOFF,
                operation=f"Iris exec in {self.task.task_id}",
            )
        except ConnectError as error:
            self._raise_if_terminated(error)
            raise
        if response.error:
            error = RuntimeError(f"Iris exec failed: {response.error}")
            self._raise_if_terminated(error)
            raise error
        return ExecOutput(exit_code=response.exit_code, stdout=response.stdout, stderr=response.stderr)

    def _raise_if_terminated(self, cause: Exception) -> None:
        """Raise ``MachineTerminated`` from ``cause`` when the sandbox task is no longer running."""
        status = self.task.status()
        if status.state != TaskState.RUNNING:
            raise MachineTerminated(
                f"Iris sandbox task {self.task.task_id} is {status.state}: {status.error_message or cause}"
            ) from cause

    async def _script(self, script: str, timeout: float | None = None) -> ExecOutput:
        return await asyncio.to_thread(self._exec_sync, ["sh", "-c", script], timeout)

    async def _checked(self, script: str) -> str:
        response = await self._script(script)
        if response.exit_code:
            raise RuntimeError(f"Iris command failed ({response.exit_code}): {response.stderr or response.stdout}")
        return response.stdout

    async def _upload_bytes(self, data: bytes, target: str) -> None:
        quoted = shlex.quote(target)
        await self._checked(f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))} && : > {quoted}")
        for offset in range(0, len(data), TRANSFER_CHUNK_BYTES):
            encoded = base64.b64encode(data[offset : offset + TRANSFER_CHUNK_BYTES]).decode("ascii")
            await self._checked(f"printf '%s' {encoded} | base64 -d >> {quoted}")

    async def _download_bytes(self, source: str, limit: int | None = None) -> tuple[bytes, bool]:
        quoted = shlex.quote(source)
        size = int((await self._checked(f"wc -c < {quoted}")).strip())
        count = size if limit is None else min(size, limit)
        chunks = []
        for offset in range(0, count, TRANSFER_CHUNK_BYTES):
            length = min(TRANSFER_CHUNK_BYTES, count - offset)
            encoded = await self._checked(f"tail -c +{offset + 1} {quoted} | head -c {length} | base64")
            chunks.append(base64.b64decode(encoded))
        return b"".join(chunks), size > count

    async def _user_identity(self) -> tuple[str, str]:
        """The ``(uid, name)`` the container runs commands as; Iris exec cannot switch users."""
        if self._container_user is None:
            uid, name = (await self._checked("id -u && id -un")).split()
            self._container_user = (uid, name)
        return self._container_user

    async def run(self, command: Command) -> Result:
        if self.execution.closed:
            raise RuntimeError("Machine is closed")
        validate_sandbox_environment({**self.spec.env, **command.env})
        if command.user is not None and command.user not in await self._user_identity():
            uid, name = await self._user_identity()
            raise UnsupportedMachineSpec(
                f"Iris runs every command as the container user {name} ({uid}); cannot run as {command.user}"
            )
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        prefix = f"/tmp/.shellbox-{uuid.uuid4().hex}"
        stdin_path, stdout_path, stderr_path = (f"{prefix}-{part}" for part in ("in", "out", "err"))
        if command.stdin:
            await self._upload_bytes(command.stdin, stdin_path)
        exports = "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in {**self.spec.env, **command.env}.items()
        )
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in {**self.spec.env, **command.env}):
            raise ValueError("Environment variable names must be shell identifiers")
        script = (
            f"{exports}\ncd {shlex.quote(command.cwd or self.spec.workdir)} || exit 1\n"
            f"{shlex.join(command.argv)} < {shlex.quote(stdin_path) if command.stdin else '/dev/null'} "
            f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
        )
        try:
            response = await asyncio.wait_for(self._script(script), timeout=command.timeout)
            stdout, stdout_truncated = await self._download_bytes(stdout_path, command.output_limit_bytes)
            stderr, stderr_truncated = await self._download_bytes(stderr_path, command.output_limit_bytes)
            return Result(response.exit_code, stdout, stderr, stdout_truncated, stderr_truncated, ExitReason.EXITED)
        except TimeoutError:
            await self.close()
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError:
            await self.close()
            raise
        finally:
            if not self.execution.closed:
                await self._script(
                    f"rm -f {shlex.quote(stdin_path)} {shlex.quote(stdout_path)} {shlex.quote(stderr_path)}"
                )

    async def upload(self, source: Path, target: str) -> None:
        if source.is_dir():
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                with tarfile.open(archive.name, "w:gz") as tar:
                    tar.add(source, arcname=".")
                remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
                await self._upload_bytes(Path(archive.name).read_bytes(), remote_archive)
            try:
                await self._checked(
                    f"mkdir -p {shlex.quote(target)} && tar xzf {shlex.quote(remote_archive)} "
                    f"--no-same-owner -C {shlex.quote(target)}"
                )
            finally:
                await self._script(f"rm -f {shlex.quote(remote_archive)}")
            return
        await self._upload_bytes(source.read_bytes(), target)

    async def download(self, source: str, target: Path) -> None:
        probe = await self._script(f"test -d {shlex.quote(source)}")
        if probe.exit_code == 0:
            remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
            await self._checked(f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source)} .")
            try:
                data, _ = await self._download_bytes(remote_archive)
                target.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                    Path(archive.name).write_bytes(data)
                    with tarfile.open(archive.name, "r:gz") as tar:
                        tar.extractall(target, filter="data")
            finally:
                await self._script(f"rm -f {shlex.quote(remote_archive)}")
            return
        data, _ = await self._download_bytes(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def _drain_in_flight(self) -> None:
        with self.execution.admission:
            if self.execution.admission.wait_for(lambda: not self.execution.in_flight, timeout=CLOSE_DRAIN_SECONDS):
                return
            stuck = [shlex.join(argv)[:STUCK_EXEC_PREVIEW_CHARS] for argv in self.execution.in_flight.values()]
        raise RuntimeError(
            f"{len(stuck)} Iris exec(s) still in flight {CLOSE_DRAIN_SECONDS} s after terminating the sandbox: "
            + "; ".join(stuck)
        )

    async def close(self) -> None:
        with self.execution.admission:
            if self.execution.closed:
                return
            self.execution.closed = True
        try:
            await asyncio.to_thread(self.job.cancel)
            # Terminating the task ends execs the controller already admitted and refuses later ones,
            # so attempts that passed the closed check return; once they have, none can start.
            await asyncio.to_thread(self._drain_in_flight)
        finally:
            try:
                await asyncio.to_thread(self.client.shutdown)
            finally:
                self.endpoint.close()


class IrisMachineFactory:
    """Submit a CPU-only gVisor job from a registry image.

    Secret references are resolved only at job submission. A factory configured
    with secrets must be reserved for trusted private grading, never actor jobs.
    """

    backend: Backend = Backend.GVISOR

    def __init__(
        self,
        *,
        cluster: str | None = None,
        controller_url: str | None = None,
        scheduling_timeout: int = DEFAULT_SCHEDULING_TIMEOUT,
        job_ttl: int = DEFAULT_JOB_TTL,
        disk_mb: int = DEFAULT_DISK_MB,
        secret_env: Mapping[str, SecretSpec] | None = None,
    ):
        if (cluster is None) == (controller_url is None):
            raise ValueError("Specify exactly one Iris cluster or controller URL")
        self.cluster = cluster
        self.controller_url = controller_url
        self.scheduling_timeout = scheduling_timeout
        self.job_ttl = job_ttl
        self.disk_mb = disk_mb
        self.secret_env = dict(secret_env or {})

    def _check_spec(self, spec: MachineSpec) -> RegistryImage:
        validate_sandbox_environment(spec.env)
        validate_sandbox_environment({name: "" for name in self.secret_env})
        if spec.gpus:
            raise UnsupportedMachineSpec("The Iris machine factory does not provide GPU allocation")
        if not isinstance(spec.source, RegistryImage):
            raise UnsupportedMachineSpec("Iris requires a registry image reference")
        return spec.source

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        source = self._check_spec(spec)
        check_execution_requirements(
            requirements,
            spec,
            semantics=CommandSemantics.LINUX_PROCESS,
            capabilities=LINUX_PROCESS_CAPABILITIES,
            image_reference=source.reference,
        )

    async def create(self, spec: MachineSpec) -> IrisMachine:
        return await self._create(spec, self._check_spec(spec))

    async def _create(
        self, spec: MachineSpec, source: RegistryImage, services: tuple[SandboxServiceSpec, ...] = ()
    ) -> IrisMachine:
        submission = asyncio.create_task(asyncio.to_thread(self._submit_sync, spec, source, services))
        try:
            machine = await asyncio.shield(submission)
        except asyncio.CancelledError:
            machine = await submission
            await asyncio.shield(machine.close())
            raise
        try:
            deadline = time.monotonic() + self.scheduling_timeout
            while time.monotonic() < deadline:
                try:
                    status = await asyncio.to_thread(machine.task.status)
                except ConnectError as error:
                    if error.code != Code.NOT_FOUND:
                        raise
                else:
                    if status.state == TaskState.RUNNING:
                        created = await asyncio.to_thread(machine._exec_sync, ["mkdir", "-p", spec.workdir])
                        if created.exit_code:
                            raise RuntimeError(f"Failed to create Iris workdir {spec.workdir}: {created.stderr}")
                        return machine
                    if status.state not in (TaskState.PENDING, TaskState.BUILDING, TaskState.ASSIGNED):
                        raise RuntimeError(f"Iris sandbox task failed before running: {status.error_message}")
                await asyncio.sleep(2)
            raise TimeoutError(f"Iris sandbox did not start within {self.scheduling_timeout} seconds")
        except BaseException:
            await machine.close()
            raise

    async def create_group(self, spec: ServiceGroupSpec) -> ServiceGroup:
        """Start a credential-free task and services in one private gVisor pod."""
        if self.secret_env:
            raise UnsupportedMachineSpec("Private service groups accept no secret-enabled machine factory")
        source = self._check_spec(spec.task)
        services = tuple(
            SandboxServiceSpec(
                name=service.name,
                image=self._check_spec(service.machine).reference,
                workdir=service.machine.workdir,
                env_vars=service.machine.env,
                entrypoint=service.entrypoint,
                arguments=service.arguments,
                resources=ResourceSpec(
                    cpu=service.machine.cpus or 1,
                    memory=(service.machine.memory_mb or DEFAULT_MEMORY_MB) * 1024 * 1024,
                    disk=(service.machine.storage_mb or self.disk_mb) * 1024 * 1024,
                ),
            )
            for service in spec.services
        )
        if not services:
            raise UnsupportedMachineSpec("A service group requires at least one service")
        task = await self._create(spec.task, source, services)
        members = {
            service.name: IrisMachine(
                task.endpoint,
                task.client,
                task.rpc,
                task.job,
                task.task,
                service.machine,
                container_name=service.name,
                execution=task.execution,
            )
            for service in spec.services
        }
        return ServiceGroup(task, members, spec, task.close)

    def _submit_sync(
        self, spec: MachineSpec, source: RegistryImage, services: tuple[SandboxServiceSpec, ...] = ()
    ) -> IrisMachine:
        info = get_job_info()
        if info is not None:
            if (
                self.cluster is not None
                or self.controller_url is None
                or info.controller_address is None
                or self.controller_url.rstrip("/") != info.controller_address.rstrip("/")
            ):
                raise UnsupportedMachineSpec("An Iris task must create sandboxes on its own controller URL")
            credentials = info.credentials
            if credentials is None:
                raise UnsupportedMachineSpec("An Iris task needs its task token to create sandboxes")
            # A task token scopes this driver to its existing identity. Do not
            # consult the host's broader cluster credentials from inside a job.
            endpoint = ControllerEndpoint(url=self.controller_url, credentials=credentials, config=None)
        else:
            endpoint = connect_controller(cluster_name=self.cluster, controller_url=self.controller_url)
        url, credentials = endpoint.url, endpoint.credentials
        client = None
        job = None
        try:
            client = IrisClient.remote(url, workspace=None, credentials=credentials)
            rpc = ControllerServiceClientSync(
                address=url,
                timeout_ms=RPC_PADDING_SECONDS * 1000,
                interceptors=credentials.interceptors() if credentials is not None else [],
                accept_compression=IRIS_RPC_COMPRESSIONS,
                send_compression=None,
            )
            environment = EnvironmentSpec(
                setup_scripts=[],
                env_vars=(
                    dict(spec.env)
                    if services
                    else {name: resolve_secret_spec(ref).value for name, ref in self.secret_env.items()}
                ),
                sandbox_services=services,
            )
            resources = ResourceSpec(
                cpu=(spec.cpus or 1) + sum(service.resources.cpu for service in services),
                memory=(spec.memory_mb or DEFAULT_MEMORY_MB) * 1024 * 1024
                + sum(int(service.resources.memory) for service in services),
                disk=(spec.storage_mb or self.disk_mb) * 1024 * 1024
                + sum(int(service.resources.disk) for service in services),
            )
            if services:
                validate_sandbox_services(
                    job_pb2.CONTAINER_PROFILE_SANDBOX,
                    EGRESS_POLICIES[spec.network],
                    environment.to_sandbox_proto(),
                    resources.to_proto(),
                    source.reference,
                )
                backends = rpc.list_backends(controller_pb2.Controller.ListBackendsRequest()).backends
                if not any(SANDBOX_SERVICE_GROUP_CAPABILITY in backend.capabilities for backend in backends):
                    raise UnsupportedMachineSpec("Controller does not advertise private sandbox service-group support")
            job = client.submit(
                # Process 1 must exit on the stop signal, or a cancelled machine keeps its node
                # capacity for the whole termination grace period.
                entrypoint=Entrypoint.from_command("sh", "-c", IDLE_ENTRYPOINT),
                name=f"shellbox-{uuid.uuid4().hex}",
                environment=environment,
                resources=resources,
                task_image=source.reference,
                container_profile=job_pb2.CONTAINER_PROFILE_SANDBOX,
                egress_policy=EGRESS_POLICIES[spec.network],
                scheduling_timeout=Duration.from_seconds(self.scheduling_timeout),
                timeout=Duration.from_seconds(self.job_ttl),
                max_retries_failure=0,
                max_retries_preemption=0,
            )
            return IrisMachine(endpoint, client, rpc, job, job.task(0), spec)
        except BaseException:
            try:
                if job is not None:
                    job.cancel()
            finally:
                try:
                    if client is not None:
                        client.shutdown()
                finally:
                    endpoint.close()
            raise
