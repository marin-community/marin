# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a registry image with direct primary pipes and unary companion exec."""

import asyncio
import base64
import math
import re
import secrets
import shlex
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cli.connect import ControllerEndpoint
from iris.client import IrisClient, Job, Task
from iris.cluster.client.job_info import get_job_info
from iris.cluster.runtime.sandbox import validate_sandbox_environment
from iris.cluster.types import ContainerSpec, DriverConnectionSpec, Entrypoint, NetworkAttachment, ResourceSpec
from iris.cluster.types import TaskSpec as IrisTaskSpec
from iris.resources.state import TaskState
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.errors import DEFAULT_RETRY_MAX_ATTEMPTS, DEFAULT_RETRY_MAX_ELAPSED
from rigging.timing import Duration, ExponentialBackoff, retry_with_backoff

from shellbox.backends.iris import process_guest
from shellbox.backends.iris.process import CONNECT_TIMEOUT, IrisProcessSession, ProcessConnectionUnavailable
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
from shellbox.process_session import acquire_owned, await_cleanup, finish_owned, own_cleanup
from shellbox.task import Task as ContainerTask
from shellbox.task import TaskSpec, single_container_task

# The worker runs `docker exec` or `kubectl exec` with the command as argv, and Linux caps one
# argument at 128 KiB (MAX_ARG_STRLEN). An upload chunk travels base64-encoded inside one
# `sh -c` script, so 64 KiB of data (87,384 encoded bytes) leaves room for the target path.
TRANSFER_CHUNK_BYTES = 64 * 1024
DEFAULT_MEMORY_MB = 2048
DEFAULT_DISK_MB = 10240
DEFAULT_SCHEDULING_TIMEOUT = 600
STARTUP_TCP_TIMEOUT = 1
STARTUP_RETRY_INTERVAL = 0.1
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
    sessions: set[IrisProcessSession] = field(default_factory=set)
    closing: asyncio.Task[None] | None = None


class IrisMachine:
    """One Iris task container, sharing allocation lifetime with its companions."""

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
        capability: bytes = b"",
        driver: DriverConnectionSpec | None = None,
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
        self._capability = capability
        self._driver = driver
        self._connection: controller_pb2.Controller.GetTaskConnectionResponse | None = None
        self._connecting = asyncio.Lock()

    async def _primary_connection(self) -> controller_pb2.Controller.GetTaskConnectionResponse:
        async with self._connecting:
            if self._connection is not None:
                return self._connection
            if self._driver is None or not self._capability:
                raise UnsupportedMachineSpec("Direct Iris processes require an owning driver task on the Pod network")
            status = await asyncio.to_thread(self.task.status)
            attempt = next(
                (attempt for attempt in status.attempts if attempt.attempt_number == status.current_attempt_number), None
            )
            if attempt is None or not attempt.attempt_uid:
                raise MachineTerminated("Iris sandbox has no current allocation identity")
            response = await asyncio.to_thread(
                self.rpc.get_task_connection,
                controller_pb2.Controller.GetTaskConnectionRequest(
                    driver_task_id=self._driver.driver_task_id.to_wire(),
                    driver_attempt_uid=self._driver.driver_attempt_uid,
                    task_id=self.task.task_id.to_wire(),
                    attempt_uid=attempt.attempt_uid,
                ),
                timeout_ms=RPC_PADDING_SECONDS * 1000,
            )
            if response.attempt_uid != attempt.attempt_uid or not response.pod_uid or response.port != self._driver.port:
                raise MachineTerminated("Iris returned a mismatched sandbox connection identity")
            self._connection = response
            return response

    async def open_process(self, command: Command) -> IrisProcessSession:
        """Open primary-container pipes directly from the owning driver Pod."""
        return await self._open_process(command, tcp_timeout=CONNECT_TIMEOUT)

    async def _open_process(self, command: Command, *, tcp_timeout: float) -> IrisProcessSession:
        if self.container_name:
            raise UnsupportedMachineSpec("Persistent Iris process sessions support the primary container only")
        if not command.argv or command.stdin or command.timeout is not None:
            raise ValueError("Process sessions require argv and own stdin and their lifetime")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        environment = {**self.spec.env, **command.env}
        validate_sandbox_environment(environment)
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in environment):
            raise ValueError("Environment variable names must be shell identifiers")
        if command.user is not None and command.user not in await self._user_identity():
            raise UnsupportedMachineSpec("Iris process sessions cannot switch the container user")

        async def acquire() -> IrisProcessSession:
            if self.execution.closed:
                raise RuntimeError("Machine is closed")
            connection = await self._primary_connection()
            if self.execution.closed:
                raise RuntimeError("Machine is closed")
            session = await IrisProcessSession.connect(
                connection.pod_ip,
                connection.port,
                self._capability,
                replace(command, cwd=command.cwd or self.spec.workdir, env=environment),
                on_close=self.execution.sessions.discard,
                tcp_timeout=tcp_timeout,
            )
            self.execution.sessions.add(session)
            if self.execution.closed:
                await session.close()
                raise MachineTerminated("Machine closed while opening the process")
            return session

        # Acquisition is owned: interruption closes a late socket/process instead
        # of replaying an Open whose command may already have started.
        try:
            return await acquire_owned(asyncio.create_task(acquire()))
        except ProcessConnectionUnavailable:
            # Only a TCP failure proves that no authentication or Open was sent.
            raise
        except BaseException as error:
            try:
                await finish_owned(own_cleanup(self.close()))
            except BaseException as cleanup_error:
                error.add_note(f"Iris sandbox cleanup failed ({type(cleanup_error).__name__})")
            raise

    async def _direct_command(self, command: Command, *, tcp_timeout: float = CONNECT_TIMEOUT) -> Result:
        limit = command.output_limit_bytes
        session = await self._open_process(
            replace(command, stdin=b"", timeout=None, output_limit_bytes=limit + 1), tcp_timeout=tcp_timeout
        )

        async def input_bytes() -> None:
            if command.stdin:
                await session.write(command.stdin)
            await session.write_eof()

        sending = asyncio.create_task(input_bytes())
        output = bytearray()
        try:
            while data := await session.read():
                output.extend(data[: max(0, limit + 1 - len(output))])
            await sending
            code = await session.wait()
            result = Result(
                code,
                bytes(output[:limit]),
                session.stderr[:limit],
                len(output) > limit,
                len(session.stderr) > limit,
                ExitReason.EXITED,
            )
        except BaseException as error:
            try:
                await finish_owned(own_cleanup(session.close()))
            except BaseException as cleanup_error:
                error.add_note(f"Iris process cleanup failed ({type(cleanup_error).__name__})")
            raise
        else:
            await session.close()
            return result
        finally:
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)

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
        if self.container_name:
            return await asyncio.to_thread(self._exec_sync, ["sh", "-c", script], timeout)
        result = await self._direct_command(Command(("sh", "-c", script)))
        assert result.exit_code is not None
        return ExecOutput(result.exit_code, result.stdout.decode(), result.stderr.decode())

    async def _checked(self, script: str) -> str:
        response = await self._script(script)
        if response.exit_code:
            raise RuntimeError(f"Iris command failed ({response.exit_code}): {response.stderr or response.stdout}")
        return response.stdout

    async def _upload_bytes(self, data: bytes, target: str) -> None:
        if not self.container_name:
            result = await self._direct_command(
                Command(
                    (
                        "sh",
                        "-c",
                        f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))} && cat > {shlex.quote(target)}",
                    ),
                    stdin=data,
                )
            )
            if result.exit_code:
                raise RuntimeError(f"Iris upload failed ({result.exit_code})")
            return
        quoted = shlex.quote(target)
        await self._checked(f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))} && : > {quoted}")
        for offset in range(0, len(data), TRANSFER_CHUNK_BYTES):
            encoded = base64.b64encode(data[offset : offset + TRANSFER_CHUNK_BYTES]).decode("ascii")
            await self._checked(f"printf '%s' {encoded} | base64 -d >> {quoted}")

    async def _download_bytes(self, source: str, limit: int | None = None) -> tuple[bytes, bool]:
        quoted = shlex.quote(source)
        size = int((await self._checked(f"wc -c < {quoted}")).strip())
        count = size if limit is None else min(size, limit)
        if not self.container_name:
            result = await self._direct_command(Command(("head", "-c", str(count), source), output_limit_bytes=count))
            if result.exit_code:
                raise RuntimeError(f"Iris download failed ({result.exit_code})")
            return result.stdout, size > count
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
            if self.execution.closing is None:
                self.execution.closed = True
                self.execution.closing = own_cleanup(self._close())
            closing = self.execution.closing
        await await_cleanup(closing)

    async def _close(self) -> None:
        try:
            await asyncio.to_thread(self.job.cancel)
            # Terminating the task ends execs the controller already admitted and refuses later ones,
            # so attempts that passed the closed check return; once they have, none can start.
            await asyncio.to_thread(self._drain_in_flight)
        finally:
            try:
                await asyncio.gather(*(session.close() for session in tuple(self.execution.sessions)))
            finally:
                try:
                    await asyncio.to_thread(self.client.shutdown)
                finally:
                    self.endpoint.close()


class IrisMachineFactory:
    """Submit a CPU-only gVisor job from a registry image."""

    backend: Backend = Backend.GVISOR

    def __init__(
        self,
        *,
        cluster: str | None = None,
        controller_url: str | None = None,
        scheduling_timeout: int = DEFAULT_SCHEDULING_TIMEOUT,
        job_ttl: int = DEFAULT_JOB_TTL,
        disk_mb: int = DEFAULT_DISK_MB,
    ):
        if (cluster is None) == (controller_url is None):
            raise ValueError("Specify exactly one Iris cluster or controller URL")
        self.cluster = cluster
        self.controller_url = controller_url
        self.scheduling_timeout = scheduling_timeout
        self.job_ttl = job_ttl
        self.disk_mb = disk_mb

    def _check_spec(self, spec: MachineSpec) -> RegistryImage:
        validate_sandbox_environment(spec.env)
        if spec.gpus:
            raise UnsupportedMachineSpec("The Iris machine factory does not provide GPU allocation")
        if not isinstance(spec.source, RegistryImage):
            raise UnsupportedMachineSpec("Iris requires a registry image reference")
        return spec.source

    @classmethod
    def in_cluster(
        cls,
        *,
        scheduling_timeout: int = DEFAULT_SCHEDULING_TIMEOUT,
        job_ttl: int = DEFAULT_JOB_TTL,
        disk_mb: int = DEFAULT_DISK_MB,
    ) -> "IrisMachineFactory":
        """Create child sandboxes using the current task's controller and identity."""
        info = get_job_info()
        if info is None or info.controller_address is None:
            raise UnsupportedMachineSpec("Current-controller creation requires an Iris task")
        return cls(
            controller_url=info.controller_address,
            scheduling_timeout=scheduling_timeout,
            job_ttl=job_ttl,
            disk_mb=disk_mb,
        )

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
        task = await self.create_task(single_container_task(spec))
        return cast(IrisMachine, task.primary)

    async def _create(
        self,
        spec: MachineSpec,
        task_spec: IrisTaskSpec,
        capability: bytes,
    ) -> IrisMachine:
        submission = asyncio.create_task(asyncio.to_thread(self._submit_sync, spec, task_spec, capability))
        try:
            machine = await asyncio.shield(submission)
        except asyncio.CancelledError as cancellation:
            try:
                machine = await finish_owned(submission)
                await finish_owned(asyncio.create_task(machine.close()))
            except BaseException as cleanup_error:
                cancellation.add_note(f"Sandbox acquisition cleanup failed ({type(cleanup_error).__name__})")
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
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            async with asyncio.timeout(remaining):
                                ready = await machine._direct_command(
                                    Command(("sh", "-c", "true")), tcp_timeout=min(STARTUP_TCP_TIMEOUT, remaining)
                                )
                        except ProcessConnectionUnavailable:
                            # RUNNING does not prove ingress is usable. Fresh
                            # short dials avoid a stale SYN retransmission wait.
                            await asyncio.sleep(min(STARTUP_RETRY_INTERVAL, max(0, deadline - time.monotonic())))
                            continue
                        if ready.exit_code:
                            raise RuntimeError(f"Iris sandbox readiness command failed ({ready.exit_code})")
                        return machine
                    if status.state not in (TaskState.PENDING, TaskState.BUILDING, TaskState.ASSIGNED):
                        raise RuntimeError(f"Iris sandbox task failed before running: {status.error_message}")
                await asyncio.sleep(2)
            raise TimeoutError(f"Iris sandbox did not start within {self.scheduling_timeout} seconds")
        except BaseException as error:
            try:
                await finish_owned(asyncio.create_task(machine.close()))
            except BaseException as cleanup_error:
                error.add_note(f"Sandbox cleanup failed ({type(cleanup_error).__name__})")
            raise

    async def create_task(self, spec: TaskSpec) -> ContainerTask:
        """Start a singleton or private peer group through the same Iris path."""
        info = get_job_info()
        if info is None or not info.controller_address or not info.attempt_uid or info.credentials is None:
            raise UnsupportedMachineSpec("Direct Iris sandboxes require an in-cluster driver task and attempt token")
        if (
            self.cluster is not None
            or self.controller_url is None
            or self.controller_url.rstrip("/") != info.controller_address.rstrip("/")
        ):
            raise UnsupportedMachineSpec("An Iris task must create sandboxes on its own controller URL")
        if any(endpoint.port == process_guest.PORT for container in spec.containers for endpoint in container.endpoints):
            raise UnsupportedMachineSpec(f"Iris reserves TCP port {process_guest.PORT} for the primary process broker")
        capability = secrets.token_bytes(32)
        primary = spec.primary_container
        companions = tuple(container for container in spec.containers if container.name != spec.primary)
        if primary.entrypoint is not None or primary.arguments is not None:
            raise UnsupportedMachineSpec("The Iris primary container is kept alive for machine commands")
        if companions and spec.network is not NetworkPolicy.DENY:
            raise UnsupportedMachineSpec("Iris multi-container tasks require denied external networking")
        containers = tuple(
            ContainerSpec(
                name=container.name,
                image=self._check_spec(container.machine).reference,
                # OCI runtimes chdir before starting Python, so bootstrap from
                # an existing directory and let the broker create the workdir.
                workdir="/" if container.name == spec.primary else container.machine.workdir,
                env_vars=container.machine.env,
                command=(
                    Entrypoint.from_command(
                        "python3",
                        "-u",
                        "-c",
                        Path(process_guest.__file__).read_text(),
                        capability.hex(),
                        str(process_guest.PORT),
                        container.machine.workdir,
                    )
                    if container.name == spec.primary
                    else None
                ),
                entrypoint=container.entrypoint,
                arguments=container.arguments,
                resources=ResourceSpec(
                    cpu=container.machine.cpus or 1,
                    memory=(container.machine.memory_mb or DEFAULT_MEMORY_MB) * 1024 * 1024,
                    disk=(container.machine.storage_mb or self.disk_mb) * 1024 * 1024,
                ),
            )
            for container in spec.containers
        )
        task = await self._create(
            primary.machine,
            IrisTaskSpec(
                containers=containers,
                primary=spec.primary,
                network_attachment=NetworkAttachment.POD,
                driver_connection=DriverConnectionSpec(info.task_id, info.attempt_uid, process_guest.PORT),
            ),
            capability,
        )
        members = {
            container.name: IrisMachine(
                task.endpoint,
                task.client,
                task.rpc,
                task.job,
                task.task,
                container.machine,
                container_name=container.name,
                execution=task.execution,
            )
            for container in companions
        }
        members[primary.name] = task
        return ContainerTask(members, spec, task.close)

    def _submit_sync(
        self,
        spec: MachineSpec,
        task_spec: IrisTaskSpec,
        capability: bytes,
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
            raise UnsupportedMachineSpec("Direct Iris sandboxes require an in-cluster driver task")
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
            job = client.submit_task(
                task=task_spec,
                name=f"shellbox-{uuid.uuid4().hex}",
                container_profile=job_pb2.CONTAINER_PROFILE_SANDBOX,
                egress_policy=EGRESS_POLICIES[spec.network],
                scheduling_timeout=Duration.from_seconds(self.scheduling_timeout),
                timeout=Duration.from_seconds(self.job_ttl),
                max_retries_failure=0,
                max_retries_preemption=0,
            )
            return IrisMachine(
                endpoint, client, rpc, job, job.task(0), spec, capability=capability, driver=task_spec.driver_connection
            )
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
