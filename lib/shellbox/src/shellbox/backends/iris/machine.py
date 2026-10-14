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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cli.connect import ControllerEndpoint, connect_controller
from iris.client import IrisClient, Job, Task
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
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
# Every poll is a controller RPC, and a sandbox waits up to a control tick to start;
# backing off keeps hundreds of starting sandboxes from crowding the controller.
START_POLL_BACKOFF = ExponentialBackoff(initial=1.0, maximum=5.0, factor=1.5)
# The line between the two base64 streams ``run`` frames in its exec output; no base64
# line can equal it.
FRAME_SEPARATOR = "."
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


def _parse_run_frame(response: ExecOutput) -> tuple[int, int, bytes, bytes]:
    """Stream sizes and heads from ``run``'s framed exec output: sizes, stdout, separator, stderr."""
    lines = response.stdout.split("\n")
    try:
        stdout_size, stderr_size = (int(size) for size in lines[0].split())
        separator = lines.index(FRAME_SEPARATOR, 1)
    except ValueError as error:
        raise RuntimeError(f"Iris command output is not framed: {response.stderr or response.stdout[:200]}") from error
    stdout = base64.b64decode("".join(lines[1:separator]))
    stderr = base64.b64decode("".join(lines[separator + 1 :]))
    return stdout_size, stderr_size, stdout, stderr


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
    ):
        self.endpoint = endpoint
        self.client = client
        self.rpc = rpc
        self.job = job
        self.task = task
        self.spec = spec
        self._closed = False
        self._container_user: tuple[str, str] | None = None
        # Guards ``_closed`` and ``_in_flight`` so ``close`` cannot interleave with an exec attempt's
        # check-then-send; it can only wait for attempts that already passed the check.
        self._admission = threading.Condition()
        self._in_flight: dict[str, list[str]] = {}

    def _exec_sync(self, argv: list[str], timeout: float | None = None) -> ExecOutput:
        seconds = math.ceil(timeout) if timeout is not None else -1
        request = controller_pb2.Controller.ExecInContainerRequest(
            task_id=self.task.task_id.to_wire(), command=argv, timeout_seconds=seconds
        )
        timeout_ms = (seconds + RPC_PADDING_SECONDS) * 1000 if seconds >= 0 else DEFAULT_JOB_TTL * 1000

        def attempt() -> controller_pb2.Controller.ExecInContainerResponse:
            # A timed-out ``run`` closes the machine while this thread may still be retrying.
            with self._admission:
                if self._closed:
                    raise RuntimeError("Machine is closed")
                attempt_id = uuid.uuid4().hex
                self._in_flight[attempt_id] = argv
            try:
                return self.rpc.exec_in_container(request, timeout_ms=timeout_ms)
            finally:
                with self._admission:
                    del self._in_flight[attempt_id]
                    self._admission.notify_all()

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
        """Write ``data`` to ``target`` in one exec per chunk; the first also creates the file and its parent."""
        quoted = shlex.quote(target)
        for offset in range(0, max(len(data), 1), TRANSFER_CHUNK_BYTES):
            encoded = base64.b64encode(data[offset : offset + TRANSFER_CHUNK_BYTES]).decode("ascii")
            if offset == 0:
                parent = shlex.quote(str(PurePosixPath(target).parent))
                await self._checked(f"mkdir -p {parent} && printf '%s' {encoded} | base64 -d > {quoted}")
            else:
                await self._checked(f"printf '%s' {encoded} | base64 -d >> {quoted}")

    async def _read_range(self, path: str, start: int, end: int) -> bytes:
        """Bytes ``[start, end)`` of ``path``, one exec per transfer chunk."""
        quoted = shlex.quote(path)
        chunks = []
        for offset in range(start, end, TRANSFER_CHUNK_BYTES):
            length = min(TRANSFER_CHUNK_BYTES, end - offset)
            encoded = await self._checked(f"tail -c +{offset + 1} {quoted} | head -c {length} | base64")
            chunks.append(base64.b64decode(encoded))
        return b"".join(chunks)

    async def _user_identity(self) -> tuple[str, str]:
        """The ``(uid, name)`` the container runs commands as; Iris exec cannot switch users."""
        if self._container_user is None:
            uid, name = (await self._checked("id -u && id -un")).split()
            self._container_user = (uid, name)
        return self._container_user

    async def run(self, command: Command) -> Result:
        if self._closed:
            raise RuntimeError("Machine is closed")
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
        files = shlex.join((stdin_path, stdout_path, stderr_path))
        out, err = shlex.quote(stdout_path), shlex.quote(stderr_path)
        if command.stdin:
            await self._upload_bytes(command.stdin, stdin_path)
        exports = "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in {**self.spec.env, **command.env}.items()
        )
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in {**self.spec.env, **command.env}):
            raise ValueError("Environment variable names must be shell identifiers")
        inline = min(command.output_limit_bytes, TRANSFER_CHUNK_BYTES)
        # One exec runs the command in a subshell whose streams go to files, then frames its
        # own stdout as both stream sizes and the head of each stream in base64. It removes
        # the files unless a stream has more within the limit to read back.
        cleanup = f"rm -f {files}"
        if command.output_limit_bytes > inline:
            cleanup = f'[ "$o" -gt {inline} ] || [ "$e" -gt {inline} ] || {cleanup}'
        script = (
            f"(\n{exports}\ncd {shlex.quote(command.cwd or self.spec.workdir)} || exit 1\n"
            f"{shlex.join(command.argv)}\n) < {shlex.quote(stdin_path) if command.stdin else '/dev/null'} "
            f"> {out} 2> {err}\n"
            "status=$?\n"
            f"o=$(wc -c < {out}) && e=$(wc -c < {err}) || exit 1\n"
            "echo $o $e\n"
            f"head -c {inline} {out} | base64\necho {FRAME_SEPARATOR}\nhead -c {inline} {err} | base64\n"
            f"{cleanup}\n"
            'exit "$status"'
        )
        removed = False
        try:
            response = await asyncio.wait_for(self._script(script), timeout=command.timeout)
            stdout_size, stderr_size, stdout, stderr = _parse_run_frame(response)
            limit = command.output_limit_bytes
            removed = limit <= inline or (stdout_size <= inline and stderr_size <= inline)
            stdout += await self._read_range(stdout_path, len(stdout), min(stdout_size, limit))
            stderr += await self._read_range(stderr_path, len(stderr), min(stderr_size, limit))
            return Result(
                response.exit_code, stdout, stderr, stdout_size > limit, stderr_size > limit, ExitReason.EXITED
            )
        except TimeoutError:
            await self.close()
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError:
            await self.close()
            raise
        finally:
            if not self._closed and not removed:
                await self._script(f"rm -f {files}")

    async def upload(self, source: Path, target: str) -> None:
        if source.is_dir():
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                with tarfile.open(archive.name, "w:gz") as tar:
                    tar.add(source, arcname=".")
                data = Path(archive.name).read_bytes()
            quoted = shlex.quote(target)
            if len(data) <= TRANSFER_CHUNK_BYTES:
                encoded = base64.b64encode(data).decode("ascii")
                await self._checked(f"mkdir -p {quoted} && printf '%s' {encoded} | base64 -d | tar xzf - -C {quoted}")
                return
            remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
            await self._upload_bytes(data, remote_archive)
            try:
                await self._checked(f"mkdir -p {quoted} && tar xzf {shlex.quote(remote_archive)} -C {quoted}")
            finally:
                await self._script(f"rm -f {shlex.quote(remote_archive)}")
            return
        await self._upload_bytes(source.read_bytes(), target)

    async def download(self, source: str, target: Path) -> None:
        quoted = shlex.quote(source)
        remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
        archive = shlex.quote(remote_archive)
        # One exec reads a file source, or archives a directory source and reads the
        # archive: it prints the kind, the size of what it read, and its first chunk.
        kind, size, encoded = (
            await self._checked(
                f"if test -d {quoted}; then tar czf {archive} -C {quoted} . || exit 1; f={archive}; echo dir\n"
                f"else f={quoted}; echo file; fi\n"
                f'wc -c < "$f" && head -c {TRANSFER_CHUNK_BYTES} "$f" | base64'
            )
        ).split("\n", 2)
        try:
            head = base64.b64decode(encoded)
            data = head + await self._read_range(remote_archive if kind == "dir" else source, len(head), int(size))
        finally:
            if kind == "dir":
                await self._script(f"rm -f {archive}")
        if kind == "file":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as local_archive:
            Path(local_archive.name).write_bytes(data)
            with tarfile.open(local_archive.name, "r:gz") as tar:
                tar.extractall(target, filter="data")

    def _drain_in_flight(self) -> None:
        with self._admission:
            if self._admission.wait_for(lambda: not self._in_flight, timeout=CLOSE_DRAIN_SECONDS):
                return
            stuck = [shlex.join(argv)[:STUCK_EXEC_PREVIEW_CHARS] for argv in self._in_flight.values()]
        raise RuntimeError(
            f"{len(stuck)} Iris exec(s) still in flight {CLOSE_DRAIN_SECONDS} s after terminating the sandbox: "
            + "; ".join(stuck)
        )

    async def close(self) -> None:
        with self._admission:
            if self._closed:
                return
            self._closed = True
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
        return await asyncio.to_thread(self._create_sync, spec, self._check_spec(spec))

    def _create_sync(self, spec: MachineSpec, source: RegistryImage) -> IrisMachine:
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
            job = client.submit(
                # Process 1 must exit on the stop signal, or a cancelled machine keeps its node
                # capacity for the whole termination grace period.
                entrypoint=Entrypoint.from_command("sh", "-c", IDLE_ENTRYPOINT),
                name=f"shellbox-{uuid.uuid4().hex}",
                environment=EnvironmentSpec(
                    setup_scripts=[],
                    env_vars={name: resolve_secret_spec(ref).value for name, ref in self.secret_env.items()},
                ),
                resources=ResourceSpec(
                    cpu=spec.cpus or 1,
                    memory=(spec.memory_mb or DEFAULT_MEMORY_MB) * 1024 * 1024,
                    disk=(spec.storage_mb or self.disk_mb) * 1024 * 1024,
                ),
                task_image=source.reference,
                container_profile=job_pb2.CONTAINER_PROFILE_SANDBOX,
                egress_policy=EGRESS_POLICIES[spec.network],
                scheduling_timeout=Duration.from_seconds(self.scheduling_timeout),
                timeout=Duration.from_seconds(self.job_ttl),
                max_retries_failure=0,
                max_retries_preemption=0,
            )
            deadline = time.monotonic() + self.scheduling_timeout
            backoff = START_POLL_BACKOFF.copy()
            while time.monotonic() < deadline:
                # One ListTasks call per poll: its statuses carry the task state.
                statuses = client.list_tasks(job.job_id)
                if statuses:
                    status = statuses[0]
                    if status.state == TaskState.RUNNING:
                        machine = IrisMachine(endpoint, client, rpc, job, Task(client, status.task_id), spec)
                        created = machine._exec_sync(["mkdir", "-p", spec.workdir])
                        if created.exit_code:
                            raise RuntimeError(f"Failed to create Iris workdir {spec.workdir}: {created.stderr}")
                        return machine
                    if status.state not in (TaskState.PENDING, TaskState.BUILDING, TaskState.ASSIGNED):
                        raise RuntimeError(f"Iris sandbox task failed before running: {status.error_message}")
                time.sleep(min(backoff.next_interval(), max(deadline - time.monotonic(), 0)))
            raise TimeoutError(f"Iris sandbox did not start within {self.scheduling_timeout} seconds")
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
