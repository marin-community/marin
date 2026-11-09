# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a registry image as a gVisor Iris job and exec into its container."""

import ast
import asyncio
import base64
import logging
import math
import re
import shlex
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

import httpx
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
    ExecSetupFailed,
    ExecutionRequirements,
    ExitReason,
    MachineCancelled,
    MachineResourceExhausted,
    MachineSpec,
    MachineTerminated,
    MachineUnavailable,
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
# How much of each stuck exec's command ``close`` quotes when the drain times out.
STUCK_EXEC_PREVIEW_CHARS = 120

# ALLOW reaches public internet addresses only; neither mode reaches the cluster.
EGRESS_POLICIES = {
    NetworkPolicy.ALLOW: job_pb2.EGRESS_POLICY_INTERNET,
    NetworkPolicy.DENY: job_pb2.EGRESS_POLICY_NONE,
}


def _controller_unavailable(error: Exception) -> bool:
    return (isinstance(error, ConnectError) and error.code == Code.UNAVAILABLE) or isinstance(
        error, (ConnectionError, httpx.ConnectError)
    )


def _exec_transport_failure(error: Exception) -> bool:
    return isinstance(error, (ConnectError, ConnectionError, httpx.TransportError))


def _controller_call[T](call: Callable[[], T], *, exec_call: bool = False) -> T:
    """Retry exec RPC transport failures under the existing Iris availability budget."""
    try:
        # Submit/status already own Iris's bounded retry; never stack another wait on them.
        if not exec_call:
            return call()
        return retry_with_backoff(
            call,
            retryable=_exec_transport_failure,
            max_attempts=DEFAULT_RETRY_MAX_ATTEMPTS,
            max_elapsed=DEFAULT_RETRY_MAX_ELAPSED,
            backoff=EXEC_SHED_BACKOFF,
            operation="Iris controller",
        )
    except Exception as error:
        if _controller_unavailable(error) or (exec_call and _exec_transport_failure(error)):
            raise MachineUnavailable(f"Iris controller unavailable: {error}") from error
        raise


@dataclass(frozen=True)
class ExecOutput:
    """The completed command's exit code and text output from one Iris container exec."""

    exit_code: int
    stdout: str
    stderr: str


RESET = """
import json, os, shutil, signal, sys
from pathlib import Path
processes = {}
for path in Path('/proc').glob('[0-9]*/stat'):
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
        processes[path.parent.name] = (fields[1], fields[19])
    except (FileNotFoundError, ProcessLookupError):
        pass
baseline = json.loads(sys.argv[1]) if sys.argv[1] else [processes, os.listdir('/')]
keep, pid = set(), str(os.getpid())
while pid in processes and pid not in keep:
    keep.add(pid)
    pid = processes[pid][0]
for pid, (_, started) in processes.items():
    if pid not in keep and baseline[0].get(pid, [None, None])[1] != started:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
task_roots = {'/app', '/workspace', '/tests', '/tmp', sys.argv[2]}
roots = set(task_roots)
roots.update('/' + name for name in set(os.listdir('/')) - set(baseline[1]))
for root in roots:
    path = Path(root)
    entries = list(path.iterdir()) if path.is_dir() and not path.is_symlink() else [path]
    for entry in entries:
        if root == '/tmp' and entry.name.startswith('.shellbox'):
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)
    if root in task_roots:
        path.mkdir(parents=True, exist_ok=True)
print(json.dumps(baseline))
"""


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
        self._reset_baseline = ""
        self._closed = False
        self._timeout_available = False
        self._container_user: tuple[str, str] | None = None
        # Guards ``_closed`` and ``_in_flight`` so ``close`` cannot interleave with an exec attempt's
        # check-then-send; it can only wait for attempts that already passed the check.
        self._admission = threading.Condition()
        self._in_flight: dict[str, list[str]] = {}

    def _exec_sync(self, argv: list[str], timeout: float | None = None) -> ExecOutput:
        seconds = math.ceil(timeout) if timeout is not None else DEFAULT_JOB_TTL
        request = controller_pb2.Controller.ExecInContainerRequest(
            task_id=self.task.task_id.to_wire(), command=argv, timeout_seconds=seconds
        )
        timeout_ms = (seconds + RPC_PADDING_SECONDS) * 1000

        def attempt() -> controller_pb2.Controller.ExecInContainerResponse:
            # A timed-out ``run`` closes the machine while this thread may still be retrying.
            with self._admission:
                if self._closed:
                    raise MachineTerminated("Machine is closed")
                attempt_id = uuid.uuid4().hex
                self._in_flight[attempt_id] = argv
            try:
                return self.rpc.exec_in_container(request, timeout_ms=timeout_ms)
            finally:
                with self._admission:
                    del self._in_flight[attempt_id]
                    self._admission.notify_all()

        try:
            response = _controller_call(attempt, exec_call=True)
        except ConnectError as error:
            self._raise_if_terminated(error)
            raise
        detail = response.error or (response.stderr if response.exit_code else "")
        # Contract on controller-owned ValueError text until Iris carries typed exec-status failure.
        status = detail.removeprefix("Non-integer exec status: ")
        prefix = "invalid literal for int() with base 10: "
        if status.startswith(prefix):
            quoted = status.removeprefix(prefix)
            try:
                decoded = ast.literal_eval(quoted)
                if isinstance(decoded, str):
                    detail = decoded
            except (ValueError, SyntaxError):
                # The controller truncates long exec errors, leaving an unterminated repr; keep its text so the
                # transport-failure markers below still classify it instead of surfacing a parser bug.
                if quoted[:1] in ("'", '"'):
                    detail = quoted[1:]
        if status.startswith("invalid literal for int()"):
            error = RuntimeError(f"Iris exec failed: {detail}")
            if status.startswith(prefix) and detail.startswith("Internal error occurred"):
                raise MachineUnavailable(str(error)) from error
            raise error
        if response.error:
            error = RuntimeError(f"Iris exec failed: {detail}")
            self._raise_if_terminated(error)
            if any(
                marker in response.error
                for marker in (
                    "error executing command in container",
                    "cannot exec in a",
                    "not running",
                    "container not found",
                )
            ):
                raise MachineTerminated(str(error)) from error
            # Older controllers flatten Kubernetes transport exceptions into this wire string.
            if (
                response.error.startswith("Websocket setup failed (status 0):")
                or response.error == "'NoneType' object has no attribute 'decode'"
            ):
                raise ExecSetupFailed(str(error)) from error
            raise error
        return ExecOutput(exit_code=response.exit_code, stdout=response.stdout, stderr=response.stderr)

    def _raise_if_terminated(self, cause: Exception) -> None:
        """Raise ``MachineTerminated`` from ``cause`` when the sandbox task is no longer running."""
        status = _controller_call(self.task.status)
        if status.state != TaskState.RUNNING:
            detail = str(status.error_message or cause)
            kind = MachineTerminated
            if "oomkilled" in detail.lower():
                kind = MachineResourceExhausted
                detail += f" (memory request: {self.spec.memory_mb or DEFAULT_MEMORY_MB} MiB)"
            raise kind(f"Iris sandbox task {self.task.task_id} is {status.state}: {detail}") from cause

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

    async def _download_bytes(
        self, source: str, limit: int | None = None, *, offset: int = 0, known_file: bool = False
    ) -> tuple[bytes, bool]:
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError("File offset and limit must be nonnegative")
        quoted = shlex.quote(source)
        inspect = f"wc -c < {quoted}"
        if not known_file:
            inspect = f"if test -d {quoted}; then exit 2; elif test -f {quoted}; then {inspect}; else exit 1; fi"
        probe = await self._script(inspect)
        if probe.exit_code == 2:
            raise IsADirectoryError(source)
        if probe.exit_code == 1:
            raise FileNotFoundError(source)
        if probe.exit_code:
            raise OSError(f"Cannot inspect {source}: {probe.stderr or probe.stdout}")
        size = int(probe.stdout.strip())
        remaining = max(0, size - offset)
        count = remaining if limit is None else min(remaining, limit)
        chunks = []
        for position in range(offset, offset + count, TRANSFER_CHUNK_BYTES):
            length = min(TRANSFER_CHUNK_BYTES, offset + count - position)
            encoded = await self._checked(f"tail -c +{position + 1} {quoted} | head -c {length} | base64")
            chunks.append(base64.b64decode(encoded))
        return b"".join(chunks), remaining > count

    async def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> bytes:
        """Read only the requested byte range, rejecting directories before transfer."""
        try:
            data, _ = await self._download_bytes(path, limit, offset=offset)
        except RuntimeError as error:
            if str(error).startswith("Iris command failed ("):
                raise OSError(str(error)) from error
            raise
        return data

    async def _user_identity(self) -> tuple[str, str]:
        """The ``(uid, name)`` the container runs commands as; Iris exec cannot switch users."""
        if self._container_user is None:
            uid, name = (await self._checked("id -u && id -un")).split()
            self._container_user = (uid, name)
        return self._container_user

    async def run(self, command: Command) -> Result:
        if self._closed:
            raise MachineTerminated("Machine is closed")
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
        timed_command = command.timeout is not None and command.timeout > 0 and self._timeout_available
        argv = shlex.join(command.argv)
        if timed_command:
            argv = f"timeout -k 5 {command.timeout}s sh -c {shlex.quote(argv)}"
        script = (
            f"{exports}\ncd {shlex.quote(command.cwd or self.spec.workdir)} || exit 1\n"
            f"{argv} < {shlex.quote(stdin_path) if command.stdin else '/dev/null'} "
            f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
        )
        try:
            started = time.monotonic()
            backstop = command.timeout if command.timeout is not None else DEFAULT_JOB_TTL
            if timed_command:
                assert command.timeout is not None
                backstop = command.timeout + RPC_PADDING_SECONDS
            rpc_timeout = (command.timeout if command.timeout is not None else DEFAULT_JOB_TTL) + RPC_PADDING_SECONDS
            response = await asyncio.wait_for(self._script(script, timeout=rpc_timeout), timeout=backstop)
            elapsed = time.monotonic() - started
            stdout, stdout_truncated = await self._download_bytes(
                stdout_path, command.output_limit_bytes, known_file=True
            )
            stderr, stderr_truncated = await self._download_bytes(
                stderr_path, command.output_limit_bytes, known_file=True
            )
            # GNU timeout returns 137 if its kill-after escalation was needed.
            expired = timed_command and command.timeout is not None and elapsed >= command.timeout
            # A nested timeout may return 124 before our own budget; this is tool feedback, not machine death.
            reason = (
                ExitReason.TIMED_OUT
                if response.exit_code == 124 or (expired and response.exit_code == 137)
                else ExitReason.EXITED
            )
            return Result(response.exit_code, stdout, stderr, stdout_truncated, stderr_truncated, reason)
        except TimeoutError:
            await self.close()
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError:
            await self.close()
            raise
        finally:
            if not self._closed:
                await self._script(
                    f"rm -f {shlex.quote(stdin_path)} {shlex.quote(stdout_path)} {shlex.quote(stderr_path)}"
                )

    async def reset(self) -> None:
        """Exclusively clear task roots and non-runtime processes; preserve mount points."""
        if self.spec.workdir == "/":
            raise MachineTerminated("Cannot reset the filesystem root as a task workdir")
        try:
            result = await self.run(
                Command(("python3", "-c", RESET, self._reset_baseline, self.spec.workdir), cwd="/", timeout=30)
            )
            if result.exit_code != 0 or result.reason != ExitReason.EXITED:
                raise MachineTerminated(f"Iris reset failed: {result.stderr.decode(errors='replace')}")
            self._reset_baseline = result.stdout.decode().strip()
        except BaseException:
            await self.close()
            raise

    async def upload(self, source: Path, target: str) -> None:
        if source.is_dir():
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                with tarfile.open(archive.name, "w:gz") as tar:
                    tar.add(source, arcname=".")
                remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
                await self._upload_bytes(Path(archive.name).read_bytes(), remote_archive)
            try:
                await self._checked(
                    f"mkdir -p {shlex.quote(target)} && tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target)}"
                )
            finally:
                await self._script(f"rm -f {shlex.quote(remote_archive)}")
            return
        await self._upload_bytes(source.read_bytes(), target)

    async def download(self, source: str, target: Path) -> None:
        try:
            data, _ = await self._download_bytes(source)
        except IsADirectoryError:
            remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
            await self._checked(f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source)} .")
            try:
                data, _ = await self._download_bytes(remote_archive, known_file=True)
                target.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                    Path(archive.name).write_bytes(data)
                    with tarfile.open(archive.name, "r:gz") as tar:
                        tar.extractall(target, filter="data")
            finally:
                await self._script(f"rm -f {shlex.quote(remote_archive)}")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

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
            try:
                await asyncio.to_thread(self.job.cancel)
            except Exception:
                logging.info(
                    "Iris kill job=%s result=failed returned_state=not_provided", self.job.job_id, exc_info=True
                )
                raise
            logging.info("Iris kill job=%s result=acknowledged returned_state=not_provided", self.job.job_id)
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

    fixed_execution_user = True
    """Exec uses the container user, including commands that request setup as root."""

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
        machine = await asyncio.to_thread(self._submit_sync, spec, self._check_spec(spec))
        try:
            deadline = time.monotonic() + self.scheduling_timeout
            while time.monotonic() < deadline:
                try:
                    status = await asyncio.to_thread(_controller_call, machine.task.status)
                except ConnectError as error:
                    if error.code != Code.NOT_FOUND:
                        raise
                else:
                    if status.state == TaskState.RUNNING:
                        created = await asyncio.to_thread(machine._exec_sync, ["mkdir", "-p", spec.workdir])
                        if created.exit_code:
                            raise RuntimeError(f"Failed to create Iris workdir {spec.workdir}: {created.stderr}")
                        probe = await machine._script("command -v timeout", timeout=RPC_PADDING_SECONDS)
                        machine._timeout_available = probe.exit_code == 0
                        return machine
                    if status.state is TaskState.KILLED:
                        raise MachineCancelled(status.error_message.lower() or "sandbox job terminated")
                    if status.state not in (TaskState.PENDING, TaskState.BUILDING, TaskState.ASSIGNED):
                        raise MachineTerminated(f"Iris sandbox task failed before running: {status.error_message}")
                await asyncio.sleep(2)
            raise TimeoutError(f"Iris sandbox did not start within {self.scheduling_timeout} seconds")
        except BaseException as error:
            try:
                await machine.close()
            except Exception as cleanup_error:
                error.add_note(f"Sandbox cleanup failed: {cleanup_error}")
                logging.warning("Sandbox startup cleanup failed", exc_info=cleanup_error)
            raise

    def _submit_sync(self, spec: MachineSpec, source: RegistryImage) -> IrisMachine:
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
            job = _controller_call(
                partial(
                    client.submit,
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
            )
            return IrisMachine(endpoint, client, rpc, job, job.task(0), spec)
        except BaseException as error:
            try:
                try:
                    if job is not None:
                        job.cancel()
                finally:
                    try:
                        if client is not None:
                            client.shutdown()
                    finally:
                        endpoint.close()
            except Exception as cleanup_error:
                error.add_note(f"Sandbox submission cleanup failed: {cleanup_error}")
                logging.warning("Sandbox submission cleanup failed", exc_info=cleanup_error)
            if isinstance(error, ConnectError) and error.code == Code.ALREADY_EXISTS:
                raise MachineUnavailable("Iris sandbox name collision; retry with a fresh sandbox id") from error
            # Contract on controller-owned text until Iris carries typed parent-termination status.
            if (
                isinstance(error, ConnectError)
                and error.code == Code.FAILED_PRECONDITION
                and re.fullmatch(r"Cannot submit job: parent job .+ has terminated \(state=JOB_STATE_\w+\)", str(error))
            ):
                raise MachineCancelled("parent job terminated") from error
            raise
