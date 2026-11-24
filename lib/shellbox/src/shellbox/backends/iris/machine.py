# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a registry image as an Iris SANDBOX job and exec into its container."""

import ast
import asyncio
import base64
import errno
import logging
import math
import os
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
    CommandOutcomeUnknown,
    CommandSemantics,
    ExecSetupFailed,
    ExecutionRequirements,
    ExitReason,
    GuestHelperFailed,
    GuestPathError,
    MachineCancelled,
    MachineResourceExhausted,
    MachineSpec,
    MachineTerminated,
    MachineUnavailable,
    NetworkPolicy,
    ProcessScope,
    Result,
    UnsupportedMachineSpec,
    check_execution_requirements,
)

# The worker runs `docker exec` or `kubectl exec` with the command as argv, and Linux caps one
# argument at 128 KiB (MAX_ARG_STRLEN). An upload chunk travels base64-encoded inside one
# `sh -c` script, so 64 KiB of data (87,384 encoded bytes) leaves room for the target path.
TRANSFER_CHUNK_BYTES = 64 * 1024
# Guest path classes, as exit statuses above those sh and coreutils use for their own failures, and their errnos.
PATH_ERRNOS = {101: errno.EISDIR, 102: errno.ENOENT, 103: errno.ENOTDIR, 104: errno.EACCES, 106: errno.ENAMETOOLONG}
# Classifies why "$p" cannot be used, with sh builtins and ``test`` only; falling through means no path fault.
# The walk drops one trailing component per pass, so it ends within the path's depth. ``test`` cannot tell a
# looping link from a dangling one, so both report ENOENT.
CLASSIFY_PATH = (
    'if [ ${{#p}} -ge 4096 ]; then exit 106; fi; if [ -d "$p" ]; then exit 101; fi; '
    'if [ -e "$p" ]; then [ -{access} "$p" ] || exit 104; else a=$p; while :; do '
    'leaf=${{a##*/}}; [ ${{#leaf}} -le 255 ] || exit 106; [ ! -L "$a" ] || exit 102; '
    "case $a in */*) a=${{a%/*}}; a=${{a:-/}};; *) a=.;; esac; "
    '[ ! -e "$a" ] || break; done; [ -d "$a" ] || exit 103; [ -x "$a" ] || exit 104; {missing}; fi; '
)
# Prints the size of a readable file; a directory exits 101 before anything is read.
READ_FILE = 'if [ ! -d "$p" ] && [ -r "$p" ]; then wc -c < "$p"; exit; fi; ' + (
    CLASSIFY_PATH.format(access="r", missing="exit 102") + "exit 1"
)
# Creates or truncates a file, then classifies only a failed attempt: a dangling link may still be writable.
WRITE_FILE = (
    "case $p in */*) d=${p%/*}; d=${d:-/};; *) d=.;; esac; "
    # ``true``, not the special builtin ``:``, whose failed redirection would exit a POSIX shell.
    'if { [ -d "$d" ] || mkdir -p -- "$d"; } && true > "$p"; then exit 0; fi; '
    + CLASSIFY_PATH.format(access="w", missing='[ -w "$a" ] || exit 104')
    + "exit 1"
)

logger = logging.getLogger(__name__)

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


# lib/iris's ``BoundedThreadExecutor`` raises this before the exec handler runs when the controller's exec pool is
# full. Iris attaches no typed detail, so the text is matched exactly; a test pins it against the real executor.
EXEC_ADMISSION_REFUSED = "Blocking RPC capacity exhausted"


def _exec_was_refused(error: BaseException) -> bool:
    """The controller's exec pool refused the exec before its handler ran.

    Only this refusal is retried. Any other RESOURCE_EXHAUSTED may follow execution: the generated client also
    raises it for a reply larger than its configured read limit.
    """
    return (
        isinstance(error, ConnectError)
        and error.code == Code.RESOURCE_EXHAUSTED
        and error.message == EXEC_ADMISSION_REFUSED
    )


def _raise_if_unreachable(error: BaseException) -> None:
    """Report a controller that stayed unreachable through the Iris client's own retry as ``MachineUnavailable``."""
    if isinstance(error, MachineUnavailable):
        return
    if (isinstance(error, ConnectError) and error.code == Code.UNAVAILABLE) or isinstance(
        error, (ConnectionError, httpx.ConnectError)
    ):
        raise MachineUnavailable(f"Iris controller unavailable: {error}") from error


def never_connected(error: BaseException) -> bool:
    """The request failed to open a connection, so the controller never received it."""
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, (httpx.ConnectError, ConnectionRefusedError)):
            return True
        cause = cause.__cause__
    return False


# The Kubernetes client parses the exec status channel as an integer; the controller reports a failure
# there as exit code 1 with this ValueError text, so the command's own outcome is unknown.
UNKNOWN_EXEC_STATUS = "invalid literal for int()"
_UNKNOWN_EXEC_STATUS_PREFIX = UNKNOWN_EXEC_STATUS + " with base 10: "
# Container runtime errors on an exec into a sandbox that is gone, even while its task still reads RUNNING.
DEAD_CONTAINER_MARKERS = (
    "error executing command in container",
    "cannot exec in a",
    "not running",
    "container not found",
)
# The Kubernetes client raises this when the exec websocket upgrade gets no reply. The client sends the command in
# the upgrade request, so the command may have run.
EXEC_SETUP_FAILURE = "'NoneType' object has no attribute 'decode'"
PARENT_TERMINATED = re.compile(r"Cannot submit job: parent job .+ has terminated \(state=JOB_STATE_\w+\)")


def _exec_status_detail(status: str) -> str:
    """The cause quoted in an unknown-status error; the controller may truncate it mid-quote."""
    quoted = status.removeprefix(_UNKNOWN_EXEC_STATUS_PREFIX)
    if quoted == status:
        return status
    try:
        decoded = ast.literal_eval(quoted)
    except (ValueError, SyntaxError):
        return quoted[1:] if quoted[:1] in ("'", '"') else quoted
    return decoded if isinstance(decoded, str) else quoted


def _command_failure(response: controller_pb2.Controller.ExecInContainerResponse) -> str | None:
    """Why an exec reply carries no exit status of the command, or ``None`` when it does."""
    lines = (response.error or (response.stderr if response.exit_code else "")).splitlines()
    if lines and lines[-1].startswith(UNKNOWN_EXEC_STATUS):
        return _exec_status_detail(lines[-1])
    return response.error or None


# Kills processes started since the baseline and empties task roots, keeping mount points and
# ``/tmp/.shellbox*`` capture files. Prints the baseline for the next reset.
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


@dataclass(frozen=True)
class ExecOutput:
    """The completed command's exit code and text output from one Iris container exec."""

    exit_code: int
    stdout: str
    stderr: str


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
        self._container_user: tuple[str, str] | None = None
        # Guards ``_closed`` and ``_in_flight`` so ``close`` cannot interleave with an exec attempt's
        # check-then-send; it can only wait for attempts that already passed the check.
        self._admission = threading.Condition()
        self._in_flight: dict[str, list[str]] = {}

    def _send(self, argv: list[str], timeout: float | None) -> controller_pb2.Controller.ExecInContainerResponse:
        """Send one exec, retrying only the exec pool's admission refusal."""
        seconds = math.ceil(timeout) if timeout is not None else -1
        request = controller_pb2.Controller.ExecInContainerRequest(
            task_id=self.task.task_id.to_wire(), command=argv, timeout_seconds=seconds
        )
        timeout_ms = (seconds + RPC_PADDING_SECONDS) * 1000 if seconds >= 0 else DEFAULT_JOB_TTL * 1000

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

        return retry_with_backoff(
            attempt,
            retryable=_exec_was_refused,
            max_attempts=DEFAULT_RETRY_MAX_ATTEMPTS,
            max_elapsed=DEFAULT_RETRY_MAX_ELAPSED,
            backoff=EXEC_SHED_BACKOFF,
            operation=f"Iris exec in {self.task.task_id}",
        )

    def _exec_sync(self, argv: list[str], timeout: float | None = None) -> ExecOutput:
        """Run a helper exec; ``run`` classifies its own command's exec separately."""
        try:
            response = self._send(argv, timeout)
        except Exception as error:
            _raise_if_unreachable(error)
            if isinstance(error, ConnectError):
                self._raise_if_terminated(error)
            raise
        if response.error == EXEC_SETUP_FAILURE:
            raise ExecSetupFailed(f"Iris exec failed: {response.error}")
        lines = (response.error or (response.stderr if response.exit_code else "")).splitlines()
        if lines and lines[-1].startswith(UNKNOWN_EXEC_STATUS):
            detail = _exec_status_detail(lines[-1])
            error = RuntimeError(f"Iris exec failed: {detail}")
            if detail.startswith("Internal error occurred"):
                raise MachineUnavailable(str(error)) from error
            raise error
        if response.error:
            error = RuntimeError(f"Iris exec failed: {response.error}")
            self._raise_if_terminated(error)
            if any(marker in response.error for marker in DEAD_CONTAINER_MARKERS):
                raise MachineTerminated(str(error)) from error
            raise error
        return ExecOutput(exit_code=response.exit_code, stdout=response.stdout, stderr=response.stderr)

    def _termination(self, cause: object) -> MachineTerminated | None:
        """The error for a sandbox task that is no longer running, or ``None`` while it runs."""
        status = self.task.status()
        if status.state == TaskState.RUNNING:
            return None
        detail = str(status.error_message or cause)
        kind = MachineTerminated
        if "oomkilled" in detail.lower():
            kind = MachineResourceExhausted
            detail += f" (memory request: {self.spec.memory_mb or DEFAULT_MEMORY_MB} MiB)"
        return kind(f"Iris sandbox task {self.task.task_id} is {status.state}: {detail}")

    def _raise_if_terminated(self, cause: Exception) -> None:
        """Raise ``MachineTerminated`` from ``cause`` when the sandbox task is no longer running."""
        try:
            termination = self._termination(cause)
        except Exception as error:
            _raise_if_unreachable(error)
            raise
        if termination is not None:
            raise termination from cause

    def _unattributed_outcome(self, failure: str, container_gone: bool) -> MachineTerminated | CommandOutcomeUnknown:
        """The outcome of a command whose exec was attempted but returned no exit status. Never raises.

        The command may have run, so no outcome is retried on this machine. The status lookup can only report the
        machine gone, so a dead sandbox restarts as terminated, or holds when OOM-killed; its own failure is detail.
        """
        try:
            termination = self._termination(failure)
        except Exception as error:
            termination, failure = None, f"{failure} (sandbox status lookup failed: {error!r})"
        if termination is not None:
            return termination
        if container_gone:
            return MachineTerminated(f"Iris exec failed: {failure}")
        return CommandOutcomeUnknown(f"The command may have run; its result was lost: {failure}")

    async def _script(self, script: str, timeout: float | None = None) -> ExecOutput:
        return await asyncio.to_thread(self._exec_sync, ["sh", "-c", script], timeout)

    async def _checked(self, script: str) -> str:
        response = await self._script(script)
        if response.exit_code:
            raise GuestHelperFailed(f"Iris command failed ({response.exit_code}): {response.stderr or response.stdout}")
        return response.stdout

    async def _path_checked(self, script: str, path: str) -> str:
        """Run a ``READ_FILE`` or ``WRITE_FILE`` script on a caller-chosen ``path``.

        A classified path fault raises ``GuestPathError``; any other failure is the guest helpers'.
        """
        response = await self._script(f"LC_ALL=C; p={shlex.quote(path)}; {script}")
        if (code := PATH_ERRNOS.get(response.exit_code)) is not None:
            raise GuestPathError(code, os.strerror(code), path)
        if response.exit_code:
            raise GuestHelperFailed(f"Iris command failed ({response.exit_code}): {response.stderr or response.stdout}")
        return response.stdout

    async def _upload_bytes(self, data: bytes, target: str, *, caller_path: bool = False) -> None:
        quoted = shlex.quote(target)
        if caller_path:
            await self._path_checked(WRITE_FILE, target)
        else:
            await self._checked(f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))} && : > {quoted}")
        for offset in range(0, len(data), TRANSFER_CHUNK_BYTES):
            encoded = base64.b64encode(data[offset : offset + TRANSFER_CHUNK_BYTES]).decode("ascii")
            await self._checked(f"printf '%s' {encoded} | base64 -d >> {quoted}")

    async def _download_bytes(
        self, source: str, limit: int | None = None, *, caller_path: bool = False
    ) -> tuple[bytes, bool]:
        quoted = shlex.quote(source)
        measured = self._path_checked(READ_FILE, source) if caller_path else self._checked(f"wc -c < {quoted}")
        size = int((await measured).strip())
        count = size if limit is None else min(size, limit)
        chunks = []
        for offset in range(0, count, TRANSFER_CHUNK_BYTES):
            length = min(TRANSFER_CHUNK_BYTES, count - offset)
            encoded = await self._checked(f"tail -c +{offset + 1} {quoted} | head -c {length} | base64")
            chunks.append(base64.b64decode(encoded))
            if len(chunks[-1]) != length:
                # The pipeline reports only its encoder's status; a file that vanished reads as a short chunk.
                if caller_path:
                    await self._path_checked(READ_FILE, source)
                raise GuestHelperFailed(f"Read {len(chunks[-1])} of {length} bytes at offset {offset} of {source}")
        return b"".join(chunks), size > count

    async def _user_identity(self) -> tuple[str, str]:
        """The ``(uid, name)`` the container runs commands as; Iris exec cannot switch users."""
        if self._container_user is None:
            uid, name = (await self._checked("id -u && id -un")).split()
            self._container_user = (uid, name)
        return self._container_user

    async def open_process_scope(self) -> ProcessScope:
        raise UnsupportedMachineSpec("Iris does not provide confirmed scoped descendant teardown")

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
        script = (
            f"{exports}\ncd {shlex.quote(command.cwd or self.spec.workdir)} || exit 1\n"
            f"{shlex.join(command.argv)} < {shlex.quote(stdin_path) if command.stdin else '/dev/null'} "
            f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
        )
        # The command may be retried on this machine only when it was positively never sent: the connection was
        # refused before any request bytes were written, or the exec pool refused admission before its handler ran.
        #   dispatch  never connected, or admission refused  -> MachineUnavailable (retry)
        #             machine closed before admitting it     -> MachineTerminated
        #             deadline                               -> TIMED_OUT, machine closed
        #             any other error                        -> unattributed outcome
        #   decide    reply without the command's exit status, including the WebSocket setup-failure text,
        #             which the Kubernetes client reports after an exec whose upgrade reply was lost
        #                                                    -> unattributed outcome
        #   capture   guest helper failed or sandbox gone    -> GuestHelperFailed / MachineTerminated
        #             any other error                        -> CommandOutcomeUnknown
        #   cleanup   any error or nonzero exit              -> logged; outcome unchanged
        # An unattributed outcome is CommandOutcomeUnknown, or MachineTerminated once the sandbox is gone.
        try:
            try:
                # An exec without a positive deadline falls back to the controller's short Kubernetes default.
                sent = asyncio.to_thread(self._send, ["sh", "-c", script], command.timeout)
                response = await asyncio.wait_for(sent, timeout=command.timeout)
            except TimeoutError:
                await self.close()
                return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
            except MachineTerminated:
                raise  # The machine closed before admitting the exec.
            except Exception as error:
                if _exec_was_refused(error) or never_connected(error):
                    raise MachineUnavailable(f"Iris exec was never sent: {error}") from error
                raise await asyncio.to_thread(self._unattributed_outcome, repr(error), False) from error
            if (failure := _command_failure(response)) is not None:
                gone = any(marker in response.error for marker in DEAD_CONTAINER_MARKERS)
                raise await asyncio.to_thread(self._unattributed_outcome, failure, gone)
            try:
                stdout, stdout_truncated = await self._download_bytes(stdout_path, command.output_limit_bytes)
                stderr, stderr_truncated = await self._download_bytes(stderr_path, command.output_limit_bytes)
            except (GuestHelperFailed, MachineTerminated):
                raise  # Known outcomes: a broken guest helper holds, and a dead or OOM-killed sandbox keeps its kind.
            except Exception as error:
                raise CommandOutcomeUnknown(f"The command ran; its output was lost: {error!r}") from error
            return Result(response.exit_code, stdout, stderr, stdout_truncated, stderr_truncated, ExitReason.EXITED)
        except asyncio.CancelledError:
            await self.close()
            raise
        finally:
            if not self._closed:
                paths = shlex.join((stdin_path, stdout_path, stderr_path))
                try:
                    removed = await self._script(f"rm -f {paths}")
                except Exception as error:
                    logger.warning("Could not remove Iris command capture files %s: %s", paths, error)
                else:
                    if removed.exit_code:
                        detail = f"(exit {removed.exit_code}): {removed.stderr or removed.stdout}"
                        logger.warning("Could not remove Iris command capture files %s %s", paths, detail)

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
        await self._upload_bytes(source.read_bytes(), target, caller_path=True)

    async def download_file(self, source: str, target: Path) -> None:
        """Download one file; a directory raises ``GuestPathError`` with ``EISDIR`` before any transfer."""
        data, _ = await self._download_bytes(source, caller_path=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def download(self, source: str, target: Path) -> None:
        try:
            await self.download_file(source, target)
            return
        except GuestPathError as error:
            if error.errno != errno.EISDIR:
                raise
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
    """Submit a CPU-only Iris SANDBOX job from a registry image.

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
        machine = await asyncio.to_thread(self._submit_sync, spec, self._check_spec(spec))
        try:
            deadline = time.monotonic() + self.scheduling_timeout
            while time.monotonic() < deadline:
                try:
                    status = await asyncio.to_thread(machine.task.status)
                except ConnectError as error:
                    _raise_if_unreachable(error)
                    if error.code != Code.NOT_FOUND:
                        raise
                else:
                    if status.state == TaskState.RUNNING:
                        created = await asyncio.to_thread(machine._exec_sync, ["mkdir", "-p", spec.workdir])
                        if created.exit_code:
                            raise RuntimeError(f"Failed to create Iris workdir {spec.workdir}: {created.stderr}")
                        return machine
                    if status.state is TaskState.KILLED:
                        raise MachineCancelled(f"Iris sandbox task was killed before running: {status.error_message}")
                    if status.state not in (TaskState.PENDING, TaskState.BUILDING, TaskState.ASSIGNED):
                        raise MachineTerminated(f"Iris sandbox task failed before running: {status.error_message}")
                await asyncio.sleep(2)
            raise TimeoutError(f"Iris sandbox did not start within {self.scheduling_timeout} seconds")
        except BaseException as error:
            # A failed cancel must not hide why the sandbox never started.
            try:
                await machine.close()
            except Exception as cleanup_error:
                error.add_note(f"Sandbox cleanup failed: {cleanup_error}")
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
            return IrisMachine(endpoint, client, rpc, job, job.task(0), spec)
        except BaseException as error:
            try:
                if job is not None:
                    job.cancel()
            finally:
                try:
                    if client is not None:
                        client.shutdown()
                finally:
                    endpoint.close()
            _raise_if_unreachable(error)
            if isinstance(error, ConnectError) and error.code == Code.ALREADY_EXISTS:
                raise MachineUnavailable("Iris sandbox name collision; retry with a fresh sandbox id") from error
            # Controller-owned text until Iris reports parent termination as a typed status.
            if (
                isinstance(error, ConnectError)
                and error.code == Code.FAILED_PRECONDITION
                and PARENT_TERMINATED.fullmatch(error.message)
            ):
                raise MachineCancelled("parent job terminated") from error
            raise
