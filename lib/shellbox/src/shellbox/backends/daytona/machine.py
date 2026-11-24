# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the machine contract on a Daytona v2 sandbox."""

import asyncio
import errno
import json
import logging
import math
import os
import re
import shlex
import stat
import tarfile
import tempfile
import uuid
from asyncio import sleep
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from http import HTTPStatus
from pathlib import Path, PurePosixPath

from daytona import (
    AsyncDaytonaV2Client,
    DaytonaConflictError,
    DaytonaConnectionError,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaTimeoutError,
)
from daytona.v2 import DISABLED, AsyncSandbox, LifecyclePolicyOptions, LimitExceededError, NetworkPolicyInput, Resources
from rigging.timing import ExponentialBackoff

from shellbox.backends.daytona.images import ensure_image, image_request_for_spec, validate_source
from shellbox.backends.daytona.process import DaytonaProcessSession
from shellbox.backends.daytona.transient import is_ambiguous_daytona_error, is_transient_daytona_error, retry_transient
from shellbox.backends.docker.machine import INTERRUPT_TIMEOUT, KILL_PROCESS_GROUP_COMMAND, START_COMMAND
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    Backend,
    Command,
    CommandOutputLost,
    CommandSemantics,
    ExecutionRequirements,
    ExitReason,
    MachineDiskFull,
    MachineSpec,
    MachineTerminated,
    NetworkPolicy,
    ProcessScope,
    Result,
    UnsupportedMachineSpec,
    check_execution_requirements,
)
from shellbox.process_session import finish_owned, own_cleanup

DEFAULT_SANDBOX_TTL_MINUTES = 360
DEFAULT_IDLE_TIMEOUT_MINUTES = 30
DEFAULT_OUTPUT_COLLECTION_TIMEOUT = 60
logger = logging.getLogger(__name__)
CONTROL_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
CONTROL_CWD = "/"
COUNT_PREFIX = "SHELLBOX_COUNT="
EXIT_PREFIX = "SHELLBOX_EXIT="
GONE_MARKER = "SHELLBOX_GONE"
CLAIMED_MARKER = "SHELLBOX_CLAIMED"
CLIPPED_OUTPUT_SUFFIX = ".limited"
PROVIDER_DIAGNOSTIC_CHAR_LIMIT = 4096
DISK_FULL_MESSAGE = os.strerror(errno.ENOSPC)
STARTED_STATE = "started"
# A v2 sandbox CPU request is in cores, and one core is two vCPUs; MachineSpec.cpus counts vCPUs.
VCPUS_PER_CORE = 2
# The shape of Daytona v0's default sandbox, used when a spec requests no CPU or memory.
DEFAULT_CPUS = 1
DEFAULT_MEMORY_MB = 1024
# The exit trap records the command's status, so a request that lost its connection can still read it.
RUN_COMMAND = (
    'pidfile=$1; completed=$2; shift 2; echo $$ > "$pidfile"; '
    'trap \'echo $? > "$completed"; rm -f "$pidfile"\' EXIT; "$@"'
)
STOP_COMMAND = (
    "pidfile=$1; completed=$2; "
    '[ ! -f "$completed" ] || exit 0; '
    'if read -r pid < "$pidfile"; then '
    'case "$pid" in ""|*[!0-9]*) exit 1;; esac; '
    '[ "$pid" -gt 1 ] || exit 1; set -- "$pid"; ' + KILL_PROCESS_GROUP_COMMAND + ' && exit 0; fi; [ -f "$completed" ]'
)


@dataclass(frozen=True)
class ExecOutput:
    """A provider command that exited, with its interleaved stdout and stderr."""

    exit_code: int
    output: str


class DaytonaCommandStopped(RuntimeError):
    """The provider stopped a command before it exited, for example at its execution timeout."""


def _control_argv(script: str) -> tuple[str, ...]:
    return (
        "/bin/sh",
        "-c",
        f"_SHELLBOX_COMMAND_PATH=$PATH; export _SHELLBOX_COMMAND_PATH; PATH={CONTROL_PATH}; export PATH; {script}",
    )


async def _exec(
    sandbox: AsyncSandbox,
    argv: Sequence[str],
    *,
    cwd: str | None,
    env: Mapping[str, str] | None = None,
    execution_timeout: float | None = None,
) -> ExecOutput:
    """Run ``argv`` without a shell and wait for it to exit."""
    result = await sandbox.process.run(argv=argv, cwd=cwd, env=env, execution_timeout=execution_timeout)
    if result.exit_code is None:
        raise DaytonaCommandStopped(f"Daytona stopped the command: reason={result.reason} signal={result.signal}")
    return ExecOutput(result.exit_code, result.output)


def sandbox_resources(spec: MachineSpec) -> Resources:
    """Request a fixed CPU and memory shape, so the guest sees what it was given.

    Without limits, v2 boots a burstable sandbox whose ``nproc`` reports the
    burst ceiling, so an unset request takes v0's default shape. Disk is only
    requested: v2 raises disk to a platform minimum and refuses a smaller limit.
    """
    if spec.gpus:
        raise UnsupportedMachineSpec("Daytona v2 Linux sandboxes do not provide GPUs")
    cores = (DEFAULT_CPUS if spec.cpus is None else spec.cpus) / VCPUS_PER_CORE
    memory = (DEFAULT_MEMORY_MB if spec.memory_mb is None else spec.memory_mb) / 1024
    disk = None if spec.storage_mb is None else spec.storage_mb / 1024
    return Resources(cpu=cores, cpu_max=cores, memory=memory, memory_max=memory, disk=disk)


async def _create_sandbox(
    client: AsyncDaytonaV2Client,
    image: str,
    *,
    name: str,
    timeout: float,
    env: Mapping[str, str],
    labels: Mapping[str, str],
    resources: Resources,
    lifecycle_policy: LifecyclePolicyOptions,
    network_policy: NetworkPolicyInput,
) -> AsyncSandbox:
    """Create a sandbox, retrying throttling and lost requests, and return it in whatever state it reached.

    A named v2 create is get-or-create, so repeating the same request after a
    lost response or a throttle returns the sandbox that was accepted. A full
    sandbox limit is not a throttle and raises at once.
    """
    backoff = ExponentialBackoff(initial=1, maximum=30, factor=2, jitter=0.5)
    budget = math.ceil(timeout)
    while True:
        try:
            sandbox = await client.create(
                image,
                name=name,
                env=env,
                labels=labels,
                resources=resources,
                user="root",
                lifecycle_policy=lifecycle_policy,
                network_policy=network_policy,
                wait_seconds=budget,
                create_timeout_seconds=budget,
            )
        except LimitExceededError:
            raise
        except DaytonaRateLimitError as error:
            delay = max(error.retry_after_seconds or 0, backoff.next_interval())
            logger.warning("Retrying throttled Daytona creation: delay=%.3f", delay)
            await sleep(delay)
            continue
        except DaytonaError as error:
            # A named create is idempotent, so a request whose answer was lost is safe to repeat.
            if not is_ambiguous_daytona_error(error):
                raise
            delay = backoff.next_interval()
            logger.warning("Retrying Daytona creation after %s: delay=%.3f", type(error).__name__, delay)
            await sleep(delay)
            continue
        return sandbox


async def _keep_active(sandbox: AsyncSandbox, interval: float) -> None:
    """Reset the idle timer while the machine is open; running guest processes do not reset it."""
    while True:
        await sleep(interval)
        try:
            async with asyncio.timeout(INTERRUPT_TIMEOUT):
                await sandbox.reset_idle_timer()
        except (DaytonaError, TimeoutError) as error:
            logger.warning("Cannot reset Daytona sandbox idle timer: %r", error)


async def _stop_keepalive(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@dataclass
class _SandboxClaim:
    """The sandbox one create owns: its handle, or whether an unanswered create may have made it."""

    name: str
    sandboxes: list[AsyncSandbox] = field(default_factory=list)
    # False once the create returned a handle or was refused outright.
    unanswered: bool = True


async def _delete_sandbox(client: AsyncDaytonaV2Client, claim: _SandboxClaim) -> None:
    """Delete the claimed sandbox, retrying gateway errors and state changes until the interrupt deadline.

    A create that never answered may still have been accepted. Sandbox search
    is eventually consistent, so its caller-owned name is looked up until the
    deadline before cleanup gives up; the lifecycle policy reclaims a sandbox
    that appears later. When the deadline expires after a provider error, raise
    that error rather than a bare timeout.
    """
    name, created = claim.name, claim.sandboxes
    if not created and not claim.unanswered:
        return
    backoff = ExponentialBackoff(initial=1, maximum=4, factor=2, jitter=0)
    retried: DaytonaError | None = None
    try:
        async with asyncio.timeout(INTERRUPT_TIMEOUT):
            while True:
                try:
                    if not created:
                        created.extend(await client.find_by_name(name))
                    for sandbox in created:
                        await client.delete(sandbox, timeout=INTERRUPT_TIMEOUT)
                    if created:
                        return
                except DaytonaNotFoundError:
                    return
                except DaytonaConflictError as error:
                    # Daytona refuses deletion while the sandbox is still changing state.
                    logger.warning("Retrying Daytona sandbox deletion during a state change: %s: %s", name, error)
                    retried = error
                except DaytonaError as error:
                    if not is_transient_daytona_error(error):
                        raise
                    logger.warning(
                        "Retrying Daytona sandbox deletion after %s: %s",
                        f"HTTP {error.status_code}" if error.status_code else type(error).__name__,
                        name,
                    )
                    retried = error
                await sleep(backoff.next_interval())
    except TimeoutError:
        if retried is not None:
            raise retried from None
        if created:
            raise
        logger.warning("No Daytona sandbox named %s appeared before cleanup ended", name)


class DaytonaNetworkMode(StrEnum):
    BLOCK_ALL = "block_all"
    UNRESTRICTED = "unrestricted"
    NETWORK_ALLOW_LIST = "network_allow_list"
    DOMAIN_ALLOW_LIST = "domain_allow_list"


@dataclass(frozen=True)
class DaytonaNetworkPolicy:
    mode: DaytonaNetworkMode
    value: str | None = None

    @classmethod
    def from_config(cls, policy: Mapping[str, str]) -> "DaytonaNetworkPolicy":
        """Parse a declarative policy, rejecting unknown fields."""
        if policy.keys() - {"mode", "value"}:
            raise ValueError("Unknown Daytona network policy fields")
        return cls(DaytonaNetworkMode(policy["mode"]), policy.get("value"))

    def parameters(self) -> NetworkPolicyInput:
        """Return the provider's egress policy; an allow list (comma-separated) blocks everything else."""
        mode = DaytonaNetworkMode(self.mode)
        if mode in (DaytonaNetworkMode.NETWORK_ALLOW_LIST, DaytonaNetworkMode.DOMAIN_ALLOW_LIST):
            if not self.value:
                raise ValueError(f"{self.mode} requires an allow list")
            entries = [entry.strip() for entry in self.value.split(",") if entry.strip()]
            if mode == DaytonaNetworkMode.NETWORK_ALLOW_LIST:
                return {"block_all": True, "allow_cidrs": entries}
            return {"block_all": True, "allow_domains": entries}
        if self.value is not None:
            raise ValueError(f"{self.mode} does not accept an allow list")
        return {"block_all": mode == DaytonaNetworkMode.BLOCK_ALL}


class DaytonaOutputUnavailable(ConnectionError):
    """A command ran, but its output could not be read back from the sandbox."""


class DaytonaMachine:
    """One Daytona sandbox. Each command gets a fresh process and shared files."""

    def __init__(
        self,
        sandbox: AsyncSandbox,
        spec: MachineSpec,
        resources: AsyncExitStack,
        *,
        output_collection_timeout: float = DEFAULT_OUTPUT_COLLECTION_TIMEOUT,
    ):
        self.sandbox = sandbox
        self.spec = spec
        self.resources = resources
        self.output_collection_timeout = output_collection_timeout
        self._closed = False
        # Why the machine closed itself, after which its sandbox is in an unknown state.
        self._lost: BaseException | None = None
        self._sessions: set[DaytonaProcessSession] = set()
        self._acquisitions: set[asyncio.Task[None]] = set()
        self._closing: asyncio.Task[None] | None = None

    def _require_open(self) -> None:
        """Raise ``MachineTerminated`` once the machine closed itself after a failure; using it after its owner
        closed it is a ``RuntimeError``."""
        if not self._closed:
            return
        if self._lost is not None:
            raise MachineTerminated(f"The Daytona machine was closed after {self._lost!r}") from self._lost
        raise RuntimeError("Machine is closed")

    async def _close_after(self, failure: BaseException) -> None:
        self._lost = self._lost or failure
        await self.close()

    async def _control(self, script: str) -> ExecOutput:
        """Run a root command that is safe to run twice, repeating a request that failed in transit."""
        return await retry_transient(lambda: _exec(self.sandbox, _control_argv(script), cwd=CONTROL_CWD))

    async def _user_argv(self, argv: tuple[str, ...], user: str | None) -> tuple[str, ...]:
        if user in (None, "root", "0"):
            return argv
        assert user is not None
        if user.isdecimal():
            account = await self._control(f"getent passwd {shlex.quote(user)}")
            if account.exit_code or not account.output.strip():
                raise ValueError(f"Execution user {user} has no guest account")
            user = account.output.split(":", 1)[0]
        capabilities = await self._control("su --help")
        if capabilities.exit_code or "--session-command" not in capabilities.output:
            raise UnsupportedMachineSpec("Non-root Daytona commands require util-linux su --session-command")
        return ("su", "-s", "/bin/sh", "-m", user, "--session-command", shlex.join(argv))

    async def open_process(self, command: Command) -> DaytonaProcessSession:
        """Attach persistent binary pipes to one provider process."""
        self._require_open()
        if not command.argv or command.stdin or command.timeout is not None:
            raise ValueError("Process sessions require argv and own stdin and their lifetime")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")

        async def cleanup_failed() -> None:
            if not self._closed:
                await self._close_after(MachineTerminated("Cannot delete a Daytona process session"))

        session = DaytonaProcessSession(
            self.sandbox.process,
            f"shellbox-{uuid.uuid4().hex}",
            stderr_limit=command.output_limit_bytes,
            release=self._sessions.discard,
            failed_cleanup=cleanup_failed,
            provider_failures=self._provider_failures,
        )
        self._sessions.add(session)

        async def start() -> None:
            async with asyncio.timeout(INTERRUPT_TIMEOUT):
                argv = await self._user_argv(command.argv, command.user)
                await session.start(argv, command.cwd or self.spec.workdir or None, {**self.spec.env, **command.env})

        acquisition = asyncio.create_task(start())
        self._acquisitions.add(acquisition)
        try:
            await asyncio.shield(acquisition)
            self._require_open()
            return session
        except BaseException as error:
            try:
                await finish_owned(acquisition)
            except BaseException as acquisition_error:
                if acquisition_error is not error:
                    error.add_note(f"Daytona process acquisition failed ({type(acquisition_error).__name__})")
            try:
                await finish_owned(own_cleanup(session.close()))
            except Exception as cleanup_error:
                error.add_note(f"Daytona process cleanup failed ({type(cleanup_error).__name__})")
            raise
        finally:
            self._acquisitions.discard(acquisition)

    async def _command_result(self, exit_code: int, stdout_path: str, stderr_path: str, limit: int) -> Result:
        # Measure and cap both streams in a single provider round trip. The files
        # stay under the command directory until its normal cleanup.
        count = await self._control(
            "measure_and_cap() { "
            'size=$(wc -c < "$1") || return; '
            'if [ "$size" -gt "$2" ] && [ "$2" -gt 0 ]; then '
            f'head -c "$2" "$1" > "$1{CLIPPED_OUTPUT_SUFFIX}" || return; '
            'fi; printf "%d" "$size"; }; '
            f"out=$(measure_and_cap {shlex.quote(stdout_path)} {limit}) && "
            f"err=$(measure_and_cap {shlex.quote(stderr_path)} {limit}) && "
            f'printf \'{COUNT_PREFIX}%s %s\\n\' "$out" "$err"'
        )
        if count.exit_code:
            raise CommandOutputLost(f"Failed to measure command output: {count.output}")
        match = re.search(rf"(?m)^{COUNT_PREFIX}([0-9]+) ([0-9]+)$", count.output)
        if match is None:
            raise DaytonaOutputUnavailable(f"Invalid command output sizes: {count.output!r}")

        stdout_size, stderr_size = map(int, match.groups())
        stdout, stdout_truncated = await self._download_output(stdout_path, stdout_size, limit)
        stderr, stderr_truncated = await self._download_output(stderr_path, stderr_size, limit)
        return Result(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            reason=ExitReason.EXITED,
        )

    async def _download_output(self, path: str, size: int, limit: int) -> tuple[bytes, bool]:
        """Return the bounded stream bytes and whether any bytes were dropped."""
        truncated = size > limit
        if not size or not limit:
            return b"", truncated
        source = f"{path}{CLIPPED_OUTPUT_SUFFIX}" if truncated else path
        return await retry_transient(lambda: self.sandbox.fs.read_bytes(source)), truncated

    async def _exit_after_lost_request(
        self, request: Callable[[], Awaitable[ExecOutput]], pidfile: str, completed: str
    ) -> tuple[int, str]:
        """Repeat a command request that failed in transit and return the command's exit code and diagnostic."""
        response = await retry_transient(request)
        if CLAIMED_MARKER not in response.output.splitlines():
            # No earlier request started the command, so this response is the command's own.
            return response.exit_code, response.output
        backoff = ExponentialBackoff(initial=0.5, maximum=5, factor=2, jitter=0)
        pidfile, completed = shlex.quote(pidfile), shlex.quote(completed)
        # The exit trap writes the exit code before it removes the pidfile. A pidfile that names a dead
        # process next to an empty exit code means the command shell was killed, so no exit code will come.
        probe = (
            f"if [ -s {completed} ]; then printf '{EXIT_PREFIX}%s\\n' \"$(cat {completed})\"; "
            f'elif {{ read -r pid < {pidfile}; }} 2>/dev/null && ! kill -0 "$pid" 2>/dev/null '
            f"&& [ ! -s {completed} ]; then echo {GONE_MARKER}; fi"
        )
        while True:
            status = (await self._control(probe)).output
            match = re.search(rf"(?m)^{EXIT_PREFIX}([0-9]+)$", status)
            if match is not None:
                return int(match.group(1)), ""
            if re.search(rf"(?m)^{GONE_MARKER}$", status):
                raise RuntimeError("Daytona command ended without an exit code after its request was lost")
            await sleep(backoff.next_interval())

    async def _stop_command(self, pidfile: str, completed_path: str) -> None:
        async with asyncio.timeout(INTERRUPT_TIMEOUT):
            stopped = await _exec(
                self.sandbox,
                _control_argv(shlex.join(("/bin/sh", "-c", STOP_COMMAND, "stop-command", pidfile, completed_path))),
                cwd=CONTROL_CWD,
            )
            if stopped.exit_code:
                raise RuntimeError(stopped.output)

    async def _remove_command_files(self, prefix: str) -> None:
        async with asyncio.timeout(INTERRUPT_TIMEOUT):
            removed = await self._control(f"rm -rf {shlex.quote(prefix)}")
            if removed.exit_code:
                raise RuntimeError("Cannot remove Daytona command files")

    async def _close_preserving_failure(self, failure: BaseException) -> None:
        try:
            await self._close_after(failure)
        except Exception as cleanup_error:
            failure.add_note(f"Daytona machine cleanup failed: {cleanup_error!r}")

    async def _cleanup_cancelled_command(
        self,
        prefix: str,
        pidfile: str,
        completed_path: str,
        launch_requested: bool,
        cancellation: asyncio.CancelledError,
    ) -> None:
        try:
            if launch_requested:
                await finish_owned(own_cleanup(self._stop_command(pidfile, completed_path)))
            # Cancellation during staging leaves only private files. Keeping the machine
            # available lets the verifier grade work from commands that already finished.
            await finish_owned(own_cleanup(self._remove_command_files(prefix)))
        except BaseException as cleanup_error:
            cancellation.add_note(f"Daytona command cancellation cleanup failed ({type(cleanup_error).__name__})")
            try:
                await finish_owned(own_cleanup(self._close_after(cleanup_error)))
            except BaseException as close_error:
                cancellation.add_note(f"Daytona machine cleanup failed ({type(close_error).__name__})")

    @asynccontextmanager
    async def _provider_failures(self) -> AsyncIterator[None]:
        """Raise a lost provider connection, or a started sandbox its proxy cannot reach, as ``ConnectionError``,
        and any failure of a sandbox that is no longer started as ``MachineTerminated``."""
        try:
            yield
        except (DaytonaConnectionError, DaytonaTimeoutError) as error:
            raise ConnectionError(f"Lost the connection to Daytona sandbox {self.sandbox.id}: {error}") from error
        except DaytonaError as error:
            try:
                async with asyncio.timeout(INTERRUPT_TIMEOUT):
                    await self.sandbox.refresh_data()
            except DaytonaNotFoundError:
                raise MachineTerminated(f"Daytona sandbox {self.sandbox.id} no longer exists: {error}") from error
            except (DaytonaError, TimeoutError) as probe_error:
                raise ConnectionError(
                    f"Cannot check Daytona sandbox {self.sandbox.id} after {error}: {probe_error!r}"
                ) from error
            if self.sandbox.state != STARTED_STATE:
                raise MachineTerminated(f"Daytona sandbox {self.sandbox.id} is {self.sandbox.state}: {error}") from error
            if error.status_code is None or error.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
                raise ConnectionError(f"Daytona cannot reach sandbox {self.sandbox.id}: {error}") from error
            raise

    async def open_process_scope(self) -> ProcessScope:
        raise UnsupportedMachineSpec("Daytona does not provide confirmed scoped descendant teardown")

    async def run(self, command: Command) -> Result:
        self._require_open()
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        prefix = f"/tmp/.shellbox-{uuid.uuid4().hex}"
        stdin_path, stdout_path, stderr_path = (f"{prefix}/{part}" for part in ("in", "out", "err"))
        script_path = f"{prefix}/command.sh"
        pidfile, completed_path, claim_path = f"{prefix}/pid", f"{prefix}/completed", f"{prefix}/claimed"
        launch_requested = False
        primary_error = None
        exit_code = None
        diagnostic = ""
        environment = {**self.spec.env, **command.env}
        try:
            async with self._provider_failures():
                async with asyncio.timeout(command.timeout) as deadline:
                    argv = (
                        "/bin/sh",
                        "-c",
                        'PATH=$_SHELLBOX_COMMAND_PATH; unset _SHELLBOX_COMMAND_PATH; export PATH; exec "$@"',
                        "shellbox-agent",
                        *command.argv,
                    )
                    argv = await self._user_argv(argv, command.user)
                    argv = (
                        "/bin/sh",
                        "-c",
                        START_COMMAND,
                        "shellbox-start",
                        "/bin/sh",
                        "-c",
                        RUN_COMMAND,
                        "shellbox-command",
                        pidfile,
                        completed_path,
                        *argv,
                    )
                    # Only the first request to claim the directory runs the command.
                    script = (
                        f"mkdir {shlex.quote(claim_path)} 2>/dev/null || {{ echo {CLAIMED_MARKER}; exit 0; }}; "
                        f"{shlex.join(argv)} < {shlex.quote(stdin_path) if command.stdin else '/dev/null'} "
                        f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
                    )
                    # The name is random, so `-p` only makes a repeated request succeed.
                    prepared = await self._control(f"umask 077; mkdir -p {shlex.quote(prefix)}")
                    if prepared.exit_code and DISK_FULL_MESSAGE in prepared.output:
                        raise MachineDiskFull(f"Daytona sandbox {self.sandbox.id} is full: {prepared.output}")
                    if prepared.exit_code:
                        raise RuntimeError(f"Cannot create private command directory: {prepared.output}")
                    if command.stdin:
                        stdin = command.stdin
                        await retry_transient(lambda: self.sandbox.fs.write_bytes(stdin_path, stdin))
                    # Nested shell quoting can exceed execve's single-argument limit
                    # even when every original command argument fits.
                    await retry_transient(lambda: self.sandbox.fs.write_bytes(script_path, script.encode()))
                    launch_requested = True

                    def request() -> Awaitable[ExecOutput]:
                        return _exec(
                            self.sandbox,
                            _control_argv(shlex.join(("/bin/sh", script_path))),
                            cwd=command.cwd or self.spec.workdir or None,
                            env=environment,
                            execution_timeout=math.ceil(command.timeout + 10) if command.timeout is not None else None,
                        )

                    try:
                        response = await request()
                        exit_code, diagnostic = response.exit_code, response.output
                    except DaytonaError as error:
                        if not is_transient_daytona_error(error):
                            raise
                        logger.warning("Reattaching to a Daytona command after %s", type(error).__name__)
                        exit_code, diagnostic = await self._exit_after_lost_request(request, pidfile, completed_path)
                async with asyncio.timeout(self.output_collection_timeout):
                    try:
                        return await self._command_result(
                            exit_code, stdout_path, stderr_path, command.output_limit_bytes
                        )
                    except Exception as error:
                        error.add_note(
                            "Daytona exec response: "
                            + json.dumps(
                                {
                                    "exit_code": exit_code,
                                    "diagnostic": diagnostic[:PROVIDER_DIAGNOSTIC_CHAR_LIMIT],
                                    "diagnostic_truncated": len(diagnostic) > PROVIDER_DIAGNOSTIC_CHAR_LIMIT,
                                }
                            )
                        )
                        raise
        except TimeoutError as error:
            if exit_code is not None:
                primary_error = DaytonaOutputUnavailable("Cannot collect Daytona command output")
                raise primary_error from error
            if not deadline.expired():
                primary_error = error
                raise
            if not launch_requested:
                return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
            try:
                await self._stop_command(pidfile, completed_path)
            except asyncio.CancelledError as cancellation:
                primary_error = cancellation
                await self._cleanup_cancelled_command(prefix, pidfile, completed_path, launch_requested, cancellation)
                raise
            except Exception as error:
                # Closing the machine ends it, so the caller sees the machine as terminated.
                primary_error = MachineTerminated(
                    "Daytona sandbox stopped responding to the command stop request"
                    if isinstance(error, TimeoutError)
                    else "Cannot stop the Daytona command process group"
                )
                await self._close_preserving_failure(primary_error)
                raise primary_error from error
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError as cancellation:
            primary_error = cancellation
            await self._cleanup_cancelled_command(prefix, pidfile, completed_path, launch_requested, cancellation)
            raise
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if not self._closed and not isinstance(primary_error, asyncio.CancelledError):
                try:
                    await self._remove_command_files(prefix)
                except asyncio.CancelledError as cancellation:
                    await self._close_preserving_failure(cancellation)
                    raise
                except Exception as error:
                    if primary_error is None:
                        primary_error = MachineTerminated(
                            "Daytona sandbox stopped responding to command file cleanup"
                            if isinstance(error, TimeoutError)
                            else "Cannot remove Daytona command files"
                        )
                        await self._close_preserving_failure(primary_error)
                        raise primary_error from error
                    primary_error.add_note(f"Daytona command cleanup failed: {error}")
                    await self._close_preserving_failure(primary_error)

    async def upload(self, source: Path, target: str) -> None:
        self._require_open()
        # Each transfer rewrites its target, so a transfer that lost its connection starts over.
        async with self._provider_failures():
            await retry_transient(lambda: self._upload(source, target))

    async def _upload(self, source: Path, target: str) -> None:
        if source.is_dir():
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                with tarfile.open(archive.name, "w:gz") as tar:
                    tar.add(source, arcname=".")
                remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
                await self.sandbox.fs.upload_file(archive.name, remote_archive)
            result = await _exec(
                self.sandbox,
                _control_argv(
                    f"mkdir -p {shlex.quote(target)} && tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target)}"
                    f"; status=$?; rm -f {shlex.quote(remote_archive)}; exit $status"
                ),
                cwd=CONTROL_CWD,
            )
        else:
            parent = str(PurePosixPath(target).parent)
            result = await _exec(self.sandbox, _control_argv(f"mkdir -p {shlex.quote(parent)}"), cwd=CONTROL_CWD)
            if result.exit_code:
                raise RuntimeError(f"Failed to create {parent}: {result.output}")
            await self.sandbox.fs.upload_file(source, target)
            mode = stat.S_IMODE(source.stat().st_mode)
            result = await _exec(self.sandbox, _control_argv(f"chmod {mode:o} {shlex.quote(target)}"), cwd=CONTROL_CWD)
        if result.exit_code:
            raise RuntimeError(f"Failed to upload {source}: {result.output}")

    async def download(self, source: str, target: Path) -> None:
        self._require_open()
        async with self._provider_failures():
            await retry_transient(lambda: self._download(source, target))

    async def _download(self, source: str, target: Path) -> None:
        probe = await _exec(self.sandbox, _control_argv(f"test -d {shlex.quote(source)}"), cwd=CONTROL_CWD)
        if probe.exit_code == 0:
            remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
            result = await _exec(
                self.sandbox,
                _control_argv(f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source)} ."),
                cwd=CONTROL_CWD,
            )
            if result.exit_code:
                raise RuntimeError(f"Failed to archive {source}: {result.output}")
            try:
                target.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as staging:
                    archive = Path(staging) / "archive.tar.gz"
                    await self.sandbox.fs.download_file(remote_archive, archive)
                    with tarfile.open(archive, "r:gz") as tar:
                        tar.extractall(target, filter="data")
            finally:
                await _exec(self.sandbox, _control_argv(f"rm -f {shlex.quote(remote_archive)}"), cwd=CONTROL_CWD)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        await self.sandbox.fs.download_file(source, target)

    async def close(self) -> None:
        if self._closing is None:
            self._closed = True
            self._closing = own_cleanup(self._close())
        try:
            await asyncio.shield(self._closing)
        except asyncio.CancelledError as error:
            try:
                await finish_owned(self._closing)
            except Exception as cleanup_error:
                error.add_note(f"Daytona cleanup failed ({type(cleanup_error).__name__})")
            raise

    async def _close(self) -> None:
        try:
            await asyncio.gather(*tuple(self._acquisitions), return_exceptions=True)
            results = await asyncio.gather(
                *(session.close() for session in tuple(self._sessions)), return_exceptions=True
            )
        finally:
            await self.resources.aclose()
        for result in results:
            if isinstance(result, BaseException):
                raise result


class DaytonaMachineFactory:
    """Create a Daytona sandbox from a registry image or Docker build context."""

    backend: Backend = Backend.DAYTONA

    def __init__(
        self,
        client_factory: Callable[[], AsyncDaytonaV2Client] | None,
        *,
        ttl_minutes: int = DEFAULT_SANDBOX_TTL_MINUTES,
        idle_timeout_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES,
        labels: Mapping[str, str] | None = None,
        create_timeout: float = 600,
        output_collection_timeout: float = DEFAULT_OUTPUT_COLLECTION_TIMEOUT,
        network_policy: DaytonaNetworkPolicy | None = None,
    ):
        if ttl_minutes <= 0:
            raise ValueError("Daytona sandbox TTL must be positive")
        if idle_timeout_minutes < 0:
            raise ValueError("Daytona idle timeout must be nonnegative")
        self.client_factory = client_factory
        self.ttl_minutes = ttl_minutes
        self.idle_timeout_minutes = idle_timeout_minutes
        self.labels = {"owner": "shellbox", **(labels or {})}
        self.create_timeout = create_timeout
        self.output_collection_timeout = output_collection_timeout
        self.network_policy = network_policy

    def _clients(self) -> Callable[[], AsyncDaytonaV2Client]:
        """Resolve the configured client factory for one provider session."""
        if self.client_factory is None:
            raise ValueError("Daytona requires an explicit host connection")
        return self.client_factory

    def _check_spec(self, spec: MachineSpec) -> DockerfileSource | RegistryImage:
        source: DockerfileSource | RegistryImage
        if isinstance(spec.source, DockerfileSource):
            # Reject a build context Daytona cannot use when the spec is checked, before any machine starts.
            source = validate_source(spec.source)
        elif isinstance(spec.source, RegistryImage):
            source = spec.source
        else:
            raise UnsupportedMachineSpec("Daytona requires a registry image or Docker build context")
        sandbox_resources(spec)
        if self.network_policy is not None:
            self.network_policy.parameters()
        return source

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        source = self._check_spec(spec)
        reference = source.reference if isinstance(source, RegistryImage) else None
        if self.network_policy is not None:
            network = (
                NetworkPolicy.DENY if self.network_policy.mode == DaytonaNetworkMode.BLOCK_ALL else NetworkPolicy.ALLOW
            )
            spec = replace(spec, network=network)
        check_execution_requirements(
            requirements,
            spec,
            semantics=CommandSemantics.LINUX_PROCESS,
            capabilities=LINUX_PROCESS_CAPABILITIES,
            image_reference=reference,
        )

    def _lifecycle_policy(self) -> LifecyclePolicyOptions:
        """Delete the sandbox at its TTL, and after the idle timeout when one is set, so orphans expire."""
        policy: LifecyclePolicyOptions = {"timeout_seconds": self.ttl_minutes * 60, "on_timeout": "delete"}
        if self.idle_timeout_minutes:
            policy["idle_expiry_seconds"] = self.idle_timeout_minutes * 60
            policy["on_idle_timeout"] = "delete"
        else:
            policy["idle_expiry_seconds"] = DISABLED
        return policy

    async def prepare(self, spec: MachineSpec) -> str:
        """Prepare the image used by create without starting a sandbox.

        Pin registry sources by digest to reuse immutable image content.
        """
        client_factory = self._clients()
        timeout = self.create_timeout if spec.startup_timeout is None else spec.startup_timeout
        self._check_spec(spec)
        request = image_request_for_spec(spec)
        async with client_factory() as client, asyncio.timeout(timeout):
            return await ensure_image(client, request)

    async def create(self, spec: MachineSpec) -> DaytonaMachine:
        network = (
            {"block_all": spec.network is NetworkPolicy.DENY}
            if self.network_policy is None
            else self.network_policy.parameters()
        )
        async with AsyncExitStack() as lifetime:
            client = await lifetime.enter_async_context(self._clients()())
            timeout = self.create_timeout if spec.startup_timeout is None else spec.startup_timeout
            async with asyncio.timeout(timeout):
                self._check_spec(spec)
                image = await ensure_image(client, image_request_for_spec(spec))
                name = f"shellbox-{uuid.uuid4().hex}"
                claim = _SandboxClaim(name)
                lifetime.push_async_callback(_delete_sandbox, client, claim)
                try:
                    sandbox = await _create_sandbox(
                        client,
                        image,
                        name=name,
                        timeout=timeout,
                        env=spec.env,
                        labels=dict(self.labels),
                        resources=sandbox_resources(spec),
                        lifecycle_policy=self._lifecycle_policy(),
                        network_policy=network,
                    )
                except DaytonaError as error:
                    # A refused create made no sandbox, so cleanup has nothing to look for.
                    claim.unanswered = is_ambiguous_daytona_error(error)
                    raise
                claim.sandboxes.append(sandbox)
                claim.unanswered = False
                if sandbox.state != STARTED_STATE:
                    raise RuntimeError(f"Daytona sandbox {sandbox.id} is {sandbox.state}: {sandbox.error}")
            if self.idle_timeout_minutes:
                keepalive = asyncio.create_task(_keep_active(sandbox, self.idle_timeout_minutes * 60 / 3))
                lifetime.push_async_callback(_stop_keepalive, keepalive)
            prepared = await _exec(
                sandbox, _control_argv("command -v setsid"), cwd=CONTROL_CWD, execution_timeout=INTERRUPT_TIMEOUT
            )
            if prepared.exit_code:
                raise UnsupportedMachineSpec("Daytona task images require setsid for command timeout recovery")
            effective_spec = spec
            if not spec.workdir:
                default_cwd = await _exec(
                    sandbox, _control_argv("pwd -L"), cwd=None, execution_timeout=INTERRUPT_TIMEOUT
                )
                image_workdir = default_cwd.output.removesuffix("\n")
                if default_cwd.exit_code or not PurePosixPath(image_workdir).is_absolute():
                    raise RuntimeError(f"Cannot resolve Daytona image workdir: {default_cwd.output!r}")
                effective_spec = replace(spec, workdir=image_workdir)
            else:
                workdir_result = await _exec(
                    sandbox,
                    _control_argv(shlex.join(("mkdir", "-p", "--", spec.workdir))),
                    cwd=CONTROL_CWD,
                    execution_timeout=INTERRUPT_TIMEOUT,
                )
                if workdir_result.exit_code:
                    raise RuntimeError(f"Failed to create workdir {spec.workdir}: {workdir_result.output!r}")
            machine = DaytonaMachine(
                sandbox, effective_spec, lifetime, output_collection_timeout=self.output_collection_timeout
            )
            machine.resources = lifetime.pop_all()
            return machine
