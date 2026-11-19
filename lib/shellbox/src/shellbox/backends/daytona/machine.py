# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the machine contract on a Daytona sandbox."""

import asyncio
import json
import logging
import re
import shlex
import stat
import tarfile
import tempfile
import time
import uuid
from asyncio import sleep
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath

from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaConflictError,
    DaytonaConnectionError,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaTimeoutError,
    SessionExecuteRequest,
)
from rigging.timing import ExponentialBackoff

from shellbox.backends.daytona.process import DaytonaProcessSession
from shellbox.backends.daytona.snapshot import (
    daytona_image,
    ensure_snapshot,
    snapshot_request_for_spec,
)
from shellbox.backends.docker.machine import INTERRUPT_TIMEOUT, KILL_PROCESS_GROUP_COMMAND, START_COMMAND
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    Backend,
    Command,
    CommandSemantics,
    ExecutionRequirements,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    ProcessScope,
    Result,
    UnsupportedMachineSpec,
    check_execution_requirements,
)
from shellbox.process_session import await_cleanup, finish_owned, own_cleanup

DEFAULT_SANDBOX_TTL_MINUTES = 360
DEFAULT_OUTPUT_COLLECTION_TIMEOUT = 60
logger = logging.getLogger(__name__)
CONTROL_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
CONTROL_CWD = "/"
COUNT_PREFIX = "SHELLBOX_COUNT="
PROVIDER_DIAGNOSTIC_CHAR_LIMIT = 4096
MAX_COMMAND_POLL_DELAY = 5
RUN_COMMAND = (
    'pidfile=$1; completed=$2; shift 2; echo $$ > "$pidfile"; trap \': > "$completed"; rm -f "$pidfile"\' EXIT; "$@"'
)
STOP_COMMAND = (
    "pidfile=$1; completed=$2; "
    '[ ! -f "$completed" ] || exit 0; '
    'if read -r pid < "$pidfile"; then '
    'case "$pid" in ""|*[!0-9]*) exit 1;; esac; '
    '[ "$pid" -gt 1 ] || exit 1; set -- "$pid"; ' + KILL_PROCESS_GROUP_COMMAND + ' && exit 0; fi; [ -f "$completed" ]'
)


def _control_script(script: str) -> str:
    return f"_SHELLBOX_COMMAND_PATH=$PATH; export _SHELLBOX_COMMAND_PATH; PATH={CONTROL_PATH}; export PATH; {script}"


def _control_command(script: str) -> str:
    return shlex.join(("/bin/sh", "-c", _control_script(script)))


def _is_transient_session_error(error: Exception) -> bool:
    if isinstance(
        error, (TimeoutError, ConnectionError, DaytonaConnectionError, DaytonaTimeoutError, DaytonaRateLimitError)
    ):
        return True
    return (
        isinstance(error, DaytonaError)
        and error.status_code is not None
        and (error.status_code in (408, 429) or 500 <= error.status_code < 600)
    )


def _retry_after(error: Exception) -> int:
    return max(
        (
            int(value)
            for name, value in (getattr(error, "headers", None) or {}).items()
            if name.lower().startswith("retry-after") and str(value).isdecimal()
        ),
        default=0,
    )


@dataclass(frozen=True)
class _CompletedCommand:
    exit_code: int
    session_id: str
    command_id: str


async def _create_sandbox(
    client: AsyncDaytona, params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams, timeout: float
) -> AsyncSandbox:
    """Retry throttling using a caller-owned name to recover an accepted creation."""
    assert params.name is not None
    backoff = ExponentialBackoff(initial=1, maximum=30, factor=2, jitter=0.5)
    while True:
        try:
            try:
                return await client.create(params, timeout=timeout)
            except DaytonaConflictError:
                sandbox = await client.get(params.name)
                await sandbox.wait_for_sandbox_start(timeout=timeout)
                return sandbox
        except DaytonaRateLimitError as error:
            delay = max(_retry_after(error), backoff.next_interval())
            logger.warning("Retrying throttled Daytona creation: delay=%.3f", delay)
            await sleep(delay)


async def _delete_named_sandbox(client: AsyncDaytona, name: str) -> None:
    async with asyncio.timeout(INTERRUPT_TIMEOUT):
        try:
            sandbox = await client.get(name)
        except DaytonaNotFoundError:
            return
        await client.delete(sandbox)


class DaytonaImageBuild(StrEnum):
    """How a sandbox gets its image.

    ``SNAPSHOT`` builds each distinct image once into a named account snapshot and starts
    sandboxes from it; snapshots persist and count against the account's snapshot quota.
    ``DECLARATIVE`` builds the image for every sandbox and creates no snapshot. The account
    must permit declarative builds.
    """

    SNAPSHOT = "snapshot"
    DECLARATIVE = "declarative"


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

    def parameters(self) -> dict[str, str | bool]:
        """Return the provider's mutually exclusive network policy fields."""
        mode = DaytonaNetworkMode(self.mode)
        if mode in (DaytonaNetworkMode.NETWORK_ALLOW_LIST, DaytonaNetworkMode.DOMAIN_ALLOW_LIST):
            if not self.value:
                raise ValueError(f"{self.mode} requires an allow list")
            return {mode.value: self.value}
        if self.value is not None:
            raise ValueError(f"{self.mode} does not accept an allow list")
        return {"network_block_all": mode == DaytonaNetworkMode.BLOCK_ALL}


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
        self._sessions: set[DaytonaProcessSession] = set()
        self._acquisitions: set[asyncio.Task[None]] = set()
        self._closing: asyncio.Task[None] | None = None

    async def _user_argv(self, argv: tuple[str, ...], user: str | None) -> tuple[str, ...]:
        if user in (None, "root", "0"):
            return argv
        assert user is not None
        if user.isdecimal():
            account = await self.sandbox.process.exec(
                _control_command(f"getent passwd {shlex.quote(user)}"), cwd=CONTROL_CWD
            )
            if account.exit_code or not account.result.strip():
                raise ValueError(f"Execution user {user} has no guest account")
            user = account.result.split(":", 1)[0]
        capabilities = await self.sandbox.process.exec(_control_command("su --help"), cwd=CONTROL_CWD)
        if capabilities.exit_code or "--session-command" not in capabilities.result:
            raise UnsupportedMachineSpec("Non-root Daytona commands require util-linux su --session-command")
        return ("su", "-s", "/bin/sh", "-m", user, "--session-command", shlex.join(argv))

    async def open_process(self, command: Command) -> DaytonaProcessSession:
        """Attach persistent binary pipes through one provider session and log stream."""
        if self._closed:
            raise RuntimeError("Machine is closed")
        if not command.argv or command.stdin or command.timeout is not None:
            raise ValueError("Process sessions require argv and own stdin and their lifetime")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")

        async def cleanup_failed() -> None:
            if not self._closed:
                await self.close()

        session = DaytonaProcessSession(
            self.sandbox.process,
            f"shellbox-{uuid.uuid4().hex}",
            stderr_limit=command.output_limit_bytes,
            release=self._sessions.discard,
            failed_cleanup=cleanup_failed,
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
            if self._closed:
                raise RuntimeError("Machine was closed while opening a process session")
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

    async def _read_output(self, path: str, limit: int) -> tuple[bytes, bool]:
        count = await self.sandbox.process.exec(
            _control_command(f"size=$(wc -c < {shlex.quote(path)}) || exit; printf '{COUNT_PREFIX}%s\\n' \"$size\""),
            cwd=CONTROL_CWD,
        )
        if count.exit_code:
            raise RuntimeError(f"Failed to measure command output: {count.result}")
        match = re.search(rf"(?m)^{COUNT_PREFIX}([0-9]+)$", count.result)
        if match is None:
            raise RuntimeError(f"Invalid command output size: {count.result!r}")
        size = int(match.group(1))
        if size <= limit:
            data = await self.sandbox.fs.download_file(path)
            assert isinstance(data, bytes)
            return data, False
        clipped = f"{path}.limited"
        result = await self.sandbox.process.exec(
            _control_command(f"head -c {limit} {shlex.quote(path)} > {shlex.quote(clipped)}"), cwd=CONTROL_CWD
        )
        if result.exit_code:
            raise RuntimeError(f"Failed to limit command output: {result.result}")
        try:
            data = await self.sandbox.fs.download_file(clipped)
            assert isinstance(data, bytes)
            return data, True
        finally:
            await self.sandbox.process.exec(_control_command(f"rm -f {shlex.quote(clipped)}"), cwd=CONTROL_CWD)

    async def _stop_command(self, pidfile: str, completed_path: str) -> None:
        async with asyncio.timeout(INTERRUPT_TIMEOUT):
            stopped = await self.sandbox.process.exec(
                _control_command(shlex.join(("/bin/sh", "-c", STOP_COMMAND, "stop-command", pidfile, completed_path))),
                cwd=CONTROL_CWD,
            )
            if stopped.exit_code:
                raise RuntimeError(stopped.result)

    async def _remove_command_files(self, prefix: str) -> None:
        async with asyncio.timeout(INTERRUPT_TIMEOUT):
            removed = await self.sandbox.process.exec(_control_command(f"rm -rf {shlex.quote(prefix)}"), cwd=CONTROL_CWD)
            if removed.exit_code:
                raise RuntimeError("Cannot remove Daytona command files")

    async def _close_preserving_failure(self, failure: BaseException) -> None:
        try:
            await self.close()
        except Exception as cleanup_error:
            failure.add_note(f"Daytona machine cleanup failed: {cleanup_error!r}")

    async def _abort_command(self, prefix: str, started: bool, failure: BaseException) -> None:
        """Stop possibly submitted work, or close after interrupted preparation.

        Finish despite repeated cancellation. If stopping fails, close the machine
        and attach cleanup errors to the original failure. Files belong to run().
        """
        try:
            if started:
                await finish_owned(own_cleanup(self._stop_command(f"{prefix}/pid", f"{prefix}/completed")))
            else:
                await finish_owned(own_cleanup(self.close()))
        except BaseException as cleanup_error:
            failure.add_note(f"Daytona command cleanup failed: {cleanup_error!r}")
            try:
                await finish_owned(own_cleanup(self.close()))
            except BaseException as close_error:
                failure.add_note(f"Daytona machine cleanup failed: {close_error!r}")

    async def open_process_scope(self) -> ProcessScope:
        raise UnsupportedMachineSpec("Daytona does not provide confirmed scoped descendant teardown")

    async def _delete_command_session(self, name: str) -> None:
        try:
            for attempt in range(2):
                try:
                    async with asyncio.timeout(INTERRUPT_TIMEOUT):
                        await self.sandbox.process.delete_session(name, request_timeout=INTERRUPT_TIMEOUT)
                    return
                except DaytonaNotFoundError:
                    return
                except Exception as error:
                    if attempt or not _is_transient_session_error(error):
                        raise
                    logger.warning("Retrying Daytona command session deletion: error=%s", type(error).__name__)
                    await sleep(0.2)
        except Exception as error:
            try:
                await self.close()
            except BaseException as close_error:
                error.add_note(f"Daytona machine cleanup failed: {close_error!r}")
            raise

    @asynccontextmanager
    async def _command_session(self) -> AsyncIterator[str]:
        """Own session acquisition and deletion through output/diagnostic collection.

        A cancelled or failed acquisition still deletes any late session. Cleanup
        finishes despite cancellation; its errors do not replace a primary failure.
        """
        name = f"shellbox-run-{uuid.uuid4().hex}"
        creation = asyncio.create_task(self.sandbox.process.create_session(name, request_timeout=INTERRUPT_TIMEOUT))
        primary_error = None
        try:
            await asyncio.shield(creation)
            yield name
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if not creation.done():
                try:
                    await finish_owned(creation)
                except BaseException as error:
                    if primary_error is not None:
                        primary_error.add_note(f"Daytona command session creation failed ({type(error).__name__})")
            cleanup = own_cleanup(self._delete_command_session(name))
            if primary_error is None:
                await await_cleanup(cleanup)
            else:
                try:
                    await finish_owned(cleanup)
                except BaseException as error:
                    primary_error.add_note(f"Daytona command session cleanup failed: {error!r}")

    async def _poll_command(self, path: str, name: str) -> _CompletedCommand:
        """Submit once, then poll bounded status reads, retrying transient read failures."""
        process = self.sandbox.process
        polling = ExponentialBackoff(initial=0.2, maximum=MAX_COMMAND_POLL_DELAY, factor=1.5)
        retry = ExponentialBackoff(initial=0.5, maximum=MAX_COMMAND_POLL_DELAY, factor=2)
        requests = failures = consecutive_failures = 0
        started = time.monotonic()
        try:
            response = await process.execute_session_command(
                name,
                SessionExecuteRequest(command=shlex.join(("/bin/sh", path)), run_async=True, suppress_input_echo=True),
                timeout=INTERRUPT_TIMEOUT,
            )
            while True:
                requests += 1
                try:
                    status = await process.get_session_command(name, response.cmd_id, request_timeout=INTERRUPT_TIMEOUT)
                except Exception as error:
                    if not _is_transient_session_error(error):
                        raise
                    failures += 1
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        raise RuntimeError("Daytona command status unavailable") from error
                    delay = max(_retry_after(error), retry.next_interval())
                    logger.warning("Retrying Daytona command status: error=%s delay=%.3f", type(error).__name__, delay)
                    await sleep(delay)
                    continue
                consecutive_failures = 0
                retry.reset()
                if status.exit_code is not None:
                    return _CompletedCommand(status.exit_code, name, response.cmd_id)
                await sleep(polling.next_interval())
        finally:
            logger.info(
                "Daytona command status polling finished: requests=%d failures=%d seconds=%.3f",
                requests,
                failures,
                time.monotonic() - started,
            )

    async def _prepare_command(self, command: Command, prefix: str) -> str:
        """Stage one command, returning its script path without submitting it."""
        environment = {**self.spec.env, **command.env}
        stdin_path, stdout_path, stderr_path = (f"{prefix}/{part}" for part in ("in", "out", "err"))
        script_path = f"{prefix}/command.sh"
        pidfile, completed_path = f"{prefix}/pid", f"{prefix}/completed"
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
        script = (
            f"{shlex.join(argv)} < {shlex.quote(stdin_path) if command.stdin else '/dev/null'} "
            f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
        )
        prepared = await self.sandbox.process.exec(
            _control_command(f"umask 077; mkdir {shlex.quote(prefix)}"), cwd=CONTROL_CWD
        )
        if prepared.exit_code:
            raise RuntimeError(f"Cannot create private command directory: {prepared.result}")
        if command.stdin:
            await self.sandbox.fs.upload_file_stream(command.stdin, stdin_path)
        lines = []
        for key, value in environment.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("Invalid command environment variable name")
            lines.append(f"export {shlex.quote(key + '=' + value)}")
        if cwd := command.cwd or self.spec.workdir:
            lines.append(f"cd {shlex.quote(cwd)} || exit $?")
        lines.append(_control_script(script))
        script = "\n".join(lines) + "\n"
        await self.sandbox.fs.upload_file_stream(script.encode(), script_path)
        return script_path

    async def _execute_command(
        self, command: Command, prefix: str, sessions: AsyncExitStack
    ) -> _CompletedCommand | None:
        """Prepare and wait under the caller's deadline; None means it expired.

        The session stays in sessions until output and diagnostics are read.
        Submission/status failures abort possibly accepted work without replay.
        Failed session acquisition has not submitted work. Caller expiry stops
        submitted work before returning None; cancellation finishes abort before
        propagating. Provider failures remain exceptions.
        """
        started = False
        try:
            async with asyncio.timeout(command.timeout) as deadline:
                script_path = await self._prepare_command(command, prefix)
                session_id = await sessions.enter_async_context(self._command_session())
                # The provider may accept the command before the submission request fails.
                started = True
                return await self._poll_command(script_path, session_id)
        except TimeoutError as error:
            if not deadline.expired():
                if started and not self._closed:
                    await self._abort_command(prefix, started, error)
                raise
            if started:
                try:
                    await self._stop_command(f"{prefix}/pid", f"{prefix}/completed")
                except asyncio.CancelledError as cancellation:
                    await self._abort_command(prefix, started, cancellation)
                    raise
                except Exception as error:
                    failure = RuntimeError("Cannot stop the Daytona command process group")
                    await self._close_preserving_failure(failure)
                    raise failure from error
            return None
        except asyncio.CancelledError as cancellation:
            await self._abort_command(prefix, started, cancellation)
            raise
        except BaseException as error:
            if started and not self._closed:
                await self._abort_command(prefix, started, error)
            raise

    async def _collect_command_output(self, command: Command, prefix: str, response: _CompletedCommand) -> Result:
        """Read bounded outputs under a separate deadline, retaining provider diagnostics.

        Sessions must remain open here. A collection timeout is an infrastructure
        failure even when the command itself has already completed successfully.
        """
        try:
            async with asyncio.timeout(self.output_collection_timeout):
                try:
                    stdout, stdout_truncated = await self._read_output(f"{prefix}/out", command.output_limit_bytes)
                    stderr, stderr_truncated = await self._read_output(f"{prefix}/err", command.output_limit_bytes)
                except Exception as error:
                    diagnostic = ""
                    try:
                        logs = await self.sandbox.process.get_session_command_logs(
                            response.session_id, response.command_id, request_timeout=INTERRUPT_TIMEOUT
                        )
                        diagnostic = logs.output or (logs.stdout or "") + (logs.stderr or "")
                    except Exception as diagnostic_error:
                        error.add_note(f"Daytona session diagnostic collection failed: {diagnostic_error!r}")
                    error.add_note(
                        "Daytona exec response: "
                        + json.dumps(
                            {
                                "exit_code": response.exit_code,
                                "diagnostic": diagnostic[:PROVIDER_DIAGNOSTIC_CHAR_LIMIT],
                                "diagnostic_truncated": len(diagnostic) > PROVIDER_DIAGNOSTIC_CHAR_LIMIT,
                            }
                        )
                    )
                    raise
        except TimeoutError as error:
            raise RuntimeError("Cannot collect Daytona command output") from error
        return Result(response.exit_code, stdout, stderr, stdout_truncated, stderr_truncated, ExitReason.EXITED)

    async def run(self, command: Command) -> Result:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        prefix = f"/tmp/.shellbox-{uuid.uuid4().hex}"
        primary_error = None
        async with AsyncExitStack() as sessions:
            try:
                response = await self._execute_command(command, prefix, sessions)
                if response is None:
                    return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
                return await self._collect_command_output(command, prefix, response)
            except BaseException as error:
                primary_error = error
                raise
            finally:
                if not self._closed:
                    try:
                        if isinstance(primary_error, asyncio.CancelledError):
                            await finish_owned(own_cleanup(self._remove_command_files(prefix)))
                        else:
                            await self._remove_command_files(prefix)
                    except asyncio.CancelledError as cancellation:
                        await self._close_preserving_failure(cancellation)
                        raise
                    except Exception as error:
                        if primary_error is None:
                            failure = RuntimeError("Cannot remove Daytona command files")
                            await self._close_preserving_failure(failure)
                            raise failure from error
                        primary_error.add_note(f"Daytona command cleanup failed: {error}")
                        await self._close_preserving_failure(primary_error)

    async def upload(self, source: Path, target: str) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if source.is_dir():
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                with tarfile.open(archive.name, "w:gz") as tar:
                    tar.add(source, arcname=".")
                remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
                await self.sandbox.fs.upload_file_stream(Path(archive.name).read_bytes(), remote_archive)
            result = await self.sandbox.process.exec(
                _control_command(
                    f"mkdir -p {shlex.quote(target)} && tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target)}"
                    f"; status=$?; rm -f {shlex.quote(remote_archive)}; exit $status"
                ),
                cwd=CONTROL_CWD,
            )
        else:
            parent = str(PurePosixPath(target).parent)
            result = await self.sandbox.process.exec(
                _control_command(f"mkdir -p {shlex.quote(parent)}"), cwd=CONTROL_CWD
            )
            if result.exit_code:
                raise RuntimeError(f"Failed to create {parent}: {result.result}")
            await self.sandbox.fs.upload_file_stream(source.read_bytes(), target)
            mode = stat.S_IMODE(source.stat().st_mode)
            result = await self.sandbox.process.exec(
                _control_command(f"chmod {mode:o} {shlex.quote(target)}"), cwd=CONTROL_CWD
            )
        if result.exit_code:
            raise RuntimeError(f"Failed to upload {source}: {result.result}")

    async def download(self, source: str, target: Path) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
        probe = await self.sandbox.process.exec(_control_command(f"test -d {shlex.quote(source)}"), cwd=CONTROL_CWD)
        if probe.exit_code == 0:
            remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
            result = await self.sandbox.process.exec(
                _control_command(f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source)} ."), cwd=CONTROL_CWD
            )
            if result.exit_code:
                raise RuntimeError(f"Failed to archive {source}: {result.result}")
            try:
                data = await self.sandbox.fs.download_file(remote_archive)
                assert isinstance(data, bytes)
                target.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                    Path(archive.name).write_bytes(data)
                    with tarfile.open(archive.name, "r:gz") as tar:
                        tar.extractall(target, filter="data")
            finally:
                await self.sandbox.process.exec(
                    _control_command(f"rm -f {shlex.quote(remote_archive)}"), cwd=CONTROL_CWD
                )
            return
        data = await self.sandbox.fs.download_file(source)
        assert isinstance(data, bytes)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

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
        client_factory: Callable[[], AsyncDaytona] | None = AsyncDaytona,
        *,
        ttl_minutes: int = DEFAULT_SANDBOX_TTL_MINUTES,
        create_timeout: float = 600,
        output_collection_timeout: float = DEFAULT_OUTPUT_COLLECTION_TIMEOUT,
        network_policy: DaytonaNetworkPolicy | None = None,
        image_build: DaytonaImageBuild = DaytonaImageBuild.SNAPSHOT,
    ):
        self.client_factory = client_factory
        self.ttl_minutes = ttl_minutes
        self.create_timeout = create_timeout
        self.output_collection_timeout = output_collection_timeout
        self.network_policy = network_policy
        self.image_build = image_build

    def _clients(self) -> Callable[[], AsyncDaytona]:
        """Resolve the configured client factory for one provider session."""
        if self.client_factory is None:
            raise ValueError("Daytona requires an explicit host connection")
        return self.client_factory

    def _check_spec(self, spec: MachineSpec) -> DockerfileSource | RegistryImage:
        source: DockerfileSource | RegistryImage
        if isinstance(spec.source, DockerfileSource):
            source = spec.source
        elif isinstance(spec.source, RegistryImage):
            source = spec.source
        else:
            raise UnsupportedMachineSpec("Daytona requires a registry image or Docker build context")
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

    async def prepare(self, spec: MachineSpec) -> str:
        """Prepare the snapshot used by create without starting a sandbox.

        Pin registry sources by digest to reuse immutable image content.
        """
        if self.image_build != DaytonaImageBuild.SNAPSHOT:
            raise ValueError(f"{self.image_build} image builds create no snapshot to prepare")
        client_factory = self._clients()
        timeout = self.create_timeout if spec.startup_timeout is None else spec.startup_timeout
        self._check_spec(spec)
        request = snapshot_request_for_spec(spec)
        async with client_factory() as client, asyncio.timeout(timeout):
            return await ensure_snapshot(client, request)

    async def create(self, spec: MachineSpec) -> DaytonaMachine:
        network = (
            {"network_block_all": spec.network is NetworkPolicy.DENY}
            if self.network_policy is None
            else self.network_policy.parameters()
        )
        async with AsyncExitStack() as lifetime:
            client = await lifetime.enter_async_context(self._clients()())
            timeout = self.create_timeout if spec.startup_timeout is None else spec.startup_timeout
            async with asyncio.timeout(timeout):
                self._check_spec(spec)
                request = snapshot_request_for_spec(spec)
                name = f"shellbox-{uuid.uuid4().hex}"
                sandbox_fields = dict(
                    name=name, os_user="root", env_vars=spec.env, **network, ttl_minutes=self.ttl_minutes
                )
                if self.image_build == DaytonaImageBuild.SNAPSHOT:
                    params = CreateSandboxFromSnapshotParams(
                        snapshot=await ensure_snapshot(client, request), **sandbox_fields
                    )
                else:
                    params = CreateSandboxFromImageParams(
                        image=daytona_image(request.source), resources=request.resources, **sandbox_fields
                    )
                lifetime.push_async_callback(_delete_named_sandbox, client, name)
                sandbox = await _create_sandbox(client, params, timeout=timeout)
            prepared = await sandbox.process.exec(
                _control_command("command -v setsid"), cwd=CONTROL_CWD, timeout=INTERRUPT_TIMEOUT
            )
            if prepared.exit_code:
                raise UnsupportedMachineSpec("Daytona task images require setsid for command timeout recovery")
            effective_spec = spec
            if not spec.workdir:
                default_cwd = await sandbox.process.exec(_control_command("pwd -L"), timeout=INTERRUPT_TIMEOUT)
                image_workdir = default_cwd.result.removesuffix("\n")
                if default_cwd.exit_code or not PurePosixPath(image_workdir).is_absolute():
                    raise RuntimeError(f"Cannot resolve Daytona image workdir: {default_cwd.result!r}")
                effective_spec = replace(spec, workdir=image_workdir)
            machine = DaytonaMachine(
                sandbox,
                effective_spec,
                lifetime,
                output_collection_timeout=self.output_collection_timeout,
            )
            if spec.workdir:
                result = await machine.run(Command(("mkdir", "-p", spec.workdir), cwd="/", env={"PATH": CONTROL_PATH}))
                if result.exit_code:
                    raise RuntimeError(f"Failed to create workdir {spec.workdir}: {result.stderr!r}")
            machine.resources = lifetime.pop_all()
            return machine
