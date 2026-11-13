# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the machine contract on a Daytona sandbox."""

import asyncio
import json
import logging
import math
import re
import shlex
import stat
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from pathlib import Path, PurePosixPath

from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromSnapshotParams,
    DaytonaConflictError,
    DaytonaNotFoundError,
)

from shellbox.backends.daytona.process import DaytonaProcessSession
from shellbox.backends.daytona.provider import (
    PROVIDER_REQUEST_TIMEOUT,
    ProviderWaitPolicy,
    cancel_provider_waits,
    provider_wait_scope,
    retry_provider_call,
)
from shellbox.backends.daytona.snapshot import (
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
from shellbox.process_session import finish_owned, own_cleanup

DEFAULT_SANDBOX_TTL_MINUTES = 360
logger = logging.getLogger(__name__)
CONTROL_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
CONTROL_CWD = "/"
COUNT_PREFIX = "SHELLBOX_COUNT="
PROVIDER_DIAGNOSTIC_CHAR_LIMIT = 4096
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


def _control_command(script: str) -> str:
    return shlex.join(
        (
            "/bin/sh",
            "-c",
            f"_SHELLBOX_COMMAND_PATH=$PATH; export _SHELLBOX_COMMAND_PATH; "
            f"PATH={CONTROL_PATH}; export PATH; {script}",
        )
    )


async def _create_sandbox(
    client: AsyncDaytona,
    params: CreateSandboxFromSnapshotParams,
    timeout: float,
    *,
    deadline: float | None = None,
    wait_policy: ProviderWaitPolicy = ProviderWaitPolicy.PHASE,
) -> AsyncSandbox:
    """Recover transient or ambiguous creation under the caller-owned name."""
    assert params.name is not None

    async def create() -> AsyncSandbox:
        try:
            return await client.create(params, timeout=timeout)
        except DaytonaConflictError:
            sandbox = await client.get(params.name)
            await sandbox.wait_for_sandbox_start(timeout=timeout)
            return sandbox

    return await retry_provider_call("sandbox.create", create, deadline=deadline, wait_policy=wait_policy)


async def _delete_named_sandbox(
    client: AsyncDaytona, name: str, wait_policy: ProviderWaitPolicy = ProviderWaitPolicy.PHASE
) -> None:
    async def delete() -> None:
        try:
            sandbox = await client.get(name)
        except DaytonaNotFoundError:
            return
        await client.delete(sandbox)

    async with wait_policy.timeout(INTERRUPT_TIMEOUT):
        if wait_policy is ProviderWaitPolicy.JOB:
            await retry_provider_call(
                "sandbox.delete", delete, wait_policy=wait_policy, request_timeout=PROVIDER_REQUEST_TIMEOUT
            )
        else:
            await delete()


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


@dataclass(frozen=True)
class _Closing:
    task: asyncio.Task[None]
    cancel_provider_waits: Callable[[], None]


class DaytonaMachine:
    """One Daytona sandbox. Each command gets a fresh process and shared files."""

    def __init__(
        self,
        sandbox: AsyncSandbox,
        spec: MachineSpec,
        resources: AsyncExitStack,
        provider_wait_policy: ProviderWaitPolicy = ProviderWaitPolicy.PHASE,
    ):
        self.sandbox = sandbox
        self.spec = spec
        self.resources = resources
        self.provider_wait_policy = provider_wait_policy
        self._closed = False
        self._sessions: set[DaytonaProcessSession] = set()
        self._acquisitions: set[asyncio.Task[None]] = set()
        self._closing: _Closing | None = None

    async def _provider_request[T](self, operation: str, request: Callable[[], Awaitable[T]]) -> T:
        if self.provider_wait_policy is ProviderWaitPolicy.PHASE:
            return await request()
        return await retry_provider_call(
            operation, request, wait_policy=self.provider_wait_policy, request_timeout=PROVIDER_REQUEST_TIMEOUT
        )

    async def _control_exec(self, command: str, *, cwd: str = CONTROL_CWD, timeout: float | None = None):
        timeout = PROVIDER_REQUEST_TIMEOUT if self.provider_wait_policy is ProviderWaitPolicy.JOB else timeout
        return await self._provider_request(
            "sandbox.control", partial(self.sandbox.process.exec, command, cwd=cwd, timeout=timeout)
        )

    async def _user_argv(self, argv: tuple[str, ...], user: str | None) -> tuple[str, ...]:
        if user in (None, "root", "0"):
            return argv
        assert user is not None
        if user.isdecimal():
            account = await self._control_exec(_control_command(f"getent passwd {shlex.quote(user)}"), cwd=CONTROL_CWD)
            if account.exit_code or not account.result.strip():
                raise ValueError(f"Execution user {user} has no guest account")
            user = account.result.split(":", 1)[0]
        capabilities = await self._control_exec(_control_command("su --help"), cwd=CONTROL_CWD)
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
            async with self.provider_wait_policy.timeout(INTERRUPT_TIMEOUT):
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
        count = await self._control_exec(
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
            data = await self._provider_request("artifact.read", partial(self.sandbox.fs.download_file, path))
            assert isinstance(data, bytes)
            return data, False
        clipped = f"{path}.limited"
        result = await self._control_exec(
            _control_command(f"head -c {limit} {shlex.quote(path)} > {shlex.quote(clipped)}"), cwd=CONTROL_CWD
        )
        if result.exit_code:
            raise RuntimeError(f"Failed to limit command output: {result.result}")
        try:
            data = await self._provider_request("artifact.read", partial(self.sandbox.fs.download_file, clipped))
            assert isinstance(data, bytes)
            return data, True
        finally:
            await self._control_exec(_control_command(f"rm -f {shlex.quote(clipped)}"), cwd=CONTROL_CWD)

    async def _stop_command(self, pidfile: str, completed_path: str) -> None:
        with self.provider_wait_policy.wait():
            await self._stop_command_once(pidfile, completed_path)

    async def _stop_command_once(self, pidfile: str, completed_path: str) -> None:
        # An ambiguous kill response requires sandbox teardown because the PID may be reused.
        async with asyncio.timeout(INTERRUPT_TIMEOUT):
            stopped = await self.sandbox.process.exec(
                _control_command(shlex.join(("/bin/sh", "-c", STOP_COMMAND, "stop-command", pidfile, completed_path))),
                cwd=CONTROL_CWD,
            )
            if stopped.exit_code:
                raise RuntimeError(stopped.result)

    async def _remove_command_files(self, prefix: str) -> None:
        async with self.provider_wait_policy.timeout(INTERRUPT_TIMEOUT):
            removed = await self._control_exec(_control_command(f"rm -rf {shlex.quote(prefix)}"), cwd=CONTROL_CWD)
            if removed.exit_code:
                raise RuntimeError("Cannot remove Daytona command files")

    async def _close_preserving_failure(self, failure: BaseException) -> None:
        try:
            await self.close()
        except Exception as cleanup_error:
            failure.add_note(f"Daytona machine cleanup failed: {cleanup_error!r}")

    async def _cleanup_cancelled_command(
        self, prefix: str, pidfile: str, completed_path: str, started: bool, cancellation: asyncio.CancelledError
    ) -> None:
        try:
            if started:
                await finish_owned(own_cleanup(self._stop_command(pidfile, completed_path)))
                await finish_owned(own_cleanup(self._remove_command_files(prefix)))
            else:
                await finish_owned(own_cleanup(self.close()))
        except BaseException as cleanup_error:
            cancellation.add_note(f"Daytona command cancellation cleanup failed ({type(cleanup_error).__name__})")
            try:
                await finish_owned(own_cleanup(self.close()))
            except BaseException as close_error:
                cancellation.add_note(f"Daytona machine cleanup failed ({type(close_error).__name__})")

    async def open_process_scope(self) -> ProcessScope:
        raise UnsupportedMachineSpec("Daytona does not provide confirmed scoped descendant teardown")

    @provider_wait_scope()
    async def run(self, command: Command) -> Result:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        prefix = f"/tmp/.shellbox-{uuid.uuid4().hex}"
        stdin_path, stdout_path, stderr_path = (f"{prefix}/{part}" for part in ("in", "out", "err"))
        script_path = f"{prefix}/command.sh"
        pidfile, completed_path = f"{prefix}/pid", f"{prefix}/completed"
        started = False
        primary_error = None
        response = None
        environment = {**self.spec.env, **command.env}
        try:
            async with self.provider_wait_policy.timeout(command.timeout) as deadline:
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
                prepared = await self._control_exec(
                    _control_command(f"umask 077; mkdir -p {shlex.quote(prefix)}"), cwd=CONTROL_CWD
                )
                if prepared.exit_code:
                    raise RuntimeError(f"Cannot create private command directory: {prepared.result}")
                if command.stdin:
                    await self._provider_request(
                        "artifact.write", partial(self.sandbox.fs.upload_file_stream, command.stdin, stdin_path)
                    )
                # Nested shell quoting can exceed execve's single-argument limit
                # even when every original command argument fits.
                await self._provider_request(
                    "artifact.write", partial(self.sandbox.fs.upload_file_stream, script.encode(), script_path)
                )
                started = True
                response = await self.sandbox.process.exec(
                    _control_command(shlex.join(("/bin/sh", script_path))),
                    cwd=command.cwd or self.spec.workdir or None,
                    env=environment,
                    timeout=math.ceil(command.timeout + 10) if command.timeout is not None else None,
                )
            async with self.provider_wait_policy.timeout(INTERRUPT_TIMEOUT):
                limit = command.output_limit_bytes
                try:
                    stdout, stdout_truncated = await self._read_output(stdout_path, limit)
                    stderr, stderr_truncated = await self._read_output(stderr_path, limit)
                except Exception as error:
                    diagnostic = response.result or ""
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
            return Result(
                response.exit_code,
                stdout,
                stderr,
                stdout_truncated,
                stderr_truncated,
                ExitReason.EXITED,
            )
        except TimeoutError as error:
            if response is not None:
                primary_error = RuntimeError("Cannot collect Daytona command output")
                raise primary_error from error
            if not deadline.expired():
                primary_error = error
                raise
            if not started:
                return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
            try:
                await self._stop_command(pidfile, completed_path)
            except asyncio.CancelledError as cancellation:
                cancel_provider_waits()
                primary_error = cancellation
                await self._cleanup_cancelled_command(prefix, pidfile, completed_path, started, cancellation)
                raise
            except Exception as error:
                primary_error = RuntimeError("Cannot stop the Daytona command process group")
                await self._close_preserving_failure(primary_error)
                raise primary_error from error
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError as cancellation:
            cancel_provider_waits()
            primary_error = cancellation
            await self._cleanup_cancelled_command(prefix, pidfile, completed_path, started, cancellation)
            raise
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if not self._closed and not isinstance(primary_error, asyncio.CancelledError):
                try:
                    await self._remove_command_files(prefix)
                except asyncio.CancelledError as cancellation:
                    cancel_provider_waits()
                    await self._close_preserving_failure(cancellation)
                    raise
                except Exception as error:
                    if primary_error is None:
                        primary_error = RuntimeError("Cannot remove Daytona command files")
                        await self._close_preserving_failure(primary_error)
                        raise primary_error from error
                    primary_error.add_note(f"Daytona command cleanup failed: {error}")
                    await self._close_preserving_failure(primary_error)

    @provider_wait_scope()
    async def upload(self, source: Path, target: str) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if source.is_dir():
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                with tarfile.open(archive.name, "w:gz") as tar:
                    tar.add(source, arcname=".")
                remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
                await self._provider_request(
                    "artifact.write",
                    partial(self.sandbox.fs.upload_file_stream, Path(archive.name).read_bytes(), remote_archive),
                )
            try:
                result = await self._control_exec(
                    _control_command(
                        f"mkdir -p {shlex.quote(target)} && "
                        f"tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target)}"
                    ),
                    cwd=CONTROL_CWD,
                )
            finally:
                await self._control_exec(_control_command(f"rm -f {shlex.quote(remote_archive)}"))
        else:
            parent = str(PurePosixPath(target).parent)
            result = await self._control_exec(_control_command(f"mkdir -p {shlex.quote(parent)}"), cwd=CONTROL_CWD)
            if result.exit_code:
                raise RuntimeError(f"Failed to create {parent}: {result.result}")
            await self._provider_request(
                "artifact.write", partial(self.sandbox.fs.upload_file_stream, source.read_bytes(), target)
            )
            mode = stat.S_IMODE(source.stat().st_mode)
            result = await self._control_exec(_control_command(f"chmod {mode:o} {shlex.quote(target)}"), cwd=CONTROL_CWD)
        if result.exit_code:
            raise RuntimeError(f"Failed to upload {source}: {result.result}")

    @provider_wait_scope()
    async def download(self, source: str, target: Path) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
        probe = await self._control_exec(_control_command(f"test -d {shlex.quote(source)}"), cwd=CONTROL_CWD)
        if probe.exit_code == 0:
            remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
            result = await self._control_exec(
                _control_command(f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source)} ."), cwd=CONTROL_CWD
            )
            if result.exit_code:
                raise RuntimeError(f"Failed to archive {source}: {result.result}")
            try:
                data = await self._provider_request(
                    "artifact.read", partial(self.sandbox.fs.download_file, remote_archive)
                )
                assert isinstance(data, bytes)
                target.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                    Path(archive.name).write_bytes(data)
                    with tarfile.open(archive.name, "r:gz") as tar:
                        tar.extractall(target, filter="data")
            finally:
                await self._control_exec(_control_command(f"rm -f {shlex.quote(remote_archive)}"), cwd=CONTROL_CWD)
            return
        data = await self._provider_request("artifact.read", partial(self.sandbox.fs.download_file, source))
        assert isinstance(data, bytes)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def close(self) -> None:
        if self._closing is None:
            self._closed = True
            async with provider_wait_scope() as cancel_waits:
                self._closing = _Closing(own_cleanup(self._close()), cancel_waits)
        closing = self._closing
        try:
            await asyncio.shield(closing.task)
        except asyncio.CancelledError as error:
            closing.cancel_provider_waits()
            closing.task.cancel()
            try:
                await finish_owned(closing.task, timeout=INTERRUPT_TIMEOUT)
            except BaseException as cleanup_error:
                error.add_note(f"Daytona cleanup failed ({type(cleanup_error).__name__})")
            raise

    async def _close(self) -> None:
        try:
            await asyncio.gather(*tuple(self._acquisitions), return_exceptions=True)
            results = await asyncio.gather(
                *(session.close() for session in tuple(self._sessions)), return_exceptions=True
            )
        except asyncio.CancelledError:
            cancel_provider_waits()
            raise
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
        network_policy: DaytonaNetworkPolicy | None = None,
        provider_wait_policy: ProviderWaitPolicy = ProviderWaitPolicy.PHASE,
    ):
        self.client_factory = client_factory
        self.ttl_minutes = ttl_minutes
        self.create_timeout = create_timeout
        self.network_policy = network_policy
        self.provider_wait_policy = ProviderWaitPolicy(provider_wait_policy)

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

    async def _prepare(self, client: AsyncDaytona, spec: MachineSpec, deadline: float | None) -> str:
        self._check_spec(spec)
        return await retry_provider_call(
            "snapshot.prepare",
            partial(
                ensure_snapshot,
                client,
                snapshot_request_for_spec(spec),
                deadline=deadline,
                wait_policy=self.provider_wait_policy,
            ),
            deadline=deadline,
            wait_policy=self.provider_wait_policy,
        )

    @provider_wait_scope()
    async def prepare(self, spec: MachineSpec) -> str:
        """Prepare the snapshot used by create without starting a sandbox.

        Pin registry sources by digest to reuse immutable image content.
        """
        client_factory = self._clients()
        timeout = self.create_timeout if spec.startup_timeout is None else spec.startup_timeout
        async with client_factory() as client, self.provider_wait_policy.timeout(timeout) as deadline:
            return await self._prepare(
                client, spec, deadline.when() if self.provider_wait_policy is ProviderWaitPolicy.PHASE else None
            )

    @provider_wait_scope()
    async def create(self, spec: MachineSpec) -> DaytonaMachine:
        network = (
            {"network_block_all": spec.network is NetworkPolicy.DENY}
            if self.network_policy is None
            else self.network_policy.parameters()
        )
        async with AsyncExitStack() as lifetime:
            client = await lifetime.enter_async_context(self._clients()())
            timeout = self.create_timeout if spec.startup_timeout is None else spec.startup_timeout
            async with self.provider_wait_policy.timeout(timeout) as deadline:
                snapshot = await self._prepare(
                    client, spec, deadline.when() if self.provider_wait_policy is ProviderWaitPolicy.PHASE else None
                )
                name = f"shellbox-{uuid.uuid4().hex}"

                async def delete_sandbox(_error_type, error, _traceback):
                    try:
                        await _delete_named_sandbox(client, name, self.provider_wait_policy)
                    except Exception as cleanup_error:
                        if error is None:
                            raise
                        error.add_note(f"Daytona sandbox cleanup failed: {cleanup_error!r}")

                lifetime.push_async_exit(delete_sandbox)
                sandbox = await _create_sandbox(
                    client,
                    CreateSandboxFromSnapshotParams(
                        name=name,
                        snapshot=snapshot,
                        os_user="root",
                        env_vars=spec.env,
                        **network,
                        ttl_minutes=self.ttl_minutes,
                    ),
                    timeout=timeout,
                    deadline=deadline.when() if self.provider_wait_policy is ProviderWaitPolicy.PHASE else None,
                    wait_policy=self.provider_wait_policy,
                )
            machine = DaytonaMachine(sandbox, spec, lifetime, self.provider_wait_policy)
            prepared = await machine._control_exec(
                _control_command("command -v setsid"), cwd=CONTROL_CWD, timeout=INTERRUPT_TIMEOUT
            )
            if prepared.exit_code:
                raise UnsupportedMachineSpec("Daytona task images require setsid for command timeout recovery")
            if spec.workdir:
                result = await machine._control_exec(_control_command(shlex.join(("mkdir", "-p", spec.workdir))))
                if result.exit_code:
                    raise RuntimeError(f"Failed to create workdir {spec.workdir}: {result.result!r}")
            machine.resources = lifetime.pop_all()
            return machine
