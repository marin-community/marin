# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the machine contract on a Daytona sandbox."""

import asyncio
import hashlib
import json
import math
import re
import shlex
import stat
import tarfile
import tempfile
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, aclosing
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import cast

from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    DaytonaConflictError,
    DaytonaNotFoundError,
    Image,
    Resources,
)
from daytona_api_client_async import SnapshotState
from rigging.timing import ExponentialBackoff

from shellbox.backends.docker.machine import INTERRUPT_TIMEOUT, KILL_PROCESS_GROUP_COMMAND, START_COMMAND
from shellbox.file_transfer import write_download
from shellbox.image import DockerfileSource, RegistryImage, image_source_key
from shellbox.machine import Backend, Command, ExitReason, MachineSpec, NetworkPolicy, Result, UnsupportedMachineSpec

DEFAULT_SANDBOX_TTL_MINUTES = 360
RUN_COMMAND = (
    'pidfile=$1; completed=$2; shift 2; echo $$ > "$pidfile"; '
    'trap \'touch "$completed"; rm -f "$pidfile"\' EXIT; "$@"'
)
STOP_COMMAND = (
    '[ -f "$1" ] || exit 1; read -r pid < "$1"; '
    'case "$pid" in ""|*[!0-9]*) exit 1;; esac; '
    '[ "$pid" -gt 1 ] || exit 1; set -- "$pid"; ' + KILL_PROCESS_GROUP_COMMAND
)


async def _snapshot(client: AsyncDaytona, source: RegistryImage | DockerfileSource, resources: Resources) -> str:
    """Get an active snapshot within the caller's startup deadline."""
    identity = f"{image_source_key(source)}\0{json.dumps(asdict(resources), sort_keys=True)}"
    name = f"shellbox-{hashlib.sha256(identity.encode()).hexdigest()[:40]}"
    try:
        snapshot = await client.snapshot.get(name)
    except DaytonaNotFoundError:
        image = source.reference if isinstance(source, RegistryImage) else Image.from_dockerfile(source.dockerfile)
        try:
            snapshot = await client.snapshot.create(CreateSnapshotParams(name=name, image=image, resources=resources))
        except DaytonaConflictError:
            # Another rollout can create this image through a separate client or event loop.
            snapshot = await client.snapshot.get(name)
    backoff = ExponentialBackoff(initial=1, maximum=10)
    while snapshot.state != SnapshotState.ACTIVE:
        if snapshot.state in (SnapshotState.ERROR, SnapshotState.BUILD_FAILED, SnapshotState.INACTIVE):
            raise RuntimeError(f"Daytona snapshot {name} is {snapshot.state}: {snapshot.error_reason}")
        await asyncio.sleep(backoff.next_interval())
        snapshot = await client.snapshot.get(name)
    return name


class DaytonaNetworkMode(StrEnum):
    BLOCK_ALL = "block_all"
    UNRESTRICTED = "unrestricted"
    NETWORK_ALLOW_LIST = "network_allow_list"
    DOMAIN_ALLOW_LIST = "domain_allow_list"


@dataclass(frozen=True)
class DaytonaNetworkPolicy:
    mode: DaytonaNetworkMode
    value: str | None = None

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

    def __init__(self, sandbox: AsyncSandbox, spec: MachineSpec, resources: AsyncExitStack):
        self.sandbox = sandbox
        self.spec = spec
        self.resources = resources
        self._closed = False

    async def _read_output(self, path: str, limit: int) -> tuple[bytes, bool]:
        count = await self.sandbox.process.exec(f"wc -c < {shlex.quote(path)}")
        if count.exit_code:
            raise RuntimeError(f"Failed to measure command output: {count.result}")
        size = int(count.result.strip())
        if size <= limit:
            data = await self.sandbox.fs.download_file(path)
            assert isinstance(data, bytes)
            return data, False
        clipped = f"{path}.limited"
        result = await self.sandbox.process.exec(f"head -c {limit} {shlex.quote(path)} > {shlex.quote(clipped)}")
        if result.exit_code:
            raise RuntimeError(f"Failed to limit command output: {result.result}")
        try:
            data = await self.sandbox.fs.download_file(clipped)
            assert isinstance(data, bytes)
            return data, True
        finally:
            await self.sandbox.process.exec(f"rm -f {shlex.quote(clipped)}")

    async def run(self, command: Command) -> Result:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        prefix = f"/tmp/.shellbox-{uuid.uuid4().hex}"
        stdin_path, stdout_path, stderr_path = (f"{prefix}/{part}" for part in ("in", "out", "err"))
        pidfile, completed_path = f"{prefix}/pid", f"{prefix}/completed"
        argv = command.argv
        if command.user not in (None, "root", "0"):
            user = command.user
            # su accepts names. Resolve a numeric UID in the guest's account database.
            if user.isdecimal():
                account = await self.sandbox.process.exec(f"getent passwd {shlex.quote(user)}")
                if account.exit_code or not account.result.strip():
                    raise ValueError(f"Execution user {user} has no guest account")
                user = account.result.split(":", 1)[0]
            capabilities = await self.sandbox.process.exec("su --help")
            if capabilities.exit_code or "--session-command" not in capabilities.result:
                raise UnsupportedMachineSpec("Non-root Daytona commands require util-linux su --session-command")
            argv = ("su", "-s", "/bin/sh", "-m", user, "--session-command", shlex.join(argv))
        argv = (
            "sh",
            "-c",
            START_COMMAND,
            "shellbox-start",
            "sh",
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
        try:
            prepared = await self.sandbox.process.exec(f"umask 077; mkdir {shlex.quote(prefix)}")
            if prepared.exit_code:
                raise RuntimeError(f"Cannot create private command directory: {prepared.result}")
            if command.stdin:
                await self.sandbox.fs.upload_file_stream(command.stdin, stdin_path)
            operation = self.sandbox.process.exec(
                script,
                cwd=command.cwd or self.spec.workdir or None,
                env={**self.spec.env, **command.env},
                timeout=math.ceil(command.timeout + 10) if command.timeout is not None else None,
            )
            response = await asyncio.wait_for(operation, timeout=command.timeout)
            limit = command.output_limit_bytes
            stdout, stdout_truncated = await self._read_output(stdout_path, limit)
            stderr, stderr_truncated = await self._read_output(stderr_path, limit)
            return Result(
                response.exit_code,
                stdout,
                stderr,
                stdout_truncated,
                stderr_truncated,
                ExitReason.EXITED,
            )
        except TimeoutError:
            try:
                async with asyncio.timeout(INTERRUPT_TIMEOUT):
                    stopped = await self.sandbox.process.exec(
                        f"if [ -f {shlex.quote(completed_path)} ]; then exit 0; fi; "
                        + shlex.join(("sh", "-c", STOP_COMMAND, "stop-command", pidfile))
                    )
                    if stopped.exit_code:
                        raise RuntimeError(stopped.result)
            except Exception as error:
                await self.close()
                raise RuntimeError("Cannot stop the Daytona command process group") from error
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError:
            await self.close()
            raise
        finally:
            if not self._closed:
                await self.sandbox.process.exec(f"rm -rf {shlex.quote(prefix)}")

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
                f"mkdir -p {shlex.quote(target)} && tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target)}"
                f"; status=$?; rm -f {shlex.quote(remote_archive)}; exit $status"
            )
        else:
            parent = str(PurePosixPath(target).parent)
            result = await self.sandbox.process.exec(f"mkdir -p {shlex.quote(parent)}")
            if result.exit_code:
                raise RuntimeError(f"Failed to create {parent}: {result.result}")
            await self.sandbox.fs.upload_file_stream(source.read_bytes(), target)
            mode = stat.S_IMODE(source.stat().st_mode)
            result = await self.sandbox.process.exec(f"chmod {mode:o} {shlex.quote(target)}")
        if result.exit_code:
            raise RuntimeError(f"Failed to upload {source}: {result.result}")

    async def download(self, source: str, target: Path, *, max_bytes: int | None = None) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if max_bytes is not None:
            regular = await self.sandbox.process.exec(f"test -f {shlex.quote(source)}")
            if regular.exit_code != 0:
                raise UnsupportedMachineSpec("A download byte limit requires a regular file")
            # The pinned SDK returns an async generator with an AsyncIterator annotation.
            stream = cast(AsyncGenerator[bytes, None], await self.sandbox.fs.download_file_stream(source))
            async with aclosing(stream):
                await write_download(stream, target, max_bytes)
            return
        probe = await self.sandbox.process.exec(f"test -d {shlex.quote(source)}")
        if probe.exit_code == 0:
            remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
            result = await self.sandbox.process.exec(f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source)} .")
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
                await self.sandbox.process.exec(f"rm -f {shlex.quote(remote_archive)}")
            return
        data = await self.sandbox.fs.download_file(source)
        assert isinstance(data, bytes)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.resources.aclose()


class DaytonaMachineFactory:
    """Create a Daytona sandbox from a registry image or Docker build context."""

    backend: Backend = Backend.DAYTONA

    def __init__(
        self,
        client_factory: Callable[[], AsyncDaytona] = AsyncDaytona,
        *,
        ttl_minutes: int = DEFAULT_SANDBOX_TTL_MINUTES,
        create_timeout: float = 600,
        network_policy: DaytonaNetworkPolicy | None = None,
    ):
        self.client_factory = client_factory
        self.ttl_minutes = ttl_minutes
        self.create_timeout = create_timeout
        self.network_policy = network_policy

    async def create(self, spec: MachineSpec) -> DaytonaMachine:
        source: DockerfileSource | RegistryImage
        if isinstance(spec.source, DockerfileSource):
            source = DockerfileSource(spec.source.context.resolve(), spec.source.dockerfile.resolve())
            if source.dockerfile.parent != source.context:
                raise UnsupportedMachineSpec("Daytona requires the Dockerfile at the build context root")
            if re.search(r"^\s*ADD(?:\s|$)", source.dockerfile.read_text(), re.IGNORECASE | re.MULTILINE):
                raise UnsupportedMachineSpec("Daytona Dockerfiles with ADD require a prebuilt registry image")
        elif isinstance(spec.source, RegistryImage):
            source = spec.source
        else:
            raise UnsupportedMachineSpec("Daytona requires a registry image or Docker build context")
        resources = Resources(
            cpu=spec.cpus,
            memory=math.ceil(spec.memory_mb / 1024) if spec.memory_mb is not None else None,
            disk=math.ceil(spec.storage_mb / 1024) if spec.storage_mb is not None else None,
            gpu=spec.gpus or None,
        )
        network = (
            {"network_block_all": spec.network is NetworkPolicy.DENY}
            if self.network_policy is None
            else self.network_policy.parameters()
        )
        async with AsyncExitStack() as lifetime:
            client = await lifetime.enter_async_context(self.client_factory())
            timeout = self.create_timeout if spec.startup_timeout is None else spec.startup_timeout
            async with asyncio.timeout(timeout):
                snapshot = await _snapshot(client, source, resources)
                sandbox = await client.create(
                    CreateSandboxFromSnapshotParams(
                        snapshot=snapshot,
                        os_user="root",
                        env_vars=spec.env,
                        **network,
                        ttl_minutes=self.ttl_minutes,
                    ),
                    timeout=timeout,
                )
            lifetime.push_async_callback(client.delete, sandbox)
            prepared = await sandbox.process.exec("command -v setsid", timeout=INTERRUPT_TIMEOUT)
            if prepared.exit_code:
                raise UnsupportedMachineSpec("Daytona task images require setsid for command timeout recovery")
            machine = DaytonaMachine(sandbox, spec, lifetime)
            if spec.workdir:
                result = await machine.run(Command(("mkdir", "-p", spec.workdir), cwd="/"))
                if result.exit_code:
                    raise RuntimeError(f"Failed to create workdir {spec.workdir}: {result.stderr!r}")
            machine.resources = lifetime.pop_all()
            return machine
