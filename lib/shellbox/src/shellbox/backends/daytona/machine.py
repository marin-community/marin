# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the machine contract on a Daytona sandbox."""

import asyncio
import math
import shlex
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from daytona import AsyncDaytona, AsyncSandbox, CreateSandboxFromImageParams, Resources

from shellbox.image import RegistryImage
from shellbox.machine import Command, ExitReason, MachineSpec, NetworkPolicy, Result, UnsupportedMachineSpec

DEFAULT_SANDBOX_TTL_MINUTES = 360


class DaytonaMachine:
    """One Daytona sandbox. Each command gets a fresh process and shared files."""

    def __init__(self, client: AsyncDaytona, sandbox: AsyncSandbox, spec: MachineSpec):
        self.client = client
        self.sandbox = sandbox
        self.spec = spec
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
        stdin_path, stdout_path, stderr_path = (f"{prefix}-{part}" for part in ("in", "out", "err"))
        if command.stdin:
            await self.sandbox.fs.upload_file_stream(command.stdin, stdin_path)
        script = (
            f"{shlex.join(command.argv)} < {shlex.quote(stdin_path) if command.stdin else '/dev/null'} "
            f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
        )
        try:
            operation = self.sandbox.process.exec(
                script,
                cwd=command.cwd or self.spec.workdir,
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
            await self.close()
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError:
            await self.close()
            raise
        finally:
            if not self._closed:
                await self.sandbox.process.exec(
                    f"rm -f {shlex.quote(stdin_path)} {shlex.quote(stdout_path)} {shlex.quote(stderr_path)}"
                )

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
            return
        if result.exit_code:
            raise RuntimeError(f"Failed to upload {source}: {result.result}")

    async def download(self, source: str, target: Path) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
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
        await self.client.delete(self.sandbox)


class DaytonaMachineFactory:
    """Create a Daytona sandbox from a registry image reference."""

    def __init__(self, client: AsyncDaytona | None = None, *, ttl_minutes: int = DEFAULT_SANDBOX_TTL_MINUTES):
        self.client = client or AsyncDaytona()
        self.ttl_minutes = ttl_minutes

    async def create(self, spec: MachineSpec) -> DaytonaMachine:
        if not isinstance(spec.source, RegistryImage):
            raise UnsupportedMachineSpec("Daytona requires a registry image reference")
        memory_gb = math.ceil(spec.memory_mb / 1024) if spec.memory_mb is not None else None
        sandbox = await self.client.create(
            CreateSandboxFromImageParams(
                image=spec.source.reference,
                resources=Resources(memory=memory_gb) if memory_gb is not None else None,
                network_block_all=spec.network is NetworkPolicy.DENY,
                ttl_minutes=self.ttl_minutes,
            )
        )
        machine = DaytonaMachine(self.client, sandbox, spec)
        try:
            result = await machine.run(Command(("mkdir", "-p", spec.workdir), cwd="/"))
            if result.exit_code:
                raise RuntimeError(f"Failed to create workdir {spec.workdir}: {result.stderr!r}")
        except BaseException:
            await machine.close()
            raise
        return machine
