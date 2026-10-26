# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker reference implementation of the machine contract."""

import asyncio
import logging
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shellbox.image import DockerfileSource, RegistryImage
from shellbox.lifecycle import CommandLifetime, finish_cleanup
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    Backend,
    Command,
    CommandSemantics,
    DockerImage,
    ExecutionRequirements,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    Result,
    UnsupportedMachineSpec,
    check_execution_requirements,
)
from shellbox.process import CLEANUP_TIMEOUT, ProcessScope

logger = logging.getLogger(__name__)
IMAGE_PLATFORM = "linux/amd64"

OUTPUT_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class DockerCommandResult:
    exit_code: int
    stdout: bytes
    stderr: bytes


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> bytes:
    retained = bytearray()
    while chunk := await stream.read(OUTPUT_READ_CHUNK_BYTES):
        retained.extend(chunk[: max(0, limit + 1 - len(retained))])
    return bytes(retained)


async def docker(
    *args: str,
    stdin: bytes = b"",
    timeout: float | None = None,
    output_limit_bytes: int | None = None,
) -> DockerCommandResult:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    assert process.stdout is not None and process.stderr is not None and process.stdin is not None
    try:
        async with asyncio.timeout(timeout):
            if output_limit_bytes is None:
                stdout, stderr = await process.communicate(stdin)
            else:
                async with asyncio.TaskGroup() as readers:
                    stdout_task = readers.create_task(_read_limited(process.stdout, output_limit_bytes))
                    stderr_task = readers.create_task(_read_limited(process.stderr, output_limit_bytes))
                    try:
                        process.stdin.write(stdin)
                        await process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        # A command can finish before it reads all of stdin.
                        pass
                    finally:
                        process.stdin.close()
                    await process.wait()
                stdout, stderr = stdout_task.result(), stderr_task.result()
    except BaseException:
        if process.returncode is None:
            process.kill()
        async with asyncio.TaskGroup() as readers:
            readers.create_task(_read_limited(process.stdout, 0))
            readers.create_task(_read_limited(process.stderr, 0))
            await process.wait()
        raise
    assert process.returncode is not None
    return DockerCommandResult(process.returncode, stdout, stderr)


class DockerMachine:
    """One Docker container with a writable filesystem."""

    def __init__(self, name: str, spec: MachineSpec):
        self.name = name
        self.spec = spec
        self._closed = False
        self._commands = CommandLifetime()

    async def run(self, command: Command) -> Result:
        with self._commands.admit():
            return await self._run(command)

    async def _run(self, command: Command) -> Result:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        args = ["exec", "-i"]
        workdir = command.cwd or self.spec.workdir
        if workdir:
            args.extend(("-w", workdir))
        if command.user is not None:
            args.extend(("--user", command.user))
        for key, value in command.env.items():
            args.extend(("-e", f"{key}={value}"))
        scope = ProcessScope()
        args.extend((self.name, *scope.command(command.argv, command.timeout)))
        try:
            completed = await docker(
                *args,
                stdin=command.stdin,
                timeout=command.timeout,
                output_limit_bytes=command.output_limit_bytes,
            )
        except (TimeoutError, asyncio.CancelledError) as interruption:
            try:
                await finish_cleanup(self._interrupt(scope, command.user))
            except Exception as error:
                await self.close()
                raise RuntimeError("Cannot confirm Docker command teardown") from error
            if isinstance(interruption, asyncio.CancelledError):
                raise
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except Exception:
            await self.close()
            raise
        try:
            status = await docker(
                "exec",
                *(("--user", command.user) if command.user is not None else ()),
                self.name,
                "/bin/sh",
                "-c",
                scope.status_command(),
                timeout=CLEANUP_TIMEOUT,
            )
            not_started = status.exit_code == 126 and completed.exit_code in (126, 127)
            if status.exit_code not in (0, 124) and not not_started:
                raise RuntimeError("Docker command teardown could not be confirmed")
        except BaseException:
            await self.close()
            raise
        timed_out = status.exit_code == 124
        limit = command.output_limit_bytes
        return Result(
            None if timed_out else completed.exit_code,
            completed.stdout[:limit],
            completed.stderr[:limit],
            len(completed.stdout) > limit,
            len(completed.stderr) > limit,
            ExitReason.TIMED_OUT if timed_out else ExitReason.EXITED,
        )

    async def _interrupt(self, scope: ProcessScope, user: str | None) -> None:
        async with asyncio.timeout(CLEANUP_TIMEOUT):
            result = await docker(
                "exec",
                *(("--user", user) if user is not None else ()),
                self.name,
                "/bin/sh",
                "-c",
                scope.cancel_command(),
            )
            if result.exit_code:
                raise RuntimeError(result.stderr.decode(errors="replace"))

    async def upload(self, source: Path, target: str) -> None:
        parent = str(PurePosixPath(target).parent)
        result = await docker("exec", "--user", "0", self.name, "mkdir", "-p", parent)
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        copy_source = f"{source}/." if source.is_dir() else str(source)
        if source.is_dir():
            result = await docker("exec", "--user", "0", self.name, "mkdir", "-p", target)
            if result.exit_code:
                raise RuntimeError(result.stderr.decode(errors="replace"))
        result = await docker("cp", copy_source, f"{self.name}:{target}")
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        if source.is_dir():
            mode = stat.S_IMODE(source.stat().st_mode)
            result = await docker("exec", "--user", "0", self.name, "chmod", f"{mode:o}", target)
            if result.exit_code:
                raise RuntimeError(result.stderr.decode(errors="replace"))

    async def download(self, source: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_source = f"{source}/." if target.is_dir() else source
        result = await docker("cp", f"{self.name}:{copy_source}", str(target))
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._commands.close()
        finally:
            result = await finish_cleanup(docker("rm", "-f", self.name, timeout=CLEANUP_TIMEOUT))
            if result.exit_code:
                raise RuntimeError(result.stderr.decode(errors="replace"))


class DockerMachineFactory:
    """Start a Docker image as a fresh trial container."""

    backend: Backend = Backend.DOCKER

    def __init__(self, *, runtime: str | None = None):
        self.runtime = runtime

    def _check_spec(self, spec: MachineSpec) -> DockerImage | RegistryImage | DockerfileSource:
        if not isinstance(spec.source, (DockerImage, RegistryImage, DockerfileSource)):
            raise UnsupportedMachineSpec("Docker requires a DockerImage, RegistryImage or DockerfileSource")
        return spec.source

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        source = self._check_spec(spec)
        reference = source.reference if isinstance(source, (DockerImage, RegistryImage)) else None
        check_execution_requirements(
            requirements,
            spec,
            semantics=CommandSemantics.LINUX_PROCESS,
            capabilities=LINUX_PROCESS_CAPABILITIES,
            image_reference=reference,
        )

    async def create(self, spec: MachineSpec) -> DockerMachine:
        source = self._check_spec(spec)
        image = source
        pull = "never"
        if isinstance(source, RegistryImage):
            pull = "missing"
            image = DockerImage(source.reference)
        elif isinstance(source, DockerfileSource):
            built = await docker(
                "build",
                "--quiet",
                "--platform",
                IMAGE_PLATFORM,
                "-f",
                str(source.dockerfile),
                str(source.context),
            )
            if built.exit_code:
                raise RuntimeError(built.stderr.decode(errors="replace"))
            image = DockerImage(built.stdout.decode().strip())
        assert isinstance(image, DockerImage)
        name = f"harbor-machine-{uuid.uuid4().hex}"
        args = [
            "run",
            "--rm",
            "--init",
            f"--pull={pull}",
            "-d",
            "--name",
            name,
            "--network",
            "none" if spec.network is NetworkPolicy.DENY else "bridge",
        ]
        if pull == "missing":
            args.extend(("--platform", IMAGE_PLATFORM))
        if spec.memory_mb is not None:
            args.extend(("--memory", f"{spec.memory_mb}m"))
        if spec.cpus is not None:
            args.extend(("--cpus", str(spec.cpus)))
        if spec.storage_mb is not None:
            args.extend(("--storage-opt", f"size={spec.storage_mb}M"))
        if spec.gpus:
            args.extend(("--gpus", str(spec.gpus)))
        if self.runtime is not None:
            args.extend(("--runtime", self.runtime))
        for key, value in spec.env.items():
            args.extend(("-e", f"{key}={value}"))
        args.extend(("--entrypoint", "/bin/sh", image.reference, "-c", "while :; do sleep 3600; done"))
        try:
            result = await docker(*args)
            if result.exit_code:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            prepared = await docker("exec", name, "/bin/sh", "-c", "command -v python3")
            if prepared.exit_code:
                raise UnsupportedMachineSpec("Docker task images require Python 3 for command teardown")
            # Commands run in the working directory, which the image need not contain; Iris machines create it too.
            if spec.workdir:
                created = await docker("exec", "--user", "0", name, "mkdir", "-p", spec.workdir)
                if created.exit_code:
                    raise RuntimeError(
                        f"Cannot create workdir {spec.workdir}: {created.stderr.decode(errors='replace')}"
                    )
        except BaseException:
            await docker("rm", "-f", name)
            raise
        return DockerMachine(name, spec)
