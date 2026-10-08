# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker reference implementation of the machine contract."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from shellbox.image import DockerfileSource, PreparedImage, RegistryImage, load_docker_image, process_image_cache
from shellbox.machine import (
    Backend,
    Command,
    DockerImage,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    Result,
    UnsupportedMachineSpec,
)

logger = logging.getLogger(__name__)

# The parent waits so setsid is not a process-group leader and cannot detach.
# Keep stdin available because non-interactive shells redirect background jobs to /dev/null.
START_COMMAND = 'exec 3<&0; setsid "$@" <&3 3<&- & wait "$!"'
RUN_COMMAND = 'pidfile=$1; shift; echo $$ > "$pidfile"; ' 'trap \'rm -f "$pidfile"\' EXIT; "$@"'
INTERRUPT_TIMEOUT = 10
OUTPUT_READ_CHUNK_BYTES = 64 * 1024
STOP_COMMAND = (
    '[ -f "$1" ] || exit 1; read -r pid < "$1"; '
    'case "$pid" in ""|*[!0-9]*) exit 1;; esac; '
    '[ "$pid" -gt 1 ] || exit 1; '
    'kill -KILL "-$pid" || exit 1; '
    'rm -f "$1"'
)


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
    *args: str, stdin: bytes = b"", timeout: float | None = None, output_limit_bytes: int | None = None
) -> DockerCommandResult:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        async with asyncio.timeout(timeout):
            if output_limit_bytes is None:
                stdout, stderr = await process.communicate(stdin)
            else:
                assert process.stdout is not None and process.stderr is not None and process.stdin is not None
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
        process.kill()
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

    async def run(self, command: Command) -> Result:
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
        pidfile = f"/tmp/.shellbox-command-{uuid.uuid4().hex}"
        args.extend(
            (
                self.name,
                "sh",
                "-c",
                START_COMMAND,
                "shellbox-start",
                "sh",
                "-c",
                RUN_COMMAND,
                "shellbox-command",
                pidfile,
                *command.argv,
            )
        )
        try:
            completed = await docker(
                *args, stdin=command.stdin, timeout=command.timeout, output_limit_bytes=command.output_limit_bytes
            )
        except (TimeoutError, asyncio.CancelledError) as interruption:
            try:
                await self._interrupt(pidfile)
            except Exception:
                logger.exception("Cannot stop the Docker command process group")
                try:
                    await self.close()
                finally:
                    raise interruption
            if isinstance(interruption, asyncio.CancelledError):
                raise
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        limit = command.output_limit_bytes
        return Result(
            completed.exit_code,
            completed.stdout[:limit],
            completed.stderr[:limit],
            len(completed.stdout) > limit,
            len(completed.stderr) > limit,
            ExitReason.EXITED,
        )

    async def _interrupt(self, pidfile: str) -> None:
        result = await docker(
            "exec",
            "--user",
            "0",
            self.name,
            "sh",
            "-c",
            STOP_COMMAND,
            "stop-command",
            pidfile,
            timeout=INTERRUPT_TIMEOUT,
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
        result = await docker("rm", "-f", self.name, timeout=INTERRUPT_TIMEOUT)
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))


class DockerMachineFactory:
    """Start a Docker image as a fresh trial container."""

    backend: Backend = Backend.DOCKER

    def __init__(
        self,
        *,
        skopeo: Path | None = None,
        image_cache: Path | None = None,
        authfile: Path | None = None,
        policy: Path | None = None,
        runtime: str | None = None,
    ):
        self.skopeo = skopeo
        self.image_cache = image_cache
        self.authfile = authfile
        self.policy = policy
        self.runtime = runtime

    async def create(self, spec: MachineSpec) -> DockerMachine:
        if isinstance(spec.source, (RegistryImage, DockerfileSource)):
            if self.image_cache is None or self.skopeo is None:
                raise UnsupportedMachineSpec("Registry images and Dockerfiles require an image cache and Skopeo")
            cache = process_image_cache(self.image_cache, self.skopeo, self.authfile, self.policy)
            image = await asyncio.to_thread(cache.prepare, spec.source)
            spec = replace(spec, source=image)
        if isinstance(spec.source, PreparedImage):
            if self.skopeo is None:
                raise UnsupportedMachineSpec("Prepared OCI images require a Skopeo path for Docker")
            reference = await asyncio.to_thread(load_docker_image, spec.source, skopeo=self.skopeo, policy=self.policy)
            spec = replace(spec, source=DockerImage(reference))
        if not isinstance(spec.source, DockerImage):
            raise UnsupportedMachineSpec("Docker requires a DockerImage source")
        name = f"harbor-machine-{uuid.uuid4().hex}"
        args = [
            "run",
            "--rm",
            "--init",
            "--pull=never",
            "-d",
            "--name",
            name,
            "--network",
            "none" if spec.network is NetworkPolicy.DENY else "bridge",
        ]
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
        args.extend(("--entrypoint", "/bin/sh", spec.source.reference, "-c", "while :; do sleep 3600; done"))
        try:
            result = await docker(*args)
            if result.exit_code:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            prepared = await docker("exec", name, "sh", "-c", "command -v setsid")
            if prepared.exit_code:
                raise UnsupportedMachineSpec("Docker task images require setsid for command cancellation")
        except BaseException:
            await docker("rm", "-f", name)
            raise
        return DockerMachine(name, spec)
