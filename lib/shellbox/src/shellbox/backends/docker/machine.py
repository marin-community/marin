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
PROCESS_GROUP_PREFIX = b"SHELLBOX_PGID:"
PROCESS_GROUP_HEADER_LIMIT_BYTES = 64
RUN_COMMAND = f'printf "{PROCESS_GROUP_PREFIX.decode()}%s\\n" "$$"; exec "$@"'
INTERRUPT_TIMEOUT = 10
PROCESS_GROUP_PROBE_INTERVAL = 0.05
OUTPUT_READ_CHUNK_BYTES = 64 * 1024
KILL_PROCESS_GROUP_COMMAND = 'kill -KILL "-$1"'
PROCESS_GROUP_ABSENT_EXIT_CODE = 3
PROBE_PROCESS_GROUP_COMMAND = f"""
group=$1
[ -r /proc/1/stat ] || exit 1
for path in /proc/[0-9]*/stat; do
    record=
    if ! {{ while IFS= read -r line; do record=$record$line; done; }} < "$path"; then
        [ ! -e "$path" ] || exit 1
        continue
    fi
    [ "$record" ] || continue
    record=${{record##*) }}
    set -- $record
    [ "$#" -ge 3 ] || exit 1
    [ "$3" != "$group" ] || [ "$1" = Z ] || exit 0
done
exit {PROCESS_GROUP_ABSENT_EXIT_CODE}
"""


@dataclass(frozen=True)
class DockerCommandResult:
    exit_code: int
    stdout: bytes
    stderr: bytes


async def _read_limited(
    stream: asyncio.StreamReader, limit: int, process_group: asyncio.Future[int] | None = None
) -> bytes:
    header = bytearray()
    if process_group is not None:
        for _ in range(PROCESS_GROUP_HEADER_LIMIT_BYTES):
            byte = await stream.read(1)
            header.extend(byte)
            if byte in (b"", b"\n"):
                break
        value = header.removeprefix(PROCESS_GROUP_PREFIX).rstrip(b"\n")
        if header.startswith(PROCESS_GROUP_PREFIX) and header.endswith(b"\n") and value.isdigit() and int(value) > 1:
            process_group.set_result(int(value))
            header.clear()
    # Docker can send exec-start errors before the wrapper emits its header.
    retained = header[: limit + 1]
    while chunk := await stream.read(OUTPUT_READ_CHUNK_BYTES):
        retained.extend(chunk[: max(0, limit + 1 - len(retained))])
    return bytes(retained)


async def docker(
    *args: str,
    stdin: bytes = b"",
    timeout: float | None = None,
    output_limit_bytes: int | None = None,
    process_group: asyncio.Future[int] | None = None,
) -> DockerCommandResult:
    assert process_group is None or output_limit_bytes is not None
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
                    stdout_task = readers.create_task(_read_limited(process.stdout, output_limit_bytes, process_group))
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
        process_group: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        args.extend(
            (
                self.name,
                "/bin/sh",
                "-c",
                START_COMMAND,
                "shellbox-start",
                "/bin/sh",
                "-c",
                RUN_COMMAND,
                "shellbox-command",
                *command.argv,
            )
        )
        try:
            completed = await docker(
                *args,
                stdin=command.stdin,
                timeout=command.timeout,
                output_limit_bytes=command.output_limit_bytes,
                process_group=process_group,
            )
        except (TimeoutError, asyncio.CancelledError) as interruption:
            try:
                if not process_group.done():
                    raise RuntimeError("Docker exec did not supply a process-group ID")
                await self._interrupt(process_group.result(), command.user)
            except Exception:
                logger.exception("Cannot stop the Docker command process group")
                try:
                    await self.close()
                finally:
                    raise interruption
            if isinstance(interruption, asyncio.CancelledError):
                raise
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except Exception:
            await self.close()
            raise
        limit = command.output_limit_bytes
        return Result(
            completed.exit_code,
            completed.stdout[:limit],
            completed.stderr[:limit],
            len(completed.stdout) > limit,
            len(completed.stderr) > limit,
            ExitReason.EXITED,
        )

    async def _interrupt(self, process_group: int, user: str | None) -> None:
        args = ["exec"]
        if user is not None:
            args.extend(("--user", user))
        async with asyncio.timeout(INTERRUPT_TIMEOUT):
            result = await docker(
                *args, self.name, "/bin/sh", "-c", KILL_PROCESS_GROUP_COMMAND, "stop-command", str(process_group)
            )
            # A group signal can succeed while members with different UIDs remain alive.
            while True:
                probe = await docker(
                    "exec",
                    "--user",
                    "0",
                    self.name,
                    "/bin/sh",
                    "-c",
                    PROBE_PROCESS_GROUP_COMMAND,
                    "probe-command",
                    str(process_group),
                )
                if probe.exit_code == PROCESS_GROUP_ABSENT_EXIT_CODE:
                    return
                if probe.exit_code != 0 or result.exit_code != 0:
                    raise RuntimeError(probe.stderr.decode(errors="replace"))
                await asyncio.sleep(PROCESS_GROUP_PROBE_INTERVAL)

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
            prepared = await docker("exec", name, "/bin/sh", "-c", "command -v setsid")
            if prepared.exit_code:
                raise UnsupportedMachineSpec("Docker task images require setsid for command cancellation")
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
