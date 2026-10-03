# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker reference implementation of the machine contract."""

import asyncio
import uuid
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from shellbox.backends.docker.terminal import (
    DockerControlPlane,
    DockerMachineState,
    DockerTerminalReader,
    validate_linux_image,
)
from shellbox.image import DockerfileSource, PreparedImage, RegistryImage, load_docker_image, process_image_cache
from shellbox.machine import (
    Command,
    DockerImage,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    Result,
    UnsupportedMachineSpec,
)

MACHINE_CLEANUP_TIMEOUT = 30


@dataclass(frozen=True)
class DockerCommandResult:
    exit_code: int
    stdout: bytes
    stderr: bytes


async def docker(*args: str, stdin: bytes = b"", timeout: float | None = None) -> DockerCommandResult:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout=timeout)
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
        self.state = DockerMachineState.RUNNING
        self.control: DockerControlPlane | None = None

    async def run(self, command: Command) -> Result:
        if self.state is not DockerMachineState.RUNNING:
            raise RuntimeError("Docker machine is not running")
        if not command.argv:
            raise ValueError("Command argv is empty")
        args = ["exec", "-i"]
        workdir = command.cwd or self.spec.workdir
        if workdir:
            args.extend(("-w", workdir))
        if command.user is not None:
            args.extend(("--user", command.user))
        for key, value in command.env.items():
            args.extend(("-e", f"{key}={value}"))
        args.extend((self.name, *command.argv))
        try:
            completed = await docker(*args, stdin=command.stdin, timeout=command.timeout)
        except TimeoutError:
            # Stop all trial processes, retaining files for Harbor output recovery.
            await self._stop()
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError:
            await self._stop()
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

    async def upload(self, source: Path, target: str) -> None:
        if self.state is not DockerMachineState.RUNNING:
            raise RuntimeError("Docker machine is not running")
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
        if self.state is DockerMachineState.CLOSED:
            raise RuntimeError("Machine is closed")
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_source = f"{source}/." if target.is_dir() else source
        result = await docker("cp", f"{self.name}:{copy_source}", str(target))
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))

    async def read_file(self, path: str, max_bytes: int) -> bytes | None:
        if self.state is DockerMachineState.CLOSED:
            raise RuntimeError("Machine is closed")
        if self.state is DockerMachineState.STOP_FAILED:
            raise RuntimeError("Docker machine stop failed; terminal collection is unavailable")
        if self.control is None:
            raise UnsupportedMachineSpec("Terminal collection requires a local Docker control plane")
        return await DockerTerminalReader(self.name, self.spec.workdir, self.control, self.state).read_file(
            path, max_bytes
        )

    async def _stop(self) -> None:
        if self.state is not DockerMachineState.RUNNING:
            return
        self.state = DockerMachineState.STOP_FAILED
        result = await docker("stop", "--time", "0", self.name, timeout=MACHINE_CLEANUP_TIMEOUT)
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        self.state = DockerMachineState.STOPPED

    async def close(self) -> None:
        if self.state is DockerMachineState.CLOSED:
            return
        result = await docker("rm", "-f", self.name, timeout=MACHINE_CLEANUP_TIMEOUT)
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        self.state = DockerMachineState.CLOSED


class DockerMachineFactory:
    """Start a Docker image as a fresh trial container."""

    def __init__(
        self,
        *,
        skopeo: Path | None = None,
        image_cache: Path | None = None,
        authfile: Path | None = None,
        policy: Path | None = None,
        runtime: str | None = None,
        control: DockerControlPlane | None = None,
    ):
        self.skopeo = skopeo
        self.image_cache = image_cache
        self.authfile = authfile
        self.policy = policy
        self.runtime = runtime
        self.control = control

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
        if self.control is not None:
            await validate_linux_image(spec.source.reference)
        name = f"harbor-machine-{uuid.uuid4().hex}"
        args = [
            "run",
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
            result = await docker(*args, timeout=spec.startup_timeout)
            if result.exit_code:
                raise RuntimeError(result.stderr.decode(errors="replace"))
        except BaseException as error:
            cleanup = await docker("rm", "-f", name, timeout=MACHINE_CLEANUP_TIMEOUT)
            if cleanup.exit_code:
                raise RuntimeError(cleanup.stderr.decode(errors="replace")) from error
            raise
        machine = DockerMachine(name, spec)
        machine.control = self.control
        return machine
