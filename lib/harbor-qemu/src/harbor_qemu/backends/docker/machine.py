# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker reference implementation of the machine contract."""

import asyncio
import uuid
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from harbor_qemu.image import DockerfileSource, PreparedImage, RegistryImage, load_docker_image, process_image_cache
from harbor_qemu.machine import (
    Command,
    DockerImage,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    Result,
    UnsupportedMachineSpec,
)


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
        self._closed = False

    async def run(self, command: Command) -> Result:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if not command.argv:
            raise ValueError("Command argv is empty")
        args = ["exec", "-i", "-w", command.cwd or self.spec.workdir]
        for key, value in command.env.items():
            args.extend(("-e", f"{key}={value}"))
        args.extend((self.name, *command.argv))
        try:
            completed = await docker(*args, stdin=command.stdin, timeout=command.timeout)
        except TimeoutError:
            # docker exec has no reliable process-tree cancellation; dispose of the trial.
            await self.close()
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError:
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

    async def upload(self, source: Path, target: str) -> None:
        parent = str(PurePosixPath(target).parent)
        result = await docker("exec", self.name, "mkdir", "-p", parent)
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        result = await docker("cp", str(source), f"{self.name}:{target}")
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))

    async def download(self, source: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        result = await docker("cp", f"{self.name}:{source}", str(target))
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        result = await docker("rm", "-f", self.name)
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))


class DockerMachineFactory:
    """Start a Docker image as a fresh trial container."""

    def __init__(
        self,
        *,
        skopeo: Path | None = None,
        image_cache: Path | None = None,
        authfile: Path | None = None,
        policy: Path | None = None,
    ):
        self.skopeo = skopeo
        self.image_cache = image_cache
        self.authfile = authfile
        self.policy = policy

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
            "--pull=never",
            "-d",
            "--name",
            name,
            "--network",
            "none" if spec.network is NetworkPolicy.DENY else "bridge",
        ]
        if spec.memory_mb is not None:
            args.extend(("--memory", f"{spec.memory_mb}m"))
        for key, value in spec.env.items():
            args.extend(("-e", f"{key}={value}"))
        args.extend(("--entrypoint", "/bin/sh", spec.source.reference, "-c", "while :; do sleep 3600; done"))
        result = await docker(*args)
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return DockerMachine(name, spec)
