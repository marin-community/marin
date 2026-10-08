# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Docker images with the host's registered gVisor runtime."""

import io
import shlex
import tarfile
from pathlib import Path, PurePosixPath

from shellbox.backends.docker.machine import DockerMachine, DockerMachineFactory, docker
from shellbox.machine import Backend, MachineSpec


class GvisorMachine(DockerMachine):
    """A gVisor sandbox with file transfer into its isolated filesystem."""

    async def upload(self, source: Path, target: str) -> None:
        # Docker cp cannot see runsc's private overlay; transfer through the guest.
        path = PurePosixPath(target)
        directory = target if source.is_dir() else str(path.parent)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            archive.add(source, arcname="." if source.is_dir() else path.name)
        result = await docker(
            "exec",
            "-i",
            "--user",
            "0",
            self.name,
            "sh",
            "-c",
            f"mkdir -p {shlex.quote(directory)} && tar -xf - -C {shlex.quote(directory)}",
            stdin=stream.getvalue(),
        )
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))

    async def download(self, source: str, target: Path, *, max_bytes: int | None = None) -> None:
        if max_bytes is not None:
            await super().download(source, target, max_bytes=max_bytes)
            return
        probe = await docker("exec", "--user", "0", self.name, "test", "-d", source)
        if probe.exit_code == 0:
            result = await docker("exec", "--user", "0", self.name, "tar", "-cf", "-", "-C", source, ".")
            if result.exit_code:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
                archive.extractall(target, filter="data")
            return
        path = PurePosixPath(source)
        result = await docker(
            "exec", "--user", "0", self.name, "tar", "-cf", "-", "-C", str(path.parent), "--", path.name
        )
        if result.exit_code:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            member = archive.getmember(path.name).replace(name=target.name)
            archive.extract(member, target.parent, filter="data")


class GvisorMachineFactory(DockerMachineFactory):
    """Require a Docker daemon with its ``runsc`` runtime registered."""

    backend: Backend = Backend.GVISOR

    def __init__(
        self,
        *,
        skopeo: Path | None = None,
        image_cache: Path | None = None,
        authfile: Path | None = None,
        policy: Path | None = None,
    ):
        super().__init__(skopeo=skopeo, image_cache=image_cache, authfile=authfile, policy=policy, runtime="runsc")

    async def create(self, spec: MachineSpec) -> GvisorMachine:
        machine = await super().create(spec)
        return GvisorMachine(machine.name, machine.spec)
