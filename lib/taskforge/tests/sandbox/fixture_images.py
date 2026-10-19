# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine factories for unit tests that need no Docker or ShellSim port, and the grader packages that run on them."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from pydantic import JsonValue
from shellbox.backends.shellsim.machine import ShellSimMachine, ShellSimMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    Backend,
    Command,
    CommandSemantics,
    DockerImage,
    ExecutionRequirements,
    ExitReason,
    Machine,
    MachineSpec,
    PackageSpec,
    Result,
    check_execution_requirements,
)
from taskcompendium.grader import GraderPackage
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.resources import inline_resource

from taskforge.spec.draft import GRADER_RESOURCE_ROOT, PYTHON_GRADER_ARGV, VERDICT_REWARD, script_grader

SHELL_GRADER_ARGV = ("sh", f"{GRADER_RESOURCE_ROOT}/grade.sh")


class PortlessShellSimFactory(ShellSimMachineFactory):
    """ShellSim machines without the catalog's CPython that ``draft.requirements(image=None)`` asks for.

    TESTING.md leaves ShellSim's guest interpreter and ports to ShellSim's own suite, so the unit suite
    never fetches the port: a machine here has ShellSim's built-in shell tools only, and stays offline.
    """

    async def create(self, spec: MachineSpec) -> ShellSimMachine:
        if isinstance(spec.source, PackageSpec):
            spec = replace(spec, source=replace(spec.source, pypi_packages=None))
        return await super().create(spec)


@dataclass(frozen=True)
class FixtureImageFactory:
    """Stand in for a container backend: every image's commands run on ShellSim's built-in filesystem.

    A bare ShellSim machine has ``sh`` and ``jq`` but no ``python3`` (shellsim 0.1.36), so a grader
    program on the grader-base image or a task image is a ``sh`` script (``shell_grader``); the image
    itself is never pulled.
    """

    backend: Backend = Backend.DOCKER

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        """Check ``requirements`` as the Docker factory does, against the image ``spec`` names."""
        reference = spec.source.reference if isinstance(spec.source, DockerImage | RegistryImage) else None
        check_execution_requirements(
            requirements,
            spec,
            semantics=CommandSemantics.LINUX_PROCESS,
            capabilities=LINUX_PROCESS_CAPABILITIES,
            image_reference=reference,
        )

    async def create(self, spec: MachineSpec) -> ShellSimMachine:
        return await ShellSimMachineFactory().create(
            replace(spec, source=PackageSpec(), workdir=spec.workdir or "/workspace")
        )


@dataclass
class ShellGradedMachine:
    """A fixture machine that runs ``PYTHON_GRADER_ARGV`` (``python3 /tests/grade.py``) with ``sh``."""

    machine: Machine

    async def run(self, command: Command) -> Result:
        if command.argv == PYTHON_GRADER_ARGV:
            command = replace(command, argv=("sh", *PYTHON_GRADER_ARGV[1:]))
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


@dataclass(frozen=True)
class ShellGraderImageFactory:
    """``FixtureImageFactory`` machines that run Taskforge's grader command, ``python3 /tests/grade.py``, with ``sh``.

    The build template packages every script grader a model writes as that command, and a bare ShellSim
    machine has no python3. A test on these machines writes its grader programs in sh: it checks how the
    template packages, runs and reads a grader, not the language of the program.
    """

    images: FixtureImageFactory = FixtureImageFactory()

    @property
    def backend(self) -> Backend:
        return self.images.backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.images.check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> ShellGradedMachine:
        return ShellGradedMachine(await self.images.create(spec))


@dataclass
class SlowGraderMachine:
    """A fixture machine whose grader command may come back ``TIMED_OUT``, as ``SlowGraderFactory`` decides."""

    machine: Machine
    factory: "SlowGraderFactory"

    async def run(self, command: Command) -> Result:
        if command.argv in (PYTHON_GRADER_ARGV, SHELL_GRADER_ARGV):
            self.factory.grader_runs += 1
            if self.factory.grader_runs <= self.factory.timeouts:
                return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


@dataclass
class SlowGraderFactory:
    """``ShellGraderImageFactory`` machines on a sandbox too slow for the grader: the first ``timeouts`` grader
    commands, counted across every machine it creates, come back ``TIMED_OUT`` without running.

    ``machines`` and ``grader_runs`` count the machines created and the grader commands run.
    """

    timeouts: int
    machines: int = 0
    grader_runs: int = 0
    images: ShellGraderImageFactory = field(default_factory=ShellGraderImageFactory)

    @property
    def backend(self) -> Backend:
        return self.images.backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.images.check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> SlowGraderMachine:
        self.machines += 1
        return SlowGraderMachine(await self.images.create(spec), self)


def shell_grader(
    script: str,
    config: Mapping[str, JsonValue],
    *,
    environment: EnvironmentRequirements,
    answer_path: str | None,
    timeout: float,
) -> GraderPackage:
    """``python_grader``'s package with a ``sh`` program: ``script`` is ``/tests/grade.sh``, run as
    ``SHELL_GRADER_ARGV``, ``config`` its private ``/tests/config.json``, and it writes its verdict to
    ``GRADER_VERDICT_PATH``."""
    files = (
        inline_resource("grade.sh", script.encode()),
        inline_resource("config.json", json.dumps(dict(config), sort_keys=True).encode()),
    )
    return script_grader(
        SHELL_GRADER_ARGV,
        VERDICT_REWARD,
        environment=environment,
        answer_path=answer_path,
        timeout=timeout,
        files=files,
    )
