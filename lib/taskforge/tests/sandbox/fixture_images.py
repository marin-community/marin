# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A container machine factory for unit tests that need no Docker."""

from dataclasses import dataclass, replace

from shellbox.backends.shellsim.machine import ShellSimMachine, ShellSimMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    Backend,
    CommandSemantics,
    DockerImage,
    ExecutionRequirements,
    MachineSpec,
    PackageSpec,
    check_execution_requirements,
)


@dataclass(frozen=True)
class FixtureImageFactory:
    """Stand in for a container backend: every image's commands run on ShellSim's built-in filesystem.

    ShellSim has ``sh`` and ``python3``, so a grader program on the grader-base image or a task image runs
    unchanged; the image itself is never pulled.
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
