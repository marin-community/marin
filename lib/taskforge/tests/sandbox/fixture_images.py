# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A container machine factory for unit tests that need no Docker."""

from dataclasses import dataclass, replace

from shellbox.backends.shellsim.machine import ShellSimMachine, ShellSimMachineFactory
from shellbox.machine import Backend, MachineSpec, ShellSimBuiltins


@dataclass(frozen=True)
class FixtureImageFactory:
    """Stand in for a container backend: every image's commands run on ShellSim's built-in filesystem.

    ShellSim has ``sh`` and ``python3``, so a grader program on the grader-base image or a task image runs
    unchanged; the image itself is never pulled.
    """

    backend: Backend = Backend.DOCKER

    async def create(self, spec: MachineSpec) -> ShellSimMachine:
        return await ShellSimMachineFactory().create(
            replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
        )
