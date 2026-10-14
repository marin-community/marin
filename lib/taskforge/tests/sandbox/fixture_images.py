# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A container machine factory for unit tests that need no Docker, and the grader packages that run on it."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace

from pydantic import JsonValue
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
from taskcompendium.grader import GraderPackage
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.resources import inline_resource

from taskforge.spec.draft import GRADER_RESOURCE_ROOT, VERDICT_REWARD, script_grader

SHELL_GRADER_ARGV = ("sh", f"{GRADER_RESOURCE_ROOT}/grade.sh")


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
