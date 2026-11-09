# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, sandbox/factories.py and taskforge_adapters/backends.py.

"""Backend selection shared with task lowering."""

from enum import StrEnum

import shellsim
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, TaskSpec, VerifyitGrader, grades_with_judge

SANDBOX_SCHEDULING_TIMEOUT = 600


class MachineHost(StrEnum):
    LAPTOP = "laptop"
    IRIS = "iris"


def container_backend(where: MachineHost) -> Backend:
    return Backend.GVISOR if where is MachineHost.IRIS else Backend.DOCKER


def local_factory() -> MachineFactory:
    # bubblewrap is an optional Linux-only dependency.
    from shellbox.backends.local.machine import LocalMachineFactory, SystemDirectories  # noqa: PLC0415

    return LocalMachineFactory(system_directories=SystemDirectories.COPY_ON_WRITE)


def shellsim_factory() -> MachineFactory:
    return ShellSimMachineFactory(context=shellsim.Context(pypi_catalog=shellsim.DEFAULT_PYPI_CATALOG))


def grading_environment(task: TaskSpec) -> EnvironmentRequirements | None:
    grader = task.grader
    if isinstance(grader, ScriptGrader | VerifyitGrader) and not grades_with_judge(grader):
        return grader.environment
    return None
