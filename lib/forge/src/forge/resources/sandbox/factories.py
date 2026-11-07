# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, sandbox/factories.py and taskforge_adapters/backends.py.

"""Backend selection shared with task lowering."""

import shellsim
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineFactory
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, TaskSpec, VerifyitGrader, grades_with_judge

SANDBOX_SCHEDULING_TIMEOUT = 600


def shellsim_factory() -> MachineFactory:
    return ShellSimMachineFactory(context=shellsim.Context(pypi_catalog=shellsim.DEFAULT_PYPI_CATALOG))


def grading_environment(task: TaskSpec) -> EnvironmentRequirements | None:
    grader = task.grader
    if isinstance(grader, ScriptGrader | VerifyitGrader) and not grades_with_judge(grader):
        return grader.environment
    return None
