# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution capabilities that individual backends declare for themselves.

Each backend's admission module owns the capability set it can serve; there is no
central capability table. ``shellbox.machine.check_execution_requirements`` checks
a task's requirements against the declaring backend's set.
"""

from enum import StrEnum


class MachineCapability(StrEnum):
    SHELL = "shell"
    FILESYSTEM = "filesystem"
    PYTHON = "python"
    PROCESS = "process"
    NETWORK = "network"
    GIT_REPOSITORY = "git_repository"
    RETAINED_COMMAND_CANCELLATION = "retained_command_cancellation"
    PROCESS_SCOPE = "process_scope"
