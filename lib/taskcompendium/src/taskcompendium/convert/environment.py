# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment requirements for graders that run an image in a fresh machine."""

from taskcompendium.models import CommandSemantics, EnvironmentRequirements


def grading_environment(image: str) -> EnvironmentRequirements:
    """A grader environment that runs the digest-pinned ``image`` with native Linux process semantics."""
    return EnvironmentRequirements(docker_image=image, command_semantics=CommandSemantics.LINUX_PROCESS)
