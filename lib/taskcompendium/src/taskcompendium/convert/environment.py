# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment requirements for graders that run an image in a fresh machine."""

from shellbox.machine import Backend

from taskcompendium.models import EnvironmentRequirements


def grading_environment(image: str, backends: tuple[Backend, ...]) -> EnvironmentRequirements:
    """A grader environment that runs the digest-pinned ``image`` on any of ``backends``."""
    return EnvironmentRequirements(docker_image=image, compatible_backends=backends)
