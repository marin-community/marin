# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment requirements for graders that run an image in a fresh machine."""

from shellbox.machine import Backend

from taskcompendium.models import EnvironmentRequirements

IMAGE_BACKENDS = (Backend.GVISOR, Backend.DOCKER)
"""Backends that start a fresh machine from a digest-pinned image."""


def grading_environment(image: str) -> EnvironmentRequirements:
    """A grader environment that runs the digest-pinned ``image`` under gVisor or Docker."""
    return EnvironmentRequirements(docker_image=image, compatible_backends=IMAGE_BACKENDS)


def local_grading_environment(image: str) -> EnvironmentRequirements:
    """A grader environment that runs on a host providing the packages of the digest-pinned ``image``.

    The image names what the host must reproduce; the local backend never starts it.
    """
    return EnvironmentRequirements(docker_image=image, compatible_backends=(Backend.LOCAL,))
