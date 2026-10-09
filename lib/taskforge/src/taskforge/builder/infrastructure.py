# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host failures inside a build, told apart from failures of the builder program.

Only errors the program cannot have caused are infrastructure:

* ``NO_FACTORY``: the build needs a machine backend this host has no factory for (a laptop without
  Docker).
* ``NO_IMAGE_BUILDER``: the build publishes a task image and this host has no ``ImageBuilder``.
* ``SCHEDULING_TIMEOUT``: the factory itself gave up waiting for a machine (Iris could not
  schedule the sandbox).
* ``HOST_UNREACHABLE``: a connection to the machine host failed, or the controller answered an RPC
  with a transport, capacity or credential error.
"""

from enum import StrEnum


class InfrastructureCause(StrEnum):
    """Why the host, not the program, failed a build."""

    NO_FACTORY = "no_factory"
    NO_IMAGE_BUILDER = "no_image_builder"
    SCHEDULING_TIMEOUT = "scheduling_timeout"
    HOST_UNREACHABLE = "host_unreachable"
