# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine admission requirements shared by preparation and selection."""

from shellbox.machine import UnsupportedMachineSpec

from taskcompendium.models import EnvironmentRequirements


def require_resolved_environment(requirements: EnvironmentRequirements) -> None:
    """Reject an unbuilt recipe before choosing or creating an execution environment."""
    if requirements.docker_build is not None:
        raise UnsupportedMachineSpec("Unresolved Docker build context must be built and pinned before execution")
