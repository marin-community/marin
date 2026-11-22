# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine routing metadata and admission requirements."""

from shellbox.machine import UnsupportedMachineSpec

from taskcompendium.models import EnvironmentRequirements

ROUTE_BACKENDS_LABEL = "machine.route_backends"
"""The JSON task label holding the passing backend and its higher admitted candidates."""


def require_resolved_environment(requirements: EnvironmentRequirements) -> None:
    """Reject an unbuilt recipe before choosing or creating an execution environment."""
    if requirements.docker_build is not None:
        raise UnsupportedMachineSpec("Unresolved Docker build context must be built and pinned before execution")
