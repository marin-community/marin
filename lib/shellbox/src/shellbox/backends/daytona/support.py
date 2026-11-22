# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona admission without creating a provider client."""

import re

from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    MachineSpec,
    TaskRequirements,
    UnsupportedMachineSpec,
    check_execution_requirements,
)


def daytona_source(spec: MachineSpec) -> DockerfileSource | RegistryImage:
    if isinstance(spec.source, DockerfileSource):
        return validate_source(spec.source)
    if isinstance(spec.source, RegistryImage):
        return spec.source
    raise UnsupportedMachineSpec("Daytona requires a registry image or Docker build context")


def check_environment(requirements: TaskRequirements, spec: MachineSpec) -> None:
    source = daytona_source(spec)
    reference = source.reference if isinstance(source, RegistryImage) else None
    check_execution_requirements(requirements, spec, capabilities=LINUX_PROCESS_CAPABILITIES, image_reference=reference)


def validate_source(
    source: RegistryImage | DockerfileSource,
) -> RegistryImage | DockerfileSource:
    """Reject Daytona build inputs the provider cannot use."""
    if isinstance(source, RegistryImage):
        return source
    source = DockerfileSource(source.context.resolve(), source.dockerfile.resolve())
    if source.dockerfile.parent != source.context:
        raise UnsupportedMachineSpec("Daytona requires the Dockerfile at the build context root")
    if re.search(
        r"^\s*ADD(?:\s|$)",
        source.dockerfile.read_text(),
        re.IGNORECASE | re.MULTILINE,
    ):
        raise UnsupportedMachineSpec("Daytona Dockerfiles with ADD require a prebuilt registry image")
    return source
