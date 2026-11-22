# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate selected machines and prepare their required software without acquiring them."""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import replace

from shellbox.image import DockerfileSource
from shellbox.machine import (
    Backend,
    HostImage,
    MachineFactory,
    MachineSpec,
    UnsupportedMachineSpec,
)

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.local import local_runtime

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def environment_identity(requirements: EnvironmentRequirements) -> tuple[str | None, str | None]:
    """Return the image identity and space-separated setup command digests.

    Each is None when the task has no corresponding image or setup commands.
    """
    image = requirements.docker_image
    if requirements.docker_build is not None:
        image = "build:sha256:" + canonical_sha256(requirements.docker_build.model_dump(mode="json"))
    setup = " ".join(hashlib.sha256(command.encode()).hexdigest() for command in requirements.setup_commands)
    return image, setup or None


def setup_user(backend: Backend) -> str | None:
    """Use the host user for local writable roots and root for image-owned directories."""
    return None if backend == Backend.LOCAL else "0"


def host_variables(environment: Mapping[str, str]) -> frozenset[str]:
    """The host variables that whole-value ``${NAME}`` and ``${NAME:-default}`` references read."""
    return frozenset(match[1] for value in environment.values() if (match := _TEMPLATE_PATTERN.fullmatch(value)))


def resolve_env_vars(environment: Mapping[str, str], host_environment: Mapping[str, str]) -> dict[str, str]:
    """Resolve whole-value ``${NAME}`` and ``${NAME:-default}`` references."""
    resolved = {}
    for key, value in environment.items():
        match = _TEMPLATE_PATTERN.fullmatch(value)
        if match is None:
            resolved[key] = value
            continue
        name, default = match.groups()
        if name in host_environment:
            resolved[key] = host_environment[name]
        elif default is not None:
            resolved[key] = default
        else:
            raise ValueError(f"Environment variable '{name}' not found in host environment")
    return resolved


def bound_machine_spec(requirements: EnvironmentRequirements, factory: MachineFactory, spec: MachineSpec) -> MachineSpec:
    """Bind package requirements and omit advisory local memory before feature checks."""
    return replace(
        spec,
        source=requirements.package_spec if requirements.package_spec is not None else spec.source,
        memory_mb=None if factory.backend == Backend.LOCAL else spec.memory_mb,
    )


def validate_machine_spec(requirements: EnvironmentRequirements, factory: MachineFactory, spec: MachineSpec) -> None:
    """Ask the selected factory to reject unsupported requirements before preparation.

    A Docker build context runs only on a machine built from a Dockerfile, which a factory that
    supports builds creates when it starts the machine.
    """
    if requirements.services or requirements.readiness is not None:
        raise UnsupportedMachineSpec("Service groups and readiness checks require a multi-container task runtime")
    if requirements.docker_build is not None and not isinstance(spec.source, DockerfileSource):
        raise UnsupportedMachineSpec("A Docker build context requires a machine built from its Dockerfile")
    factory.check_environment(requirements.requirements, bound_machine_spec(requirements, factory, spec))


def prepare_machine_spec(
    requirements: EnvironmentRequirements,
    factory: MachineFactory,
    spec: MachineSpec,
    host_environment: Mapping[str, str],
) -> MachineSpec:
    """Bind required software, working directory and variables to a selected machine.

    Lock preparation performs I/O. Async callers run it off-thread under their startup deadline.
    Resource installation, setup commands, machine creation and cleanup remain with the caller.
    """
    spec = bound_machine_spec(requirements, factory, spec)
    validate_machine_spec(requirements, factory, spec)
    required_variables = resolve_env_vars(requirements.environment_variables, host_environment)
    variables: Mapping[str, str] = {}
    if requirements.packages_lock is not None:
        source = spec.source
        if not isinstance(source, HostImage):
            raise UnsupportedMachineSpec("Package locks require a HostImage source")
        runtime = local_runtime(requirements.packages_lock)
        runtime.ensure_built()
        spec = replace(
            spec,
            source=HostImage(
                read_only=tuple(dict.fromkeys((*source.read_only, runtime.root))),
                bin_dirs=tuple(dict.fromkeys((runtime.bin_dir, *source.bin_dirs))),
                mounts=(*source.mounts, *runtime.mounts),
            ),
        )
        variables = runtime.env
    return replace(
        spec,
        workdir=requirements.working_directory if requirements.working_directory is not None else spec.workdir,
        env={**variables, **spec.env, **required_variables},
    )
