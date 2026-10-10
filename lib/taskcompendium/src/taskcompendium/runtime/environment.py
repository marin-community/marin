# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate selected machines and prepare their required software without acquiring them."""

import json
import re
from collections.abc import Mapping
from dataclasses import replace

from shellbox.image import PreparedImage, RegistryImage
from shellbox.machine import (
    Backend,
    DockerImage,
    HostImage,
    MachineFactory,
    MachineSpec,
    NetworkPolicy,
    QemuBundle,
    ShellSimBuiltins,
    UnsupportedMachineSpec,
)

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, require_resolved_environment
from taskcompendium.runtime.local import local_runtime

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


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


def require_image(spec: MachineSpec, image: str) -> None:
    """Require the pinned image, including the source identity of a staged guest."""
    source = spec.source
    if source in (DockerImage(image), RegistryImage(image)):
        return
    if isinstance(source, PreparedImage) and source.source == RegistryImage(image):
        return
    if isinstance(source, QemuBundle):
        metadata = json.loads((source.path / "image.json").read_text())
        if metadata.get("image_reference") == image:
            return
    raise UnsupportedMachineSpec("Machine must use the task's pinned image")


def validate_machine_spec(requirements: EnvironmentRequirements, factory: MachineFactory, spec: MachineSpec) -> None:
    """Reject semantic incompatibilities before building software or acquiring a machine."""
    require_resolved_environment(requirements)
    if requirements.command_semantics == CommandSemantics.SHELL_SIMULATOR:
        if factory.backend != Backend.SHELLSIM or not isinstance(spec.source, ShellSimBuiltins):
            raise UnsupportedMachineSpec("Shell simulator semantics require ShellSim built-ins")
        if spec.network != NetworkPolicy.DENY:
            raise UnsupportedMachineSpec("The shell simulator has no guest network")
        return
    if requirements.command_semantics != CommandSemantics.LINUX_PROCESS:
        raise UnsupportedMachineSpec("Machine execution requires explicit command semantics")
    if factory.backend == Backend.SHELLSIM:
        raise UnsupportedMachineSpec("The shell simulator cannot execute native Linux processes")
    if requirements.docker_image is not None:
        if factory.backend == Backend.LOCAL:
            raise UnsupportedMachineSpec("The local factory cannot execute a required image")
        # Staged guest metadata is read during preparation, under the acquisition deadline.
        if not isinstance(spec.source, QemuBundle):
            require_image(spec, requirements.docker_image)
        return
    if requirements.packages_lock is not None:
        if factory.backend != Backend.LOCAL or not isinstance(spec.source, HostImage):
            raise UnsupportedMachineSpec("Package locks currently require a local HostImage machine")
        return
    raise UnsupportedMachineSpec("Native Linux execution requires a pinned image or package lock")


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
    validate_machine_spec(requirements, factory, spec)
    required_variables = resolve_env_vars(requirements.environment_variables, host_environment)
    if requirements.docker_image is not None:
        require_image(spec, requirements.docker_image)
    variables = {}
    if requirements.packages_lock is not None:
        runtime = local_runtime(requirements.packages_lock)
        runtime.ensure_built()
        source = spec.source
        if not isinstance(source, HostImage):
            raise UnsupportedMachineSpec("Package locks require a HostImage source")
        spec = replace(
            spec,
            source=HostImage(
                read_only=tuple(dict.fromkeys((*source.read_only, runtime.root))),
                bin_dirs=tuple(dict.fromkeys((runtime.bin_dir, *source.bin_dirs))),
            ),
        )
        variables = runtime.variables
    return replace(
        spec,
        workdir=requirements.working_directory if requirements.working_directory is not None else spec.workdir,
        env={**variables, **spec.env, **required_variables},
        # Memory is advisory for local grading; bubblewrap has no allocation API.
        memory_mb=None if factory.backend == Backend.LOCAL else spec.memory_mb,
    )
