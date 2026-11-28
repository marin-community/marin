# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordered factory selection using each backend's admission checks."""

from dataclasses import dataclass
from typing import Protocol

from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    Backend,
    CommandSemantics,
    ExecutionRequirements,
    HostImage,
    MachineFactory,
    MachineSpec,
    NetworkPolicy,
    PackageSpec,
    TaskRequirements,
    UnsupportedMachineSpec,
)


class EnvironmentSource(ExecutionRequirements, Protocol):
    @property
    def packages_lock(self) -> str | None: ...

    @property
    def package_spec(self) -> PackageSpec | None: ...

    @property
    def working_directory(self) -> str | None: ...


def declared_spec(
    requirements: EnvironmentSource,
    network: NetworkPolicy,
    *,
    build_source: DockerfileSource | None = None,
    workspace: str,
) -> MachineSpec:
    """Bind the declared software and its default working directory without installing it.

    ``build_source`` supplies a prepared DockerfileSource for an unresolved build context.
    An image machine starts in the image's own working directory (an empty spec workdir);
    package and lock environments start in ``workspace`` — the same shared constant for
    curation and RL, so both admit a task against the same spec; a declared working
    directory wins.
    """
    if requirements.docker_image is not None:
        source, image = RegistryImage(requirements.docker_image), True
    elif build_source is not None:
        source, image = build_source, True
    elif requirements.packages_lock is not None:
        source, image = HostImage(), False
    elif requirements.package_spec is not None:
        source, image = requirements.package_spec, False
    elif requirements.command_semantics is CommandSemantics.SHELL_SIMULATOR:
        source, image = PackageSpec(), False
    else:
        raise UnsupportedMachineSpec("Machine execution requires declared software")
    workdir = requirements.working_directory
    if workdir is None:
        workdir = "" if image else workspace
    return MachineSpec(source, network=network, workdir=workdir)


@dataclass(frozen=True)
class MachineRoute:
    """A factory and its allocation inputs in a deployment's priority order."""

    factory: MachineFactory
    spec: MachineSpec

    @property
    def backend(self) -> Backend:
        return self.factory.backend


@dataclass(frozen=True)
class ShellboxContext:
    """Select factories statically from each backend's admission checks.

    The context owns ordered candidates and per-task admission only. Callers bind
    allocation inputs into each ``MachineRoute`` and acquire machines from the
    selected route at their own execution boundary; the context does not create
    machines until it owns an explicit preparation and lifecycle contract.
    """

    candidates: tuple[MachineRoute, ...]

    def routes(self, requirements: TaskRequirements) -> tuple[MachineRoute, ...]:
        selected = []
        for route in self.candidates:
            try:
                route.factory.check_environment(requirements, route.spec)
            except UnsupportedMachineSpec:
                continue
            selected.append(route)
        return tuple(selected)
