# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordered factory selection using each backend's admission checks."""

from dataclasses import dataclass
from typing import Protocol

from shellbox.image import RegistryImage
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


def source_spec(requirements: EnvironmentSource, network: NetworkPolicy) -> MachineSpec:
    """Bind the declared software without installing it."""
    if requirements.docker_image is not None:
        return MachineSpec(RegistryImage(requirements.docker_image), network=network)
    if requirements.packages_lock is not None:
        return MachineSpec(HostImage(), network=network)
    if requirements.package_spec is not None:
        return MachineSpec(requirements.package_spec, network=network)
    if requirements.command_semantics is CommandSemantics.SHELL_SIMULATOR:
        return MachineSpec(PackageSpec(), network=network)
    raise UnsupportedMachineSpec("Machine execution requires declared software")


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
