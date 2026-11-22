# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind campaign environments to ordered Shellbox factories."""

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

from shellbox.context import MachineRoute, ShellboxContext, source_spec
from shellbox.machine import Backend, MachineFactory, NetworkPolicy, PackageSpec
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.local import LocalGraderMachines
from taskcompendium.runtime.requirements import require_resolved_environment

IMAGE_MACHINE_CPUS = 4


@dataclass(frozen=True)
class CampaignMachines:
    """Bind declared software and semantics to caller-selected Shellbox factories."""

    image_factory: MachineFactory | None = None
    image_cpus: int = IMAGE_MACHINE_CPUS
    local: LocalGraderMachines = field(default_factory=LocalGraderMachines)
    simulator_catalog: str | None = None

    @cached_property
    def _simulator_factory(self) -> MachineFactory | None:
        try:
            import shellsim  # noqa: PLC0415
            from shellbox.backends.shellsim.machine import ShellSimMachineFactory  # noqa: PLC0415
        except ImportError:
            return None
        return ShellSimMachineFactory(
            context=shellsim.Context(
                pypi_catalog=self.simulator_catalog,
                trusted_precompiled=self.simulator_catalog in (None, shellsim.DEFAULT_PYPI_CATALOG),
            )
        )

    def identity(self) -> dict[str, Any]:
        factory = self.image_factory
        return {
            "image": (
                {
                    "backend": factory.backend.value,
                    "factory": f"{type(factory).__module__}.{type(factory).__qualname__}",
                    "cpus": self.image_cpus,
                }
                if factory is not None
                else None
            ),
            "local": self.local.identity(),
            "simulator": {
                "backend": Backend.SHELLSIM.value,
                "factory": "shellbox.backends.shellsim.machine.ShellSimMachineFactory",
                "catalog": self.simulator_catalog,
            },
            "network": NetworkPolicy.DENY.value,
        }

    def routes(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineRoute, ...]:
        """Bind supported factory candidates in deployment order with denied networking."""
        require_resolved_environment(environment)
        spec = source_spec(environment, NetworkPolicy.DENY)
        routes: list[MachineRoute] = []
        if isinstance(spec.source, PackageSpec):
            simulator = self._simulator_factory
            if simulator is not None:
                routes.append(MachineRoute(simulator, replace(spec, memory_mb=memory_mb)))
        routes.extend(self.local.routes(environment, memory_mb))
        if self.image_factory is not None:
            routes.append(MachineRoute(self.image_factory, replace(spec, memory_mb=memory_mb, cpus=self.image_cpus)))
        return ShellboxContext(tuple(routes)).routes(environment.requirements)
