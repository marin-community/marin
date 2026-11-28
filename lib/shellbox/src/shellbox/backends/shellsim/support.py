# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ShellSim admission without initializing its execution context."""

from shellbox.capabilities import MachineCapability
from shellbox.machine import (
    MachineSpec,
    NetworkPolicy,
    PackageSpec,
    TaskRequirements,
    UnsupportedMachineSpec,
    check_execution_requirements,
)

SHELLSIM_CAPABILITIES = frozenset(
    {
        MachineCapability.SHELL,
        MachineCapability.FILESYSTEM,
        MachineCapability.PYTHON,
    }
)
"""The ShellSim backend's declared execution capabilities."""


def package_source(spec: MachineSpec) -> PackageSpec:
    if spec.cpus is not None or spec.gpus:
        raise UnsupportedMachineSpec("ShellSim does not provide host CPU or GPU allocations")
    if not isinstance(spec.source, PackageSpec):
        raise UnsupportedMachineSpec("ShellSim requires a PackageSpec source")
    if spec.network is not NetworkPolicy.DENY:
        raise UnsupportedMachineSpec("ShellSim has no guest network")
    return spec.source


def check_environment(requirements: TaskRequirements, spec: MachineSpec) -> None:
    package_source(spec)
    check_execution_requirements(requirements, spec, capabilities=SHELLSIM_CAPABILITIES)
