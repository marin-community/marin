# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ShellSim admission without initializing its execution context."""

from shellbox.machine import (
    SHELL_SIMULATOR_CAPABILITIES,
    MachineSpec,
    NetworkPolicy,
    PackageSpec,
    TaskRequirements,
    UnsupportedMachineSpec,
    check_execution_requirements,
)


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
    check_execution_requirements(requirements, spec, capabilities=SHELL_SIMULATOR_CAPABILITIES)
