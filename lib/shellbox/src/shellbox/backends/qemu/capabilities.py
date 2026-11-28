# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The QEMU backend's declared execution capabilities."""

from shellbox.capabilities import MachineCapability

QEMU_CAPABILITIES = frozenset(
    {
        MachineCapability.SHELL,
        MachineCapability.FILESYSTEM,
        MachineCapability.PYTHON,
        MachineCapability.PROCESS,
        MachineCapability.NETWORK,
        MachineCapability.GIT_REPOSITORY,
    }
)
