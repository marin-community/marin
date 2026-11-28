# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local sandbox admission without probing or allocating a sandbox."""

from pathlib import Path

from shellbox.backends.local.cgroup import check_cgroup_parent
from shellbox.capabilities import MachineCapability
from shellbox.image import PreparedImage, RegistryImage, is_digest_pinned_image
from shellbox.machine import (
    HostImage,
    MachineSpec,
    PackageSpec,
    TaskRequirements,
    UnsupportedMachineSpec,
    check_execution_requirements,
)

LOCAL_CAPABILITIES = frozenset(
    {
        MachineCapability.SHELL,
        MachineCapability.FILESYSTEM,
        MachineCapability.PYTHON,
        MachineCapability.PROCESS,
        MachineCapability.NETWORK,
        MachineCapability.GIT_REPOSITORY,
    }
)
"""The local backend's declared execution capabilities without a cgroup unit."""


def local_source(spec: MachineSpec) -> HostImage | PackageSpec | PreparedImage | RegistryImage:
    if not isinstance(spec.source, (HostImage, PackageSpec, PreparedImage, RegistryImage)):
        raise UnsupportedMachineSpec(
            "The local backend requires HostImage, PackageSpec, or a prepared/pinned OCI image, "
            f"not {type(spec.source).__name__}"
        )
    if spec.storage_mb is not None or spec.gpus:
        raise UnsupportedMachineSpec("The local backend does not provide disk quotas or GPU allocations")
    if spec.memory_mb is not None and spec.memory_mb <= 0:
        raise ValueError("Local memory_mb must be positive")
    if isinstance(spec.source, RegistryImage) and not is_digest_pinned_image(spec.source.reference):
        raise UnsupportedMachineSpec("Local registry images must be pinned by sha256 digest")
    return spec.source


def check_local_environment(
    requirements: TaskRequirements, spec: MachineSpec, *, cgroup_parent: Path | None = None
) -> None:
    """Check local execution support without acquiring a bubblewrap sandbox."""
    source = local_source(spec)
    origin = source.source if isinstance(source, PreparedImage) else source
    reference = origin.reference if isinstance(origin, RegistryImage) else None
    capabilities = LOCAL_CAPABILITIES
    if cgroup_parent is not None:
        check_cgroup_parent(cgroup_parent)
        capabilities = capabilities | {MachineCapability.RETAINED_COMMAND_CANCELLATION, MachineCapability.PROCESS_SCOPE}
    check_execution_requirements(
        requirements,
        spec,
        capabilities=capabilities,
        image_reference=reference,
    )
