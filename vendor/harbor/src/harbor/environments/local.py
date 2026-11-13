# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute prepared Harbor tasks in persistent private bubblewrap sandboxes."""

from pathlib import Path
from typing import cast

from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.prepared import PreparedEnvironment
from harbor.environments.resource_policies import validate_local_resources
from harbor.models.environment_type import EnvironmentType
from shellbox.backends.local.disk import LocalDiskPool
from shellbox.backends.local.machine import LocalMachine, LocalMachineFactory


class LocalEnvironment(PreparedEnvironment):
    """Use separate root filesystems and a private shared loopback namespace."""

    def __init__(
        self,
        *args,
        bwrap: str | None = None,
        umoci: str = "umoci",
        image_cache: str | None = None,
        registry_authfile: str | None = None,
        signature_policy: str | None = None,
        cgroup_parent: str | None = None,
        disk_root: str | None = None,
        ttl: float = 3600,
        **kwargs,
    ):
        self.cgroup_parent = Path(cgroup_parent) if cgroup_parent is not None else None
        self.disk_pool = LocalDiskPool(Path(disk_root)) if disk_root is not None else None
        super().__init__(*args, **kwargs)
        self.factory = LocalMachineFactory(
            bwrap=Path(bwrap) if bwrap is not None else None,
            umoci=Path(umoci),
            image_cache=Path(image_cache) if image_cache is not None else None,
            authfile=Path(registry_authfile) if registry_authfile is not None else None,
            policy=Path(signature_policy) if signature_policy is not None else None,
            cgroup_parent=self.cgroup_parent,
            ttl=ttl,
            disk_pool=self.disk_pool,
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.LOCAL

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_task_resources(self) -> None:
        validate_local_resources(
            self.task_env_config,
            cpu_enforcement_policy=self._cpu_resource_mode,
            memory_enforcement_policy=self._memory_resource_mode,
            disk_pool=self.disk_pool,
        )

    async def quiesce_agent_commands(self) -> None:
        if self.machine is not None:
            await cast(LocalMachine, self.machine).quiesce_commands()
