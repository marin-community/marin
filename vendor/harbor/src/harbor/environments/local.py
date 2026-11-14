# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute prepared Harbor tasks in persistent private bubblewrap sandboxes."""

import asyncio
from pathlib import Path
from typing import cast

from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.prepared import PreparedEnvironment
from harbor.environments.resource_policies import validate_local_resources
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode
from shellbox.backends.local.disk import LocalDiskPool, verified_local_disk_pool
from shellbox.backends.local.machine import LocalMachine, LocalMachineFactory

DEFAULT_LOCAL_TTL = 3600


def local_machine_factory(
    *,
    bwrap: str | None = None,
    umoci: str = "umoci",
    image_cache: str | None = None,
    registry_authfile: str | None = None,
    signature_policy: str | None = None,
    cgroup_parent: str | None = None,
    ttl: float = DEFAULT_LOCAL_TTL,
    disk_pool: LocalDiskPool | None = None,
) -> LocalMachineFactory:
    """Normalize LOCAL options for native and TaskSpec workloads."""
    return LocalMachineFactory(
        bwrap=Path(bwrap) if bwrap is not None else None,
        umoci=Path(umoci),
        image_cache=Path(image_cache) if image_cache is not None else None,
        authfile=Path(registry_authfile) if registry_authfile is not None else None,
        policy=Path(signature_policy) if signature_policy is not None else None,
        cgroup_parent=Path(cgroup_parent) if cgroup_parent is not None else None,
        ttl=ttl,
        disk_pool=disk_pool,
    )


async def local_disk_pool(
    task: EnvironmentConfig,
    *,
    disk_root: Path | None,
    cpu_enforcement_policy: ResourceMode,
    memory_enforcement_policy: ResourceMode,
) -> LocalDiskPool | None:
    """Qualify storage and validate LOCAL resource policy before allocation."""
    pool = await verified_local_disk_pool(disk_root) if disk_root is not None else None
    validate_local_resources(
        task,
        cpu_enforcement_policy=cpu_enforcement_policy,
        memory_enforcement_policy=memory_enforcement_policy,
        disk_pool=pool,
    )
    return pool


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
        ttl: float = DEFAULT_LOCAL_TTL,
        **kwargs,
    ):
        self.cgroup_parent = Path(cgroup_parent) if cgroup_parent is not None else None
        self.disk_root = Path(disk_root) if disk_root is not None else None
        self.disk_pool: LocalDiskPool | None = None
        super().__init__(*args, **kwargs)
        self.factory = local_machine_factory(
            bwrap=bwrap,
            umoci=umoci,
            image_cache=image_cache,
            registry_authfile=registry_authfile,
            signature_policy=signature_policy,
            cgroup_parent=cgroup_parent,
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
        task = self.task_env_config
        if self.disk_root is not None and self.disk_pool is None:
            # Storage admission completes asynchronously before any acquisition.
            task = task.model_copy(update={"storage_mb": None})
        validate_local_resources(
            task,
            cpu_enforcement_policy=self._cpu_resource_mode,
            memory_enforcement_policy=self._memory_resource_mode,
            disk_pool=self.disk_pool,
        )

    async def start(self, force_build: bool, *, service_tasks: asyncio.TaskGroup) -> None:
        self.disk_pool = await local_disk_pool(
            self.task_env_config,
            disk_root=self.disk_root,
            cpu_enforcement_policy=self._cpu_resource_mode,
            memory_enforcement_policy=self._memory_resource_mode,
        )
        cast(LocalMachineFactory, self.factory).disk_pool = self.disk_pool
        await super().start(force_build, service_tasks=service_tasks)

    async def quiesce_agent_commands(self) -> None:
        if self.machine is not None:
            await cast(LocalMachine, self.machine).quiesce_commands()
