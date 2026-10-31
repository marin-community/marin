# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute prepared Harbor tasks in persistent private bubblewrap sandboxes."""

from dataclasses import replace
from pathlib import Path

from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.prepared import PreparedEnvironment
from harbor.environments.resource_policies import validate_local_resources
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import ResourceMode
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import MachineSpec


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
        ttl: float = 3600,
        **kwargs,
    ):
        self.cgroup_parent = Path(cgroup_parent) if cgroup_parent is not None else None
        super().__init__(*args, **kwargs)
        self.factory = LocalMachineFactory(
            bwrap=Path(bwrap) if bwrap is not None else None,
            umoci=Path(umoci),
            image_cache=Path(image_cache) if image_cache is not None else None,
            authfile=Path(registry_authfile) if registry_authfile is not None else None,
            policy=Path(signature_policy) if signature_policy is not None else None,
            cgroup_parent=self.cgroup_parent,
            ttl=ttl,
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.LOCAL

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_definition(self) -> None:
        super()._validate_definition()
        validate_local_resources(
            self.task_env_config,
            cpu_enforcement_policy=self._cpu_resource_mode,
            memory_enforcement_policy=self._memory_resource_mode,
            cgroup_parent=self.cgroup_parent,
        )

    def _resource_spec(self, spec: MachineSpec) -> MachineSpec:
        return replace(
            spec,
            cpus=self._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT),
            memory_mb=self._resource_limit_value("memory", auto_mode=ResourceMode.LIMIT),
        )

    def _command_user(self, user: str | int | None) -> str:
        selected = self._resolve_user(user)
        if selected is None or str(selected) in {"root", "0", "0:0"}:
            return "0"
        raise ValueError("Local prepared sandboxes support only root commands")
