# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute prepared Harbor tasks in persistent private bubblewrap sandboxes."""

from dataclasses import replace
from pathlib import Path

from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.prepared import PreparedEnvironment
from harbor.environments.prepared_compose import validate_prepared_task
from harbor.environments.resource_policies import validate_resource_capabilities
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode
from shellbox.machine import MachineSpec


def _validate_local_resources(
    task: EnvironmentConfig,
    *,
    cpu_enforcement_policy: ResourceMode,
    memory_enforcement_policy: ResourceMode,
) -> None:
    validate_resource_capabilities(
        environment_label="local",
        resource_capabilities=EnvironmentResourceCapabilities(),
        cpu_enforcement_policy=cpu_enforcement_policy,
        memory_enforcement_policy=memory_enforcement_policy,
    )
    if task.storage_mb is not None or task.gpus or task.gpu_types or task.tpu is not None:
        raise ValueError("Local prepared sandboxes do not enforce storage, GPU, or TPU allocations")
    for name, value, mode in (
        ("CPU", task.cpus, cpu_enforcement_policy),
        ("memory", task.memory_mb, memory_enforcement_policy),
    ):
        if value is not None and mode is not ResourceMode.IGNORE:
            raise ValueError(f"Local {name} declarations require explicit enforcement_policy=ignore")


def validate_local_task(
    task_env_config: EnvironmentConfig,
    environment_dir: Path,
    *,
    cpu_enforcement_policy: ResourceMode,
    memory_enforcement_policy: ResourceMode,
) -> None:
    """Check prepared task admission and explicit local resource consent without allocation."""
    validate_prepared_task(task_env_config, environment_dir)
    _validate_local_resources(
        task_env_config,
        cpu_enforcement_policy=cpu_enforcement_policy,
        memory_enforcement_policy=memory_enforcement_policy,
    )


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
        ttl: float = 3600,
        **kwargs,
    ):
        # The local backend requires Linux modules; offline task validation is
        # also used by hosts that run their tasks on Iris or Daytona.
        from shellbox.backends.local.machine import LocalMachineFactory

        super().__init__(*args, **kwargs)
        self.factory = LocalMachineFactory(
            bwrap=Path(bwrap) if bwrap is not None else None,
            umoci=Path(umoci),
            image_cache=Path(image_cache) if image_cache is not None else None,
            authfile=Path(registry_authfile) if registry_authfile is not None else None,
            policy=Path(signature_policy) if signature_policy is not None else None,
            ttl=ttl,
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.LOCAL

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

    def _validate_definition(self) -> None:
        super()._validate_definition()
        _validate_local_resources(
            self.task_env_config,
            cpu_enforcement_policy=self._cpu_resource_mode,
            memory_enforcement_policy=self._memory_resource_mode,
        )

    def _resource_spec(self, spec: MachineSpec) -> MachineSpec:
        return replace(spec, cpus=None, memory_mb=None)

    def _command_user(self, user: str | int | None) -> str:
        selected = self._resolve_user(user)
        if selected is None or str(selected) in {"root", "0", "0:0"}:
            return "0"
        raise ValueError("Local prepared sandboxes support only root commands")
