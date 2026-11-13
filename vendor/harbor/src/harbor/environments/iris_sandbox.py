# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute a prepared task image as a single Iris SANDBOX pod."""

from typing import Literal

from iris.rpc.proto_display import priority_band_value
from shellbox.backends.iris.machine import KubernetesIrisMachineFactory

from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.prepared import PreparedEnvironment
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import ResourceMode


class IrisSandboxEnvironment(PreparedEnvironment):
    """One task per pod, with files retained for verification after command deadlines."""

    def __init__(
        self,
        *args,
        cluster: str,
        pool: str | None = None,
        priority: Literal["production", "batch"] = "batch",
        ttl: float = 3600,
        **kwargs,
    ):
        if kwargs.get("model_endpoint") or kwargs.get("candidate_model_endpoint"):
            raise ValueError("Iris sandbox does not support process-session model relays")
        super().__init__(*args, **kwargs)
        self.factory = KubernetesIrisMachineFactory(
            cluster=cluster,
            pool=pool,
            job_ttl=int(ttl),
            priority_band=priority_band_value(priority),
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.IRIS_SANDBOX

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # Enable storage only after a SANDBOX-profile overflow qualifies its runtime.
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_task_resources(self) -> None:
        KubernetesIrisMachineFactory.check_task(self._prepared_task.task)
        if ResourceMode.IGNORE in (self._cpu_resource_mode, self._memory_resource_mode):
            raise ValueError("Iris sandbox requires the declared CPU and memory allocations")
        if self.task_env_config.storage_mb is not None:
            raise ValueError("Iris SANDBOX storage enforcement is unqualified; declared storage is unsupported")
        if self.task_env_config.gpus or self.task_env_config.gpu_types or self.task_env_config.tpu is not None:
            raise ValueError("Iris sandbox supports only CPU tasks")
