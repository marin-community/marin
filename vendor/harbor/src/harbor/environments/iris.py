# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute prepared Harbor tasks in credential-free Iris sandboxes."""

from dataclasses import replace

from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.prepared import PreparedEnvironment
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import ResourceMode
from shellbox.backends.iris.machine import (
    DEFAULT_JOB_TTL,
    DEFAULT_SCHEDULING_TIMEOUT,
    IrisMachineFactory,
)
from shellbox.machine import MachineSpec


class IrisEnvironment(PreparedEnvironment):
    """Use gVisor and Iris's denied external networking for a prepared task."""

    def __init__(
        self,
        *args,
        cluster: str | None = None,
        controller_url: str | None = None,
        scheduling_timeout: int = DEFAULT_SCHEDULING_TIMEOUT,
        job_ttl: int = DEFAULT_JOB_TTL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if cluster is None and controller_url is None:
            self.factory = IrisMachineFactory.in_cluster(
                scheduling_timeout=scheduling_timeout, job_ttl=job_ttl
            )
        else:
            self.factory = IrisMachineFactory(
                cluster=cluster,
                controller_url=controller_url,
                scheduling_timeout=scheduling_timeout,
                job_ttl=job_ttl,
            )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.IRIS

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_request=True, memory_request=True)

    def _resource_spec(self, spec: MachineSpec) -> MachineSpec:
        return replace(
            spec,
            cpus=self._resource_request_value("cpu", auto_mode=ResourceMode.REQUEST),
            memory_mb=self._resource_request_value("memory", auto_mode=ResourceMode.REQUEST),
        )
