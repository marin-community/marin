# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from rigging.connect import capability_path, federated_capability_path

from iris.backends.protocol import BackendCapability, TaskBackend
from iris.cluster.config import BackendConfig
from iris.cluster.controller.auth import (
    ControllerAuth,
)
from iris.cluster.controller.persistence.database import ControllerDB
from iris.cluster.controller.persistence.projections.endpoints import (
    EndpointRow,
)
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.federation.manager import FederationManager
from iris.cluster.types import (
    WorkerId,
)
from iris.resources.log import LogReader


class ResourceRuntime(Protocol):
    """Controller capabilities needed by resource operations."""

    @property
    def backends(self) -> dict[str, TaskBackend]: ...

    @property
    def federation(self) -> FederationManager: ...

    @property
    def capabilities(self) -> frozenset[BackendCapability]: ...

    def wake(self) -> None: ...

    def liveness_for_worker(self, worker_id: WorkerId) -> WorkerLiveness: ...

    def backend_id_for_scale_group(self, scale_group: str) -> str: ...

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None: ...


class EndpointRegistry(Protocol):
    """Native endpoint registry capabilities consumed by resource operations."""

    def system_endpoints(self) -> tuple[tuple[str, str], ...]: ...

    def resolve_task_endpoint(self, name: str) -> EndpointRow | None: ...


@dataclass(frozen=True, slots=True)
class CapabilityUrlConfig:
    """Origins used to construct endpoint capability URLs."""

    cluster_name: str = ""
    local_origin: str = ""
    parent_origin: str = ""

    def build(self, name: str, token: str) -> str:
        if self.parent_origin and self.cluster_name:
            return f"{self.parent_origin.rstrip('/')}{federated_capability_path(self.cluster_name, name, token)}"
        if self.local_origin:
            return f"{self.local_origin.rstrip('/')}{capability_path(name, token)}"
        return ""


@dataclass(frozen=True, slots=True)
class ResourceDependencies:
    """Shared immutable dependencies for resource noun services."""

    cluster_id: str
    db: ControllerDB
    runtime: ResourceRuntime
    endpoint_registry: EndpointRegistry
    auth: ControllerAuth
    capability_url_config: CapabilityUrlConfig
    backends: Mapping[str, TaskBackend]
    backend_configs: Mapping[str, BackendConfig]
    log_reader: LogReader | None = None

    def __post_init__(self) -> None:
        if not self.cluster_id.strip():
            raise ValueError("cluster_id is required for resource identities")
        backends = dict(self.backends)
        backend_configs = dict(self.backend_configs)
        if backend_configs.keys() != backends.keys():
            raise ValueError("backend_configs keys must exactly match live backend keys")
        object.__setattr__(self, "backends", MappingProxyType(backends))
        object.__setattr__(self, "backend_configs", MappingProxyType(backend_configs))
