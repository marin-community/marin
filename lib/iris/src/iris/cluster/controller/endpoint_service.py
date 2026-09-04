# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin protobuf adapter for endpoint request operations."""

from collections.abc import Callable

from connectrpc.request import RequestContext
from rigging.timing import Duration

from iris.cluster.controller import endpoints
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.endpoints import (
    ENDPOINT_LEASE,
    ProxyMappingDelta,
    ProxyRegistryReset,
    ProxyRegistrySnapshot,
)
from iris.cluster.controller.projections.endpoints import EndpointRow
from iris.rpc import controller_pb2, job_pb2


class EndpointServiceImpl:
    """EndpointService adapter and its shared endpoint registry."""

    def __init__(
        self,
        *,
        db: ControllerDB,
        system_endpoints: dict[str, str] | None = None,
        lease: Duration = ENDPOINT_LEASE,
    ) -> None:
        self._registry = endpoints.EndpointRegistry(db, system_endpoints)
        self._dependencies = endpoints.EndpointDependencies(db=db, registry=self._registry, lease=lease)

    def register_endpoint(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
        context: RequestContext,
    ) -> controller_pb2.Controller.RegisterEndpointResponse:
        return endpoints.register_endpoint(self._dependencies, request, context)

    def unregister_endpoint(
        self,
        request: controller_pb2.Controller.UnregisterEndpointRequest,
        context: RequestContext,
    ) -> job_pb2.Empty:
        return endpoints.unregister_endpoint(self._dependencies, request, context)

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest,
        context: RequestContext,
    ) -> controller_pb2.Controller.ListEndpointsResponse:
        return endpoints.list_endpoints(self._dependencies, request, context)

    def register_system_endpoint(self, name: str, address: str) -> None:
        self._registry.register_system_endpoint(name, address)

    def subscribe_proxy_updates(
        self,
        listener: Callable[[ProxyMappingDelta | ProxyRegistryReset], None],
    ) -> None:
        self._registry.subscribe_proxy_updates(listener)

    def proxy_registry_snapshot(self) -> ProxyRegistrySnapshot:
        return self._registry.proxy_registry_snapshot()

    def resolve_endpoint(self, name: str) -> str | None:
        return self._registry.resolve_endpoint(name)

    def resolve_task_endpoint(self, name: str) -> EndpointRow | None:
        return self._registry.resolve_task_endpoint(name)
