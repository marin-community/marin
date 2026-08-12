# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic Connect RPC boundary for registered Iris resources."""

from collections.abc import Callable
from typing import cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.rpc import resource_pb2
from iris.rpc.resource_registry import ResourceRouteRegistry, ResourceVerb

PROTOCOL_VERSION = "resource.v1"


class ResourceServiceImpl:
    """Dispatch ResourceService verbs without importing concrete resource types."""

    def __init__(
        self,
        registry: ResourceRouteRegistry,
        *,
        controller_generation: str = "",
    ) -> None:
        self._registry = registry
        self._controller_generation = controller_generation

    def create_resource(
        self,
        request: resource_pb2.CreateResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.Operation:
        _require_mutation(request.mutation)
        route = self._registry.require(request.type, ResourceVerb.CREATE)
        route.require_type_url(request.type, request.body.type_url)
        handler = cast(
            Callable[[resource_pb2.CreateResourceRequest, RequestContext], resource_pb2.Operation],
            route.endpoint,
        )
        return handler(request, context)

    def get_resource(
        self,
        request: resource_pb2.GetResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.GetResourceResponse:
        _require_ref(request.ref)
        route = self._registry.require(request.ref.type, ResourceVerb.GET)
        route.require_view(request.ref.type, request.view)
        handler = cast(
            Callable[[resource_pb2.GetResourceRequest, RequestContext], resource_pb2.GetResourceResponse],
            route.endpoint,
        )
        return handler(request, context)

    def batch_get_resources(
        self,
        request: resource_pb2.BatchGetResourcesRequest,
        context: RequestContext,
    ) -> resource_pb2.BatchGetResourcesResponse:
        route = self._registry.require(request.type, ResourceVerb.BATCH_GET)
        route.require_view(request.type, request.view)
        for ref in request.refs:
            _require_ref(ref)
            if ref.type != request.type:
                raise ConnectError(Code.INVALID_ARGUMENT, f"batch ref type {ref.type!r} does not match {request.type!r}")
        handler = cast(
            Callable[[resource_pb2.BatchGetResourcesRequest, RequestContext], resource_pb2.BatchGetResourcesResponse],
            route.endpoint,
        )
        return handler(request, context)

    def list_resources(
        self,
        request: resource_pb2.ListResourcesRequest,
        context: RequestContext,
    ) -> resource_pb2.ListResourcesResponse:
        route = self._registry.require(request.type, ResourceVerb.LIST)
        route.require_view(request.type, request.view)
        route.require_type_url(request.type, request.query.type_url)
        handler = cast(
            Callable[[resource_pb2.ListResourcesRequest, RequestContext], resource_pb2.ListResourcesResponse],
            route.endpoint,
        )
        return handler(request, context)

    def update_resource(
        self,
        request: resource_pb2.UpdateResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.Operation:
        _require_mutation(request.mutation)
        _require_ref(request.ref)
        route = self._registry.require(request.ref.type, ResourceVerb.UPDATE)
        route.require_type_url(request.ref.type, request.update.type_url)
        handler = cast(
            Callable[[resource_pb2.UpdateResourceRequest, RequestContext], resource_pb2.Operation],
            route.endpoint,
        )
        return handler(request, context)

    def delete_resource(
        self,
        request: resource_pb2.DeleteResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.Operation:
        _require_mutation(request.mutation)
        _require_ref(request.ref)
        route = self._registry.require(request.ref.type, ResourceVerb.DELETE)
        handler = cast(
            Callable[[resource_pb2.DeleteResourceRequest, RequestContext], resource_pb2.Operation],
            route.endpoint,
        )
        return handler(request, context)

    def get_service_info(
        self,
        _request: resource_pb2.GetServiceInfoRequest,
        _context: RequestContext,
    ) -> resource_pb2.GetServiceInfoResponse:
        return resource_pb2.GetServiceInfoResponse(
            protocol_version=PROTOCOL_VERSION,
            controller_generation=self._controller_generation,
            resources=self._registry.capabilities,
            backend_resources=self._registry.backend_capabilities,
        )


def _require_ref(ref: resource_pb2.ResourceRef) -> None:
    if not ref.authority_cluster_id or not ref.type or not ref.id:
        raise ConnectError(Code.INVALID_ARGUMENT, "resource authority, type, and id are required")


def _require_mutation(mutation: resource_pb2.MutationMetadata) -> None:
    if not mutation.request_id.strip():
        raise ConnectError(Code.INVALID_ARGUMENT, "mutation request_id is required")
