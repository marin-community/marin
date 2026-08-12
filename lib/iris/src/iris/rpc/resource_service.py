# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic Connect RPC boundary for registered Iris resources."""

from collections.abc import Callable
from typing import TypeVar, cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.message import Message

from iris.cluster.federation.protocol import PeerCallError
from iris.resources.errors import (
    ActionIdempotencyConflict,
    ActionPolicyRejected,
    AmbiguousResourceMigration,
    BackendIdentityUnknown,
    InvalidPageToken,
    InvalidResourceKey,
    InvalidResourceRequest,
    ResourceConflict,
    ResourceError,
    ResourceExhausted,
    ResourceNotFound,
    ResourcePermissionDenied,
    ResourcePreconditionFailed,
    ResourceReplaced,
    ResourceSourceUnavailable,
    UnsupportedResourceSchema,
    UnsupportedResourceVerb,
)
from iris.rpc import resource_pb2
from iris.rpc.federation_client import peer_connect_error
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
        return _invoke(handler, request, context)

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
        return _invoke(handler, request, context)

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
        return _invoke(handler, request, context)

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
        return _invoke(handler, request, context)

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
        return _invoke(handler, request, context)

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
        return _invoke(handler, request, context)

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


_RequestT = TypeVar("_RequestT", bound=Message)
_ResponseT = TypeVar("_ResponseT", bound=Message)


def _invoke(
    handler: Callable[[_RequestT, RequestContext], _ResponseT],
    request: _RequestT,
    context: RequestContext,
) -> _ResponseT:
    try:
        return handler(request, context)
    except ResourceError as error:
        raise _resource_connect_error(error) from error
    except PeerCallError as error:
        raise peer_connect_error(error) from error
    except ValueError as error:
        raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error


def _resource_connect_error(error: ResourceError) -> ConnectError:
    if isinstance(error, (InvalidResourceKey, InvalidResourceRequest, InvalidPageToken)):
        code = Code.INVALID_ARGUMENT
    elif isinstance(error, ResourceNotFound):
        code = Code.NOT_FOUND
    elif isinstance(error, ResourcePermissionDenied):
        code = Code.PERMISSION_DENIED
    elif isinstance(
        error,
        (
            ResourceReplaced,
            ResourcePreconditionFailed,
            ActionPolicyRejected,
            BackendIdentityUnknown,
            UnsupportedResourceSchema,
            AmbiguousResourceMigration,
        ),
    ):
        code = Code.FAILED_PRECONDITION
    elif isinstance(error, (ResourceConflict, ActionIdempotencyConflict)):
        code = Code.ALREADY_EXISTS
    elif isinstance(error, ResourceExhausted):
        code = Code.RESOURCE_EXHAUSTED
    elif isinstance(error, ResourceSourceUnavailable):
        code = Code.UNAVAILABLE
    elif isinstance(error, UnsupportedResourceVerb):
        code = Code.UNIMPLEMENTED
    else:
        code = Code.INTERNAL
    return ConnectError(code, str(error))
