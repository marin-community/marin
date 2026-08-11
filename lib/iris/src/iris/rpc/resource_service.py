# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic Connect RPC boundary for registered Iris resources."""

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.rpc import resource_pb2
from iris.rpc.resource_registry import ResourceCatalog

PROTOCOL_VERSION = "resource.v1"


class ResourceServiceImpl:
    """Dispatch ResourceService verbs without importing concrete resource types."""

    def __init__(self, catalog: ResourceCatalog, *, controller_generation: str = "") -> None:
        self._catalog = catalog
        self._controller_generation = controller_generation

    @property
    def catalog(self) -> ResourceCatalog:
        return self._catalog

    def create_resource(
        self,
        request: resource_pb2.CreateResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.Operation:
        registration = self._catalog.get(request.type)
        _require_mutation(request.mutation)
        registration.require_create_body(request.body.type_url)
        return self._catalog.require(request.type, "create")(request, context)

    def get_resource(
        self,
        request: resource_pb2.GetResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.GetResourceResponse:
        _require_ref(request.ref)
        registration = self._catalog.get(request.ref.type)
        registration.require_view(request.view)
        return self._catalog.require(request.ref.type, "get")(request, context)

    def batch_get_resources(
        self,
        request: resource_pb2.BatchGetResourcesRequest,
        context: RequestContext,
    ) -> resource_pb2.BatchGetResourcesResponse:
        registration = self._catalog.get(request.type)
        registration.require_view(request.view)
        for ref in request.refs:
            _require_ref(ref)
            if ref.type != request.type:
                raise ConnectError(Code.INVALID_ARGUMENT, f"batch ref type {ref.type!r} does not match {request.type!r}")
        return self._catalog.require(request.type, "batch_get")(request, context)

    def list_resources(
        self,
        request: resource_pb2.ListResourcesRequest,
        context: RequestContext,
    ) -> resource_pb2.ListResourcesResponse:
        registration = self._catalog.get(request.type)
        registration.require_view(request.view)
        registration.require_query(request.query.type_url)
        return self._catalog.require(request.type, "list")(request, context)

    def update_resource(
        self,
        request: resource_pb2.UpdateResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.Operation:
        _require_mutation(request.mutation)
        _require_ref(request.ref)
        registration = self._catalog.get(request.ref.type)
        registration.require_update(request.update.patch.type_url)
        return self._catalog.require(request.ref.type, "update")(request, context)

    def delete_resource(
        self,
        request: resource_pb2.DeleteResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.Operation:
        _require_mutation(request.mutation)
        _require_ref(request.ref)
        return self._catalog.require(request.ref.type, "delete")(request, context)

    def get_service_info(
        self,
        _request: resource_pb2.GetServiceInfoRequest,
        _context: RequestContext,
    ) -> resource_pb2.GetServiceInfoResponse:
        return resource_pb2.GetServiceInfoResponse(
            protocol_version=PROTOCOL_VERSION,
            controller_generation=self._controller_generation,
            resources=self._catalog.capabilities,
            backend_resources=self._catalog.backend_capabilities,
        )


def _require_ref(ref: resource_pb2.ResourceRef) -> None:
    if not ref.authority_cluster_id or not ref.type or not ref.id:
        raise ConnectError(Code.INVALID_ARGUMENT, "resource authority, type, and id are required")


def _require_mutation(mutation: resource_pb2.MutationMetadata) -> None:
    if not mutation.request_id.strip():
        raise ConnectError(Code.INVALID_ARGUMENT, "mutation request_id is required")
