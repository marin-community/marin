# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic Connect boundary over controller-installed resource bindings."""

from typing import cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf import any_pb2
from google.protobuf.message import Message

from iris.rpc import resource_pb2
from iris.rpc.resource_registry import ResourceBinding, ResourceRegistry, ResourceVerb

PROTOCOL_VERSION = "resource.v1"


class ResourceServiceImpl:
    """Validate generic envelopes and dispatch without naming resource nouns."""

    def __init__(self, registry: ResourceRegistry) -> None:
        self._registry = registry

    def get_resource(
        self,
        request: resource_pb2.GetResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.GetResourceResponse:
        _require_ref(request.ref)
        binding = self._registry.require(request.ref.type, ResourceVerb.GET)
        try:
            return cast(resource_pb2.GetResourceResponse, binding.handler(request, context))
        except ValueError as error:
            raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error

    def update_resource(
        self,
        request: resource_pb2.UpdateResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.Operation:
        if not request.mutation.request_id.strip():
            raise ConnectError(Code.INVALID_ARGUMENT, "mutation request_id is required")
        _require_ref(request.ref)
        binding = self._registry.require(request.ref.type, ResourceVerb.UPDATE)
        payload = _unpack(binding, request.update)
        try:
            return cast(resource_pb2.Operation, binding.handler(request, payload, context))
        except ValueError as error:
            raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error

    def get_service_info(
        self,
        _request: resource_pb2.GetServiceInfoRequest,
        _context: RequestContext,
    ) -> resource_pb2.GetServiceInfoResponse:
        return resource_pb2.GetServiceInfoResponse(
            protocol_version=PROTOCOL_VERSION,
            resources=self._registry.capabilities,
        )


def _require_ref(ref: resource_pb2.ResourceRef) -> None:
    if not ref.authority_cluster_id or not ref.type or not ref.id:
        raise ConnectError(Code.INVALID_ARGUMENT, "resource authority, type, and id are required")


def _unpack(binding: ResourceBinding, packed: any_pb2.Any) -> Message:
    payload_type = binding.payload_type
    if payload_type is None:
        raise RuntimeError("update binding has no payload type")
    if not packed.Is(payload_type.DESCRIPTOR):
        raise ConnectError(Code.INVALID_ARGUMENT, f"resource update does not accept {packed.type_url!r}")
    payload = payload_type()
    if not packed.Unpack(payload):
        raise ConnectError(Code.INVALID_ARGUMENT, f"could not decode {packed.type_url!r}")
    return payload
