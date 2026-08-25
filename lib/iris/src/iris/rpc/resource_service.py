# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect boundary for controller-registered resource operations."""

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.any_pb2 import Any as AnyMessage

from iris.rpc import resource_pb2
from iris.rpc.resource_registry import ResourceRegistry, ResourceVerb


class ResourceServiceImpl:
    """Dispatch resource verbs through an immutable registry."""

    def __init__(self, registry: ResourceRegistry) -> None:
        self._registry = registry

    def get(self, request: resource_pb2.ResourceRequest, context: RequestContext) -> resource_pb2.ResourceResponse:
        return resource_pb2.ResourceResponse(
            output=self._invoke(request.resource_type, ResourceVerb.GET, request.input, context)
        )

    def list(self, request: resource_pb2.ResourceRequest, context: RequestContext) -> resource_pb2.ResourceResponse:
        return resource_pb2.ResourceResponse(
            output=self._invoke(request.resource_type, ResourceVerb.LIST, request.input, context)
        )

    def batch_get(
        self,
        request: resource_pb2.ResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.ResourceResponse:
        return resource_pb2.ResourceResponse(
            output=self._invoke(request.resource_type, ResourceVerb.BATCH_GET, request.input, context)
        )

    def _invoke(
        self,
        resource_type: str,
        verb: ResourceVerb,
        payload: AnyMessage,
        context: RequestContext,
    ) -> AnyMessage:
        if not resource_type:
            raise ConnectError(Code.INVALID_ARGUMENT, "resource_type is required")
        return self._registry.require(resource_type, verb).invoke(payload, context)
