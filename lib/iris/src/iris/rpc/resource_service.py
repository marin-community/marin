# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect boundary for controller-registered resource operations."""

from typing import TypeVar

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.any_pb2 import Any as AnyMessage
from google.protobuf.message import Message

from iris.rpc import resource_pb2
from iris.rpc.resource_registry import ResourceRegistry, ResourceVerb

WireResponse = TypeVar("WireResponse", bound=Message)


class ResourceServiceImpl:
    """Dispatch resource verbs through an immutable registry."""

    def __init__(self, registry: ResourceRegistry) -> None:
        self._registry = registry

    def get(self, request: resource_pb2.ResourceRequest, context: RequestContext) -> resource_pb2.GetResponse:
        return self._invoke(
            request.resource_type,
            ResourceVerb.GET,
            request.input,
            context,
            resource_pb2.GetResponse,
        )

    def list(self, request: resource_pb2.ResourceRequest, context: RequestContext) -> resource_pb2.ListResponse:
        return self._invoke(
            request.resource_type,
            ResourceVerb.LIST,
            request.input,
            context,
            resource_pb2.ListResponse,
        )

    def batch_get(
        self,
        request: resource_pb2.ResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.BatchGetResponse:
        return self._invoke(
            request.resource_type,
            ResourceVerb.BATCH_GET,
            request.input,
            context,
            resource_pb2.BatchGetResponse,
        )

    def _invoke(
        self,
        resource_type: str,
        verb: ResourceVerb,
        payload: AnyMessage,
        context: RequestContext,
        response_type: type[WireResponse],
    ) -> WireResponse:
        if not resource_type:
            raise ConnectError(Code.INVALID_ARGUMENT, "resource_type is required")
        response = self._registry.require(resource_type, verb).invoke(payload, context)
        if not isinstance(response, response_type):
            raise TypeError(f"resource codec returned {type(response).__name__}; expected {response_type.__name__}")
        return response
