# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect boundary for controller-registered resource operations."""

from typing import cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.any_pb2 import Any as AnyMessage
from google.protobuf.message import Message

from iris.rpc import resource_pb2
from iris.rpc.auth import authorize_resource_access
from iris.rpc.resource_registry import ResourceRegistry, ResourceVerb


class ResourceServiceImpl:
    """Dispatch resource verbs through an immutable registry."""

    def __init__(self, registry: ResourceRegistry) -> None:
        self._registry = registry

    def get(self, request: resource_pb2.ResourceRequest, context: RequestContext) -> resource_pb2.GetResponse:
        return cast(
            resource_pb2.GetResponse,
            self._invoke(request.resource_type, ResourceVerb.GET, request.input, context),
        )

    def list(self, request: resource_pb2.ResourceRequest, context: RequestContext) -> resource_pb2.ListResponse:
        return cast(
            resource_pb2.ListResponse,
            self._invoke(request.resource_type, ResourceVerb.LIST, request.input, context),
        )

    def batch_get(
        self,
        request: resource_pb2.ResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.BatchGetResponse:
        return cast(
            resource_pb2.BatchGetResponse,
            self._invoke(request.resource_type, ResourceVerb.BATCH_GET, request.input, context),
        )

    def _invoke(
        self,
        resource_type: str,
        verb: ResourceVerb,
        payload: AnyMessage,
        context: RequestContext,
    ) -> Message:
        if not resource_type:
            raise ConnectError(Code.INVALID_ARGUMENT, "resource_type is required")
        binding = self._registry.require(resource_type, verb)
        authorize_resource_access(binding.dashboard_readable, resource_type, verb)
        return binding.invoke(payload, context)
