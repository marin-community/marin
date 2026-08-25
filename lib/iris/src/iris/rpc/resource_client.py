# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client for controller-registered resource operations."""

import builtins
from collections.abc import Callable
from typing import TypeVar

from google.protobuf.any_pb2 import Any as AnyMessage
from google.protobuf.message import Message

from iris.rpc import resource_pb2
from iris.rpc.resource_connect import ResourceServiceClientSync

ResourceBody = TypeVar("ResourceBody", bound=Message)
WireResponse = TypeVar("WireResponse", bound=Message)


class ResourceClient:
    """Invoke resource operations using their registered protobuf types."""

    def __init__(self, stub: ResourceServiceClientSync) -> None:
        self._stub = stub

    def get(
        self,
        resource_type: str,
        request: Message,
        body_type: type[ResourceBody],
        *,
        timeout_ms: int | None = None,
    ) -> ResourceBody:
        response = self._invoke(self._stub.get, resource_type, request, timeout_ms)
        return _unpack(response.resource, body_type)

    def list(
        self,
        resource_type: str,
        request: Message,
        body_type: type[ResourceBody],
        *,
        timeout_ms: int | None = None,
    ) -> tuple[list[ResourceBody], resource_pb2.PageInfo]:
        response = self._invoke(self._stub.list, resource_type, request, timeout_ms)
        return [_unpack(resource, body_type) for resource in response.resources], response.page

    def batch_get(
        self,
        resource_type: str,
        request: Message,
        body_type: type[ResourceBody],
        *,
        timeout_ms: int | None = None,
    ) -> builtins.list[ResourceBody]:
        response = self._invoke(self._stub.batch_get, resource_type, request, timeout_ms)
        return [_unpack(resource, body_type) for resource in response.resources]

    def _invoke(
        self,
        rpc: Callable[..., WireResponse],
        resource_type: str,
        request: Message,
        timeout_ms: int | None,
    ) -> WireResponse:
        return rpc(
            resource_pb2.ResourceRequest(resource_type=resource_type, input=_pack(request)),
            timeout_ms=timeout_ms,
        )

    def close(self) -> None:
        self._stub.close()


def _pack(message: Message) -> AnyMessage:
    packed = AnyMessage()
    packed.Pack(message)
    return packed


def _unpack(resource: resource_pb2.Resource, body_type: type[ResourceBody]) -> ResourceBody:
    body = body_type()
    if not resource.body.Unpack(body):
        raise TypeError(f"resource body is not type.googleapis.com/{body_type.DESCRIPTOR.full_name}")
    return body
