# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client for controller-registered resource operations."""

from collections.abc import Callable
from typing import TypeVar

from google.protobuf.any_pb2 import Any as AnyMessage
from google.protobuf.message import Message

from iris.rpc import resource_pb2
from iris.rpc.resource_connect import ResourceServiceClientSync

Response = TypeVar("Response", bound=Message)


class ResourceClient:
    """Invoke resource operations using their registered protobuf types."""

    def __init__(self, stub: ResourceServiceClientSync) -> None:
        self._stub = stub

    def get(
        self,
        resource_type: str,
        request: Message,
        response_type: type[Response],
        *,
        timeout_ms: int | None = None,
    ) -> Response:
        return self._invoke(self._stub.get, resource_type, request, response_type, timeout_ms)

    def list(
        self,
        resource_type: str,
        request: Message,
        response_type: type[Response],
        *,
        timeout_ms: int | None = None,
    ) -> Response:
        return self._invoke(self._stub.list, resource_type, request, response_type, timeout_ms)

    def batch_get(
        self,
        resource_type: str,
        request: Message,
        response_type: type[Response],
        *,
        timeout_ms: int | None = None,
    ) -> Response:
        return self._invoke(self._stub.batch_get, resource_type, request, response_type, timeout_ms)

    def _invoke(
        self,
        rpc: Callable[..., resource_pb2.ResourceResponse],
        resource_type: str,
        request: Message,
        response_type: type[Response],
        timeout_ms: int | None,
    ) -> Response:
        response = rpc(
            resource_pb2.ResourceRequest(resource_type=resource_type, input=_pack(request)),
            timeout_ms=timeout_ms,
        )
        return _unpack(response.output, response_type)

    def close(self) -> None:
        self._stub.close()


def _pack(message: Message) -> AnyMessage:
    packed = AnyMessage()
    packed.Pack(message)
    return packed


def _unpack(payload: AnyMessage, response_type: type[Response]) -> Response:
    response = response_type()
    if not payload.Unpack(response):
        raise TypeError(f"resource response is not type.googleapis.com/{response_type.DESCRIPTOR.full_name}")
    return response
