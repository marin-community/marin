# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed client for the generic resource transport."""

from typing import TypeVar

from google.protobuf.any_pb2 import Any as AnyMessage
from google.protobuf.message import Message

from iris.rpc import resource_pb2
from iris.rpc.resource_connect import ResourceServiceClientSync

Response = TypeVar("Response", bound=Message)


class ResourceClient:
    """Pack registered requests and validate their typed responses."""

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
        response = self._stub.get(
            resource_pb2.ResourceRequest(resource_type=resource_type, input=_pack(request)),
            timeout_ms=timeout_ms,
        )
        return _unpack(response.output, response_type)

    def list(
        self,
        resource_type: str,
        request: Message,
        response_type: type[Response],
        *,
        timeout_ms: int | None = None,
    ) -> Response:
        response = self._stub.list(
            resource_pb2.ResourceRequest(resource_type=resource_type, input=_pack(request)),
            timeout_ms=timeout_ms,
        )
        return _unpack(response.output, response_type)

    def batch_get(
        self,
        resource_type: str,
        request: Message,
        response_type: type[Response],
        *,
        timeout_ms: int | None = None,
    ) -> Response:
        response = self._stub.batch_get(
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
