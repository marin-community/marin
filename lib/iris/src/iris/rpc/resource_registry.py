# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable bindings for generic resource verbs."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TypeVar, cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.any_pb2 import Any as AnyMessage
from google.protobuf.message import Message


class ResourceVerb(StrEnum):
    GET = "get"
    LIST = "list"
    BATCH_GET = "batch-get"


Request = TypeVar("Request", bound=Message)
Response = TypeVar("Response", bound=Message)
Handler = Callable[[Message, RequestContext], Message]


@dataclass(frozen=True, slots=True)
class ResourceBinding:
    request_type: type[Message]
    response_type: type[Message]
    handler: Handler

    def invoke(self, payload: AnyMessage, context: RequestContext) -> AnyMessage:
        request = self.request_type()
        if not payload.Unpack(request):
            raise ConnectError(Code.INVALID_ARGUMENT, f"expected {_type_url(self.request_type)}")
        response = self.handler(request, context)
        if not isinstance(response, self.response_type):
            raise TypeError(
                f"resource handler returned {type(response).__name__}; expected {self.response_type.__name__}"
            )
        packed = AnyMessage()
        packed.Pack(response)
        return packed


class ResourceRegistryBuilder:
    """Build the resource catalog at controller composition time."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, ResourceVerb], ResourceBinding] = {}

    def bind(
        self,
        path: str,
        request_type: type[Request],
        response_type: type[Response],
        handler: Callable[[Request, RequestContext], Response],
    ) -> None:
        resource_type, verb = _parse_path(path)
        key = (resource_type, verb)
        if key in self._bindings:
            raise ValueError(f"duplicate resource binding: {path}")
        self._bindings[key] = ResourceBinding(
            request_type=request_type,
            response_type=response_type,
            handler=cast(Handler, handler),
        )

    def freeze(self) -> "ResourceRegistry":
        return ResourceRegistry(self._bindings)


class ResourceRegistry:
    """Frozen resource catalog used by the RPC dispatcher."""

    def __init__(self, bindings: Mapping[tuple[str, ResourceVerb], ResourceBinding]) -> None:
        self._bindings = MappingProxyType(dict(bindings))
        self._resource_types = frozenset(resource_type for resource_type, _ in bindings)

    def require(self, resource_type: str, verb: ResourceVerb) -> ResourceBinding:
        binding = self._bindings.get((resource_type, verb))
        if binding is not None:
            return binding
        if resource_type not in self._resource_types:
            raise ConnectError(Code.NOT_FOUND, f"unknown resource type: {resource_type!r}")
        raise ConnectError(Code.UNIMPLEMENTED, f"resource type {resource_type!r} does not support {verb.value}")


def _parse_path(path: str) -> tuple[str, ResourceVerb]:
    parts = path.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("resource binding path must be /noun/verb")
    resource_type, verb_name = parts
    try:
        verb = ResourceVerb(verb_name)
    except ValueError as error:
        raise ValueError(f"unknown resource verb: {verb_name!r}") from error
    return resource_type, verb


def _type_url(message_type: type[Message]) -> str:
    return f"type.googleapis.com/{message_type.DESCRIPTOR.full_name}"
