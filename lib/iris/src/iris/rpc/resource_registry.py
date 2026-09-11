# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable bindings for generic resource verbs."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Generic, TypeVar, cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.any_pb2 import Any as AnyMessage
from google.protobuf.message import Message

from iris.rpc import resource_pb2


class ResourceVerb(StrEnum):
    GET = "get"
    LIST = "list"
    BATCH_GET = "batch-get"


Request = TypeVar("Request", bound=Message)
Response = TypeVar("Response", bound=Message)
Handler = Callable[[Message, RequestContext], Message]


@dataclass(frozen=True, slots=True)
class ResourceCodec(Generic[Request, Response]):
    verb: ResourceVerb
    request_type: type[Request]
    response_type: type[Response]
    wire_response_type: type[Message]
    encoder: Callable[[Request, Response], Message]

    def decode(self, payload: AnyMessage) -> Request:
        request = self.request_type()
        if not payload.Unpack(request):
            raise ConnectError(Code.INVALID_ARGUMENT, f"expected {_type_url(self.request_type)}")
        return request

    def encode(self, request: Request, response: Message) -> Message:
        if not isinstance(response, self.response_type):
            raise TypeError(
                f"resource handler returned {type(response).__name__}; expected {self.response_type.__name__}"
            )
        encoded = self.encoder(request, cast(Response, response))
        if not isinstance(encoded, self.wire_response_type):
            raise TypeError(
                f"resource codec returned {type(encoded).__name__}; expected {self.wire_response_type.__name__}"
            )
        return encoded


@dataclass(frozen=True, slots=True)
class ResourceBinding:
    codec: ResourceCodec
    handler: Handler
    dashboard_readable: bool

    def invoke(self, payload: AnyMessage, context: RequestContext) -> Message:
        request = self.codec.decode(payload)
        response = self.handler(request, context)
        return self.codec.encode(request, response)


def get_codec(
    request_type: type[Request],
    response_type: type[Response],
) -> ResourceCodec[Request, Response]:
    def encode(_request: Request, response: Response) -> Message:
        return resource_pb2.GetResponse(resource=_resource(response))

    return ResourceCodec(ResourceVerb.GET, request_type, response_type, resource_pb2.GetResponse, encode)


def list_codec(
    request_type: type[Request],
    response_type: type[Response],
    resources: Callable[[Response], Iterable[Message]],
    page: Callable[[Response], resource_pb2.PageInfo] | None = None,
    metadata: Callable[[Response], Message] | None = None,
) -> ResourceCodec[Request, Response]:
    def encode(_request: Request, response: Response) -> Message:
        bodies = tuple(resources(response))
        result = resource_pb2.ListResponse(
            resources=[_resource(body) for body in bodies],
            page=page(response) if page is not None else resource_pb2.PageInfo(total_count=len(bodies)),
        )
        if metadata is not None:
            result.metadata.CopyFrom(_pack(metadata(response)))
        return result

    return ResourceCodec(ResourceVerb.LIST, request_type, response_type, resource_pb2.ListResponse, encode)


def batch_get_codec(
    request_type: type[Request],
    response_type: type[Response],
    resources: Callable[[Request, Response], Iterable[Message]],
) -> ResourceCodec[Request, Response]:
    def encode(request: Request, response: Response) -> Message:
        return resource_pb2.BatchGetResponse(resources=[_resource(body) for body in resources(request, response)])

    return ResourceCodec(
        ResourceVerb.BATCH_GET,
        request_type,
        response_type,
        resource_pb2.BatchGetResponse,
        encode,
    )


class ResourceRegistryBuilder:
    """Build the resource catalog at controller composition time."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, ResourceVerb], ResourceBinding] = {}

    def get(
        self,
        path: str,
        handler: Callable[[Request, RequestContext], Response],
        *,
        request_type: type[Request],
        response_type: type[Response],
        dashboard_readable: bool,
    ) -> None:
        self._bind(
            path,
            get_codec(request_type, response_type),
            handler,
            dashboard_readable=dashboard_readable,
        )

    def list(
        self,
        path: str,
        handler: Callable[[Request, RequestContext], Response],
        *,
        request_type: type[Request],
        response_type: type[Response],
        resources: Callable[[Response], Iterable[Message]],
        page: Callable[[Response], resource_pb2.PageInfo] | None = None,
        metadata: Callable[[Response], Message] | None = None,
        dashboard_readable: bool,
    ) -> None:
        self._bind(
            path,
            list_codec(request_type, response_type, resources, page, metadata),
            handler,
            dashboard_readable=dashboard_readable,
        )

    def batch_get(
        self,
        path: str,
        handler: Callable[[Request, RequestContext], Response],
        *,
        request_type: type[Request],
        response_type: type[Response],
        resources: Callable[[Request, Response], Iterable[Message]],
        dashboard_readable: bool,
    ) -> None:
        self._bind(
            path,
            batch_get_codec(request_type, response_type, resources),
            handler,
            dashboard_readable=dashboard_readable,
        )

    def _bind(
        self,
        path: str,
        codec: ResourceCodec[Request, Response],
        handler: Callable[[Request, RequestContext], Response],
        *,
        dashboard_readable: bool,
    ) -> None:
        resource_type, verb = _parse_path(path)
        if codec.verb is not verb:
            raise ValueError(f"resource binding {path} requires a {verb.value} codec")
        key = (resource_type, verb)
        if key in self._bindings:
            raise ValueError(f"duplicate resource binding: {path}")
        self._bindings[key] = ResourceBinding(
            codec=codec,
            handler=cast(Handler, handler),
            dashboard_readable=dashboard_readable,
        )

    def freeze(self) -> "ResourceRegistry":
        return ResourceRegistry(self._bindings)


class ResourceRegistry:
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


def _resource(body: Message) -> resource_pb2.Resource:
    return resource_pb2.Resource(body=_pack(body))


def _pack(message: Message) -> AnyMessage:
    packed = AnyMessage()
    packed.Pack(message)
    return packed
