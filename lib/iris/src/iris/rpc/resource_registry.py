# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Noun-agnostic binding registry for the generic ResourceService verbs."""

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, TypeVar, cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.message import Message

from iris.rpc import resource_pb2


class ResourceVerb(StrEnum):
    """Operations understood by the generic resource transport."""

    CREATE = "create"
    GET = "get"
    BATCH_GET = "batch-get"
    LIST = "list"
    UPDATE = "update"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class ResourceWireContract:
    """Wire types and features advertised for one installed endpoint."""

    views: tuple[int, ...] = ()
    body_type_urls: tuple[str, ...] = ()
    accepted_type_urls: tuple[str, ...] = ()
    features: tuple[str, ...] = ()


_RequestT = TypeVar("_RequestT", bound=Message, contravariant=True)
_ResponseT = TypeVar("_ResponseT", bound=Message, covariant=True)


class ResourceEndpoint(Protocol[_RequestT, _ResponseT]):
    """One concrete noun/verb endpoint installed by controller composition."""

    contract: ResourceWireContract

    def __call__(self, request: _RequestT, context: RequestContext) -> _ResponseT: ...


@dataclass(frozen=True, slots=True)
class ResourceEndpointBinding:
    """One endpoint callable and its advertised wire contract."""

    endpoint: Callable[[Message, RequestContext], Message]
    contract: ResourceWireContract = ResourceWireContract()

    def require_view(self, resource_type: str, view: int) -> None:
        if self.contract.views and view not in self.contract.views:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"resource type {resource_type!r} does not support view {view}",
            )

    def require_type_url(self, resource_type: str, type_url: str) -> None:
        if type_url not in self.contract.accepted_type_urls:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"resource type {resource_type!r} does not accept payload {type_url!r}",
            )


@dataclass(frozen=True, slots=True)
class _BackendBinding:
    backend_id: str
    resource_type: str
    verb: ResourceVerb
    features: tuple[str, ...]


class ResourceRouteRegistryBuilder:
    """Composition-time builder for the immutable endpoint registry."""

    def __init__(self) -> None:
        self._routes: dict[tuple[str, ResourceVerb], ResourceEndpointBinding] = {}
        self._types: set[str] = set()
        self._backend_routes: list[_BackendBinding] = []

    def bind(
        self,
        path: str,
        endpoint: ResourceEndpoint[_RequestT, _ResponseT],
    ) -> None:
        """Bind `/noun/verb` to one controller endpoint operation."""
        resource_type, verb = _parse_path(path)
        key = (resource_type, verb)
        if key in self._routes:
            raise ValueError(f"duplicate resource binding: {path}")
        handler = cast(Callable[[Message, RequestContext], Message], endpoint)
        self._routes[key] = ResourceEndpointBinding(handler, endpoint.contract)
        self._types.add(resource_type)

    def register_backend(
        self,
        backend_id: str,
        resource_type: str,
        verb: ResourceVerb,
        *,
        features: tuple[str, ...] = (),
    ) -> None:
        """Advertise one backend contribution to an installed endpoint."""
        if not backend_id.strip():
            raise ValueError("backend_id must be non-empty")
        if resource_type not in self._types:
            raise ValueError(f"backend binding refers to unknown resource type: {resource_type}")
        self._backend_routes.append(_BackendBinding(backend_id, resource_type, verb, features))

    def freeze(self) -> "ResourceRouteRegistry":
        """Return an immutable snapshot for one ResourceService instance."""
        return ResourceRouteRegistry(self._routes, tuple(self._backend_routes))


class ResourceRouteRegistry:
    """Immutable lookup table used by ResourceService at runtime."""

    def __init__(
        self,
        routes: Mapping[tuple[str, ResourceVerb], ResourceEndpointBinding],
        backend_routes: tuple[_BackendBinding, ...] = (),
    ) -> None:
        self._routes = MappingProxyType(dict(routes))
        self._types = frozenset(resource_type for resource_type, _ in routes)
        self._backend_routes = backend_routes

    def require(self, resource_type: str, verb: ResourceVerb) -> ResourceEndpointBinding:
        """Return a bound endpoint or a stable Connect error."""
        route = self._routes.get((resource_type, verb))
        if route is not None:
            return route
        if resource_type not in self._types:
            raise ConnectError(Code.NOT_FOUND, f"unknown resource type: {resource_type!r}")
        raise ConnectError(
            Code.UNIMPLEMENTED,
            f"resource type {resource_type!r} does not support {verb.value}",
        )

    @property
    def capabilities(self) -> tuple[resource_pb2.ResourceCapability, ...]:
        """Aggregate installed endpoints into the public service catalog."""
        by_type: dict[str, list[tuple[ResourceVerb, ResourceEndpointBinding]]] = defaultdict(list)
        for (resource_type, verb), route in self._routes.items():
            by_type[resource_type].append((verb, route))

        capabilities: list[resource_pb2.ResourceCapability] = []
        for resource_type in sorted(by_type):
            routes = sorted(by_type[resource_type], key=lambda item: item[0].value)
            capability = resource_pb2.ResourceCapability(
                type=resource_type,
                verbs=[verb.value.replace("-", "_") for verb, _ in routes],
                views=sorted({view for _, route in routes for view in route.contract.views}),
                body_type_urls=sorted({url for _, route in routes for url in route.contract.body_type_urls}),
                features=sorted({feature for _, route in routes for feature in route.contract.features}),
            )
            for verb, route in routes:
                if verb is ResourceVerb.CREATE:
                    capability.create_type_urls.extend(route.contract.accepted_type_urls)
                elif verb is ResourceVerb.LIST:
                    capability.query_type_urls.extend(route.contract.accepted_type_urls)
                elif verb is ResourceVerb.UPDATE:
                    capability.update_type_urls.extend(route.contract.accepted_type_urls)
            capabilities.append(capability)
        return tuple(capabilities)

    @property
    def backend_capabilities(self) -> tuple[resource_pb2.BackendResourceCapability, ...]:
        """Aggregate backend contributions without changing public dispatch."""
        grouped: dict[tuple[str, str], list[_BackendBinding]] = defaultdict(list)
        for route in self._backend_routes:
            grouped[(route.backend_id, route.resource_type)].append(route)
        return tuple(
            resource_pb2.BackendResourceCapability(
                backend_id=backend_id,
                type=resource_type,
                verbs=sorted({route.verb.value.replace("-", "_") for route in routes}),
                features=sorted({feature for route in routes for feature in route.features}),
            )
            for (backend_id, resource_type), routes in sorted(grouped.items())
        )


def _parse_path(path: str) -> tuple[str, ResourceVerb]:
    parts = path.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("resource binding path must be /noun/verb")
    noun, verb_name = parts
    try:
        verb = ResourceVerb(verb_name)
    except ValueError as error:
        raise ValueError(f"unknown resource verb: {verb_name!r}") from error
    return f"iris/{noun}", verb
