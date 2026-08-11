# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open registration catalog for the generic ResourceService endpoint."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.rpc import resource_pb2

GetResourceHandler = Callable[
    [resource_pb2.GetResourceRequest, RequestContext],
    resource_pb2.GetResourceResponse,
]
BatchGetResourcesHandler = Callable[
    [resource_pb2.BatchGetResourcesRequest, RequestContext],
    resource_pb2.BatchGetResourcesResponse,
]
ListResourcesHandler = Callable[
    [resource_pb2.ListResourcesRequest, RequestContext],
    resource_pb2.ListResourcesResponse,
]
CreateResourceHandler = Callable[
    [resource_pb2.CreateResourceRequest, RequestContext],
    resource_pb2.Operation,
]
UpdateResourceHandler = Callable[
    [resource_pb2.UpdateResourceRequest, RequestContext],
    resource_pb2.Operation,
]
DeleteResourceHandler = Callable[
    [resource_pb2.DeleteResourceRequest, RequestContext],
    resource_pb2.Operation,
]


@dataclass(frozen=True, slots=True)
class ResourceRegistration:
    """One resource type's wire codecs and controller-backed operations."""

    resource_type: str
    views: tuple[int, ...] = ()
    body_type_urls: tuple[str, ...] = ()
    query_type_urls: tuple[str, ...] = ()
    update_type_urls: tuple[str, ...] = ()
    create_type_urls: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    get: GetResourceHandler | None = None
    batch_get: BatchGetResourcesHandler | None = None
    list: ListResourcesHandler | None = None
    create: CreateResourceHandler | None = None
    update: UpdateResourceHandler | None = None
    delete: DeleteResourceHandler | None = None

    def __post_init__(self) -> None:
        if not self.resource_type.strip():
            raise ValueError("resource_type must be non-empty")
        if not self.resource_type.startswith("iris/"):
            raise ValueError("resource_type must use the iris/ namespace")

    def capability(self) -> resource_pb2.ResourceCapability:
        verbs: list[str] = []
        for name in ("get", "batch_get", "list", "create", "update", "delete"):
            if getattr(self, name) is not None:
                verbs.append(name)
        return resource_pb2.ResourceCapability(
            type=self.resource_type,
            verbs=verbs,
            views=self.views,
            body_type_urls=self.body_type_urls,
            query_type_urls=self.query_type_urls,
            update_type_urls=self.update_type_urls,
            features=self.features,
            create_type_urls=self.create_type_urls,
        )

    def require_view(self, view: int) -> None:
        if self.views and view not in self.views:
            raise ConnectError(
                Code.INVALID_ARGUMENT, f"resource type {self.resource_type!r} does not support view {view}"
            )

    def require_query(self, type_url: str) -> None:
        self._require_type_url(type_url, self.query_type_urls, "query")

    def require_create_body(self, type_url: str) -> None:
        self._require_type_url(type_url, self.create_type_urls, "create body")

    def require_update(self, type_url: str) -> None:
        if type_url:
            self._require_type_url(type_url, self.update_type_urls, "update body")

    def _require_type_url(self, type_url: str, allowed: tuple[str, ...], field: str) -> None:
        if type_url not in allowed:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"resource type {self.resource_type!r} does not accept {field} {type_url!r}",
            )


@dataclass(frozen=True, slots=True)
class BackendResourceRegistration:
    """Backend contribution to the endpoint's installed resource behavior."""

    backend_id: str
    resource_type: str
    verbs: tuple[str, ...]
    features: tuple[str, ...] = ()

    def capability(self) -> resource_pb2.BackendResourceCapability:
        return resource_pb2.BackendResourceCapability(
            backend_id=self.backend_id,
            type=self.resource_type,
            verbs=self.verbs,
            features=self.features,
        )


class ResourceCatalog:
    """Immutable lookup table used by RPC, federation, and compatibility adapters."""

    def __init__(
        self,
        registrations: Iterable[ResourceRegistration],
        backend_registrations: Iterable[BackendResourceRegistration] = (),
    ) -> None:
        by_type: dict[str, ResourceRegistration] = {}
        for registration in registrations:
            if registration.resource_type in by_type:
                raise ValueError(f"duplicate resource registration: {registration.resource_type}")
            by_type[registration.resource_type] = registration
        self._by_type = by_type
        self._backend_registrations = tuple(
            sorted(backend_registrations, key=lambda item: (item.backend_id, item.resource_type))
        )

    @property
    def capabilities(self) -> tuple[resource_pb2.ResourceCapability, ...]:
        return tuple(self._by_type[key].capability() for key in sorted(self._by_type))

    @property
    def backend_capabilities(self) -> tuple[resource_pb2.BackendResourceCapability, ...]:
        return tuple(registration.capability() for registration in self._backend_registrations)

    def get(self, resource_type: str) -> ResourceRegistration:
        registration = self._by_type.get(resource_type)
        if registration is None:
            raise ConnectError(Code.NOT_FOUND, f"unknown resource type: {resource_type!r}")
        return registration

    def require(self, resource_type: str, verb: str):
        registration = self.get(resource_type)
        handler = getattr(registration, verb)
        if handler is None:
            raise ConnectError(
                Code.UNIMPLEMENTED,
                f"resource type {resource_type!r} does not support {verb}",
            )
        return handler
