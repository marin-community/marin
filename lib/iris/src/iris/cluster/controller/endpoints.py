# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint registry state and endpoint request operations."""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any
from urllib.parse import urlsplit

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.timing import Duration, Timestamp

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.endpoints import (
    AddEndpointOutcome,
    EndpointDelta,
    EndpointQuery,
    EndpointReset,
    EndpointRow,
    EndpointsProjection,
)
from iris.cluster.types import PROXY_TIMEOUT_METADATA_KEY, EndpointAccess, JobName
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_from_proto, duration_to_proto

logger = logging.getLogger(__name__)

ENDPOINT_LEASE = Duration.from_minutes(10)
MIN_ENDPOINT_LEASE = Duration.from_minutes(3)
SYSTEM_PROXY_ENDPOINT_PREFIX = "system:"


def proxy_name_to_endpoint_names(proxy_name: str) -> tuple[str, str]:
    """Decode a proxy ``.``-encoded name into endpoint-name lookup candidates."""
    slashed = proxy_name.replace(".", "/")
    return f"/{slashed}", slashed


def parse_proxy_timeout(metadata: dict[str, str]) -> float | None:
    """Return the per-endpoint proxy timeout in seconds, when valid."""
    raw = metadata.get(PROXY_TIMEOUT_METADATA_KEY)
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        seconds = 0.0
    if seconds <= 0:
        logger.warning("Ignoring invalid %s=%r on endpoint metadata", PROXY_TIMEOUT_METADATA_KEY, raw)
        return None
    return seconds


@dataclass(frozen=True, slots=True)
class ProxyEndpointMapping:
    """Endpoint fields required by the native proxy data plane."""

    endpoint_id: str
    name: str
    address: str
    link_access: bool
    peer_id: str | None
    task_id: str | None
    timeout_seconds: float | None
    lease_deadline_epoch_ms: int | None


@dataclass(frozen=True, slots=True)
class ProxyMappingDelta:
    """One atomic registry transition between adjacent generations."""

    base_generation: int
    next_generation: int
    upserts: tuple[ProxyEndpointMapping, ...]
    deletes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProxyRegistryReset:
    """Notification that consumers must install a complete current snapshot."""


@dataclass(frozen=True, slots=True)
class ProxyRegistrySnapshot:
    """Complete native-proxy bootstrap or recovery state."""

    generation: int
    endpoints: tuple[ProxyEndpointMapping, ...]


def _proxyable_address(address: str) -> bool:
    candidate = address if "://" in address else f"http://{address}"
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class EndpointRegistry:
    """Leased endpoint state shared by request handling and the native proxy."""

    def __init__(self, db: ControllerDB, system_endpoints: dict[str, str] | None = None) -> None:
        self._db = db
        self._system_endpoints = system_endpoints or {}
        self._proxy_lock = RLock()
        self._proxy_generation = 0
        self._proxy_listeners: list[Callable[[ProxyMappingDelta | ProxyRegistryReset], None]] = []
        db.caches[EndpointsProjection].subscribe(self._endpoint_mutated)

    def register_system_endpoint(self, name: str, address: str) -> None:
        """Register a never-expiring ``/system/`` endpoint."""
        with self._proxy_lock:
            if self._system_endpoints.get(name) == address:
                return
            self._system_endpoints[name] = address
        mapping = self._system_proxy_mapping(name, address)
        self._publish_proxy_delta(
            upserts=(mapping,) if mapping is not None else (),
            deletes=() if mapping is not None else (f"{SYSTEM_PROXY_ENDPOINT_PREFIX}{name}",),
        )

    def subscribe_proxy_updates(self, listener: Callable[[ProxyMappingDelta | ProxyRegistryReset], None]) -> None:
        """Subscribe to committed endpoint-registry transitions."""
        with self._proxy_lock:
            self._proxy_listeners.append(listener)

    def proxy_registry_snapshot(self) -> ProxyRegistrySnapshot:
        """Return a complete generation-stamped native proxy registry."""
        with self._proxy_lock:
            generation = self._proxy_generation
            system_endpoints = tuple(self._system_endpoints.items())
            task_endpoints = tuple(self._db.caches[EndpointsProjection].all())
        mappings = tuple(mapping for row in task_endpoints if (mapping := self._task_proxy_mapping(row)) is not None)
        mappings += tuple(
            mapping
            for name, address in system_endpoints
            if (mapping := self._system_proxy_mapping(name, address)) is not None
        )
        return ProxyRegistrySnapshot(generation=generation, endpoints=mappings)

    def resolve_endpoint(self, name: str) -> str | None:
        """Resolve an endpoint name to its address, or return None."""
        row = self._db.caches[EndpointsProjection].resolve(name)
        if row is not None:
            return row.address
        with self._proxy_lock:
            return self._system_endpoints.get(name)

    def resolve_task_endpoint(self, name: str) -> EndpointRow | None:
        """Resolve a task endpoint row by slash or proxy-encoded name."""
        for candidate in proxy_name_to_endpoint_names(name):
            row = self._db.caches[EndpointsProjection].resolve(candidate)
            if row is not None:
                return row
        return None

    def system_endpoints(self) -> tuple[tuple[str, str], ...]:
        """Return a stable snapshot of registered system endpoints."""
        with self._proxy_lock:
            return tuple(self._system_endpoints.items())

    def _endpoint_mutated(self, mutation: EndpointDelta | EndpointReset) -> None:
        if isinstance(mutation, EndpointReset):
            self._publish_proxy_reset()
            return
        upserts: list[ProxyEndpointMapping] = []
        deletes = list(mutation.deletes)
        for row in mutation.upserts:
            mapping = self._task_proxy_mapping(row)
            if mapping is None:
                deletes.append(row.endpoint_id)
            else:
                upserts.append(mapping)
        self._publish_proxy_delta(upserts=tuple(upserts), deletes=tuple(deletes))

    def _publish_proxy_delta(
        self,
        *,
        upserts: tuple[ProxyEndpointMapping, ...],
        deletes: tuple[str, ...],
    ) -> None:
        with self._proxy_lock:
            base_generation = self._proxy_generation
            next_generation = base_generation + 1
            self._proxy_generation = next_generation
            listeners = tuple(self._proxy_listeners)
        delta = ProxyMappingDelta(
            base_generation=base_generation,
            next_generation=next_generation,
            upserts=upserts,
            deletes=deletes,
        )
        for listener in listeners:
            listener(delta)

    def _publish_proxy_reset(self) -> None:
        with self._proxy_lock:
            self._proxy_generation += 1
            listeners = tuple(self._proxy_listeners)
        for listener in listeners:
            listener(ProxyRegistryReset())

    @staticmethod
    def _task_proxy_mapping(row: EndpointRow) -> ProxyEndpointMapping | None:
        if not _proxyable_address(row.address):
            return None
        return ProxyEndpointMapping(
            endpoint_id=row.endpoint_id,
            name=row.name,
            address=row.address,
            link_access=row.access == EndpointAccess.ENDPOINT_ACCESS_LINK,
            peer_id=row.peer_id,
            task_id=row.task_id.to_wire(),
            timeout_seconds=parse_proxy_timeout(row.metadata),
            lease_deadline_epoch_ms=row.lease_deadline.epoch_ms() if row.lease_deadline is not None else None,
        )

    @staticmethod
    def _system_proxy_mapping(name: str, address: str) -> ProxyEndpointMapping | None:
        if not _proxyable_address(address):
            return None
        return ProxyEndpointMapping(
            endpoint_id=f"{SYSTEM_PROXY_ENDPOINT_PREFIX}{name}",
            name=name,
            address=address,
            link_access=False,
            peer_id=None,
            task_id=None,
            timeout_seconds=None,
            lease_deadline_epoch_ms=None,
        )


@dataclass(frozen=True, slots=True)
class EndpointDependencies:
    db: ControllerDB
    registry: EndpointRegistry
    lease: Duration


def register_endpoint(
    dependencies: EndpointDependencies,
    request: controller_pb2.Controller.RegisterEndpointRequest,
    context: Any,
) -> controller_pb2.Controller.RegisterEndpointResponse:
    """Register or renew a service endpoint and return its granted lease."""
    del context
    endpoint_id = request.endpoint_id or str(uuid.uuid4())
    task_id = JobName.from_wire(request.task_id)
    task_id.require_task()
    granted = _granted_lease(dependencies, request)
    endpoint = EndpointRow(
        endpoint_id=endpoint_id,
        name=request.name,
        address=request.address,
        task_id=task_id,
        metadata=dict(request.metadata),
        registered_at=Timestamp.now(),
        lease_deadline=Timestamp.now().add(granted),
        access=request.access,
    )

    with dependencies.db.transaction() as cur:
        outcome = cur.caches[EndpointsProjection].add(cur, endpoint, attempt_id=request.attempt_id)
    if outcome is AddEndpointOutcome.NOT_FOUND:
        raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")
    if outcome is AddEndpointOutcome.TERMINAL:
        raise ConnectError(
            Code.FAILED_PRECONDITION,
            f"Task {request.task_id} is already terminal; endpoint not registered",
        )
    if outcome is AddEndpointOutcome.STALE_ATTEMPT:
        raise ConnectError(
            Code.FAILED_PRECONDITION,
            f"Task {request.task_id} attempt {request.attempt_id} is no longer current",
        )

    return controller_pb2.Controller.RegisterEndpointResponse(
        endpoint_id=endpoint_id,
        lease_duration=duration_to_proto(granted),
    )


def unregister_endpoint(
    dependencies: EndpointDependencies,
    request: controller_pb2.Controller.UnregisterEndpointRequest,
    context: Any,
) -> job_pb2.Empty:
    """Unregister a service endpoint idempotently."""
    del context
    with dependencies.db.transaction() as cur:
        cur.caches[EndpointsProjection].remove(cur, request.endpoint_id)
    return job_pb2.Empty()


def list_endpoints(
    dependencies: EndpointDependencies,
    request: controller_pb2.Controller.ListEndpointsRequest,
    context: Any,
) -> controller_pb2.Controller.ListEndpointsResponse:
    """List live task or system endpoints matching the request."""
    del context
    prefix = request.prefix
    if prefix.startswith("/system/"):
        return _list_system_endpoints(dependencies.registry, prefix, exact=request.exact)

    rows = dependencies.db.caches[EndpointsProjection].query(
        EndpointQuery(
            exact_name=prefix if request.exact else None,
            name_prefix=None if request.exact else prefix,
            task_ids=tuple(JobName.from_wire(task_id) for task_id in request.task_ids),
        ),
    )
    return controller_pb2.Controller.ListEndpointsResponse(
        endpoints=[
            controller_pb2.Controller.Endpoint(
                endpoint_id=row.endpoint_id,
                name=row.name,
                address=row.address,
                task_id=row.task_id.to_wire(),
                metadata=row.metadata,
                access=row.access,
                peer_id=row.peer_id or "",
            )
            for row in rows
        ]
    )


def _granted_lease(
    dependencies: EndpointDependencies,
    request: controller_pb2.Controller.RegisterEndpointRequest,
) -> Duration:
    if not request.HasField("lease_duration"):
        return dependencies.lease
    requested = duration_from_proto(request.lease_duration)
    if requested < MIN_ENDPOINT_LEASE:
        return MIN_ENDPOINT_LEASE
    if requested > dependencies.lease:
        return dependencies.lease
    return requested


def _list_system_endpoints(
    registry: EndpointRegistry,
    prefix: str,
    *,
    exact: bool,
) -> controller_pb2.Controller.ListEndpointsResponse:
    results = [
        controller_pb2.Controller.Endpoint(endpoint_id=name, name=name, address=address)
        for name, address in registry.system_endpoints()
        if (name == prefix if exact else name.startswith(prefix))
    ]
    return controller_pb2.Controller.ListEndpointsResponse(endpoints=results)
