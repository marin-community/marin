# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native leased endpoint registry and proxy projection."""

import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from urllib.parse import urlsplit

from rigging.timing import Duration, Timestamp

from iris.cluster.controller.persistence.database import ControllerDB
from iris.cluster.controller.persistence.projections.endpoints import (
    AddEndpointOutcome,
    EndpointDelta,
    EndpointQuery,
    EndpointReset,
    EndpointRow,
    EndpointsProjection,
)
from iris.cluster.types import PROXY_TIMEOUT_METADATA_KEY, EndpointAccess, JobName

logger = logging.getLogger(__name__)

ENDPOINT_LEASE = Duration.from_minutes(10)
MIN_ENDPOINT_LEASE = Duration.from_minutes(3)
SYSTEM_PROXY_ENDPOINT_PREFIX = "system:"


class EndpointTaskNotFound(ValueError):
    """The endpoint's Task does not exist."""


class EndpointTaskTerminal(ValueError):
    """The endpoint's Task is already terminal."""


class EndpointAttemptStale(ValueError):
    """The endpoint registration names a superseded Attempt."""


def proxy_name_to_endpoint_names(proxy_name: str) -> tuple[str, str]:
    """Decode a proxy ``.``-encoded name into endpoint-name candidates."""
    slashed = proxy_name.replace(".", "/")
    return f"/{slashed}", slashed


def parse_proxy_timeout(metadata: dict[str, str]) -> float | None:
    """Read a positive per-endpoint proxy timeout, if configured."""
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
    base_generation: int
    next_generation: int
    upserts: tuple[ProxyEndpointMapping, ...]
    deletes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProxyRegistryReset:
    """Notification that consumers must install a complete snapshot."""


@dataclass(frozen=True, slots=True)
class ProxyRegistrySnapshot:
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
    """Persistence-backed endpoint operations with an in-memory system registry."""

    def __init__(
        self,
        *,
        db: ControllerDB,
        system_endpoints: dict[str, str] | None = None,
        lease: Duration = ENDPOINT_LEASE,
    ) -> None:
        self._db = db
        self._system_endpoints: dict[str, str] = system_endpoints or {}
        self._lease = lease
        self._proxy_lock = RLock()
        self._proxy_generation = 0
        self._proxy_listeners: list[Callable[[ProxyMappingDelta | ProxyRegistryReset], None]] = []
        db.caches[EndpointsProjection].subscribe(self._endpoint_mutated)

    def register_system_endpoint(self, name: str, address: str) -> None:
        with self._proxy_lock:
            if self._system_endpoints.get(name) == address:
                return
            self._system_endpoints[name] = address
        mapping = self._system_proxy_mapping(name, address)
        self._publish_proxy_delta(
            upserts=(mapping,) if mapping is not None else (),
            deletes=() if mapping is not None else (f"{SYSTEM_PROXY_ENDPOINT_PREFIX}{name}",),
        )

    def system_endpoints(self) -> tuple[tuple[str, str], ...]:
        with self._proxy_lock:
            return tuple(sorted(self._system_endpoints.items()))

    def subscribe_proxy_updates(self, listener: Callable[[ProxyMappingDelta | ProxyRegistryReset], None]) -> None:
        with self._proxy_lock:
            self._proxy_listeners.append(listener)

    def proxy_registry_snapshot(self) -> ProxyRegistrySnapshot:
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

    def granted_lease(self, requested: Duration | None) -> Duration:
        if requested is None:
            return self._lease
        if requested < MIN_ENDPOINT_LEASE:
            return MIN_ENDPOINT_LEASE
        if requested > self._lease:
            return self._lease
        return requested

    def register_endpoint(
        self,
        *,
        endpoint_id: str | None,
        name: str,
        address: str,
        task_id: JobName,
        attempt_id: int,
        metadata: dict[str, str],
        access: int,
        requested_lease: Duration | None,
    ) -> tuple[str, Duration]:
        resolved_endpoint_id = endpoint_id or str(uuid.uuid4())
        task_id.require_task()
        granted = self.granted_lease(requested_lease)
        endpoint = EndpointRow(
            endpoint_id=resolved_endpoint_id,
            name=name,
            address=address,
            task_id=task_id,
            metadata=dict(metadata),
            registered_at=Timestamp.now(),
            lease_deadline=Timestamp.now().add(granted),
            access=access,
        )
        with self._db.transaction() as cur:
            outcome = cur.caches[EndpointsProjection].add(cur, endpoint, attempt_id=attempt_id)
        if outcome is AddEndpointOutcome.NOT_FOUND:
            raise EndpointTaskNotFound(task_id.to_wire())
        if outcome is AddEndpointOutcome.TERMINAL:
            raise EndpointTaskTerminal(task_id.to_wire())
        if outcome is AddEndpointOutcome.STALE_ATTEMPT:
            raise EndpointAttemptStale(f"{task_id.to_wire()}:{attempt_id}")
        return resolved_endpoint_id, granted

    def unregister_endpoint(self, endpoint_id: str) -> None:
        with self._db.transaction() as cur:
            cur.caches[EndpointsProjection].remove(cur, endpoint_id)

    def list_endpoint_rows(
        self,
        *,
        prefix: str,
        exact: bool,
        task_ids: Sequence[JobName],
    ) -> tuple[EndpointRow, ...]:
        return tuple(
            self._db.caches[EndpointsProjection].query(
                EndpointQuery(
                    exact_name=prefix if exact else None,
                    name_prefix=None if exact else prefix,
                    task_ids=tuple(task_ids),
                )
            )
        )

    def resolve_endpoint(self, name: str) -> str | None:
        row = self._db.caches[EndpointsProjection].resolve(name)
        if row is not None:
            return row.address
        with self._proxy_lock:
            return self._system_endpoints.get(name)

    def resolve_task_endpoint(self, name: str) -> EndpointRow | None:
        for candidate in proxy_name_to_endpoint_names(name):
            row = self._db.caches[EndpointsProjection].resolve(candidate)
            if row is not None:
                return row
        return None

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
        delta = ProxyMappingDelta(base_generation, next_generation, upserts, deletes)
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
