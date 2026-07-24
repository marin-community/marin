# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EndpointService: the leased service-discovery registry.

Registration grants a lease, returned as ``lease_duration``; re-registering with
the same ``endpoint_id`` renews it, and an unrenewed endpoint expires (hidden
from reads, swept by the pruner) independent of its task row. The legacy
``ControllerService`` endpoint RPCs forward here in-process. ``/system/``
endpoints are served from an in-memory map and never expire.
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

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

# Default lease granted when the client does not request one, and the ceiling a
# requested lease is clamped to. Endpoint registrations renew on a cadence
# (register-or-renew), so the lease governs liveness: a crashed or unrenewed
# endpoint expires within one lease. Renewal runs at 1/3 of the granted lease, so
# 10m yields a ~3.3m cadence and a ~10m worst-case expiry for a registrant that
# stops renewing.
ENDPOINT_LEASE = Duration.from_minutes(10)
# Floor on a granted lease: bounds how often a client may force the controller to
# re-register by capping the renewal rate a short requested lease can ask for.
MIN_ENDPOINT_LEASE = Duration.from_minutes(3)


def proxy_name_to_endpoint_names(proxy_name: str) -> tuple[str, str]:
    """Decode a proxy ``.``-encoded name into endpoint-name lookup candidates.

    Proxy URLs and subdomains encode ``/`` as ``.`` (``user.jobX.dash`` ->
    ``/user/jobX/dash``). Endpoint names start with ``/``, so the
    slash-prefixed form is tried first; the bare form covers endpoints
    registered without a leading slash.
    """
    slashed = proxy_name.replace(".", "/")
    return f"/{slashed}", slashed


def parse_proxy_timeout(metadata: dict[str, str]) -> float | None:
    """Per-endpoint proxy timeout (seconds) from endpoint metadata, or None.

    None when the key is absent or the value is not a positive number: a malformed
    override falls back to the proxy default rather than breaking resolution.
    """
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


class EndpointServiceImpl:
    """Leased service-discovery registry over the shared endpoints projection."""

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
        """Register a never-expiring ``/system/`` endpoint (e.g. the log server)."""
        with self._proxy_lock:
            if self._system_endpoints.get(name) == address:
                return
            self._system_endpoints[name] = address
        self._publish_proxy_delta(
            upserts=(self._system_proxy_mapping(name, address),),
            deletes=(),
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
        mappings = tuple(self._task_proxy_mapping(row) for row in task_endpoints)
        mappings += tuple(self._system_proxy_mapping(name, address) for name, address in system_endpoints)
        return ProxyRegistrySnapshot(generation=generation, endpoints=mappings)

    def _endpoint_mutated(self, mutation: EndpointDelta | EndpointReset) -> None:
        if isinstance(mutation, EndpointReset):
            self._publish_proxy_reset()
            return
        self._publish_proxy_delta(
            upserts=tuple(self._task_proxy_mapping(row) for row in mutation.upserts),
            deletes=mutation.deletes,
        )

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
    def _task_proxy_mapping(row: EndpointRow) -> ProxyEndpointMapping:
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
    def _system_proxy_mapping(name: str, address: str) -> ProxyEndpointMapping:
        return ProxyEndpointMapping(
            endpoint_id=f"system:{name}",
            name=name,
            address=address,
            link_access=False,
            peer_id=None,
            task_id=None,
            timeout_seconds=None,
            lease_deadline_epoch_ms=None,
        )

    def _granted_lease(self, request: controller_pb2.Controller.RegisterEndpointRequest) -> Duration:
        """Lease to grant: the client's request clamped to ``[MIN_ENDPOINT_LEASE, self._lease]``.

        Unset selects the default (``self._lease``), so old clients that never
        set the field keep the long lease. A renewing client requests a short
        one and gets it, down to the floor.
        """
        if not request.HasField("lease_duration"):
            return self._lease
        requested = duration_from_proto(request.lease_duration)
        if requested < MIN_ENDPOINT_LEASE:
            return MIN_ENDPOINT_LEASE
        if requested > self._lease:
            return self._lease
        return requested

    # --- RPC surface ---------------------------------------------------------

    def register_endpoint(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.RegisterEndpointResponse:
        """Register or renew a service endpoint, returning the granted lease.

        Re-registering with the same ``endpoint_id`` renews the lease.
        Registration is refused if the task is already terminal (see
        :meth:`EndpointsProjection.add`); once registered, the endpoint is served
        to lookup/list until its lease lapses.
        """
        endpoint_id = request.endpoint_id or str(uuid.uuid4())

        task_id = JobName.from_wire(request.task_id)
        task_id.require_task()

        granted = self._granted_lease(request)
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

        # Validation runs inside the writer transaction in
        # ``EndpointsProjection.add``: NOT_FOUND if the task row is missing,
        # FAILED_PRECONDITION if the task is terminal.
        with self._db.transaction() as cur:
            outcome = cur.caches[EndpointsProjection].add(cur, endpoint)
        if outcome is AddEndpointOutcome.NOT_FOUND:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")
        if outcome is AddEndpointOutcome.TERMINAL:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Task {request.task_id} is already terminal; endpoint not registered",
            )

        return controller_pb2.Controller.RegisterEndpointResponse(
            endpoint_id=endpoint_id,
            lease_duration=duration_to_proto(granted),
        )

    def unregister_endpoint(
        self,
        request: controller_pb2.Controller.UnregisterEndpointRequest,
        ctx: Any,
    ) -> job_pb2.Empty:
        """Unregister a service endpoint. Idempotent."""
        with self._db.transaction() as cur:
            cur.caches[EndpointsProjection].remove(cur, request.endpoint_id)
        return job_pb2.Empty()

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListEndpointsResponse:
        """List endpoints by name prefix (or exact name when ``request.exact`` is set).

        ``request.task_ids``, if set, ANDs with the name match. Expired leases
        are excluded; ``/system/`` names resolve from the in-memory map.
        """
        prefix = request.prefix
        if prefix.startswith("/system/"):
            return self._list_system_endpoints(prefix, exact=request.exact)

        endpoints = self._db.caches[EndpointsProjection].query(
            EndpointQuery(
                exact_name=prefix if request.exact else None,
                name_prefix=None if request.exact else prefix,
                task_ids=tuple(JobName.from_wire(t) for t in request.task_ids),
            ),
        )
        return controller_pb2.Controller.ListEndpointsResponse(
            endpoints=[
                controller_pb2.Controller.Endpoint(
                    endpoint_id=e.endpoint_id,
                    name=e.name,
                    address=e.address,
                    task_id=e.task_id.to_wire(),
                    metadata=e.metadata,
                    access=e.access,
                    peer_id=e.peer_id or "",
                )
                for e in endpoints
            ]
        )

    # --- Internal helpers ----------------------------------------------------

    def resolve_endpoint(self, name: str) -> str | None:
        """Resolve an endpoint name to its address, or None.

        Task endpoints (live leases) take priority over ``/system/`` endpoints.
        """
        row = self._db.caches[EndpointsProjection].resolve(name)
        if row is not None:
            return row.address
        with self._proxy_lock:
            return self._system_endpoints.get(name)

    def resolve_task_endpoint(self, name: str) -> EndpointRow | None:
        """Resolve a task-registered endpoint row by wire name, or None.

        Used for owner authorization on token minting; ``/system/`` endpoints
        (no owning task) are intentionally not returned. Accepts either the
        ``/``-prefixed name or the bare form.
        """
        for candidate in proxy_name_to_endpoint_names(name):
            row = self._db.caches[EndpointsProjection].resolve(candidate)
            if row is not None:
                return row
        return None

    def _list_system_endpoints(self, prefix: str, *, exact: bool) -> controller_pb2.Controller.ListEndpointsResponse:
        """Resolve system endpoints from the in-memory map."""
        results: list[controller_pb2.Controller.Endpoint] = []
        with self._proxy_lock:
            system_endpoints = tuple(self._system_endpoints.items())
        for name, address in system_endpoints:
            matches = name == prefix if exact else name.startswith(prefix)
            if matches:
                results.append(controller_pb2.Controller.Endpoint(endpoint_id=name, name=name, address=address))
        return controller_pb2.Controller.ListEndpointsResponse(endpoints=results)
