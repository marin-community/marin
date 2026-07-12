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
from dataclasses import dataclass
from typing import Any

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.timing import Duration, Timestamp

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.endpoints import (
    AddEndpointOutcome,
    EndpointQuery,
    EndpointRow,
    EndpointsProjection,
)
from iris.cluster.types import EndpointAccess, JobName
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_from_proto, duration_to_proto

logger = logging.getLogger(__name__)

# Lease granted when the client does not request one. Long so an old client that
# registers once and never renews keeps its endpoint served; a renewing client
# requests a much shorter lease and gets it (see MIN_ENDPOINT_LEASE). This only
# bounds the dead-but-non-terminal backstop case — a task's endpoints are still
# reclaimed promptly when it goes terminal — so it is sized generously (5 days).
ENDPOINT_LEASE = Duration.from_hours(120)
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


@dataclass(frozen=True, slots=True)
class ResolvedEndpoint:
    """A proxy request's resolved target: canonical endpoint name, address, and access mode."""

    name: str
    address: str
    # A Controller.EndpointAccess value.
    access: int
    # Set when the endpoint is mirrored from a federated child: the /proxy route
    # forwards to this peer's controller instead of dialing ``address``. None = local.
    peer_id: str | None = None
    # The task that owns the endpoint; used to authorize a federation-peer's proxy
    # against the owning job's federation handle. None for ``/system/`` endpoints.
    task_id: JobName | None = None


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

    def register_system_endpoint(self, name: str, address: str) -> None:
        """Register a never-expiring ``/system/`` endpoint (e.g. the log server)."""
        self._system_endpoints[name] = address

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

        Re-registering with the same ``endpoint_id`` renews the lease. The
        endpoint is bound to ``request.task_id`` so retry cleanup removes
        endpoints from superseded attempts. It is visible to lookup/list only
        while that task is non-terminal and the lease is unexpired.
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
        # FAILED_PRECONDITION if the task is terminal or the attempt is stale.
        with self._db.transaction() as cur:
            outcome = cur.caches[EndpointsProjection].add(cur, endpoint, expected_attempt_id=request.attempt_id)
        if outcome is AddEndpointOutcome.NOT_FOUND:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")
        if outcome is AddEndpointOutcome.STALE_ATTEMPT:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Stale attempt for task {request.task_id} (attempt {request.attempt_id})",
            )
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

    def resolve_proxy_target(self, encoded_name: str) -> ResolvedEndpoint | None:
        """Resolve a proxy request's ``encoded_name`` to its target, or None.

        A remote (federated) row carries ``peer_id`` so the proxy forwards to that
        peer. When one name resolves to disagreeing targets — a local and a remote
        row, or remote rows from different peers — this fails closed (returns None),
        since a mistaken pick would forward a request to the wrong place or apply the
        wrong access mode. ``/system/`` endpoints always resolve as ``PRIVATE``.
        """
        for name in proxy_name_to_endpoint_names(encoded_name):
            rows = self._db.caches[EndpointsProjection].resolve_all(name)
            if rows:
                if len({row.peer_id for row in rows}) > 1:
                    logger.warning("Ambiguous endpoint %r resolves to multiple peers; refusing to proxy", name)
                    return None
                row = rows[0]
                return ResolvedEndpoint(
                    name=row.name, address=row.address, access=row.access, peer_id=row.peer_id, task_id=row.task_id
                )
            address = self._system_endpoints.get(name)
            if address is not None:
                return ResolvedEndpoint(name=name, address=address, access=EndpointAccess.ENDPOINT_ACCESS_PRIVATE)
        return None

    def _list_system_endpoints(self, prefix: str, *, exact: bool) -> controller_pb2.Controller.ListEndpointsResponse:
        """Resolve system endpoints from the in-memory map."""
        results: list[controller_pb2.Controller.Endpoint] = []
        for name, address in self._system_endpoints.items():
            matches = name == prefix if exact else name.startswith(prefix)
            if matches:
                results.append(controller_pb2.Controller.Endpoint(endpoint_id=name, name=name, address=address))
        return controller_pb2.Controller.ListEndpointsResponse(endpoints=results)
