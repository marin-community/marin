# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EndpointService: the leased service-discovery registry.

Registration grants a time-bounded lease. The registrant renews it by
re-registering with the same ``endpoint_id`` before it elapses; an unrenewed
endpoint expires — hidden from reads immediately, swept from storage by the
pruner — independent of the owning task row's lifetime. ``RegisterEndpoint``
returns the granted ``lease_duration`` so the client paces its renewals off the
server's policy.

The legacy ``ControllerService.{RegisterEndpoint,UnregisterEndpoint,
ListEndpoints}`` RPCs forward into this backend in-process, so clients that call
the old surface keep working; clients that want to renew call ``EndpointService``
directly to learn their lease.

``/system/`` endpoints (e.g. the log server) are served from an in-memory map
rather than the leased table, so they never expire.
"""

import logging
import uuid
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
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_to_proto

logger = logging.getLogger(__name__)

# Endpoint lease length. Generous so a client that registers once and never
# renews keeps its endpoint served far longer than any realistic gap before it
# re-registers (e.g. across a controller restart). A renewing client renews at a
# fraction of this, so the absolute value only bounds how long a crashed task's
# endpoint lingers before the sweep reclaims it.
ENDPOINT_LEASE = Duration.from_hours(72)


class EndpointServiceImpl:
    """Leased service-discovery registry over the shared endpoints projection."""

    def __init__(
        self,
        *,
        db: ControllerDB,
        endpoints: EndpointsProjection,
        system_endpoints: dict[str, str] | None = None,
        lease: Duration = ENDPOINT_LEASE,
    ) -> None:
        self._db = db
        self._endpoints = endpoints
        self._system_endpoints: dict[str, str] = system_endpoints or {}
        self._lease = lease

    def register_system_endpoint(self, name: str, address: str) -> None:
        """Register a never-expiring ``/system/`` endpoint (e.g. the log server)."""
        self._system_endpoints[name] = address

    # --- RPC surface ---------------------------------------------------------

    def register_endpoint(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.RegisterEndpointResponse:
        """Register or renew a service endpoint, returning the granted lease.

        Re-registering with the same ``endpoint_id`` renews the lease (the
        underlying upsert keys off ``endpoint_id``). The ``task_id`` field
        carries the calling task's wire-format task ID; the endpoint is bound to
        that task so retry cleanup removes endpoints from superseded attempts.

        Endpoints register regardless of job state but only become visible to
        clients (lookup/list) while the task is executing (not terminal) and the
        lease is unexpired.
        """
        endpoint_id = request.endpoint_id or str(uuid.uuid4())

        task_id = JobName.from_wire(request.task_id)
        task_id.require_task()

        endpoint = EndpointRow(
            endpoint_id=endpoint_id,
            name=request.name,
            address=request.address,
            task_id=task_id,
            metadata=dict(request.metadata),
            registered_at=Timestamp.now(),
            lease_deadline=Timestamp.now().add(self._lease),
        )

        # Validation runs inside the writer transaction in
        # ``EndpointsProjection.add``: NOT_FOUND if the task row is missing,
        # FAILED_PRECONDITION if the task is terminal or the attempt is stale.
        with self._db.transaction() as cur:
            outcome = self._endpoints.add(cur, endpoint, expected_attempt_id=request.attempt_id)
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
            lease_duration=duration_to_proto(self._lease),
        )

    def unregister_endpoint(
        self,
        request: controller_pb2.Controller.UnregisterEndpointRequest,
        ctx: Any,
    ) -> job_pb2.Empty:
        """Unregister a service endpoint. Idempotent."""
        with self._db.transaction() as cur:
            self._endpoints.remove(cur, request.endpoint_id)
        return job_pb2.Empty()

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListEndpointsResponse:
        """List endpoints by name prefix (or exact name when ``request.exact`` is set).

        When ``request.task_ids`` is set, only endpoints registered by those
        tasks are returned, ANDed with any prefix/exact match. Expired leases are
        excluded. Names starting with ``/system/`` resolve from the in-memory
        system map instead of the leased table.
        """
        prefix = request.prefix
        if prefix.startswith("/system/"):
            return self._list_system_endpoints(prefix, exact=request.exact)

        endpoints = self._endpoints.query(
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
                )
                for e in endpoints
            ]
        )

    # --- Internal helpers ----------------------------------------------------

    def resolve_endpoint(self, name: str) -> str | None:
        """Resolve an endpoint name to its address, or None.

        Task endpoints (live leases) take priority over ``/system/`` endpoints.
        """
        row = self._endpoints.resolve(name)
        if row is not None:
            return row.address
        return self._system_endpoints.get(name)

    def _list_system_endpoints(self, prefix: str, *, exact: bool) -> controller_pb2.Controller.ListEndpointsResponse:
        """Resolve system endpoints from the in-memory map."""
        results: list[controller_pb2.Controller.Endpoint] = []
        for name, address in self._system_endpoints.items():
            matches = name == prefix if exact else name.startswith(prefix)
            if matches:
                results.append(controller_pb2.Controller.Endpoint(endpoint_id=name, name=name, address=address))
        return controller_pb2.Controller.ListEndpointsResponse(endpoints=results)
