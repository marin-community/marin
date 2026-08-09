# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

import secrets
from collections.abc import Sequence

from rigging.timing import Duration, Timestamp

from iris.cluster.authorization import authorize_resource_owner
from iris.cluster.controller.auth import (
    DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS,
    MAX_ENDPOINT_TOKEN_TTL_SECONDS,
)
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.pagination import (
    _decode_page_token,
    _encode_page_token,
    _page_size,
    _query_fingerprint,
    _require_kind,
)
from iris.cluster.controller.persistence import reads
from iris.cluster.controller.persistence.database import Tx
from iris.cluster.controller.persistence.projections.endpoints import (
    EndpointQuery as ProjectionEndpointQuery,
)
from iris.cluster.controller.persistence.projections.endpoints import (
    EndpointRow,
    EndpointsProjection,
)
from iris.cluster.controller.source_status import (
    _available_source,
    peer_source_statuses,
)
from iris.cluster.federation.protocol import FederationDirection
from iris.cluster.types import (
    JobName,
)
from iris.resources.endpoint import (
    EndpointAccess,
    EndpointDetail,
    EndpointQuery,
    EndpointSummary,
    EndpointToken,
)
from iris.resources.errors import (
    ResourceNotFound,
)
from iris.resources.identity import (
    ResourceKey,
    ResourceKind,
)
from iris.resources.source import (
    Page,
)

_MAX_ENDPOINT_PAGE = 500


class EndpointResources:
    """Endpoint resource operations."""

    def __init__(self, dependencies: ResourceDependencies) -> None:
        self._dependencies = dependencies

    def list_endpoints(self, query: EndpointQuery = EndpointQuery()) -> Page[EndpointSummary]:
        page_size = _page_size(query.page_size, _MAX_ENDPOINT_PAGE)
        fingerprint = _query_fingerprint(
            "endpoints",
            {
                "name_prefix": query.name_prefix,
                "task": query.task.resource_id if query.task is not None else None,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        task_ids = (JobName.from_wire(query.task.resource_id),) if query.task is not None else ()
        rows = sorted(
            self._dependencies.db.caches[EndpointsProjection].query(
                ProjectionEndpointQuery(name_prefix=query.name_prefix, task_ids=task_ids)
            ),
            key=lambda row: (row.name, row.endpoint_id),
        )
        system_endpoints = ()
        if query.task is None:
            system_endpoints = tuple(
                (name, address)
                for name, address in self._dependencies.endpoint_registry.system_endpoints()
                if query.name_prefix is None or name.startswith(query.name_prefix)
            )
        entries: list[tuple[str, str, EndpointRow | None, str]] = [(row.name, row.endpoint_id, row, "") for row in rows]
        entries.extend((name, name, None, address) for name, address in system_endpoints)
        entries.sort(key=lambda entry: (entry[0], entry[1]))
        if position is not None:
            last_key = (str(position["name"]), str(position["endpoint_id"]))
            entries = [entry for entry in entries if (entry[0], entry[1]) > last_key]
        page_entries = entries[:page_size]
        page_rows = [row for _, _, row, _ in page_entries if row is not None]
        coordinates = self._endpoint_coordinates(page_rows)
        peer_ids = {
            execution for authority, execution in coordinates.values() if execution != self._dependencies.cluster_id
        }
        next_token = None
        if len(entries) > page_size:
            last_name, last_endpoint_id, _, _ = page_entries[-1]
            next_token = _encode_page_token(
                fingerprint,
                {"name": last_name, "endpoint_id": last_endpoint_id},
            )
        return Page(
            items=tuple(
                (
                    self._endpoint_summary(row, coordinates[row.endpoint_id])
                    if row is not None
                    else self._system_endpoint_summary(name)
                )
                for name, _, row, _ in page_entries
            ),
            next_page_token=next_token,
            source_statuses=(
                _available_source(f"controller:{self._dependencies.cluster_id}"),
                *peer_source_statuses(self._dependencies, peer_ids),
            ),
        )

    def describe_endpoint(self, key: ResourceKey) -> EndpointDetail:
        return self.describe_endpoints((key,))[0]

    def describe_endpoints(self, keys: Sequence[ResourceKey]) -> tuple[EndpointDetail, ...]:
        """Return details for a bounded sequence of Endpoint keys."""
        if len(keys) > _MAX_ENDPOINT_PAGE:
            raise ValueError(f"Endpoint detail batch cannot exceed {_MAX_ENDPOINT_PAGE} items")
        for key in keys:
            _require_kind(key, ResourceKind.ENDPOINT)

        system_endpoints = dict(self._dependencies.endpoint_registry.system_endpoints())
        endpoint_ids = tuple(key.resource_id for key in keys if key.resource_id not in system_endpoints)
        rows = self._dependencies.db.caches[EndpointsProjection].query(
            ProjectionEndpointQuery(endpoint_ids=endpoint_ids)
        )
        rows_by_id = {row.endpoint_id: row for row in rows}
        coordinates = self._endpoint_coordinates(rows)
        details: list[EndpointDetail] = []
        for key in keys:
            system_address = system_endpoints.get(key.resource_id)
            if system_address is not None and key.cluster_id == self._dependencies.cluster_id:
                details.append(
                    EndpointDetail(
                        summary=self._system_endpoint_summary(key.resource_id),
                        address=system_address,
                        metadata={},
                    )
                )
                continue
            row = rows_by_id.get(key.resource_id)
            if row is None:
                raise ResourceNotFound(key.resource_id)
            row_coordinates = coordinates[row.endpoint_id]
            if row_coordinates[0] != key.cluster_id:
                raise ResourceNotFound(key.resource_id)
            details.append(
                EndpointDetail(
                    summary=self._endpoint_summary(row, row_coordinates),
                    address=row.address,
                    metadata=dict(row.metadata),
                )
            )
        return tuple(details)

    def mint_endpoint_token(self, key: ResourceKey, ttl: Duration | None) -> EndpointToken:
        detail = self.describe_endpoint(key)
        if self._dependencies.auth.jwt_manager is None:
            raise RuntimeError("JWT manager not configured")
        row = self._dependencies.endpoint_registry.resolve_task_endpoint(detail.summary.name)
        if row is None:
            raise ResourceNotFound(detail.summary.name)
        if self._dependencies.auth.provider:
            authorize_resource_owner(row.task_id.user)
        ttl_seconds = DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS
        if ttl is not None:
            ttl_seconds = max(1, min(int(ttl.to_seconds()), MAX_ENDPOINT_TOKEN_TTL_SECONDS))
        now = Timestamp.now()
        expires_at = Timestamp.from_ms(now.epoch_ms() + ttl_seconds * 1_000)
        token = self._dependencies.auth.jwt_manager.create_endpoint_token(
            row.name,
            f"iris_ket_{secrets.token_urlsafe(8)}",
            ttl_seconds=ttl_seconds,
        )
        return EndpointToken(
            token=token,
            expires_at=expires_at,
            capability_url=self._dependencies.capability_url_config.build(row.name, token),
        )

    def _endpoint_summary(self, row: EndpointRow, coordinates: tuple[str, str]) -> EndpointSummary:
        authority, execution = coordinates
        task = ResourceKey(authority, ResourceKind.TASK, row.task_id.to_wire())
        return EndpointSummary(
            key=ResourceKey(authority, ResourceKind.ENDPOINT, row.endpoint_id),
            endpoint_id=row.endpoint_id,
            name=row.name,
            task=task,
            execution_cluster_id=execution,
            access=EndpointAccess.from_storage(row.access),
            lease_deadline=row.lease_deadline,
        )

    def _system_endpoint_summary(self, name: str) -> EndpointSummary:
        return EndpointSummary(
            key=ResourceKey(self._dependencies.cluster_id, ResourceKind.ENDPOINT, name),
            endpoint_id=name,
            name=name,
            task=None,
            execution_cluster_id=self._dependencies.cluster_id,
            access=EndpointAccess.PRIVATE,
            lease_deadline=None,
        )

    def _endpoint_coordinates(self, rows: list[EndpointRow]) -> dict[str, tuple[str, str]]:
        if not rows:
            return {}
        roots = {row.task_id.root_job for row in rows}
        with self._dependencies.db.read_snapshot() as tx:
            jobs = self._job_rows(tx, roots)
        coordinates: dict[str, tuple[str, str]] = {}
        for row in rows:
            job = jobs.get(row.task_id.root_job)
            if job is None:
                raise ResourceNotFound(row.task_id.root_job.to_wire())
            coordinates[row.endpoint_id] = (
                self._authority_cluster(job),
                row.peer_id or self._dependencies.cluster_id,
            )
        return coordinates

    def _authority_cluster(self, row: reads.JobCoordinates) -> str:
        direction = getattr(row, "direction", None)
        if direction == int(FederationDirection.RECEIVED):
            return str(row.peer_id)
        return self._dependencies.cluster_id

    def _job_rows(self, tx: Tx, job_ids: set[JobName]) -> dict[JobName, reads.JobCoordinates]:
        return reads.job_coordinates(tx, job_ids)
