# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller operations that serve federation peers."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.server_auth import require_identity
from rigging.timing import Duration, Timestamp
from sqlalchemy import Row

from iris.cluster.controller import jobs, reads, tasks
from iris.cluster.controller.codec import resource_spec_from_job_row
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.schema import tasks_table
from iris.cluster.federation.manager import FederationManager
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import FEDERATION_PEER_ROLE
from iris.time_proto import duration_to_proto, timestamp_to_proto


class FederationRuntime(Protocol):
    @property
    def federation(self) -> FederationManager: ...


@dataclass(frozen=True, slots=True)
class FederationDependencies:
    db: ControllerDB
    runtime: FederationRuntime


def list_peers(
    dependencies: FederationDependencies,
    _request: controller_pb2.Controller.ListPeersRequest,
    _ctx: RequestContext,
) -> controller_pb2.Controller.ListPeersResponse:
    """List peers this controller may delegate whole jobs to."""
    require_identity()
    return controller_pb2.Controller.ListPeersResponse(peers=dependencies.runtime.federation.peer_summaries())


def _job_summary(q: Tx, job) -> job_pb2.JobStatus:
    summaries = reads.task_summaries_for_jobs(
        q,
        {job.job_id},
        attempt_counts=q.caches[AttemptCountsProjection].get_jobs(q, [job.job_id]),
    )
    status = job_pb2.JobStatus(
        job_id=job.job_id.to_wire(),
        state=job.state,
        error=job.error or "",
        exit_code=job.exit_code or 0,
        name=job.name,
        backend_id=job.backend_id or "",
        cluster=job.cluster,
        resources=resource_spec_from_job_row(job),
    )
    jobs.apply_job_status_counts(status, summaries.get(job.job_id), job.job_id)
    if job.started_at_ms:
        status.started_at.CopyFrom(timestamp_to_proto(job.started_at_ms))
    if job.finished_at_ms:
        status.finished_at.CopyFrom(timestamp_to_proto(job.finished_at_ms))
    if job.submitted_at_ms:
        status.submitted_at.CopyFrom(timestamp_to_proto(job.submitted_at_ms))
    return status


def _job_delta(
    q: Tx,
    job_id: JobName,
    *,
    task_indexes: set[int] | None,
) -> controller_pb2.Controller.FederationJobDelta | None:
    job = reads.get_job_detail(q, job_id)
    if job is None:
        return None
    task_rows = [
        row
        for row in q.execute(reads.task_detail_query().where(tasks_table.c.job_id == job_id)).all()
        if task_indexes is None or row.task_id.task_index in task_indexes
    ]
    attempts_by_task = reads.all_attempts_for_tasks(q, [row.task_id for row in task_rows])
    changed_tasks = [
        tasks.task_to_proto(tasks.TaskWithAttempts.from_row(row, attempts_by_task.get(row.task_id, ())))
        for row in task_rows
    ]
    return controller_pb2.Controller.FederationJobDelta(
        job_id=job_id.to_wire(),
        summary=_job_summary(q, job),
        changed_tasks=changed_tasks,
    )


def _endpoint_snapshot(
    q: Tx,
    requester_id: str,
    now: Timestamp,
) -> list[controller_pb2.Controller.FederationEndpoint]:
    endpoints: list[controller_pb2.Controller.FederationEndpoint] = []
    for endpoint in reads.live_endpoints_for_requester(q, requester_id, now):
        proto = controller_pb2.Controller.FederationEndpoint(
            endpoint_id=endpoint.endpoint_id,
            name=endpoint.name,
            address=endpoint.address,
            task_id=endpoint.task_id.to_wire(),
            access=endpoint.access,
            metadata=endpoint.metadata,
        )
        if endpoint.lease_deadline is not None:
            remaining_ms = max(0, endpoint.lease_deadline.epoch_ms() - now.epoch_ms())
            proto.lease_remaining.CopyFrom(duration_to_proto(Duration.from_ms(remaining_ms)))
        endpoints.append(proto)
    return endpoints


def _authorize_sync(requester_id: str) -> None:
    identity = require_identity()
    if identity.role == FEDERATION_PEER_ROLE:
        if requester_id != identity.user_id:
            raise ConnectError(
                Code.PERMISSION_DENIED,
                f"Peer {identity.user_id!r} may not sync jobs for requester {requester_id!r}",
            )
        return
    if identity.role != "admin":
        raise ConnectError(Code.PERMISSION_DENIED, "federation_sync requires a federation-peer or admin identity")


@dataclass(slots=True)
class _FederationChange:
    job_id: JobName
    tombstone: bool = False
    task_indexes: set[int] | None = field(default_factory=set)


def _fold_changelog(rows: Iterable[Row]) -> list[_FederationChange]:
    changes_by_job: dict[JobName, _FederationChange] = {}
    for row in rows:
        change = changes_by_job.setdefault(row.job_id, _FederationChange(job_id=row.job_id))
        if row.tombstone:
            change.tombstone = True
        elif change.tombstone:
            change.tombstone = False
            change.task_indexes = None
        elif row.task_index is None:
            change.task_indexes = None
        elif change.task_indexes is not None:
            change.task_indexes.add(row.task_index)
    return list(changes_by_job.values())


def federation_sync(
    dependencies: FederationDependencies,
    request: controller_pb2.Controller.FederationSyncRequest,
    _ctx: RequestContext,
) -> controller_pb2.Controller.FederationSyncResponse:
    """Return the peer-visible job changes since the requester's cursor."""
    requester_id = request.requester_id
    _authorize_sync(requester_id)
    cursor = request.cursor
    cursor_seq = int(cursor) if cursor else 0
    deltas: list[controller_pb2.Controller.FederationJobDelta] = []

    with dependencies.db.read_snapshot() as q:
        min_seq = reads.changelog_min_seq(q)
        next_cursor = str(reads.changelog_max_seq(q))
        endpoints = _endpoint_snapshot(q, requester_id, Timestamp.now())
        stale = not cursor or (min_seq > 0 and cursor_seq < min_seq - 1)
        if stale:
            for job_id in reads.received_jobs_for_requester(q, requester_id):
                delta = _job_delta(q, job_id, task_indexes=None)
                if delta is not None:
                    deltas.append(delta)
            return controller_pb2.Controller.FederationSyncResponse(
                deltas=deltas,
                next_cursor=next_cursor,
                cursor_stale=True,
                endpoints=endpoints,
            )

        changes = _fold_changelog(reads.changelog_rows_since(q, requester_id, cursor_seq))
        for change in changes:
            if change.tombstone:
                deltas.append(
                    controller_pb2.Controller.FederationJobDelta(job_id=change.job_id.to_wire(), tombstone=True)
                )
                continue
            delta = _job_delta(q, change.job_id, task_indexes=change.task_indexes)
            if delta is not None:
                deltas.append(delta)

    return controller_pb2.Controller.FederationSyncResponse(
        deltas=deltas,
        next_cursor=next_cursor,
        cursor_stale=False,
        endpoints=endpoints,
    )
