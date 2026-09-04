# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller operations that serve federation peers."""

from dataclasses import dataclass
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.server_auth import require_identity
from rigging.timing import Duration, Timestamp

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
    all_tasks: bool,
    task_indexes: set[int],
) -> controller_pb2.Controller.FederationJobDelta | None:
    job = reads.get_job_detail(q, job_id)
    if job is None:
        return None
    task_rows = [
        row
        for row in q.execute(reads.task_detail_query().where(tasks_table.c.job_id == job_id)).all()
        if all_tasks or row.task_id.task_index in task_indexes
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
                delta = _job_delta(q, job_id, all_tasks=True, task_indexes=set())
                if delta is not None:
                    deltas.append(delta)
            return controller_pb2.Controller.FederationSyncResponse(
                deltas=deltas,
                next_cursor=next_cursor,
                cursor_stale=True,
                endpoints=endpoints,
            )

        tombstoned: dict[JobName, bool] = {}
        all_tasks: dict[JobName, bool] = {}
        indexes: dict[JobName, set[int]] = {}
        order: list[JobName] = []
        for row in reads.changelog_rows_since(q, requester_id, cursor_seq):
            if row.job_id not in tombstoned:
                tombstoned[row.job_id] = False
                all_tasks[row.job_id] = False
                indexes[row.job_id] = set()
                order.append(row.job_id)
            if row.tombstone:
                tombstoned[row.job_id] = True
            elif tombstoned[row.job_id]:
                tombstoned[row.job_id] = False
                all_tasks[row.job_id] = True
            elif row.task_index is None:
                all_tasks[row.job_id] = True
            else:
                indexes[row.job_id].add(row.task_index)

        for job_id in order:
            if tombstoned[job_id]:
                deltas.append(controller_pb2.Controller.FederationJobDelta(job_id=job_id.to_wire(), tombstone=True))
                continue
            delta = _job_delta(q, job_id, all_tasks=all_tasks[job_id], task_indexes=indexes[job_id])
            if delta is not None:
                deltas.append(delta)

    return controller_pb2.Controller.FederationSyncResponse(
        deltas=deltas,
        next_cursor=next_cursor,
        cursor_stale=False,
        endpoints=endpoints,
    )
