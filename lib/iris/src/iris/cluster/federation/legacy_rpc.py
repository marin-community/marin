# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate federation sync records at the legacy ControllerService boundary."""

from rigging.timing import Timestamp

from iris.cluster.federation.store import (
    FederationJobDelta,
    FederationSyncBatch,
    SyncedAttempt,
    SyncedEndpoint,
    SyncedJob,
    SyncedTask,
)
from iris.cluster.resources.endpoint import EndpointAccess
from iris.cluster.types import JobName, ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_from_proto, duration_to_proto, timestamp_from_proto, timestamp_to_proto


def federation_batch_from_legacy(response: controller_pb2.Controller.FederationSyncResponse) -> FederationSyncBatch:
    """Decode the peer's legacy response into canonical federation records."""

    def maybe_timestamp(message, field: str) -> Timestamp | None:
        return timestamp_from_proto(getattr(message, field)) if message.HasField(field) else None

    deltas = []
    for delta in response.deltas:
        summary = None
        if delta.HasField("summary"):
            wire = delta.summary
            device = None
            if wire.resources.HasField("device"):
                device = job_pb2.DeviceConfig()
                device.CopyFrom(wire.resources.device)
            summary = SyncedJob(
                job_id=JobName.from_wire(delta.job_id),
                state=wire.state,
                error_message=wire.error,
                exit_code=wire.exit_code or None,
                submitted_at=maybe_timestamp(wire, "submitted_at"),
                started_at=maybe_timestamp(wire, "started_at"),
                finished_at=maybe_timestamp(wire, "finished_at"),
                task_count=wire.task_count,
                backend_id=wire.backend_id,
                resources=ResourceSpec(
                    cpu=wire.resources.cpu_millicores / 1_000,
                    memory=wire.resources.memory_bytes,
                    disk=wire.resources.disk_bytes,
                    device=device,
                ),
            )
        tasks = []
        for wire in delta.changed_tasks:
            attempts = tuple(
                SyncedAttempt(
                    attempt_id=attempt.attempt_id,
                    state=attempt.state,
                    exit_code=attempt.exit_code or None,
                    error_message=attempt.error,
                    attempt_uid=attempt.attempt_uid,
                    started_at=maybe_timestamp(attempt, "started_at"),
                    finished_at=maybe_timestamp(attempt, "finished_at"),
                )
                for attempt in wire.attempts
            )
            tasks.append(
                SyncedTask(
                    task_id=JobName.from_wire(wire.task_id),
                    state=wire.state,
                    error_message=wire.error,
                    exit_code=wire.exit_code or None,
                    submitted_at=maybe_timestamp(wire, "submitted_at"),
                    started_at=maybe_timestamp(wire, "started_at"),
                    finished_at=maybe_timestamp(wire, "finished_at"),
                    current_attempt_id=wire.current_attempt_id,
                    worker_address=wire.worker_address,
                    worker_label=wire.worker_id or wire.worker_address,
                    status_message=wire.status_message,
                    backend_id=wire.backend_id,
                    attempts=attempts,
                )
            )
        deltas.append(
            FederationJobDelta(
                job_id=JobName.from_wire(delta.job_id),
                summary=summary,
                changed_tasks=tuple(tasks),
                tombstone=delta.tombstone,
            )
        )
    endpoints = tuple(
        SyncedEndpoint(
            endpoint_id=wire.endpoint_id,
            name=wire.name,
            address=wire.address,
            task_id=JobName.from_wire(wire.task_id),
            access=(
                EndpointAccess.LINK
                if wire.access == controller_pb2.Controller.ENDPOINT_ACCESS_LINK
                else EndpointAccess.PRIVATE
            ),
            metadata=dict(wire.metadata),
            lease_remaining=(duration_from_proto(wire.lease_remaining) if wire.HasField("lease_remaining") else None),
        )
        for wire in response.endpoints
    )
    return FederationSyncBatch(tuple(deltas), response.next_cursor, response.cursor_stale, endpoints)


def federation_batch_to_legacy(batch: FederationSyncBatch) -> controller_pb2.Controller.FederationSyncResponse:
    """Encode canonical federation records for a legacy peer response."""
    deltas = []
    for delta in batch.deltas:
        wire_delta = controller_pb2.Controller.FederationJobDelta(
            job_id=delta.job_id.to_wire(), tombstone=delta.tombstone
        )
        if delta.summary is not None:
            summary = delta.summary
            wire_summary = job_pb2.JobStatus(
                job_id=summary.job_id.to_wire(),
                state=summary.state,
                error=summary.error_message,
                exit_code=summary.exit_code or 0,
                backend_id=summary.backend_id,
                task_count=summary.task_count,
                resources=summary.resources.to_exact_proto(),
            )
            _set_timestamps(wire_summary, summary.submitted_at, summary.started_at, summary.finished_at)
            wire_delta.summary.CopyFrom(wire_summary)
        for task in delta.changed_tasks:
            wire_task = job_pb2.TaskStatus(
                task_id=task.task_id.to_wire(),
                state=task.state,
                error=task.error_message,
                exit_code=task.exit_code or 0,
                current_attempt_id=task.current_attempt_id,
                worker_address=task.worker_address,
                worker_id=task.worker_label,
                status_message=task.status_message,
                backend_id=task.backend_id,
            )
            _set_timestamps(wire_task, task.submitted_at, task.started_at, task.finished_at)
            for attempt in task.attempts:
                wire_attempt = job_pb2.TaskAttempt(
                    attempt_id=attempt.attempt_id,
                    state=attempt.state,
                    exit_code=attempt.exit_code or 0,
                    error=attempt.error_message,
                    attempt_uid=attempt.attempt_uid,
                )
                if attempt.started_at is not None:
                    wire_attempt.started_at.CopyFrom(timestamp_to_proto(attempt.started_at))
                if attempt.finished_at is not None:
                    wire_attempt.finished_at.CopyFrom(timestamp_to_proto(attempt.finished_at))
                wire_task.attempts.append(wire_attempt)
            wire_delta.changed_tasks.append(wire_task)
        deltas.append(wire_delta)

    endpoints = []
    for endpoint in batch.endpoints:
        wire_endpoint = controller_pb2.Controller.FederationEndpoint(
            endpoint_id=endpoint.endpoint_id,
            name=endpoint.name,
            address=endpoint.address,
            task_id=endpoint.task_id.to_wire(),
            access=(
                controller_pb2.Controller.ENDPOINT_ACCESS_LINK
                if endpoint.access == EndpointAccess.LINK
                else controller_pb2.Controller.ENDPOINT_ACCESS_PRIVATE
            ),
            metadata=dict(endpoint.metadata),
        )
        if endpoint.lease_remaining is not None:
            wire_endpoint.lease_remaining.CopyFrom(duration_to_proto(endpoint.lease_remaining))
        endpoints.append(wire_endpoint)
    return controller_pb2.Controller.FederationSyncResponse(
        deltas=deltas,
        next_cursor=batch.next_cursor,
        cursor_stale=batch.cursor_stale,
        endpoints=endpoints,
    )


def _set_timestamps(
    message, submitted_at: Timestamp | None, started_at: Timestamp | None, finished_at: Timestamp | None
):
    if submitted_at is not None:
        message.submitted_at.CopyFrom(timestamp_to_proto(submitted_at))
    if started_at is not None:
        message.started_at.CopyFrom(timestamp_to_proto(started_at))
    if finished_at is not None:
        message.finished_at.CopyFrom(timestamp_to_proto(finished_at))
