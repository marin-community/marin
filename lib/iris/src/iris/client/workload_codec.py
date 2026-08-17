# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode the deployed ControllerService workload messages for public clients."""

from rigging.timing import Timestamp

from iris.client.workload import (
    AttemptStatus,
    BuildMetrics,
    Device,
    DeviceKind,
    JobStatus,
    ResourceRequest,
    ResourceUsage,
    TaskStatus,
)
from iris.cluster.types import JobName
from iris.resources.state import FederationState, JobState, TaskState
from iris.rpc import job_pb2, time_pb2
from iris.time_proto import timestamp_from_proto

_FEDERATION_STATES: dict[int, FederationState] = {
    job_pb2.PEER_STATUS_NONE: FederationState.LOCAL,
    job_pb2.PEER_STATUS_PENDING_SCHEDULING: FederationState.PENDING,
    job_pb2.PEER_STATUS_ASSIGNED: FederationState.ASSIGNED,
    job_pb2.PEER_STATUS_SYNCED: FederationState.SYNCED,
    job_pb2.PEER_STATUS_REJECTED: FederationState.REJECTED,
}


def job_state_from_proto(value: int) -> JobState:
    return JobState(job_pb2.JobState.Name(value).removeprefix("JOB_STATE_").lower())


def _task_state_from_proto(value: int) -> TaskState:
    return TaskState(job_pb2.TaskState.Name(value).removeprefix("TASK_STATE_").lower())


def _timestamp(value: time_pb2.Timestamp, *, present: bool) -> Timestamp | None:
    return timestamp_from_proto(value) if present else None


def _device(value: job_pb2.DeviceConfig) -> Device | None:
    if value.HasField("gpu"):
        return Device(DeviceKind.GPU, value.gpu.variant, value.gpu.count)
    if value.HasField("tpu"):
        return Device(DeviceKind.TPU, value.tpu.variant, value.tpu.count, value.tpu.topology)
    if value.HasField("cpu"):
        return Device(DeviceKind.CPU, value.cpu.variant, 0)
    return None


def _resources(value: job_pb2.ResourceSpecProto) -> ResourceRequest:
    return ResourceRequest(
        cpu_millicores=value.cpu_millicores,
        memory_bytes=value.memory_bytes,
        disk_bytes=value.disk_bytes,
        device=_device(value.device) if value.HasField("device") else None,
    )


def _usage(value: job_pb2.ResourceUsage) -> ResourceUsage:
    return ResourceUsage(
        memory_mb=value.memory_mb,
        disk_mb=value.disk_mb,
        cpu_millicores=value.cpu_millicores,
        memory_peak_mb=value.memory_peak_mb,
        process_count=value.process_count,
    )


def _build_metrics(value: job_pb2.BuildMetrics) -> BuildMetrics:
    return BuildMetrics(
        started_at=_timestamp(value.build_started, present=value.HasField("build_started")),
        finished_at=_timestamp(value.build_finished, present=value.HasField("build_finished")),
        from_cache=value.from_cache,
        image_tag=value.image_tag,
    )


def attempt_status_from_proto(value: job_pb2.TaskAttempt) -> AttemptStatus:
    return AttemptStatus(
        attempt_number=value.attempt_id,
        attempt_uid=value.attempt_uid,
        state=_task_state_from_proto(value.state),
        worker_id=value.worker_id,
        exit_code=value.exit_code,
        error_message=value.error,
        started_at=_timestamp(value.started_at, present=value.HasField("started_at")),
        finished_at=_timestamp(value.finished_at, present=value.HasField("finished_at")),
        is_worker_failure=value.is_worker_failure,
        pod_name=value.pod_name,
        pod_uid=value.pod_uid,
        node_name=value.node_name,
        terminal_reason=value.terminal_reason,
    )


def task_status_from_proto(value: job_pb2.TaskStatus) -> TaskStatus:
    return TaskStatus(
        task_id=JobName.from_wire(value.task_id),
        state=_task_state_from_proto(value.state),
        worker_id=value.worker_id,
        worker_address=value.worker_address,
        exit_code=value.exit_code,
        error_message=value.error,
        submitted_at=_timestamp(value.submitted_at, present=value.HasField("submitted_at")),
        started_at=_timestamp(value.started_at, present=value.HasField("started_at")),
        finished_at=_timestamp(value.finished_at, present=value.HasField("finished_at")),
        ports=value.ports,
        resource_usage=_usage(value.resource_usage) if value.HasField("resource_usage") else None,
        build_metrics=_build_metrics(value.build_metrics) if value.HasField("build_metrics") else None,
        current_attempt_number=value.current_attempt_id,
        attempts=tuple(attempt_status_from_proto(attempt) for attempt in value.attempts),
        pending_reason=value.pending_reason,
        can_be_scheduled=value.can_be_scheduled,
        container_id=value.container_id,
        backend_id=value.backend_id,
        execution_cluster_id=value.cluster,
        status_message=value.status_message,
    )


def _task_state_counts(values: dict[str, int]) -> dict[TaskState, int]:
    result: dict[TaskState, int] = {}
    for name, count in values.items():
        try:
            state = TaskState[name.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown Task state count {name!r}") from exc
        result[state] = count
    return result


def job_status_from_proto(value: job_pb2.JobStatus) -> JobStatus:
    parent_job_id = JobName.from_wire(value.parent_job_id) if value.parent_job_id else None
    return JobStatus(
        job_id=JobName.from_wire(value.job_id),
        state=job_state_from_proto(value.state),
        exit_code=value.exit_code,
        error_message=value.error,
        submitted_at=_timestamp(value.submitted_at, present=value.HasField("submitted_at")),
        started_at=_timestamp(value.started_at, present=value.HasField("started_at")),
        finished_at=_timestamp(value.finished_at, present=value.HasField("finished_at")),
        ports=value.ports,
        status_message=value.status_message,
        build_metrics=_build_metrics(value.build_metrics) if value.HasField("build_metrics") else None,
        failure_count=value.failure_count,
        preemption_count=value.preemption_count,
        tasks=tuple(task_status_from_proto(task) for task in value.tasks),
        name=value.name,
        resources=_resources(value.resources),
        task_state_counts=_task_state_counts(dict(value.task_state_counts)),
        task_count=value.task_count,
        completed_count=value.completed_count,
        pending_reason=value.pending_reason,
        has_children=value.has_children,
        parent_job_id=parent_job_id,
        backend_id=value.backend_id,
        execution_cluster_id=value.cluster,
        federation_state=_FEDERATION_STATES[value.peer_status],
        submitting_user=value.submitting_user,
    )
