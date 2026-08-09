# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate WorkerService protobuf messages at the RPC boundary."""

from rigging.provenance import Provenance
from rigging.timing import Duration

from iris.cluster.resources.attempt import AttemptLaunch, AttemptLaunchTemplate, AttemptObservation
from iris.cluster.resources.endpoint import ExecResult, ProfileConfiguration
from iris.cluster.resources.job import ContainerProfile, CoschedulingConfig, PriorityBand
from iris.cluster.resources.state import TaskState
from iris.cluster.resources.system import ProcessInfo
from iris.cluster.resources.worker import (
    AttemptStatus,
    DesiredAttempt,
    ResourceUsage,
    StopReason,
    WorkerMetadata,
    WorkerReconcileRequest,
    WorkerReconcileResponse,
    WorkerResourceSnapshot,
    WorkerTaskStatus,
)
from iris.cluster.types import AttemptUid, JobName
from iris.rpc import job_pb2, worker_pb2
from iris.rpc.legacy_job_codec import (
    attribute_value_from_proto,
    attribute_value_to_proto,
    constraint_from_proto,
    constraint_to_proto,
    device_from_proto,
    device_to_proto,
    environment_from_proto,
    environment_to_proto,
    resource_spec_from_proto,
    resource_spec_to_proto,
    runtime_entrypoint_from_proto,
    runtime_entrypoint_to_proto,
)
from iris.rpc.profile_codec import profile_configuration_from_proto
from iris.time_proto import duration_to_proto, timestamp_from_proto, timestamp_to_proto


def provenance_to_proto(value: Provenance) -> job_pb2.Provenance:
    return job_pb2.Provenance(
        tree_hash=value.tree_hash,
        base_commit=value.base_commit,
        dirty=value.dirty,
        branch=value.branch or "",
        built_by=value.built_by or "",
    )


def provenance_from_proto(value: job_pb2.Provenance) -> Provenance:
    return Provenance(
        tree_hash=value.tree_hash,
        base_commit=value.base_commit or value.tree_hash,
        dirty=value.dirty,
        branch=value.branch or None,
        built_by=value.built_by or None,
    )


def worker_metadata_to_proto(value: WorkerMetadata) -> job_pb2.WorkerMetadata:
    result = job_pb2.WorkerMetadata(
        hostname=value.hostname,
        ip_address=value.ip_address,
        cpu_count=value.cpu_count,
        memory_bytes=value.memory_bytes,
        disk_bytes=value.disk_bytes,
        tpu_name=value.tpu_name,
        tpu_worker_hostnames=value.tpu_worker_hostnames,
        tpu_worker_id=value.tpu_worker_id,
        tpu_chips_per_host_bounds=value.tpu_chips_per_host_bounds,
        gpu_count=value.gpu_count,
        gpu_name=value.gpu_name,
        gpu_memory_mb=value.gpu_memory_mb,
        gce_instance_name=value.gce_instance_name,
        gce_zone=value.gce_zone,
        attributes={key: attribute_value_to_proto(item) for key, item in value.attributes.items()},
    )
    if value.device is not None:
        result.device.CopyFrom(device_to_proto(value.device))
    if value.provenance is not None:
        result.provenance.CopyFrom(provenance_to_proto(value.provenance))
    return result


def worker_metadata_from_proto(value: job_pb2.WorkerMetadata) -> WorkerMetadata:
    return WorkerMetadata(
        hostname=value.hostname,
        ip_address=value.ip_address,
        cpu_count=value.cpu_count,
        memory_bytes=value.memory_bytes,
        disk_bytes=value.disk_bytes,
        device=device_from_proto(value.device) if value.HasField("device") else None,
        tpu_name=value.tpu_name,
        tpu_worker_hostnames=value.tpu_worker_hostnames,
        tpu_worker_id=value.tpu_worker_id,
        tpu_chips_per_host_bounds=value.tpu_chips_per_host_bounds,
        gpu_count=value.gpu_count,
        gpu_name=value.gpu_name,
        gpu_memory_mb=value.gpu_memory_mb,
        gce_instance_name=value.gce_instance_name,
        gce_zone=value.gce_zone,
        attributes={key: attribute_value_from_proto(item) for key, item in value.attributes.items()},
        provenance=provenance_from_proto(value.provenance) if value.HasField("provenance") else None,
    )


def resource_usage_to_proto(value: ResourceUsage) -> job_pb2.ResourceUsage:
    return job_pb2.ResourceUsage(
        memory_mb=value.memory_mb,
        disk_mb=value.disk_mb,
        cpu_millicores=value.cpu_millicores,
        memory_peak_mb=value.memory_peak_mb,
        process_count=value.process_count,
    )


def resource_usage_from_proto(value: job_pb2.ResourceUsage) -> ResourceUsage:
    return ResourceUsage(
        memory_mb=value.memory_mb,
        disk_mb=value.disk_mb,
        cpu_millicores=value.cpu_millicores,
        memory_peak_mb=value.memory_peak_mb,
        process_count=value.process_count,
    )


def worker_resource_snapshot_to_proto(value: WorkerResourceSnapshot) -> job_pb2.WorkerResourceSnapshot:
    result = job_pb2.WorkerResourceSnapshot(
        host_cpu_percent=value.host_cpu_percent,
        memory_used_bytes=value.memory_used_bytes,
        memory_total_bytes=value.memory_total_bytes,
        disk_used_bytes=value.disk_used_bytes,
        disk_total_bytes=value.disk_total_bytes,
        running_task_count=value.running_task_count,
        total_process_count=value.total_process_count,
        net_recv_bytes=value.net_recv_bytes,
        net_sent_bytes=value.net_sent_bytes,
    )
    if value.timestamp is not None:
        result.timestamp.CopyFrom(timestamp_to_proto(value.timestamp))
    return result


def worker_resource_snapshot_from_proto(value: job_pb2.WorkerResourceSnapshot) -> WorkerResourceSnapshot:
    return WorkerResourceSnapshot(
        timestamp=timestamp_from_proto(value.timestamp) if value.HasField("timestamp") else None,
        host_cpu_percent=value.host_cpu_percent,
        memory_used_bytes=value.memory_used_bytes,
        memory_total_bytes=value.memory_total_bytes,
        disk_used_bytes=value.disk_used_bytes,
        disk_total_bytes=value.disk_total_bytes,
        running_task_count=value.running_task_count,
        total_process_count=value.total_process_count,
        net_recv_bytes=value.net_recv_bytes,
        net_sent_bytes=value.net_sent_bytes,
    )


def attempt_launch_from_proto(value: job_pb2.RunTaskRequest) -> AttemptLaunch:
    timeout = None
    if value.HasField("timeout") and value.timeout.milliseconds > 0:
        timeout = Duration.from_ms(value.timeout.milliseconds)
    coscheduling = None
    if value.HasField("coscheduling") and value.coscheduling.group_by:
        coscheduling = CoschedulingConfig(value.coscheduling.group_by)
    return AttemptLaunch(
        task_id=JobName.from_wire(value.task_id),
        attempt_id=value.attempt_id,
        attempt_uid=AttemptUid(value.attempt_uid),
        template=AttemptLaunchTemplate(
            num_tasks=value.num_tasks,
            entrypoint=runtime_entrypoint_from_proto(value.entrypoint),
            environment=environment_from_proto(value.environment),
            bundle_id=value.bundle_id,
            resources=resource_spec_from_proto(value.resources),
            timeout=timeout,
            ports=tuple(value.ports),
            constraints=tuple(constraint_from_proto(item) for item in value.constraints),
            task_image=value.task_image,
            coscheduling=coscheduling,
            priority_band=PriorityBand(value.priority),
            container_profile=ContainerProfile(value.container_profile),
        ),
    )


def attempt_launch_to_proto(value: AttemptLaunch) -> job_pb2.RunTaskRequest:
    template = value.template
    result = job_pb2.RunTaskRequest(
        task_id=value.task_id.to_wire(),
        attempt_id=value.attempt_id,
        attempt_uid=value.attempt_uid,
        num_tasks=template.num_tasks,
        entrypoint=runtime_entrypoint_to_proto(template.entrypoint),
        environment=environment_to_proto(template.environment),
        bundle_id=template.bundle_id,
        resources=resource_spec_to_proto(template.resources),
        ports=template.ports,
        constraints=[constraint_to_proto(item) for item in template.constraints],
        task_image=template.task_image,
        priority=int(template.priority_band),
        container_profile=int(template.container_profile),
    )
    if template.timeout is not None:
        result.timeout.CopyFrom(duration_to_proto(template.timeout))
    if template.coscheduling is not None:
        result.coscheduling.group_by = template.coscheduling.group_by
    return result


def task_status_to_proto(value: WorkerTaskStatus) -> job_pb2.TaskStatus:
    result = job_pb2.TaskStatus(
        task_id=value.task_id.to_wire(),
        state=int(value.state),
        worker_id=value.worker_id,
        worker_address=value.worker_address,
        exit_code=value.exit_code or 0,
        error=value.error,
        ports=dict(value.ports),
        current_attempt_id=value.attempt_id,
        container_id=value.container_id,
        status_message=value.status_message,
    )
    if value.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(value.started_at))
    if value.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(value.finished_at))
    if value.resource_usage is not None:
        result.resource_usage.CopyFrom(resource_usage_to_proto(value.resource_usage))
    if value.build_metrics is not None:
        metrics = value.build_metrics
        result.build_metrics.from_cache = metrics.from_cache
        result.build_metrics.image_tag = metrics.image_tag
        if metrics.build_started is not None:
            result.build_metrics.build_started.CopyFrom(timestamp_to_proto(metrics.build_started))
        if metrics.build_finished is not None:
            result.build_metrics.build_finished.CopyFrom(timestamp_to_proto(metrics.build_finished))
    return result


def profile_configuration_request_from_proto(
    value: job_pb2.ProfileTaskRequest,
) -> tuple[str, Duration, ProfileConfiguration | None]:
    return (
        value.target,
        Duration.from_seconds(value.duration_seconds),
        profile_configuration_from_proto(value.profile_type) if value.HasField("profile_type") else None,
    )


def exec_result_to_proto(value: ExecResult) -> worker_pb2.Worker.ExecInContainerResponse:
    return worker_pb2.Worker.ExecInContainerResponse(
        exit_code=value.exit_code,
        stdout=value.stdout,
        stderr=value.stderr,
        error=value.error_message,
    )


def process_info_to_proto(value: ProcessInfo) -> job_pb2.ProcessInfo:
    return job_pb2.ProcessInfo(
        hostname=value.hostname,
        pid=value.pid,
        python_version=value.python_version,
        uptime_ms=value.uptime_ms,
        memory_rss_bytes=value.memory_rss_bytes,
        memory_vms_bytes=value.memory_vms_bytes,
        thread_count=value.thread_count,
        open_fd_count=value.open_fd_count,
        memory_total_bytes=value.memory_total_bytes,
        cpu_count=value.cpu_count,
        cpu_millicores=value.cpu_millicores,
        provenance=provenance_to_proto(value.provenance),
    )


def process_info_from_proto(value: job_pb2.ProcessInfo) -> ProcessInfo:
    return ProcessInfo(
        hostname=value.hostname,
        pid=value.pid,
        python_version=value.python_version,
        uptime_ms=value.uptime_ms,
        memory_rss_bytes=value.memory_rss_bytes,
        memory_vms_bytes=value.memory_vms_bytes,
        thread_count=value.thread_count,
        open_fd_count=value.open_fd_count,
        memory_total_bytes=value.memory_total_bytes,
        cpu_count=value.cpu_count,
        cpu_millicores=value.cpu_millicores,
        provenance=provenance_from_proto(value.provenance),
    )


def reconcile_request_from_proto(value: worker_pb2.Worker.ReconcileRequest) -> WorkerReconcileRequest:
    desired = []
    for item in value.desired:
        if item.HasField("run"):
            launch = attempt_launch_from_proto(item.run.request) if item.run.HasField("request") else None
            desired.append(DesiredAttempt(AttemptUid(item.attempt_uid), launch=launch))
        else:
            desired.append(DesiredAttempt(AttemptUid(item.attempt_uid), stop_reason=StopReason(item.stop)))
    return WorkerReconcileRequest(worker_id=value.worker_id, desired=tuple(desired))


def _attempt_status_to_proto(value: AttemptStatus) -> worker_pb2.Worker.AttemptObservation:
    observation = value.observation
    result = worker_pb2.Worker.AttemptObservation(
        attempt_uid=observation.attempt_uid,
        state=int(observation.state),
        exit_code=observation.exit_code or 0,
        error=observation.error or "",
        container_id=observation.container_id or "",
    )
    if value.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(value.finished_at))
    if value.resource_usage is not None:
        result.resource_usage.CopyFrom(resource_usage_to_proto(value.resource_usage))
    return result


def attempt_observation_from_proto(value: worker_pb2.Worker.AttemptObservation) -> AttemptObservation:
    """Decode the reconciliation fields consumed by execution backends."""
    return AttemptObservation(
        attempt_uid=AttemptUid(value.attempt_uid),
        state=TaskState(value.state),
        exit_code=value.exit_code,
        error=value.error or None,
        container_id=value.container_id or None,
    )


def reconcile_response_to_proto(value: WorkerReconcileResponse) -> worker_pb2.Worker.ReconcileResponse:
    health = worker_pb2.Worker.WorkerHealth(
        healthy=value.health.healthy,
        health_error=value.health.health_error,
    )
    if value.health.resources is not None:
        health.resources.CopyFrom(worker_resource_snapshot_to_proto(value.health.resources))
    return worker_pb2.Worker.ReconcileResponse(
        worker_id=value.worker_id,
        observed=[_attempt_status_to_proto(item) for item in value.observed],
        health=health,
    )
