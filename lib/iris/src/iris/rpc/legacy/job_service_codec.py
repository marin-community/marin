# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapters from resource records to the retired ControllerService wire shape."""

from rigging.redaction import redact_value

from iris.cluster.types import LOCAL_CLUSTER, TERMINAL_TASK_STATES
from iris.resources.job import (
    ContainerProfile,
    CoschedulingConfig,
    ExistingJobPolicy,
    JobPreemptionPolicy,
    JobSpec,
    JobSummary,
    JobTaskAggregate,
    PriorityBand,
)
from iris.resources.task import TaskDetail
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.legacy.job_codec import (
    constraint_from_proto,
    constraint_to_proto,
    environment_from_proto,
    environment_to_proto,
    resource_spec_from_proto,
    resource_spec_to_proto,
    runtime_entrypoint_from_proto,
    runtime_entrypoint_to_proto,
)
from iris.rpc.proto_display import task_state_friendly
from iris.time_proto import duration_from_proto, duration_to_proto, timestamp_to_proto


def redact_request_env_vars(
    request: controller_pb2.Controller.LaunchJobRequest,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Copy an old-wire request and redact sensitive environment values."""
    if not request.environment.env_vars:
        return request
    redacted = controller_pb2.Controller.LaunchJobRequest()
    redacted.CopyFrom(request)
    env_vars = redacted.environment.env_vars
    values = redact_value(dict(env_vars))
    env_vars.clear()
    env_vars.update(values)
    return redacted


def job_spec_from_legacy_request(request: controller_pb2.Controller.LaunchJobRequest) -> JobSpec:
    """Decode the retired LaunchJob wire request into the resource contract."""
    resources = resource_spec_from_proto(request.resources)
    coscheduling = (
        CoschedulingConfig(group_by=request.coscheduling.group_by) if request.HasField("coscheduling") else None
    )
    return JobSpec(
        version=1,
        name=request.name,
        entrypoint=runtime_entrypoint_from_proto(request.entrypoint),
        resources=resources,
        environment=environment_from_proto(request.environment),
        bundle_id=request.bundle_id,
        scheduling_timeout=(
            duration_from_proto(request.scheduling_timeout) if request.HasField("scheduling_timeout") else None
        ),
        ports=tuple(request.ports),
        max_task_failures=request.max_task_failures,
        max_retries_failure=request.max_retries_failure,
        max_retries_preemption=request.max_retries_preemption,
        constraints=tuple(constraint_from_proto(candidate) for candidate in request.constraints),
        coscheduling=coscheduling,
        replicas=request.replicas,
        timeout=duration_from_proto(request.timeout) if request.HasField("timeout") else None,
        fail_if_exists=request.fail_if_exists,
        preemption_policy=JobPreemptionPolicy(request.preemption_policy),
        existing_job_policy=ExistingJobPolicy(request.existing_job_policy),
        priority_band=PriorityBand(request.priority_band),
        task_image=request.task_image,
        submit_argv=tuple(request.submit_argv),
        client_revision_date=request.client_revision_date,
        container_profile=ContainerProfile(request.container_profile),
    )


def job_spec_to_legacy_request(spec: JobSpec) -> controller_pb2.Controller.LaunchJobRequest:
    request = controller_pb2.Controller.LaunchJobRequest(
        name=spec.name,
        entrypoint=runtime_entrypoint_to_proto(spec.entrypoint),
        resources=resource_spec_to_proto(spec.resources),
        environment=environment_to_proto(spec.environment),
        bundle_id=spec.bundle_id,
        ports=spec.ports,
        max_task_failures=spec.max_task_failures,
        max_retries_failure=spec.max_retries_failure,
        max_retries_preemption=spec.max_retries_preemption,
        constraints=[constraint_to_proto(constraint) for constraint in spec.constraints],
        replicas=spec.replicas,
        fail_if_exists=spec.fail_if_exists,
        preemption_policy=spec.preemption_policy,
        existing_job_policy=spec.existing_job_policy,
        priority_band=spec.priority_band,
        task_image=spec.task_image,
        submit_argv=spec.submit_argv,
        client_revision_date=spec.client_revision_date,
        container_profile=spec.container_profile,
    )
    if spec.scheduling_timeout is not None:
        request.scheduling_timeout.CopyFrom(duration_to_proto(spec.scheduling_timeout))
    if spec.coscheduling is not None:
        request.coscheduling.group_by = spec.coscheduling.group_by
    if spec.timeout is not None:
        request.timeout.CopyFrom(duration_to_proto(spec.timeout))
    return request


def job_status_to_legacy(
    summary: JobSummary,
    tasks: JobTaskAggregate,
    *,
    has_children: bool,
    local_cluster_id: str,
) -> job_pb2.JobStatus:
    state_counts = {entry.state: entry.count for entry in tasks.state_counts}
    status = job_pb2.JobStatus(
        job_id=summary.identity.key.resource_id,
        state=summary.state,
        error=summary.error_message,
        exit_code=summary.exit_code or 0,
        name=summary.identity.key.resource_id.lower(),
        has_children=has_children,
        parent_job_id=summary.parent.key.resource_id if summary.parent is not None else "",
        backend_id=summary.backend_id,
        cluster=(LOCAL_CLUSTER if summary.execution_cluster_id == local_cluster_id else summary.execution_cluster_id),
        pending_reason=summary.pending_reason,
        resources=resource_spec_to_proto(summary.resources),
        task_count=tasks.task_count if tasks.task_count else summary.num_tasks,
        completed_count=tasks.completed_count,
        failure_count=tasks.failure_count,
        preemption_count=tasks.preemption_count,
        task_state_counts={task_state_friendly(state): count for state, count in state_counts.items()},
    )
    status.submitted_at.CopyFrom(timestamp_to_proto(summary.submitted_at))
    if summary.started_at is not None:
        status.started_at.CopyFrom(timestamp_to_proto(summary.started_at))
    if summary.finished_at is not None:
        status.finished_at.CopyFrom(timestamp_to_proto(summary.finished_at))
    return status


def task_detail_to_legacy(
    detail: TaskDetail,
    *,
    local_cluster_id: str,
) -> job_pb2.TaskStatus:
    summary = detail.summary
    attempts = []
    for attempt in detail.attempts:
        item = job_pb2.TaskAttempt(
            attempt_id=attempt.identity.attempt_number,
            worker_id=attempt.node.key.resource_id if attempt.node is not None else "",
            state=attempt.state,
            exit_code=attempt.exit_code or 0,
            error=attempt.error_message,
            is_worker_failure=attempt.state
            in (
                job_pb2.TASK_STATE_WORKER_FAILED,
                job_pb2.TASK_STATE_PREEMPTED,
            ),
            attempt_uid=attempt.identity.attempt_uid,
            node_name=attempt.node.key.resource_id if attempt.node is not None else "",
            terminal_reason=attempt.terminal_reason,
        )
        if attempt.started_at is not None:
            item.started_at.CopyFrom(timestamp_to_proto(attempt.started_at))
        if attempt.finished_at is not None:
            item.finished_at.CopyFrom(timestamp_to_proto(attempt.finished_at))
        attempts.append(item)
    current = next(
        (
            attempt
            for attempt in detail.attempts
            if summary.current_attempt is not None and attempt.identity == summary.current_attempt
        ),
        None,
    )
    current_worker_label = summary.current_node.key.resource_id if summary.current_node is not None else ""
    result = job_pb2.TaskStatus(
        task_id=summary.identity.key.resource_id,
        state=summary.state,
        worker_id=current_worker_label,
        exit_code=current.exit_code if current is not None and current.exit_code is not None else 0,
        error=summary.error_message,
        current_attempt_id=summary.current_attempt.attempt_number if summary.current_attempt is not None else 0,
        attempts=attempts,
        backend_id=summary.backend_id,
        cluster=(LOCAL_CLUSTER if summary.execution_cluster_id == local_cluster_id else summary.execution_cluster_id),
        status_message=summary.status_message,
        can_be_scheduled=summary.state == job_pb2.TASK_STATE_PENDING,
    )
    result.submitted_at.CopyFrom(timestamp_to_proto(summary.submitted_at))
    if summary.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(summary.started_at))
    if summary.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(summary.finished_at))
    if summary.state == job_pb2.TASK_STATE_PENDING and detail.attempts:
        previous = detail.attempts[-1]
        if previous.state in TERMINAL_TASK_STATES:
            result.pending_reason = (
                f"Retrying (attempt {summary.current_attempt.attempt_number + 1 if summary.current_attempt else 1}, "
                f"last: {task_state_friendly(previous.state)})"
            )
    return result
