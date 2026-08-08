# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect RPC boundary for typed Iris resources."""

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.server_auth import ANONYMOUS_ADMIN, get_verified_identity

from iris.cluster.constraints import Constraint
from iris.cluster.controller.resources.facade import ResourceController
from iris.cluster.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.cluster.resources.activity import ActivityQuery
from iris.cluster.resources.endpoint import EndpointAccess, EndpointQuery
from iris.cluster.resources.errors import (
    ActionIdempotencyConflict,
    ActionPolicyRejected,
    InvalidPageToken,
    InvalidResourceKey,
    ResourceNotFound,
    ResourceReplaced,
)
from iris.cluster.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    NodeLocator,
    ResourceKey,
    ResourceKind,
    SliceLocator,
    TaskIdentity,
)
from iris.cluster.resources.job import JobQuery, JobSpec
from iris.cluster.resources.log import LogQuery
from iris.cluster.resources.node import NodeAttributeKind, NodeHealth, NodeQuery
from iris.cluster.resources.slice import MembershipState, SliceLifecycle, SliceQuery
from iris.cluster.resources.source import Freshness, ResourceSourceStatus, SourceState
from iris.cluster.resources.task import TaskQuery
from iris.cluster.types import CoschedulingConfig, JobName, ResourceSpec
from iris.rpc import controller_pb2, resource_pb2
from iris.rpc.auth import DASHBOARD_ROLE, authorize_resource_owner
from iris.time_proto import duration_from_proto, duration_to_proto, timestamp_from_proto, timestamp_to_proto

_RESOURCE_KIND_FROM_PROTO = {
    resource_pb2.RESOURCE_KIND_JOB: ResourceKind.JOB,
    resource_pb2.RESOURCE_KIND_TASK: ResourceKind.TASK,
    resource_pb2.RESOURCE_KIND_ATTEMPT: ResourceKind.ATTEMPT,
    resource_pb2.RESOURCE_KIND_ENDPOINT: ResourceKind.ENDPOINT,
    resource_pb2.RESOURCE_KIND_NODE: ResourceKind.NODE,
    resource_pb2.RESOURCE_KIND_SLICE: ResourceKind.SLICE,
}
_RESOURCE_KIND_TO_PROTO = {value: key for key, value in _RESOURCE_KIND_FROM_PROTO.items()}
_SOURCE_STATE_TO_PROTO = {
    SourceState.AVAILABLE: resource_pb2.SOURCE_STATE_AVAILABLE,
    SourceState.UNAVAILABLE: resource_pb2.SOURCE_STATE_UNAVAILABLE,
    SourceState.UNSUPPORTED: resource_pb2.SOURCE_STATE_UNSUPPORTED,
}
_FRESHNESS_TO_PROTO = {
    Freshness.CURRENT: resource_pb2.FRESHNESS_CURRENT,
    Freshness.STALE: resource_pb2.FRESHNESS_STALE,
    Freshness.UNKNOWN: resource_pb2.FRESHNESS_UNKNOWN,
}
_ACTION_KIND_TO_PROTO = {
    ActionKind.CANCEL_JOB: resource_pb2.ACTION_KIND_CANCEL_JOB,
    ActionKind.RETRY_TASK: resource_pb2.ACTION_KIND_RETRY_TASK,
    ActionKind.TERMINATE_ATTEMPT: resource_pb2.ACTION_KIND_TERMINATE_ATTEMPT,
}
_ACTION_STATE_TO_PROTO = {
    ActionState.ACCEPTED: resource_pb2.ACTION_STATE_ACCEPTED,
    ActionState.VERIFYING: resource_pb2.ACTION_STATE_VERIFYING,
    ActionState.SUCCEEDED: resource_pb2.ACTION_STATE_SUCCEEDED,
    ActionState.FAILED: resource_pb2.ACTION_STATE_FAILED,
}
_ACTION_RESULT_TO_PROTO = {
    ActionResult.NONE: resource_pb2.ACTION_RESULT_NONE,
    ActionResult.SATISFIED: resource_pb2.ACTION_RESULT_SATISFIED,
    ActionResult.TARGET_ABSENT: resource_pb2.ACTION_RESULT_TARGET_ABSENT,
    ActionResult.PROVIDER_REJECTED: resource_pb2.ACTION_RESULT_PROVIDER_REJECTED,
    ActionResult.INTERNAL_ERROR: resource_pb2.ACTION_RESULT_INTERNAL_ERROR,
}
_ENDPOINT_ACCESS_TO_PROTO = {
    EndpointAccess.PRIVATE: resource_pb2.ENDPOINT_ACCESS_PRIVATE,
    EndpointAccess.LINK: resource_pb2.ENDPOINT_ACCESS_LINK,
}
_NODE_HEALTH_FROM_PROTO = {
    resource_pb2.NODE_HEALTH_READY: NodeHealth.READY,
    resource_pb2.NODE_HEALTH_DEGRADED: NodeHealth.DEGRADED,
    resource_pb2.NODE_HEALTH_UNAVAILABLE: NodeHealth.UNAVAILABLE,
    resource_pb2.NODE_HEALTH_RETIRED: NodeHealth.RETIRED,
}
_NODE_HEALTH_TO_PROTO = {value: key for key, value in _NODE_HEALTH_FROM_PROTO.items()}
_SLICE_LIFECYCLE_TO_PROTO = {
    SliceLifecycle.CREATING: resource_pb2.SLICE_LIFECYCLE_CREATING,
    SliceLifecycle.READY: resource_pb2.SLICE_LIFECYCLE_READY,
    SliceLifecycle.DELETING: resource_pb2.SLICE_LIFECYCLE_DELETING,
    SliceLifecycle.FAILED: resource_pb2.SLICE_LIFECYCLE_FAILED,
}
_MEMBERSHIP_STATE_TO_PROTO = {
    MembershipState.UNKNOWN: resource_pb2.MEMBERSHIP_STATE_UNKNOWN,
    MembershipState.OBSERVED: resource_pb2.MEMBERSHIP_STATE_OBSERVED,
}


def _resource_key_from_proto(value: resource_pb2.ResourceKey) -> ResourceKey:
    try:
        kind = _RESOURCE_KIND_FROM_PROTO[value.kind]
    except KeyError as exc:
        raise InvalidResourceKey("resource kind is required") from exc
    return ResourceKey(value.cluster_id, kind, value.resource_id)


def _resource_key_to_proto(value: ResourceKey) -> resource_pb2.ResourceKey:
    return resource_pb2.ResourceKey(
        cluster_id=value.cluster_id,
        kind=_RESOURCE_KIND_TO_PROTO[value.kind],
        resource_id=value.resource_id,
    )


def _job_identity_to_proto(value: JobIdentity) -> resource_pb2.JobIdentity:
    return resource_pb2.JobIdentity(key=_resource_key_to_proto(value.key), job_uid=value.job_uid)


def _task_identity_to_proto(value) -> resource_pb2.TaskIdentity:
    return resource_pb2.TaskIdentity(key=_resource_key_to_proto(value.key), task_uid=value.task_uid)


def _attempt_identity_to_proto(value) -> resource_pb2.AttemptIdentity:
    return resource_pb2.AttemptIdentity(
        task=_resource_key_to_proto(value.task),
        attempt_number=value.attempt_number,
        attempt_uid=value.attempt_uid,
    )


def _job_identity_from_proto(value: resource_pb2.JobIdentity) -> JobIdentity:
    return JobIdentity(_resource_key_from_proto(value.key), value.job_uid)


def _task_identity_from_proto(value: resource_pb2.TaskIdentity) -> TaskIdentity:
    return TaskIdentity(_resource_key_from_proto(value.key), value.task_uid)


def _attempt_identity_from_proto(value: resource_pb2.AttemptIdentity) -> AttemptIdentity:
    return AttemptIdentity(
        _resource_key_from_proto(value.task),
        value.attempt_number,
        value.attempt_uid,
    )


def _node_identity_to_proto(value) -> resource_pb2.NodeIdentity:
    return resource_pb2.NodeIdentity(
        key=_resource_key_to_proto(value.key),
        backend_id=value.backend_id,
        node_uid=value.node_uid,
    )


def _slice_identity_to_proto(value) -> resource_pb2.SliceIdentity:
    return resource_pb2.SliceIdentity(
        key=_resource_key_to_proto(value.key),
        backend_id=value.backend_id,
        slice_uid=value.slice_uid,
    )


def _node_summary_to_proto(value) -> resource_pb2.NodeSummary:
    result = resource_pb2.NodeSummary(
        identity=_node_identity_to_proto(value.identity),
        health=_NODE_HEALTH_TO_PROTO[value.health],
        schedulable=value.schedulable,
        capacity=resource_pb2.NodeCapacity(
            cpu_millicores=value.capacity.cpu_millicores,
            memory_bytes=value.capacity.memory_bytes,
            disk_bytes=value.capacity.disk_bytes,
            accelerator_kind=value.capacity.accelerator_kind,
            accelerator_variant=value.capacity.accelerator_variant,
            accelerator_count=value.capacity.accelerator_count,
        ),
        scaling_group_id=value.scaling_group_id or "",
        running_task_count=value.running_task_count,
        observed_at=timestamp_to_proto(value.observed_at),
    )
    if value.slice is not None:
        result.slice.CopyFrom(_slice_identity_to_proto(value.slice))
    return result


def _node_attribute_to_proto(value) -> resource_pb2.NodeAttribute:
    result = resource_pb2.NodeAttribute(key=value.key)
    if value.kind is NodeAttributeKind.STRING:
        result.string_value = value.string_value or ""
    elif value.kind is NodeAttributeKind.INTEGER:
        result.integer_value = value.integer_value or 0
    else:
        result.float_value = value.float_value or 0.0
    return result


def _slice_summary_to_proto(value) -> resource_pb2.SliceSummary:
    result = resource_pb2.SliceSummary(
        identity=_slice_identity_to_proto(value.identity),
        scaling_group_id=value.scaling_group_id,
        lifecycle=_SLICE_LIFECYCLE_TO_PROTO[value.lifecycle],
        membership_state=_MEMBERSHIP_STATE_TO_PROTO[value.membership_state],
        observed_member_count=value.observed_member_count,
        error_message=value.error_message,
    )
    if value.observed_at is not None:
        result.observed_at.CopyFrom(timestamp_to_proto(value.observed_at))
    return result


def _slice_member_to_proto(value) -> resource_pb2.SliceMember:
    result = resource_pb2.SliceMember(
        provider_node_id=value.provider_node_id,
        observed_at=timestamp_to_proto(value.observed_at),
    )
    if value.node is not None:
        result.node.CopyFrom(_node_identity_to_proto(value.node))
    return result


def _source_status_to_proto(value: ResourceSourceStatus) -> resource_pb2.ResourceSourceStatus:
    result = resource_pb2.ResourceSourceStatus(
        source_id=value.source_id,
        backend_id=value.backend_id,
        state=_SOURCE_STATE_TO_PROTO[value.state],
        freshness=_FRESHNESS_TO_PROTO[value.freshness],
        error_code=value.error_code,
        error_message=value.error_message,
    )
    if value.observed_at is not None:
        result.observed_at.CopyFrom(timestamp_to_proto(value.observed_at))
    return result


def _page_info(next_page_token: str | None, statuses: tuple[ResourceSourceStatus, ...]) -> resource_pb2.PageInfo:
    return resource_pb2.PageInfo(
        next_page_token=next_page_token or "",
        source_statuses=[_source_status_to_proto(status) for status in statuses],
    )


def _job_summary_to_proto(value) -> resource_pb2.JobSummary:
    result = resource_pb2.JobSummary(
        identity=_job_identity_to_proto(value.identity),
        owner_id=value.owner_id,
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        num_tasks=value.num_tasks,
        submitted_at=timestamp_to_proto(value.submitted_at),
        error_message=value.error_message,
    )
    if value.parent is not None:
        result.parent.CopyFrom(_job_identity_to_proto(value.parent))
    if value.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(value.started_at))
    if value.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(value.finished_at))
    return result


def _task_summary_to_proto(value) -> resource_pb2.TaskSummary:
    result = resource_pb2.TaskSummary(
        identity=_task_identity_to_proto(value.identity),
        job=_job_identity_to_proto(value.job),
        task_index=value.task_index,
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        failure_count=value.failure_count,
        preemption_count=value.preemption_count,
        submitted_at=timestamp_to_proto(value.submitted_at),
        status_message=value.status_message,
        error_message=value.error_message,
    )
    if value.current_attempt is not None:
        result.current_attempt.CopyFrom(_attempt_identity_to_proto(value.current_attempt))
    if value.current_node is not None:
        result.current_node.CopyFrom(_node_identity_to_proto(value.current_node))
    if value.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(value.started_at))
    if value.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(value.finished_at))
    return result


def _attempt_summary_to_proto(value) -> resource_pb2.AttemptSummary:
    result = resource_pb2.AttemptSummary(
        identity=_attempt_identity_to_proto(value.identity),
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        created_at=timestamp_to_proto(value.created_at),
        error_message=value.error_message,
        terminal_reason=value.terminal_reason,
    )
    if value.node is not None:
        result.node.CopyFrom(_node_identity_to_proto(value.node))
    if value.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(value.started_at))
    if value.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(value.finished_at))
    if value.exit_code is not None:
        result.exit_code = value.exit_code
    return result


def _action_receipt_to_proto(value: ActionReceipt) -> resource_pb2.ActionReceipt:
    result = resource_pb2.ActionReceipt(
        action_id=value.action_id,
        kind=_ACTION_KIND_TO_PROTO[value.kind],
        target=_resource_key_to_proto(value.target),
        expected_target_uid=value.expected_target_uid,
        expected_attempt_uid=value.expected_attempt_uid or "",
        state=_ACTION_STATE_TO_PROTO[value.state],
        result_code=_ACTION_RESULT_TO_PROTO[value.result_code],
        result_message=value.result_message,
        created_at=timestamp_to_proto(value.created_at),
        updated_at=timestamp_to_proto(value.updated_at),
    )
    if value.completed_at is not None:
        result.completed_at.CopyFrom(timestamp_to_proto(value.completed_at))
    return result


def _activity_entry_to_proto(value) -> resource_pb2.ActivityEntry:
    return resource_pb2.ActivityEntry(
        entry_id=value.entry_id,
        occurred_at=timestamp_to_proto(value.occurred_at),
        source=value.source,
        severity=value.severity,
        kind=value.kind,
        message=value.message,
        target=_resource_key_to_proto(value.target),
        attempt_uid=value.attempt_uid or "",
        correlation_id=value.correlation_id or "",
        attributes=value.attributes,
    )


def _endpoint_summary_to_proto(value) -> resource_pb2.EndpointSummary:
    result = resource_pb2.EndpointSummary(
        key=_resource_key_to_proto(value.key),
        endpoint_id=value.endpoint_id,
        name=value.name,
        execution_cluster_id=value.execution_cluster_id,
        access=_ENDPOINT_ACCESS_TO_PROTO[value.access],
    )
    if value.task is not None:
        result.task.CopyFrom(_resource_key_to_proto(value.task))
    if value.lease_deadline is not None:
        result.lease_deadline.CopyFrom(timestamp_to_proto(value.lease_deadline))
    return result


def _action_principal(resource_id: str) -> str:
    owner = JobName.from_wire(resource_id.rpartition(":")[0] or resource_id).user
    identity = get_verified_identity()
    if identity is None:
        return ANONYMOUS_ADMIN.user_id
    return authorize_resource_owner(owner).user_id


def _authorized_owner(requested_owner: str | None = None) -> str | None:
    identity = get_verified_identity()
    if identity is None or identity.role in {"admin", DASHBOARD_ROLE}:
        return requested_owner
    if requested_owner is not None and requested_owner != identity.user_id:
        authorize_resource_owner(requested_owner)
    return identity.user_id


def _authorize_key_owner(key: ResourceKey) -> None:
    identity = get_verified_identity()
    if identity is None or identity.role in {"admin", DASHBOARD_ROLE}:
        return
    owner = JobName.from_wire(key.resource_id.rpartition(":")[0] or key.resource_id).user
    authorize_resource_owner(owner)


def _job_spec_from_proto(value: resource_pb2.JobSpec) -> JobSpec:
    return JobSpec(
        version=value.version,
        name=value.name,
        entrypoint=value.entrypoint,
        resources=ResourceSpec(
            cpu=value.resources.cpu_millicores / 1_000,
            memory=value.resources.memory_bytes,
            disk=value.resources.disk_bytes,
            device=value.resources.device if value.resources.HasField("device") else None,
        ),
        environment=value.environment,
        bundle_id=value.bundle_id,
        scheduling_timeout=(
            duration_from_proto(value.scheduling_timeout) if value.HasField("scheduling_timeout") else None
        ),
        ports=tuple(value.ports),
        max_task_failures=value.max_task_failures,
        max_retries_failure=value.max_retries_failure,
        max_retries_preemption=value.max_retries_preemption,
        constraints=tuple(Constraint.from_proto(candidate) for candidate in value.constraints),
        coscheduling=(
            CoschedulingConfig(group_by=value.coscheduling.group_by) if value.HasField("coscheduling") else None
        ),
        replicas=value.replicas,
        timeout=duration_from_proto(value.timeout) if value.HasField("timeout") else None,
        fail_if_exists=value.fail_if_exists,
        preemption_policy=value.preemption_policy,
        existing_job_policy=value.existing_job_policy,
        priority_band=value.priority_band,
        task_image=value.task_image,
        submit_argv=tuple(value.submit_argv),
        client_revision_date=value.client_revision_date,
        container_profile=value.container_profile,
    )


def _job_spec_to_proto(value: JobSpec) -> resource_pb2.JobSpec:
    launch = controller_pb2.Controller.LaunchJobRequest(
        name=value.name,
        entrypoint=value.entrypoint,
        resources=value.resources.to_proto(),
        environment=value.environment,
        bundle_id=value.bundle_id,
        ports=value.ports,
        max_task_failures=value.max_task_failures,
        max_retries_failure=value.max_retries_failure,
        max_retries_preemption=value.max_retries_preemption,
        constraints=[constraint.to_proto() for constraint in value.constraints],
        replicas=value.replicas,
        fail_if_exists=value.fail_if_exists,
        preemption_policy=value.preemption_policy,
        existing_job_policy=value.existing_job_policy,
        priority_band=value.priority_band,
        task_image=value.task_image,
        submit_argv=value.submit_argv,
        client_revision_date=value.client_revision_date,
        container_profile=value.container_profile,
    )
    if value.scheduling_timeout is not None:
        launch.scheduling_timeout.CopyFrom(duration_to_proto(value.scheduling_timeout))
    if value.coscheduling is not None:
        launch.coscheduling.CopyFrom(value.coscheduling.to_proto())
    if value.timeout is not None:
        launch.timeout.CopyFrom(duration_to_proto(value.timeout))
    result = resource_pb2.JobSpec()
    result.version = value.version
    result.name = launch.name
    result.entrypoint.CopyFrom(launch.entrypoint)
    result.resources.CopyFrom(launch.resources)
    result.environment.CopyFrom(launch.environment)
    result.bundle_id = launch.bundle_id
    if launch.HasField("scheduling_timeout"):
        result.scheduling_timeout.CopyFrom(launch.scheduling_timeout)
    result.ports.extend(launch.ports)
    result.max_task_failures = launch.max_task_failures
    result.max_retries_failure = launch.max_retries_failure
    result.max_retries_preemption = launch.max_retries_preemption
    result.constraints.extend(launch.constraints)
    if launch.HasField("coscheduling"):
        result.coscheduling.CopyFrom(launch.coscheduling)
    result.replicas = launch.replicas
    if launch.HasField("timeout"):
        result.timeout.CopyFrom(launch.timeout)
    result.fail_if_exists = launch.fail_if_exists
    result.preemption_policy = launch.preemption_policy
    result.existing_job_policy = launch.existing_job_policy
    result.priority_band = launch.priority_band
    result.task_image = launch.task_image
    result.submit_argv.extend(launch.submit_argv)
    result.client_revision_date = launch.client_revision_date
    result.container_profile = launch.container_profile
    return result


class ResourceServiceImpl:
    """Serve public resource reads through the typed controller boundary."""

    def __init__(self, resources: ResourceController) -> None:
        self._resources = resources

    def submit_job(self, request: resource_pb2.SubmitJobRequest, _ctx: RequestContext) -> resource_pb2.SubmitJobResponse:
        try:
            identity = self._resources.submit_job(_job_spec_from_proto(request.spec), request.bundle_blob)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return resource_pb2.SubmitJobResponse(job=_job_identity_to_proto(identity))

    def list_jobs(self, request: resource_pb2.ListJobsRequest, _ctx: RequestContext) -> resource_pb2.ListJobsResponse:
        query = request.query
        try:
            owner_id = _authorized_owner(query.owner_id or None)
            page = self._resources.list_jobs(
                JobQuery(
                    owner_id=owner_id,
                    parent=_resource_key_from_proto(query.parent) if query.HasField("parent") else None,
                    job_id_prefix=query.job_id_prefix or None,
                    states=frozenset(query.states),
                    backend_id=query.backend_id or None,
                    execution_cluster_id=query.execution_cluster_id or None,
                    page_size=query.page.page_size or 50,
                    page_token=query.page.page_token or None,
                )
            )
        except (InvalidPageToken, InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return resource_pb2.ListJobsResponse(
            jobs=[_job_summary_to_proto(item) for item in page.items],
            page=_page_info(page.next_page_token, page.source_statuses),
        )

    def describe_job(
        self, request: resource_pb2.DescribeJobRequest, _ctx: RequestContext
    ) -> resource_pb2.DescribeJobResponse:
        try:
            key = _resource_key_from_proto(request.job)
            _authorize_key_owner(key)
            detail = self._resources.describe_job(key)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return resource_pb2.DescribeJobResponse(
            job=resource_pb2.JobDetail(
                summary=_job_summary_to_proto(detail.summary), spec=_job_spec_to_proto(detail.spec)
            )
        )

    def list_tasks(self, request: resource_pb2.ListTasksRequest, _ctx: RequestContext) -> resource_pb2.ListTasksResponse:
        query = request.query
        try:
            job = _resource_key_from_proto(query.job) if query.HasField("job") else None
            if job is not None:
                _authorize_key_owner(job)
            owner_id = _authorized_owner()
            job_id_prefix = query.job_id_prefix or None
            if owner_id is not None:
                owner_prefix = f"/{owner_id}/"
                if job_id_prefix is not None and not job_id_prefix.startswith(owner_prefix):
                    raise ConnectError(
                        Code.PERMISSION_DENIED,
                        f"User {owner_id!r} cannot list Tasks outside {owner_prefix!r}",
                    )
                job_id_prefix = job_id_prefix or owner_prefix
            page = self._resources.list_tasks(
                TaskQuery(
                    job=job,
                    job_id_prefix=job_id_prefix,
                    states=frozenset(query.states),
                    backend_id=query.backend_id or None,
                    authority_cluster_id=query.authority_cluster_id or None,
                    execution_cluster_id=query.execution_cluster_id or None,
                    page_size=query.page.page_size or 100,
                    page_token=query.page.page_token or None,
                )
            )
        except (InvalidPageToken, InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return resource_pb2.ListTasksResponse(
            tasks=[_task_summary_to_proto(item) for item in page.items],
            page=_page_info(page.next_page_token, page.source_statuses),
        )

    def describe_task(
        self, request: resource_pb2.DescribeTaskRequest, _ctx: RequestContext
    ) -> resource_pb2.DescribeTaskResponse:
        try:
            key = _resource_key_from_proto(request.task)
            _authorize_key_owner(key)
            detail = self._resources.describe_task(key)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return resource_pb2.DescribeTaskResponse(
            task=resource_pb2.TaskDetail(
                summary=_task_summary_to_proto(detail.summary),
                attempts=[_attempt_summary_to_proto(item) for item in detail.attempts],
                source_statuses=[_source_status_to_proto(status) for status in detail.source_statuses],
                root_cause_highlights=detail.root_cause_highlights,
            )
        )

    def describe_attempt(
        self, request: resource_pb2.DescribeAttemptRequest, _ctx: RequestContext
    ) -> resource_pb2.DescribeAttemptResponse:
        try:
            locator = AttemptLocator(
                _resource_key_from_proto(request.attempt.task),
                request.attempt.attempt_number if request.attempt.HasField("attempt_number") else None,
            )
            _authorize_key_owner(locator.task)
            detail = self._resources.describe_attempt(locator)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        result = resource_pb2.AttemptDetail(
            summary=_attempt_summary_to_proto(detail.summary),
            source_statuses=[_source_status_to_proto(status) for status in detail.source_statuses],
        )
        if detail.runtime is not None:
            result.runtime.CopyFrom(
                resource_pb2.AttemptRuntimeObject(
                    provider_kind=detail.runtime.provider_kind,
                    namespace=detail.runtime.namespace,
                    name=detail.runtime.name,
                    provider_uid=detail.runtime.provider_uid,
                    provider_node_id=detail.runtime.provider_node_id,
                    provider_node_uid=detail.runtime.provider_node_uid,
                    container_id=detail.runtime.container_id,
                    observed_at=timestamp_to_proto(detail.runtime.observed_at),
                )
            )
        return resource_pb2.DescribeAttemptResponse(attempt=result)

    def list_nodes(self, request: resource_pb2.ListNodesRequest, _ctx: RequestContext) -> resource_pb2.ListNodesResponse:
        query = request.query
        try:
            health = frozenset(_NODE_HEALTH_FROM_PROTO[value] for value in query.health)
            page = self._resources.list_nodes(
                NodeQuery(
                    backend_id=query.backend_id or None,
                    contains=query.contains or None,
                    health=health,
                    page_size=query.page.page_size or 100,
                    page_token=query.page.page_token or None,
                )
            )
        except (InvalidPageToken, KeyError, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return resource_pb2.ListNodesResponse(
            nodes=[_node_summary_to_proto(item) for item in page.items],
            page=_page_info(page.next_page_token, page.source_statuses),
        )

    def describe_node(
        self, request: resource_pb2.DescribeNodeRequest, _ctx: RequestContext
    ) -> resource_pb2.DescribeNodeResponse:
        try:
            detail = self._resources.describe_node(
                NodeLocator(
                    key=_resource_key_from_proto(request.node.key),
                    backend_id=request.node.backend_id,
                    node_uid=request.node.node_uid or None,
                )
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ActionPolicyRejected as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        result = resource_pb2.NodeDetail(
            summary=_node_summary_to_proto(detail.summary),
            address=detail.address or "",
            attributes=[_node_attribute_to_proto(item) for item in detail.attributes],
            recent_attempts=[_attempt_summary_to_proto(item) for item in detail.recent_attempts],
            bootstrap_log_key=detail.bootstrap_log_key or "",
            source_statuses=[_source_status_to_proto(status) for status in detail.source_statuses],
        )
        return resource_pb2.DescribeNodeResponse(node=result)

    def list_slices(
        self, request: resource_pb2.ListSlicesRequest, _ctx: RequestContext
    ) -> resource_pb2.ListSlicesResponse:
        query = request.query
        try:
            page = self._resources.list_slices(
                SliceQuery(
                    backend_id=query.backend_id or None,
                    scaling_group_id=query.scaling_group_id or None,
                    page_size=query.page.page_size or 100,
                    page_token=query.page.page_token or None,
                )
            )
        except (InvalidPageToken, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return resource_pb2.ListSlicesResponse(
            slices=[_slice_summary_to_proto(item) for item in page.items],
            page=_page_info(page.next_page_token, page.source_statuses),
        )

    def describe_slice(
        self, request: resource_pb2.DescribeSliceRequest, _ctx: RequestContext
    ) -> resource_pb2.DescribeSliceResponse:
        try:
            detail = self._resources.describe_slice(
                SliceLocator(
                    key=_resource_key_from_proto(request.slice.key),
                    backend_id=request.slice.backend_id,
                    slice_uid=request.slice.slice_uid or None,
                )
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ActionPolicyRejected as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        return resource_pb2.DescribeSliceResponse(
            slice=resource_pb2.SliceDetail(
                summary=_slice_summary_to_proto(detail.summary),
                members=[_slice_member_to_proto(item) for item in detail.members],
                source_statuses=[_source_status_to_proto(status) for status in detail.source_statuses],
            )
        )

    def list_endpoints(
        self, request: resource_pb2.ListEndpointsRequest, _ctx: RequestContext
    ) -> resource_pb2.ListEndpointsResponse:
        query = request.query
        try:
            page = self._resources.list_endpoints(
                EndpointQuery(
                    name_prefix=query.name_prefix or None,
                    task=_resource_key_from_proto(query.task) if query.HasField("task") else None,
                    page_size=query.page.page_size or 100,
                    page_token=query.page.page_token or None,
                )
            )
        except (InvalidPageToken, InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return resource_pb2.ListEndpointsResponse(
            endpoints=[_endpoint_summary_to_proto(item) for item in page.items],
            page=_page_info(page.next_page_token, page.source_statuses),
        )

    def describe_endpoint(
        self, request: resource_pb2.DescribeEndpointRequest, _ctx: RequestContext
    ) -> resource_pb2.DescribeEndpointResponse:
        try:
            detail = self._resources.describe_endpoint(_resource_key_from_proto(request.endpoint))
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return resource_pb2.DescribeEndpointResponse(
            endpoint=resource_pb2.EndpointDetail(
                summary=_endpoint_summary_to_proto(detail.summary),
                address=detail.address,
                metadata=detail.metadata,
            )
        )

    def mint_endpoint_token(
        self, request: resource_pb2.MintEndpointTokenRequest, _ctx: RequestContext
    ) -> resource_pb2.MintEndpointTokenResponse:
        try:
            token = self._resources.mint_endpoint_token(
                _resource_key_from_proto(request.endpoint),
                duration_from_proto(request.ttl) if request.HasField("ttl") else None,
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return resource_pb2.MintEndpointTokenResponse(
            token=token.token,
            expires_at=timestamp_to_proto(token.expires_at),
            capability_url=token.capability_url,
        )

    def list_activity(
        self, request: resource_pb2.ListActivityRequest, _ctx: RequestContext
    ) -> resource_pb2.ListActivityResponse:
        try:
            target = _resource_key_from_proto(request.query.target)
            _action_principal(target.resource_id)
            page = self._resources.list_activity(
                ActivityQuery(
                    target=target,
                    attempt_uid=request.query.attempt_uid or None,
                    after=timestamp_from_proto(request.query.after) if request.query.HasField("after") else None,
                    page_size=request.query.page.page_size or 200,
                    page_token=request.query.page.page_token or None,
                )
            )
        except (InvalidPageToken, InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        return resource_pb2.ListActivityResponse(
            entries=[_activity_entry_to_proto(item) for item in page.items],
            page=_page_info(page.next_page_token, page.source_statuses),
        )

    def fetch_logs(self, request: resource_pb2.FetchLogsRequest, _ctx: RequestContext) -> resource_pb2.FetchLogsResponse:
        try:
            target_kind = request.target.WhichOneof("target")
            if target_kind == "job":
                target = _job_identity_from_proto(request.target.job)
            elif target_kind == "task":
                target = _task_identity_from_proto(request.target.task)
            elif target_kind == "attempt":
                target = _attempt_identity_from_proto(request.target.attempt)
            else:
                raise InvalidResourceKey("log target is required")
            resource_id = target.task.resource_id if isinstance(target, AttemptIdentity) else target.key.resource_id
            _action_principal(resource_id)
            page = self._resources.fetch_logs(
                target,
                LogQuery(
                    after=timestamp_from_proto(request.query.after) if request.query.HasField("after") else None,
                    cursor=request.query.cursor,
                    max_lines=request.query.max_lines or 1_000,
                    substring=request.query.substring,
                    minimum_level=request.query.minimum_level,
                    tail=request.query.tail,
                ),
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        return resource_pb2.FetchLogsResponse(
            entries=page.entries,
            next_cursor=page.next_cursor,
            source_statuses=[_source_status_to_proto(status) for status in page.source_statuses],
        )

    def cancel_job(self, request: resource_pb2.CancelJobRequest, _ctx: RequestContext) -> resource_pb2.ActionResponse:
        try:
            identity = _job_identity_from_proto(request.job)
            receipt = self._resources.cancel_job(
                identity,
                idempotency_key=request.idempotency_key,
                principal_id=_action_principal(identity.key.resource_id),
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ActionIdempotencyConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        return resource_pb2.ActionResponse(receipt=_action_receipt_to_proto(receipt))

    def retry_task(self, request: resource_pb2.RetryTaskRequest, _ctx: RequestContext) -> resource_pb2.ActionResponse:
        try:
            identity = _task_identity_from_proto(request.task)
            receipt = self._resources.retry_task(
                identity,
                expected_attempt_uid=request.expected_attempt_uid,
                idempotency_key=request.idempotency_key,
                principal_id=_action_principal(identity.key.resource_id),
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except (ResourceReplaced, ActionPolicyRejected) as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ActionIdempotencyConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        return resource_pb2.ActionResponse(receipt=_action_receipt_to_proto(receipt))

    def terminate_attempt(
        self, request: resource_pb2.TerminateAttemptRequest, _ctx: RequestContext
    ) -> resource_pb2.ActionResponse:
        try:
            identity = _attempt_identity_from_proto(request.attempt)
            receipt = self._resources.terminate_attempt(
                identity,
                idempotency_key=request.idempotency_key,
                principal_id=_action_principal(identity.task.resource_id),
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except (ResourceReplaced, ActionPolicyRejected) as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ActionIdempotencyConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        return resource_pb2.ActionResponse(receipt=_action_receipt_to_proto(receipt))

    def get_action_receipt(
        self, request: resource_pb2.GetActionReceiptRequest, _ctx: RequestContext
    ) -> resource_pb2.ActionResponse:
        try:
            receipt = self._resources.get_action_receipt(request.action_id)
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        _action_principal(receipt.target.resource_id)
        return resource_pb2.ActionResponse(receipt=_action_receipt_to_proto(receipt))

    def exec_attempt(
        self, request: resource_pb2.ExecAttemptRequest, _ctx: RequestContext
    ) -> resource_pb2.ExecAttemptResponse:
        try:
            locator = AttemptLocator(
                _resource_key_from_proto(request.attempt.task),
                request.attempt.attempt_number if request.attempt.HasField("attempt_number") else None,
            )
            _action_principal(locator.task.resource_id)
            response = self._resources.exec_attempt(
                locator,
                tuple(request.command),
                duration_from_proto(request.timeout) if request.HasField("timeout") else None,
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        return resource_pb2.ExecAttemptResponse(
            exit_code=response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
            error_message=response.error,
        )

    def profile_attempt(
        self, request: resource_pb2.ProfileAttemptRequest, _ctx: RequestContext
    ) -> resource_pb2.ProfileAttemptResponse:
        try:
            locator = AttemptLocator(
                _resource_key_from_proto(request.attempt.task),
                request.attempt.attempt_number if request.attempt.HasField("attempt_number") else None,
            )
            _action_principal(locator.task.resource_id)
            response = self._resources.profile_attempt(
                locator,
                request.profile,
                duration_from_proto(request.duration) if request.HasField("duration") else None,
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        return resource_pb2.ProfileAttemptResponse(
            profile_data=response.profile_data,
            error_message=response.error,
        )
