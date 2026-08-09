# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ResourceService wire codecs.

The frozen records in :mod:`iris.cluster.resources` are Iris's public Python API.
Generated protobuf messages stay at the RPC boundary because their mutable,
presence-sensitive semantics are transport concerns rather than resource behavior.
"""

from iris.cluster.constraints import Constraint
from iris.cluster.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.cluster.resources.activity import ActivityEntry, ActivityQuery
from iris.cluster.resources.attempt import AttemptDetail, AttemptRuntimeObject, AttemptSummary
from iris.cluster.resources.endpoint import (
    EndpointAccess,
    EndpointDetail,
    EndpointQuery,
    EndpointSummary,
    EndpointToken,
    ExecResult,
    ProfileResult,
)
from iris.cluster.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    NodeIdentity,
    NodeLocator,
    ResourceKey,
    ResourceKind,
    SliceIdentity,
    SliceLocator,
    TaskIdentity,
)
from iris.cluster.resources.job import JobDetail, JobQuery, JobSpec, JobSummary
from iris.cluster.resources.log import LogPage, LogQuery
from iris.cluster.resources.node import (
    NodeAttribute,
    NodeAttributeKind,
    NodeCapacity,
    NodeDetail,
    NodeHealth,
    NodeQuery,
    NodeSummary,
)
from iris.cluster.resources.slice import (
    MembershipState,
    SliceDetail,
    SliceLifecycle,
    SliceMember,
    SliceQuery,
    SliceSummary,
)
from iris.cluster.resources.source import Freshness, Page, ResourceSourceStatus, SourceState
from iris.cluster.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.cluster.types import CoschedulingConfig, ResourceSpec
from iris.rpc import resource_pb2
from iris.time_proto import duration_from_proto, duration_to_proto, timestamp_from_proto, timestamp_to_proto

_RESOURCE_KIND_TO_PROTO = {
    ResourceKind.JOB: resource_pb2.RESOURCE_KIND_JOB,
    ResourceKind.TASK: resource_pb2.RESOURCE_KIND_TASK,
    ResourceKind.ATTEMPT: resource_pb2.RESOURCE_KIND_ATTEMPT,
    ResourceKind.ENDPOINT: resource_pb2.RESOURCE_KIND_ENDPOINT,
    ResourceKind.NODE: resource_pb2.RESOURCE_KIND_NODE,
    ResourceKind.SLICE: resource_pb2.RESOURCE_KIND_SLICE,
}
_RESOURCE_KIND_FROM_PROTO = {value: key for key, value in _RESOURCE_KIND_TO_PROTO.items()}
_SOURCE_STATE_FROM_PROTO = {
    resource_pb2.SOURCE_STATE_AVAILABLE: SourceState.AVAILABLE,
    resource_pb2.SOURCE_STATE_UNAVAILABLE: SourceState.UNAVAILABLE,
    resource_pb2.SOURCE_STATE_UNSUPPORTED: SourceState.UNSUPPORTED,
}
_FRESHNESS_FROM_PROTO = {
    resource_pb2.FRESHNESS_CURRENT: Freshness.CURRENT,
    resource_pb2.FRESHNESS_STALE: Freshness.STALE,
    resource_pb2.FRESHNESS_UNKNOWN: Freshness.UNKNOWN,
}
_NODE_HEALTH_TO_PROTO = {
    NodeHealth.READY: resource_pb2.NODE_HEALTH_READY,
    NodeHealth.DEGRADED: resource_pb2.NODE_HEALTH_DEGRADED,
    NodeHealth.UNAVAILABLE: resource_pb2.NODE_HEALTH_UNAVAILABLE,
    NodeHealth.RETIRED: resource_pb2.NODE_HEALTH_RETIRED,
}
_NODE_HEALTH_FROM_PROTO = {value: key for key, value in _NODE_HEALTH_TO_PROTO.items()}
_SLICE_LIFECYCLE_FROM_PROTO = {
    resource_pb2.SLICE_LIFECYCLE_CREATING: SliceLifecycle.CREATING,
    resource_pb2.SLICE_LIFECYCLE_READY: SliceLifecycle.READY,
    resource_pb2.SLICE_LIFECYCLE_DELETING: SliceLifecycle.DELETING,
    resource_pb2.SLICE_LIFECYCLE_FAILED: SliceLifecycle.FAILED,
}
_MEMBERSHIP_STATE_FROM_PROTO = {
    resource_pb2.MEMBERSHIP_STATE_UNKNOWN: MembershipState.UNKNOWN,
    resource_pb2.MEMBERSHIP_STATE_OBSERVED: MembershipState.OBSERVED,
}
_ENDPOINT_ACCESS_FROM_PROTO = {
    resource_pb2.ENDPOINT_ACCESS_PRIVATE: EndpointAccess.PRIVATE,
    resource_pb2.ENDPOINT_ACCESS_LINK: EndpointAccess.LINK,
}
_ACTION_KIND_FROM_PROTO = {
    resource_pb2.ACTION_KIND_CANCEL_JOB: ActionKind.CANCEL_JOB,
    resource_pb2.ACTION_KIND_RETRY_TASK: ActionKind.RETRY_TASK,
    resource_pb2.ACTION_KIND_TERMINATE_ATTEMPT: ActionKind.TERMINATE_ATTEMPT,
}
_ACTION_STATE_FROM_PROTO = {
    resource_pb2.ACTION_STATE_ACCEPTED: ActionState.ACCEPTED,
    resource_pb2.ACTION_STATE_VERIFYING: ActionState.VERIFYING,
    resource_pb2.ACTION_STATE_SUCCEEDED: ActionState.SUCCEEDED,
    resource_pb2.ACTION_STATE_FAILED: ActionState.FAILED,
}
_ACTION_RESULT_FROM_PROTO = {
    resource_pb2.ACTION_RESULT_NONE: ActionResult.NONE,
    resource_pb2.ACTION_RESULT_SATISFIED: ActionResult.SATISFIED,
    resource_pb2.ACTION_RESULT_TARGET_ABSENT: ActionResult.TARGET_ABSENT,
    resource_pb2.ACTION_RESULT_PROVIDER_REJECTED: ActionResult.PROVIDER_REJECTED,
    resource_pb2.ACTION_RESULT_INTERNAL_ERROR: ActionResult.INTERNAL_ERROR,
}


def resource_key_to_proto(value: ResourceKey) -> resource_pb2.ResourceKey:
    return resource_pb2.ResourceKey(
        cluster_id=value.cluster_id,
        kind=_RESOURCE_KIND_TO_PROTO[value.kind],
        resource_id=value.resource_id,
    )


def _resource_key_from_proto(value: resource_pb2.ResourceKey) -> ResourceKey:
    try:
        kind = _RESOURCE_KIND_FROM_PROTO[value.kind]
    except KeyError as exc:
        raise ValueError("resource response contains an unspecified kind") from exc
    return ResourceKey(value.cluster_id, kind, value.resource_id)


def job_identity_to_proto(value: JobIdentity) -> resource_pb2.JobIdentity:
    return resource_pb2.JobIdentity(key=resource_key_to_proto(value.key), job_uid=value.job_uid)


def job_identity_from_proto(value: resource_pb2.JobIdentity) -> JobIdentity:
    return JobIdentity(_resource_key_from_proto(value.key), value.job_uid)


def task_identity_to_proto(value: TaskIdentity) -> resource_pb2.TaskIdentity:
    return resource_pb2.TaskIdentity(key=resource_key_to_proto(value.key), task_uid=value.task_uid)


def _task_identity_from_proto(value: resource_pb2.TaskIdentity) -> TaskIdentity:
    return TaskIdentity(_resource_key_from_proto(value.key), value.task_uid)


def attempt_locator_to_proto(value: AttemptLocator) -> resource_pb2.AttemptLocator:
    result = resource_pb2.AttemptLocator(task=resource_key_to_proto(value.task))
    if value.attempt_number is not None:
        result.attempt_number = value.attempt_number
    return result


def attempt_identity_to_proto(value: AttemptIdentity) -> resource_pb2.AttemptIdentity:
    return resource_pb2.AttemptIdentity(
        task=resource_key_to_proto(value.task),
        attempt_number=value.attempt_number,
        attempt_uid=value.attempt_uid,
    )


def _attempt_identity_from_proto(value: resource_pb2.AttemptIdentity) -> AttemptIdentity:
    return AttemptIdentity(_resource_key_from_proto(value.task), value.attempt_number, value.attempt_uid)


def _node_identity_from_proto(value: resource_pb2.NodeIdentity) -> NodeIdentity:
    return NodeIdentity(_resource_key_from_proto(value.key), value.backend_id, value.node_uid)


def node_locator_to_proto(value: NodeLocator) -> resource_pb2.NodeLocator:
    result = resource_pb2.NodeLocator(key=resource_key_to_proto(value.key), backend_id=value.backend_id)
    if value.node_uid is not None:
        result.node_uid = value.node_uid
    return result


def _slice_identity_from_proto(value: resource_pb2.SliceIdentity) -> SliceIdentity:
    return SliceIdentity(_resource_key_from_proto(value.key), value.backend_id, value.slice_uid)


def slice_locator_to_proto(value: SliceLocator) -> resource_pb2.SliceLocator:
    result = resource_pb2.SliceLocator(key=resource_key_to_proto(value.key), backend_id=value.backend_id)
    if value.slice_uid is not None:
        result.slice_uid = value.slice_uid
    return result


def _source_status_from_proto(value: resource_pb2.ResourceSourceStatus) -> ResourceSourceStatus:
    try:
        state = _SOURCE_STATE_FROM_PROTO[value.state]
        freshness = _FRESHNESS_FROM_PROTO[value.freshness]
    except KeyError as exc:
        raise ValueError("resource response contains an unspecified source status") from exc
    return ResourceSourceStatus(
        source_id=value.source_id,
        backend_id=value.backend_id,
        state=state,
        freshness=freshness,
        observed_at=timestamp_from_proto(value.observed_at) if value.HasField("observed_at") else None,
        error_code=value.error_code,
        error_message=value.error_message,
    )


def _page_request(page_size: int, page_token: str | None) -> resource_pb2.PageRequest:
    return resource_pb2.PageRequest(page_size=page_size, page_token=page_token or "")


def _page[T](items: tuple[T, ...], value: resource_pb2.PageInfo) -> Page[T]:
    return Page(
        items=items,
        next_page_token=value.next_page_token or None,
        source_statuses=tuple(_source_status_from_proto(status) for status in value.source_statuses),
    )


def job_query_to_proto(value: JobQuery) -> resource_pb2.JobQuery:
    result = resource_pb2.JobQuery(
        owner_id=value.owner_id or "",
        job_id_prefix=value.job_id_prefix or "",
        states=value.states,
        backend_id=value.backend_id or "",
        execution_cluster_id=value.execution_cluster_id or "",
        page=_page_request(value.page_size, value.page_token),
    )
    if value.parent is not None:
        result.parent.CopyFrom(resource_key_to_proto(value.parent))
    return result


def _job_summary_from_proto(value: resource_pb2.JobSummary) -> JobSummary:
    return JobSummary(
        identity=job_identity_from_proto(value.identity),
        owner_id=value.owner_id,
        parent=job_identity_from_proto(value.parent) if value.HasField("parent") else None,
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        num_tasks=value.num_tasks,
        submitted_at=timestamp_from_proto(value.submitted_at),
        started_at=timestamp_from_proto(value.started_at) if value.HasField("started_at") else None,
        finished_at=timestamp_from_proto(value.finished_at) if value.HasField("finished_at") else None,
        error_message=value.error_message,
        pending_reason=value.pending_reason,
    )


def job_spec_to_proto(value: JobSpec) -> resource_pb2.JobSpec:
    result = resource_pb2.JobSpec(
        version=value.version,
        name=value.name,
        entrypoint=value.entrypoint,
        resources=value.resources.to_exact_proto(),
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
        result.scheduling_timeout.CopyFrom(duration_to_proto(value.scheduling_timeout))
    if value.coscheduling is not None:
        result.coscheduling.CopyFrom(value.coscheduling.to_proto())
    if value.timeout is not None:
        result.timeout.CopyFrom(duration_to_proto(value.timeout))
    return result


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


def job_detail_from_proto(value: resource_pb2.JobDetail) -> JobDetail:
    return JobDetail(_job_summary_from_proto(value.summary), _job_spec_from_proto(value.spec))


def job_page_from_proto(value: resource_pb2.ListJobsResponse) -> Page[JobSummary]:
    return _page(tuple(_job_summary_from_proto(item) for item in value.jobs), value.page)


def task_query_to_proto(value: TaskQuery) -> resource_pb2.TaskQuery:
    result = resource_pb2.TaskQuery(
        job_id_prefix=value.job_id_prefix or "",
        states=value.states,
        backend_id=value.backend_id or "",
        authority_cluster_id=value.authority_cluster_id or "",
        execution_cluster_id=value.execution_cluster_id or "",
        page=_page_request(value.page_size, value.page_token),
    )
    if value.job is not None:
        result.job.CopyFrom(resource_key_to_proto(value.job))
    return result


def _task_summary_from_proto(value: resource_pb2.TaskSummary) -> TaskSummary:
    return TaskSummary(
        identity=_task_identity_from_proto(value.identity),
        job=job_identity_from_proto(value.job),
        task_index=value.task_index,
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        current_attempt=(
            _attempt_identity_from_proto(value.current_attempt) if value.HasField("current_attempt") else None
        ),
        current_node=_node_identity_from_proto(value.current_node) if value.HasField("current_node") else None,
        failure_count=value.failure_count,
        preemption_count=value.preemption_count,
        submitted_at=timestamp_from_proto(value.submitted_at),
        started_at=timestamp_from_proto(value.started_at) if value.HasField("started_at") else None,
        finished_at=timestamp_from_proto(value.finished_at) if value.HasField("finished_at") else None,
        status_message=value.status_message,
        error_message=value.error_message,
    )


def _attempt_summary_from_proto(value: resource_pb2.AttemptSummary) -> AttemptSummary:
    return AttemptSummary(
        identity=_attempt_identity_from_proto(value.identity),
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        node=_node_identity_from_proto(value.node) if value.HasField("node") else None,
        created_at=timestamp_from_proto(value.created_at),
        started_at=timestamp_from_proto(value.started_at) if value.HasField("started_at") else None,
        finished_at=timestamp_from_proto(value.finished_at) if value.HasField("finished_at") else None,
        exit_code=value.exit_code if value.HasField("exit_code") else None,
        error_message=value.error_message,
        terminal_reason=value.terminal_reason,
    )


def task_page_from_proto(value: resource_pb2.ListTasksResponse) -> Page[TaskSummary]:
    return _page(tuple(_task_summary_from_proto(item) for item in value.tasks), value.page)


def task_detail_from_proto(value: resource_pb2.TaskDetail) -> TaskDetail:
    return TaskDetail(
        summary=_task_summary_from_proto(value.summary),
        attempts=tuple(_attempt_summary_from_proto(item) for item in value.attempts),
        source_statuses=tuple(_source_status_from_proto(status) for status in value.source_statuses),
        root_cause_highlights=tuple(value.root_cause_highlights),
    )


def task_details_from_proto(value: resource_pb2.BatchDescribeTasksResponse) -> tuple[TaskDetail, ...]:
    return tuple(task_detail_from_proto(item) for item in value.tasks)


def attempt_detail_from_proto(value: resource_pb2.AttemptDetail) -> AttemptDetail:
    runtime = None
    if value.HasField("runtime"):
        runtime = AttemptRuntimeObject(
            provider_kind=value.runtime.provider_kind,
            namespace=value.runtime.namespace,
            name=value.runtime.name,
            provider_uid=value.runtime.provider_uid,
            provider_node_id=value.runtime.provider_node_id,
            provider_node_uid=value.runtime.provider_node_uid,
            container_id=value.runtime.container_id,
            observed_at=timestamp_from_proto(value.runtime.observed_at),
        )
    return AttemptDetail(
        summary=_attempt_summary_from_proto(value.summary),
        runtime=runtime,
        source_statuses=tuple(_source_status_from_proto(status) for status in value.source_statuses),
    )


def node_query_to_proto(value: NodeQuery) -> resource_pb2.NodeQuery:
    return resource_pb2.NodeQuery(
        backend_id=value.backend_id or "",
        contains=value.contains or "",
        health=[_NODE_HEALTH_TO_PROTO[item] for item in value.health],
        page=_page_request(value.page_size, value.page_token),
    )


def _node_summary_from_proto(value: resource_pb2.NodeSummary) -> NodeSummary:
    try:
        health = _NODE_HEALTH_FROM_PROTO[value.health]
    except KeyError as exc:
        raise ValueError("node response contains unspecified health") from exc
    capacity = value.capacity
    return NodeSummary(
        identity=_node_identity_from_proto(value.identity),
        health=health,
        schedulable=value.schedulable,
        capacity=NodeCapacity(
            cpu_millicores=capacity.cpu_millicores,
            memory_bytes=capacity.memory_bytes,
            disk_bytes=capacity.disk_bytes,
            accelerator_kind=capacity.accelerator_kind,
            accelerator_variant=capacity.accelerator_variant,
            accelerator_count=capacity.accelerator_count,
        ),
        scaling_group_id=value.scaling_group_id or None,
        slice=_slice_identity_from_proto(value.slice) if value.HasField("slice") else None,
        running_task_count=value.running_task_count,
        observed_at=timestamp_from_proto(value.observed_at),
        region=value.region or None,
    )


def _node_attribute(value: resource_pb2.NodeAttribute) -> NodeAttribute:
    selected = value.WhichOneof("value")
    if selected == "string_value":
        return NodeAttribute(value.key, NodeAttributeKind.STRING, string_value=value.string_value)
    if selected == "integer_value":
        return NodeAttribute(value.key, NodeAttributeKind.INTEGER, integer_value=value.integer_value)
    if selected == "float_value":
        return NodeAttribute(value.key, NodeAttributeKind.FLOAT, float_value=value.float_value)
    raise ValueError("node response contains an attribute without a value")


def node_page_from_proto(value: resource_pb2.ListNodesResponse) -> Page[NodeSummary]:
    return _page(tuple(_node_summary_from_proto(item) for item in value.nodes), value.page)


def node_detail_from_proto(value: resource_pb2.NodeDetail) -> NodeDetail:
    return NodeDetail(
        summary=_node_summary_from_proto(value.summary),
        address=value.address or None,
        attributes=tuple(_node_attribute(item) for item in value.attributes),
        recent_attempts=tuple(_attempt_summary_from_proto(item) for item in value.recent_attempts),
        bootstrap_log_key=value.bootstrap_log_key or None,
        source_statuses=tuple(_source_status_from_proto(status) for status in value.source_statuses),
    )


def slice_query_to_proto(value: SliceQuery) -> resource_pb2.SliceQuery:
    return resource_pb2.SliceQuery(
        backend_id=value.backend_id or "",
        scaling_group_id=value.scaling_group_id or "",
        page=_page_request(value.page_size, value.page_token),
    )


def _slice_summary_from_proto(value: resource_pb2.SliceSummary) -> SliceSummary:
    try:
        lifecycle = _SLICE_LIFECYCLE_FROM_PROTO[value.lifecycle]
        membership = _MEMBERSHIP_STATE_FROM_PROTO[value.membership_state]
    except KeyError as exc:
        raise ValueError("slice response contains an unspecified state") from exc
    return SliceSummary(
        identity=_slice_identity_from_proto(value.identity),
        scaling_group_id=value.scaling_group_id,
        lifecycle=lifecycle,
        membership_state=membership,
        observed_member_count=value.observed_member_count,
        observed_at=timestamp_from_proto(value.observed_at) if value.HasField("observed_at") else None,
        error_message=value.error_message,
    )


def slice_page_from_proto(value: resource_pb2.ListSlicesResponse) -> Page[SliceSummary]:
    return _page(tuple(_slice_summary_from_proto(item) for item in value.slices), value.page)


def slice_detail_from_proto(value: resource_pb2.SliceDetail) -> SliceDetail:
    return SliceDetail(
        summary=_slice_summary_from_proto(value.summary),
        members=tuple(
            SliceMember(
                provider_node_id=item.provider_node_id,
                node=_node_identity_from_proto(item.node) if item.HasField("node") else None,
                observed_at=timestamp_from_proto(item.observed_at),
            )
            for item in value.members
        ),
        source_statuses=tuple(_source_status_from_proto(status) for status in value.source_statuses),
    )


def endpoint_query_to_proto(value: EndpointQuery) -> resource_pb2.EndpointQuery:
    result = resource_pb2.EndpointQuery(
        name_prefix=value.name_prefix or "",
        page=_page_request(value.page_size, value.page_token),
    )
    if value.task is not None:
        result.task.CopyFrom(resource_key_to_proto(value.task))
    return result


def _endpoint_summary_from_proto(value: resource_pb2.EndpointSummary) -> EndpointSummary:
    try:
        access = _ENDPOINT_ACCESS_FROM_PROTO[value.access]
    except KeyError as exc:
        raise ValueError("endpoint response contains unspecified access") from exc
    return EndpointSummary(
        key=_resource_key_from_proto(value.key),
        endpoint_id=value.endpoint_id,
        name=value.name,
        task=_resource_key_from_proto(value.task) if value.HasField("task") else None,
        execution_cluster_id=value.execution_cluster_id,
        access=access,
        lease_deadline=timestamp_from_proto(value.lease_deadline) if value.HasField("lease_deadline") else None,
    )


def endpoint_page_from_proto(value: resource_pb2.ListEndpointsResponse) -> Page[EndpointSummary]:
    return _page(tuple(_endpoint_summary_from_proto(item) for item in value.endpoints), value.page)


def endpoint_detail_from_proto(value: resource_pb2.EndpointDetail) -> EndpointDetail:
    return EndpointDetail(_endpoint_summary_from_proto(value.summary), value.address, dict(value.metadata))


def endpoint_details_from_proto(value: resource_pb2.BatchDescribeEndpointsResponse) -> tuple[EndpointDetail, ...]:
    return tuple(endpoint_detail_from_proto(endpoint) for endpoint in value.endpoints)


def endpoint_token_from_proto(value: resource_pb2.MintEndpointTokenResponse) -> EndpointToken:
    return EndpointToken(value.token, timestamp_from_proto(value.expires_at), value.capability_url)


def activity_query_to_proto(value: ActivityQuery) -> resource_pb2.ActivityQuery:
    result = resource_pb2.ActivityQuery(
        target=resource_key_to_proto(value.target),
        attempt_uid=value.attempt_uid or "",
        page=_page_request(value.page_size, value.page_token),
    )
    if value.after is not None:
        result.after.CopyFrom(timestamp_to_proto(value.after))
    return result


def activity_page_from_proto(value: resource_pb2.ListActivityResponse) -> Page[ActivityEntry]:
    return _page(
        tuple(
            ActivityEntry(
                entry_id=item.entry_id,
                occurred_at=timestamp_from_proto(item.occurred_at),
                source=item.source,
                severity=item.severity,
                kind=item.kind,
                message=item.message,
                target=_resource_key_from_proto(item.target),
                attempt_uid=item.attempt_uid or None,
                correlation_id=item.correlation_id or None,
                attributes=dict(item.attributes),
            )
            for item in value.entries
        ),
        value.page,
    )


def log_query_to_proto(value: LogQuery) -> resource_pb2.LogQuery:
    result = resource_pb2.LogQuery(
        cursor=value.cursor,
        max_lines=value.max_lines,
        substring=value.substring,
        minimum_level=value.minimum_level,
        tail=value.tail,
    )
    if value.after is not None:
        result.after.CopyFrom(timestamp_to_proto(value.after))
    return result


def log_page_from_proto(value: resource_pb2.FetchLogsResponse) -> LogPage:
    return LogPage(
        entries=tuple(value.entries),
        next_cursor=value.next_cursor,
        source_statuses=tuple(_source_status_from_proto(status) for status in value.source_statuses),
    )


def action_receipt_from_proto(value: resource_pb2.ActionReceipt) -> ActionReceipt:
    try:
        kind = _ACTION_KIND_FROM_PROTO[value.kind]
        state = _ACTION_STATE_FROM_PROTO[value.state]
        result = _ACTION_RESULT_FROM_PROTO[value.result_code]
    except KeyError as exc:
        raise ValueError("action response contains an unspecified state") from exc
    return ActionReceipt(
        action_id=value.action_id,
        kind=kind,
        target=_resource_key_from_proto(value.target),
        expected_target_uid=value.expected_target_uid,
        expected_attempt_uid=value.expected_attempt_uid or None,
        state=state,
        result_code=result,
        result_message=value.result_message,
        created_at=timestamp_from_proto(value.created_at),
        updated_at=timestamp_from_proto(value.updated_at),
        completed_at=timestamp_from_proto(value.completed_at) if value.HasField("completed_at") else None,
    )


def exec_result_from_proto(value: resource_pb2.ExecAttemptResponse) -> ExecResult:
    return ExecResult(value.exit_code, value.stdout, value.stderr, value.error_message)


def profile_result_from_proto(value: resource_pb2.ProfileAttemptResponse) -> ProfileResult:
    return ProfileResult(value.profile_data, value.error_message)
