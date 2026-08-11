# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Type-specific ResourceService registrations."""

import uuid
from typing import TypeVar

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf import any_pb2
from google.protobuf.message import Message
from rigging.server_auth import ANONYMOUS_ADMIN, get_verified_identity
from rigging.timing import Timestamp

from iris.backends.protocol import BackendCapability
from iris.cluster.controller.auth import WORKER_ROLE
from iris.cluster.controller.controller import Controller
from iris.cluster.federation.protocol import PeerCallError
from iris.resources.action import ActionKind
from iris.resources.activity import ActivityEntry, ActivityQuery
from iris.resources.attempt import AttemptSummary
from iris.resources.capacity import (
    CapacityBackend,
    CapacityKubernetesStatus,
    CapacityPeerBackend,
    CapacityRouting,
    CapacityScalingGroup,
    ResourceAvailability,
)
from iris.resources.endpoint import EndpointDetail, EndpointQuery, EndpointSummary
from iris.resources.errors import (
    ActionIdempotencyConflict,
    ActionPolicyRejected,
    InvalidPageToken,
    InvalidResourceKey,
    InvalidResourceRequest,
    ResourceConflict,
    ResourceExhausted,
    ResourceNotFound,
    ResourcePermissionDenied,
    ResourcePreconditionFailed,
    ResourceReplaced,
    ResourceSourceUnavailable,
)
from iris.resources.identity import (
    AttemptIdentity,
    ResourceKey,
    ResourceKind,
)
from iris.resources.job import JobQuery, JobSummary
from iris.resources.log import LogEntry, LogLevel, LogQuery
from iris.resources.names import JobName
from iris.resources.node import NodeAttribute, NodeAttributeKind, NodeQuery, NodeSummary
from iris.resources.slice import SliceMember, SliceQuery, SliceSummary
from iris.resources.source import ResourceSourceStatus
from iris.resources.state import JobState, TaskState
from iris.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.resources.user import UserSummary
from iris.rpc import iris_logging_pb2, resource_pb2
from iris.rpc.auth import DASHBOARD_ROLE, FEDERATION_PEER_ROLE, authorize_resource_owner
from iris.rpc.federation_client import peer_connect_error
from iris.rpc.proto_display import job_state_friendly, task_state_friendly
from iris.rpc.resource_codec import (
    action_receipt_to_proto as _action_receipt_to_proto,
)
from iris.rpc.resource_codec import (
    attempt_identity_from_proto as _attempt_identity_from_proto,
)
from iris.rpc.resource_codec import (
    attempt_identity_to_proto as _attempt_identity_to_proto,
)
from iris.rpc.resource_codec import (
    attempt_locator_from_proto,
    endpoint_access_to_proto,
    job_spec_from_proto,
    membership_state_to_proto,
    node_health_from_proto,
    node_health_to_proto,
    node_locator_from_proto,
    profile_configuration_from_proto,
    redacted_job_spec_to_proto,
    resource_spec_to_proto,
    slice_capacity_state_to_proto,
    slice_lifecycle_to_proto,
    slice_locator_from_proto,
)
from iris.rpc.resource_codec import (
    job_identity_from_proto as _job_identity_from_proto,
)
from iris.rpc.resource_codec import (
    job_identity_to_proto as _job_identity_to_proto,
)
from iris.rpc.resource_codec import (
    node_identity_to_proto as _node_identity_to_proto,
)
from iris.rpc.resource_codec import (
    resource_key_from_proto as _resource_key_from_proto,
)
from iris.rpc.resource_codec import (
    resource_key_to_proto as _resource_key_to_proto,
)
from iris.rpc.resource_codec import (
    resource_source_status_to_proto as _source_status_to_proto,
)
from iris.rpc.resource_codec import (
    slice_identity_to_proto as _slice_identity_to_proto,
)
from iris.rpc.resource_codec import (
    task_identity_from_proto as _task_identity_from_proto,
)
from iris.rpc.resource_codec import (
    task_identity_to_proto as _task_identity_to_proto,
)
from iris.rpc.resource_registry import BackendResourceRegistration, ResourceCatalog, ResourceRegistration
from iris.rpc.resource_types import (
    ACTIVITY_ENTRY,
    ATTEMPT,
    CAPACITY,
    ENDPOINT,
    ENDPOINT_CAPABILITY,
    EXEC_SESSION,
    JOB,
    LOG_ENTRY,
    NODE,
    OPERATION,
    PROFILE_CAPTURE,
    SLICE,
    TASK,
    USER_SUMMARY,
)
from iris.time_proto import duration_from_proto, timestamp_from_proto, timestamp_to_proto

_DEFAULT_JOB_PAGE_SIZE = 50
_DEFAULT_RESOURCE_PAGE_SIZE = 100
_DEFAULT_ACTIVITY_PAGE_SIZE = 200


def _node_summary_to_proto(value: NodeSummary) -> resource_pb2.NodeSummary:
    result = resource_pb2.NodeSummary(
        identity=_node_identity_to_proto(value.identity),
        health=node_health_to_proto(value.health),
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
        region=value.region or "",
    )
    if value.slice is not None:
        result.slice.CopyFrom(_slice_identity_to_proto(value.slice))
    return result


def _node_attribute_to_proto(value: NodeAttribute) -> resource_pb2.NodeAttribute:
    result = resource_pb2.NodeAttribute(key=value.key)
    if value.kind is NodeAttributeKind.STRING:
        result.string_value = value.string_value or ""
    elif value.kind is NodeAttributeKind.INTEGER:
        result.integer_value = value.integer_value or 0
    else:
        result.float_value = value.float_value or 0.0
    return result


def _slice_summary_to_proto(value: SliceSummary) -> resource_pb2.SliceSummary:
    result = resource_pb2.SliceSummary(
        identity=_slice_identity_to_proto(value.identity),
        scaling_group_id=value.scaling_group_id,
        lifecycle=slice_lifecycle_to_proto(value.lifecycle),
        membership_state=membership_state_to_proto(value.membership_state),
        observed_member_count=value.observed_member_count,
        error_message=value.error_message,
        capacity_state=slice_capacity_state_to_proto(value.capacity_state),
        healthy_member_count=value.healthy_member_count,
        degraded_member_count=value.degraded_member_count,
        running_task_count=value.running_task_count,
    )
    if value.observed_at is not None:
        result.observed_at.CopyFrom(timestamp_to_proto(value.observed_at))
    if value.created_at is not None:
        result.created_at.CopyFrom(timestamp_to_proto(value.created_at))
    if value.last_active_at is not None:
        result.last_active_at.CopyFrom(timestamp_to_proto(value.last_active_at))
    return result


def _slice_member_to_proto(value: SliceMember) -> resource_pb2.SliceMember:
    result = resource_pb2.SliceMember(
        provider_node_id=value.provider_node_id,
        observed_at=timestamp_to_proto(value.observed_at),
        worker_id=value.worker_id,
        healthy=value.healthy,
        usability=value.usability,
        running_task_count=value.running_task_count,
        zone=value.zone,
    )
    if value.node is not None:
        result.node.CopyFrom(_node_identity_to_proto(value.node))
    return result


def _capacity_availability_to_proto(value: ResourceAvailability) -> resource_pb2.CapacityResourceAvailability:
    return resource_pb2.CapacityResourceAvailability(
        version=value.version,
        observed_at=timestamp_to_proto(value.observed_at),
        amounts=value.amounts,
        total_amounts=value.total_amounts,
        held_by_band=[
            resource_pb2.CapacityBandAvailability(band=band, amounts=amounts)
            for band, amounts in sorted(value.held_by_band.items())
        ],
    )


def _user_summary_to_proto(value: UserSummary) -> resource_pb2.UserSummary:
    result = resource_pb2.UserSummary(
        user_id=value.user_id,
        task_state_counts={task_state_friendly(state): count for state, count in value.task_state_counts},
        job_state_counts={job_state_friendly(state): count for state, count in value.job_state_counts},
        role=value.role,
        budget_spent=value.budget_spent,
        budget_configured=value.budget_limit is not None,
    )
    if value.budget_limit is not None:
        result.budget_limit = value.budget_limit
    if value.max_band is not None:
        result.max_band = value.max_band
    return result


def _capacity_scaling_group_to_proto(value: CapacityScalingGroup) -> resource_pb2.CapacityScalingGroup:
    result = resource_pb2.CapacityScalingGroup(
        name=value.name,
        backend_id=value.backend_id,
        device_type=value.device_type,
        device_variant=value.device_variant,
        quota_pool=value.quota_pool,
        allocation_tier=value.allocation_tier,
        region=value.region,
        current_demand=value.current_demand,
        peak_demand=value.peak_demand,
        consecutive_failures=value.consecutive_failures,
        slices=[
            resource_pb2.CapacitySlice(
                summary=_slice_summary_to_proto(item.summary),
                members=[_slice_member_to_proto(member) for member in item.members],
            )
            for item in value.slices
        ],
        slice_state_counts=value.slice_state_counts,
        availability_status=value.availability_status,
        availability_reason=value.availability_reason,
        idle_threshold_ms=value.idle_threshold_ms,
    )
    for field, timestamp in (
        (result.backoff_until, value.backoff_until),
        (result.last_scale_up, value.last_scale_up),
        (result.last_scale_down, value.last_scale_down),
        (result.blocked_until, value.blocked_until),
        (result.scale_up_cooldown_until, value.scale_up_cooldown_until),
    ):
        if timestamp is not None:
            field.CopyFrom(timestamp_to_proto(timestamp))
    return result


def _capacity_routing_to_proto(value: CapacityRouting) -> resource_pb2.CapacityRouting:
    return resource_pb2.CapacityRouting(
        unmet=[
            resource_pb2.CapacityUnmetDemand(
                entry=resource_pb2.CapacityDemandEntry(
                    task_ids=item.entry.task_ids,
                    coschedule_group_id=item.entry.coschedule_group_id,
                    device_type=item.entry.device_type,
                    device_variant=item.entry.device_variant,
                    preemptible=item.entry.preemptible,
                ),
                reason=item.reason,
            )
            for item in value.unmet
        ],
        groups=[
            resource_pb2.CapacityGroupRouting(
                scaling_group_id=item.scaling_group_id,
                priority=item.priority,
                assigned=item.assigned,
                launch=item.launch,
                decision=item.decision,
                reason=item.reason,
            )
            for item in value.groups
        ],
    )


def _capacity_kubernetes_to_proto(value: CapacityKubernetesStatus) -> resource_pb2.CapacityKubernetesStatus:
    return resource_pb2.CapacityKubernetesStatus(
        namespace=value.namespace,
        total_nodes=value.total_nodes,
        schedulable_nodes=value.schedulable_nodes,
        allocatable_cpu=value.allocatable_cpu,
        allocatable_memory=value.allocatable_memory,
        pods=[
            resource_pb2.CapacityKubernetesPod(
                pod_name=item.pod_name,
                task_id=item.task_id,
                phase=item.phase,
                reason=item.reason,
                message=item.message,
                last_transition=(timestamp_to_proto(item.last_transition) if item.last_transition is not None else None),
                node_name=item.node_name,
            )
            for item in value.pods
        ],
        provider_version=value.provider_version,
        pools=[
            resource_pb2.CapacityKubernetesPool(
                name=item.name,
                instance_type=item.instance_type,
                scaling_group_id=item.scaling_group_id,
                target_nodes=item.target_nodes,
                current_nodes=item.current_nodes,
                queued_nodes=item.queued_nodes,
                in_progress_nodes=item.in_progress_nodes,
                autoscaling=item.autoscaling,
                min_nodes=item.min_nodes,
                max_nodes=item.max_nodes,
                capacity=item.capacity,
                quota=item.quota,
            )
            for item in value.pools
        ],
        nodes=[
            resource_pb2.CapacityKubernetesNode(
                name=item.name,
                ready=item.ready,
                schedulable=item.schedulable,
                status_summary=item.status_summary,
                instance_type=item.instance_type,
                region=item.region,
                accelerator_count=item.accelerator_count,
                accelerator_variant=item.accelerator_variant,
                cpu_millicores=item.cpu_millicores,
                memory_bytes=item.memory_bytes,
                disk_bytes=item.disk_bytes,
                running_pods=item.running_pods,
                created=item.created,
            )
            for item in value.nodes
        ],
    )


def _capacity_backend_to_proto(value: CapacityBackend) -> resource_pb2.CapacityBackend:
    result = resource_pb2.CapacityBackend(
        backend_id=value.backend_id,
        name=value.name,
        kind=value.kind,
        capabilities=value.capabilities,
        worker_count=value.worker_count,
        pending_task_count=value.pending_task_count,
        running_task_count=value.running_task_count,
        has_autoscaler=value.has_autoscaler,
        capacity_health=value.capacity_health,
        scaling_groups=[_capacity_scaling_group_to_proto(item) for item in value.scaling_groups],
        recent_actions=[
            resource_pb2.CapacityAction(
                timestamp=(timestamp_to_proto(item.timestamp) if item.timestamp is not None else None),
                action_type=item.action_type,
                scaling_group_id=item.scaling_group_id,
                slice_id=item.slice_id,
                reason=item.reason,
                status=item.status,
            )
            for item in value.recent_actions
        ],
        healthy_worker_count=value.healthy_worker_count,
    )
    for key, values in value.advertised_attributes.items():
        result.advertised_attributes[key].values.extend(values)
    if value.availability is not None:
        result.availability.CopyFrom(_capacity_availability_to_proto(value.availability))
    if value.routing is not None:
        result.routing.CopyFrom(_capacity_routing_to_proto(value.routing))
    if value.last_evaluation is not None:
        result.last_evaluation.CopyFrom(timestamp_to_proto(value.last_evaluation))
    if value.kubernetes is not None:
        result.kubernetes.CopyFrom(_capacity_kubernetes_to_proto(value.kubernetes))
    return result


def _capacity_peer_backend_to_proto(value: CapacityPeerBackend) -> resource_pb2.CapacityPeerBackend:
    result = resource_pb2.CapacityPeerBackend(
        backend_id=value.backend_id,
        name=value.name,
        kind=value.kind,
        capabilities=value.capabilities,
        scaling_groups=value.scale_groups,
        worker_count=value.worker_count,
        pending_task_count=value.pending_task_count,
        running_task_count=value.running_task_count,
        has_autoscaler=value.has_autoscaler,
        capacity_health=value.capacity_health,
    )
    for key, values in value.advertised_attributes.items():
        result.advertised_attributes[key].values.extend(values)
    if value.availability is not None:
        result.availability.CopyFrom(_capacity_availability_to_proto(value.availability))
    return result


def _page_info(next_page_token: str | None, statuses: tuple[ResourceSourceStatus, ...]) -> resource_pb2.PageInfo:
    return resource_pb2.PageInfo(
        next_page_token=next_page_token or "",
        source_statuses=[_source_status_to_proto(status) for status in statuses],
    )


def _log_entry_to_proto(value: LogEntry) -> iris_logging_pb2.LogEntry:
    result = iris_logging_pb2.LogEntry(
        source=value.source,
        data=value.data,
        attempt_id=value.attempt_id,
        level=int(value.level),
        key=value.key,
        seq=value.sequence,
    )
    if value.timestamp is not None:
        result.timestamp.CopyFrom(timestamp_to_proto(value.timestamp))
    return result


def _job_summary_to_proto(value: JobSummary) -> resource_pb2.JobSummary:
    result = resource_pb2.JobSummary(
        identity=_job_identity_to_proto(value.identity),
        owner_id=value.owner_id,
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        num_tasks=value.num_tasks,
        submitted_at=timestamp_to_proto(value.submitted_at),
        error_message=value.error_message,
        pending_reason=value.pending_reason,
        resources=resource_spec_to_proto(value.resources),
    )
    if value.exit_code is not None:
        result.exit_code = value.exit_code
    if value.parent is not None:
        result.parent.CopyFrom(_job_identity_to_proto(value.parent))
    if value.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(value.started_at))
    if value.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(value.finished_at))
    return result


def _task_summary_to_proto(value: TaskSummary) -> resource_pb2.TaskSummary:
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


def _attempt_summary_to_proto(value: AttemptSummary) -> resource_pb2.AttemptSummary:
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


def _task_detail_to_proto(value: TaskDetail) -> resource_pb2.TaskDetail:
    return resource_pb2.TaskDetail(
        summary=_task_summary_to_proto(value.summary),
        attempts=[_attempt_summary_to_proto(item) for item in value.attempts],
        source_statuses=[_source_status_to_proto(status) for status in value.source_statuses],
        root_cause_highlights=value.root_cause_highlights,
    )


def _activity_entry_to_proto(value: ActivityEntry) -> resource_pb2.ActivityEntry:
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


def _endpoint_summary_to_proto(value: EndpointSummary) -> resource_pb2.EndpointSummary:
    result = resource_pb2.EndpointSummary(
        key=_resource_key_to_proto(value.key),
        endpoint_id=value.endpoint_id,
        name=value.name,
        execution_cluster_id=value.execution_cluster_id,
        access=endpoint_access_to_proto(value.access),
    )
    if value.task is not None:
        result.task.CopyFrom(_resource_key_to_proto(value.task))
    if value.lease_deadline is not None:
        result.lease_deadline.CopyFrom(timestamp_to_proto(value.lease_deadline))
    return result


def _endpoint_detail_to_proto(value: EndpointDetail) -> resource_pb2.EndpointDetail:
    return resource_pb2.EndpointDetail(
        summary=_endpoint_summary_to_proto(value.summary),
        address=value.address,
        metadata=value.metadata,
    )


def _action_principal(resource_id: str) -> str:
    owner = JobName.from_wire(resource_id.rpartition(":")[0] or resource_id).user
    identity = get_verified_identity()
    if identity is None:
        return ANONYMOUS_ADMIN.user_id
    if identity.role in {"admin", DASHBOARD_ROLE}:
        return identity.user_id
    return authorize_resource_owner(owner).user_id


def _resource_principal(resources: Controller, resource_id: str) -> str:
    identity = get_verified_identity()
    if identity is None or identity.role != FEDERATION_PEER_ROLE:
        return _action_principal(resource_id)
    root_job = JobName.from_wire(resource_id.rpartition(":")[0] or resource_id).root_job
    if resources.received_job_from_peer(root_job, identity.user_id):
        return root_job.user
    raise ConnectError(Code.PERMISSION_DENIED, f"Peer {identity.user_id!r} did not federate job {root_job}")


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


def _authorize_endpoint_owner(detail: EndpointDetail) -> None:
    if detail.summary.task is not None:
        _authorize_key_owner(detail.summary.task)
        return
    identity = get_verified_identity()
    if identity is not None and identity.role not in {"admin", DASHBOARD_ROLE}:
        raise ConnectError(Code.PERMISSION_DENIED, "System Endpoints require administrator access")


def _authorize_endpoint_read(detail: EndpointDetail) -> None:
    if detail.summary.task is not None:
        _authorize_key_owner(detail.summary.task)
        return
    identity = get_verified_identity()
    if identity is not None and identity.role == WORKER_ROLE:
        return
    _authorize_endpoint_owner(detail)


class _TypedResourceOperations:
    """Type-specific behavior used by generic registrations and legacy adapters."""

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def submit_job(self, request: resource_pb2.SubmitJobRequest, _ctx: RequestContext) -> resource_pb2.SubmitJobResponse:
        try:
            identity = self._resources.submit_job(
                job_spec_from_proto(request.spec),
                request.bundle_blob,
                enforce_client_freshness=_ctx is not None,
            )
        except (InvalidResourceKey, InvalidResourceRequest, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourcePermissionDenied as exc:
            raise ConnectError(Code.PERMISSION_DENIED, str(exc)) from exc
        except ResourcePreconditionFailed as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ResourceConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        except ResourceExhausted as exc:
            raise ConnectError(Code.RESOURCE_EXHAUSTED, str(exc)) from exc
        return resource_pb2.SubmitJobResponse(job=_job_identity_to_proto(identity))

    def list_jobs(self, request: resource_pb2.ListJobsRequest, _ctx: RequestContext) -> resource_pb2.ListJobsResponse:
        query = request.query
        try:
            owner_id = _authorized_owner(query.owner_id or None)
            page = self._resources.list_jobs(
                JobQuery(
                    resource_id=query.resource_id or None,
                    owner_id=owner_id,
                    parent=_resource_key_from_proto(query.parent) if query.HasField("parent") else None,
                    job_id_prefix=query.job_id_prefix or None,
                    states=frozenset(JobState(state) for state in query.states),
                    backend_id=query.backend_id or None,
                    execution_cluster_id=query.execution_cluster_id or None,
                    top_level_only=query.top_level_only,
                    page_size=query.page.page_size or _DEFAULT_JOB_PAGE_SIZE,
                    page_token=query.page.page_token or None,
                )
            )
        except (InvalidPageToken, InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return resource_pb2.ListJobsResponse(
            jobs=[_job_summary_to_proto(item) for item in page.items],
            page=_page_info(page.next_page_token, page.source_statuses),
        )

    def list_users(
        self, _request: resource_pb2.ListUsersRequest, _ctx: RequestContext
    ) -> resource_pb2.ListUsersResponse:
        owner_id = _authorized_owner()
        users = self._resources.list_users()
        if owner_id is not None:
            users = tuple(user for user in users if user.user_id == owner_id)
        return resource_pb2.ListUsersResponse(users=[_user_summary_to_proto(user) for user in users])

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
                summary=_job_summary_to_proto(detail.summary), spec=redacted_job_spec_to_proto(detail.spec)
            )
        )

    def get_job_state(
        self, request: resource_pb2.GetJobStateRequest, _ctx: RequestContext
    ) -> resource_pb2.GetJobStateResponse:
        try:
            identity = _job_identity_from_proto(request.job)
            _authorize_key_owner(identity.key)
            state = self._resources.job_state(identity)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        return resource_pb2.GetJobStateResponse(state=state)

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
                    states=frozenset(TaskState(state) for state in query.states),
                    backend_id=query.backend_id or None,
                    authority_cluster_id=query.authority_cluster_id or None,
                    execution_cluster_id=query.execution_cluster_id or None,
                    page_size=query.page.page_size or _DEFAULT_RESOURCE_PAGE_SIZE,
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
        return resource_pb2.DescribeTaskResponse(task=_task_detail_to_proto(detail))

    def batch_describe_tasks(
        self, request: resource_pb2.BatchDescribeTasksRequest, _ctx: RequestContext
    ) -> resource_pb2.BatchDescribeTasksResponse:
        try:
            keys = tuple(_resource_key_from_proto(item) for item in request.tasks)
            for key in keys:
                _authorize_key_owner(key)
            details = self._resources.describe_tasks(keys)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return resource_pb2.BatchDescribeTasksResponse(tasks=[_task_detail_to_proto(item) for item in details])

    def describe_attempt(
        self, request: resource_pb2.DescribeAttemptRequest, _ctx: RequestContext
    ) -> resource_pb2.DescribeAttemptResponse:
        try:
            locator = attempt_locator_from_proto(request.attempt)
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
            health = frozenset(node_health_from_proto(value) for value in query.health)
            page = self._resources.list_nodes(
                NodeQuery(
                    backend_id=query.backend_id or None,
                    contains=query.contains or None,
                    health=health,
                    page_size=query.page.page_size or _DEFAULT_RESOURCE_PAGE_SIZE,
                    page_token=query.page.page_token or None,
                )
            )
        except (InvalidPageToken, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return resource_pb2.ListNodesResponse(
            nodes=[_node_summary_to_proto(item) for item in page.items],
            page=_page_info(page.next_page_token, page.source_statuses),
        )

    def describe_node(
        self, request: resource_pb2.DescribeNodeRequest, _ctx: RequestContext
    ) -> resource_pb2.DescribeNodeResponse:
        try:
            detail = self._resources.describe_node(node_locator_from_proto(request.node))
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
            bootstrap_logs=detail.bootstrap_logs or "",
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
                    page_size=query.page.page_size or _DEFAULT_RESOURCE_PAGE_SIZE,
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
            detail = self._resources.describe_slice(slice_locator_from_proto(request.slice))
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

    def get_capacity_status(
        self, _request: resource_pb2.GetCapacityStatusRequest, _ctx: RequestContext
    ) -> resource_pb2.GetCapacityStatusResponse:
        status = self._resources.capacity_status()
        return resource_pb2.GetCapacityStatusResponse(
            backends=[_capacity_backend_to_proto(item) for item in status.backends],
            peers=[
                resource_pb2.CapacityPeer(
                    peer_id=peer.peer_id,
                    controller_address=peer.controller_address,
                    reachable=peer.reachable,
                    last_contact_ms=peer.last_contact_ms,
                    active_federated_jobs=peer.active_federated_jobs,
                    backends=[_capacity_peer_backend_to_proto(item) for item in peer.backends],
                )
                for peer in status.peers
            ],
            running_placements=[
                resource_pb2.CapacityRunningPlacement(
                    backend_id=item.backend_id,
                    worker_id=item.worker_id,
                    job_id=item.job_id,
                    user_id=item.user_id,
                    task_count=item.task_count,
                )
                for item in status.running_placements
            ],
            unroutable_job_count=status.unroutable_job_count,
            unroutable_jobs=[
                resource_pb2.CapacityUnroutableJob(job_id=item.job_id, reason=item.reason)
                for item in status.unroutable_jobs
            ],
            source_statuses=[_source_status_to_proto(item) for item in status.source_statuses],
        )

    def list_endpoints(
        self, request: resource_pb2.ListEndpointsRequest, _ctx: RequestContext
    ) -> resource_pb2.ListEndpointsResponse:
        query = request.query
        try:
            task = _resource_key_from_proto(query.task) if query.HasField("task") else None
            identity = get_verified_identity()
            worker_system_lookup = identity is not None and identity.role == WORKER_ROLE
            if worker_system_lookup and task is not None:
                raise ConnectError(Code.PERMISSION_DENIED, "Workers may only resolve system Endpoints")
            if task is not None:
                _authorize_key_owner(task)
            page = self._resources.list_endpoints(
                EndpointQuery(
                    name_prefix=query.name_prefix or None,
                    task=task,
                    owner_id=None if worker_system_lookup else _authorized_owner(),
                    page_size=query.page.page_size or _DEFAULT_RESOURCE_PAGE_SIZE,
                    page_token=query.page.page_token or None,
                    system_only=worker_system_lookup,
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
            _authorize_endpoint_read(detail)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return resource_pb2.DescribeEndpointResponse(endpoint=_endpoint_detail_to_proto(detail))

    def batch_describe_endpoints(
        self,
        request: resource_pb2.BatchDescribeEndpointsRequest,
        _ctx: RequestContext,
    ) -> resource_pb2.BatchDescribeEndpointsResponse:
        try:
            details = self._resources.describe_endpoints(
                tuple(_resource_key_from_proto(key) for key in request.endpoints)
            )
            for detail in details:
                _authorize_endpoint_read(detail)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return resource_pb2.BatchDescribeEndpointsResponse(
            endpoints=[_endpoint_detail_to_proto(detail) for detail in details]
        )

    def mint_endpoint_token(
        self, request: resource_pb2.MintEndpointTokenRequest, _ctx: RequestContext
    ) -> resource_pb2.MintEndpointTokenResponse:
        try:
            key = _resource_key_from_proto(request.endpoint)
            _authorize_endpoint_owner(self._resources.describe_endpoint(key))
            token = self._resources.mint_endpoint_token(
                key,
                duration_from_proto(request.ttl) if request.HasField("ttl") else None,
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourcePermissionDenied as exc:
            raise ConnectError(Code.PERMISSION_DENIED, str(exc)) from exc
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
                    page_size=request.query.page.page_size or _DEFAULT_ACTIVITY_PAGE_SIZE,
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
                    minimum_level=LogLevel(request.query.minimum_level),
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
            entries=[_log_entry_to_proto(entry) for entry in page.entries],
            next_cursor=page.next_cursor,
            source_statuses=[_source_status_to_proto(status) for status in page.source_statuses],
        )

    def cancel_job(
        self,
        request: resource_pb2.CancelJobRequest,
        _ctx: RequestContext,
        *,
        reason: str = "Cancelled through the resource API",
    ) -> resource_pb2.ActionResponse:
        try:
            identity = _job_identity_from_proto(request.job)
            receipt = self._resources.cancel_job(
                identity,
                idempotency_key=request.idempotency_key,
                reason=reason,
                principal_id=_resource_principal(self._resources, identity.key.resource_id),
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ActionIdempotencyConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        except PeerCallError as exc:
            raise peer_connect_error(exc) from exc
        return resource_pb2.ActionResponse(receipt=_action_receipt_to_proto(receipt))

    def retry_task(
        self,
        request: resource_pb2.RetryTaskRequest,
        _ctx: RequestContext,
        *,
        reason: str = "Requested through the resource API",
    ) -> resource_pb2.ActionResponse:
        try:
            identity = _task_identity_from_proto(request.task)
            receipt = self._resources.retry_task(
                identity,
                expected_attempt_uid=request.expected_attempt_uid,
                idempotency_key=request.idempotency_key,
                reason=reason,
                principal_id=_resource_principal(self._resources, identity.key.resource_id),
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except (ResourceReplaced, ActionPolicyRejected) as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ActionIdempotencyConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        except PeerCallError as exc:
            raise peer_connect_error(exc) from exc
        return resource_pb2.ActionResponse(receipt=_action_receipt_to_proto(receipt))

    def terminate_attempt(
        self,
        request: resource_pb2.TerminateAttemptRequest,
        _ctx: RequestContext,
        *,
        reason: str = "Requested through the resource API",
    ) -> resource_pb2.ActionResponse:
        try:
            identity = _attempt_identity_from_proto(request.attempt)
            receipt = self._resources.terminate_attempt(
                identity,
                idempotency_key=request.idempotency_key,
                reason=reason,
                principal_id=_resource_principal(self._resources, identity.task.resource_id),
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except (ResourceReplaced, ActionPolicyRejected) as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ActionIdempotencyConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        except PeerCallError as exc:
            raise peer_connect_error(exc) from exc
        return resource_pb2.ActionResponse(receipt=_action_receipt_to_proto(receipt))

    def fail_attempt(
        self,
        identity: resource_pb2.AttemptIdentity,
        *,
        idempotency_key: str,
        reason: str,
    ) -> resource_pb2.ActionResponse:
        try:
            native_identity = _attempt_identity_from_proto(identity)
            receipt = self._resources.fail_attempt(
                native_identity,
                idempotency_key=idempotency_key,
                reason=reason,
                principal_id=_resource_principal(self._resources, native_identity.task.resource_id),
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except (ResourceReplaced, ActionPolicyRejected) as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ActionIdempotencyConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        except PeerCallError as exc:
            raise peer_connect_error(exc) from exc
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

    def replay_action(
        self,
        *,
        resource_id: str,
        idempotency_key: str,
        kind: ActionKind,
        reason: str,
    ) -> resource_pb2.ActionResponse | None:
        try:
            receipt = self._resources.replay_action(
                principal_id=_resource_principal(self._resources, resource_id),
                idempotency_key=idempotency_key,
                kind=kind,
                reason=reason,
            )
        except ActionIdempotencyConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        if receipt is None:
            return None
        return resource_pb2.ActionResponse(receipt=_action_receipt_to_proto(receipt))

    def resolve_job_update(self, ref: resource_pb2.ResourceRef) -> resource_pb2.JobIdentity:
        try:
            _resource_principal(self._resources, ref.id)
            detail = self._resources.describe_job(ResourceKey(ref.authority_cluster_id, ResourceKind.JOB, ref.id))
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return _job_identity_to_proto(detail.summary.identity)

    def resolve_task_update(self, ref: resource_pb2.ResourceRef) -> resource_pb2.TaskSummary:
        try:
            _resource_principal(self._resources, ref.id)
            detail = self._resources.describe_task(ResourceKey(ref.authority_cluster_id, ResourceKind.TASK, ref.id))
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return _task_summary_to_proto(detail.summary)

    def resolve_attempt_update(self, ref: resource_pb2.ResourceRef) -> resource_pb2.AttemptIdentity:
        try:
            _resource_principal(self._resources, ref.id)
            detail = self._resources.describe_attempt(attempt_locator_from_proto(_attempt_locator(ref)))
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        return _attempt_identity_to_proto(detail.summary.identity)

    def exec_attempt(
        self, request: resource_pb2.ExecAttemptRequest, _ctx: RequestContext
    ) -> resource_pb2.ExecAttemptResponse:
        try:
            identity = _attempt_identity_from_proto(request.attempt)
            _resource_principal(self._resources, identity.task.resource_id)
            response = self._resources.exec_attempt(
                identity,
                tuple(request.command),
                duration_from_proto(request.timeout) if request.HasField("timeout") else None,
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ResourceSourceUnavailable as exc:
            raise ConnectError(Code.UNAVAILABLE, str(exc)) from exc
        except PeerCallError as exc:
            raise peer_connect_error(exc) from exc
        return resource_pb2.ExecAttemptResponse(
            exit_code=response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
            error_message=response.error_message,
        )

    def profile_attempt(
        self, request: resource_pb2.ProfileAttemptRequest, _ctx: RequestContext
    ) -> resource_pb2.ProfileAttemptResponse:
        try:
            identity = _attempt_identity_from_proto(request.attempt)
            _resource_principal(self._resources, identity.task.resource_id)
            response = self._resources.profile_attempt(
                identity,
                profile_configuration_from_proto(request.profile),
                duration_from_proto(request.duration) if request.HasField("duration") else None,
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ResourceSourceUnavailable as exc:
            raise ConnectError(Code.UNAVAILABLE, str(exc)) from exc
        except PeerCallError as exc:
            raise peer_connect_error(exc) from exc
        return resource_pb2.ProfileAttemptResponse(
            profile_data=response.profile_data,
            error_message=response.error_message,
        )


_MessageT = TypeVar("_MessageT", bound=Message)


def _type_url(message_type: type[Message]) -> str:
    return f"type.googleapis.com/{message_type.DESCRIPTOR.full_name}"


def _pack(value: Message) -> any_pb2.Any:
    packed = any_pb2.Any()
    packed.Pack(value)
    return packed


def _unpack(value: any_pb2.Any, message_type: type[_MessageT]) -> _MessageT:
    if not value.Is(message_type.DESCRIPTOR):
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            f"expected {_type_url(message_type)}, got {value.type_url or 'an empty body'}",
        )
    result = message_type()
    if not value.Unpack(result):
        raise ConnectError(Code.INVALID_ARGUMENT, f"invalid {_type_url(message_type)} body")
    return result


def _resource_ref(
    authority_cluster_id: str,
    resource_type: str,
    resource_id: str,
    uid: str | None = None,
) -> resource_pb2.ResourceRef:
    result = resource_pb2.ResourceRef(
        authority_cluster_id=authority_cluster_id,
        type=resource_type,
        id=resource_id,
    )
    if uid is not None:
        result.uid = uid
    return result


def _resource(ref: resource_pb2.ResourceRef, body: Message) -> resource_pb2.Resource:
    return resource_pb2.Resource(ref=ref, body=_pack(body))


def _require_ref_type(ref: resource_pb2.ResourceRef, expected: str) -> None:
    if ref.type != expected:
        raise ConnectError(Code.INVALID_ARGUMENT, f"expected resource type {expected!r}, got {ref.type!r}")
    if not ref.authority_cluster_id or not ref.id:
        raise ConnectError(Code.INVALID_ARGUMENT, "resource authority and id are required")


def _require_exact_uid(ref: resource_pb2.ResourceRef, actual_uid: str) -> None:
    if ref.HasField("uid") and ref.uid != actual_uid:
        raise ConnectError(Code.FAILED_PRECONDITION, f"resource {ref.id!r} was replaced")


def _legacy_key(ref: resource_pb2.ResourceRef, kind: int) -> resource_pb2.ResourceKey:
    return resource_pb2.ResourceKey(
        cluster_id=ref.authority_cluster_id,
        kind=kind,
        resource_id=ref.id,
    )


def _backend_ref_id(backend_id: str, resource_id: str) -> str:
    if ":" in backend_id:
        raise ValueError("backend IDs used in ResourceRef must not contain ':'")
    return f"{backend_id}:{resource_id}"


def _parse_backend_ref_id(value: str) -> tuple[str, str]:
    backend_id, separator, resource_id = value.partition(":")
    if not separator or not backend_id or not resource_id:
        raise ConnectError(Code.INVALID_ARGUMENT, "backend resource id must be '<backend>:<id>'")
    return backend_id, resource_id


def _attempt_locator(ref: resource_pb2.ResourceRef) -> resource_pb2.AttemptLocator:
    task_id, separator, attempt = ref.id.rpartition(":")
    if not separator or not task_id:
        raise ConnectError(Code.INVALID_ARGUMENT, "Attempt id must be '<task>:<number|current>'")
    locator = resource_pb2.AttemptLocator(
        task=_legacy_key(
            _resource_ref(ref.authority_cluster_id, TASK, task_id),
            resource_pb2.RESOURCE_KIND_TASK,
        )
    )
    if attempt != "current":
        if not attempt.isdecimal() or str(int(attempt)) != attempt:
            raise ConnectError(Code.INVALID_ARGUMENT, "Attempt number must be canonical and non-negative")
        locator.attempt_number = int(attempt)
    return locator


def _job_ref(identity: resource_pb2.JobIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(identity.key.cluster_id, JOB, identity.key.resource_id, identity.job_uid)


def _task_ref(identity: resource_pb2.TaskIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(identity.key.cluster_id, TASK, identity.key.resource_id, identity.task_uid)


def _attempt_ref(identity: resource_pb2.AttemptIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(
        identity.task.cluster_id,
        ATTEMPT,
        f"{identity.task.resource_id}:{identity.attempt_number}",
        identity.attempt_uid,
    )


def _node_ref(identity: resource_pb2.NodeIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(
        identity.key.cluster_id,
        NODE,
        _backend_ref_id(identity.backend_id, identity.key.resource_id),
        identity.node_uid,
    )


def _slice_ref(identity: resource_pb2.SliceIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(
        identity.key.cluster_id,
        SLICE,
        _backend_ref_id(identity.backend_id, identity.key.resource_id),
        identity.slice_uid,
    )


def _endpoint_ref(summary: resource_pb2.EndpointSummary) -> resource_pb2.ResourceRef:
    return _resource_ref(summary.key.cluster_id, ENDPOINT, summary.key.resource_id)


def _operation(
    request_id: str,
    *,
    verb: str,
    requested_ref: resource_pb2.ResourceRef | None,
    resolved_ref: resource_pb2.ResourceRef | None,
    result: Message,
    requested_state: int = resource_pb2.REQUESTED_RESOURCE_STATE_UNSPECIFIED,
    phase: int = resource_pb2.OPERATION_PHASE_VERIFIED,
) -> resource_pb2.Operation:
    operation_id = request_id or uuid.uuid4().hex
    cluster_id = "system"
    if resolved_ref is not None:
        cluster_id = resolved_ref.authority_cluster_id
    elif requested_ref is not None:
        cluster_id = requested_ref.authority_cluster_id
    now = timestamp_to_proto(Timestamp.now())
    operation = resource_pb2.Operation(
        ref=_resource_ref(cluster_id, OPERATION, operation_id, operation_id),
        phase=phase,
        verb=verb,
        requested_state=requested_state,
        result=_pack(result),
        accepted_at=now,
        applied_at=now,
    )
    if requested_ref is not None:
        operation.requested_ref.CopyFrom(requested_ref)
    if resolved_ref is not None:
        operation.resolved_ref.CopyFrom(resolved_ref)
    if phase in {resource_pb2.OPERATION_PHASE_VERIFIED, resource_pb2.OPERATION_PHASE_FAILED}:
        operation.completed_at.CopyFrom(now)
    return operation


def _action_target_ref(receipt: resource_pb2.ActionReceipt) -> resource_pb2.ResourceRef:
    if receipt.target.kind == resource_pb2.RESOURCE_KIND_JOB:
        return _resource_ref(
            receipt.target.cluster_id,
            JOB,
            receipt.target.resource_id,
            receipt.expected_target_uid,
        )
    if receipt.target.kind == resource_pb2.RESOURCE_KIND_TASK:
        return _resource_ref(
            receipt.target.cluster_id,
            TASK,
            receipt.target.resource_id,
            receipt.expected_target_uid,
        )
    return _resource_ref(
        receipt.target.cluster_id,
        ATTEMPT,
        receipt.target.resource_id,
        receipt.expected_attempt_uid,
    )


def _selected_ref_from_action(
    requested_ref: resource_pb2.ResourceRef,
    receipt: resource_pb2.ActionReceipt,
) -> resource_pb2.ResourceRef:
    if requested_ref.type == JOB:
        return _resource_ref(
            receipt.target.cluster_id,
            JOB,
            requested_ref.id,
            receipt.expected_target_uid,
        )
    if requested_ref.type == TASK:
        task_id = receipt.target.resource_id
        if receipt.target.kind == resource_pb2.RESOURCE_KIND_ATTEMPT:
            task_id, _, _ = task_id.rpartition(":")
        return _resource_ref(
            receipt.target.cluster_id,
            TASK,
            task_id,
            receipt.expected_target_uid,
        )
    if requested_ref.type == ATTEMPT:
        if not receipt.HasField("expected_attempt_number") or not receipt.expected_attempt_uid:
            raise ConnectError(Code.INTERNAL, "Attempt action did not record its exact target")
        task_id = receipt.target.resource_id
        if receipt.target.kind == resource_pb2.RESOURCE_KIND_ATTEMPT:
            task_id, _, _ = task_id.rpartition(":")
        return _resource_ref(
            receipt.target.cluster_id,
            ATTEMPT,
            f"{task_id}:{receipt.expected_attempt_number}",
            receipt.expected_attempt_uid,
        )
    raise ConnectError(Code.INTERNAL, f"action cannot resolve selected resource type {requested_ref.type!r}")


def _operation_from_action(
    requested_ref: resource_pb2.ResourceRef,
    resolved_ref: resource_pb2.ResourceRef,
    requested_state: int,
    response: resource_pb2.ActionResponse,
) -> resource_pb2.Operation:
    receipt = response.receipt
    state_to_phase = {
        resource_pb2.ACTION_STATE_ACCEPTED: resource_pb2.OPERATION_PHASE_ACCEPTED,
        resource_pb2.ACTION_STATE_VERIFYING: resource_pb2.OPERATION_PHASE_VERIFYING,
        resource_pb2.ACTION_STATE_SUCCEEDED: resource_pb2.OPERATION_PHASE_APPLIED,
        resource_pb2.ACTION_STATE_FAILED: resource_pb2.OPERATION_PHASE_FAILED,
    }
    operation = _operation(
        receipt.action_id,
        verb="update",
        requested_ref=requested_ref,
        resolved_ref=resolved_ref,
        requested_state=requested_state,
        result=receipt,
        phase=state_to_phase[receipt.state],
    )
    if receipt.HasField("expected_attempt_number"):
        task_id = receipt.target.resource_id
        if receipt.target.kind == resource_pb2.RESOURCE_KIND_ATTEMPT:
            task_id, _, _ = task_id.rpartition(":")
        operation.affected.append(
            _resource_ref(
                receipt.target.cluster_id,
                ATTEMPT,
                f"{task_id}:{receipt.expected_attempt_number}",
                receipt.expected_attempt_uid,
            )
        )
    operation.ref.CopyFrom(_resource_ref(receipt.target.cluster_id, OPERATION, receipt.action_id, receipt.action_id))
    operation.accepted_at.CopyFrom(receipt.created_at)
    operation.applied_at.CopyFrom(receipt.updated_at)
    if operation.phase in {
        resource_pb2.OPERATION_PHASE_VERIFIED,
        resource_pb2.OPERATION_PHASE_FAILED,
    } and receipt.HasField("completed_at"):
        operation.completed_at.CopyFrom(receipt.completed_at)
    return operation


class _GenericResourceHandlers:
    def __init__(self, typed: _TypedResourceOperations) -> None:
        self._typed = typed

    def _replay_update(
        self,
        request: resource_pb2.UpdateResourceRequest,
        *,
        kind: ActionKind,
        reason: str,
    ) -> resource_pb2.Operation | None:
        response = self._typed.replay_action(
            resource_id=request.ref.id,
            idempotency_key=request.mutation.request_id,
            kind=kind,
            reason=reason,
        )
        if response is None:
            return None
        return _operation_from_action(
            request.ref,
            _selected_ref_from_action(request.ref, response.receipt),
            request.update.new_state,
            response,
        )

    def get_job(
        self, request: resource_pb2.GetResourceRequest, context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, JOB)
        response = self._typed.describe_job(
            resource_pb2.DescribeJobRequest(job=_legacy_key(request.ref, resource_pb2.RESOURCE_KIND_JOB)),
            context,
        )
        _require_exact_uid(request.ref, response.job.summary.identity.job_uid)
        body: Message = response.job.summary if request.view == resource_pb2.RESOURCE_VIEW_BASIC else response.job
        return resource_pb2.GetResourceResponse(resource=_resource(_job_ref(response.job.summary.identity), body))

    def list_jobs(
        self, request: resource_pb2.ListResourcesRequest, context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_pb2.JobQuery)
        query.page.CopyFrom(
            resource_pb2.PageRequest(
                page_size=request.page_size or query.page.page_size,
                page_token=request.page_token or query.page.page_token,
            )
        )
        response = self._typed.list_jobs(resource_pb2.ListJobsRequest(query=query), context)
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_job_ref(item.identity), item) for item in response.jobs],
            next_page_token=response.page.next_page_token,
            source_statuses=response.page.source_statuses,
        )

    def create_job(self, request: resource_pb2.CreateResourceRequest, context: RequestContext) -> resource_pb2.Operation:
        body = _unpack(request.body, resource_pb2.SubmitJobRequest)
        response = self._typed.submit_job(body, context)
        resolved = _job_ref(response.job)
        requested = _resource_ref(response.job.key.cluster_id, JOB, request.id or response.job.key.resource_id)
        return _operation(
            request.mutation.request_id,
            verb="create",
            requested_ref=requested,
            resolved_ref=resolved,
            result=response,
        )

    def update_job(self, request: resource_pb2.UpdateResourceRequest, context: RequestContext) -> resource_pb2.Operation:
        _require_ref_type(request.ref, JOB)
        if request.update.new_state != resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED:
            raise ConnectError(Code.INVALID_ARGUMENT, "Job supports only CANCELLED updates")
        reason = request.mutation.reason or "Cancelled through the resource API"
        replay = self._replay_update(request, kind=ActionKind.CANCEL_JOB, reason=reason)
        if replay is not None:
            return replay
        identity = self._typed.resolve_job_update(request.ref)
        _require_exact_uid(request.ref, identity.job_uid)
        response = self._typed.cancel_job(
            resource_pb2.CancelJobRequest(
                job=identity,
                idempotency_key=request.mutation.request_id,
            ),
            context,
            reason=reason,
        )
        return _operation_from_action(
            request.ref,
            _job_ref(identity),
            request.update.new_state,
            response,
        )

    def get_task(
        self, request: resource_pb2.GetResourceRequest, context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, TASK)
        response = self._typed.describe_task(
            resource_pb2.DescribeTaskRequest(task=_legacy_key(request.ref, resource_pb2.RESOURCE_KIND_TASK)),
            context,
        )
        _require_exact_uid(request.ref, response.task.summary.identity.task_uid)
        body: Message = response.task.summary if request.view == resource_pb2.RESOURCE_VIEW_BASIC else response.task
        return resource_pb2.GetResourceResponse(
            resource=_resource(_task_ref(response.task.summary.identity), body),
            source_statuses=response.task.source_statuses,
        )

    def list_tasks(
        self, request: resource_pb2.ListResourcesRequest, context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_pb2.TaskQuery)
        query.page.CopyFrom(
            resource_pb2.PageRequest(
                page_size=request.page_size or query.page.page_size,
                page_token=request.page_token or query.page.page_token,
            )
        )
        response = self._typed.list_tasks(resource_pb2.ListTasksRequest(query=query), context)
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_task_ref(item.identity), item) for item in response.tasks],
            next_page_token=response.page.next_page_token,
            source_statuses=response.page.source_statuses,
        )

    def batch_get_tasks(
        self, request: resource_pb2.BatchGetResourcesRequest, context: RequestContext
    ) -> resource_pb2.BatchGetResourcesResponse:
        refs = tuple(request.refs)
        for ref in refs:
            _require_ref_type(ref, TASK)
        response = self._typed.batch_describe_tasks(
            resource_pb2.BatchDescribeTasksRequest(
                tasks=[_legacy_key(ref, resource_pb2.RESOURCE_KIND_TASK) for ref in refs]
            ),
            context,
        )
        results: list[resource_pb2.BatchGetResourceResult] = []
        for ref, detail in zip(refs, response.tasks, strict=True):
            _require_exact_uid(ref, detail.summary.identity.task_uid)
            body: Message = detail.summary if request.view == resource_pb2.RESOURCE_VIEW_BASIC else detail
            results.append(
                resource_pb2.BatchGetResourceResult(resource=_resource(_task_ref(detail.summary.identity), body))
            )
        return resource_pb2.BatchGetResourcesResponse(results=results)

    def update_task(
        self, request: resource_pb2.UpdateResourceRequest, context: RequestContext
    ) -> resource_pb2.Operation:
        _require_ref_type(request.ref, TASK)
        if request.update.new_state in {
            resource_pb2.REQUESTED_RESOURCE_STATE_PENDING,
            resource_pb2.REQUESTED_RESOURCE_STATE_PREEMPTED,
        }:
            kind = ActionKind.RETRY_TASK
            reason = request.mutation.reason or "Requested through the resource API"
        elif request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED:
            kind = ActionKind.TERMINATE_ATTEMPT
            reason = request.mutation.reason or "Requested through the resource API"
        elif request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_FAILED:
            kind = ActionKind.FAIL_ATTEMPT
            reason = request.mutation.reason or "Failed through the resource API"
        else:
            raise ConnectError(Code.INVALID_ARGUMENT, "Task update requires a supported new_state")
        replay = self._replay_update(request, kind=kind, reason=reason)
        if replay is not None:
            return replay
        summary = self._typed.resolve_task_update(request.ref)
        _require_exact_uid(request.ref, summary.identity.task_uid)
        attempt = summary.current_attempt
        if not attempt.attempt_uid:
            raise ConnectError(Code.FAILED_PRECONDITION, "Task has no current Attempt")
        if request.update.HasField("patch"):
            condition = _unpack(request.update.patch, resource_pb2.RetryTaskRequest)
            if condition.expected_attempt_uid and condition.expected_attempt_uid != attempt.attempt_uid:
                raise ConnectError(Code.FAILED_PRECONDITION, "Task's current Attempt was replaced")
        if request.update.new_state in {
            resource_pb2.REQUESTED_RESOURCE_STATE_PENDING,
            resource_pb2.REQUESTED_RESOURCE_STATE_PREEMPTED,
        }:
            response = self._typed.retry_task(
                resource_pb2.RetryTaskRequest(
                    task=summary.identity,
                    expected_attempt_uid=attempt.attempt_uid,
                    idempotency_key=request.mutation.request_id,
                ),
                context,
                reason=reason,
            )
        elif request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED:
            response = self._typed.terminate_attempt(
                resource_pb2.TerminateAttemptRequest(
                    attempt=attempt,
                    idempotency_key=request.mutation.request_id,
                ),
                context,
                reason=reason,
            )
        elif request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_FAILED:
            response = self._typed.fail_attempt(
                attempt,
                idempotency_key=request.mutation.request_id,
                reason=reason,
            )
        return _operation_from_action(
            request.ref,
            _task_ref(summary.identity),
            request.update.new_state,
            response,
        )

    def get_attempt(
        self, request: resource_pb2.GetResourceRequest, context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, ATTEMPT)
        response = self._typed.describe_attempt(
            resource_pb2.DescribeAttemptRequest(attempt=_attempt_locator(request.ref)),
            context,
        )
        _require_exact_uid(request.ref, response.attempt.summary.identity.attempt_uid)
        return resource_pb2.GetResourceResponse(
            resource=_resource(_attempt_ref(response.attempt.summary.identity), response.attempt),
            source_statuses=response.attempt.source_statuses,
        )

    def update_attempt(
        self, request: resource_pb2.UpdateResourceRequest, context: RequestContext
    ) -> resource_pb2.Operation:
        _require_ref_type(request.ref, ATTEMPT)
        if request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED:
            kind = ActionKind.TERMINATE_ATTEMPT
            reason = request.mutation.reason or "Requested through the resource API"
        elif request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_PREEMPTED:
            kind = ActionKind.RETRY_TASK
            reason = request.mutation.reason or "Requested through the resource API"
        elif request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_FAILED:
            kind = ActionKind.FAIL_ATTEMPT
            reason = request.mutation.reason or "Failed through the resource API"
        else:
            raise ConnectError(Code.INVALID_ARGUMENT, "Attempt supports PREEMPTED, FAILED, or CANCELLED")
        replay = self._replay_update(request, kind=kind, reason=reason)
        if replay is not None:
            return replay
        identity = self._typed.resolve_attempt_update(request.ref)
        _require_exact_uid(request.ref, identity.attempt_uid)
        if request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED:
            response = self._typed.terminate_attempt(
                resource_pb2.TerminateAttemptRequest(
                    attempt=identity,
                    idempotency_key=request.mutation.request_id,
                ),
                context,
                reason=reason,
            )
        elif request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_PREEMPTED:
            task = self._typed.resolve_task_update(
                _resource_ref(identity.task.cluster_id, TASK, identity.task.resource_id)
            ).identity
            response = self._typed.retry_task(
                resource_pb2.RetryTaskRequest(
                    task=task,
                    expected_attempt_uid=identity.attempt_uid,
                    idempotency_key=request.mutation.request_id,
                ),
                context,
                reason=reason,
            )
        elif request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_FAILED:
            response = self._typed.fail_attempt(
                identity,
                idempotency_key=request.mutation.request_id,
                reason=reason,
            )
        return _operation_from_action(
            request.ref,
            _attempt_ref(identity),
            request.update.new_state,
            response,
        )

    def get_node(
        self, request: resource_pb2.GetResourceRequest, context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, NODE)
        backend_id, node_id = _parse_backend_ref_id(request.ref.id)
        locator = resource_pb2.NodeLocator(
            key=resource_pb2.ResourceKey(
                cluster_id=request.ref.authority_cluster_id,
                kind=resource_pb2.RESOURCE_KIND_NODE,
                resource_id=node_id,
            ),
            backend_id=backend_id,
        )
        if request.ref.HasField("uid"):
            locator.node_uid = request.ref.uid
        response = self._typed.describe_node(resource_pb2.DescribeNodeRequest(node=locator), context)
        return resource_pb2.GetResourceResponse(
            resource=_resource(_node_ref(response.node.summary.identity), response.node),
            source_statuses=response.node.source_statuses,
        )

    def list_nodes(
        self, request: resource_pb2.ListResourcesRequest, context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_pb2.NodeQuery)
        query.page.CopyFrom(
            resource_pb2.PageRequest(
                page_size=request.page_size or query.page.page_size,
                page_token=request.page_token or query.page.page_token,
            )
        )
        response = self._typed.list_nodes(resource_pb2.ListNodesRequest(query=query), context)
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_node_ref(item.identity), item) for item in response.nodes],
            next_page_token=response.page.next_page_token,
            source_statuses=response.page.source_statuses,
        )

    def get_slice(
        self, request: resource_pb2.GetResourceRequest, context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, SLICE)
        backend_id, slice_id = _parse_backend_ref_id(request.ref.id)
        locator = resource_pb2.SliceLocator(
            key=resource_pb2.ResourceKey(
                cluster_id=request.ref.authority_cluster_id,
                kind=resource_pb2.RESOURCE_KIND_SLICE,
                resource_id=slice_id,
            ),
            backend_id=backend_id,
        )
        if request.ref.HasField("uid"):
            locator.slice_uid = request.ref.uid
        response = self._typed.describe_slice(resource_pb2.DescribeSliceRequest(slice=locator), context)
        return resource_pb2.GetResourceResponse(
            resource=_resource(_slice_ref(response.slice.summary.identity), response.slice),
            source_statuses=response.slice.source_statuses,
        )

    def list_slices(
        self, request: resource_pb2.ListResourcesRequest, context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_pb2.SliceQuery)
        query.page.CopyFrom(
            resource_pb2.PageRequest(
                page_size=request.page_size or query.page.page_size,
                page_token=request.page_token or query.page.page_token,
            )
        )
        response = self._typed.list_slices(resource_pb2.ListSlicesRequest(query=query), context)
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_slice_ref(item.identity), item) for item in response.slices],
            next_page_token=response.page.next_page_token,
            source_statuses=response.page.source_statuses,
        )

    def get_endpoint(
        self, request: resource_pb2.GetResourceRequest, context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, ENDPOINT)
        if request.ref.HasField("uid"):
            raise ConnectError(Code.FAILED_PRECONDITION, "Endpoint exact identity is not available")
        response = self._typed.describe_endpoint(
            resource_pb2.DescribeEndpointRequest(endpoint=_legacy_key(request.ref, resource_pb2.RESOURCE_KIND_ENDPOINT)),
            context,
        )
        return resource_pb2.GetResourceResponse(
            resource=_resource(_endpoint_ref(response.endpoint.summary), response.endpoint)
        )

    def list_endpoints(
        self, request: resource_pb2.ListResourcesRequest, context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_pb2.EndpointQuery)
        query.page.CopyFrom(
            resource_pb2.PageRequest(
                page_size=request.page_size or query.page.page_size,
                page_token=request.page_token or query.page.page_token,
            )
        )
        response = self._typed.list_endpoints(resource_pb2.ListEndpointsRequest(query=query), context)
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_endpoint_ref(item), item) for item in response.endpoints],
            next_page_token=response.page.next_page_token,
            source_statuses=response.page.source_statuses,
        )

    def batch_get_endpoints(
        self, request: resource_pb2.BatchGetResourcesRequest, context: RequestContext
    ) -> resource_pb2.BatchGetResourcesResponse:
        refs = tuple(request.refs)
        for ref in refs:
            _require_ref_type(ref, ENDPOINT)
            if ref.HasField("uid"):
                raise ConnectError(Code.FAILED_PRECONDITION, "Endpoint exact identity is not available")
        response = self._typed.batch_describe_endpoints(
            resource_pb2.BatchDescribeEndpointsRequest(
                endpoints=[_legacy_key(ref, resource_pb2.RESOURCE_KIND_ENDPOINT) for ref in refs]
            ),
            context,
        )
        return resource_pb2.BatchGetResourcesResponse(
            results=[
                resource_pb2.BatchGetResourceResult(resource=_resource(_endpoint_ref(detail.summary), detail))
                for detail in response.endpoints
            ]
        )

    def get_capacity(
        self, request: resource_pb2.GetResourceRequest, context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, CAPACITY)
        if request.ref.id != "capacity" or request.ref.HasField("uid"):
            raise ConnectError(Code.INVALID_ARGUMENT, "Capacity is the current-only 'capacity' resource")
        body = self._typed.get_capacity_status(resource_pb2.GetCapacityStatusRequest(), context)
        return resource_pb2.GetResourceResponse(
            resource=_resource(_resource_ref(request.ref.authority_cluster_id, CAPACITY, "capacity"), body),
            source_statuses=body.source_statuses,
        )

    def list_users(
        self, request: resource_pb2.ListResourcesRequest, context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        _unpack(request.query, resource_pb2.ListUsersRequest)
        response = self._typed.list_users(resource_pb2.ListUsersRequest(), context)
        return resource_pb2.ListResourcesResponse(
            resources=[
                _resource(
                    _resource_ref("system", USER_SUMMARY, user.user_id),
                    user,
                )
                for user in response.users
            ]
        )

    def list_activity(
        self, request: resource_pb2.ListResourcesRequest, context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_pb2.ActivityQuery)
        query.page.CopyFrom(
            resource_pb2.PageRequest(
                page_size=request.page_size or query.page.page_size,
                page_token=request.page_token or query.page.page_token,
            )
        )
        response = self._typed.list_activity(resource_pb2.ListActivityRequest(query=query), context)
        return resource_pb2.ListResourcesResponse(
            resources=[
                _resource(
                    _resource_ref(entry.target.cluster_id, ACTIVITY_ENTRY, entry.entry_id),
                    entry,
                )
                for entry in response.entries
            ],
            next_page_token=response.page.next_page_token,
            source_statuses=response.page.source_statuses,
        )

    def list_logs(
        self, request: resource_pb2.ListResourcesRequest, context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_pb2.FetchLogsRequest)
        if request.page_token:
            try:
                query.query.cursor = int(request.page_token)
            except ValueError as exc:
                raise ConnectError(Code.INVALID_ARGUMENT, "log page token must be an integer cursor") from exc
        if request.page_size:
            query.query.max_lines = request.page_size
        response = self._typed.fetch_logs(query, context)
        target_kind = query.target.WhichOneof("target")
        target = getattr(query.target, target_kind) if target_kind is not None else None
        if target is None:
            raise ConnectError(Code.INVALID_ARGUMENT, "log target is required")
        key = target.task if target_kind == "attempt" else target.key
        return resource_pb2.ListResourcesResponse(
            resources=[
                _resource(
                    _resource_ref(
                        key.cluster_id,
                        LOG_ENTRY,
                        f"{entry.key}:{entry.seq}",
                    ),
                    entry,
                )
                for entry in response.entries
            ],
            next_page_token=str(response.next_cursor),
            source_statuses=response.source_statuses,
        )

    def get_operation(
        self, request: resource_pb2.GetResourceRequest, context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, OPERATION)
        response = self._typed.get_action_receipt(
            resource_pb2.GetActionReceiptRequest(action_id=request.ref.id),
            context,
        )
        receipt = response.receipt
        _require_exact_uid(request.ref, receipt.action_id)
        requested_state = {
            resource_pb2.ACTION_KIND_CANCEL_JOB: resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED,
            resource_pb2.ACTION_KIND_RETRY_TASK: resource_pb2.REQUESTED_RESOURCE_STATE_PENDING,
            resource_pb2.ACTION_KIND_TERMINATE_ATTEMPT: resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED,
            resource_pb2.ACTION_KIND_FAIL_ATTEMPT: resource_pb2.REQUESTED_RESOURCE_STATE_FAILED,
        }[receipt.kind]
        operation = _operation_from_action(
            _action_target_ref(receipt),
            _action_target_ref(receipt),
            requested_state,
            response,
        )
        return resource_pb2.GetResourceResponse(
            resource=_resource(
                _resource_ref(receipt.target.cluster_id, OPERATION, receipt.action_id, receipt.action_id),
                operation,
            )
        )

    def create_exec_session(
        self, request: resource_pb2.CreateResourceRequest, context: RequestContext
    ) -> resource_pb2.Operation:
        _require_ref_type(request.parent, ATTEMPT)
        body = _unpack(request.body, resource_pb2.ExecAttemptRequest)
        response = self._typed.exec_attempt(body, context)
        return _operation(
            request.mutation.request_id,
            verb="create",
            requested_ref=request.parent,
            resolved_ref=request.parent,
            result=response,
        )

    def create_profile_capture(
        self, request: resource_pb2.CreateResourceRequest, context: RequestContext
    ) -> resource_pb2.Operation:
        _require_ref_type(request.parent, ATTEMPT)
        body = _unpack(request.body, resource_pb2.ProfileAttemptRequest)
        response = self._typed.profile_attempt(body, context)
        return _operation(
            request.mutation.request_id,
            verb="create",
            requested_ref=request.parent,
            resolved_ref=request.parent,
            result=response,
        )

    def create_endpoint_capability(
        self, request: resource_pb2.CreateResourceRequest, context: RequestContext
    ) -> resource_pb2.Operation:
        _require_ref_type(request.parent, ENDPOINT)
        body = _unpack(request.body, resource_pb2.MintEndpointTokenRequest)
        response = self._typed.mint_endpoint_token(body, context)
        return _operation(
            request.mutation.request_id,
            verb="create",
            requested_ref=request.parent,
            resolved_ref=request.parent,
            result=response,
        )


def resource_registrations(resources: Controller) -> tuple[ResourceRegistration, ...]:
    """Build the installed ResourceService behavior from controller-backed parts."""
    handlers = _GenericResourceHandlers(_TypedResourceOperations(resources))
    basic_full = (
        resource_pb2.RESOURCE_VIEW_BASIC,
        resource_pb2.RESOURCE_VIEW_FULL,
    )
    return (
        ResourceRegistration(
            JOB,
            views=basic_full,
            body_type_urls=(_type_url(resource_pb2.JobSummary), _type_url(resource_pb2.JobDetail)),
            create_type_urls=(_type_url(resource_pb2.SubmitJobRequest),),
            query_type_urls=(_type_url(resource_pb2.JobQuery),),
            features=("current-ref-v1", "exact-ref-v1"),
            get=handlers.get_job,
            list=handlers.list_jobs,
            create=handlers.create_job,
            update=handlers.update_job,
        ),
        ResourceRegistration(
            TASK,
            views=basic_full,
            body_type_urls=(_type_url(resource_pb2.TaskSummary), _type_url(resource_pb2.TaskDetail)),
            query_type_urls=(_type_url(resource_pb2.TaskQuery),),
            update_type_urls=(_type_url(resource_pb2.RetryTaskRequest),),
            features=("current-ref-v1", "exact-ref-v1"),
            get=handlers.get_task,
            batch_get=handlers.batch_get_tasks,
            list=handlers.list_tasks,
            update=handlers.update_task,
        ),
        ResourceRegistration(
            ATTEMPT,
            views=(resource_pb2.RESOURCE_VIEW_FULL,),
            body_type_urls=(_type_url(resource_pb2.AttemptDetail),),
            features=("current-ref-v1", "exact-ref-v1"),
            get=handlers.get_attempt,
            update=handlers.update_attempt,
        ),
        ResourceRegistration(
            NODE,
            views=basic_full,
            body_type_urls=(_type_url(resource_pb2.NodeSummary), _type_url(resource_pb2.NodeDetail)),
            query_type_urls=(_type_url(resource_pb2.NodeQuery),),
            features=("current-ref-v1", "exact-ref-v1"),
            get=handlers.get_node,
            list=handlers.list_nodes,
        ),
        ResourceRegistration(
            SLICE,
            views=basic_full,
            body_type_urls=(_type_url(resource_pb2.SliceSummary), _type_url(resource_pb2.SliceDetail)),
            query_type_urls=(_type_url(resource_pb2.SliceQuery),),
            features=("current-ref-v1", "exact-ref-v1"),
            get=handlers.get_slice,
            list=handlers.list_slices,
        ),
        ResourceRegistration(
            ENDPOINT,
            views=basic_full,
            body_type_urls=(_type_url(resource_pb2.EndpointSummary), _type_url(resource_pb2.EndpointDetail)),
            query_type_urls=(_type_url(resource_pb2.EndpointQuery),),
            features=("current-ref-v1",),
            get=handlers.get_endpoint,
            batch_get=handlers.batch_get_endpoints,
            list=handlers.list_endpoints,
        ),
        ResourceRegistration(
            CAPACITY,
            views=(resource_pb2.RESOURCE_VIEW_FULL,),
            body_type_urls=(_type_url(resource_pb2.GetCapacityStatusResponse),),
            features=("current-only-v1",),
            get=handlers.get_capacity,
        ),
        ResourceRegistration(
            USER_SUMMARY,
            views=(resource_pb2.RESOURCE_VIEW_BASIC,),
            body_type_urls=(_type_url(resource_pb2.UserSummary),),
            query_type_urls=(_type_url(resource_pb2.ListUsersRequest),),
            list=handlers.list_users,
        ),
        ResourceRegistration(
            ACTIVITY_ENTRY,
            views=(resource_pb2.RESOURCE_VIEW_FULL,),
            body_type_urls=(_type_url(resource_pb2.ActivityEntry),),
            query_type_urls=(_type_url(resource_pb2.ActivityQuery),),
            list=handlers.list_activity,
        ),
        ResourceRegistration(
            LOG_ENTRY,
            views=(resource_pb2.RESOURCE_VIEW_FULL,),
            body_type_urls=(_type_url(iris_logging_pb2.LogEntry),),
            query_type_urls=(_type_url(resource_pb2.FetchLogsRequest),),
            list=handlers.list_logs,
        ),
        ResourceRegistration(
            OPERATION,
            views=(resource_pb2.RESOURCE_VIEW_FULL,),
            body_type_urls=(_type_url(resource_pb2.Operation),),
            features=("durable-update-v1",),
            get=handlers.get_operation,
        ),
        ResourceRegistration(
            EXEC_SESSION,
            body_type_urls=(_type_url(resource_pb2.ExecAttemptRequest), _type_url(resource_pb2.ExecAttemptResponse)),
            create_type_urls=(_type_url(resource_pb2.ExecAttemptRequest),),
            create=handlers.create_exec_session,
        ),
        ResourceRegistration(
            PROFILE_CAPTURE,
            body_type_urls=(
                _type_url(resource_pb2.ProfileAttemptRequest),
                _type_url(resource_pb2.ProfileAttemptResponse),
            ),
            create_type_urls=(_type_url(resource_pb2.ProfileAttemptRequest),),
            create=handlers.create_profile_capture,
        ),
        ResourceRegistration(
            ENDPOINT_CAPABILITY,
            body_type_urls=(
                _type_url(resource_pb2.MintEndpointTokenRequest),
                _type_url(resource_pb2.MintEndpointTokenResponse),
            ),
            create_type_urls=(_type_url(resource_pb2.MintEndpointTokenRequest),),
            create=handlers.create_endpoint_capability,
        ),
    )


def resource_catalog(resources: Controller) -> ResourceCatalog:
    """Return the immutable catalog installed for one controller."""
    backend_registrations: list[BackendResourceRegistration] = []
    for backend_id, backend in resources.backends.items():
        features = tuple(sorted(capability.value for capability in backend.capabilities))
        backend_registrations.extend(
            (
                BackendResourceRegistration(backend_id, ATTEMPT, ("update",), features),
                BackendResourceRegistration(backend_id, EXEC_SESSION, ("create",), features),
                BackendResourceRegistration(backend_id, PROFILE_CAPTURE, ("create",), features),
                BackendResourceRegistration(backend_id, NODE, ("get", "list"), features),
            )
        )
        if backend.capabilities & {BackendCapability.IRIS_AUTOSCALER, BackendCapability.CLUSTER_VIEW}:
            backend_registrations.append(BackendResourceRegistration(backend_id, SLICE, ("get", "list"), features))
    return ResourceCatalog(resource_registrations(resources), backend_registrations)
