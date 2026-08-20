# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registered fleet and capacity operations owned by the controller."""

from connectrpc.request import RequestContext
from google.protobuf.message import Message

from iris.cluster.controller.controller import Controller
from iris.cluster.controller.resource_operations.support import (
    _DEFAULT_RESOURCE_PAGE_SIZE,
    _node_ref,
    _parse_backend_ref_id,
    _require_ref_type,
    _resource,
    _resource_ref,
    _slice_ref,
)
from iris.cluster.controller.resource_operations.task import _attempt_summary_to_proto
from iris.resources.capacity import (
    CapacityBackend,
    CapacityKubernetesStatus,
    CapacityPeerBackend,
    CapacityRouting,
    CapacityScalingGroup,
    ResourceAvailability,
)
from iris.resources.errors import InvalidResourceRequest
from iris.resources.node import NodeAttribute, NodeAttributeKind, NodeQuery, NodeSummary
from iris.resources.slice import SliceMember, SliceQuery, SliceSummary
from iris.rpc import resource_fleet_pb2, resource_identity_pb2, resource_pb2
from iris.rpc.resource_codec import (
    membership_state_to_proto,
    node_health_from_proto,
    node_health_to_proto,
    node_locator_from_proto,
    slice_capacity_state_to_proto,
    slice_lifecycle_to_proto,
    slice_locator_from_proto,
)
from iris.rpc.resource_codec import (
    node_identity_to_proto as _node_identity_to_proto,
)
from iris.rpc.resource_codec import (
    resource_source_status_to_proto as _source_status_to_proto,
)
from iris.rpc.resource_codec import (
    slice_identity_to_proto as _slice_identity_to_proto,
)
from iris.rpc.resource_registry import ResourceWireContract
from iris.rpc.resource_types import CAPACITY, NODE, SLICE
from iris.time_proto import timestamp_to_proto


def _node_summary_to_proto(value: NodeSummary) -> resource_fleet_pb2.NodeSummary:
    result = resource_fleet_pb2.NodeSummary(
        identity=_node_identity_to_proto(value.identity),
        health=node_health_to_proto(value.health),
        schedulable=value.schedulable,
        capacity=resource_fleet_pb2.NodeCapacity(
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


def _node_attribute_to_proto(value: NodeAttribute) -> resource_fleet_pb2.NodeAttribute:
    result = resource_fleet_pb2.NodeAttribute(key=value.key)
    if value.kind is NodeAttributeKind.STRING:
        result.string_value = value.string_value or ""
    elif value.kind is NodeAttributeKind.INTEGER:
        result.integer_value = value.integer_value or 0
    else:
        result.float_value = value.float_value or 0.0
    return result


def _slice_summary_to_proto(value: SliceSummary) -> resource_fleet_pb2.SliceSummary:
    result = resource_fleet_pb2.SliceSummary(
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


def _slice_member_to_proto(value: SliceMember) -> resource_fleet_pb2.SliceMember:
    result = resource_fleet_pb2.SliceMember(
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


def _capacity_availability_to_proto(value: ResourceAvailability) -> resource_fleet_pb2.CapacityResourceAvailability:
    return resource_fleet_pb2.CapacityResourceAvailability(
        version=value.version,
        observed_at=timestamp_to_proto(value.observed_at),
        amounts=value.amounts,
        total_amounts=value.total_amounts,
        held_by_band=[
            resource_fleet_pb2.CapacityBandAvailability(band=band, amounts=amounts)
            for band, amounts in sorted(value.held_by_band.items())
        ],
    )


def _capacity_scaling_group_to_proto(value: CapacityScalingGroup) -> resource_fleet_pb2.CapacityScalingGroup:
    result = resource_fleet_pb2.CapacityScalingGroup(
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
            resource_fleet_pb2.CapacitySlice(
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


def _capacity_routing_to_proto(value: CapacityRouting) -> resource_fleet_pb2.CapacityRouting:
    return resource_fleet_pb2.CapacityRouting(
        unmet=[
            resource_fleet_pb2.CapacityUnmetDemand(
                entry=resource_fleet_pb2.CapacityDemandEntry(
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
            resource_fleet_pb2.CapacityGroupRouting(
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


def _capacity_kubernetes_to_proto(value: CapacityKubernetesStatus) -> resource_fleet_pb2.CapacityKubernetesStatus:
    return resource_fleet_pb2.CapacityKubernetesStatus(
        namespace=value.namespace,
        total_nodes=value.total_nodes,
        schedulable_nodes=value.schedulable_nodes,
        allocatable_cpu=value.allocatable_cpu,
        allocatable_memory=value.allocatable_memory,
        pods=[
            resource_fleet_pb2.CapacityKubernetesPod(
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
            resource_fleet_pb2.CapacityKubernetesPool(
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
            resource_fleet_pb2.CapacityKubernetesNode(
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


def _capacity_backend_to_proto(value: CapacityBackend) -> resource_fleet_pb2.CapacityBackend:
    result = resource_fleet_pb2.CapacityBackend(
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
            resource_fleet_pb2.CapacityAction(
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


def _capacity_peer_backend_to_proto(value: CapacityPeerBackend) -> resource_fleet_pb2.CapacityPeerBackend:
    result = resource_fleet_pb2.CapacityPeerBackend(
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


def _node_detail_to_proto(detail) -> resource_fleet_pb2.NodeDetail:
    return resource_fleet_pb2.NodeDetail(
        summary=_node_summary_to_proto(detail.summary),
        address=detail.address or "",
        attributes=[_node_attribute_to_proto(item) for item in detail.attributes],
        recent_attempts=[_attempt_summary_to_proto(item) for item in detail.recent_attempts],
        bootstrap_logs=detail.bootstrap_logs or "",
        source_statuses=[_source_status_to_proto(status) for status in detail.source_statuses],
    )


class GetNode:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_types=(resource_fleet_pb2.NodeSummary, resource_fleet_pb2.NodeDetail),
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self, request: resource_pb2.GetResourceRequest, _context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, NODE)
        backend_id, node_id = _parse_backend_ref_id(request.ref.id)
        locator = resource_identity_pb2.NodeLocator(
            key=resource_identity_pb2.ResourceKey(
                cluster_id=request.ref.authority_cluster_id,
                kind=resource_identity_pb2.RESOURCE_KIND_NODE,
                resource_id=node_id,
            ),
            backend_id=backend_id,
        )
        if request.ref.HasField("uid"):
            locator.node_uid = request.ref.uid
        detail = self._resources.describe_node(node_locator_from_proto(locator))
        proto = _node_detail_to_proto(detail)
        body: Message = proto.summary if request.view == resource_pb2.RESOURCE_VIEW_BASIC else proto
        return resource_pb2.GetResourceResponse(
            resource=_resource(_node_ref(proto.summary.identity), body),
            source_statuses=proto.source_statuses,
        )


class ListNodes:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_types=(resource_fleet_pb2.NodeSummary,),
        input_type=resource_fleet_pb2.NodeQuery,
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self,
        request: resource_pb2.ListResourcesRequest,
        query: resource_fleet_pb2.NodeQuery,
        _context: RequestContext,
    ) -> resource_pb2.ListResourcesResponse:
        page = self._resources.list_nodes(
            NodeQuery(
                backend_id=query.backend_id or None,
                contains=query.contains or None,
                health=frozenset(node_health_from_proto(value) for value in query.health),
                page_size=request.page_size or query.page.page_size or _DEFAULT_RESOURCE_PAGE_SIZE,
                page_token=request.page_token or query.page.page_token or None,
            )
        )
        summaries = [_node_summary_to_proto(item) for item in page.items]
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_node_ref(item.identity), item) for item in summaries],
            next_page_token=page.next_page_token or "",
            source_statuses=[_source_status_to_proto(status) for status in page.source_statuses],
        )


def _slice_detail_to_proto(detail) -> resource_fleet_pb2.SliceDetail:
    return resource_fleet_pb2.SliceDetail(
        summary=_slice_summary_to_proto(detail.summary),
        members=[_slice_member_to_proto(item) for item in detail.members],
        source_statuses=[_source_status_to_proto(status) for status in detail.source_statuses],
    )


class GetSlice:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_types=(resource_fleet_pb2.SliceSummary, resource_fleet_pb2.SliceDetail),
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self, request: resource_pb2.GetResourceRequest, _context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, SLICE)
        backend_id, slice_id = _parse_backend_ref_id(request.ref.id)
        locator = resource_identity_pb2.SliceLocator(
            key=resource_identity_pb2.ResourceKey(
                cluster_id=request.ref.authority_cluster_id,
                kind=resource_identity_pb2.RESOURCE_KIND_SLICE,
                resource_id=slice_id,
            ),
            backend_id=backend_id,
        )
        if request.ref.HasField("uid"):
            locator.slice_uid = request.ref.uid
        detail = self._resources.describe_slice(slice_locator_from_proto(locator))
        proto = _slice_detail_to_proto(detail)
        body: Message = proto.summary if request.view == resource_pb2.RESOURCE_VIEW_BASIC else proto
        return resource_pb2.GetResourceResponse(
            resource=_resource(_slice_ref(proto.summary.identity), body),
            source_statuses=proto.source_statuses,
        )


class ListSlices:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_types=(resource_fleet_pb2.SliceSummary,),
        input_type=resource_fleet_pb2.SliceQuery,
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self,
        request: resource_pb2.ListResourcesRequest,
        query: resource_fleet_pb2.SliceQuery,
        _context: RequestContext,
    ) -> resource_pb2.ListResourcesResponse:
        page = self._resources.list_slices(
            SliceQuery(
                backend_id=query.backend_id or None,
                scaling_group_id=query.scaling_group_id or None,
                page_size=request.page_size or query.page.page_size or _DEFAULT_RESOURCE_PAGE_SIZE,
                page_token=request.page_token or query.page.page_token or None,
            )
        )
        summaries = [_slice_summary_to_proto(item) for item in page.items]
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_slice_ref(item.identity), item) for item in summaries],
            next_page_token=page.next_page_token or "",
            source_statuses=[_source_status_to_proto(status) for status in page.source_statuses],
        )


def _capacity_to_proto(status) -> resource_fleet_pb2.CapacityStatus:
    return resource_fleet_pb2.CapacityStatus(
        backends=[_capacity_backend_to_proto(item) for item in status.backends],
        peers=[
            resource_fleet_pb2.CapacityPeer(
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
            resource_fleet_pb2.CapacityRunningPlacement(
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
            resource_fleet_pb2.CapacityUnroutableJob(job_id=item.job_id, reason=item.reason)
            for item in status.unroutable_jobs
        ],
        source_statuses=[_source_status_to_proto(item) for item in status.source_statuses],
    )


class GetCapacity:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_FULL,),
        body_types=(resource_fleet_pb2.CapacityStatus,),
        features=("current-only-v1",),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self, request: resource_pb2.GetResourceRequest, _context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, CAPACITY)
        if request.ref.id != "capacity" or request.ref.HasField("uid"):
            raise InvalidResourceRequest("Capacity is the current-only 'capacity' resource")
        body = _capacity_to_proto(self._resources.capacity_status())
        return resource_pb2.GetResourceResponse(
            resource=_resource(_resource_ref(request.ref.authority_cluster_id, CAPACITY, "capacity"), body),
            source_statuses=body.source_statuses,
        )
