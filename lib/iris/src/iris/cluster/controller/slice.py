# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

from collections.abc import Mapping
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.backends.protocol import BackendCapability, ProviderError, TaskBackend
from iris.backends.status import SliceStatus
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.pagination import (
    _decode_page_token,
    _encode_page_token,
    _page_size,
    _query_fingerprint,
)
from iris.cluster.controller.resource_identity import _opaque_uid
from iris.cluster.controller.source_status import (
    _available_source,
    _unavailable_backend_source,
    _unsupported_source,
)
from iris.resources.errors import (
    ActionPolicyRejected,
    ResourceNotFound,
)
from iris.resources.identity import (
    NodeIdentity,
    ResourceKey,
    ResourceKind,
    SliceIdentity,
    SliceLocator,
)
from iris.resources.slice import (
    MembershipState,
    SliceCapacityState,
    SliceDetail,
    SliceLifecycle,
    SliceMember,
    SliceQuery,
    SliceSummary,
)
from iris.resources.source import (
    MAX_PROVIDER_SNAPSHOT_ITEMS,
    Page,
    ResourceSourceStatus,
)

_MAX_SLICE_PAGE = 500


@dataclass(frozen=True, slots=True)
class _SliceSnapshot:
    slices: tuple[SliceSummary, ...]
    members: Mapping[tuple[str, str], tuple[SliceMember, ...]]
    source_statuses: tuple[ResourceSourceStatus, ...]


@dataclass(frozen=True, slots=True)
class _ProviderSliceObservation:
    slice_id: str
    scaling_group_id: str
    lifecycle_state: str
    created_at: Timestamp | None
    members: tuple["_ProviderMemberObservation", ...]
    error_message: str
    last_active_at: Timestamp | None
    capacity_state: str
    degraded_member_count: int


@dataclass(frozen=True, slots=True)
class _ProviderMemberObservation:
    provider_node_id: str
    worker_id: str
    healthy: bool
    usability: str
    running_task_count: int
    zone: str


def _provider_slice_observations(
    backend: TaskBackend,
) -> tuple[Timestamp, tuple[_ProviderSliceObservation, ...]]:
    status = backend.autoscaler_status()
    return status.last_evaluation or Timestamp.now(), tuple(
        _ProviderSliceObservation(
            slice_id=item.slice_id,
            scaling_group_id=group.name,
            lifecycle_state=item.state,
            created_at=item.created_at,
            members=tuple(
                _ProviderMemberObservation(
                    provider_node_id=vm.vm_id,
                    worker_id=vm.worker_id,
                    healthy=vm.worker_healthy,
                    usability=vm.usability,
                    running_task_count=vm.running_task_count,
                    zone=vm.zone,
                )
                for vm in item.vms
            ),
            error_message=item.error_message,
            last_active_at=item.last_active,
            capacity_state=item.capacity_status,
            degraded_member_count=item.degraded_slot_count,
        )
        for group in status.groups
        for item in group.slices
    )


class SliceResources:
    """Slice resource operations."""

    def __init__(self, dependencies: ResourceDependencies) -> None:
        self._dependencies = dependencies

    def list_slices(self, query: SliceQuery = SliceQuery()) -> Page[SliceSummary]:
        page_size = _page_size(query.page_size, _MAX_SLICE_PAGE)
        fingerprint = _query_fingerprint(
            "slices",
            {
                "backend_id": query.backend_id,
                "scaling_group_id": query.scaling_group_id,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        snapshot = self._slice_snapshot()
        filtered = [
            item
            for item in snapshot.slices
            if (query.backend_id is None or item.identity.backend_id == query.backend_id)
            and (query.scaling_group_id is None or item.scaling_group_id == query.scaling_group_id)
        ]
        filtered.sort(
            key=lambda item: (
                item.identity.backend_id,
                item.identity.key.resource_id,
                item.identity.slice_uid,
            )
        )
        if position is not None:
            last_key = (
                str(position["backend_id"]),
                str(position["slice_id"]),
                str(position["slice_uid"]),
            )
            filtered = [
                item
                for item in filtered
                if (item.identity.backend_id, item.identity.key.resource_id, item.identity.slice_uid) > last_key
            ]
        items = tuple(filtered[:page_size])
        next_token = None
        if len(filtered) > page_size:
            last = items[-1]
            next_token = _encode_page_token(
                fingerprint,
                {
                    "backend_id": last.identity.backend_id,
                    "slice_id": last.identity.key.resource_id,
                    "slice_uid": last.identity.slice_uid,
                },
            )
        return Page(items=items, next_page_token=next_token, source_statuses=snapshot.source_statuses)

    def describe_slice(self, locator: SliceLocator) -> SliceDetail:
        snapshot = self._slice_snapshot()
        matches = [
            item
            for item in snapshot.slices
            if item.identity.key == locator.key
            and item.identity.backend_id == locator.backend_id
            and (locator.slice_uid is None or item.identity.slice_uid == locator.slice_uid)
        ]
        if not matches:
            raise ResourceNotFound(locator.key.resource_id)
        if len(matches) != 1:
            raise ActionPolicyRejected(f"Slice locator {locator.key.resource_id!r} is ambiguous")
        item = matches[0]
        return SliceDetail(
            summary=item,
            members=snapshot.members.get((item.identity.backend_id, item.identity.slice_uid), ()),
            source_statuses=snapshot.source_statuses,
        )

    def _slice_snapshot(self) -> _SliceSnapshot:
        slices: list[SliceSummary] = []
        members: dict[tuple[str, str], tuple[SliceMember, ...]] = {}
        statuses: list[ResourceSourceStatus] = []
        for backend_id, backend in sorted(self._dependencies.backends.items()):
            if BackendCapability.IRIS_AUTOSCALER not in backend.capabilities:
                statuses.append(_unsupported_source(f"backend:{backend_id}", backend_id=backend_id))
                continue
            try:
                observed_at, observations = _provider_slice_observations(backend)
            except (ConnectionError, ProviderError) as exc:
                statuses.append(_unavailable_backend_source(backend_id, exc))
                continue
            if len(observations) > MAX_PROVIDER_SNAPSHOT_ITEMS:
                statuses.append(
                    _unavailable_backend_source(
                        backend_id,
                        ValueError(f"provider returned more than {MAX_PROVIDER_SNAPSHOT_ITEMS} slices"),
                    )
                )
                continue
            statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))
            for item in observations:
                slice_uid = _opaque_uid(
                    f"rpc:{backend_id}:{item.slice_id}:{item.created_at.epoch_ms() if item.created_at else 0}"
                )
                identity = SliceIdentity(
                    ResourceKey(self._dependencies.cluster_id, ResourceKind.SLICE, item.slice_id),
                    backend_id,
                    slice_uid,
                )
                lifecycle = _slice_lifecycle(item.lifecycle_state)
                membership_state = (
                    MembershipState.OBSERVED if lifecycle is SliceLifecycle.READY else MembershipState.UNKNOWN
                )
                slices.append(
                    SliceSummary(
                        identity=identity,
                        scaling_group_id=item.scaling_group_id,
                        lifecycle=lifecycle,
                        membership_state=membership_state,
                        observed_member_count=len(item.members),
                        observed_at=observed_at,
                        error_message=item.error_message,
                        created_at=item.created_at,
                        last_active_at=item.last_active_at,
                        capacity_state=_slice_capacity_state(item.capacity_state),
                        healthy_member_count=sum(
                            member.usability == "healthy" or member.healthy for member in item.members
                        ),
                        degraded_member_count=item.degraded_member_count,
                        running_task_count=sum(member.running_task_count for member in item.members),
                    )
                )
                members[(backend_id, slice_uid)] = tuple(
                    SliceMember(
                        provider_node_id=member.provider_node_id,
                        node=(
                            NodeIdentity(
                                ResourceKey(
                                    self._dependencies.cluster_id,
                                    ResourceKind.NODE,
                                    member.worker_id,
                                ),
                                backend_id,
                                member.worker_id,
                            )
                            if member.worker_id
                            else None
                        ),
                        observed_at=observed_at,
                        worker_id=member.worker_id,
                        healthy=member.healthy,
                        usability=member.usability,
                        running_task_count=member.running_task_count,
                        zone=member.zone,
                    )
                    for member in item.members
                )
        return _SliceSnapshot(tuple(slices), members, tuple(statuses))


def _slice_lifecycle(value: str) -> SliceLifecycle:
    if value == "ready":
        return SliceLifecycle.READY
    if value == "failed":
        return SliceLifecycle.FAILED
    if value in {"deleting", "stopping", "terminated"}:
        return SliceLifecycle.DELETING
    return SliceLifecycle.CREATING


def _slice_capacity_state(value: str) -> SliceCapacityState:
    try:
        return SliceCapacityState(value)
    except ValueError:
        return SliceCapacityState.UNKNOWN


def slice_detail_from_status(
    *,
    cluster_id: str,
    backend_id: str,
    scaling_group_id: str,
    value: SliceStatus,
    observed_at: Timestamp,
) -> SliceDetail:
    """Convert one backend-authored Slice observation into its resource form."""
    slice_uid = _opaque_uid(
        f"rpc:{backend_id}:{value.slice_id}:{value.created_at.epoch_ms() if value.created_at else 0}"
    )
    lifecycle = _slice_lifecycle(value.state)
    members = tuple(
        SliceMember(
            provider_node_id=vm.vm_id,
            node=(
                NodeIdentity(
                    ResourceKey(cluster_id, ResourceKind.NODE, vm.worker_id),
                    backend_id,
                    vm.worker_id,
                )
                if vm.worker_id
                else None
            ),
            observed_at=observed_at,
            worker_id=vm.worker_id,
            healthy=vm.worker_healthy,
            usability=vm.usability,
            running_task_count=vm.running_task_count,
            zone=vm.zone,
        )
        for vm in value.vms
    )
    return SliceDetail(
        summary=SliceSummary(
            identity=SliceIdentity(
                ResourceKey(cluster_id, ResourceKind.SLICE, value.slice_id),
                backend_id,
                slice_uid,
            ),
            scaling_group_id=scaling_group_id,
            lifecycle=lifecycle,
            membership_state=(
                MembershipState.OBSERVED if lifecycle is SliceLifecycle.READY else MembershipState.UNKNOWN
            ),
            observed_member_count=len(members),
            observed_at=observed_at,
            error_message=value.error_message,
            created_at=value.created_at,
            last_active_at=value.last_active,
            capacity_state=_slice_capacity_state(value.capacity_status),
            healthy_member_count=sum(member.usability == "healthy" or member.healthy for member in members),
            degraded_member_count=value.degraded_slot_count,
            running_task_count=sum(member.running_task_count for member in members),
        ),
        members=members,
        source_statuses=(),
    )
