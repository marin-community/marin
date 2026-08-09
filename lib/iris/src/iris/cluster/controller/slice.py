# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

from collections.abc import Mapping
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.backends.protocol import BackendCapability, ProviderError, TaskBackend
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
    ResourceKey,
    ResourceKind,
    SliceIdentity,
    SliceLocator,
)
from iris.resources.slice import (
    MembershipState,
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
    provider_node_ids: tuple[str, ...]
    error_message: str


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
            provider_node_ids=tuple(vm.vm_id for vm in item.vms),
            error_message=item.error_message,
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
                        observed_member_count=len(item.provider_node_ids),
                        observed_at=observed_at,
                        error_message=item.error_message,
                    )
                )
                members[(backend_id, slice_uid)] = tuple(
                    SliceMember(
                        provider_node_id=provider_node_id,
                        node=None,
                        observed_at=observed_at,
                    )
                    for provider_node_id in item.provider_node_ids
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
