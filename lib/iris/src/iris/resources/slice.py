# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Slice inventory and membership records."""

from dataclasses import dataclass
from enum import StrEnum

from rigging.timing import Timestamp

from iris.resources.identity import (
    NodeIdentity,
)
from iris.resources.identity import (
    SliceIdentity as SliceIdentity,
)
from iris.resources.identity import (
    SliceLocator as SliceLocator,
)
from iris.resources.source import ResourceSourceStatus


class SliceLifecycle(StrEnum):
    CREATING = "creating"
    READY = "ready"
    DELETING = "deleting"
    FAILED = "failed"


class MembershipState(StrEnum):
    UNKNOWN = "unknown"
    OBSERVED = "observed"


class SliceCapacityState(StrEnum):
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    IN_USE = "in_use"
    IDLE = "idle"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class SliceQuery:
    backend_id: str | None = None
    scaling_group_id: str | None = None
    page_size: int = 100
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class SliceSummary:
    identity: SliceIdentity
    scaling_group_id: str
    lifecycle: SliceLifecycle
    membership_state: MembershipState
    observed_member_count: int
    observed_at: Timestamp | None
    error_message: str
    created_at: Timestamp | None = None
    last_active_at: Timestamp | None = None
    capacity_state: SliceCapacityState = SliceCapacityState.UNKNOWN
    healthy_member_count: int = 0
    degraded_member_count: int = 0
    running_task_count: int = 0


@dataclass(frozen=True, slots=True)
class SliceMember:
    provider_node_id: str
    node: NodeIdentity | None
    observed_at: Timestamp
    worker_id: str = ""
    healthy: bool = False
    usability: str = ""
    running_task_count: int = 0
    zone: str = ""


@dataclass(frozen=True, slots=True)
class SliceDetail:
    summary: SliceSummary
    members: tuple[SliceMember, ...]
    source_statuses: tuple[ResourceSourceStatus, ...]
