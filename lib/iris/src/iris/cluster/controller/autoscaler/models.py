# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared autoscaler data structures."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from enum import Enum, StrEnum

from rigging.timing import Timestamp

from iris.cluster.constraints import Constraint, PlacementRequirements
from iris.cluster.providers.types import SliceHandle
from iris.rpc import job_pb2


class SliceLifecycleState(StrEnum):
    """Lifecycle state for a slice (VM group) in the autoscaler.

    These states represent the dominant state of a slice based on its constituent VMs.
    String values are lowercase names for use as dictionary keys and proto map keys.

    States:
    - REQUESTING: Scale-up operation in progress (tracked at ScalingGroup level)
    - BOOTING: At least one VM is booting (VM_STATE_BOOTING)
    - INITIALIZING: At least one VM is initializing (VM_STATE_INITIALIZING)
    - READY: All VMs are ready (VM_STATE_READY)
    - FAILED: At least one VM has failed (VM_STATE_FAILED or VM_STATE_PREEMPTED)

    Note: These are slice-level aggregate states, not direct VM states.
    """

    REQUESTING = enum.auto()
    BOOTING = enum.auto()
    INITIALIZING = enum.auto()
    READY = enum.auto()
    FAILED = enum.auto()


@dataclass
class SliceState:
    """Per-slice state tracked by ScalingGroup.

    Consolidates the slice handle with its associated tracking state
    (idle timeout, lifecycle) into a single structure.
    lifecycle and worker_ids are populated eagerly by the bootstrap thread.
    """

    handle: SliceHandle
    # Timestamp at which the slice's workers all became idle. None means
    # the slice is currently active or has never been observed; in both
    # cases the slice is not eligible for scale-down. Scale-down kicks in
    # once `now - quiet_since >= idle_threshold`. Memory-only.
    quiet_since: Timestamp | None = None
    lifecycle: SliceLifecycleState = SliceLifecycleState.BOOTING
    worker_ids: list[str] = field(default_factory=list)
    # worker_id -> reachable http://host:port URL. Populated at READY transition
    # from the slice handle's worker info. Empty after a controller restart for
    # already-READY slices — the health probe lazy-fetches via
    # handle.describe() in that case. Memory-only.
    worker_urls: dict[str, str] = field(default_factory=dict)
    # worker_id -> consecutive /health probe failures since the last
    # success. Reset on a healthy response; once any worker crosses
    # PING_FAILURE_THRESHOLD the slice is terminated. Memory-only.
    ping_failures: dict[str, int] = field(default_factory=dict)
    error_message: str = ""


class ScalingAction(Enum):
    """Type of scaling action."""

    SCALE_UP = "scale_up"


@dataclass(frozen=True)
class ScalingDecision:
    """A single scaling decision for a scale group."""

    scale_group: str
    action: ScalingAction
    slice_id: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class DemandEntry:
    """A demand entry specifying resource requirements and constraints."""

    task_ids: list[str]
    coschedule_group_id: str | None
    normalized: PlacementRequirements
    constraints: list[Constraint]
    resources: job_pb2.ResourceSpecProto
    invalid_reason: str | None = None


@dataclass(frozen=True)
class AdditiveReq:
    """Additive (packable) resource request for one non-coscheduled entry."""

    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int


@dataclass(frozen=True)
class UnmetDemand:
    entry: DemandEntry
    reason: str


@dataclass(frozen=True)
class GroupRoutingStatus:
    group: str
    priority: int
    assigned: int
    launch: int
    decision: str
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    group_to_launch: dict[str, int]
    group_required_slices: dict[str, int]
    routed_entries: dict[str, list[DemandEntry]]
    unmet_entries: list[UnmetDemand]
    group_reasons: dict[str, str]
    group_statuses: list[GroupRoutingStatus]
