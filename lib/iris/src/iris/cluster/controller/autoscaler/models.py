# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared autoscaler data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum

from iris.cluster.constraints import Constraint, PlacementRequirements
from iris.cluster.providers.types import SliceHandle
from iris.rpc import job_pb2
from rigging.timing import Timestamp


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

    REQUESTING = "requesting"
    BOOTING = "booting"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class SliceState:
    """Per-slice state tracked by ScalingGroup.

    Consolidates the slice handle with its associated tracking state
    (idle timeout, lifecycle) into a single structure.
    lifecycle and worker_ids are populated eagerly by the bootstrap thread.
    """

    handle: SliceHandle
    last_active: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    lifecycle: SliceLifecycleState = SliceLifecycleState.BOOTING
    worker_ids: list[str] = field(default_factory=list)
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
