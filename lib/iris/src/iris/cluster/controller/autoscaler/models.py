# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared autoscaler data structures."""

from dataclasses import dataclass
from enum import Enum

from iris.cluster.constraints import Constraint, PlacementRequirements
from iris.rpc import job_pb2

# Band for a demand entry whose tasks carry no resolved effective band; sorts after
# every real band (lower band = higher priority) so unranked demand yields chips to
# ranked demand for the same fungible pool.
UNRANKED_DEMAND_BAND = 1 << 30


class ScalingAction(Enum):
    """Type of scaling action."""

    SCALE_UP = "scale_up"


@dataclass(frozen=True)
class ScalingDecision:
    """A single scaling decision for a scale group."""

    scale_group: str
    action: ScalingAction
    reason: str = ""


@dataclass(frozen=True)
class DemandEntry:
    """A demand entry specifying resource requirements and constraints."""

    task_ids: tuple[str, ...]
    coschedule_group_id: str | None
    normalized: PlacementRequirements
    constraints: list[Constraint]
    resources: job_pb2.ResourceSpecProto
    invalid_reason: str | None = None
    band: int = UNRANKED_DEMAND_BAND
    """Effective band (min over the entry's tasks; lower = higher priority) the
    reservation-aware launch cap admits fungible-pool slices by. Stamped by the
    scheduler from its resolved band map; defaults to unranked for demand emitted
    where no band was resolved (e.g. the no-schedulable-work path)."""


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
