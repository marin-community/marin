# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale-up planning helpers built on top of routed demand."""

from __future__ import annotations

from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller.autoscaler.models import RoutingDecision, ScalingAction, ScalingDecision
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup, SliceLifecycleState


@dataclass(frozen=True)
class GroupSliceCounts:
    """Grouped slice counts used by the scale-up planner."""

    ready: int
    requesting: int
    pending: int
    total: int
    capacity_slices: int

    @classmethod
    def from_group(cls, group: ScalingGroup) -> GroupSliceCounts:
        counts = group.slice_state_counts()
        requesting = counts[SliceLifecycleState.REQUESTING]
        pending = counts[SliceLifecycleState.BOOTING] + counts[SliceLifecycleState.INITIALIZING] + requesting
        ready = counts[SliceLifecycleState.READY]
        total = sum(counts.values())
        return cls(
            ready=ready,
            requesting=requesting,
            pending=pending,
            total=total,
            capacity_slices=ready + pending,
        )


@dataclass(frozen=True)
class GroupScalePlan:
    """Canonical scale-up plan for one scaling group."""

    group: str
    required_slices: int
    buffer_slices: int
    max_slices: int
    counts: GroupSliceCounts
    target_slices: int
    demand_gap: int
    buffer_gap: int
    slices_to_add: int
    scale_up_blocked: bool

    def decisions(self) -> list[ScalingDecision]:
        if self.slices_to_add <= 0 or self.scale_up_blocked:
            return []
        return [
            ScalingDecision(
                scale_group=self.group,
                action=ScalingAction.SCALE_UP,
                reason=(
                    f"target={self.target_slices} (demand={self.required_slices}+buffer={self.buffer_slices}), "
                    f"total={self.counts.total}, pending={self.counts.pending} "
                    f"(scaling {index + 1}/{self.slices_to_add})"
                ),
            )
            for index in range(self.slices_to_add)
        ]


@dataclass(frozen=True)
class ScalePlan:
    """Canonical autoscaler plan for one evaluation cycle."""

    routing_decision: RoutingDecision
    group_plans: dict[str, GroupScalePlan]

    def launch_counts(self) -> dict[str, int]:
        """Return canonical launch counts keyed by scale-group name."""

        return {name: plan.slices_to_add for name, plan in self.group_plans.items()}

    def decisions(self) -> list[ScalingDecision]:
        """Flatten all planned scaling decisions for execution."""

        decisions: list[ScalingDecision] = []
        for plan in self.group_plans.values():
            decisions.extend(plan.decisions())
        return decisions


def build_group_scale_plan(group: ScalingGroup, required_slices: int, ts: Timestamp) -> GroupScalePlan:
    """Build the actionable scale-up plan for a group."""

    counts = GroupSliceCounts.from_group(group)
    target_slices = min(required_slices + group.buffer_slices, group.max_slices)
    demand_gap = max(0, required_slices - counts.pending)
    buffer_gap = max(0, target_slices - counts.total)
    slices_needed = max(demand_gap, buffer_gap)
    slices_to_add = 0
    blocked = False
    if slices_needed > 0 and counts.total < group.max_slices:
        blocked = not group.can_scale_up(ts)
        if not blocked:
            slices_to_add = min(slices_needed, group.max_slices - counts.total)

    return GroupScalePlan(
        group=group.name,
        required_slices=required_slices,
        buffer_slices=group.buffer_slices,
        max_slices=group.max_slices,
        counts=counts,
        target_slices=target_slices,
        demand_gap=demand_gap,
        buffer_gap=buffer_gap,
        slices_to_add=slices_to_add,
        scale_up_blocked=blocked,
    )


def build_scale_plan(
    groups: dict[str, ScalingGroup],
    routing_decision: RoutingDecision,
    ts: Timestamp,
) -> ScalePlan:
    """Build the canonical autoscaler plan for the current routing decision."""

    group_plans = {
        name: build_group_scale_plan(group, routing_decision.group_required_slices.get(name, 0), ts)
        for name, group in groups.items()
    }
    return ScalePlan(routing_decision=routing_decision, group_plans=group_plans)
