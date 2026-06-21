# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale-up planning helpers built on top of routed demand."""

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import NamedTuple

from rigging.timing import Timestamp

from iris.cluster.controller.autoscaler.models import (
    UNRANKED_DEMAND_BAND,
    RoutingDecision,
    ScalingAction,
    ScalingDecision,
)
from iris.cluster.controller.autoscaler.reserved_pool import reserved_pool_usage
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup, SliceLifecycleState
from iris.cluster.tpu_topology import get_tpu_topology


@dataclass(frozen=True)
class GroupSliceCounts:
    """Grouped slice counts used by the scale-up planner."""

    ready: int
    requesting: int
    pending: int
    total: int

    @classmethod
    def from_group(cls, group: ScalingGroup) -> "GroupSliceCounts":
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
    """Build a scale-up plan for one group.

    ``slices_to_add`` is the desired launch count for this tick before any
    rate-limiting. Token-bucket throttling is applied at execution time.
    """

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
            headroom = group.max_slices - counts.total
            slices_to_add = min(slices_needed, headroom)

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


class _PoolCandidate(NamedTuple):
    """A scaling group of one fungible pool that wants to launch slices this tick."""

    name: str
    band: int
    chips_per_slice: int
    want_slices: int


def _admit_in_band_order(candidates: list[_PoolCandidate], free_chips: int) -> dict[str, int]:
    """Admit per-group new-slice counts under one fungible pool's chip budget.

    ``candidates`` are the groups of one ``quota_pool`` that want to launch this tick
    (lower band = higher priority). Returns ``group_name -> admitted_slices``. Groups
    are admitted highest priority first; once a band cannot fully launch, every
    strictly-lower-priority group on the pool is denied — so the remaining chips are
    held for the high-priority slice (e.g. accumulating across a multi-tick drain)
    instead of being re-grabbed by the lower-priority slice they were freed from.
    Same-band groups share the remaining chips greedily.
    """
    remaining = max(0, free_chips)
    admitted: dict[str, int] = {}
    blocking_band: int | None = None
    for cand in sorted(candidates, key=lambda c: (c.band, c.name)):
        if blocking_band is not None and cand.band > blocking_band:
            admitted[cand.name] = 0
            continue
        grant = (
            cand.want_slices if cand.chips_per_slice <= 0 else min(cand.want_slices, remaining // cand.chips_per_slice)
        )
        admitted[cand.name] = grant
        remaining -= grant * cand.chips_per_slice
        if grant < cand.want_slices and blocking_band is None:
            blocking_band = cand.band
    return admitted


def _cap_fungible_pool_launches(
    group_plans: dict[str, GroupScalePlan],
    groups: dict[str, ScalingGroup],
    routing_decision: RoutingDecision,
) -> dict[str, GroupScalePlan]:
    """Trim new launches per fungible reservation pool to its chip budget.

    A fungible reservation's per-size groups share one physical chip pool, but each
    group's ``max_slices`` bounds it independently and ``route_demand`` never reads
    the shared budget — so unconstrained planning can request far more chips than the
    reservation holds (256 v4-8 *and* 128 v4-16 = 2048 chips against a 1024-chip
    pool). This caps each pool's total ``slices_to_add * chips`` to the chips free
    against the reservation (live + in-flight slices already counted by
    ``reserved_pool_usage``), admitting groups highest priority first and holding
    chips for an unsatisfied high-priority slice rather than a lower one. Non-fungible
    groups are untouched.
    """
    usage = reserved_pool_usage(groups.values())
    if not usage:
        return group_plans

    candidates_by_pool: dict[str, list[_PoolCandidate]] = defaultdict(list)
    for name, group in groups.items():
        plan = group_plans[name]
        if group.reservation_chips <= 0 or plan.slices_to_add <= 0:
            continue
        band = min(
            (entry.band for entry in routing_decision.routed_entries.get(name, ())),
            default=UNRANKED_DEMAND_BAND,
        )
        chips = get_tpu_topology(group.accelerator_variant).chip_count
        candidates_by_pool[group.config.quota_pool].append(
            _PoolCandidate(name=name, band=band, chips_per_slice=chips, want_slices=plan.slices_to_add)
        )

    if not candidates_by_pool:
        return group_plans

    capped = dict(group_plans)
    for pool_id, candidates in candidates_by_pool.items():
        for name, grant in _admit_in_band_order(candidates, usage[pool_id].free_chips).items():
            if grant != capped[name].slices_to_add:
                capped[name] = replace(capped[name], slices_to_add=grant)
    return capped


def build_scale_plan(
    groups: dict[str, ScalingGroup],
    routing_decision: RoutingDecision,
    ts: Timestamp,
) -> ScalePlan:
    """Build the canonical autoscaler plan for the current routing decision.

    Fungible reservation pools are capped to their shared chip budget after per-group
    planning, so a high-priority job's larger slice claims the reservation before a
    lower-priority job's and the pool is never over-committed.
    """

    group_plans = {
        name: build_group_scale_plan(group, routing_decision.group_required_slices.get(name, 0), ts)
        for name, group in groups.items()
    }
    group_plans = _cap_fungible_pool_launches(group_plans, groups, routing_decision)
    return ScalePlan(routing_decision=routing_decision, group_plans=group_plans)
