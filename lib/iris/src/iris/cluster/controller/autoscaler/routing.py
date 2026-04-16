# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure demand routing and capacity estimation for the autoscaler."""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from iris.cluster.constraints import (
    Constraint,
    ConstraintIndex,
    DeviceType,
    PlacementRequirements,
    extract_placement_requirements,
    get_device_type_enum,
    routing_constraints,
    soft_constraint_score,
    split_hard_soft,
)
from iris.cluster.controller.autoscaler.models import (
    AdditiveReq,
    DemandEntry,
    GroupRoutingStatus,
    RoutingDecision,
    UnmetDemand,
)
from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability, ScalingGroup, SliceLifecycleState
from iris.rpc import config_pb2
from rigging.timing import Timestamp


def additive_req(entry: DemandEntry) -> AdditiveReq:
    """Extract additive resource dimensions from a demand entry."""

    return AdditiveReq(
        cpu_millicores=entry.resources.cpu_millicores,
        memory_bytes=entry.resources.memory_bytes,
        disk_bytes=entry.resources.disk_bytes,
    )


@dataclass
class VmBin:
    """Represents one VM's remaining capacity during bin packing."""

    cpu_remaining: int
    memory_remaining: int
    disk_remaining: int

    def can_fit(self, req: AdditiveReq) -> bool:
        return (
            req.cpu_millicores <= self.cpu_remaining
            and req.memory_bytes <= self.memory_remaining
            and req.disk_bytes <= self.disk_remaining
        )

    def place(self, req: AdditiveReq) -> None:
        self.cpu_remaining -= req.cpu_millicores
        self.memory_remaining -= req.memory_bytes
        self.disk_remaining -= req.disk_bytes


def first_fit_decreasing(reqs: list[AdditiveReq], vm_capacity: AdditiveReq) -> int:
    """Pack requests into VMs using first-fit decreasing, returning VMs needed."""

    if not reqs:
        return 0
    reqs_sorted = sorted(
        reqs,
        key=lambda r: (r.disk_bytes, r.memory_bytes, r.cpu_millicores),
        reverse=True,
    )
    used: list[VmBin] = []
    for req in reqs_sorted:
        placed = False
        for bin_state in used:
            if bin_state.can_fit(req):
                bin_state.place(req)
                placed = True
                break
        if not placed:
            bin_state = VmBin(
                cpu_remaining=vm_capacity.cpu_millicores,
                memory_remaining=vm_capacity.memory_bytes,
                disk_remaining=vm_capacity.disk_bytes,
            )
            bin_state.place(req)
            used.append(bin_state)
    return len(used)


def _effective_vm_capacity(group: ScalingGroup) -> AdditiveReq | None:
    """Per-VM capacity for bin packing, with 0-means-unlimited semantics."""

    resources = group.resources
    if resources is None:
        return None
    return AdditiveReq(
        cpu_millicores=resources.cpu_millicores or 2**63,
        memory_bytes=resources.memory_bytes or 2**63,
        disk_bytes=resources.disk_bytes or 2**63,
    )


@dataclass
class RoutingBudget:
    """Per-group routing state with per-VM bin packing for fungible resources."""

    group: ScalingGroup
    vm_capacity: AdditiveReq | None
    max_vms: int
    packable_bins: list[VmBin]
    coscheduled_slices: int
    assigned_entries: list[DemandEntry]

    @property
    def name(self) -> str:
        return self.group.name

    @property
    def vms_used(self) -> int:
        return self.coscheduled_slices * self.group.num_vms + len(self.packable_bins)

    @property
    def required_slices(self) -> int:
        if not self.assigned_entries:
            return 0
        noncsc = math.ceil(len(self.packable_bins) / self.group.num_vms) if self.packable_bins else 0
        return self.coscheduled_slices + noncsc

    def try_assign(self, entry: DemandEntry) -> bool:
        if not _matches_filters(self.group, entry):
            return False
        if entry.invalid_reason:
            return False
        if not self.group.can_fit_resources(entry.resources):
            return False

        if entry.coschedule_group_id:
            if self.group.num_vms != len(entry.task_ids):
                return False
            return self._assign_coscheduled(entry)
        return self._assign_packable(entry)

    def _assign_packable(self, entry: DemandEntry) -> bool:
        is_accel = get_device_type_enum(entry.resources.device) != DeviceType.CPU

        if self.vm_capacity is None or is_accel:
            if self.vms_used >= self.max_vms:
                return False
            self.packable_bins.append(VmBin(cpu_remaining=0, memory_remaining=0, disk_remaining=0))
            self.assigned_entries.append(entry)
            return True

        req = additive_req(entry)
        for bin_state in self.packable_bins:
            if bin_state.can_fit(req):
                bin_state.place(req)
                self.assigned_entries.append(entry)
                return True
        if self.vms_used >= self.max_vms:
            return False

        cap = self.vm_capacity
        bin_state = VmBin(
            cpu_remaining=cap.cpu_millicores,
            memory_remaining=cap.memory_bytes,
            disk_remaining=cap.disk_bytes,
        )
        bin_state.place(req)
        self.packable_bins.append(bin_state)
        self.assigned_entries.append(entry)
        return True

    def _assign_coscheduled(self, entry: DemandEntry) -> bool:
        needed = self.group.num_vms
        if self.vms_used + needed > self.max_vms:
            return False
        self.coscheduled_slices += 1
        self.assigned_entries.append(entry)
        return True


def _make_routing_budget(group: ScalingGroup) -> RoutingBudget:
    counts = group.slice_state_counts()
    inflight = (
        counts.get(SliceLifecycleState.REQUESTING, 0)
        + counts.get(SliceLifecycleState.BOOTING, 0)
        + counts.get(SliceLifecycleState.INITIALIZING, 0)
    )
    ready = counts.get(SliceLifecycleState.READY, 0)
    current = sum(counts.values())
    headroom = max(0, group.max_slices - current)
    return RoutingBudget(
        group=group,
        vm_capacity=_effective_vm_capacity(group),
        max_vms=(ready + inflight + headroom) * group.num_vms,
        packable_bins=[],
        coscheduled_slices=0,
        assigned_entries=[],
    )


def _make_committed_budget(group: ScalingGroup) -> RoutingBudget | None:
    """Create a requesting-capped budget for groups with in-flight slices."""

    counts = group.slice_state_counts()
    requesting = counts.get(SliceLifecycleState.REQUESTING, 0)
    if requesting == 0:
        return None
    return RoutingBudget(
        group=group,
        vm_capacity=_effective_vm_capacity(group),
        max_vms=requesting * group.num_vms,
        packable_bins=[],
        coscheduled_slices=0,
        assigned_entries=[],
    )


def _format_variants(variants: frozenset[str] | None) -> str:
    if not variants:
        return "*"
    return ",".join(sorted(variants))


# GCP zones end with -{single letter}, e.g. us-central1-a.
_ZONE_PATTERN = re.compile(r".+-[a-z]$")


def _looks_like_zone(value: str) -> bool:
    return bool(_ZONE_PATTERN.fullmatch(value))


def _diagnose(
    placement: PlacementRequirements,
    groups: Sequence[ScalingGroup],
) -> str:
    """Explain why no scaling group satisfies a placement requirement.

    Layered analysis (device → preemptible → zone → region) with zone/region
    confusion heuristics and fuzzy-match hints. Returned string has no prefix;
    callers prepend their own (e.g. "no_matching_group: ") when needed.
    """
    device_type = placement.device_type or DeviceType.CPU
    device_matches = [g for g in groups if g.matches_device_requirement(device_type, placement.device_variants)]
    variants_str = _format_variants(placement.device_variants)

    if not device_matches:
        available = ", ".join(g.name for g in groups)
        return f"no scaling group provides device {device_type.value}:{variants_str} (available: {available})"

    if placement.preemptible is not None:
        preempt_matches = [
            g
            for g in device_matches
            if (g.config.resources.capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE) == placement.preemptible
        ]
        if not preempt_matches:
            want = "preemptible" if placement.preemptible else "non-preemptible"
            return f"no {want} group provides device {device_type.value}:{variants_str}"
        device_matches = preempt_matches

    if placement.required_zones:
        available_zones = {g.zone for g in device_matches} - {None}
        available_regions = {g.region for g in device_matches} - {None}
        requested = sorted(placement.required_zones)
        parts = [f"no groups in zone {', '.join(requested)}"]
        for z in requested:
            if not _looks_like_zone(z) and z in available_regions:
                parts.append(f"'{z}' looks like a region, not a zone; use a region constraint instead")
            else:
                close = difflib.get_close_matches(z, available_zones, n=1, cutoff=0.7)
                if close:
                    parts.append(f"did you mean {close[0]}?")
        return "; ".join(parts)

    if placement.required_regions:
        available_regions = {g.region for g in device_matches} - {None}
        available_zones = {g.zone for g in device_matches} - {None}
        requested = sorted(placement.required_regions)
        parts = [f"no groups in region {', '.join(requested)}"]
        for r in requested:
            if _looks_like_zone(r) and r in available_zones:
                parts.append(f"'{r}' looks like a zone, not a region; use a zone constraint instead")
            else:
                close = difflib.get_close_matches(r, available_regions, n=1, cutoff=0.7)
                if close:
                    parts.append(f"did you mean {close[0]}?")
        return "; ".join(parts)

    available = ", ".join(g.name for g in groups)
    return f"no scaling group matches constraints (available: {available})"


@dataclass(frozen=True)
class GroupFeasibility:
    """Result of the job_feasibility predicate.

    `feasible` is the subset of groups whose hard routing constraints match
    and (if coscheduled) have a compatible num_vms. Non-empty means the job
    can, in principle, be scheduled; an autoscaler tick may still need to
    grow a group before capacity appears.

    `reason` is populated iff `feasible` is empty, with a user-facing
    explanation suitable for rejecting the job at submit time.
    """

    feasible: list[ScalingGroup]
    reason: str | None


def job_feasibility(
    groups: Sequence[ScalingGroup],
    constraints: Sequence[Constraint],
    replicas: int | None = None,
) -> GroupFeasibility:
    """Answer: can any scaling group ever host this job shape?

    Ignores runtime availability (quota, cooldown, in-flight capacity) — that
    is the autoscaler's job on each tick. This predicate gates LaunchJob at
    submit time so jobs that can never be scheduled fail fast.

    Args:
        groups: scaling groups to consider.
        constraints: the job's hard + soft routing constraints.
        replicas: for coscheduled jobs, the required replica count; None for
            non-coscheduled jobs. When set, groups must also have num_vms that
            divides replicas evenly.
    """
    groups_list = list(groups)
    if not groups_list:
        return GroupFeasibility(feasible=[], reason=None)

    group_attrs = {g.name: g.to_attributes() for g in groups_list}
    group_index = ConstraintIndex.build(group_attrs)
    hard_cs, _ = split_hard_soft(routing_constraints(constraints))
    matching_names = group_index.matching_entities(hard_cs)
    matching = [g for g in groups_list if g.name in matching_names]

    if not matching:
        placement = extract_placement_requirements(constraints)
        return GroupFeasibility(feasible=[], reason=_diagnose(placement, groups_list))

    if replicas is not None:
        compatible = [g for g in matching if g.num_vms > 0 and replicas % g.num_vms == 0]
        if not compatible:
            sizes = {g.name: g.num_vms for g in matching}
            reason = (
                f"job requires {replicas} coscheduled replicas but no matching scaling group "
                f"has a compatible size (replicas must be an exact multiple of num_vms); "
                f"matching group sizes: {sizes}"
            )
            return GroupFeasibility(feasible=[], reason=reason)
        matching = compatible

    return GroupFeasibility(feasible=matching, reason=None)


def _diagnose_no_capacity(
    entry: DemandEntry,
    matching_groups: list[ScalingGroup],
    budgets: dict[str, RoutingBudget],
    ts: Timestamp,
) -> str:
    """Produce a specific reason when matching groups exist but none can accept demand."""

    del entry

    per_group: list[str] = []
    for group in matching_groups:
        availability = group.availability(ts)
        if not group.can_accept_demand(ts):
            per_group.append(f"{group.name}={availability.status.value}")
        elif group.name in budgets:
            per_group.append(f"{group.name}=exhausted")
        else:
            per_group.append(f"{group.name}=unknown")
    return f"no_capacity: {', '.join(per_group)}"


def _matches_filters(group: ScalingGroup, entry: DemandEntry) -> bool:
    return group.matches_constraints(entry.constraints)


def _build_group_statuses(
    sorted_groups: list[ScalingGroup],
    routed: dict[str, list[DemandEntry]],
    group_to_launch: dict[str, int],
    group_reasons: dict[str, str],
    ts: Timestamp,
) -> list[GroupRoutingStatus]:
    statuses: list[GroupRoutingStatus] = []
    for group in sorted_groups:
        name = group.name
        availability = group.availability(ts)
        assigned = len(routed.get(name, []))
        launch = group_to_launch.get(name, 0)

        if assigned > 0:
            decision = "selected"
            reason = group_reasons.get(name, "demand-routed")
        elif availability.status in {GroupAvailability.BACKOFF, GroupAvailability.QUOTA_EXCEEDED}:
            decision = "blocked"
            reason = availability.reason
        elif availability.status == GroupAvailability.REQUESTING:
            decision = "requesting"
            reason = availability.reason
        elif availability.status == GroupAvailability.COOLDOWN:
            decision = "cooldown"
            reason = availability.reason
        elif availability.status == GroupAvailability.AT_MAX_SLICES:
            decision = "blocked"
            reason = "at max_slices"
        else:
            decision = "idle"
            reason = ""

        statuses.append(
            GroupRoutingStatus(
                group=name,
                priority=group.config.priority or 100,
                assigned=assigned,
                launch=launch,
                decision=decision,
                reason=reason,
            )
        )
    return statuses


def _pool_blocked_tiers(groups: list[ScalingGroup], ts: Timestamp) -> dict[str, int]:
    """Return the minimum failed tier per quota_pool."""

    blocked: dict[str, int] = {}
    for group in groups:
        pool = group.config.quota_pool
        tier = group.config.allocation_tier
        if not pool or not tier:
            continue
        availability = group.availability(ts)
        if availability.status in (GroupAvailability.QUOTA_EXCEEDED, GroupAvailability.BACKOFF):
            if pool not in blocked or tier < blocked[pool]:
                blocked[pool] = tier
    return blocked


def _is_tier_blocked(group: ScalingGroup, pool_blocked: dict[str, int]) -> bool:
    pool = group.config.quota_pool
    tier = group.config.allocation_tier
    if not pool or not tier:
        return False
    min_blocked = pool_blocked.get(pool)
    if min_blocked is None:
        return False
    return tier >= min_blocked


def route_demand(
    groups: list[ScalingGroup],
    demand_entries: list[DemandEntry],
    timestamp: Timestamp | None = None,
) -> RoutingDecision:
    """Route demand to groups using two-phase routing with committed budgets."""

    ts = timestamp or Timestamp.now()
    sorted_groups = sorted(groups, key=lambda group: group.config.priority or 100)
    group_attrs = {group.name: group.to_attributes() for group in sorted_groups}
    group_index = ConstraintIndex.build(group_attrs)

    routed: dict[str, list[DemandEntry]] = {}
    unmet: list[UnmetDemand] = []
    group_reasons: dict[str, str] = {}

    committed_budgets: dict[str, RoutingBudget] = {}
    for group in sorted_groups:
        if not group.can_accept_demand(ts):
            continue
        budget = _make_committed_budget(group)
        if budget is not None:
            committed_budgets[group.name] = budget

    full_budgets: dict[str, RoutingBudget] = {}
    for group in sorted_groups:
        if group.can_accept_demand(ts):
            full_budgets[group.name] = _make_routing_budget(group)

    pool_blocked = _pool_blocked_tiers(sorted_groups, ts)

    for entry in demand_entries:
        if entry.invalid_reason:
            unmet.append(UnmetDemand(entry=entry, reason=entry.invalid_reason))
            continue

        routing_cs = routing_constraints(entry.constraints)
        hard_routing_cs, soft_routing_cs = split_hard_soft(routing_cs)
        matching_names = group_index.matching_entities(hard_routing_cs)
        matching_groups = [group for group in sorted_groups if group.name in matching_names]

        pre_tier_count = len(matching_groups)
        if pool_blocked:
            matching_groups = [group for group in matching_groups if not _is_tier_blocked(group, pool_blocked)]

        if not matching_groups:
            reason = (
                f"tier_blocked: {pre_tier_count} matching group(s) blocked by quota-pool tier monotonicity"
                if pre_tier_count > 0
                else f"no_matching_group: {_diagnose(entry.normalized, sorted_groups)}"
            )
            unmet.append(UnmetDemand(entry=entry, reason=reason))
            continue

        if soft_routing_cs:
            matching_groups = sorted(
                matching_groups,
                key=lambda group: (
                    -soft_constraint_score(group.to_attributes(), soft_routing_cs),
                    group.config.priority or 100,
                ),
            )

        if entry.coschedule_group_id and not any(group.num_vms == len(entry.task_ids) for group in matching_groups):
            group_detail = ", ".join(f"{group.name}={group.num_vms}" for group in matching_groups)
            unmet.append(
                UnmetDemand(
                    entry=entry,
                    reason=(
                        f"coschedule_mismatch: job needs {len(entry.task_ids)} tasks coscheduled"
                        f" but no matching group has num_vms={len(entry.task_ids)} ({group_detail})"
                    ),
                )
            )
            continue

        fit_reasons = [group.check_resource_fit(entry.resources) for group in matching_groups]
        if all(reason is not None for reason in fit_reasons):
            details = "; ".join(reason for reason in fit_reasons if reason is not None)
            unmet.append(UnmetDemand(entry=entry, reason=f"insufficient_resources: {details}"))
            continue

        matched = False
        matching_group_names = [group.name for group in matching_groups]

        for name in matching_group_names:
            budget = committed_budgets.get(name)
            if budget is not None and budget.try_assign(entry):
                full_budgets[budget.name].try_assign(entry)
                routed.setdefault(budget.name, []).append(entry)
                group_reasons.setdefault(budget.name, "demand-routed")
                matched = True
                break

        if not matched:
            for name in matching_group_names:
                budget = full_budgets.get(name)
                if budget is not None and budget.try_assign(entry):
                    routed.setdefault(budget.name, []).append(entry)
                    group_reasons.setdefault(budget.name, "demand-routed")
                    matched = True
                    break

        if not matched:
            unmet.append(
                UnmetDemand(entry=entry, reason=_diagnose_no_capacity(entry, matching_groups, full_budgets, ts))
            )

    group_to_launch: dict[str, int] = {}
    group_required_slices: dict[str, int] = {}
    for name, budget in full_budgets.items():
        required = budget.required_slices
        group_required_slices[name] = required
        if not budget.assigned_entries:
            continue
        counts = budget.group.slice_state_counts()
        capacity_slices = (
            counts.get(SliceLifecycleState.READY, 0)
            + counts.get(SliceLifecycleState.BOOTING, 0)
            + counts.get(SliceLifecycleState.INITIALIZING, 0)
            + counts.get(SliceLifecycleState.REQUESTING, 0)
        )
        group_to_launch[name] = max(0, required - capacity_slices)

    group_statuses = _build_group_statuses(sorted_groups, routed, group_to_launch, group_reasons, ts)
    return RoutingDecision(
        group_to_launch=group_to_launch,
        group_required_slices=group_required_slices,
        routed_entries=routed,
        unmet_entries=unmet,
        group_reasons=group_reasons,
        group_statuses=group_statuses,
    )
