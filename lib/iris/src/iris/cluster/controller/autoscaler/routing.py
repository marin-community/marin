# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure demand routing and capacity estimation for the autoscaler."""

import difflib
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.constraints import (
    AVAILABILITY_PREFIX,
    AttributeValue,
    Constraint,
    ConstraintIndex,
    ConstraintOp,
    DeviceType,
    PlacementRequirements,
    availability_key,
    device_variant_constraint,
    evaluate_constraint,
    extract_placement_requirements,
    get_device_type_enum,
    is_availability_key,
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
from iris.cluster.types import AcceleratorType, CapacityType, gpu_device, tpu_device
from iris.rpc import job_pb2

# Synthetic task id stem for an availability probe (see availability_probe_entries).
_AVAILABILITY_PROBE_TASK = "__availability_probe__"


def empirical_zone_capabilities(groups: Iterable[ScalingGroup], timestamp: Timestamp) -> dict[str, frozenset[str]]:
    """Map zone -> accelerator variants the cluster has EMPIRICALLY confirmed available.

    A variant counts for a zone when an accelerator group in that zone's *region*
    currently has >0 successfully-allocated (``READY``) slices and is not erroring
    (``QUOTA_EXCEEDED``/``BACKOFF``) — i.e. we actually scaled it up and got capacity,
    not merely that a group is configured there. This replaces the old optimistic
    "configured and not quota-blocked" map: a configured-but-never-launched group
    advertises nothing until a scale-up (e.g. an availability probe) succeeds.

    Rolled up to region so every zone in a region inherits the region's live variants
    ("schedule me where the accelerator can be found" is a regional question); the
    returned map stays keyed by zone so the worker/group enrichment consumers are
    unchanged. Only zones belonging to a region with ≥1 live variant appear.
    """
    region_variants: dict[str, set[str]] = defaultdict(set)
    zones_by_region: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        zone = group.zone
        if zone is None:
            continue
        region = group.region or zone
        zones_by_region[region].add(zone)
        resources = group.resources
        variant = resources.device_variant if resources is not None else ""
        if not variant or group.ready_slice_count() == 0:
            continue
        if group.availability(timestamp).status in (GroupAvailability.QUOTA_EXCEEDED, GroupAvailability.BACKOFF):
            continue
        region_variants[region].add(variant.lower())

    caps: dict[str, frozenset[str]] = {}
    for region, zones in zones_by_region.items():
        variants = frozenset(region_variants.get(region, ()))
        if not variants:
            continue
        for zone in zones:
            caps[zone] = variants
    return caps


def availability_probe_entries(
    groups: Sequence[ScalingGroup],
    demand_entries: Sequence[DemandEntry],
    available_variants: frozenset[str],
) -> list[DemandEntry]:
    """Convert unmet ``availability:<variant>`` demand into accelerator scale-up demand.

    A job carrying ``availability:V`` cannot be placed until some region has live V
    capacity, but empirical availability can only be discovered by attempting a
    scale-up. For each V that a pending entry constrains on and that is not yet
    available, emit **one** synthetic demand entry for a V slice — routed to V's group
    by device variant (NOT by availability, which would be circular). When the
    scale-up succeeds V becomes available and the constrained job places, leaving the
    pending set, so this probe demand naturally subsides on the next tick. Probe
    slices an orchestrator does not promptly claim are reclaimed by idle-scaledown.
    """
    wanted: set[str] = set()
    for entry in demand_entries:
        for constraint in entry.constraints:
            if is_availability_key(constraint.key):
                wanted.add(constraint.key[len(AVAILABILITY_PREFIX) :])
    to_probe = wanted - available_variants
    if not to_probe:
        return []

    group_by_variant: dict[str, ScalingGroup] = {}
    for group in groups:
        resources = group.resources
        variant = resources.device_variant.lower() if resources is not None and resources.device_variant else ""
        if variant in to_probe and variant not in group_by_variant:
            group_by_variant[variant] = group

    probes: list[DemandEntry] = []
    for variant in sorted(to_probe):
        group = group_by_variant.get(variant)
        if group is None:
            continue  # no configured group provides it — nothing to probe
        probes.append(_availability_probe_entry(variant, group))
    return probes


def _availability_probe_entry(variant: str, group: ScalingGroup) -> DemandEntry:
    """One non-coscheduled demand entry shaped to scale a single slice of ``group``."""
    resources = group.resources
    if resources is not None and resources.device_type == AcceleratorType.GPU:
        device = gpu_device(resources.device_variant, resources.device_count or 1)
    else:
        # tpu_device infers the per-VM chip count, which matches the group's per-VM
        # device_count, so check_resource_fit accepts one VM's worth of accelerator.
        device = tpu_device(variant)
    constraints = [device_variant_constraint([variant])]
    return DemandEntry(
        task_ids=(f"{_AVAILABILITY_PROBE_TASK}:{variant}",),
        coschedule_group_id=None,
        normalized=extract_placement_requirements(constraints),
        constraints=constraints,
        resources=job_pb2.ResourceSpecProto(device=device),
    )


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
        if self.group.check_resource_fit(entry.resources) is not None:
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


def format_variants(variants: frozenset[str] | None) -> str:
    if not variants:
        return "*"
    return ",".join(sorted(variants))


# GCP zones end with -{single letter}, e.g. us-central1-a.
_ZONE_PATTERN = re.compile(r".+-[a-z]$")


def _looks_like_zone(value: str) -> bool:
    return bool(_ZONE_PATTERN.fullmatch(value))


def _diagnose_locality(
    kind: str,
    other_kind: str,
    requested: frozenset[str],
    available_same: set[str],
    available_other: set[str],
    confused_with_other: Callable[[str], bool],
) -> str:
    """Explain why no group matches a zone or region constraint.

    `confused_with_other(value)` returns True when `value` looks like it
    belongs to `other_kind` rather than `kind` (e.g. a region string passed as
    a zone). When the value is also a known `other_kind`, the message suggests
    switching constraint types; otherwise it offers a fuzzy-match hint against
    the available values of the same kind.
    """
    requested_sorted = sorted(requested)
    parts = [f"no groups in {kind} {', '.join(requested_sorted)}"]
    for value in requested_sorted:
        if confused_with_other(value) and value in available_other:
            parts.append(f"'{value}' looks like a {other_kind}, not a {kind}; use a {other_kind} constraint instead")
        else:
            close = difflib.get_close_matches(value, available_same, n=1, cutoff=0.7)
            if close:
                parts.append(f"did you mean {close[0]}?")
    return "; ".join(parts)


# Human-readable operator glyphs for _format_constraint (EXISTS/NOT_EXISTS/IN render as words).
_OP_SYMBOLS = {
    ConstraintOp.EQ: "==",
    ConstraintOp.NE: "!=",
    ConstraintOp.GT: ">",
    ConstraintOp.GE: ">=",
    ConstraintOp.LT: "<",
    ConstraintOp.LE: "<=",
}

# Caps for the per-constraint coverage report: enough groups to see the conflict, not the fleet.
_COVERAGE_MAX_SIGNATURES = 6
_COVERAGE_MAX_EXAMPLES = 5


def _format_constraint(c: Constraint) -> str:
    """Render a constraint compactly for a diagnostic (``region==us-east5``,
    ``availability:v5litepod-16 exists``), instead of the raw dataclass repr."""
    if c.op == ConstraintOp.EXISTS:
        return f"{c.key} exists"
    if c.op == ConstraintOp.NOT_EXISTS:
        return f"{c.key} absent"
    if c.op == ConstraintOp.IN:
        return f"{c.key} in {{{', '.join(str(v.value) for v in c.values)}}}"
    return f"{c.key}{_OP_SYMBOLS[c.op]}{c.values[0].value}"


def _constraint_coverage(
    hard_constraints: Sequence[Constraint], group_attrs: Mapping[str, dict[str, AttributeValue]]
) -> str:
    """How each hard constraint fares across the fleet, and the groups that come closest.

    Reports, per constraint, how many groups satisfy it on its own, then lists the groups that
    fail the fewest constraints — each annotated with the specific ones it violates. This surfaces
    a pairwise conflict: e.g. every group satisfies the availability constraint or the preemptible
    constraint, but none satisfies both. Returns "" when there are no constraints or no groups.
    """
    if not hard_constraints or not group_attrs:
        return ""

    satisfied = [0] * len(hard_constraints)
    # group name -> indices of the constraints it fails (its failure "signature").
    failures: dict[str, tuple[int, ...]] = {}
    for name, attrs in group_attrs.items():
        failed = []
        for i, c in enumerate(hard_constraints):
            if evaluate_constraint(attrs.get(c.key), c):
                satisfied[i] += 1
            else:
                failed.append(i)
        failures[name] = tuple(failed)

    total = len(group_attrs)
    lines = [f"constraint coverage across {total} group(s):"]
    lines += [f"  {_format_constraint(c)}: {satisfied[i]}/{total} satisfy" for i, c in enumerate(hard_constraints)]

    # The caller only reaches here when no group matched all constraints, so every signature is
    # non-empty; the closest groups fail the fewest.
    fewest = min(len(sig) for sig in failures.values())
    by_signature: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for name, sig in failures.items():
        if len(sig) == fewest:
            by_signature[sig].append(name)

    lines.append(f"closest group(s) each fail {fewest} of {len(hard_constraints)} constraint(s):")
    for sig, names in list(by_signature.items())[:_COVERAGE_MAX_SIGNATURES]:
        failed_str = ", ".join(_format_constraint(hard_constraints[i]) for i in sig)
        shown = sorted(names)
        suffix = f", +{len(shown) - _COVERAGE_MAX_EXAMPLES} more" if len(shown) > _COVERAGE_MAX_EXAMPLES else ""
        lines.append(f"  fail [{failed_str}]: {', '.join(shown[:_COVERAGE_MAX_EXAMPLES])}{suffix}")
    return "\n".join(lines)


def _diagnose(
    placement: PlacementRequirements,
    groups: Sequence[ScalingGroup],
    *,
    coverage: str | None = None,
) -> str:
    """Explain why no scaling group satisfies a placement requirement.

    Layered analysis (device → preemptible → zone → region) with zone/region
    confusion heuristics and fuzzy-match hints. For a failure that is none of those
    structured cases (e.g. a generic ``availability:<variant>`` constraint),
    ``coverage`` — when given — replaces the bare list of group names. Returned
    string has no prefix; callers prepend their own (e.g. "no_matching_group: ")
    when needed.
    """
    device_type = placement.device_type or DeviceType.CPU
    device_matches = [g for g in groups if g.matches_device_requirement(device_type, placement.device_variants)]
    variants_str = format_variants(placement.device_variants)

    if not device_matches:
        available = ", ".join(g.name for g in groups)
        return f"no scaling group provides device {device_type.value}:{variants_str} (available: {available})"

    if placement.preemptible is not None:
        preempt_matches = [
            g
            for g in device_matches
            if (g.config.resources is not None and g.config.resources.capacity_type == CapacityType.PREEMPTIBLE)
            == placement.preemptible
        ]
        if not preempt_matches:
            want = "preemptible" if placement.preemptible else "non-preemptible"
            return f"no {want} group provides device {device_type.value}:{variants_str}"
        device_matches = preempt_matches

    available_zones = {g.zone for g in device_matches} - {None}
    available_regions = {g.region for g in device_matches} - {None}

    if placement.required_zones:
        return _diagnose_locality(
            kind="zone",
            other_kind="region",
            requested=placement.required_zones,
            available_same=available_zones,
            available_other=available_regions,
            confused_with_other=lambda value: not _looks_like_zone(value),
        )

    if placement.required_regions:
        return _diagnose_locality(
            kind="region",
            other_kind="zone",
            requested=placement.required_regions,
            available_same=available_regions,
            available_other=available_zones,
            confused_with_other=_looks_like_zone,
        )

    if coverage:
        return f"no scaling group matches all constraints.\n{coverage}"
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


def _feasibility_group_attrs(group: ScalingGroup) -> dict[str, AttributeValue]:
    """A group's routing attributes plus a CONFIGURED ``availability:<variant>`` marker.

    Lets :func:`job_feasibility` treat a hard ``availability:<variant>`` constraint as
    satisfiable against any group configured for that variant.
    """
    attrs = group.to_attributes()
    resources = group.resources
    if resources is not None and resources.device_variant:
        # CONFIGURED availability — must stay distinct from the EMPIRICAL (live-slice)
        # marker _enriched_group_attrs injects under the same key for runtime routing.
        # Feasibility is the static "can this ever schedule" gate, so a cold region with
        # no live slice (which the autoscaler probe will scale up) must still pass;
        # gating submission on live capacity would reject every cold-start availability
        # job. availability_key normalizes the variant to match the constraint.
        attrs[availability_key(resources.device_variant)] = AttributeValue("true")
    return attrs


def job_feasibility(
    groups: Sequence[ScalingGroup],
    constraints: Sequence[Constraint],
    replicas: int | None = None,
    resources: job_pb2.ResourceSpecProto | None = None,
) -> GroupFeasibility:
    """Answer: can any scaling group ever host this job shape?

    Ignores runtime availability (quota, cooldown, in-flight capacity) — that
    is the autoscaler's job on each tick. This predicate gates LaunchJob at
    submit time so jobs that can never be scheduled fail fast.

    A hard ``availability:<variant>`` constraint is satisfiable against any group
    *configured* for that variant (see :func:`_feasibility_group_attrs`); the
    constraint is ANDed with the job's other routing constraints, so an availability
    job with an incompatible region/zone still fails fast.

    When ``resources`` is given, a matching group must also have enough per-VM
    capacity for the request's additive dimensions (cpu, memory, disk, device
    count). This catches over-requests like 300GB disk on a pool that advertises
    100GB, which would otherwise route to no group and sit pending forever.

    Args:
        groups: scaling groups to consider.
        constraints: the job's hard + soft routing constraints.
        replicas: for coscheduled jobs, the required replica count; None for
            non-coscheduled jobs. When set, groups must also have num_vms that
            divides replicas evenly.
        resources: the job's per-task resource spec. When set, groups whose
            advertised per-VM capacity can't hold it are dropped from the
            feasible set.
    """
    groups_list = list(groups)
    if not groups_list:
        return GroupFeasibility(feasible=[], reason=None)

    group_attrs = {g.name: _feasibility_group_attrs(g) for g in groups_list}
    group_index = ConstraintIndex.build(group_attrs)
    hard_cs, _ = split_hard_soft(routing_constraints(constraints))
    matching_names = group_index.matching_entities(hard_cs)
    matching = [g for g in groups_list if g.name in matching_names]

    if not matching:
        placement = extract_placement_requirements(constraints)
        coverage = _constraint_coverage(hard_cs, group_attrs)
        return GroupFeasibility(feasible=[], reason=_diagnose(placement, groups_list, coverage=coverage))

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

    if resources is not None:
        fit_reasons = {g.name: g.check_resource_fit(resources) for g in matching}
        fitting = [g for g in matching if fit_reasons[g.name] is None]
        if not fitting:
            details = "; ".join(f"{name} ({reason})" for name, reason in fit_reasons.items() if reason)
            return GroupFeasibility(
                feasible=[],
                reason=f"no matching scaling group has enough per-VM capacity: {details}",
            )
        matching = fitting

    return GroupFeasibility(feasible=matching, reason=None)


def _diagnose_no_capacity(
    matching_groups: list[ScalingGroup],
    budgets: dict[str, RoutingBudget],
    ts: Timestamp,
) -> str:
    """Produce a specific reason when matching groups exist but none can accept demand."""

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


def _enriched_group_attrs(
    group: ScalingGroup, zone_capabilities: Mapping[str, frozenset[str]]
) -> dict[str, AttributeValue]:
    """A group's routing attributes plus its zone's ``availability:<variant>`` markers.

    A hard ``availability:<variant>`` constraint on a job's demand restricts routing
    to groups whose zone has empirically yielded that variant (live, non-erroring
    slices — see :func:`empirical_zone_capabilities`), so a CPU orchestrator's CPU
    demand is steered into a zone where its accelerator has actually been found (and
    is held back from zones that have never provided it).
    """
    attrs = group.to_attributes()
    for variant in zone_capabilities.get(group.zone or "", ()):
        attrs[availability_key(variant)] = AttributeValue("true")
    return attrs


def route_demand(
    groups: list[ScalingGroup],
    demand_entries: list[DemandEntry],
    timestamp: Timestamp | None = None,
    zone_capabilities: Mapping[str, frozenset[str]] | None = None,
) -> RoutingDecision:
    """Route demand to groups using two-phase routing with committed budgets."""

    ts = timestamp or Timestamp.now()
    zone_caps = zone_capabilities or {}
    sorted_groups = sorted(groups, key=lambda group: group.config.priority or 100)
    # Enrich once and reuse for BOTH hard filtering (the index) and soft ranking,
    # since the soft-rank sort reads this map directly rather than recomputing
    # group.to_attributes() (which would omit the injected availability markers).
    group_attrs = {group.name: _enriched_group_attrs(group, zone_caps) for group in sorted_groups}
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
                    -soft_constraint_score(group_attrs[group.name], soft_routing_cs),
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
            unmet.append(UnmetDemand(entry=entry, reason=_diagnose_no_capacity(matching_groups, full_budgets, ts)))

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
