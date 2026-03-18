# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler manages scaling across scale groups.

The autoscaler coordinates scaling decisions across multiple scale groups,
delegating slice ownership to ScalingGroup.

Key design principles:
- Autoscaler does NOT track slices directly - that's ScalingGroup's job
- Scale-up decisions come from Autoscaler, scale-down is delegated to ScalingGroup
- ScalingGroup owns per-slice idle tracking and decides which slices to scale down
- Bootstrap is handled internally by each Platform implementation, not by the autoscaler

The run_once() flow splits into two phases:
- refresh(): state-read phase — scale down idle slices from tracked state
- update(): CPU phase — evaluate demand and execute scale-up decisions
"""

from __future__ import annotations

import difflib
import logging
import math
from collections import deque
from collections.abc import Callable

from dataclasses import dataclass
from enum import Enum

from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    CommandResult,
    Platform,
    QuotaExhaustedError,
    RemoteWorkerHandle,
    SliceHandle,
    WorkerStatus,
)
from iris.cluster.constraints import (
    ConstraintIndex,
    DeviceType,
    PlacementRequirements,
    get_device_type_enum,
    routing_constraints,
)
from iris.cluster.controller.db import SCALING_GROUPS, SLICES, TRACKED_WORKERS, ControllerDB
from iris.cluster.types import WorkerStatusMap
from iris.cluster.controller.scaling_group import (
    GroupAvailability,
    GroupSnapshot,
    ScalingGroup,
    SliceLifecycleState,
    SliceSnapshot,
    restore_scaling_group,
)
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import cluster_pb2, config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


class _RestoredWorkerHandle:
    """Minimal handle placeholder used for restored tracked workers.

    Provides just enough of the RemoteWorkerHandle interface to let restored
    workers participate in the autoscaler until they are replaced by real
    handles from the platform or pruned by heartbeat failures.
    """

    def __init__(self, worker_id: str, internal_address: str) -> None:
        self._worker_id = worker_id
        self._internal_address = internal_address

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def vm_id(self) -> str:
        return self._worker_id

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return None

    @property
    def bootstrap_log(self) -> str:
        return ""

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=CloudWorkerState.RUNNING)

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        raise NotImplementedError("RestoredWorkerHandle does not support run_command")

    def reboot(self) -> None:
        raise NotImplementedError("RestoredWorkerHandle does not support reboot")


@dataclass(frozen=True)
class _TrackedWorkerRow:
    """Lightweight record for a tracked worker read from the DB."""

    worker_id: str
    slice_id: str
    scale_group: str
    internal_address: str


def _restore_tracked_workers(rows: list[_TrackedWorkerRow]) -> dict[str, TrackedWorker]:
    """Restore tracked workers from DB rows."""
    workers: dict[str, TrackedWorker] = {}
    for row in rows:
        handle = _RestoredWorkerHandle(worker_id=row.worker_id, internal_address=row.internal_address)
        tw = TrackedWorker(
            worker_id=row.worker_id,
            slice_id=row.slice_id,
            scale_group=row.scale_group,
            handle=handle,
        )
        workers[tw.worker_id] = tw
    return workers


@dataclass
class TrackedWorker:
    """Per-worker state tracked by the autoscaler across bootstrap and lifecycle."""

    worker_id: str
    slice_id: str
    scale_group: str
    handle: RemoteWorkerHandle
    bootstrap_log: str = ""


# Slices that die within this time of creation trigger backoff (preemption detection)
SHORT_LIVED_SLICE_THRESHOLD = Duration.from_minutes(5)

# After this long in UNKNOWN state, treat the slice as FAILED (quota timeout is 5 min, so this is conservative)
DEFAULT_UNRESOLVABLE_TIMEOUT = Duration.from_minutes(15)


class ScalingAction(Enum):
    """Type of scaling action."""

    SCALE_UP = "scale_up"


@dataclass
class ScalingDecision:
    """A single scaling decision for a scale group.

    Attributes:
        scale_group: Name of the scale group to scale
        action: Whether to scale up or down
        slice_id: For scale_down, the specific slice to terminate (None for scale_up)
        reason: Human-readable explanation of why this decision was made
    """

    scale_group: str
    action: ScalingAction
    slice_id: str | None = None
    reason: str = ""


@dataclass
class DemandEntry:
    """A demand entry specifying resource requirements and constraints.

    The `normalized` field carries all categorical placement constraints
    (device type, variant, preemptible, region, zone) as a single object,
    replacing what was previously spread across separate fields.
    """

    task_ids: list[str]
    coschedule_group_id: str | None
    normalized: PlacementRequirements
    constraints: list[cluster_pb2.Constraint]
    resources: cluster_pb2.ResourceSpecProto
    invalid_reason: str | None = None


@dataclass(frozen=True)
class AdditiveReq:
    """Additive (packable) resource request for one non-coscheduled entry."""

    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int


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
    """Pack requests into VMs using first-fit decreasing, returning VMs needed.

    This estimates required capacity for the autoscaler. It does not model
    placement onto existing READY workers — the scheduler handles that.

    Args:
        reqs: List of additive resource requests to pack.
        vm_capacity: Per-VM resource capacity used as the template for new bins.

    Returns:
        Number of VMs required to fit all requests.
    """
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
        for b in used:
            if b.can_fit(req):
                b.place(req)
                placed = True
                break
        if not placed:
            b = VmBin(
                cpu_remaining=vm_capacity.cpu_millicores,
                memory_remaining=vm_capacity.memory_bytes,
                disk_remaining=vm_capacity.disk_bytes,
            )
            b.place(req)
            used.append(b)
    return len(used)


def compute_required_slices(group: ScalingGroup, entries: list[DemandEntry]) -> int:
    """Compute the number of slices required to serve a group's routed entries.

    Coscheduled entries each consume one full slice. Non-coscheduled entries
    are bin-packed by additive resources (CPU, memory, disk) to estimate how
    many VMs are needed, then converted to slices via ceil(vms / num_vms).

    If the group has no per-VM resources configured, falls back to treating
    each entry as requiring one slice (the pre-packing behavior).
    """
    if not entries:
        return 0

    vm_capacity = _effective_vm_capacity(group)
    if vm_capacity is None:
        return len(entries)

    coscheduled_count = 0
    accel_vm_count = 0
    noncsc_reqs: list[AdditiveReq] = []
    for entry in entries:
        if entry.coschedule_group_id:
            coscheduled_count += 1
        elif get_device_type_enum(entry.resources.device) != DeviceType.CPU:
            # Accelerator entries are not bin-packable — each task needs
            # exclusive access to the device, so treat as 1 VM per entry.
            accel_vm_count += 1
        else:
            noncsc_reqs.append(additive_req(entry))

    required_vms = first_fit_decreasing(noncsc_reqs, vm_capacity) if noncsc_reqs else 0
    required_vms += accel_vm_count

    num_vms = group.num_vms
    required_slices_for_noncsc = math.ceil(required_vms / num_vms) if required_vms > 0 else 0
    return coscheduled_count + required_slices_for_noncsc


def _effective_vm_capacity(group: ScalingGroup) -> AdditiveReq | None:
    """Per-VM capacity for bin packing, with 0-means-unlimited semantics. None if unconfigured."""
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
    """Per-group routing state with per-VM bin packing for fungible resources.

    During routing, entries are packed into VmBin objects representing individual
    VMs. This prevents premature overflow to lower-priority groups when multiple
    entries fit in a single VM (e.g. 4x 32GB entries in a 128GB VM).
    """

    group: ScalingGroup
    vm_capacity: AdditiveReq | None  # None = no resources configured → 1 entry per VM
    max_vms: int  # (ready + inflight + headroom) * num_vms
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
        """Slices needed to serve all assigned entries, derived from bin packing."""
        if not self.assigned_entries:
            return 0
        noncsc = math.ceil(len(self.packable_bins) / self.group.num_vms) if self.packable_bins else 0
        return self.coscheduled_slices + noncsc

    def try_assign(self, entry: DemandEntry) -> bool:
        """Try to assign a demand entry. Checks categorical filters, per-VM fit,
        and attempts bin packing. Returns True if placed successfully.
        """
        group = self.group
        if not _matches_filters(group, entry):
            return False
        if entry.invalid_reason:
            return False
        if not group.can_fit_resources(entry.resources):
            return False

        if entry.coschedule_group_id:
            if group.num_vms != len(entry.task_ids):
                return False
            return self._assign_coscheduled(entry)
        return self._assign_packable(entry)

    def _assign_packable(self, entry: DemandEntry) -> bool:
        # TODO: track accelerator counts in VmBin so we can bin-pack mixed workloads.
        # For now, accelerator entries are VM-exclusive (one VM per entry).
        is_accel = get_device_type_enum(entry.resources.device) != DeviceType.CPU

        if self.vm_capacity is None or is_accel:
            if self.vms_used >= self.max_vms:
                return False
            self.packable_bins.append(VmBin(cpu_remaining=0, memory_remaining=0, disk_remaining=0))
            self.assigned_entries.append(entry)
            return True

        req = additive_req(entry)
        for b in self.packable_bins:
            if b.can_fit(req):
                b.place(req)
                self.assigned_entries.append(entry)
                return True
        if self.vms_used >= self.max_vms:
            return False
        cap = self.vm_capacity
        b = VmBin(
            cpu_remaining=cap.cpu_millicores,
            memory_remaining=cap.memory_bytes,
            disk_remaining=cap.disk_bytes,
        )
        b.place(req)
        self.packable_bins.append(b)
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
    """Create a requesting-capped budget for groups with in-flight slices.

    Returns None if the group has no requesting slices. The committed budget
    limits routing to the capacity already being provisioned, preventing
    demand from being re-routed away when other groups exit cooldown.
    """
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


@dataclass
class UnmetDemand:
    entry: DemandEntry
    reason: str


@dataclass
class RoutingDecision:
    group_to_launch: dict[str, int]
    group_required_slices: dict[str, int]
    routed_entries: dict[str, list[DemandEntry]]
    unmet_entries: list[UnmetDemand]
    group_reasons: dict[str, str]
    group_statuses: list[GroupRoutingStatus]


@dataclass
class GroupRoutingStatus:
    group: str
    priority: int
    assigned: int
    launch: int
    decision: str
    reason: str


def _format_variants(variants: frozenset[str] | None) -> str:
    """Format device variants for diagnostic messages."""
    if not variants:
        return "*"
    return ",".join(sorted(variants))


def _diagnose_no_matching_group(entry: DemandEntry, groups: list[ScalingGroup]) -> str:
    """Produce a concise, actionable reason when no group matches a demand entry.

    Checks filters in order (device → preemptible → zone → region) and reports
    the first mismatch with enough context to fix the issue.
    """
    n = entry.normalized
    device_type = n.device_type or DeviceType.CPU
    device_matches = [g for g in groups if g.matches_device_requirement(device_type, n.device_variants)]

    variants_str = _format_variants(n.device_variants)

    if not device_matches:
        return f"no_matching_group: no groups with device {device_type.value}:{variants_str}"

    if n.preemptible is not None:
        preempt_matches = [g for g in device_matches if g.config.resources.preemptible == n.preemptible]
        if not preempt_matches:
            want = "preemptible" if n.preemptible else "non-preemptible"
            return f"no_matching_group: no {want} groups for device {device_type.value}:{variants_str}"
        device_matches = preempt_matches

    if n.required_zones:
        available_zones = {g.zone for g in device_matches} - {None}
        requested = sorted(n.required_zones)
        msg = f"no_matching_group: no groups in zone {', '.join(requested)}"
        for req_zone in requested:
            close = difflib.get_close_matches(req_zone, available_zones, n=1, cutoff=0.7)
            if close:
                msg += f" (did you mean {close[0]}?)"
        return msg

    if n.required_regions:
        requested = sorted(n.required_regions)
        return f"no_matching_group: no groups in region {', '.join(requested)}"

    return f"no_matching_group: no groups match device={device_type.value}:{_format_variants(n.device_variants)}"


def _diagnose_no_capacity(
    entry: DemandEntry,
    matching_groups: list[ScalingGroup],
    budgets: dict[str, RoutingBudget],
    ts: Timestamp,
) -> str:
    """Produce a specific reason when matching groups exist but none can accept demand.

    This replaces the opaque "no_capacity" with the actual blocking condition
    per group (e.g. at_max_slices, backoff, quota_exceeded, or exhausted).
    """
    per_group: list[str] = []
    for g in matching_groups:
        avail = g.availability(ts)
        if not g.can_accept_demand(ts):
            per_group.append(f"{g.name}={avail.status.value}")
        elif g.name in budgets:
            per_group.append(f"{g.name}=exhausted")
        else:
            per_group.append(f"{g.name}=unknown")

    return f"no_capacity: {', '.join(per_group)}"


def _matches_filters(group: ScalingGroup, entry: DemandEntry) -> bool:
    """Check device type, preemptible preference, region, and zone constraints.

    Does NOT check resource capacity or accept-demand readiness.
    """
    return group.matches_constraints(entry.constraints)


def _build_group_statuses(
    sorted_groups: list[ScalingGroup],
    routed: dict[str, list[DemandEntry]],
    group_to_launch: dict[str, int],
    group_reasons: dict[str, str],
    ts: Timestamp,
) -> list[GroupRoutingStatus]:
    group_statuses: list[GroupRoutingStatus] = []
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

        group_statuses.append(
            GroupRoutingStatus(
                group=name,
                priority=group.config.priority or 100,
                assigned=assigned,
                launch=launch,
                decision=decision,
                reason=reason,
            )
        )
    return group_statuses


def route_demand(
    groups: list[ScalingGroup],
    demand_entries: list[DemandEntry],
    timestamp: Timestamp | None = None,
) -> RoutingDecision:
    """Route demand to groups using two-phase routing with committed budgets.

    Phase 1 routes entries to committed budgets (capped at requesting slice capacity)
    so that in-flight provisioning isn't orphaned when other groups exit cooldown.
    Phase 2 routes remaining entries through the normal priority-based waterfall
    with full-capacity budgets.
    """
    ts = timestamp or Timestamp.now()
    sorted_groups = sorted(groups, key=lambda g: g.config.priority or 100)

    # Build a ConstraintIndex over scaling groups for O(1) constraint matching
    group_attrs = {g.name: g.to_attributes() for g in sorted_groups}
    group_index = ConstraintIndex.build(group_attrs)

    routed: dict[str, list[DemandEntry]] = {}
    unmet: list[UnmetDemand] = []
    group_reasons: dict[str, str] = {}

    # Phase 1 budgets: requesting-capped for groups with in-flight slices
    committed_budgets: dict[str, RoutingBudget] = {}
    for group in sorted_groups:
        if not group.can_accept_demand(ts):
            continue
        budget = _make_committed_budget(group)
        if budget is not None:
            committed_budgets[group.name] = budget

    # Phase 2 budgets: full capacity for all accepting groups
    full_budgets: dict[str, RoutingBudget] = {}
    for group in sorted_groups:
        if group.can_accept_demand(ts):
            full_budgets[group.name] = _make_routing_budget(group)

    for entry in demand_entries:
        if entry.invalid_reason:
            unmet.append(UnmetDemand(entry=entry, reason=entry.invalid_reason))
            continue

        routing_cs = routing_constraints(entry.constraints)
        matching_names = group_index.matching_entities(routing_cs)
        matching_groups = [g for g in sorted_groups if g.name in matching_names]
        if not matching_groups:
            unmet.append(UnmetDemand(entry=entry, reason=_diagnose_no_matching_group(entry, sorted_groups)))
            continue

        if entry.coschedule_group_id and not any(g.num_vms == len(entry.task_ids) for g in matching_groups):
            group_detail = ", ".join(f"{g.name}={g.num_vms}" for g in matching_groups)
            reason = (
                f"coschedule_mismatch: job needs {len(entry.task_ids)} tasks coscheduled"
                f" but no matching group has num_vms={len(entry.task_ids)} ({group_detail})"
            )
            unmet.append(UnmetDemand(entry=entry, reason=reason))
            continue

        fit_reasons = [g.check_resource_fit(entry.resources) for g in matching_groups]
        if all(r is not None for r in fit_reasons):
            details = "; ".join(r for r in fit_reasons if r is not None)
            unmet.append(UnmetDemand(entry=entry, reason=f"insufficient_resources: {details}"))
            continue

        matched = False

        # Phase 1: try committed budgets first (groups with requesting slices)
        for budget in committed_budgets.values():
            if budget.try_assign(entry):
                # Mirror into full budget so phase 2 sees consumed capacity
                full_budgets[budget.name].try_assign(entry)
                routed.setdefault(budget.name, []).append(entry)
                group_reasons.setdefault(budget.name, "demand-routed")
                matched = True
                break

        # Phase 2: fall through to full waterfall
        if not matched:
            for budget in full_budgets.values():
                if budget.try_assign(entry):
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


class Autoscaler:
    """Manages scaling across scale groups.

    The autoscaler:
    - Receives demand from a DemandSource
    - Evaluates scaling decisions based on demand vs capacity
    - Executes decisions by calling ScalingGroup.scale_up/scale_down

    It does NOT:
    - Track VM groups (ScalingGroup does that)
    - Know about controller internals (DemandSource abstracts that)
    """

    def __init__(
        self,
        scale_groups: dict[str, ScalingGroup],
        evaluation_interval: Duration,
        platform: Platform,
        threads: ThreadContainer | None = None,
        base_worker_config: config_pb2.WorkerConfig | None = None,
        db: ControllerDB | None = None,
        unresolvable_timeout: Duration = DEFAULT_UNRESOLVABLE_TIMEOUT,
    ):
        """Create autoscaler with explicit parameters.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            evaluation_interval: How often to evaluate scaling decisions
            platform: Platform instance for shutdown lifecycle
            threads: Optional thread container for testing
            base_worker_config: Base worker config merged with per-group overrides
                and passed to platform.create_slice(). None disables bootstrap (test/local mode).
            db: Optional DB handle for write-through persistence of tracked workers.
            unresolvable_timeout: How long a slice can remain UNKNOWN before being treated as FAILED.
        """
        self._groups = scale_groups
        self._platform = platform
        self._db = db
        self.evaluation_interval = evaluation_interval
        self._base_worker_config = base_worker_config
        self._unresolvable_timeout = unresolvable_timeout

        # Centralized per-worker state indexed by worker_id
        self._workers: dict[str, TrackedWorker] = {}

        # Bounded log of recent autoscaler actions for dashboard/debugging
        self._action_log: deque[vm_pb2.AutoscalerAction] = deque(maxlen=100)

        # Most recent routing decision (for status API)
        self._last_routing_decision: RoutingDecision | None = None
        self._last_evaluation: Timestamp = Timestamp.from_ms(0)

        # Thread management
        self._threads = threads if threads is not None else get_thread_container()

    @classmethod
    def from_config(
        cls,
        scale_groups: dict[str, ScalingGroup],
        config: config_pb2.AutoscalerConfig,
        platform: Platform,
        threads: ThreadContainer | None = None,
        base_worker_config: config_pb2.WorkerConfig | None = None,
        db: ControllerDB | None = None,
    ) -> Autoscaler:
        """Create autoscaler from proto config.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            config: Autoscaler configuration proto (with defaults already applied)
            platform: Platform instance for shutdown lifecycle
            threads: Optional thread container for testing
            base_worker_config: Base worker config merged with per-group overrides
            db: Optional DB handle for write-through persistence.

        Returns:
            Configured Autoscaler instance
        """
        return cls(
            scale_groups=scale_groups,
            evaluation_interval=Duration.from_proto(config.evaluation_interval),
            platform=platform,
            threads=threads,
            base_worker_config=base_worker_config,
            db=db,
        )

    def _wait_for_inflight(self) -> None:
        """Wait for in-flight scale-ups to complete without terminating anything.

        Test-only: Waits for all scale-up threads to complete.
        """
        self._threads.wait()

    def shutdown(self) -> None:
        """Shutdown the autoscaler, terminate all VM groups, and clean up platform.

        Shutdown ordering:
        1. Stop all threads in the autoscaler's ThreadContainer. This signals
           stop_events for both in-flight scale-up threads AND worker lifecycle
           threads (via child containers), then joins with timeout.
        2. Terminate all VM groups — calls Worker.stop() for final cleanup
           of any workers that didn't exit in step 1.
        3. Shutdown platform — clears local tracking state.
        """
        # Stop all threads (scale-ups + workers) via ThreadContainer.
        # Using stop() rather than wait() because wait() doesn't signal
        # stop_events and would block forever on worker-lifecycle threads.
        self._threads.stop()

        # Step 2: Terminate VMs and cleanup (idempotent with step 1)
        for group in self._groups.values():
            group.terminate_all()

        # Step 3: Shutdown platform (cleanup remaining threads)
        self._platform.shutdown()

    def __enter__(self) -> Autoscaler:
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def _log_action(
        self,
        action_type: str,
        scale_group: str,
        slice_id: str = "",
        reason: str = "",
        status: str = "completed",
    ) -> vm_pb2.AutoscalerAction:
        """Log an autoscaler action to the bounded action log.

        Args:
            action_type: Type of action (scale_up, scale_down, etc.)
            scale_group: Name of the scale group
            slice_id: ID of the slice (if applicable)
            reason: Human-readable reason for the action
            status: Action status ("pending", "completed", "failed")

        Returns:
            The action object. The caller may mutate this object to update
            status after execution (e.g., from "pending" to "completed").
            This works because the deque holds references to the proto objects.
        """

        action = vm_pb2.AutoscalerAction(
            timestamp=Timestamp.now().to_proto(),
            action_type=action_type,
            scale_group=scale_group,
            slice_id=slice_id,
            reason=reason,
            status=status,
        )
        self._action_log.append(action)
        return action

    def evaluate(
        self,
        demand_entries: list[DemandEntry],
        timestamp: Timestamp | None = None,
    ) -> list[ScalingDecision]:
        """Compute scaling decisions based on demand.

        Routes demand to groups based on accelerator_type requirements and
        priority. Higher-priority groups (lower priority number) receive
        demand first; overflow routes to lower-priority groups.

        Args:
            demand_entries: List of demand entries with requirements and counts.
            timestamp: Optional timestamp for testing. If None, uses Timestamp.now().

        Returns:
            List of scaling decisions to execute.
        """
        ts = timestamp or Timestamp.now()

        result = route_demand(list(self._groups.values()), demand_entries, ts)
        self._last_routing_decision = result

        if result.unmet_entries:
            logger.debug(
                "Unmet demand: %d entries cannot be satisfied (visible in dashboard)",
                len(result.unmet_entries),
            )

        decisions = []
        for name, group in self._groups.items():
            required_slices = result.group_required_slices.get(name, 0)
            group.update_demand(required_slices)
            decisions.extend(self._evaluate_group(group, required_slices, ts))

        return decisions

    def _evaluate_group(
        self,
        group: ScalingGroup,
        required_slices: int,
        ts: Timestamp,
    ) -> list[ScalingDecision]:
        """Evaluate scaling decisions for a single group.

        Returns multiple SCALE_UP decisions when demand exceeds capacity by
        more than one slice, allowing multi-slice scale-up in a single cycle.
        """
        counts = group.slice_state_counts()
        ready = counts[SliceLifecycleState.READY]
        requesting = counts[SliceLifecycleState.REQUESTING]
        pending = counts[SliceLifecycleState.BOOTING] + counts[SliceLifecycleState.INITIALIZING] + requesting
        total = sum(counts.values())

        logger.debug(
            "Evaluating group %s: total=%d, ready=%d, pending=%d, required_slices=%d, min=%d, max=%d",
            group.name,
            total,
            ready,
            pending,
            required_slices,
            group.min_slices,
            group.max_slices,
        )

        # Priority 1: Enforce min_slices - scale up if below minimum
        if total < group.min_slices:
            if not group.can_scale_up(ts):
                logger.debug(
                    "Scale group %s: below min_slices (%d < %d) but scale up blocked",
                    group.name,
                    total,
                    group.min_slices,
                )
                return []

            return [
                ScalingDecision(
                    scale_group=group.name,
                    action=ScalingAction.SCALE_UP,
                    reason=f"below min_slices ({total} < {group.min_slices})",
                )
            ]

        # Priority 2: Scale UP when required slices exceed pending capacity.
        # Compare against pending only — ready slices were already tested by the
        # dry-run and found insufficient (e.g. RAM-full), so counting them as
        # available capacity would double-count and cause deadlock.
        if required_slices > pending and total < group.max_slices:
            if not group.can_scale_up(ts):
                logger.debug("Scale group %s: scale up blocked", group.name)
                return []

            slices_to_add = min(required_slices - pending, group.max_slices - total)
            return [
                ScalingDecision(
                    scale_group=group.name,
                    action=ScalingAction.SCALE_UP,
                    reason=(
                        f"required_slices={required_slices} > pending={pending}" f" (scaling {i + 1}/{slices_to_add})"
                    ),
                )
                for i in range(slices_to_add)
            ]

        return []

    def execute(
        self,
        decisions: list[ScalingDecision],
        timestamp: Timestamp,
    ) -> None:
        """Execute scale-up decisions.

        Args:
            decisions: List of scaling decisions to execute.
            timestamp: Current timestamp.
        """
        for decision in decisions:
            group = self._groups.get(decision.scale_group)
            if not group:
                logger.warning("Unknown scale group in decision: %s", decision.scale_group)
                continue

            if decision.action == ScalingAction.SCALE_UP:
                if not group.acquire_scale_up_token(timestamp):
                    logger.info("Rate-limited scale-up for %s: %s", decision.scale_group, decision.reason)
                    self._log_action(
                        "rate_limited",
                        decision.scale_group,
                        reason=decision.reason,
                    )
                    continue
                self._execute_scale_up(group, timestamp, reason=decision.reason)

    def _execute_scale_up(self, group: ScalingGroup, ts: Timestamp, reason: str = "") -> None:
        """Initiate async scale-up for a scale group.

        Increments the group's pending scale-up counter and spawns a background
        thread for the actual scale-up work. The counter is included in
        slice_count(), preventing double scale-up.
        """
        group.begin_scale_up(timestamp=ts)

        def _scale_up_wrapper(stop_event):
            self._do_scale_up(group, ts, reason)

        self._threads.spawn(
            target=_scale_up_wrapper,
            name=f"scale-up-{group.name}",
        )

    def _do_scale_up(self, group: ScalingGroup, ts: Timestamp, reason: str = "") -> bool:
        """Execute the actual blocking scale-up work.

        This runs in a background thread and should not be called directly.
        Use _execute_scale_up instead. Bootstrap is handled internally by the
        platform when cluster_config is provided.

        Returns:
            True if scale-up succeeded, False otherwise.
        """
        action = self._log_action("scale_up", group.name, reason=reason, status="pending")

        try:
            logger.info("Scaling up %s: %s", group.name, reason)
            wc = self._per_group_worker_config(group)
            slice_obj = group.scale_up(worker_config=wc, timestamp=ts)
            group.complete_scale_up(slice_obj, ts)
            logger.info("Created slice %s for group %s", slice_obj.slice_id, group.name)
            action.slice_id = slice_obj.slice_id
            action.status = "completed"
            return True
        except QuotaExhaustedError as e:
            group.cancel_scale_up()
            group.record_quota_exceeded(str(e), ts)
            logger.warning("Quota exceeded for %s: %s", group.name, e)
            action.action_type = "quota_exceeded"
            action.status = "failed"
            action.reason = str(e)
            return False
        except Exception as e:
            group.cancel_scale_up()
            logger.exception("Failed to create slice for %s: %s", group.name, e)
            action.status = "failed"
            action.reason = f"{reason} - error: {e}"
            group.record_failure(ts)
            return False

    def _per_group_worker_config(self, group: ScalingGroup) -> config_pb2.WorkerConfig | None:
        """Build per-group WorkerConfig by merging base config with scale group overrides."""
        if not self._base_worker_config:
            return None

        wc = config_pb2.WorkerConfig()
        wc.CopyFrom(self._base_worker_config)

        # Accelerator config from scale group resources
        resources = group.config.resources if group.config.HasField("resources") else None
        if resources is not None:
            wc.accelerator_type = resources.device_type
            if resources.device_variant:
                wc.accelerator_variant = resources.device_variant
            if resources.device_type == config_pb2.ACCELERATOR_TYPE_GPU and resources.device_count > 0:
                wc.gpu_count = resources.device_count
            wc.preemptible = resources.preemptible

        # Worker settings from scale group
        if group.config.HasField("worker"):
            for k, v in group.config.worker.attributes.items():
                wc.worker_attributes[k] = v
            for k, v in group.config.worker.env.items():
                wc.default_task_env[k] = v

        if group.config.name:
            wc.worker_attributes["scale-group"] = group.config.name

        return wc

    def _register_slice_workers(self, workers: list[RemoteWorkerHandle], slice_id: str, scale_group: str) -> None:
        """Register all workers from a slice into the worker registry."""
        for worker in workers:
            self._workers[worker.worker_id] = TrackedWorker(
                worker_id=worker.worker_id,
                slice_id=slice_id,
                scale_group=scale_group,
                handle=worker,
                bootstrap_log=worker.bootstrap_log,
            )
            if self._db is not None:
                with self._db.transaction() as cur:
                    cur.execute(
                        "INSERT OR REPLACE INTO tracked_workers(worker_id, slice_id, scale_group, internal_address) "
                        "VALUES (?, ?, ?, ?)",
                        (worker.worker_id, slice_id, scale_group, worker.internal_address),
                    )

    def _unregister_slice_workers(self, slice_id: str) -> None:
        """Remove all TrackedWorker entries belonging to a slice."""
        to_remove = [wid for wid, tw in self._workers.items() if tw.slice_id == slice_id]
        for wid in to_remove:
            del self._workers[wid]
        if self._db is not None and to_remove:
            with self._db.transaction() as cur:
                for wid in to_remove:
                    cur.execute("DELETE FROM tracked_workers WHERE worker_id = ?", (wid,))

    def refresh(self, worker_status_map: WorkerStatusMap, timestamp: Timestamp | None = None) -> None:
        """State-read phase: scale down idle slices from currently tracked state."""
        timestamp = timestamp or Timestamp.now()

        for group in self._groups.values():
            for slice_id, handle in group.non_ready_slice_handles():
                try:
                    status = handle.describe()
                except Exception as e:
                    logger.warning("Failed to poll slice %s: %s", slice_id, e)
                    continue

                if status.state == CloudSliceState.READY:
                    worker_ids = [w.worker_id for w in status.workers]
                    group.mark_slice_ready(slice_id, worker_ids)
                    self._register_slice_workers(status.workers, slice_id, group.name)
                    self._log_action(
                        "slice_ready",
                        group.name,
                        slice_id,
                        reason=f"bootstrap completed ({len(worker_ids)} workers)",
                    )
                elif status.state == CloudSliceState.FAILED:
                    group.mark_slice_failed(slice_id, error_message=status.error_message)
                    group.scale_down(slice_id)
                    self._unregister_slice_workers(slice_id)
                    group.record_failure()
                    reason = status.error_message if status.error_message else "bootstrap failed"
                    self._log_action(
                        "slice_failed",
                        group.name,
                        slice_id,
                        reason=reason,
                        status="failed",
                    )
                elif status.state == CloudSliceState.UNKNOWN:
                    age = Duration.from_ms(timestamp.epoch_ms() - handle.created_at.epoch_ms())
                    if age >= self._unresolvable_timeout:
                        group.mark_slice_failed(slice_id, error_message="unresolvable after timeout")
                        group.scale_down(slice_id)
                        self._unregister_slice_workers(slice_id)
                        group.record_failure()
                        self._log_action(
                            "slice_failed",
                            group.name,
                            slice_id,
                            reason=f"TPU unresolvable for {age}",
                            status="failed",
                        )
                    else:
                        logger.debug(
                            "Slice %s UNKNOWN (age %s < timeout %s); will retry",
                            slice_id,
                            age,
                            self._unresolvable_timeout,
                        )

        for group in self._groups.values():
            target_capacity = max(group.current_demand, group.min_slices)
            ready_before = group.ready_slice_count()
            scaled_down_handles = group.scale_down_if_idle(worker_status_map, target_capacity, timestamp)
            for handle in scaled_down_handles:
                self._unregister_slice_workers(handle.slice_id)
                self._log_action(
                    "scale_down",
                    group.name,
                    slice_id=handle.slice_id,
                    reason=f"idle slice (target={target_capacity}, ready={ready_before})",
                )

    def update(
        self,
        demand_entries: list[DemandEntry],
        timestamp: Timestamp | None = None,
    ) -> list[ScalingDecision]:
        """CPU phase: evaluate demand and execute scale-up decisions."""
        timestamp = timestamp or Timestamp.now()
        self._last_evaluation = timestamp

        decisions = self.evaluate(demand_entries, timestamp)
        if decisions:
            logger.info("Autoscaler decisions: %s", [(d.scale_group, d.action.value, d.reason) for d in decisions])
        self.execute(decisions, timestamp)
        return decisions

    def run_once(
        self,
        demand_entries: list[DemandEntry],
        worker_status_map: WorkerStatusMap,
        timestamp: Timestamp | None = None,
    ) -> list[ScalingDecision]:
        """Full cycle: refresh + update. Preserved for tests."""
        timestamp = timestamp or Timestamp.now()
        logger.debug("Autoscaler run_once: demand_entries=%s", demand_entries)
        self.refresh(worker_status_map, timestamp)
        return self.update(demand_entries, timestamp)

    def restore_tracked_workers(self, workers: dict[str, TrackedWorker]) -> None:
        """Restore tracked worker state from a snapshot. Called before loops start."""
        self._workers.update(workers)

    def restore_from_db(self, db: ControllerDB, platform: Platform) -> None:
        """Reconcile DB-checkpointed autoscaler state against live cloud.

        Reads scaling group and slice rows from proper DB tables,
        reconciles each group against the cloud in parallel, and restores
        tracked workers. Call at startup before loops begin.
        """
        with db.snapshot() as snapshot:
            scaling_rows = snapshot.select(
                SCALING_GROUPS,
                columns=(
                    SCALING_GROUPS.c.name,
                    SCALING_GROUPS.c.consecutive_failures,
                    SCALING_GROUPS.c.backoff_until_ms,
                    SCALING_GROUPS.c.last_scale_up_ms,
                    SCALING_GROUPS.c.last_scale_down_ms,
                    SCALING_GROUPS.c.quota_exceeded_until_ms,
                    SCALING_GROUPS.c.quota_reason,
                ),
            )
            slice_rows = snapshot.select(
                SLICES,
                columns=(
                    SLICES.c.slice_id,
                    SLICES.c.scale_group,
                    SLICES.c.lifecycle,
                    SLICES.c.worker_ids,
                    SLICES.c.created_at_ms,
                    SLICES.c.last_active_ms,
                    SLICES.c.error_message,
                ),
            )
            tracked_rows = snapshot.select(
                TRACKED_WORKERS,
                columns=(
                    TRACKED_WORKERS.c.worker_id,
                    TRACKED_WORKERS.c.slice_id,
                    TRACKED_WORKERS.c.scale_group,
                    TRACKED_WORKERS.c.internal_address,
                ),
            )

        # Build GroupSnapshot objects from DB rows
        slices_by_group: dict[str, list[SliceSnapshot]] = {}
        for row in slice_rows:
            slices_by_group.setdefault(row.scale_group, []).append(
                SliceSnapshot(
                    slice_id=row.slice_id,
                    scale_group=row.scale_group,
                    lifecycle=row.lifecycle,
                    worker_ids=row.worker_ids,
                    created_at_ms=row.created_at_ms.epoch_ms(),
                    last_active_ms=row.last_active_ms.epoch_ms(),
                    error_message=row.error_message,
                )
            )

        group_snapshots: dict[str, GroupSnapshot] = {}
        for row in scaling_rows:
            group_snapshots[row.name] = GroupSnapshot(
                name=row.name,
                slices=slices_by_group.get(row.name, []),
                consecutive_failures=row.consecutive_failures,
                backoff_until_ms=row.backoff_until_ms.epoch_ms(),
                last_scale_up_ms=row.last_scale_up_ms.epoch_ms(),
                last_scale_down_ms=row.last_scale_down_ms.epoch_ms(),
                quota_exceeded_until_ms=row.quota_exceeded_until_ms.epoch_ms(),
                quota_reason=row.quota_reason,
            )

        tracked_worker_rows = [
            _TrackedWorkerRow(
                worker_id=row.worker_id,
                slice_id=row.slice_id,
                scale_group=row.scale_group,
                internal_address=row.internal_address,
            )
            for row in tracked_rows
        ]

        # Prefetch all managed slices in one shot (2 gcloud calls on GCP),
        # then partition by scale group for pure in-memory restore.
        all_cloud_slices = platform.list_all_slices()
        cloud_by_group: dict[str, list[SliceHandle]] = {}
        for handle in all_cloud_slices:
            cloud_by_group.setdefault(handle.scale_group, []).append(handle)

        for group_snap in group_snapshots.values():
            group = self._groups.get(group_snap.name)
            if group is None:
                logger.warning(
                    "Checkpoint references scaling group %s which does not exist in config, skipping",
                    group_snap.name,
                )
                continue
            restore_result = restore_scaling_group(
                group_snapshot=group_snap,
                cloud_handles=cloud_by_group.get(group_snap.name, []),
                label_prefix=group.label_prefix,
            )
            group.restore_from_snapshot(
                slices=restore_result.slices,
                consecutive_failures=restore_result.consecutive_failures,
                last_scale_up=restore_result.last_scale_up,
                last_scale_down=restore_result.last_scale_down,
                backoff_until=restore_result.backoff_until,
                quota_exceeded_until=restore_result.quota_exceeded_until,
                quota_reason=restore_result.quota_reason,
            )

        # Workers from discarded slices remain in the DB as healthy.
        # They will naturally fail heartbeat checks and be pruned once
        # consecutive failures exceed the threshold.
        restored_workers = _restore_tracked_workers(tracked_worker_rows)
        self.restore_tracked_workers(restored_workers)
        logger.info("Restored %d tracked workers", len(restored_workers))

    def get_vm(self, vm_id: str) -> vm_pb2.VmInfo | None:
        """Get VM info by platform worker ID from the centralized worker registry."""
        tracked = self._workers.get(vm_id)
        if not tracked:
            return None

        worker_status = tracked.handle.status()
        if worker_status.state == CloudWorkerState.RUNNING:
            iris_state = vm_pb2.VM_STATE_READY
        elif worker_status.state == CloudWorkerState.STOPPED:
            iris_state = vm_pb2.VM_STATE_FAILED
        elif worker_status.state == CloudWorkerState.TERMINATED:
            iris_state = vm_pb2.VM_STATE_TERMINATED
        else:
            iris_state = vm_pb2.VM_STATE_BOOTING

        return vm_pb2.VmInfo(
            vm_id=tracked.worker_id,
            state=iris_state,
            address=tracked.handle.internal_address,
            scale_group=tracked.scale_group,
            slice_id=tracked.slice_id,
        )

    def get_init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get bootstrap log for a VM by platform worker ID."""
        tracked = self._workers.get(vm_id)
        if not tracked:
            return ""
        log = tracked.bootstrap_log
        if tail and log:
            lines = log.splitlines()
            return "\n".join(lines[-tail:])
        return log

    def check_coscheduling_feasibility(
        self,
        replicas: int,
        constraints: list[cluster_pb2.Constraint],
    ) -> str | None:
        """Check if a coscheduled job with the given replicas can ever be scheduled.

        A coscheduled job is feasible when its replica count is an exact multiple of
        some matching group's num_vms (e.g. 4 VMs can serve 4, 8, 12, ... replicas).

        Returns None if feasible, or a human-readable error message if no scaling
        group can accommodate the replica count.
        """
        groups = list(self._groups.values())
        if not groups:
            return None

        group_attrs = {g.name: g.to_attributes() for g in groups}
        group_index = ConstraintIndex.build(group_attrs)
        routing_cs = routing_constraints(constraints)
        matching_names = group_index.matching_entities(routing_cs)
        matching_groups = [g for g in groups if g.name in matching_names]

        if not matching_groups:
            return f"no scaling group matches the job constraints; " f"available groups: {[g.name for g in groups]}"

        if any(replicas % g.num_vms == 0 for g in matching_groups):
            return None

        group_sizes = {g.name: g.num_vms for g in matching_groups}
        return (
            f"job requires {replicas} coscheduled replicas but no matching scaling group "
            f"has a compatible size (replicas must be an exact multiple of num_vms); "
            f"matching group sizes: {group_sizes}"
        )

    def get_status(self) -> vm_pb2.AutoscalerStatus:
        """Build status for the status API."""
        status = vm_pb2.AutoscalerStatus(
            groups=[g.to_status() for g in self._groups.values()],
            current_demand={g.name: g.current_demand for g in self._groups.values()},
            last_evaluation=self._last_evaluation.to_proto(),
            recent_actions=list(self._action_log),
        )
        if self._last_routing_decision is not None:
            status.last_routing_decision.CopyFrom(self._routing_decision_to_proto(self._last_routing_decision))
        return status

    def _routing_decision_to_proto(self, decision: RoutingDecision) -> vm_pb2.RoutingDecision:
        def _resource_spec_proto(resources: cluster_pb2.ResourceSpecProto) -> vm_pb2.ResourceSpec:
            gpu_count = 0
            tpu_count = 0
            if resources.HasField("device"):
                if resources.device.HasField("gpu"):
                    gpu_count = resources.device.gpu.count or 1
                if resources.device.HasField("tpu"):
                    tpu_count = resources.device.tpu.count or 0
            return vm_pb2.ResourceSpec(
                cpu_millicores=resources.cpu_millicores,
                memory_bytes=resources.memory_bytes,
                disk_bytes=resources.disk_bytes,
                gpu_count=gpu_count,
                tpu_count=tpu_count,
            )

        def _entry_to_proto(entry: DemandEntry) -> vm_pb2.DemandEntryStatus:
            n = entry.normalized
            return vm_pb2.DemandEntryStatus(
                task_ids=entry.task_ids,
                coschedule_group_id=entry.coschedule_group_id or "",
                device_type=n.device_type.value if n.device_type else "",
                device_variant=_format_variants(n.device_variants),
                preemptible=bool(n.preemptible),
                resources=_resource_spec_proto(entry.resources),
            )

        routed_entries = {
            name: vm_pb2.DemandEntryStatusList(entries=[_entry_to_proto(e) for e in entries])
            for name, entries in decision.routed_entries.items()
        }
        unmet_entries = [
            vm_pb2.UnmetDemand(entry=_entry_to_proto(u.entry), reason=u.reason) for u in decision.unmet_entries
        ]

        return vm_pb2.RoutingDecision(
            group_to_launch=decision.group_to_launch,
            group_reasons=decision.group_reasons,
            routed_entries=routed_entries,
            unmet_entries=unmet_entries,
            group_statuses=[
                vm_pb2.GroupRoutingStatus(
                    group=s.group,
                    priority=s.priority,
                    assigned=s.assigned,
                    launch=s.launch,
                    decision=s.decision,
                    reason=s.reason,
                )
                for s in decision.group_statuses
            ],
        )

    def get_group(self, name: str) -> ScalingGroup | None:
        """Get a scale group by name."""
        return self._groups.get(name)

    @property
    def groups(self) -> dict[str, ScalingGroup]:
        """All scale groups."""
        return self._groups

    def notify_worker_failed(self, worker_id: str) -> list[str]:
        """Called by controller when a worker fails. Terminates the containing slice.

        This integrates with the existing controller failure cascade:
        1. Controller detects worker timeout/failure
        2. Controller emits WorkerFailedEvent (cascades to tasks)
        3. Controller calls this method (with the failed worker's ID)
        4. Autoscaler terminates the slice containing the failed worker

        If the slice was short-lived (died soon after creation), applies backoff
        to the scale group to prevent thrashing on bad zones/preemption.

        Returns:
            List of sibling worker IDs from the same slice (excluding the
            originally failed worker). The controller uses these to immediately
            fail sibling workers, since the entire slice is being terminated.
        """
        slice_id, group = self._find_slice_for_worker(worker_id)
        if not slice_id or not group:
            logger.debug("Worker %s not found in any managed slice", worker_id)
            return []

        sibling_worker_ids = [wid for wid in group.get_slice_worker_ids(slice_id) if wid != worker_id]

        logger.info("Worker %s failed, terminating slice %s", worker_id, slice_id)
        self._log_action(
            "worker_failed",
            group.name,
            slice_id=slice_id,
            reason=f"worker {worker_id} failed",
        )

        # Check if this was a short-lived slice (preemption detection)
        self._record_slice_failure(slice_id, group)

        group.scale_down(slice_id)
        self._unregister_slice_workers(slice_id)

        return sibling_worker_ids

    def _find_slice_for_worker(self, worker_id: str) -> tuple[str | None, ScalingGroup | None]:
        """Find the slice and group containing a worker by worker ID."""
        for group in self._groups.values():
            slice_id = group.find_slice_for_worker(worker_id)
            if slice_id is not None:
                return slice_id, group
        return None, None

    def _record_slice_failure(self, slice_id: str, group: ScalingGroup) -> None:
        """Record slice failure and apply backoff if it was short-lived.

        Short-lived slices (died within SHORT_LIVED_SLICE_THRESHOLD of creation)
        indicate bad zone/quota issues or preemption. Apply backoff to prevent thrashing.
        Uses slice handle's created_at timestamp retrieved from the ScalingGroup.
        """
        slice_handle = group.get_slice(slice_id)
        if slice_handle is None:
            return

        age_ms = Timestamp.now().epoch_ms() - slice_handle.created_at.epoch_ms()
        age = Duration.from_ms(age_ms)
        if age < SHORT_LIVED_SLICE_THRESHOLD:
            logger.warning(
                "Short-lived slice %s (age=%dms) in %s, applying backoff",
                slice_id,
                age_ms,
                group.name,
            )
            group.record_failure()
            self._log_action(
                "backoff_triggered",
                group.name,
                slice_id=slice_id,
                reason=f"short-lived slice (age={age_ms}ms)",
            )
