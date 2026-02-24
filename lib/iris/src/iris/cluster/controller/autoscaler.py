# Copyright 2025 The Marin Authors
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
- refresh(): I/O phase — poll slice states, scale down idle slices
- update(): CPU phase — evaluate demand and execute scale-up decisions
"""

from __future__ import annotations

import difflib
import json
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    Platform,
    QuotaExhaustedError,
    RemoteWorkerHandle,
    SliceHandle,
)
from iris.cluster.platform.bootstrap import rewrite_ghcr_to_ar_remote, zone_to_multi_region
from iris.cluster.types import DeviceType, REGION_ATTRIBUTE_KEY, VmWorkerStatusMap, ZONE_ATTRIBUTE_KEY
from iris.cluster.controller.scaling_group import GroupAvailability, ScalingGroup, SliceLifecycleState
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import cluster_pb2, config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


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
    """A demand entry specifying resource requirements and constraints."""

    task_ids: list[str]
    coschedule_group_id: str | None
    device_type: DeviceType
    device_variant: str | None  # None = any variant of this type
    constraints: list[cluster_pb2.Constraint]
    resources: cluster_pb2.ResourceSpecProto
    preemptible: bool | None = None  # None = no preference
    required_regions: frozenset[str] | None = None
    required_zones: frozenset[str] | None = None
    invalid_reason: str | None = None


@dataclass
class PendingGroup:
    name: str
    pending_slices: int
    remaining_slices: int
    assigned_entries: list[DemandEntry]
    reason: str


@dataclass
class UnmetDemand:
    entry: DemandEntry
    reason: str


@dataclass
class RoutingDecision:
    group_to_launch: dict[str, int]
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


def _diagnose_no_matching_group(
    entry: DemandEntry,
    groups: list[ScalingGroup],
    group_region: Callable[[ScalingGroup], str | None],
    group_zone: Callable[[ScalingGroup], str | None],
) -> str:
    """Produce a concise, actionable reason when no group matches a demand entry.

    Checks filters in order (device → preemptible → zone → region) and reports
    the first mismatch with enough context to fix the issue.
    """
    device_matches = [g for g in groups if g.matches_device_requirement(entry.device_type, entry.device_variant)]

    if not device_matches:
        return f"no_matching_group: no groups with device {entry.device_type.value}:{entry.device_variant or '*'}"

    if entry.preemptible is not None:
        preempt_matches = [g for g in device_matches if g.config.slice_template.preemptible == entry.preemptible]
        if not preempt_matches:
            want = "preemptible" if entry.preemptible else "non-preemptible"
            return (
                f"no_matching_group: no {want} groups for device {entry.device_type.value}:{entry.device_variant or '*'}"
            )
        device_matches = preempt_matches

    if entry.required_zones:
        available_zones = {group_zone(g) for g in device_matches} - {None}
        requested = sorted(entry.required_zones)
        msg = f"no_matching_group: no groups in zone {', '.join(requested)}"
        for req_zone in requested:
            close = difflib.get_close_matches(req_zone, available_zones, n=1, cutoff=0.7)
            if close:
                msg += f" (did you mean {close[0]}?)"
        return msg

    if entry.required_regions:
        requested = sorted(entry.required_regions)
        return f"no_matching_group: no groups in region {', '.join(requested)}"

    return f"no_matching_group: no groups match device={entry.device_type.value}:{entry.device_variant or '*'}"


def route_demand(
    groups: list[ScalingGroup],
    demand_entries: list[DemandEntry],
    timestamp: Timestamp | None = None,
) -> RoutingDecision:
    """Route demand to groups based on requirements and priority."""
    ts = timestamp or Timestamp.now()
    sorted_groups = sorted(groups, key=lambda g: g.config.priority or 100)
    group_by_name = {g.name: g for g in sorted_groups}

    pending: dict[str, PendingGroup] = {}
    routed: dict[str, list[DemandEntry]] = {}
    unmet: list[UnmetDemand] = []
    group_reasons: dict[str, str] = {}

    def group_region(group: ScalingGroup) -> str | None:
        if group.config.HasField("worker"):
            region = group.config.worker.attributes.get(REGION_ATTRIBUTE_KEY, "").strip()
            if region:
                return region
        template = group.config.slice_template
        if template.HasField("gcp") and template.gcp.zone:
            return template.gcp.zone.rsplit("-", 1)[0]
        if template.HasField("coreweave") and template.coreweave.region:
            return template.coreweave.region
        return None

    def group_zone(group: ScalingGroup) -> str | None:
        if group.config.HasField("worker"):
            zone = group.config.worker.attributes.get(ZONE_ATTRIBUTE_KEY, "").strip()
            if zone:
                return zone
        template = group.config.slice_template
        if template.HasField("gcp") and template.gcp.zone:
            return template.gcp.zone
        if template.HasField("coreweave") and template.coreweave.region:
            return template.coreweave.region
        return None

    def can_fit_group(group: ScalingGroup, entry: DemandEntry, *, check_accept: bool = True) -> bool:
        if not group.matches_device_requirement(entry.device_type, entry.device_variant):
            return False
        if entry.preemptible is not None and group.config.slice_template.preemptible != entry.preemptible:
            return False
        if entry.required_regions:
            region = group_region(group)
            if region not in entry.required_regions:
                return False
        if entry.required_zones:
            zone = group_zone(group)
            if zone not in entry.required_zones:
                return False
        if entry.invalid_reason:
            return False
        if entry.coschedule_group_id and group.num_vms != len(entry.task_ids):
            return False
        if check_accept and not group.can_accept_demand(ts):
            return False
        if not group.can_fit_resources(entry.resources):
            return False
        return True

    def can_fit_pending(pg: PendingGroup, group: ScalingGroup, entry: DemandEntry) -> bool:
        if pg.remaining_slices <= 0:
            return False
        return can_fit_group(group, entry, check_accept=False)

    def assign(pg: PendingGroup, entry: DemandEntry) -> None:
        pg.remaining_slices -= 1
        pg.assigned_entries.append(entry)
        routed.setdefault(pg.name, []).append(entry)

    def make_pending(group: ScalingGroup) -> PendingGroup:
        counts = group.slice_state_counts()
        inflight = (
            counts.get(SliceLifecycleState.REQUESTING, 0)
            + counts.get(SliceLifecycleState.BOOTING, 0)
            + counts.get(SliceLifecycleState.INITIALIZING, 0)
        )
        ready = counts.get(SliceLifecycleState.READY, 0)
        current = sum(counts.values())
        headroom = group.max_slices - current

        if headroom > 0:
            # Group can still create new slices. Only count in-flight
            # slices as pending capacity; headroom determines how many
            # new demand entries can trigger scale-ups.
            return PendingGroup(
                name=group.name,
                pending_slices=inflight,
                remaining_slices=inflight + headroom,
                assigned_entries=[],
                reason="demand-routed",
            )

        # Group is at max_slices (AT_CAPACITY). Include ready slices so
        # demand can still be routed here for demand tracking. Without
        # this, current_demand drops to 0 when a group hits max_slices,
        # causing immediate scale-down of newly ready slices.
        existing_capacity = inflight + ready
        return PendingGroup(
            name=group.name,
            pending_slices=existing_capacity,
            remaining_slices=existing_capacity,
            assigned_entries=[],
            reason="demand-routed",
        )

    for group in sorted_groups:
        if group.availability(ts).status == GroupAvailability.REQUESTING:
            pending[group.name] = make_pending(group)

    for entry in demand_entries:
        if entry.invalid_reason:
            unmet.append(UnmetDemand(entry=entry, reason=entry.invalid_reason))
            continue

        matching_groups = [
            g
            for g in sorted_groups
            if g.matches_device_requirement(entry.device_type, entry.device_variant)
            and (entry.preemptible is None or g.config.slice_template.preemptible == entry.preemptible)
            and (not entry.required_regions or group_region(g) in entry.required_regions)
            and (not entry.required_zones or group_zone(g) in entry.required_zones)
        ]
        if not matching_groups:
            reason = _diagnose_no_matching_group(entry, sorted_groups, group_region, group_zone)
            unmet.append(UnmetDemand(entry=entry, reason=reason))
            continue

        if entry.coschedule_group_id and not any(g.num_vms == len(entry.task_ids) for g in matching_groups):
            group_sizes = [g.num_vms for g in matching_groups]
            reason = (
                f"coschedule_mismatch: job needs {len(entry.task_ids)} tasks coscheduled"
                f" but matching groups have num_vms={group_sizes}"
            )
            unmet.append(UnmetDemand(entry=entry, reason=reason))
            continue

        fit_reasons = [g.check_resource_fit(entry.resources) for g in matching_groups]
        if all(r is not None for r in fit_reasons):
            details = "; ".join(r for r in fit_reasons if r is not None)
            reason = f"insufficient_resources: {details}"
            unmet.append(UnmetDemand(entry=entry, reason=reason))
            continue

        matched_pending = False
        for name, pg in pending.items():
            group = group_by_name.get(name)
            if group is None:
                continue
            if can_fit_pending(pg, group, entry):
                assign(pg, entry)
                matched_pending = True
                break
        if matched_pending:
            continue

        matched_group = False
        for group in sorted_groups:
            if not can_fit_group(group, entry):
                continue
            if group.name not in pending:
                pending[group.name] = make_pending(group)
            if pending[group.name].remaining_slices <= 0:
                continue
            assign(pending[group.name], entry)
            matched_group = True
            group_reasons.setdefault(group.name, "demand-routed")
            break

        if not matched_group:
            unmet.append(UnmetDemand(entry=entry, reason="no_capacity"))

    group_to_launch: dict[str, int] = {}
    for name, pg in pending.items():
        if not pg.assigned_entries:
            continue
        needed = max(0, len(pg.assigned_entries) - pg.pending_slices)
        group_to_launch[name] = needed

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
        elif availability.status == GroupAvailability.AT_CAPACITY:
            decision = "at_capacity"
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

    return RoutingDecision(
        group_to_launch=group_to_launch,
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
        bootstrap_config: config_pb2.BootstrapConfig | None = None,
        gcp_project: str = "",
    ):
        """Create autoscaler with explicit parameters.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            evaluation_interval: How often to evaluate scaling decisions
            platform: Platform instance for shutdown lifecycle
            threads: Optional thread container for testing
            bootstrap_config: Worker bootstrap settings passed to platform.create_slice().
                None disables bootstrap (test/local mode).
            gcp_project: GCP project ID for AR remote repo image rewriting.
        """
        self._groups = scale_groups
        self._platform = platform
        self.evaluation_interval = evaluation_interval
        self._bootstrap_config = bootstrap_config
        self._gcp_project = gcp_project

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
        bootstrap_config: config_pb2.BootstrapConfig | None = None,
        gcp_project: str = "",
    ) -> Autoscaler:
        """Create autoscaler from proto config.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            config: Autoscaler configuration proto (with defaults already applied)
            platform: Platform instance for shutdown lifecycle
            threads: Optional thread container for testing
            bootstrap_config: Worker bootstrap settings passed to platform.create_slice()
            gcp_project: GCP project ID for AR remote repo image rewriting.

        Returns:
            Configured Autoscaler instance
        """
        return cls(
            scale_groups=scale_groups,
            evaluation_interval=Duration.from_proto(config.evaluation_interval),
            platform=platform,
            threads=threads,
            bootstrap_config=bootstrap_config,
            gcp_project=gcp_project,
        )

    def _wait_for_inflight(self) -> None:
        """Wait for in-flight scale-ups to complete without terminating anything.

        Test-only: Waits for all scale-up threads to complete.
        """
        # Wait for all threads in the container to finish
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
            logger.error(
                "CAPACITY INSUFFICIENT: %d demand entries cannot be satisfied by any group",
                len(result.unmet_entries),
            )
            for unmet in result.unmet_entries[:10]:
                entry = unmet.entry
                logger.warning(
                    "Unmet demand: reason=%s device=%s:%s resources=%s tasks=%s",
                    unmet.reason,
                    entry.device_type,
                    entry.device_variant,
                    entry.resources,
                    entry.task_ids,
                )

        decisions = []
        for name, group in self._groups.items():
            allocated_entries = result.routed_entries.get(name, [])
            demand = len(allocated_entries)
            group.update_demand(demand)
            decision = self._evaluate_group(group, demand, ts)
            if decision:
                decisions.append(decision)

        return decisions

    def _evaluate_group(
        self,
        group: ScalingGroup,
        demand: int,
        ts: Timestamp,
    ) -> ScalingDecision | None:
        """Evaluate scaling decision for a single group."""
        counts = group.slice_state_counts()
        ready = counts[SliceLifecycleState.READY]
        requesting = counts[SliceLifecycleState.REQUESTING]
        pending = counts[SliceLifecycleState.BOOTING] + counts[SliceLifecycleState.INITIALIZING] + requesting
        total = sum(counts.values())

        capacity = ready + pending

        logger.debug(
            "Evaluating group %s: total=%d, ready=%d, pending=%d, demand=%d, min=%d, max=%d",
            group.name,
            total,
            ready,
            pending,
            demand,
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
                return None

            return ScalingDecision(
                scale_group=group.name,
                action=ScalingAction.SCALE_UP,
                reason=f"below min_slices ({total} < {group.min_slices})",
            )

        # Priority 2: Scale UP for demand exceeding capacity
        if demand > capacity and total < group.max_slices:
            if not group.can_scale_up(ts):
                logger.debug("Scale group %s: scale up blocked", group.name)
                return None

            return ScalingDecision(
                scale_group=group.name,
                action=ScalingAction.SCALE_UP,
                reason=f"demand={demand} > capacity={capacity}",
            )

        return None

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
                self._execute_scale_up(group, timestamp, reason=decision.reason)

    def _execute_scale_up(self, group: ScalingGroup, ts: Timestamp, reason: str = "") -> None:
        """Initiate async scale-up for a scale group.

        Increments the group's pending scale-up counter and spawns a background
        thread for the actual scale-up work. The counter is included in
        slice_count(), preventing double scale-up.
        """
        group.begin_scale_up()

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
            bc = self._per_group_bootstrap_config(group)
            slice_obj = group.scale_up(bootstrap_config=bc, timestamp=ts)
            group.complete_scale_up(slice_obj, ts)
            logger.info("Created slice %s for group %s", slice_obj.slice_id, group.name)
            action.slice_id = slice_obj.slice_id
            action.status = "completed"
            self._register_slice_workers(slice_obj, group.name)
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

    def _per_group_bootstrap_config(self, group: ScalingGroup) -> config_pb2.BootstrapConfig | None:
        """Build a per-group BootstrapConfig by merging worker attributes, image rewrite, and accelerator settings.

        Copies the base bootstrap config and:
        1. Rewrites GHCR docker_image to an AR remote repo for the group's continent
        2. Injects IRIS_WORKER_ATTRIBUTES, IRIS_TASK_DEFAULT_ENV_JSON, and
           IRIS_SCALE_GROUP from the group's worker settings into env_vars.
        3. Injects accelerator type/variant/GPU count env vars for the group.
        """
        if not self._bootstrap_config:
            return None

        has_worker = group.config.HasField("worker")

        bc = config_pb2.BootstrapConfig()
        bc.CopyFrom(self._bootstrap_config)

        # Rewrite GHCR image to AR remote repo for this group's continent
        template = group.config.slice_template
        if template.HasField("gcp") and template.gcp.zone and bc.docker_image.startswith("ghcr.io/"):
            multi_region = zone_to_multi_region(template.gcp.zone)
            if multi_region:
                project = self._gcp_project
                assert project, "gcp_project required for GHCR→AR worker image rewrite"
                bc.docker_image = rewrite_ghcr_to_ar_remote(bc.docker_image, multi_region, project)

        if has_worker:
            attributes = dict(group.config.worker.attributes)
            if attributes:
                bc.env_vars["IRIS_WORKER_ATTRIBUTES"] = json.dumps(attributes, sort_keys=True)
            if group.config.worker.env:
                bc.env_vars["IRIS_TASK_DEFAULT_ENV_JSON"] = json.dumps(dict(group.config.worker.env))

        if group.config.name:
            bc.env_vars["IRIS_SCALE_GROUP"] = group.config.name
        accel_type = group.config.accelerator_type
        if accel_type == config_pb2.ACCELERATOR_TYPE_GPU:
            bc.env_vars["IRIS_ACCELERATOR_TYPE"] = "gpu"
        elif accel_type == config_pb2.ACCELERATOR_TYPE_TPU:
            bc.env_vars["IRIS_ACCELERATOR_TYPE"] = "tpu"
        elif accel_type == config_pb2.ACCELERATOR_TYPE_CPU:
            bc.env_vars["IRIS_ACCELERATOR_TYPE"] = "cpu"
        if group.config.accelerator_variant:
            bc.env_vars["IRIS_ACCELERATOR_VARIANT"] = group.config.accelerator_variant
        if (
            accel_type == config_pb2.ACCELERATOR_TYPE_GPU
            and group.config.HasField("resources")
            and group.config.resources.gpu_count > 0
        ):
            bc.env_vars["IRIS_GPU_COUNT"] = str(group.config.resources.gpu_count)
        return bc

    def _register_slice_workers(
        self,
        handle: SliceHandle,
        scale_group: str,
    ) -> None:
        """Register all workers from a slice handle into the worker registry."""
        for worker in handle.describe().workers:
            self._workers[worker.worker_id] = TrackedWorker(
                worker_id=worker.worker_id,
                slice_id=handle.slice_id,
                scale_group=scale_group,
                handle=worker,
                bootstrap_log=worker.bootstrap_log,
            )

    def _unregister_slice_workers(self, slice_id: str) -> None:
        """Remove all TrackedWorker entries belonging to a slice."""
        to_remove = [wid for wid, tw in self._workers.items() if tw.slice_id == slice_id]
        for wid in to_remove:
            del self._workers[wid]

    def refresh(self, vm_status_map: VmWorkerStatusMap, timestamp: Timestamp | None = None) -> None:
        """I/O phase: poll non-READY slices for state transitions, then scale down idle slices."""
        timestamp = timestamp or Timestamp.now()

        self._poll_slice_states()

        for group in self._groups.values():
            target_capacity = max(group.current_demand, group.min_slices)
            scaled_down = group.scale_down_if_idle(vm_status_map, target_capacity, timestamp)
            if scaled_down:
                self._unregister_slice_workers(scaled_down.slice_id)
                self._log_action(
                    "scale_down",
                    group.name,
                    slice_id=scaled_down.slice_id,
                    reason=f"idle slice (target={target_capacity}, ready={group.ready_slice_count() + 1})",
                )

    def _poll_slice_states(self) -> None:
        """Poll describe() on non-READY slices to detect state transitions.

        When a platform's internal bootstrap completes, describe() will return
        READY. This method detects that transition and updates the ScalingGroup.
        """
        for group in self._groups.values():
            with group._slices_lock:
                snapshot = list(group._slices.items())
            for slice_id, slice_state in snapshot:
                if slice_state.lifecycle in (SliceLifecycleState.BOOTING, SliceLifecycleState.INITIALIZING):
                    try:
                        status = slice_state.handle.describe()
                    except Exception as e:
                        logger.warning("Failed to poll slice %s: %s", slice_id, e)
                        continue
                    if status.state == CloudSliceState.READY:
                        addrs = [w.internal_address for w in status.workers]
                        group.mark_slice_ready(slice_id, addrs)
                        self._register_slice_workers(slice_state.handle, group.name)
                        self._log_action(
                            "slice_ready",
                            group.name,
                            slice_id,
                            reason=f"bootstrap completed ({len(addrs)} workers)",
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
        vm_status_map: VmWorkerStatusMap,
        timestamp: Timestamp | None = None,
    ) -> list[ScalingDecision]:
        """Full cycle: refresh + update. Preserved for tests."""
        timestamp = timestamp or Timestamp.now()
        logger.debug("Autoscaler run_once: demand_entries=%s", demand_entries)
        self.refresh(vm_status_map, timestamp)
        return self.update(demand_entries, timestamp)

    def get_vm(self, vm_id: str) -> vm_pb2.VmInfo | None:
        """Get worker info by ID from the centralized worker registry."""
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
        """Get bootstrap log for a worker from the centralized worker registry."""
        tracked = self._workers.get(vm_id)
        if not tracked:
            return ""
        log = tracked.bootstrap_log
        if tail and log:
            lines = log.splitlines()
            return "\n".join(lines[-tail:])
        return log

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
        def _device_type_to_proto(device_type: DeviceType) -> int:
            if device_type == DeviceType.GPU:
                return config_pb2.ACCELERATOR_TYPE_GPU
            if device_type == DeviceType.TPU:
                return config_pb2.ACCELERATOR_TYPE_TPU
            return config_pb2.ACCELERATOR_TYPE_CPU

        def _resource_spec_proto(resources: cluster_pb2.ResourceSpecProto) -> vm_pb2.ResourceSpec:
            gpu_count = 0
            tpu_count = 0
            if resources.HasField("device"):
                if resources.device.HasField("gpu"):
                    gpu_count = resources.device.gpu.count or 1
                if resources.device.HasField("tpu"):
                    tpu_count = resources.device.tpu.count or 0
            return vm_pb2.ResourceSpec(
                cpu=resources.cpu,
                memory_bytes=resources.memory_bytes,
                disk_bytes=resources.disk_bytes,
                gpu_count=gpu_count,
                tpu_count=tpu_count,
            )

        def _entry_to_proto(entry: DemandEntry) -> vm_pb2.DemandEntryStatus:
            return vm_pb2.DemandEntryStatus(
                task_ids=entry.task_ids,
                coschedule_group_id=entry.coschedule_group_id or "",
                accelerator_type=_device_type_to_proto(entry.device_type),
                accelerator_variant=entry.device_variant or "",
                preemptible=bool(entry.preemptible),
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

    def notify_worker_failed(self, vm_address: str) -> None:
        """Called by controller when a worker fails. Terminates the containing slice.

        This integrates with the existing controller failure cascade:
        1. Controller detects worker timeout/failure
        2. Controller emits WorkerFailedEvent (cascades to tasks)
        3. Controller calls this method (with worker's vm_address)
        4. Autoscaler terminates the slice containing the failed worker

        If the slice was short-lived (died soon after creation), applies backoff
        to the scale group to prevent thrashing on bad zones/preemption.
        """
        slice_id, group = self._find_slice_for_worker(vm_address)
        if not slice_id or not group:
            logger.debug("VM %s not found in any managed slice", vm_address)
            return

        logger.info("Worker at VM %s failed, terminating slice %s", vm_address, slice_id)
        self._log_action(
            "worker_failed",
            group.name,
            slice_id=slice_id,
            reason=f"worker at VM {vm_address} failed",
        )

        # Check if this was a short-lived slice (preemption detection)
        self._record_slice_failure(slice_id, group)

        try:
            group.scale_down(slice_id)
            self._unregister_slice_workers(slice_id)
        except Exception as e:
            logger.warning("Failed to terminate slice %s: %s", slice_id, e)

    def _find_slice_for_worker(self, vm_address: str) -> tuple[str | None, ScalingGroup | None]:
        """Find the slice and group containing a worker by VM address."""
        for group in self._groups.values():
            slice_id = group.find_slice_for_vm(vm_address)
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
