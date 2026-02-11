# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler manages scaling across scale groups.

The autoscaler coordinates scaling decisions across multiple scale groups,
delegating slice ownership to ScalingGroup.

Key design principles:
- Autoscaler does NOT track slices directly - that's ScalingGroup's job
- Scale-up decisions come from Autoscaler, scale-down is delegated to ScalingGroup
- ScalingGroup owns per-slice idle tracking and decides which slices to scale down

The run_once() flow splits into two phases:
- refresh(): I/O phase — cleanup failed/dead slices, scale down idle
- update(): CPU phase — evaluate demand and execute scale-up decisions

Status queries use cached VmGroupStatus from ScalingGroup monitor threads.
The remaining blocking operations are terminate() calls for failed/dead/idle slices.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

from iris.cluster.platform.base import Platform, QuotaExhaustedError, SliceHandle, VmHandle
from iris.cluster.platform.bootstrap import WorkerBootstrap
from iris.cluster.types import DeviceType, VmWorkerStatusMap
from iris.cluster.controller.scaling_group import GroupAvailability, ScalingGroup, SliceLifecycleState
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import cluster_pb2, config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


@dataclass
class TrackedVm:
    """Per-VM state tracked by the autoscaler across bootstrap and lifecycle."""

    vm_id: str
    slice_id: str
    scale_group: str
    handle: VmHandle
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

    def can_fit_group(group: ScalingGroup, entry: DemandEntry, *, check_accept: bool = True) -> bool:
        if not group.matches_device_requirement(entry.device_type, entry.device_variant):
            return False
        if entry.preemptible is not None and group.config.slice_template.preemptible != entry.preemptible:
            return False
        if entry.invalid_reason:
            return False
        if entry.coschedule_group_id and group.slice_size != len(entry.task_ids):
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
        pending_slices = (
            counts.get(SliceLifecycleState.REQUESTING, 0)
            + counts.get(SliceLifecycleState.BOOTING, 0)
            + counts.get(SliceLifecycleState.INITIALIZING, 0)
        )
        current = sum(counts.values())
        headroom = group.max_slices - current
        return PendingGroup(
            name=group.name,
            pending_slices=pending_slices,
            remaining_slices=pending_slices + headroom,
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
        ]
        if not matching_groups:
            reason = (
                f"no_matching_group: need device={entry.device_type}:{entry.device_variant}"
                f", available groups={[g.name for g in sorted_groups]}"
            )
            unmet.append(UnmetDemand(entry=entry, reason=reason))
            continue

        if entry.coschedule_group_id and not any(g.slice_size == len(entry.task_ids) for g in matching_groups):
            group_sizes = [g.slice_size for g in matching_groups]
            reason = (
                f"coschedule_mismatch: job needs {len(entry.task_ids)} tasks coscheduled"
                f" but matching groups have slice_size={group_sizes}"
            )
            unmet.append(UnmetDemand(entry=entry, reason=reason))
            continue

        if not any(g.can_fit_resources(entry.resources) for g in matching_groups):
            reason = f"insufficient_resources: task needs {entry.resources}" f" but no matching group can fit it"
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

    return RoutingDecision(
        group_to_launch=group_to_launch,
        routed_entries=routed,
        unmet_entries=unmet,
        group_reasons=group_reasons,
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
        worker_bootstrap: WorkerBootstrap | None = None,
    ):
        """Create autoscaler with explicit parameters.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            evaluation_interval: How often to evaluate scaling decisions
            platform: Platform instance for shutdown lifecycle
            threads: Optional thread container for testing
            worker_bootstrap: WorkerBootstrap for initializing new VMs (None in test/local mode)
        """
        self._groups = scale_groups
        self._platform = platform
        self.evaluation_interval = evaluation_interval
        self._worker_bootstrap = worker_bootstrap

        # Centralized per-VM state indexed by vm_id
        self._vms: dict[str, TrackedVm] = {}

        # Bounded log of recent autoscaler actions for dashboard/debugging
        self._action_log: deque[vm_pb2.AutoscalerAction] = deque(maxlen=100)

        # Most recent routing decision (for status API)
        self._last_routing_decision: RoutingDecision | None = None

        # Thread management
        self._threads = threads if threads is not None else get_thread_container()

    @classmethod
    def from_config(
        cls,
        scale_groups: dict[str, ScalingGroup],
        config: config_pb2.AutoscalerConfig,
        platform: Platform,
        threads: ThreadContainer | None = None,
        worker_bootstrap: WorkerBootstrap | None = None,
    ) -> Autoscaler:
        """Create autoscaler from proto config.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            config: Autoscaler configuration proto (with defaults already applied)
            platform: Platform instance for shutdown lifecycle
            threads: Optional thread container for testing
            worker_bootstrap: WorkerBootstrap for initializing new VMs

        Returns:
            Configured Autoscaler instance
        """
        return cls(
            scale_groups=scale_groups,
            evaluation_interval=Duration.from_proto(config.evaluation_interval),
            platform=platform,
            threads=threads,
            worker_bootstrap=worker_bootstrap,
        )

    def _wait_for_inflight(self) -> None:
        """Wait for in-flight scale-ups to complete without terminating anything.

        Test-only: Waits for all scale-up threads to complete.
        """
        # Wait for all threads in the container to finish
        self._threads.wait()

    def start_monitors(self) -> None:
        """Start per-group monitor threads that periodically refresh slice status."""
        for group in self._groups.values():
            group.start_monitor()

    def stop_monitors(self) -> None:
        """Stop per-group monitor threads."""
        for group in self._groups.values():
            group.stop_monitor()

    def shutdown(self) -> None:
        """Shutdown the autoscaler, terminate all VM groups, and clean up platform.

        Shutdown ordering:
        1. Stop monitor threads so no new status refreshes occur.
        2. Stop all threads in the autoscaler's ThreadContainer. This signals
           stop_events for both in-flight scale-up threads AND worker lifecycle
           threads (via child containers), then joins with timeout.
        3. Terminate all VM groups — calls Worker.stop() for final cleanup
           of any workers that didn't exit in step 2.
        4. Shutdown platform — clears local tracking state.
        """
        self.stop_monitors()

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

    def reconcile(self) -> None:
        """Reconcile all groups (discover existing slices from cloud).

        Called once at startup to recover state from a previous controller.
        Each ScalingGroup queries its VmManager for existing slices.
        """
        for group in self._groups.values():
            group.reconcile()
            for slice_obj in group.slice_handles():
                self._register_slice_vms(slice_obj, group.name)

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
        Use _execute_scale_up instead.

        Returns:
            True if scale-up succeeded, False otherwise.
        """
        action = self._log_action("scale_up", group.name, reason=reason, status="pending")

        completed = False
        slice_obj: SliceHandle | None = None
        try:
            logger.info("Scaling up %s: %s", group.name, reason)
            slice_obj = group.scale_up(timestamp=ts)
            group.complete_scale_up(slice_obj, ts)
            completed = True
            logger.info("Created slice %s for group %s", slice_obj.slice_id, group.name)
            action.slice_id = slice_obj.slice_id
            action.status = "completed"
            self._bootstrap_slice(slice_obj, group.name)
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
            if completed and slice_obj is not None:
                logger.error(
                    "Bootstrap failed for slice %s in %s, cleaning up: %s",
                    slice_obj.slice_id,
                    group.name,
                    e,
                )
                group.scale_down(slice_obj.slice_id)
                self._unregister_slice_vms(slice_obj.slice_id)
            else:
                group.cancel_scale_up()
            logger.error("Failed to create slice for %s: %s", group.name, e)
            action.status = "failed"
            action.reason = f"{reason} - error: {e}"
            group.record_failure(ts)
            return False

    def _bootstrap_slice(self, handle: SliceHandle, scale_group: str) -> None:
        """Bootstrap all VMs in a newly created slice and register them in the VM registry.

        Collects bootstrap logs from the bootstrap process (if available) and
        stores them on TrackedVm entries for later retrieval via get_init_log().
        """
        bootstrap_logs: dict[str, str] = {}
        if self._worker_bootstrap:
            bootstrap_logs = self._worker_bootstrap.bootstrap_slice(handle)

        self._register_slice_vms(handle, scale_group, bootstrap_logs)

    def _register_slice_vms(
        self,
        handle: SliceHandle,
        scale_group: str,
        bootstrap_logs: dict[str, str] | None = None,
    ) -> None:
        """Register all VMs from a slice handle into the VM registry."""
        logs = bootstrap_logs or {}
        for vm in handle.describe().vms:
            self._vms[vm.vm_id] = TrackedVm(
                vm_id=vm.vm_id,
                slice_id=handle.slice_id,
                scale_group=scale_group,
                handle=vm,
                bootstrap_log=logs.get(vm.vm_id, ""),
            )

    def _unregister_slice_vms(self, slice_id: str) -> None:
        """Remove all TrackedVm entries belonging to a slice."""
        to_remove = [vm_id for vm_id, tv in self._vms.items() if tv.slice_id == slice_id]
        for vm_id in to_remove:
            del self._vms[vm_id]

    def refresh(self, vm_status_map: VmWorkerStatusMap, timestamp: Timestamp | None = None) -> None:
        """I/O phase: cleanup failed/dead slices, scale down idle.

        Status queries use cached VmGroupStatus from monitor threads.
        The remaining blocking operations are terminate() calls.
        """
        timestamp = timestamp or Timestamp.now()

        for group in self._groups.values():
            cleaned = group.cleanup_failed_slices(timestamp)
            for slice_obj in cleaned:
                self._unregister_slice_vms(slice_obj.slice_id)
                self._log_action(
                    "failed_cleanup",
                    group.name,
                    slice_id=slice_obj.slice_id,
                    reason="cleaning up failed slice",
                )

            group.update_slice_liveness(vm_status_map, timestamp)
            dead = group.cleanup_dead_slices(timestamp)
            for slice_obj in dead:
                self._unregister_slice_vms(slice_obj.slice_id)
                self._log_action(
                    "liveness_reap",
                    group.name,
                    slice_id=slice_obj.slice_id,
                    reason="slice missed liveness deadline",
                )

        for group in self._groups.values():
            target_capacity = max(group.current_demand, group.min_slices)
            scaled_down = group.scale_down_if_idle(vm_status_map, target_capacity, timestamp)
            if scaled_down:
                self._unregister_slice_vms(scaled_down.slice_id)
                self._log_action(
                    "scale_down",
                    group.name,
                    slice_id=scaled_down.slice_id,
                    reason=f"idle slice (target={target_capacity}, ready={group.ready_slice_count() + 1})",
                )

    def update(
        self,
        demand_entries: list[DemandEntry],
        timestamp: Timestamp | None = None,
    ) -> list[ScalingDecision]:
        """CPU phase: evaluate demand and execute scale-up decisions."""
        timestamp = timestamp or Timestamp.now()

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
        """Get VM info by ID from the centralized VM registry."""
        tracked = self._vms.get(vm_id)
        if not tracked:
            return None

        from iris.cluster.controller.scaling_group import _cloud_vm_state_to_iris

        vm_status = tracked.handle.status()
        iris_state = _cloud_vm_state_to_iris(vm_status.state)

        return vm_pb2.VmInfo(
            vm_id=tracked.vm_id,
            state=iris_state,
            address=tracked.handle.internal_address,
            scale_group=tracked.scale_group,
            slice_id=tracked.slice_id,
        )

    def get_init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get bootstrap log for a VM from the centralized VM registry."""
        tracked = self._vms.get(vm_id)
        if not tracked:
            return ""
        log = tracked.bootstrap_log
        if tail and log:
            lines = log.splitlines()
            return "\n".join(lines[-tail:])
        return log

    def get_status(self) -> vm_pb2.AutoscalerStatus:
        """Build status for the status API."""
        from iris.rpc import time_pb2

        status = vm_pb2.AutoscalerStatus(
            groups=[g.to_status() for g in self._groups.values()],
            current_demand={g.name: g.current_demand for g in self._groups.values()},
            last_evaluation=time_pb2.Timestamp(epoch_ms=0),  # Controlled by controller now
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
            self._unregister_slice_vms(slice_id)
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
