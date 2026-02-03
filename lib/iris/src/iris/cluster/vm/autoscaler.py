# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autoscaler manages scaling across scale groups.

The autoscaler coordinates scaling decisions across multiple scale groups,
delegating slice ownership to ScalingGroup and VM tracking to VmRegistry.

Key design principles:
- Autoscaler does NOT own VMs directly - that's VmRegistry's job
- Autoscaler does NOT track slices directly - that's ScalingGroup's job
- Scale-up decisions come from Autoscaler, scale-down is delegated to ScalingGroup
- ScalingGroup owns per-slice idle tracking and decides which slices to scale down

The run_once() flow is: cleanup -> evaluate -> execute -> idle scale-down
1. cleanup_failed_slices(): Clean up failed slices first (triggers backoff)
2. evaluate(): Compute scale-up decisions based on demand
3. execute(): Execute scale-up decisions
4. scale_down_if_idle(): Scale down idle slices based on per-slice idle tracking
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

from iris.cluster.types import DeviceType, VmWorkerStatusMap
from iris.cluster.vm.managed_vm import VmRegistry
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)

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
    """A demand entry specifying resource requirements and count."""

    device_type: DeviceType = DeviceType.CPU
    device_variant: str | None = None  # None = any variant of this type
    count: int = 0
    total_cpu: int = 0
    total_memory_bytes: int = 0
    preemptible: bool | None = None  # None = no preference


@dataclass
class RoutingResult:
    """Result of demand routing across groups."""

    allocations: dict[str, int]  # group_name -> allocated count
    unmet_demand: int


def route_demand(
    groups: list[ScalingGroup],
    demand_entries: list[DemandEntry],
    timestamp: Timestamp | None = None,
) -> RoutingResult:
    """Route demand to groups based on requirements and priority.

    For each demand entry:
    1. Find groups that match the device type/variant requirement
    2. Sort matching groups by priority (lower = higher priority)
    3. Route demand through available groups until satisfied or all exhausted

    Groups are skipped if not available (backoff, quota exceeded, at capacity).
    """
    ts = timestamp or Timestamp.now()
    allocations: dict[str, int] = {g.name: 0 for g in groups}
    total_unmet = 0

    for entry in demand_entries:
        matching = [g for g in groups if g.matches_device_requirement(entry.device_type, entry.device_variant)]
        if entry.preemptible is not None:
            matching = [g for g in matching if g.config.preemptible == entry.preemptible]
        # Sort by priority (lower = higher priority). Default to 100 if not set (proto3 defaults to 0).
        matching.sort(key=lambda g: g.config.priority or 100)

        remaining = entry.count
        for group in matching:
            if remaining <= 0:
                break

            # Check availability (handles backoff, quota, capacity)
            if not group.can_accept_demand(ts):
                continue

            # Compute headroom accounting for already-allocated demand
            counts = group.slice_state_counts()
            current = sum(counts.values())
            headroom = group.max_slices - current - allocations[group.name]

            if headroom <= 0:
                continue

            absorbed = min(remaining, headroom)
            allocations[group.name] += absorbed
            remaining -= absorbed

        if remaining > 0:
            logger.warning(
                "Demand overflow: %d/%d slices for device_type=%s, device_variant=%s unmet (tried %d groups)",
                remaining,
                entry.count,
                entry.device_type.value,
                entry.device_variant,
                len(matching),
            )
            total_unmet += remaining

    routed_groups = [(name, count) for name, count in allocations.items() if count > 0]
    if routed_groups:
        logger.debug("Demand routed: %s", ", ".join(f"{name}={count}" for name, count in routed_groups))

    return RoutingResult(allocations=allocations, unmet_demand=total_unmet)


class Autoscaler:
    """Manages scaling across scale groups.

    The autoscaler:
    - Receives demand from a DemandSource
    - Evaluates scaling decisions based on demand vs capacity
    - Executes decisions by calling ScalingGroup.scale_up/scale_down
    - Reports status via VmRegistry

    It does NOT:
    - Own VMs (VmRegistry does that)
    - Track VM groups (ScalingGroup does that)
    - Know about controller internals (DemandSource abstracts that)
    """

    def __init__(
        self,
        scale_groups: dict[str, ScalingGroup],
        vm_registry: VmRegistry,
        evaluation_interval: Duration,
        requesting_timeout: Duration,
        threads: ThreadContainer | None = None,
    ):
        """Create autoscaler with explicit parameters.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            vm_registry: Shared VM registry for tracking all VMs
            evaluation_interval: How often to evaluate scaling decisions
            requesting_timeout: How long to wait for VMs to provision before timing out
            threads: Optional thread container for testing
        """
        self._groups = scale_groups
        self._vm_registry = vm_registry
        self._evaluation_interval = evaluation_interval
        self._requesting_timeout = requesting_timeout

        # Track slice creation times for short-lived slice detection
        self._slice_created_at: dict[str, int] = {}

        # Bounded log of recent autoscaler actions for dashboard/debugging
        self._action_log: deque[vm_pb2.AutoscalerAction] = deque(maxlen=100)

        # Thread management
        self._threads = threads if threads is not None else get_thread_container()

    @classmethod
    def from_config(
        cls,
        scale_groups: dict[str, ScalingGroup],
        vm_registry: VmRegistry,
        config: config_pb2.AutoscalerDefaults,
        threads: ThreadContainer | None = None,
    ) -> Autoscaler:
        """Create autoscaler from proto config.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            vm_registry: Shared VM registry for tracking all VMs
            config: Autoscaler configuration proto (with defaults already applied)
            threads: Optional thread container for testing

        Returns:
            Configured Autoscaler instance
        """
        return cls(
            scale_groups=scale_groups,
            vm_registry=vm_registry,
            evaluation_interval=Duration.from_proto(config.evaluation_interval),
            requesting_timeout=Duration.from_proto(config.requesting_timeout),
            threads=threads,
        )

    def _wait_for_inflight(self) -> None:
        """Wait for in-flight scale-ups to complete without terminating anything.

        Test-only: Waits for all scale-up threads to complete.
        """
        # Wait for all threads in the container to finish
        self._threads.wait()

    def shutdown(self) -> None:
        """Shutdown the autoscaler, terminate all VM groups, and wait for in-flight scale-ups.

        Shutdown ordering:
        1. Wait for in-flight scale-up threads to complete
        2. Stop all VM bootstrap threads (call vm.stop() on each VM)
        3. Terminate VMs and cleanup (group.terminate_all())
        4. Stop VM managers (cleanup local platform threads if present)
        """
        # Step 1: Wait for in-flight scale-up threads
        self._threads.wait()

        # Step 2: Stop all VM bootstrap threads
        for group in self._groups.values():
            for vm_group in group.vm_groups():
                for vm in vm_group.vms():
                    vm.stop()

        # Step 3: Terminate VMs and cleanup
        for group in self._groups.values():
            group.terminate_all()

        # Step 4: Stop VM managers (cleanup local platform threads if present)
        for group in self._groups.values():
            group._vm_manager.stop()

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
            # Track creation times for discovered slices
            for slice_obj in group.vm_groups():
                self._slice_created_at[slice_obj.slice_id] = slice_obj.created_at_ms

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

        if result.unmet_demand > 0:
            logger.error(
                "CAPACITY INSUFFICIENT: %d slices of demand cannot be satisfied by any group",
                result.unmet_demand,
            )

        decisions = []
        for name, group in self._groups.items():
            allocated = result.allocations.get(name, 0)
            group.update_demand(allocated)
            decision = self._evaluate_group(group, allocated, ts)
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
        ready = counts["ready"]
        pending = counts["booting"] + counts["initializing"]
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
                logger.debug(
                    "Scale group %s: scale up blocked (backoff_until=%d, last_scale_up=%d)",
                    group.name,
                    group.backoff_until_ms,
                    group.last_scale_up_ms,
                )
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

        Marks the group as REQUESTING and spawns a background thread for the
        actual scale-up work. Returns immediately without blocking.
        """
        # Mark group as requesting before spawning thread
        group.mark_requesting(ts, self._requesting_timeout)

        # Spawn background thread for scale-up
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
        from iris.cluster.vm.managed_vm import QuotaExceededError

        # Log action as pending BEFORE execution
        action = self._log_action("scale_up", group.name, reason=reason, status="pending")

        try:
            logger.info("Scaling up %s: %s", group.name, reason)
            slice_obj = group.scale_up(timestamp=ts)
            self._slice_created_at[slice_obj.slice_id] = slice_obj.created_at_ms
            logger.info("Created slice %s for group %s", slice_obj.slice_id, group.name)
            # Update action with result
            action.slice_id = slice_obj.slice_id
            action.status = "completed"
            return True
        except QuotaExceededError as e:
            logger.warning("Quota exceeded for %s: %s", group.name, e)
            # Update the pending action to reflect quota failure
            action.action_type = "quota_exceeded"
            action.status = "failed"
            action.reason = str(e)
            return False
        except Exception as e:
            logger.error("Failed to create slice for %s: %s", group.name, e)
            action.status = "failed"
            action.reason = f"{reason} - error: {e}"
            group.record_failure(ts)
            return False
        finally:
            # Always clear requesting state when done
            group.clear_requesting()

    def run_once(
        self,
        demand_entries: list[DemandEntry],
        vm_status_map: VmWorkerStatusMap,
        timestamp: Timestamp | None = None,
    ) -> list[ScalingDecision]:
        """Run one evaluation cycle: cleanup -> evaluate -> execute -> idle scale-down.

        Args:
            demand_entries: List of demand entries with requirements and counts.
            vm_status_map: Map of VM address to worker status (required for scale-down).
            timestamp: Optional timestamp for testing.

        Returns the decisions that were made (for testing/logging).
        """
        timestamp = timestamp or Timestamp.now()
        logger.debug("Autoscaler run_once: demand_entries=%s", demand_entries)

        # Step 1: Clean up failed slices FIRST
        for group in self._groups.values():
            cleaned = group.cleanup_failed_slices(timestamp)
            for slice_obj in cleaned:
                self._slice_created_at.pop(slice_obj.slice_id, None)
                self._log_action(
                    "failed_cleanup",
                    group.name,
                    slice_id=slice_obj.slice_id,
                    reason="cleaning up failed slice",
                )

        # Step 2: Evaluate (scale-up only)
        decisions = self.evaluate(demand_entries, timestamp)
        if decisions:
            logger.info("Autoscaler decisions: %s", [(d.scale_group, d.action.value, d.reason) for d in decisions])

        # Step 3: Execute scale-up
        self.execute(decisions, timestamp)

        # Step 4: Idle scale-down
        for group in self._groups.values():
            target_capacity = max(group.current_demand, group.min_slices)
            scaled_down = group.scale_down_if_idle(vm_status_map, target_capacity, timestamp)
            if scaled_down:
                self._slice_created_at.pop(scaled_down.slice_id, None)
                self._log_action(
                    "scale_down",
                    group.name,
                    slice_id=scaled_down.slice_id,
                    reason=f"idle slice (target={target_capacity}, ready={group.ready_slice_count() + 1})",
                )

        return decisions

    # Status reporting via VmRegistry

    def get_vm(self, vm_id: str) -> vm_pb2.VmInfo | None:
        """Get VM info by ID."""
        vm = self._vm_registry.get_vm(vm_id)
        return vm.info if vm else None

    def get_init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get initialization log for a VM."""
        return self._vm_registry.get_init_log(vm_id, tail)

    def get_status(self) -> vm_pb2.AutoscalerStatus:
        """Build status for the status API."""
        from iris.rpc import time_pb2

        return vm_pb2.AutoscalerStatus(
            groups=[g.to_status() for g in self._groups.values()],
            current_demand={g.name: g.current_demand for g in self._groups.values()},
            last_evaluation=time_pb2.Timestamp(epoch_ms=0),  # Controlled by controller now
            recent_actions=list(self._action_log),
        )

    def get_group(self, name: str) -> ScalingGroup | None:
        """Get a scale group by name."""
        return self._groups.get(name)

    @property
    def vm_registry(self) -> VmRegistry:
        """Access the VM registry for RPC/status use."""
        return self._vm_registry

    @property
    def groups(self) -> dict[str, ScalingGroup]:
        """All scale groups."""
        return self._groups

    @property
    def evaluation_interval_seconds(self) -> float:
        """Configured evaluation interval in seconds."""
        return self._evaluation_interval.to_seconds()

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
            self._slice_created_at.pop(slice_id, None)
        except Exception as e:
            logger.warning("Failed to terminate slice %s: %s", slice_id, e)

    def _find_slice_for_worker(self, vm_address: str) -> tuple[str | None, ScalingGroup | None]:
        """Find the slice and group containing a worker by VM address."""
        for group in self._groups.values():
            for slice_obj in group.vm_groups():
                for vm in slice_obj.vms():
                    if vm.info.address == vm_address:
                        return slice_obj.slice_id, group
        return None, None

    def _record_slice_failure(self, slice_id: str, group: ScalingGroup) -> None:
        """Record slice failure and apply backoff if it was short-lived.

        Short-lived slices (died within SHORT_LIVED_SLICE_THRESHOLD of creation)
        indicate bad zone/quota issues or preemption. Apply backoff to prevent thrashing.
        """
        created_at = self._slice_created_at.get(slice_id)
        if created_at is None:
            return

        age_ms = Timestamp.now().epoch_ms() - created_at
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
