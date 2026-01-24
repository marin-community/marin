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

from iris.cluster.types import WorkerIdleMap
from iris.cluster.vm.managed_vm import VmRegistry
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.rpc import vm_pb2
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)

# Slices that die within this time of creation trigger backoff (preemption detection)
SHORT_LIVED_SLICE_THRESHOLD_MS = 5 * 60 * 1000  # 5 minutes


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

    accelerator_type: str | None = None  # None = any accelerator
    count: int = 0


@dataclass
class RoutingResult:
    """Result of demand routing across groups."""

    allocations: dict[str, int]  # group_name -> allocated count
    unmet_demand: int


@dataclass
class AutoscalerConfig:
    """Configuration for the autoscaler."""

    evaluation_interval_seconds: float = 10.0


def route_demand(
    groups: list[ScalingGroup],
    demand_entries: list[DemandEntry],
    timestamp_ms: int | None = None,
) -> RoutingResult:
    """Route demand to groups based on requirements and priority.

    For each demand entry:
    1. Find groups that match the accelerator_type requirement
    2. Sort matching groups by priority (lower = higher priority)
    3. Route demand through available groups until satisfied or all exhausted

    Groups are skipped if not available (backoff, quota exceeded, at capacity).
    """
    ts = timestamp_ms or now_ms()
    allocations: dict[str, int] = {g.name: 0 for g in groups}
    total_unmet = 0

    for entry in demand_entries:
        matching = [g for g in groups if g.matches_requirements(entry.accelerator_type)]
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
                "Demand overflow: %d/%d slices for accelerator_type=%s unmet (tried %d groups)",
                remaining,
                entry.count,
                entry.accelerator_type,
                len(matching),
            )
            total_unmet += remaining

    routed_groups = [(name, count) for name, count in allocations.items() if count > 0]
    if routed_groups:
        logger.info("Demand routed: %s", ", ".join(f"{name}={count}" for name, count in routed_groups))

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
        config: AutoscalerConfig | None = None,
    ):
        self._groups = scale_groups
        self._vm_registry = vm_registry
        self._config = config or AutoscalerConfig()
        self._last_evaluation_ms: int = 0

        # Track slice creation times for short-lived slice detection
        self._slice_created_at: dict[str, int] = {}

        # Bounded log of recent autoscaler actions for dashboard/debugging
        self._action_log: deque[vm_pb2.AutoscalerAction] = deque(maxlen=100)

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
            timestamp_ms=now_ms(),
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
        timestamp_ms: int | None = None,
    ) -> list[ScalingDecision]:
        """Compute scaling decisions based on demand.

        Routes demand to groups based on accelerator_type requirements and
        priority. Higher-priority groups (lower priority number) receive
        demand first; overflow routes to lower-priority groups.

        Args:
            demand_entries: List of demand entries with requirements and counts.
            timestamp_ms: Optional timestamp for testing. If None, uses now_ms().

        Returns:
            List of scaling decisions to execute.
        """
        ts = timestamp_ms or now_ms()

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
        ts: int,
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
        timestamp_ms: int,
    ) -> None:
        """Execute scale-up decisions.

        Args:
            decisions: List of scaling decisions to execute.
            timestamp_ms: Current timestamp.
        """
        self._last_evaluation_ms = timestamp_ms

        for decision in decisions:
            group = self._groups.get(decision.scale_group)
            if not group:
                logger.warning("Unknown scale group in decision: %s", decision.scale_group)
                continue

            if decision.action == ScalingAction.SCALE_UP:
                self._execute_scale_up(group, timestamp_ms, reason=decision.reason)

    def _execute_scale_up(self, group: ScalingGroup, ts: int, reason: str = "") -> bool:
        """Create a new slice for a scale group.

        Returns:
            True if scale-up succeeded, False otherwise.
        """
        from iris.cluster.vm.managed_vm import QuotaExceededError

        # Log action as pending BEFORE execution
        action = self._log_action("scale_up", group.name, reason=reason, status="pending")

        logger.info("Scaling up %s: %s", group.name, reason)
        try:
            slice_obj = group.scale_up(ts=ts)
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

    def run_once(
        self,
        demand_entries: list[DemandEntry],
        worker_idle_map: WorkerIdleMap,
        timestamp_ms: int | None = None,
    ) -> list[ScalingDecision]:
        """Run one evaluation cycle: cleanup -> evaluate -> execute -> idle scale-down.

        Args:
            demand_entries: List of demand entries with requirements and counts.
            worker_idle_map: Map of worker_id to idle info (required for scale-down).
            timestamp_ms: Optional timestamp for testing.

        Returns the decisions that were made (for testing/logging).
        """
        timestamp_ms = timestamp_ms or now_ms()
        logger.debug("Autoscaler run_once: demand_entries=%s", demand_entries)

        # Step 1: Clean up failed slices FIRST
        for group in self._groups.values():
            cleaned = group.cleanup_failed_slices(timestamp_ms)
            for slice_obj in cleaned:
                self._slice_created_at.pop(slice_obj.slice_id, None)
                self._log_action(
                    "failed_cleanup",
                    group.name,
                    slice_id=slice_obj.slice_id,
                    reason="cleaning up failed slice",
                )

        # Step 2: Evaluate (scale-up only)
        decisions = self.evaluate(demand_entries, timestamp_ms)
        if decisions:
            logger.info("Autoscaler decisions: %s", [(d.scale_group, d.action.value, d.reason) for d in decisions])

        # Step 3: Execute scale-up
        self.execute(decisions, timestamp_ms)

        # Step 4: Idle scale-down
        for group in self._groups.values():
            target_capacity = max(group.current_demand, group.min_slices)
            scaled_down = group.scale_down_if_idle(worker_idle_map, target_capacity, timestamp_ms)
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
        return vm_pb2.AutoscalerStatus(
            groups=[g.to_status() for g in self._groups.values()],
            current_demand={g.name: g.current_demand for g in self._groups.values()},
            last_evaluation_ms=self._last_evaluation_ms,
            recent_actions=list(self._action_log),
        )

    def get_group(self, name: str) -> ScalingGroup | None:
        """Get a scale group by name."""
        return self._groups.get(name)

    @property
    def groups(self) -> dict[str, ScalingGroup]:
        """All scale groups."""
        return self._groups

    def notify_worker_failed(self, worker_id: str) -> None:
        """Called by controller when a worker fails. Terminates the containing slice.

        This integrates with the existing controller failure cascade:
        1. Controller detects worker timeout/failure
        2. Controller emits WorkerFailedEvent (cascades to tasks)
        3. Controller calls this method
        4. Autoscaler terminates the slice containing the failed worker

        If the slice was short-lived (died soon after creation), applies backoff
        to the scale group to prevent thrashing on bad zones/preemption.
        """
        slice_id, group = self._find_slice_for_worker(worker_id)
        if not slice_id or not group:
            logger.debug("Worker %s not found in any managed slice", worker_id)
            return

        logger.info("Worker %s failed, terminating slice %s", worker_id, slice_id)
        self._log_action(
            "worker_failed",
            group.name,
            slice_id=slice_id,
            reason=f"worker {worker_id} failed",
        )

        # Check if this was a short-lived slice (preemption detection)
        self._record_slice_failure(slice_id, group)

        try:
            group.scale_down(slice_id)
            self._slice_created_at.pop(slice_id, None)
        except Exception as e:
            logger.warning("Failed to terminate slice %s: %s", slice_id, e)

    def _find_slice_for_worker(self, worker_id: str) -> tuple[str | None, ScalingGroup | None]:
        """Find the slice and group containing a worker."""
        for group in self._groups.values():
            for slice_obj in group.vm_groups():
                for vm in slice_obj.vms():
                    if vm.info.worker_id == worker_id:
                        return slice_obj.slice_id, group
        return None, None

    def _record_slice_failure(self, slice_id: str, group: ScalingGroup) -> None:
        """Record slice failure and apply backoff if it was short-lived.

        Short-lived slices (died within SHORT_LIVED_SLICE_THRESHOLD_MS of creation)
        indicate bad zone/quota issues or preemption. Apply backoff to prevent thrashing.
        """
        created_at = self._slice_created_at.get(slice_id)
        if created_at is None:
            return

        age_ms = now_ms() - created_at
        if age_ms < SHORT_LIVED_SLICE_THRESHOLD_MS:
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
