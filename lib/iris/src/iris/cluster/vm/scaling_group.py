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

"""ScalingGroup owns VM groups and manages scaling state for a single group.

This merges the old ScaleGroupState (stats tracking) with VM group ownership.
Each ScalingGroup has its own VmManager instance and tracks its own VM groups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from iris.cluster.types import WorkerIdleMap
from iris.cluster.vm.vm_platform import VmGroupProtocol, VmManagerProtocol
from iris.rpc import vm_pb2
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)


class GroupAvailability(Enum):
    """Availability state for a scale group in waterfall routing."""

    AVAILABLE = "available"
    BACKOFF = "backoff"
    AT_CAPACITY = "at_capacity"
    QUOTA_EXCEEDED = "quota_exceeded"


@dataclass(frozen=True)
class AvailabilityState:
    """Availability state with reason and optional expiry."""

    status: GroupAvailability
    reason: str = ""
    until_ms: int | None = None


DEFAULT_SCALE_UP_COOLDOWN_MS = 60_000  # 1 minute
DEFAULT_SCALE_DOWN_COOLDOWN_MS = 300_000  # 5 minutes
DEFAULT_BACKOFF_INITIAL_MS = 5_000  # 5 seconds
DEFAULT_BACKOFF_MAX_MS = 300_000  # 5 minutes
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_IDLE_THRESHOLD_MS = 300_000  # 5 minutes
DEFAULT_QUOTA_TIMEOUT_MS = 300_000  # 5 minutes


class ScalingGroup:
    """Owns VM groups for a single scale group.

    Merges the old ScaleGroupState (stats tracking) with VM group ownership.
    Each ScalingGroup:
    - Has its own VmManager for creating/discovering VM groups
    - Tracks VM groups it owns
    - Maintains scaling stats (per-slice idle tracking, backoff, cooldowns)
    - Provides scaling policy helpers (can_scale_up, can_scale_down)
    - Owns scale-down logic via per-slice idle timeout tracking
    """

    def __init__(
        self,
        config: vm_pb2.ScaleGroupConfig,
        vm_manager: VmManagerProtocol,
        scale_up_cooldown_ms: int = DEFAULT_SCALE_UP_COOLDOWN_MS,
        scale_down_cooldown_ms: int = DEFAULT_SCALE_DOWN_COOLDOWN_MS,
        backoff_initial_ms: int = DEFAULT_BACKOFF_INITIAL_MS,
        backoff_max_ms: int = DEFAULT_BACKOFF_MAX_MS,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        idle_threshold_ms: int = DEFAULT_IDLE_THRESHOLD_MS,
        quota_timeout_ms: int = DEFAULT_QUOTA_TIMEOUT_MS,
    ):
        self._config = config
        self._vm_manager = vm_manager
        self._vm_groups: dict[str, VmGroupProtocol] = {}

        # Demand tracking (simple current/peak, no history)
        self._current_demand: int = 0
        self._peak_demand: int = 0

        # Per-slice idle tracking: slice_id -> last active timestamp (ms)
        self._slice_last_active: dict[str, int] = {}
        self._idle_threshold_ms = idle_threshold_ms

        # Backoff state
        self._backoff_until_ms: int = 0
        self._consecutive_failures: int = 0
        self._backoff_initial_ms = backoff_initial_ms
        self._backoff_max_ms = backoff_max_ms
        self._backoff_factor = backoff_factor

        # Rate limiting
        self._last_scale_up_ms: int = 0
        self._last_scale_down_ms: int = 0
        self._scale_up_cooldown_ms = scale_up_cooldown_ms
        self._scale_down_cooldown_ms = scale_down_cooldown_ms

        # Quota state (set by scale_up when QuotaExceededError is raised)
        self._quota_exceeded_until_ms: int = 0
        self._quota_reason: str = ""
        self._quota_timeout_ms = quota_timeout_ms

    @property
    def config(self) -> vm_pb2.ScaleGroupConfig:
        """Configuration for this scale group."""
        return self._config

    @property
    def name(self) -> str:
        """Name of this scale group."""
        return self._config.name

    @property
    def min_slices(self) -> int:
        """Minimum number of VM groups to maintain."""
        return self._config.min_slices

    @property
    def max_slices(self) -> int:
        """Maximum number of VM groups allowed."""
        return self._config.max_slices

    @property
    def current_demand(self) -> int:
        """Current demand level."""
        return self._current_demand

    @property
    def peak_demand(self) -> int:
        """Peak demand seen."""
        return self._peak_demand

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive scale-up failures."""
        return self._consecutive_failures

    @property
    def backoff_until_ms(self) -> int:
        """Timestamp until which scale-up is blocked due to backoff."""
        return self._backoff_until_ms

    @property
    def last_scale_up_ms(self) -> int:
        """Timestamp of last scale-up operation."""
        return self._last_scale_up_ms

    @property
    def last_scale_down_ms(self) -> int:
        """Timestamp of last scale-down operation."""
        return self._last_scale_down_ms

    def reconcile(self) -> None:
        """Discover and adopt existing VM groups from the cloud.

        Called once at startup to recover state from a previous controller.
        """
        for vm_group in self._vm_manager.discover_vm_groups():
            self._vm_groups[vm_group.group_id] = vm_group

    def scale_up(self, tags: dict[str, str] | None = None, ts: int | None = None) -> VmGroupProtocol:
        """Create a new VM group.

        Args:
            tags: Optional labels/tags for the VM group
            ts: Optional timestamp (for testing)

        Returns:
            The newly created VM group

        Raises:
            QuotaExceededError: When quota is exceeded (quota state is set internally)
        """
        from iris.cluster.vm.managed_vm import QuotaExceededError

        ts = ts or now_ms()
        try:
            vm_group = self._vm_manager.create_vm_group(tags)
            self._vm_groups[vm_group.group_id] = vm_group
            self._last_scale_up_ms = ts
            self._consecutive_failures = 0
            self._backoff_until_ms = 0
            self._quota_exceeded_until_ms = 0
            self._quota_reason = ""
            return vm_group
        except QuotaExceededError as e:
            self._quota_exceeded_until_ms = ts + self._quota_timeout_ms
            self._quota_reason = str(e)
            raise

    def scale_down(self, group_id: str, timestamp_ms: int | None = None) -> None:
        """Terminate a VM group.

        Args:
            group_id: ID of the VM group to terminate
            timestamp_ms: Optional timestamp (for testing)
        """
        timestamp_ms = timestamp_ms or now_ms()
        vm_group = self._vm_groups.pop(group_id, None)
        if vm_group:
            vm_group.terminate()
            self._last_scale_down_ms = timestamp_ms
            self._slice_last_active.pop(group_id, None)  # Clean up tracking

    def cleanup_failed_slices(self, timestamp_ms: int | None = None) -> list[VmGroupProtocol]:
        """Find and terminate all failed slices, triggering backoff once.

        Does NOT respect scale_down_cooldown - failed slices are always cleaned immediately.
        """
        timestamp_ms = timestamp_ms or now_ms()
        cleaned: list[VmGroupProtocol] = []

        failed_slice_ids = [slice_id for slice_id, slice_obj in self._vm_groups.items() if slice_obj.status().any_failed]

        for slice_id in failed_slice_ids:
            slice_obj = self._vm_groups.get(slice_id)
            if slice_obj:
                logger.info("Cleaning up failed slice %s in group %s", slice_id, self.name)
                slice_obj.terminate()
                self._vm_groups.pop(slice_id, None)
                self._slice_last_active.pop(slice_id, None)
                cleaned.append(slice_obj)

        if cleaned:
            logger.info("Failed slice cleanup triggers backoff for group %s (%d slices)", self.name, len(cleaned))
            self.record_failure(timestamp_ms)

        return cleaned

    def vm_groups(self) -> list[VmGroupProtocol]:
        """All VM groups in this scale group."""
        return list(self._vm_groups.values())

    def slice_count(self) -> int:
        """Total number of VM groups (regardless of state)."""
        return len(self._vm_groups)

    def ready_slice_count(self) -> int:
        """Count of VM groups where all VMs are ready."""
        return sum(1 for g in self._vm_groups.values() if g.status().all_ready)

    def get_slice(self, group_id: str) -> VmGroupProtocol | None:
        """Get a specific VM group by ID."""
        return self._vm_groups.get(group_id)

    def update_demand(self, demand: int) -> None:
        """Update current demand."""
        self._current_demand = demand
        self._peak_demand = max(self._peak_demand, demand)

    def update_slice_activity(self, worker_idle_map: WorkerIdleMap, timestamp_ms: int) -> None:
        """Update activity timestamps for all slices based on worker status.

        For each slice, if any worker has running tasks, update its last_active timestamp.
        """
        for slice_id, slice_obj in self._vm_groups.items():
            if self._slice_has_active_workers(slice_obj, worker_idle_map):
                self._slice_last_active[slice_id] = timestamp_ms

    def _slice_has_active_workers(self, slice_obj: VmGroupProtocol, worker_idle_map: WorkerIdleMap) -> bool:
        """Check if any worker in a slice has running tasks."""
        for vm in slice_obj.vms():
            worker_id = vm.info.worker_id
            if not worker_id:
                continue
            idle_info = worker_idle_map.get(worker_id)
            if idle_info is not None and not idle_info.is_idle:
                return True
        return False

    def is_slice_eligible_for_scaledown(self, slice_id: str, timestamp_ms: int) -> bool:
        """Check if a specific slice has been idle long enough to scale down.

        Eligible if:
        - Slice not tracked (never had activity) -> eligible
        - OR idle for at least idle_threshold_ms
        """
        last_active = self._slice_last_active.get(slice_id)
        if last_active is None:
            return True  # Never had activity, eligible for scaledown
        return (timestamp_ms - last_active) >= self._idle_threshold_ms

    def get_idle_slices(self, timestamp_ms: int) -> list[VmGroupProtocol]:
        """Get all slices that are eligible for scaledown, sorted by idle time (longest first)."""
        eligible = []
        for slice_id, slice_obj in self._vm_groups.items():
            if slice_obj.status().all_ready and self.is_slice_eligible_for_scaledown(slice_id, timestamp_ms):
                last_active = self._slice_last_active.get(slice_id, 0)
                eligible.append((slice_obj, last_active))
        # Sort by last_active ascending (oldest activity first = longest idle)
        eligible.sort(key=lambda x: x[1])
        return [s[0] for s in eligible]

    def scale_down_if_idle(
        self,
        worker_idle_map: WorkerIdleMap,
        target_capacity: int,
        timestamp_ms: int,
    ) -> VmGroupProtocol | None:
        """Scale down one idle slice if we're over target capacity.

        This method handles the complete scale-down decision and execution:
        1. Update slice activity based on worker idle status
        2. Check if we're over target capacity
        3. Find an eligible idle slice and terminate it

        Args:
            worker_idle_map: Map of worker_id to idle info
            target_capacity: Target number of ready slices (typically max(demand, min_slices))
            timestamp_ms: Current timestamp for idle calculation

        Returns:
            The terminated slice, or None if no scale-down occurred
        """
        # Update activity tracking
        self.update_slice_activity(worker_idle_map, timestamp_ms)

        ready = self.ready_slice_count()
        if ready <= target_capacity:
            return None

        if not self.can_scale_down(timestamp_ms):
            logger.debug("Scale group %s: scale down blocked by cooldown", self.name)
            return None

        # Find idle slices and verify they're still idle before termination
        idle_slices = self.get_idle_slices(timestamp_ms)
        for slice_obj in idle_slices:
            if self._verify_slice_idle(slice_obj, worker_idle_map):
                last_active = self._slice_last_active.get(slice_obj.slice_id, 0)
                idle_duration_ms = timestamp_ms - last_active if last_active else 0
                logger.info(
                    "Scale group %s: scaling down slice %s (idle for %dms, ready=%d > target=%d)",
                    self.name,
                    slice_obj.slice_id,
                    idle_duration_ms,
                    ready,
                    target_capacity,
                )
                self.scale_down(slice_obj.slice_id, timestamp_ms)
                return slice_obj

        return None

    def _verify_slice_idle(self, slice_obj: VmGroupProtocol, worker_idle_map: WorkerIdleMap) -> bool:
        """Verify all workers in a slice are still idle before termination."""
        for vm in slice_obj.vms():
            if not vm.info.worker_id:
                return False  # VM hasn't registered worker
            idle_info = worker_idle_map.get(vm.info.worker_id)
            if idle_info is None or not idle_info.is_idle:
                return False
        return True

    def can_scale_up(self, ts: int | None = None) -> bool:
        """Check if scale-up is allowed.

        Scale-up is blocked if:
        - Currently in backoff due to previous failures
        - Scale-up cooldown period has not elapsed
        - Already at max_slices
        """
        ts = ts or now_ms()
        if ts < self._backoff_until_ms:
            return False
        if self._last_scale_up_ms > 0 and ts < self._last_scale_up_ms + self._scale_up_cooldown_ms:
            return False
        if len(self._vm_groups) >= self._config.max_slices:
            return False
        return True

    def can_scale_down(self, ts: int | None = None) -> bool:
        """Check if scale-down is allowed.

        Scale-down is blocked if:
        - Scale-down cooldown period has not elapsed
        - Already at min_slices
        """
        ts = ts or now_ms()
        if self._last_scale_down_ms > 0 and ts < self._last_scale_down_ms + self._scale_down_cooldown_ms:
            return False
        if len(self._vm_groups) <= self._config.min_slices:
            return False
        return True

    def record_failure(self, ts: int | None = None) -> None:
        """Record a scale-up failure and apply exponential backoff.

        Each consecutive failure doubles the backoff time, up to a maximum.
        """
        ts = ts or now_ms()
        self._consecutive_failures += 1

        backoff_ms = self._backoff_initial_ms * (self._backoff_factor ** (self._consecutive_failures - 1))
        backoff_ms = min(backoff_ms, self._backoff_max_ms)
        self._backoff_until_ms = ts + int(backoff_ms)

    def reset_backoff(self) -> None:
        """Reset backoff state (typically after successful operation)."""
        self._consecutive_failures = 0
        self._backoff_until_ms = 0

    def slice_state_counts(self) -> dict[str, int]:
        """Count VM groups by their dominant lifecycle state.

        Each VM group is categorized as:
        - "failed": any VM in the group has failed
        - "ready": all VMs in the group are ready
        - "initializing": at least one VM is initializing (but none failed)
        - "booting": at least one VM is booting (but none failed or initializing)

        Returns dict with keys: "booting", "initializing", "ready", "failed"
        """
        counts = {"booting": 0, "initializing": 0, "ready": 0, "failed": 0}
        for g in self._vm_groups.values():
            status = g.status()
            vms = status.vms

            # Skip terminated VM groups
            if all(vm.state == vm_pb2.VM_STATE_TERMINATED for vm in vms):
                continue

            if status.any_failed:
                counts["failed"] += 1
            elif status.all_ready:
                counts["ready"] += 1
            elif any(vm.state == vm_pb2.VM_STATE_INITIALIZING for vm in vms):
                counts["initializing"] += 1
            elif any(vm.state == vm_pb2.VM_STATE_BOOTING for vm in vms):
                counts["booting"] += 1
        return counts

    def matches_requirements(self, accelerator_type: str | None) -> bool:
        """Check if this group can satisfy the given requirements.

        Args:
            accelerator_type: Required accelerator type, or None for any.
        """
        if accelerator_type is not None:
            return self._config.accelerator_type == accelerator_type
        return True

    def availability(self, timestamp_ms: int | None = None) -> AvailabilityState:
        """Compute current availability state for waterfall routing.

        All states are computed from timestamps—no external state setting.
        Priority: QUOTA_EXCEEDED > BACKOFF > AT_CAPACITY > AVAILABLE
        """
        ts = timestamp_ms or now_ms()

        # Quota exceeded
        if self._quota_exceeded_until_ms and ts < self._quota_exceeded_until_ms:
            return AvailabilityState(
                GroupAvailability.QUOTA_EXCEEDED,
                self._quota_reason,
                self._quota_exceeded_until_ms,
            )

        # Backoff from failures
        if ts < self._backoff_until_ms:
            return AvailabilityState(
                GroupAvailability.BACKOFF,
                f"backoff until {self._backoff_until_ms}",
                self._backoff_until_ms,
            )

        # At capacity
        if len(self._vm_groups) >= self._config.max_slices:
            return AvailabilityState(GroupAvailability.AT_CAPACITY)

        return AvailabilityState(GroupAvailability.AVAILABLE)

    def can_accept_demand(self, timestamp_ms: int | None = None) -> bool:
        """Whether this group can accept demand for waterfall routing."""
        return self.availability(timestamp_ms).status == GroupAvailability.AVAILABLE

    def to_status(self) -> vm_pb2.ScaleGroupStatus:
        """Build a ScaleGroupStatus proto for the status API."""
        return vm_pb2.ScaleGroupStatus(
            name=self.name,
            config=self._config,
            current_demand=self._current_demand,
            peak_demand=self._peak_demand,
            backoff_until_ms=self._backoff_until_ms,
            consecutive_failures=self._consecutive_failures,
            last_scale_up_ms=self._last_scale_up_ms,
            last_scale_down_ms=self._last_scale_down_ms,
            slices=[g.to_proto() for g in self._vm_groups.values()],
        )
