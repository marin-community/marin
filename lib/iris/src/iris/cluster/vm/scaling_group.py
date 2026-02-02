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
import threading
from dataclasses import dataclass
from enum import Enum

from iris.cluster.types import DeviceType, VmWorkerStatusMap
from iris.cluster.vm.vm_platform import VmGroupProtocol, VmManagerProtocol
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


class GroupAvailability(Enum):
    """Availability state for a scale group in waterfall routing."""

    AVAILABLE = "available"
    BACKOFF = "backoff"
    AT_CAPACITY = "at_capacity"
    QUOTA_EXCEEDED = "quota_exceeded"
    REQUESTING = "requesting"


@dataclass(frozen=True)
class AvailabilityState:
    """Availability state with reason and optional expiry."""

    status: GroupAvailability
    reason: str = ""
    until: Timestamp | None = None


DEFAULT_SCALE_UP_COOLDOWN = Duration.from_minutes(1)
DEFAULT_SCALE_DOWN_COOLDOWN = Duration.from_minutes(5)
DEFAULT_BACKOFF_INITIAL = Duration.from_seconds(5.0)
DEFAULT_BACKOFF_MAX = Duration.from_minutes(5)
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_IDLE_THRESHOLD = Duration.from_minutes(5)
DEFAULT_QUOTA_TIMEOUT = Duration.from_minutes(5)


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
        config: config_pb2.ScaleGroupConfig,
        vm_manager: VmManagerProtocol,
        scale_up_cooldown: Duration = DEFAULT_SCALE_UP_COOLDOWN,
        scale_down_cooldown: Duration = DEFAULT_SCALE_DOWN_COOLDOWN,
        backoff_initial: Duration = DEFAULT_BACKOFF_INITIAL,
        backoff_max: Duration = DEFAULT_BACKOFF_MAX,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        idle_threshold: Duration = DEFAULT_IDLE_THRESHOLD,
        quota_timeout: Duration = DEFAULT_QUOTA_TIMEOUT,
    ):
        self._config = config
        self._vm_manager = vm_manager
        self._vm_groups: dict[str, VmGroupProtocol] = {}
        self._vm_groups_lock = threading.Lock()

        # Demand tracking (simple current/peak, no history)
        self._current_demand: int = 0
        self._peak_demand: int = 0

        # Per-slice idle tracking: slice_id -> last active timestamp
        self._slice_last_active: dict[str, Timestamp] = {}
        self._idle_threshold = idle_threshold

        # Backoff state
        self._backoff_until: Timestamp = Timestamp.from_ms(0)
        self._consecutive_failures: int = 0
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._backoff_factor = backoff_factor

        # Rate limiting
        self._last_scale_up: Timestamp = Timestamp.from_ms(0)
        self._last_scale_down: Timestamp = Timestamp.from_ms(0)
        self._scale_up_cooldown = scale_up_cooldown
        self._scale_down_cooldown = scale_down_cooldown

        # Quota state (set by scale_up when QuotaExceededError is raised)
        self._quota_exceeded_until: Timestamp = Timestamp.from_ms(0)
        self._quota_reason: str = ""
        self._quota_timeout = quota_timeout

        # Requesting state (set during async scale-up)
        self._requesting_until: Timestamp = Timestamp.from_ms(0)

    @property
    def config(self) -> config_pb2.ScaleGroupConfig:
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
        return self._backoff_until.epoch_ms()

    @property
    def last_scale_up_ms(self) -> int:
        """Timestamp of last scale-up operation."""
        return self._last_scale_up.epoch_ms()

    @property
    def last_scale_down_ms(self) -> int:
        """Timestamp of last scale-down operation."""
        return self._last_scale_down.epoch_ms()

    def mark_requesting(self, timestamp: Timestamp, timeout: Duration) -> None:
        """Mark this group as REQUESTING (scale-up in progress).

        Args:
            timestamp: Current timestamp
            timeout: How long to stay in REQUESTING state
        """
        self._requesting_until = timestamp.add(timeout)

    def clear_requesting(self) -> None:
        """Clear REQUESTING state (scale-up completed or failed)."""
        self._requesting_until = Timestamp.from_ms(0)

    def reconcile(self) -> None:
        """Discover and adopt existing VM groups from the cloud.

        Called once at startup to recover state from a previous controller.
        """
        with self._vm_groups_lock:
            for vm_group in self._vm_manager.discover_vm_groups():
                self._vm_groups[vm_group.group_id] = vm_group

    def scale_up(self, tags: dict[str, str] | None = None, timestamp: Timestamp | None = None) -> VmGroupProtocol:
        """Create a new VM group.

        Args:
            tags: Optional labels/tags for the VM group
            timestamp: Optional timestamp (for testing)

        Returns:
            The newly created VM group

        Raises:
            QuotaExceededError: When quota is exceeded (quota state is set internally)
        """
        from iris.chaos import chaos_raise
        from iris.cluster.vm.managed_vm import QuotaExceededError

        timestamp = timestamp or Timestamp.now()

        try:
            # Chaos injection for VM creation failures
            chaos_raise("vm.create")
            vm_group = self._vm_manager.create_vm_group(tags)
            with self._vm_groups_lock:
                self._vm_groups[vm_group.group_id] = vm_group
            self._last_scale_up = timestamp
            self._consecutive_failures = 0
            self._backoff_until = Timestamp.from_ms(0)
            self._quota_exceeded_until = Timestamp.from_ms(0)
            self._quota_reason = ""
            return vm_group
        except QuotaExceededError as e:
            self._quota_exceeded_until = timestamp.add(self._quota_timeout)
            self._quota_reason = str(e)
            raise

    def scale_down(self, group_id: str, timestamp: Timestamp | None = None) -> None:
        """Terminate a VM group.

        Args:
            group_id: ID of the VM group to terminate
            timestamp: Optional timestamp (for testing)
        """
        timestamp = timestamp or Timestamp.now()
        with self._vm_groups_lock:
            vm_group = self._vm_groups.pop(group_id, None)
        if vm_group:
            vm_group.terminate()
            self._last_scale_down = timestamp
            self._slice_last_active.pop(group_id, None)  # Clean up tracking

    def cleanup_failed_slices(self, timestamp: Timestamp | None = None) -> list[VmGroupProtocol]:
        """Find and terminate all failed slices, triggering backoff once.

        Does NOT respect scale_down_cooldown - failed slices are always cleaned immediately.
        """
        timestamp = timestamp or Timestamp.now()
        cleaned: list[VmGroupProtocol] = []

        with self._vm_groups_lock:
            snapshot = dict(self._vm_groups)
        failed_slice_ids = [slice_id for slice_id, slice_obj in snapshot.items() if slice_obj.status().any_failed]

        for slice_id in failed_slice_ids:
            with self._vm_groups_lock:
                slice_obj = self._vm_groups.pop(slice_id, None)
            if slice_obj:
                logger.info("Cleaning up failed slice %s in group %s", slice_id, self.name)
                slice_obj.terminate()
                self._slice_last_active.pop(slice_id, None)
                cleaned.append(slice_obj)

        if cleaned:
            logger.info("Failed slice cleanup triggers backoff for group %s (%d slices)", self.name, len(cleaned))
            self.record_failure(timestamp)

        return cleaned

    def vm_groups(self) -> list[VmGroupProtocol]:
        """All VM groups in this scale group."""
        with self._vm_groups_lock:
            return list(self._vm_groups.values())

    def slice_count(self) -> int:
        """Total number of VM groups (regardless of state)."""
        with self._vm_groups_lock:
            return len(self._vm_groups)

    def ready_slice_count(self) -> int:
        """Count of VM groups where all VMs are ready."""
        with self._vm_groups_lock:
            snapshot = list(self._vm_groups.values())
        return sum(1 for g in snapshot if g.status().all_ready)

    def get_slice(self, group_id: str) -> VmGroupProtocol | None:
        """Get a specific VM group by ID."""
        with self._vm_groups_lock:
            return self._vm_groups.get(group_id)

    def update_demand(self, demand: int) -> None:
        """Update current demand."""
        self._current_demand = demand
        self._peak_demand = max(self._peak_demand, demand)

    def update_slice_activity(self, vm_status_map: VmWorkerStatusMap, timestamp: Timestamp) -> None:
        """Update activity timestamps for all slices based on worker status.

        For each slice, if any worker has running tasks, update its last_active timestamp.
        """
        with self._vm_groups_lock:
            snapshot = dict(self._vm_groups)
        for slice_id, slice_obj in snapshot.items():
            if self._slice_has_active_workers(slice_obj, vm_status_map):
                self._slice_last_active[slice_id] = timestamp

    def _slice_has_active_workers(self, slice_obj: VmGroupProtocol, vm_status_map: VmWorkerStatusMap) -> bool:
        """Check if any worker in a slice has running tasks (lookup by VM address)."""
        for vm in slice_obj.vms():
            vm_address = vm.info.address
            if not vm_address:
                continue
            status = vm_status_map.get(vm_address)
            if status is not None and not status.is_idle:
                return True
        return False

    def is_slice_eligible_for_scaledown(self, slice_id: str, timestamp: Timestamp) -> bool:
        """Check if a specific slice has been idle long enough to scale down.

        Eligible if:
        - Slice not tracked (never had activity) -> eligible
        - OR idle for at least idle_threshold
        """
        last_active = self._slice_last_active.get(slice_id)
        if last_active is None:
            return True  # Never had activity, eligible for scaledown
        idle_duration = Duration.from_ms(timestamp.epoch_ms() - last_active.epoch_ms())
        return idle_duration >= self._idle_threshold

    def get_idle_slices(self, timestamp: Timestamp) -> list[VmGroupProtocol]:
        """Get all slices that are eligible for scaledown, sorted by idle time (longest first)."""
        with self._vm_groups_lock:
            snapshot = dict(self._vm_groups)
        eligible = []
        for slice_id, slice_obj in snapshot.items():
            if slice_obj.status().all_ready and self.is_slice_eligible_for_scaledown(slice_id, timestamp):
                last_active = self._slice_last_active.get(slice_id, Timestamp.from_ms(0))
                eligible.append((slice_obj, last_active.epoch_ms()))
        # Sort by last_active ascending (oldest activity first = longest idle)
        eligible.sort(key=lambda x: x[1])
        return [s[0] for s in eligible]

    def scale_down_if_idle(
        self,
        vm_status_map: VmWorkerStatusMap,
        target_capacity: int,
        timestamp: Timestamp,
    ) -> VmGroupProtocol | None:
        """Scale down one idle slice if we're over target capacity.

        This method handles the complete scale-down decision and execution:
        1. Update slice activity based on worker idle status
        2. Check if we're over target capacity (using ready + pending)
        3. Find an eligible idle slice and terminate it

        Args:
            vm_status_map: Map of VM address to worker status
            target_capacity: Target number of slices (typically max(demand, min_slices))
            timestamp: Current timestamp for idle calculation

        Returns:
            The terminated slice, or None if no scale-down occurred
        """
        # Update activity tracking
        self.update_slice_activity(vm_status_map, timestamp)

        # Use ready + pending for capacity check to prevent churn during boot
        counts = self.slice_state_counts()
        ready = counts["ready"]
        pending = counts["booting"] + counts["initializing"]

        # Don't scale down if total capacity (ready + pending) is at or below target
        if ready + pending <= target_capacity:
            return None

        # Don't scale down ready slices if we're still waiting for pending
        if ready <= target_capacity:
            return None

        if not self.can_scale_down(timestamp):
            logger.debug("Scale group %s: scale down blocked by cooldown", self.name)
            return None

        # Find idle slices and verify they're still idle before termination
        idle_slices = self.get_idle_slices(timestamp)
        for slice_obj in idle_slices:
            if self._verify_slice_idle(slice_obj, vm_status_map):
                last_active = self._slice_last_active.get(slice_obj.slice_id, Timestamp.from_ms(0))
                idle_duration = Duration.from_ms(timestamp.epoch_ms() - last_active.epoch_ms())
                logger.info(
                    "Scale group %s: scaling down slice %s (idle for %dms, ready=%d, pending=%d, target=%d)",
                    self.name,
                    slice_obj.slice_id,
                    idle_duration.to_ms(),
                    ready,
                    pending,
                    target_capacity,
                )
                self.scale_down(slice_obj.slice_id, timestamp)
                return slice_obj

        return None

    def _verify_slice_idle(self, slice_obj: VmGroupProtocol, vm_status_map: VmWorkerStatusMap) -> bool:
        """Verify all workers in a slice are still idle before termination (lookup by VM address)."""
        for vm in slice_obj.vms():
            vm_address = vm.info.address
            if not vm_address:
                return False  # VM hasn't registered
            status = vm_status_map.get(vm_address)
            if status is None or not status.is_idle:
                return False
        return True

    def can_scale_up(self, timestamp: Timestamp | None = None) -> bool:
        """Check if scale-up is allowed.

        Scale-up is blocked if:
        - Currently in backoff due to previous failures
        - Scale-up cooldown period has not elapsed
        - Already at max_slices
        - Scale-up request is in progress (REQUESTING state)
        """
        timestamp = timestamp or Timestamp.now()
        if timestamp.before(self._backoff_until):
            return False
        if timestamp.before(self._requesting_until):
            return False
        cooldown_end = self._last_scale_up.add(self._scale_up_cooldown)
        if self._last_scale_up.epoch_ms() > 0 and timestamp.before(cooldown_end):
            return False
        with self._vm_groups_lock:
            vm_group_count = len(self._vm_groups)
        if vm_group_count >= self._config.max_slices:
            return False
        return True

    def can_scale_down(self, timestamp: Timestamp | None = None) -> bool:
        """Check if scale-down is allowed.

        Scale-down is blocked if:
        - Scale-down cooldown period has not elapsed
        - Already at min_slices
        """
        timestamp = timestamp or Timestamp.now()
        cooldown_end = self._last_scale_down.add(self._scale_down_cooldown)
        if self._last_scale_down.epoch_ms() > 0 and timestamp.before(cooldown_end):
            return False
        with self._vm_groups_lock:
            vm_group_count = len(self._vm_groups)
        if vm_group_count <= self._config.min_slices:
            return False
        return True

    def record_failure(self, timestamp: Timestamp | None = None) -> None:
        """Record a scale-up failure and apply exponential backoff.

        Each consecutive failure doubles the backoff time, up to a maximum.
        """
        timestamp = timestamp or Timestamp.now()
        self._consecutive_failures += 1

        backoff_duration = self._backoff_initial * (self._backoff_factor ** (self._consecutive_failures - 1))
        backoff_duration = min(backoff_duration, self._backoff_max)
        self._backoff_until = timestamp.add(backoff_duration)

    def reset_backoff(self) -> None:
        """Reset backoff state (typically after successful operation)."""
        self._consecutive_failures = 0
        self._backoff_until = Timestamp.from_ms(0)

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
        with self._vm_groups_lock:
            snapshot = list(self._vm_groups.values())
        for g in snapshot:
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

    def matches_device_requirement(self, device_type: DeviceType, device_variant: str | None) -> bool:
        """Check if this group can satisfy the given device requirements.

        Matching rules:
        - CPU demand: matches ANY group (all VMs have CPUs)
        - GPU/TPU with variant=None: matches any group of the same device type
        - GPU/TPU with specific variant: requires exact variant match
        """
        if device_type == DeviceType.CPU:
            return True  # CPU jobs can run on ANY group

        # Check device type matches
        group_type = self._get_device_type()
        if group_type != device_type:
            return False

        # None variant = any group of this type; specific variant = exact match
        if device_variant is None:
            return True
        return self._config.accelerator_variant == device_variant

    def _get_device_type(self) -> DeviceType:
        """Get device type from config."""
        accel = self._config.accelerator_type
        if accel == config_pb2.ACCELERATOR_TYPE_GPU:
            return DeviceType.GPU
        elif accel == config_pb2.ACCELERATOR_TYPE_TPU:
            return DeviceType.TPU
        return DeviceType.CPU

    def availability(self, timestamp: Timestamp | None = None) -> AvailabilityState:
        """Compute current availability state for waterfall routing.

        All states are computed from timestamps—no external state setting.
        Priority: QUOTA_EXCEEDED > BACKOFF > REQUESTING > AT_CAPACITY > AVAILABLE
        """
        timestamp = timestamp or Timestamp.now()

        # Quota exceeded
        if self._quota_exceeded_until.epoch_ms() > 0 and timestamp.before(self._quota_exceeded_until):
            return AvailabilityState(
                GroupAvailability.QUOTA_EXCEEDED,
                self._quota_reason,
                self._quota_exceeded_until,
            )

        # Backoff from failures
        if timestamp.before(self._backoff_until):
            return AvailabilityState(
                GroupAvailability.BACKOFF,
                f"backoff until {self._backoff_until.epoch_ms()}",
                self._backoff_until,
            )

        # Requesting (scale-up in progress)
        if timestamp.before(self._requesting_until):
            return AvailabilityState(
                GroupAvailability.REQUESTING,
                "scale-up in progress",
                self._requesting_until,
            )

        # At capacity
        with self._vm_groups_lock:
            vm_group_count = len(self._vm_groups)
        if vm_group_count >= self._config.max_slices:
            return AvailabilityState(GroupAvailability.AT_CAPACITY)

        return AvailabilityState(GroupAvailability.AVAILABLE)

    def can_accept_demand(self, timestamp: Timestamp | None = None) -> bool:
        """Whether this group can accept demand for waterfall routing."""
        return self.availability(timestamp).status == GroupAvailability.AVAILABLE

    def terminate_all(self) -> None:
        """Terminate all VM groups in this scale group."""
        with self._vm_groups_lock:
            snapshot = list(self._vm_groups.values())
            self._vm_groups.clear()
        for vm_group in snapshot:
            vm_group.terminate()

    def to_status(self) -> vm_pb2.ScaleGroupStatus:
        """Build a ScaleGroupStatus proto for the status API."""
        with self._vm_groups_lock:
            snapshot = list(self._vm_groups.values())
        return vm_pb2.ScaleGroupStatus(
            name=self.name,
            config=self._config,
            current_demand=self._current_demand,
            peak_demand=self._peak_demand,
            backoff_until=self._backoff_until.to_proto(),
            consecutive_failures=self._consecutive_failures,
            last_scale_up=self._last_scale_up.to_proto(),
            last_scale_down=self._last_scale_down.to_proto(),
            slices=[g.to_proto() for g in snapshot],
        )
