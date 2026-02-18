# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ScalingGroup owns slices and manages scaling state for a single group.

Each ScalingGroup uses a Platform to create/discover slices, storing SliceHandle
references directly for internal tracking. It maintains scaling stats
(per-slice idle tracking, backoff, cooldowns) and provides scaling policy helpers.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, StrEnum

from iris.cluster.platform.base import Platform, SliceHandle
from iris.cluster.types import DeviceType, VmWorkerStatusMap, get_gpu_count, get_tpu_count
from iris.rpc import cluster_pb2, config_pb2, time_pb2, vm_pb2
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)


class SliceLifecycleState(StrEnum):
    """Lifecycle state for a slice (VM group) in the autoscaler.

    These states represent the dominant state of a slice based on its constituent VMs.
    String values are lowercase names for use as dictionary keys and proto map keys.

    States:
    - REQUESTING: Scale-up operation in progress (tracked at ScalingGroup level)
    - BOOTING: At least one VM is booting (VM_STATE_BOOTING)
    - INITIALIZING: At least one VM is initializing (VM_STATE_INITIALIZING)
    - READY: All VMs are ready (VM_STATE_READY)
    - FAILED: At least one VM has failed (VM_STATE_FAILED or VM_STATE_PREEMPTED)

    Note: These are slice-level aggregate states, not direct VM states.
    """

    REQUESTING = "requesting"
    BOOTING = "booting"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"


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


@dataclass
class SliceState:
    """Per-slice state tracked by ScalingGroup.

    Consolidates the slice handle with its associated tracking state
    (idle timeout, lifecycle) into a single structure.
    lifecycle and vm_addresses are populated eagerly by the bootstrap thread.
    """

    handle: SliceHandle
    last_active: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    lifecycle: SliceLifecycleState = SliceLifecycleState.BOOTING
    vm_addresses: list[str] = field(default_factory=list)


def prepare_slice_config(
    template: config_pb2.SliceConfig,
    parent_config: config_pb2.ScaleGroupConfig,
    label_prefix: str,
) -> config_pb2.SliceConfig:
    """Build a SliceConfig for platform.create_slice() from a template.

    Copies the template and sets the name_prefix and managed/scale-group labels.
    The template must already have accelerator_type, accelerator_variant, and
    num_vms set directly.
    """
    config = config_pb2.SliceConfig()
    config.CopyFrom(template)
    config.name_prefix = f"{label_prefix}-{parent_config.name}"
    config.labels[f"{label_prefix}-managed"] = "true"
    config.labels[f"{label_prefix}-scale-group"] = parent_config.name

    return config


def _zones_from_config(config: config_pb2.ScaleGroupConfig) -> list[str]:
    """Extract zones from ScaleGroupConfig's slice_template.

    Raises ValueError for GCP configs with no zones, since reconcile and
    list_slices would silently do nothing.
    """
    if not config.HasField("slice_template") or not config.slice_template.HasField("gcp"):
        return []
    gcp = config.slice_template.gcp
    if gcp.zone:
        return [gcp.zone]
    raise ValueError(
        f"ScaleGroupConfig '{config.name}' has a GCP slice_template but no zone configured. "
        "Set 'zone' in the GCP slice template."
    )


def _lifecycle_to_vm_state(lifecycle: SliceLifecycleState) -> vm_pb2.VmState:
    """Map slice lifecycle state to a VM state for proto APIs."""
    return {
        SliceLifecycleState.REQUESTING: vm_pb2.VM_STATE_BOOTING,
        SliceLifecycleState.BOOTING: vm_pb2.VM_STATE_BOOTING,
        SliceLifecycleState.INITIALIZING: vm_pb2.VM_STATE_INITIALIZING,
        SliceLifecycleState.READY: vm_pb2.VM_STATE_READY,
        SliceLifecycleState.FAILED: vm_pb2.VM_STATE_FAILED,
    }[lifecycle]


def slice_state_to_proto(state: SliceState) -> vm_pb2.SliceInfo:
    """Convert a SliceState to a SliceInfo proto for RPC APIs."""
    created_at = state.handle.created_at
    vm_state = _lifecycle_to_vm_state(state.lifecycle)
    return vm_pb2.SliceInfo(
        slice_id=state.handle.slice_id,
        scale_group=state.handle.scale_group,
        created_at=time_pb2.Timestamp(epoch_ms=created_at.epoch_ms()),
        vms=[
            vm_pb2.VmInfo(
                vm_id=f"{state.handle.slice_id}-vm-{i}",
                state=vm_state,
                address=addr,
                created_at=time_pb2.Timestamp(epoch_ms=created_at.epoch_ms()),
                state_changed_at=time_pb2.Timestamp(epoch_ms=created_at.epoch_ms()),
            )
            for i, addr in enumerate(state.vm_addresses)
        ],
    )


class ScalingGroup:
    """Owns slices for a single scale group.

    Each ScalingGroup:
    - Uses a Platform to create/discover slices
    - Stores SliceHandle references directly for internal tracking
    - Maintains scaling stats (per-slice idle tracking, backoff, cooldowns)
    - Provides scaling policy helpers (can_scale_up, can_scale_down)
    - Owns scale-down logic via per-slice idle timeout tracking
    """

    def __init__(
        self,
        config: config_pb2.ScaleGroupConfig,
        platform: Platform,
        label_prefix: str = "iris",
        scale_up_cooldown: Duration = DEFAULT_SCALE_UP_COOLDOWN,
        scale_down_cooldown: Duration = DEFAULT_SCALE_DOWN_COOLDOWN,
        backoff_initial: Duration = DEFAULT_BACKOFF_INITIAL,
        backoff_max: Duration = DEFAULT_BACKOFF_MAX,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        idle_threshold: Duration = DEFAULT_IDLE_THRESHOLD,
        quota_timeout: Duration = DEFAULT_QUOTA_TIMEOUT,
    ):
        self._config = config
        self._platform = platform
        self._label_prefix = label_prefix
        self._slices: dict[str, SliceState] = {}
        self._pending_scale_ups: int = 0
        self._slices_lock = threading.Lock()

        # Demand tracking (simple current/peak, no history)
        self._current_demand: int = 0
        self._peak_demand: int = 0

        self._idle_threshold = idle_threshold

        # Backoff state
        self._backoff_until: Deadline | None = None
        self._consecutive_failures: int = 0
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._backoff_factor = backoff_factor

        # Rate limiting
        self._last_scale_up: Timestamp = Timestamp.from_ms(0)
        self._last_scale_down: Timestamp = Timestamp.from_ms(0)
        self._scale_up_cooldown = scale_up_cooldown
        self._scale_down_cooldown = scale_down_cooldown

        # Quota state (set by scale_up when QuotaExhaustedError is raised)
        self._quota_exceeded_until: Deadline | None = None
        self._quota_reason: str = ""
        self._quota_timeout = quota_timeout

    @property
    def config(self) -> config_pb2.ScaleGroupConfig:
        """Configuration for this scale group."""
        return self._config

    @property
    def name(self) -> str:
        """Name of this scale group."""
        return self._config.name

    @property
    def num_vms(self) -> int:
        """Number of tasks per slice (coscheduling group size)."""
        return self._config.num_vms or 1

    @property
    def resources(self) -> config_pb2.ScaleGroupResources | None:
        """Per-VM resource capacity for this scale group."""
        if self._config.HasField("resources"):
            return self._config.resources
        return None

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

    def begin_scale_up(self) -> None:
        """Mark that a scale-up is in progress.

        Increments the pending counter, which is included in slice_count()
        and slice_state_counts(REQUESTING) to prevent over-provisioning.
        """
        with self._slices_lock:
            self._pending_scale_ups += 1

    def complete_scale_up(self, handle: SliceHandle, timestamp: Timestamp | None = None) -> None:
        """Record a successful scale-up: add the slice and decrement the pending counter."""
        timestamp = timestamp or Timestamp.now()
        with self._slices_lock:
            self._pending_scale_ups = max(0, self._pending_scale_ups - 1)
            self._slices[handle.slice_id] = SliceState(handle=handle)
        self._last_scale_up = timestamp
        self._consecutive_failures = 0
        self._backoff_until = None
        self._quota_exceeded_until = None
        self._quota_reason = ""

    def cancel_scale_up(self) -> None:
        """Record a failed scale-up: decrement the pending counter."""
        with self._slices_lock:
            self._pending_scale_ups = max(0, self._pending_scale_ups - 1)

    def mark_slice_ready(self, slice_id: str, vm_addresses: list[str]) -> None:
        """Mark a slice as READY with its VM addresses. Called after successful bootstrap."""
        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is not None:
                state.lifecycle = SliceLifecycleState.READY
                state.vm_addresses = vm_addresses

    def mark_slice_failed(self, slice_id: str) -> None:
        """Mark a slice as FAILED. Called when bootstrap fails."""
        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is not None:
                state.lifecycle = SliceLifecycleState.FAILED

    def reconcile(self) -> None:
        """Discover and adopt existing slices from the cloud.

        Called once at startup to recover state from a previous controller.
        Uses platform.list_slices() with the managed label to find our slices.
        """
        zones = _zones_from_config(self._config)
        labels = {f"{self._label_prefix}-scale-group": self._config.name}
        slice_handles = self._platform.list_slices(zones, labels)
        with self._slices_lock:
            for handle in slice_handles:
                self._slices[handle.slice_id] = SliceState(handle=handle)

    def scale_up(
        self,
        tags: dict[str, str] | None = None,
        timestamp: Timestamp | None = None,
        cluster_config: config_pb2.IrisClusterConfig | None = None,
    ) -> SliceHandle:
        """Create a new slice via the platform.

        Does NOT add to _slices tracking. Use begin_scale_up/complete_scale_up
        for lifecycle tracking. QuotaExhaustedError propagates to the caller.

        Args:
            tags: Optional extra labels/tags for the slice (merged with managed labels)
            timestamp: Optional timestamp (for testing)
            cluster_config: Full cluster config passed to platform.create_slice() for bootstrap

        Returns:
            The newly created SliceHandle
        """
        from iris.chaos import chaos_raise

        chaos_raise("vm.create")
        slice_config = prepare_slice_config(
            self._config.slice_template,
            self._config,
            self._label_prefix,
        )
        if tags:
            for k, v in tags.items():
                slice_config.labels[k] = v

        return self._platform.create_slice(slice_config, cluster_config=cluster_config)

    def scale_down(self, slice_id: str, timestamp: Timestamp | None = None) -> None:
        """Terminate a slice.

        Args:
            slice_id: ID of the slice to terminate
            timestamp: Optional timestamp (for testing)
        """
        timestamp = timestamp or Timestamp.now()
        with self._slices_lock:
            state = self._slices.get(slice_id)
        if state:
            state.handle.terminate()
            with self._slices_lock:
                self._slices.pop(slice_id, None)
            self._last_scale_down = timestamp

    def slice_handles(self) -> list[SliceHandle]:
        """All slice handles in this scale group."""
        with self._slices_lock:
            return [s.handle for s in self._slices.values()]

    def slice_count(self) -> int:
        """Total number of slices including in-flight scale-ups."""
        with self._slices_lock:
            return len(self._slices) + self._pending_scale_ups

    def ready_slice_count(self) -> int:
        """Count of slices where all VMs are ready."""
        with self._slices_lock:
            return sum(1 for s in self._slices.values() if s.lifecycle == SliceLifecycleState.READY)

    def get_slice(self, slice_id: str) -> SliceHandle | None:
        """Get a specific slice handle by ID."""
        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is None:
                return None
            return state.handle

    def update_demand(self, demand: int) -> None:
        """Update current demand."""
        self._current_demand = demand
        self._peak_demand = max(self._peak_demand, demand)

    def can_fit_resources(self, resources: cluster_pb2.ResourceSpecProto) -> bool:
        """Check whether a demand entry's resources fit within one VM."""
        sg_resources = self.resources
        if sg_resources is None:
            return False

        if resources.cpu and resources.cpu > sg_resources.cpu:
            return False
        if resources.memory_bytes and resources.memory_bytes > sg_resources.memory_bytes:
            return False
        if resources.disk_bytes and resources.disk_bytes > sg_resources.disk_bytes:
            return False

        gpu_count = get_gpu_count(resources.device)
        if gpu_count > sg_resources.gpu_count:
            return False

        tpu_count = get_tpu_count(resources.device)
        if tpu_count > sg_resources.tpu_count:
            return False

        return True

    def update_slice_activity(self, vm_status_map: VmWorkerStatusMap, timestamp: Timestamp) -> None:
        """Update activity timestamps for all slices based on worker status.

        For each slice, if any worker has running tasks, update its last_active timestamp.
        """
        with self._slices_lock:
            snapshot = list(self._slices.items())
        for _slice_id, state in snapshot:
            if self._slice_has_active_workers(state, vm_status_map):
                state.last_active = timestamp

    def _slice_has_active_workers(self, state: SliceState, vm_status_map: VmWorkerStatusMap) -> bool:
        """Check if any worker in a slice has running tasks (lookup by VM address)."""
        for vm_address in self._get_slice_vm_addresses(state):
            status = vm_status_map.get(vm_address)
            if status is not None and not status.is_idle:
                return True
        return False

    def is_slice_eligible_for_scaledown(self, slice_id: str, timestamp: Timestamp) -> bool:
        """Check if a specific slice has been idle long enough to scale down.

        Eligible if:
        - Slice not tracked -> eligible
        - last_active is epoch 0 (never had activity) -> eligible
        - OR idle for at least idle_threshold
        """
        with self._slices_lock:
            state = self._slices.get(slice_id)
        if state is None:
            return True
        if state.last_active.epoch_ms() == 0:
            return True
        idle_duration = Duration.from_ms(timestamp.epoch_ms() - state.last_active.epoch_ms())
        return idle_duration >= self._idle_threshold

    def get_idle_slices(self, timestamp: Timestamp) -> list[SliceState]:
        """Get all slice states eligible for scaledown, sorted by idle time (longest first)."""
        with self._slices_lock:
            snapshot = list(self._slices.items())
        eligible = []
        for slice_id, state in snapshot:
            if state.lifecycle == SliceLifecycleState.READY and self.is_slice_eligible_for_scaledown(
                slice_id, timestamp
            ):
                eligible.append((state, state.last_active.epoch_ms()))
        eligible.sort(key=lambda x: x[1])
        return [s[0] for s in eligible]

    def scale_down_if_idle(
        self,
        vm_status_map: VmWorkerStatusMap,
        target_capacity: int,
        timestamp: Timestamp,
    ) -> SliceHandle | None:
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
            The terminated slice handle, or None if no scale-down occurred
        """
        # Update activity tracking
        self.update_slice_activity(vm_status_map, timestamp)

        # Use ready + pending for capacity check to prevent churn during boot
        counts = self.slice_state_counts()
        ready = counts[SliceLifecycleState.READY]
        pending = counts[SliceLifecycleState.BOOTING] + counts[SliceLifecycleState.INITIALIZING]

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
        for slice_state in idle_slices:
            if self._verify_slice_idle(slice_state, vm_status_map):
                with self._slices_lock:
                    state = self._slices.get(slice_state.handle.slice_id)
                last_active = state.last_active if state else Timestamp.from_ms(0)
                idle_duration = Duration.from_ms(timestamp.epoch_ms() - last_active.epoch_ms())
                logger.info(
                    "Scale group %s: scaling down slice %s (idle for %dms, ready=%d, pending=%d, target=%d)",
                    self.name,
                    slice_state.handle.slice_id,
                    idle_duration.to_ms(),
                    ready,
                    pending,
                    target_capacity,
                )
                self.scale_down(slice_state.handle.slice_id, timestamp)
                return slice_state.handle

        return None

    def _verify_slice_idle(self, state: SliceState, vm_status_map: VmWorkerStatusMap) -> bool:
        """Verify all workers in a slice are idle before termination (lookup by VM address).

        Requires at least one known worker to be idle. If no workers are known at all
        (none in vm_status_map), returns False — the slice may still be booting.
        Zombie slices where workers have disappeared are handled by worker heartbeat timeouts.
        """
        has_known_worker = False
        for vm_address in self._get_slice_vm_addresses(state):
            status = vm_status_map.get(vm_address)
            if status is None:
                continue
            has_known_worker = True
            if not status.is_idle:
                return False
        return has_known_worker

    def can_scale_up(self, timestamp: Timestamp | None = None) -> bool:
        """Check if scale-up is allowed.

        Scale-up is blocked if:
        - Currently in backoff due to previous failures
        - Scale-up cooldown period has not elapsed
        - Already at max_slices (includes in-flight scale-ups)
        """
        timestamp = timestamp or Timestamp.now()
        if self._quota_exceeded_until is not None and not self._quota_exceeded_until.expired(now=timestamp):
            return False
        if self._backoff_until is not None and not self._backoff_until.expired(now=timestamp):
            return False
        cooldown_end = self._last_scale_up.add(self._scale_up_cooldown)
        if self._last_scale_up.epoch_ms() > 0 and timestamp.before(cooldown_end):
            return False
        with self._slices_lock:
            count = len(self._slices) + self._pending_scale_ups
        if count >= self._config.max_slices:
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
        with self._slices_lock:
            count = len(self._slices)
        if count <= self._config.min_slices:
            return False
        return True

    def record_quota_exceeded(self, reason: str, timestamp: Timestamp | None = None) -> None:
        """Record a quota exhaustion event, blocking scale-up until the quota timeout elapses."""
        timestamp = timestamp or Timestamp.now()
        self._quota_exceeded_until = Deadline.after(timestamp, self._quota_timeout)
        self._quota_reason = reason

    def record_failure(self, timestamp: Timestamp | None = None) -> None:
        """Record a scale-up failure and apply exponential backoff.

        Each consecutive failure doubles the backoff time, up to a maximum.
        """
        timestamp = timestamp or Timestamp.now()
        self._consecutive_failures += 1

        backoff_duration = self._backoff_initial * (self._backoff_factor ** (self._consecutive_failures - 1))
        backoff_duration = min(backoff_duration, self._backoff_max)
        self._backoff_until = Deadline.after(timestamp, backoff_duration)

    def reset_backoff(self) -> None:
        """Reset backoff state (typically after successful operation)."""
        self._consecutive_failures = 0
        self._backoff_until = None

    def slice_state_counts(self) -> dict[SliceLifecycleState, int]:
        """Count slices by their lifecycle state.

        Returns dict with SliceLifecycleState enum keys.
        """
        counts = {state: 0 for state in SliceLifecycleState}
        with self._slices_lock:
            counts[SliceLifecycleState.REQUESTING] = self._pending_scale_ups
            for state in self._slices.values():
                counts[state.lifecycle] += 1
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
        if self._quota_exceeded_until is not None and not self._quota_exceeded_until.expired(now=timestamp):
            return AvailabilityState(
                GroupAvailability.QUOTA_EXCEEDED,
                self._quota_reason,
                self._quota_exceeded_until.as_timestamp(),
            )

        # Backoff from failures
        if self._backoff_until is not None and not self._backoff_until.expired(now=timestamp):
            return AvailabilityState(
                GroupAvailability.BACKOFF,
                f"backoff until {self._backoff_until.as_timestamp().epoch_ms()}",
                self._backoff_until.as_timestamp(),
            )

        # Requesting (scale-up in progress)
        with self._slices_lock:
            pending = self._pending_scale_ups
            count = len(self._slices) + pending
        if pending > 0:
            return AvailabilityState(
                GroupAvailability.REQUESTING,
                "scale-up in progress",
            )

        # At capacity
        if count >= self._config.max_slices:
            return AvailabilityState(GroupAvailability.AT_CAPACITY)

        return AvailabilityState(GroupAvailability.AVAILABLE)

    def can_accept_demand(self, timestamp: Timestamp | None = None) -> bool:
        """Whether this group can accept demand for waterfall routing."""
        return self.availability(timestamp).status in {
            GroupAvailability.AVAILABLE,
            GroupAvailability.REQUESTING,
            GroupAvailability.BACKOFF,
        }

    def _get_slice_vm_addresses(self, state: SliceState) -> list[str]:
        """Get VM addresses for a slice."""
        return state.vm_addresses

    def find_slice_for_vm(self, vm_address: str) -> str | None:
        """Find slice_id containing a VM with the given address."""
        with self._slices_lock:
            snapshot = list(self._slices.items())
        for slice_id, state in snapshot:
            if vm_address in self._get_slice_vm_addresses(state):
                return slice_id
        return None

    def terminate_all(self) -> None:
        """Terminate all slices in this scale group."""
        with self._slices_lock:
            snapshot = [s.handle for s in self._slices.values()]
            self._slices.clear()
            self._pending_scale_ups = 0
        for handle in snapshot:
            handle.terminate()

    def to_status(self) -> vm_pb2.ScaleGroupStatus:
        """Build a ScaleGroupStatus proto for the status API."""
        with self._slices_lock:
            snapshot = list(self._slices.values())
        backoff_ts = self._backoff_until.as_timestamp() if self._backoff_until else Timestamp.from_ms(0)
        counts = self.slice_state_counts()
        status = vm_pb2.ScaleGroupStatus(
            name=self.name,
            config=self._config,
            current_demand=self._current_demand,
            peak_demand=self._peak_demand,
            backoff_until=backoff_ts.to_proto(),
            consecutive_failures=self._consecutive_failures,
            last_scale_up=self._last_scale_up.to_proto(),
            last_scale_down=self._last_scale_down.to_proto(),
            slices=[slice_state_to_proto(state) for state in snapshot],
        )
        for state_name, count in counts.items():
            status.slice_state_counts[state_name] = count
        return status
