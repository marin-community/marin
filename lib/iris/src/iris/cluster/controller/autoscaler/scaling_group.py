# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ScalingGroup owns slices and manages scaling state for a single group.

Each ScalingGroup uses a WorkerInfraProvider to create/discover slices, storing
SliceHandle references directly for internal tracking.  It maintains scaling
stats (per-slice idle tracking, backoff, cooldowns) and provides scaling policy
helpers.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from rigging.timing import Deadline, Duration, Timestamp, TokenBucket

from iris.cluster.constraints import (
    CONSTRAINT_REGISTRY,
    AttributeValue,
    Constraint,
    DeviceType,
    ResourceCapacity,
    WellKnownAttribute,
    check_resource_fit,
    evaluate_constraint,
    is_cpu_device_type_constraint,
)
from iris.cluster.controller.autoscaler.models import SliceLifecycleState, SliceState
from iris.cluster.controller.autoscaler.slice_lifecycle import (
    SHORT_LIVED_SLICE_THRESHOLD,
    TRANSITIONS,
    BecameFailed,
    BecameReady,
    CloudFailed,
    CloudReady,
    IdleTimeout,
    InternalTransition,
    NoOp,
    SliceEvent,
    TransitionOutcome,
)
from iris.cluster.controller.db import ControllerDB
from iris.cluster.providers.protocols import WorkerInfraProvider
from iris.cluster.providers.types import Labels, QuotaExhaustedError, SliceHandle
from iris.cluster.types import (
    WorkerStatusMap,
    get_gpu_count,
    get_tpu_count,
)
from iris.rpc import config_pb2, job_pb2, time_pb2, vm_pb2
from iris.time_proto import timestamp_to_proto

logger = logging.getLogger(__name__)


class GroupAvailability(Enum):
    """Availability state for waterfall routing.

    ACCEPTING states (demand stays here, scale-up may be deferred):
    - AVAILABLE: can create new slices immediately
    - COOLDOWN: recently scaled up, next scale-up deferred until cooldown expires
    - REQUESTING: scale-up in progress, capacity incoming

    REJECTING states (demand falls through to lower-priority groups):
    - BACKOFF: slice creation failed, exponential backoff active
    - QUOTA_EXCEEDED: cloud quota exhausted
    - AT_MAX_SLICES: configured slice limit reached
    """

    AVAILABLE = "available"
    COOLDOWN = "cooldown"
    REQUESTING = "requesting"
    AT_MAX_SLICES = "at_max_slices"
    BACKOFF = "backoff"
    QUOTA_EXCEEDED = "quota_exceeded"


@dataclass(frozen=True)
class AvailabilityState:
    """Availability state with reason and optional expiry."""

    status: GroupAvailability
    reason: str = ""
    until: Timestamp | None = None


DEFAULT_SCALE_UP_RATE_LIMIT = 16  # per minute
DEFAULT_SCALE_DOWN_RATE_LIMIT = 32  # per minute
DEFAULT_SCALE_UP_COOLDOWN = Duration.from_minutes(1)
DEFAULT_BACKOFF_INITIAL = Duration.from_minutes(5)
DEFAULT_BACKOFF_MAX = Duration.from_minutes(15)
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_IDLE_THRESHOLD = Duration.from_minutes(10)
DEFAULT_QUOTA_TIMEOUT = Duration.from_minutes(5)


def prepare_slice_config(
    template: config_pb2.SliceConfig,
    parent_config: config_pb2.ScaleGroupConfig,
    label_prefix: str,
) -> config_pb2.SliceConfig:
    """Build a SliceConfig for WorkerInfraProvider.create_slice() from a template.

    Copies the template and sets the name_prefix and managed/scale-group labels.
    Propagates num_vms from the parent ScaleGroupConfig when the template
    doesn't set it. accelerator_type, accelerator_variant, preemptible,
    gpu_count, and disk_size_gb are already derived from resources onto the
    template by _derive_slice_config_from_resources() during config loading.
    """
    labels = Labels(label_prefix)
    config = config_pb2.SliceConfig()
    config.CopyFrom(template)
    config.name_prefix = f"{label_prefix}-{parent_config.name}"
    config.labels[labels.iris_managed] = "true"
    config.labels[labels.iris_scale_group] = parent_config.name

    if not config.num_vms and parent_config.HasField("num_vms"):
        config.num_vms = parent_config.num_vms

    return config


def _region_from_template(template: config_pb2.SliceConfig) -> str | None:
    """Region derived from a scale group's slice template."""
    if template.HasField("gcp") and template.gcp.zone:
        return template.gcp.zone.rsplit("-", 1)[0]
    if template.HasField("coreweave") and template.coreweave.region:
        return template.coreweave.region
    return None


def _zone_from_template(template: config_pb2.SliceConfig) -> str | None:
    """Zone derived from a scale group's slice template."""
    if template.HasField("gcp") and template.gcp.zone:
        return template.gcp.zone
    if template.HasField("coreweave") and template.coreweave.region:
        return template.coreweave.region
    return None


def build_worker_config_for_group(
    base_worker_config: config_pb2.WorkerConfig | None,
    group_config: config_pb2.ScaleGroupConfig,
) -> config_pb2.WorkerConfig | None:
    """Merge base worker config with per-scale-group overrides.

    Returns None when base_worker_config is None (test/local mode).
    """
    if not base_worker_config:
        return None

    wc = config_pb2.WorkerConfig()
    wc.CopyFrom(base_worker_config)

    resources = group_config.resources if group_config.HasField("resources") else None
    if resources is not None:
        wc.accelerator_type = resources.device_type
        if resources.device_variant:
            wc.accelerator_variant = resources.device_variant
        if resources.device_type == config_pb2.ACCELERATOR_TYPE_GPU and resources.device_count > 0:
            wc.gpu_count = resources.device_count
        wc.capacity_type = resources.capacity_type

    if group_config.HasField("worker"):
        for k, v in group_config.worker.attributes.items():
            wc.worker_attributes[k] = v
        if group_config.worker.cache_dir:
            wc.cache_dir = group_config.worker.cache_dir

    template = group_config.slice_template
    region = _region_from_template(template)
    if region and not wc.worker_attributes.get(WellKnownAttribute.REGION):
        wc.worker_attributes[WellKnownAttribute.REGION] = region

    zone = _zone_from_template(template)
    if zone and not wc.worker_attributes.get(WellKnownAttribute.ZONE):
        wc.worker_attributes[WellKnownAttribute.ZONE] = zone

    if group_config.name:
        wc.worker_attributes["scale-group"] = group_config.name

    return wc


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


def slice_state_to_proto(state: SliceState, idle_threshold: Duration | None = None) -> vm_pb2.SliceInfo:
    """Convert a SliceState to a SliceInfo proto for RPC APIs."""
    created_at = state.handle.created_at
    vm_state = _lifecycle_to_vm_state(state.lifecycle)

    is_idle = False
    if idle_threshold is not None and state.lifecycle == SliceLifecycleState.READY:
        if state.last_active.epoch_ms() == 0:
            is_idle = True
        else:
            idle_duration = Duration.from_ms(Timestamp.now().epoch_ms() - state.last_active.epoch_ms())
            is_idle = idle_duration >= idle_threshold

    return vm_pb2.SliceInfo(
        slice_id=state.handle.slice_id,
        scale_group=state.handle.scale_group,
        created_at=time_pb2.Timestamp(epoch_ms=created_at.epoch_ms()),
        vms=[
            vm_pb2.VmInfo(
                vm_id=worker_id,
                state=vm_state,
                created_at=time_pb2.Timestamp(epoch_ms=created_at.epoch_ms()),
                state_changed_at=time_pb2.Timestamp(epoch_ms=created_at.epoch_ms()),
            )
            for worker_id in state.worker_ids
        ],
        error_message=state.error_message,
        last_active=timestamp_to_proto(state.last_active),
        idle=is_idle,
    )


class ScalingGroup:
    """Owns slices for a single scale group.

    Each ScalingGroup:
    - Uses a WorkerInfraProvider to create/discover slices
    - Stores SliceHandle references directly for internal tracking
    - Maintains scaling stats (per-slice idle tracking, backoff, cooldowns)
    - Provides scaling policy helpers (can_scale_up)
    - Owns scale-down logic via per-slice idle timeout tracking
    """

    def __init__(
        self,
        config: config_pb2.ScaleGroupConfig,
        platform: WorkerInfraProvider,
        label_prefix: str = "iris",
        scale_up_cooldown: Duration = DEFAULT_SCALE_UP_COOLDOWN,
        backoff_initial: Duration = DEFAULT_BACKOFF_INITIAL,
        backoff_max: Duration = DEFAULT_BACKOFF_MAX,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        idle_threshold: Duration = DEFAULT_IDLE_THRESHOLD,
        quota_timeout: Duration = DEFAULT_QUOTA_TIMEOUT,
        scale_up_rate_limit: int = DEFAULT_SCALE_UP_RATE_LIMIT,
        scale_down_rate_limit: int = DEFAULT_SCALE_DOWN_RATE_LIMIT,
        db: ControllerDB | None = None,
    ):
        self._config = config
        self._platform = platform
        self._db = db
        self._label_prefix = label_prefix
        self._labels = Labels(label_prefix)
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
        # Quota state (set by scale_up when QuotaExhaustedError is raised)
        self._quota_exceeded_until: Deadline | None = None
        self._quota_reason: str = ""
        self._quota_timeout = quota_timeout

        # Per-group token bucket rate limiter for scale-up API calls
        self._scale_up_bucket = TokenBucket(capacity=scale_up_rate_limit, refill_period=Duration.from_minutes(1))

        # Per-group token bucket rate limiter for scale-down API calls.
        # This replaces the old cooldown-only gate so that multiple idle slices
        # can be terminated in a single cycle, up to the token budget.
        self._scale_down_bucket = TokenBucket(capacity=scale_down_rate_limit, refill_period=Duration.from_minutes(1))

        # Upsert scaling group row so it exists for future updates
        if self._db is not None:
            with self._db.transaction() as cur:
                cur.execute(
                    "INSERT OR IGNORE INTO scaling_groups(name, updated_at_ms) VALUES (?, ?)",
                    (self.name, Timestamp.now().epoch_ms()),
                )

    # -----------------------------------------------------------------------
    # DB write-through helpers
    # -----------------------------------------------------------------------

    def _db_upsert_slice(self, slice_id: str, state: SliceState) -> None:
        if self._db is None:
            return
        with self._db.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO slices "
                "(slice_id, scale_group, lifecycle, worker_ids, created_at_ms, last_active_ms, error_message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    slice_id,
                    self.name,
                    state.lifecycle.value,
                    json.dumps(list(state.worker_ids)),
                    state.handle.created_at.epoch_ms(),
                    state.last_active.epoch_ms(),
                    state.error_message,
                ),
            )

    def _db_remove_slice(self, slice_id: str) -> None:
        if self._db is None:
            return
        with self._db.transaction() as cur:
            cur.execute("DELETE FROM slices WHERE slice_id = ?", (slice_id,))

    def _db_update_group(self) -> None:
        if self._db is None:
            return
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE scaling_groups SET "
                "consecutive_failures=?, backoff_until_ms=?, last_scale_up_ms=?, "
                "last_scale_down_ms=?, quota_exceeded_until_ms=?, quota_reason=?, updated_at_ms=? "
                "WHERE name=?",
                (
                    self._consecutive_failures,
                    self._backoff_until.as_timestamp().epoch_ms() if self._backoff_until else 0,
                    self._last_scale_up.epoch_ms(),
                    self._last_scale_down.epoch_ms(),
                    self._quota_exceeded_until.as_timestamp().epoch_ms() if self._quota_exceeded_until else 0,
                    self._quota_reason,
                    Timestamp.now().epoch_ms(),
                    self.name,
                ),
            )

    def _db_clear_slices(self) -> None:
        if self._db is None:
            return
        with self._db.transaction() as cur:
            cur.execute("DELETE FROM slices WHERE scale_group = ?", (self.name,))

    @property
    def platform(self) -> WorkerInfraProvider:
        """Worker infrastructure provider for this scale group."""
        return self._platform

    @property
    def label_prefix(self) -> str:
        """Label prefix used for slice labels."""
        return self._label_prefix

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
    def buffer_slices(self) -> int:
        """Extra slices to keep warm beyond current demand."""
        return self._config.buffer_slices

    @property
    def max_slices(self) -> int:
        """Maximum number of VM groups allowed."""
        return self._config.max_slices

    @property
    def region(self) -> str | None:
        """Region derived from the slice template."""
        template = self._config.slice_template
        if template.HasField("gcp") and template.gcp.zone:
            return template.gcp.zone.rsplit("-", 1)[0]
        if template.HasField("coreweave") and template.coreweave.region:
            return template.coreweave.region
        return None

    @property
    def zone(self) -> str | None:
        """Zone derived from the slice template."""
        template = self._config.slice_template
        if template.HasField("gcp") and template.gcp.zone:
            return template.gcp.zone
        if template.HasField("coreweave") and template.coreweave.region:
            return template.coreweave.region
        return None

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

    def begin_scale_up(self, timestamp: Timestamp | None = None) -> None:
        """Mark that a scale-up is in progress.

        Increments the pending counter, which is included in slice_count()
        and slice_state_counts(REQUESTING) to prevent over-provisioning.
        Also updates _last_scale_up so the cooldown gates subsequent requests
        even while this one is still in-flight.
        """
        timestamp = timestamp or Timestamp.now()
        with self._slices_lock:
            self._pending_scale_ups += 1
        self._last_scale_up = timestamp
        self._db_update_group()

    def complete_scale_up(self, handle: SliceHandle, timestamp: Timestamp | None = None) -> None:
        """Record a successful scale-up: add the slice and decrement the pending counter."""
        timestamp = timestamp or Timestamp.now()
        with self._slices_lock:
            self._pending_scale_ups = max(0, self._pending_scale_ups - 1)
            state = SliceState(handle=handle)
            self._slices[handle.slice_id] = state
        self._consecutive_failures = 0
        self._backoff_until = None
        self._quota_exceeded_until = None
        self._quota_reason = ""
        self._db_upsert_slice(handle.slice_id, state)
        self._db_update_group()

    def cancel_scale_up(self) -> None:
        """Record a failed scale-up: decrement the pending counter."""
        with self._slices_lock:
            self._pending_scale_ups = max(0, self._pending_scale_ups - 1)

    def dispatch(self, slice_id: str, event: SliceEvent, *, now: Timestamp | None = None) -> TransitionOutcome:
        """Apply a typed lifecycle event, atomically with any cross-machine cascade.

        Behavior note: a FAILED transition removes the slice from _slices and
        hands the SliceHandle to the caller via BecameFailed.handle. The caller
        is responsible for unregistering workers and async-terminating the
        cloud handle. As a consequence, slice_state_counts() and to_status()
        will never observe a slice in the FAILED state for this group — failure
        is surfaced via the action log and the returned outcome, not via
        lingering FAILED entries in the tracked slice map.

        Holds _slices_lock across both the slice mutation and the group
        failure accounting (_apply_failure_locked) so that short-lived
        failures bump consecutive_failures/backoff_until consistently with
        the slice detach. See test_concurrent_failures_account_atomically.
        """
        now = now or Timestamp.now()

        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is None:
                logger.warning("slice event for unknown slice %s (event=%s)", slice_id, event.label)
                return NoOp(slice_id, reason="unknown_slice")

            to_state = TRANSITIONS.get((state.lifecycle, type(event)))
            if to_state is None:
                logger.warning(
                    "no transition for slice %s in state %s on event %s",
                    slice_id,
                    state.lifecycle,
                    event.label,
                )
                return NoOp(slice_id, reason=f"no_transition/{state.lifecycle.value}+{event.label}")

            prior = state.lifecycle
            state.lifecycle = to_state

            outcome: TransitionOutcome
            if to_state == SliceLifecycleState.READY:
                assert isinstance(event, CloudReady)
                state.last_active = now
                state.worker_ids = [w.worker_id for w in event.workers]
                outcome = BecameReady(slice_id=slice_id, workers=event.workers)

            elif to_state == SliceLifecycleState.FAILED:
                handle = state.handle
                age = Duration.from_ms(now.epoch_ms() - handle.created_at.epoch_ms())
                short_lived = age < SHORT_LIVED_SLICE_THRESHOLD
                triggered = event.counts_toward_backoff and short_lived
                if triggered:
                    self._apply_failure_locked(now)
                if isinstance(event, CloudFailed) and event.error_message:
                    state.error_message = event.error_message
                self._slices.pop(slice_id, None)
                self._last_scale_down = now
                outcome = BecameFailed(slice_id=slice_id, handle=handle, event=event, triggered_backoff=triggered)

            else:  # BOOTING -> INITIALIZING
                outcome = InternalTransition(slice_id=slice_id, prior=prior, new_state=to_state)

        # Persistence outside the lock.
        if isinstance(outcome, BecameFailed):
            self._db_remove_slice(slice_id)
            if outcome.triggered_backoff:
                self._db_update_group()
        elif slice_id in self._slices:
            self._db_upsert_slice(slice_id, self._slices[slice_id])

        logger.info(
            "slice_transition group=%s slice=%s %s→%s event=%s backoff=%s",
            self.name,
            slice_id,
            prior.value,
            to_state.value,
            event.label,
            isinstance(outcome, BecameFailed) and outcome.triggered_backoff,
        )
        return outcome

    def _apply_failure_locked(self, now: Timestamp) -> None:
        """Mutate group failure state. Caller must hold _slices_lock."""
        self._consecutive_failures += 1
        backoff_duration = self._backoff_initial * (self._backoff_factor ** (self._consecutive_failures - 1))
        backoff_duration = min(backoff_duration, self._backoff_max)
        self._backoff_until = Deadline.after(now, backoff_duration)

    def reconcile(self) -> None:
        """Discover and adopt existing slices from the cloud.

        Used in tests to populate a scaling group with pre-injected slices.
        Production restore uses prepare_for_restore() + restore_scaling_group().
        Skips operator-created manual slices (iris_manual=true), which the
        autoscaler must not track or scale down.
        """
        zones = _zones_from_config(self._config)
        labels = {self._labels.iris_scale_group: self._config.name}
        slice_handles = self._platform.list_slices(zones, labels)
        with self._slices_lock:
            for handle in slice_handles:
                if handle.labels.get(self._labels.iris_manual) == "true":
                    continue
                state = SliceState(handle=handle)
                self._slices[handle.slice_id] = state
                self._db_upsert_slice(handle.slice_id, state)

    def scale_up(
        self,
        tags: dict[str, str] | None = None,
        timestamp: Timestamp | None = None,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> SliceHandle:
        """Create a new slice via the platform.

        Does NOT add to _slices tracking. Use begin_scale_up/complete_scale_up
        for lifecycle tracking. QuotaExhaustedError propagates to the caller.

        Args:
            tags: Optional extra labels/tags for the slice (merged with managed labels)
            timestamp: Optional timestamp (for testing)
            worker_config: Worker settings passed to platform.create_slice()

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
        logger.info(
            "Scale group %s: create_slice accel=%s:%s gpu_count=%d labels=%s coreweave_instance=%s",
            self.name,
            slice_config.accelerator_type,
            slice_config.accelerator_variant,
            slice_config.gpu_count,
            dict(slice_config.labels),
            slice_config.coreweave.instance_type if slice_config.HasField("coreweave") else "n/a",
        )

        return self._platform.create_slice(slice_config, worker_config=worker_config)

    def scale_down(self, slice_id: str, timestamp: Timestamp | None = None) -> None:
        """Terminate a slice.

        Always removes the slice from in-memory tracking and the DB, even if
        the cloud terminate call fails (e.g. resource already deleted by
        preemption). This prevents ghost slices that the autoscaler counts as
        live capacity but that no longer exist.

        Args:
            slice_id: ID of the slice to terminate
            timestamp: Optional timestamp (for testing)
        """
        handle = self.detach_slice(slice_id, timestamp=timestamp)
        if handle is not None:
            self._terminate_slice_handle(handle, context="cleaning up anyway")

    def detach_slice(self, slice_id: str, timestamp: Timestamp | None = None) -> SliceHandle | None:
        """Remove a slice from tracking and persistence without terminating it."""
        timestamp = timestamp or Timestamp.now()
        with self._slices_lock:
            state = self._slices.pop(slice_id, None)
        if state is None:
            return None
        self._last_scale_down = timestamp
        self._db_remove_slice(slice_id)
        self._db_update_group()
        return state.handle

    def _terminate_slice_handle(self, handle: SliceHandle, *, context: str) -> None:
        try:
            handle.terminate()
        except QuotaExhaustedError as e:
            # Delete quota retries are exhausted inside the provider; log a
            # terse warning without a stack trace since this is a known-benign
            # rate limit, not a crash.
            logger.warning(
                "Scale group %s: terminate() rate-limited for slice %s (%s), %s",
                self.name,
                handle.slice_id,
                e,
                context,
            )
        except Exception:
            logger.warning(
                "Scale group %s: terminate() failed for slice %s, %s",
                self.name,
                handle.slice_id,
                context,
                exc_info=True,
            )

    def slice_handles(self) -> list[SliceHandle]:
        """All slice handles in this scale group."""
        with self._slices_lock:
            return [s.handle for s in self._slices.values()]

    def non_ready_slice_handles(self) -> list[tuple[str, SliceHandle]]:
        """Snapshot non-READY slice handles for background lifecycle polling."""
        with self._slices_lock:
            return [
                (slice_id, state.handle)
                for slice_id, state in self._slices.items()
                if state.lifecycle in (SliceLifecycleState.BOOTING, SliceLifecycleState.INITIALIZING)
            ]

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

    def can_fit_resources(self, resources: job_pb2.ResourceSpecProto) -> bool:
        """Check whether a demand entry's resources fit within one VM."""
        return self.check_resource_fit(resources) is None

    def check_resource_fit(self, resources: job_pb2.ResourceSpecProto) -> str | None:
        """Check whether a demand entry's resources fit within one VM.

        Unconfigured group resource values (0 in the proto) are passed as None
        to ResourceCapacity, meaning unlimited for that dimension.

        Returns None if the resources fit, or a human-readable reason string.
        """
        sg_resources = self.resources
        if sg_resources is None:
            return f"group '{self.name}' has no resources configured"

        device_count = get_gpu_count(resources.device) + get_tpu_count(resources.device)
        available = ResourceCapacity(
            cpu_millicores=sg_resources.cpu_millicores or None,
            memory_bytes=sg_resources.memory_bytes or None,
            disk_bytes=sg_resources.disk_bytes or None,
            gpu_count=sg_resources.device_count or None,
        )
        required = ResourceCapacity(
            cpu_millicores=resources.cpu_millicores,
            memory_bytes=resources.memory_bytes,
            disk_bytes=resources.disk_bytes,
            gpu_count=device_count,
        )
        return check_resource_fit(available, required)

    # Minimum elapsed time before a last_active change is flushed to the DB.
    # Avoids one transaction per active slice per autoscaler tick for stable workloads.
    _ACTIVITY_WRITE_THRESHOLD_MS: int = 30_000

    def update_slice_activity(self, worker_status_map: WorkerStatusMap, timestamp: Timestamp) -> None:
        """Update activity timestamps for all slices based on worker status.

        For each slice, if any worker has running tasks, update its last_active timestamp.
        All DB writes are batched into a single transaction.
        """
        with self._slices_lock:
            snapshot = list(self._slices.items())

        # Determine which slices are active and whether they need a DB write.
        # _slice_has_active_workers is called outside the lock (reads worker_status_map, not _slices).
        active_slice_ids: list[str] = []
        needs_db_write: list[str] = []
        for slice_id, state in snapshot:
            if not self._slice_has_active_workers(state, worker_status_map):
                continue
            active_slice_ids.append(slice_id)
            elapsed_ms = timestamp.epoch_ms() - state.last_active.epoch_ms()
            if elapsed_ms >= self._ACTIVITY_WRITE_THRESHOLD_MS:
                needs_db_write.append(slice_id)

        if not active_slice_ids:
            return

        # Mutate last_active under the lock so readers see a consistent value.
        with self._slices_lock:
            for slice_id in active_slice_ids:
                if slice_id in self._slices:
                    self._slices[slice_id].last_active = timestamp

        if not needs_db_write or self._db is None:
            return

        # Snapshot state under lock before opening the transaction to avoid
        # holding _slices_lock and _db._lock simultaneously.
        with self._slices_lock:
            to_write = [(sid, self._slices[sid]) for sid in needs_db_write if sid in self._slices]

        with self._db.transaction() as cur:
            for slice_id, state in to_write:
                cur.execute(
                    "INSERT OR REPLACE INTO slices "
                    "(slice_id, scale_group, lifecycle, worker_ids, created_at_ms, last_active_ms, error_message) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        slice_id,
                        self.name,
                        state.lifecycle.value,
                        json.dumps(list(state.worker_ids)),
                        state.handle.created_at.epoch_ms(),
                        state.last_active.epoch_ms(),
                        state.error_message,
                    ),
                )

    def _slice_has_active_workers(self, state: SliceState, worker_status_map: WorkerStatusMap) -> bool:
        """Check if any worker in a slice has running tasks."""
        for worker_id in self._get_slice_worker_ids(state):
            status = worker_status_map.get(worker_id)
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
        worker_status_map: WorkerStatusMap,
        target_capacity: int,
        timestamp: Timestamp,
    ) -> list[BecameFailed]:
        """Scale down idle slices that exceed target capacity.

        Dispatches IdleTimeout through the state machine for each eligible
        slice so that all slice teardowns go through the same path. Returns
        the list of BecameFailed outcomes so the caller can execute
        side-effect work (unregister workers, async terminate the handle).

        Terminates multiple idle slices in a single call, rate-limited by a
        token bucket.
        """
        self.update_slice_activity(worker_status_map, timestamp)

        counts = self.slice_state_counts()
        ready = counts[SliceLifecycleState.READY]
        pending = counts[SliceLifecycleState.BOOTING] + counts[SliceLifecycleState.INITIALIZING]

        if ready + pending <= target_capacity or ready <= target_capacity:
            return []

        results: list[BecameFailed] = []
        for slice_state in self.get_idle_slices(timestamp):
            if ready - len(results) <= target_capacity:
                break

            # Verify idle before acquiring a rate-limit token so that
            # failed verifications don't waste tokens.
            if not self._verify_slice_idle(slice_state, worker_status_map):
                continue

            if not self.acquire_scale_down_token(timestamp):
                if results:
                    logger.info(
                        "Scale group %s: scale down rate-limited after %d terminations",
                        self.name,
                        len(results),
                    )
                break

            last_active = slice_state.last_active
            idle_duration = Duration.from_ms(timestamp.epoch_ms() - last_active.epoch_ms())
            logger.info(
                "Scale group %s: scaling down slice %s "
                "(idle for %dms, never_active=%s, ready=%d/%d, pending=%d, target=%d)",
                self.name,
                slice_state.handle.slice_id,
                idle_duration.to_ms(),
                last_active.epoch_ms() == 0,
                ready,
                self.num_vms,
                pending,
                target_capacity,
            )
            event = IdleTimeout(target_capacity=target_capacity, ready_before=ready)
            outcome = self.dispatch(slice_state.handle.slice_id, event, now=timestamp)
            if isinstance(outcome, BecameFailed):
                results.append(outcome)

        return results

    def _verify_slice_idle(self, state: SliceState, worker_status_map: WorkerStatusMap) -> bool:
        """Verify all workers in a slice are idle before termination.

        Requires at least one known worker to be idle. If no workers are known at all
        (none in worker_status_map), returns False -- the slice may still be booting.
        Zombie slices where workers have disappeared are handled by worker heartbeat timeouts.
        """
        has_known_worker = False
        for worker_id in self._get_slice_worker_ids(state):
            status = worker_status_map.get(worker_id)
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

    def acquire_scale_up_token(self, timestamp: Timestamp | None = None) -> bool:
        """Try to acquire a scale-up rate limit token. Returns False if rate-limited."""
        return self._scale_up_bucket.try_acquire(now=timestamp)

    def acquire_scale_down_token(self, timestamp: Timestamp | None = None) -> bool:
        """Try to acquire a scale-down rate limit token. Returns False if rate-limited."""
        return self._scale_down_bucket.try_acquire(now=timestamp)

    def record_quota_exceeded(self, reason: str, timestamp: Timestamp | None = None) -> None:
        """Record a quota exhaustion event, blocking scale-up until the quota timeout elapses."""
        timestamp = timestamp or Timestamp.now()
        self._quota_exceeded_until = Deadline.after(timestamp, self._quota_timeout)
        self._quota_reason = reason
        self._db_update_group()

    def record_failure(self, timestamp: Timestamp | None = None) -> None:
        """Record a scale-up failure (e.g. platform.create_slice exception).

        Used by the autoscaler when a slice never makes it into _slices and
        therefore can't go through the slice state machine. Slice-lifecycle
        cascades use _apply_failure_locked directly under the dispatch lock.
        """
        timestamp = timestamp or Timestamp.now()
        with self._slices_lock:
            self._apply_failure_locked(timestamp)
        self._db_update_group()

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

    def matches_device_requirement(self, device_type: DeviceType, device_variants: frozenset[str] | None) -> bool:
        """Check if this group can satisfy the given device requirements.

        Matching rules:
        - CPU demand: matches ANY group (all VMs have CPUs)
        - GPU/TPU with device_variants=None: matches any group of the same device type
        - GPU/TPU with specific variants: group variant must be in the set (case-insensitive)
        """
        if device_type == DeviceType.CPU:
            return True  # CPU jobs can run on ANY group

        group_type = self._get_device_type()
        if group_type != device_type:
            return False

        if device_variants is None:
            return True
        group_variant = self._config.resources.device_variant if self._config.HasField("resources") else ""
        return group_variant.lower() in {v.lower() for v in device_variants}

    def to_attributes(self) -> dict[str, AttributeValue]:
        """Express this group's routing properties as worker-style attributes.

        Enables the same evaluate_constraint + ConstraintIndex infrastructure
        used for worker matching to also work for scaling group routing.
        """
        attrs: dict[str, AttributeValue] = {}
        attrs[WellKnownAttribute.DEVICE_TYPE] = AttributeValue(self._get_device_type().value)
        if self._config.HasField("resources") and self._config.resources.device_variant:
            attrs[WellKnownAttribute.DEVICE_VARIANT] = AttributeValue(self._config.resources.device_variant.lower())
        if self._config.HasField("resources"):
            is_preemptible = self._config.resources.capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE
            attrs[WellKnownAttribute.PREEMPTIBLE] = AttributeValue(str(is_preemptible).lower())
        region = self.region
        if region:
            attrs[WellKnownAttribute.REGION] = AttributeValue(region)
        zone = self.zone
        if zone:
            attrs[WellKnownAttribute.ZONE] = AttributeValue(zone)
        return attrs

    def matches_constraints(self, constraints: Sequence[Constraint]) -> bool:
        """Check if this group satisfies the given constraints.

        Only evaluates routing constraints (device-type, device-variant,
        preemptible, region, zone). Non-routing constraints (tpu-name, etc.)
        and unknown constraints are scheduler-only and skipped here. CPU
        device-type constraints are also skipped since CPU jobs match any group.
        """
        attrs = self.to_attributes()
        for c in constraints:
            if is_cpu_device_type_constraint(c):
                continue
            desc = CONSTRAINT_REGISTRY.get(c.key)
            if desc is None or not desc.routing:
                continue
            if not evaluate_constraint(attrs.get(c.key), c):
                return False
        return True

    def _get_device_type(self) -> DeviceType:
        """Get device type from resources."""
        if not self._config.HasField("resources"):
            return DeviceType.CPU
        accel = self._config.resources.device_type
        if accel == config_pb2.ACCELERATOR_TYPE_GPU:
            return DeviceType.GPU
        elif accel == config_pb2.ACCELERATOR_TYPE_TPU:
            return DeviceType.TPU
        return DeviceType.CPU

    def availability(self, timestamp: Timestamp | None = None) -> AvailabilityState:
        """Compute current availability state for waterfall routing.

        All states are computed from timestamps — no external state setting.
        Priority: QUOTA_EXCEEDED > BACKOFF > REQUESTING > AT_MAX_SLICES > COOLDOWN > AVAILABLE
        """
        timestamp = timestamp or Timestamp.now()

        if self._quota_exceeded_until is not None and not self._quota_exceeded_until.expired(now=timestamp):
            return AvailabilityState(
                GroupAvailability.QUOTA_EXCEEDED,
                self._quota_reason,
                self._quota_exceeded_until.as_timestamp(),
            )

        if self._backoff_until is not None and not self._backoff_until.expired(now=timestamp):
            return AvailabilityState(
                GroupAvailability.BACKOFF,
                f"{self._consecutive_failures} consecutive failure{'s' if self._consecutive_failures != 1 else ''}",
                self._backoff_until.as_timestamp(),
            )

        with self._slices_lock:
            pending = self._pending_scale_ups
            count = len(self._slices) + pending
            has_inflight = pending > 0 or any(
                s.lifecycle in (SliceLifecycleState.BOOTING, SliceLifecycleState.INITIALIZING)
                for s in self._slices.values()
            )
        if pending > 0:
            return AvailabilityState(
                GroupAvailability.REQUESTING,
                "scale-up in progress",
            )

        if count >= self._config.max_slices:
            # When all slice slots are filled but some are still booting/initializing,
            # accept demand so it doesn't waterfall to other groups. The routing budget
            # already accounts for in-flight capacity via max_vms.
            if has_inflight:
                return AvailabilityState(GroupAvailability.COOLDOWN, "at max_slices with in-flight capacity")
            return AvailabilityState(GroupAvailability.AT_MAX_SLICES)

        cooldown_end = self._last_scale_up.add(self._scale_up_cooldown)
        if self._last_scale_up.epoch_ms() > 0 and timestamp.before(cooldown_end):
            return AvailabilityState(GroupAvailability.COOLDOWN, "scale-up cooldown", cooldown_end)

        return AvailabilityState(GroupAvailability.AVAILABLE)

    def can_accept_demand(self, timestamp: Timestamp | None = None) -> bool:
        """Whether this group can accept demand for waterfall routing.

        ACCEPTING states keep demand here: AVAILABLE, COOLDOWN, REQUESTING.
        REJECTING states cause demand to fall through: BACKOFF, QUOTA_EXCEEDED, AT_MAX_SLICES.
        """
        return self.availability(timestamp).status in {
            GroupAvailability.AVAILABLE,
            GroupAvailability.COOLDOWN,
            GroupAvailability.REQUESTING,
        }

    def _get_slice_worker_ids(self, state: SliceState) -> list[str]:
        """Get worker IDs for a slice."""
        return state.worker_ids

    def find_slice_for_worker(self, worker_id: str) -> str | None:
        """Find slice_id containing a worker with the given ID."""
        with self._slices_lock:
            snapshot = list(self._slices.items())
        for slice_id, state in snapshot:
            if worker_id in self._get_slice_worker_ids(state):
                return slice_id
        return None

    def get_slice_worker_ids(self, slice_id: str) -> list[str]:
        """Get all worker IDs for a slice. Returns empty list if not found."""
        with self._slices_lock:
            state = self._slices.get(slice_id)
        if state is None:
            return []
        return list(self._get_slice_worker_ids(state))

    def terminate_all(self) -> None:
        """Terminate all slices in this scale group.

        Continues terminating remaining slices even if individual terminate
        calls fail, to avoid leaking cloud resources.
        """
        with self._slices_lock:
            snapshot = [s.handle for s in self._slices.values()]
            self._slices.clear()
            self._pending_scale_ups = 0
        for handle in snapshot:
            try:
                handle.terminate()
            except QuotaExhaustedError as e:
                logger.warning(
                    "Scale group %s: terminate() rate-limited for slice %s (%s) during terminate_all, continuing",
                    self.name,
                    handle.slice_id,
                    e,
                )
            except Exception:
                logger.warning(
                    "Scale group %s: terminate() failed for slice %s during terminate_all, continuing",
                    self.name,
                    handle.slice_id,
                    exc_info=True,
                )
        self._db_clear_slices()

    def restore_from_snapshot(
        self,
        slices: dict[str, SliceState],
        consecutive_failures: int,
        last_scale_up: Timestamp,
        last_scale_down: Timestamp,
        backoff_until: Deadline | None,
        quota_exceeded_until: Deadline | None,
        quota_reason: str,
    ) -> None:
        """Restore state from a snapshot. Called before the autoscaler loop starts."""
        with self._slices_lock:
            self._slices = slices
        self._consecutive_failures = consecutive_failures
        self._last_scale_up = last_scale_up
        self._last_scale_down = last_scale_down
        if backoff_until is not None:
            self._backoff_until = backoff_until
        if quota_exceeded_until is not None:
            self._quota_exceeded_until = quota_exceeded_until
            self._quota_reason = quota_reason

    def to_status(self) -> vm_pb2.ScaleGroupStatus:
        """Build a ScaleGroupStatus proto for the status API."""
        with self._slices_lock:
            snapshot = list(self._slices.values())
        now = Timestamp.now()
        availability = self.availability(now)
        backoff_ts = self._backoff_until.as_timestamp() if self._backoff_until else Timestamp.from_ms(0)
        blocked_until = availability.until if availability.until is not None else Timestamp.from_ms(0)
        counts = self.slice_state_counts()

        cooldown_until = Timestamp.from_ms(0)
        if self._last_scale_up.epoch_ms() > 0:
            cooldown_end = self._last_scale_up.add(self._scale_up_cooldown)
            if now.before(cooldown_end):
                cooldown_until = cooldown_end

        status = vm_pb2.ScaleGroupStatus(
            name=self.name,
            config=self._config,
            current_demand=self._current_demand,
            peak_demand=self._peak_demand,
            backoff_until=timestamp_to_proto(backoff_ts),
            consecutive_failures=self._consecutive_failures,
            last_scale_up=timestamp_to_proto(self._last_scale_up),
            last_scale_down=timestamp_to_proto(self._last_scale_down),
            availability_status=availability.status.value,
            availability_reason=availability.reason,
            blocked_until=timestamp_to_proto(blocked_until),
            scale_up_cooldown_until=timestamp_to_proto(cooldown_until),
            slices=[slice_state_to_proto(state, idle_threshold=self._idle_threshold) for state in snapshot],
            idle_threshold_ms=self._idle_threshold.to_ms(),
        )
        for state_name, count in counts.items():
            status.slice_state_counts[state_name] = count
        return status


# ---------------------------------------------------------------------------
# Checkpoint restore: reconcile checkpointed group state against live cloud
# ---------------------------------------------------------------------------


@dataclass
class SliceSnapshot:
    """Lightweight record of a persisted slice, read from the slices DB table."""

    slice_id: str
    scale_group: str
    lifecycle: str
    worker_ids: list[str] = field(default_factory=list)
    created_at_ms: int = 0
    last_active_ms: int = 0
    error_message: str = ""


@dataclass
class GroupSnapshot:
    """Lightweight record of a persisted scaling group, read from the DB."""

    name: str
    slices: list[SliceSnapshot] = field(default_factory=list)
    consecutive_failures: int = 0
    backoff_until_ms: int = 0
    last_scale_up_ms: int = 0
    last_scale_down_ms: int = 0
    quota_exceeded_until_ms: int = 0
    quota_reason: str = ""


@dataclass
class ScalingGroupRestoreResult:
    """Result of restoring a single scaling group from checkpoint metadata."""

    slices: dict[str, SliceState] = field(default_factory=dict)
    consecutive_failures: int = 0
    backoff_active: bool = False
    quota_exceeded_active: bool = False
    quota_reason: str = ""
    discarded_count: int = 0
    adopted_count: int = 0
    last_scale_up: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    last_scale_down: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    backoff_until: Deadline | None = None
    quota_exceeded_until: Deadline | None = None


def restore_scaling_group(
    group_snapshot: GroupSnapshot,
    cloud_handles: list[SliceHandle],
    label_prefix: str,
) -> ScalingGroupRestoreResult:
    """Reconcile checkpointed group slices against pre-fetched cloud handles."""
    cloud_by_id: dict[str, SliceHandle] = {h.slice_id: h for h in cloud_handles}
    checkpoint_slices = {s.slice_id: s for s in group_snapshot.slices}

    result = ScalingGroupRestoreResult()
    result.consecutive_failures = group_snapshot.consecutive_failures

    for slice_id, slice_snap in checkpoint_slices.items():
        cloud_handle = cloud_by_id.get(slice_id)
        if cloud_handle is None:
            logger.info("Scaling group %s: discarding slice %s (missing from cloud)", group_snapshot.name, slice_id)
            result.discarded_count += 1
            continue

        try:
            lifecycle = SliceLifecycleState(slice_snap.lifecycle)
        except ValueError:
            logger.warning(
                "Scaling group %s: unknown lifecycle %r for slice %s, defaulting to BOOTING",
                group_snapshot.name,
                slice_snap.lifecycle,
                slice_id,
            )
            lifecycle = SliceLifecycleState.BOOTING

        result.slices[slice_id] = SliceState(
            handle=cloud_handle,
            lifecycle=lifecycle,
            worker_ids=list(slice_snap.worker_ids),
            last_active=Timestamp.from_ms(slice_snap.last_active_ms),
            error_message=slice_snap.error_message,
        )

    for slice_id, cloud_handle in cloud_by_id.items():
        if slice_id in checkpoint_slices:
            continue
        logger.info("Scaling group %s: adopting unknown cloud slice %s as BOOTING", group_snapshot.name, slice_id)
        result.slices[slice_id] = SliceState(handle=cloud_handle, lifecycle=SliceLifecycleState.BOOTING)
        result.adopted_count += 1

    if group_snapshot.backoff_until_ms > 0:
        backoff_ts = Timestamp.from_ms(group_snapshot.backoff_until_ms)
        result.backoff_until = Deadline.after(backoff_ts, Duration.from_ms(0))
        result.backoff_active = not result.backoff_until.expired()

    if group_snapshot.quota_exceeded_until_ms > 0:
        quota_ts = Timestamp.from_ms(group_snapshot.quota_exceeded_until_ms)
        result.quota_exceeded_until = Deadline.after(quota_ts, Duration.from_ms(0))
        result.quota_exceeded_active = not result.quota_exceeded_until.expired()
        result.quota_reason = group_snapshot.quota_reason

    if group_snapshot.last_scale_up_ms > 0:
        result.last_scale_up = Timestamp.from_ms(group_snapshot.last_scale_up_ms)
    if group_snapshot.last_scale_down_ms > 0:
        result.last_scale_down = Timestamp.from_ms(group_snapshot.last_scale_down_ms)

    logger.info(
        "Restored scaling group %s: %d slices (%d discarded, %d adopted), consecutive_failures=%d, "
        "backoff_active=%s, quota_exceeded=%s",
        group_snapshot.name,
        len(result.slices),
        result.discarded_count,
        result.adopted_count,
        result.consecutive_failures,
        result.backoff_active,
        result.quota_exceeded_active,
    )
    return result
