# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ScalingGroup owns slices and manages scaling state for a single group.

Each ScalingGroup uses a WorkerInfraProvider to create/discover slices, storing
SliceHandle references directly for internal tracking.  It maintains scaling
stats (per-slice idle tracking, backoff, cooldowns) and provides scaling policy
helpers.
"""

import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum

from rigging.timing import Duration, Timestamp, TokenBucket

from iris.chaos import chaos_raise
from iris.cluster.config import ScaleGroupConfig, ScaleGroupResources, SliceConfig, WorkerConfig, slice_template_region
from iris.cluster.constraints import (
    CONSTRAINT_REGISTRY,
    AttributeValue,
    Constraint,
    DeviceType,
    ResourceCapacity,
    WellKnownAttribute,
    accelerator_type_to_string,
    check_resource_fit,
    evaluate_constraint,
    is_cpu_device_type_constraint,
)
from iris.cluster.controller.autoscaler.backoff_detector import (
    DEFAULT_SHORT_LIVED_THRESHOLD,
    BackoffDetector,
    GroupHealth,
    SliceFate,
)
from iris.cluster.controller.autoscaler.state import GroupPersist, SlicePersist
from iris.cluster.platforms.protocols import WorkerInfraProvider
from iris.cluster.platforms.types import Labels, QuotaExhaustedError, SliceHandle
from iris.cluster.types import AcceleratorType, CapacityType, WorkerStatusMap, get_gpu_count, get_tpu_count
from iris.rpc import job_pb2, time_pb2, vm_pb2
from iris.time_proto import timestamp_to_proto

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
DEFAULT_IDLE_THRESHOLD = Duration.from_minutes(10)
DEFAULT_QUOTA_TIMEOUT = Duration.from_minutes(5)


@dataclass
class SliceState:
    """Per-slice state tracked by ScalingGroup.

    Consolidates the slice handle with its associated tracking state
    (idle timeout, lifecycle) into a single structure.
    lifecycle and worker_ids are populated eagerly by the bootstrap thread.
    """

    handle: SliceHandle
    # Timestamp at which the slice's workers all became idle. None means
    # the slice is currently active or has never been observed; in both
    # cases the slice is not eligible for scale-down. Scale-down kicks in
    # once `now - quiet_since >= idle_threshold`. Memory-only.
    quiet_since: Timestamp | None = None
    lifecycle: SliceLifecycleState = SliceLifecycleState.BOOTING
    worker_ids: list[str] = field(default_factory=list)
    # worker_id -> reachable http://host:port URL. Populated at READY transition
    # from the slice handle's worker info. Empty after a controller restart for
    # already-READY slices — the health probe lazy-fetches via
    # handle.describe() in that case. Memory-only.
    worker_urls: dict[str, str] = field(default_factory=dict)
    # worker_id -> consecutive /health probe failures since the last
    # success. Reset on a healthy response; once any worker crosses
    # CONSECUTIVE_FAILURE_THRESHOLD the slice is terminated. Memory-only.
    ping_failures: dict[str, int] = field(default_factory=dict)
    # Consecutive health-probe passes where the slice's cloud allocation
    # reported zero workers (so there was nothing to /health-probe). Reset the
    # moment any worker becomes probeable; once it crosses CONSECUTIVE_FAILURE_THRESHOLD
    # the slice is terminated. This catches a slice whose backing allocation
    # vanished while it had no cached worker URLs (e.g. preempted after a
    # controller restart), which the per-worker counters never observe.
    # Memory-only.
    no_worker_probes: int = 0
    # Timestamp of the first describe in the current run of UNKNOWN observations,
    # cleared whenever the slice resolves to a concrete state (e.g. READY). The
    # unresolvable-timeout is measured from here, NOT from created_at, so a single
    # transient UNKNOWN never terminates a long-running or freshly-adopted slice.
    # Memory-only.
    unknown_since: Timestamp | None = None
    error_message: str = ""


def prepare_slice_config(
    template: SliceConfig,
    parent_config: ScaleGroupConfig,
    label_prefix: str,
) -> SliceConfig:
    """Build a SliceConfig for WorkerInfraProvider.create_slice() from a template.

    Copies the template and sets the name_prefix and managed/scale-group labels.
    Propagates num_vms from the parent ScaleGroupConfig when the template
    doesn't set it. accelerator_type, accelerator_variant, preemptible,
    gpu_count, and disk_size_gb are already derived from resources onto the
    template during config loading.
    """
    labels = Labels(label_prefix)
    config = template.model_copy(deep=True)
    config.name_prefix = f"{label_prefix}-{parent_config.name}"
    config.labels[labels.iris_managed] = "true"
    config.labels[labels.iris_scale_group] = parent_config.name

    if not config.num_vms and parent_config.num_vms is not None:
        config.num_vms = parent_config.num_vms

    return config


def _region_from_template(template: SliceConfig) -> str | None:
    """Region derived from a scale group's slice template."""
    if template.gcp is not None and template.gcp.zone:
        return template.gcp.zone.rsplit("-", 1)[0]
    if template.coreweave is not None and template.coreweave.region:
        return template.coreweave.region
    return None


def _zone_from_template(template: SliceConfig) -> str | None:
    """Zone derived from a scale group's slice template."""
    if template.gcp is not None and template.gcp.zone:
        return template.gcp.zone
    if template.coreweave is not None and template.coreweave.region:
        return template.coreweave.region
    return None


def build_worker_config_for_group(
    base_worker_config: WorkerConfig | None,
    group_config: ScaleGroupConfig,
) -> WorkerConfig | None:
    """Merge base worker config with per-scale-group overrides.

    Returns None when base_worker_config is None (test/local mode).
    """
    if not base_worker_config:
        return None

    wc = base_worker_config.model_copy(deep=True)

    resources = group_config.resources
    if resources is not None:
        wc.accelerator_type = resources.device_type
        if resources.device_variant:
            wc.accelerator_variant = resources.device_variant
        if resources.device_type == AcceleratorType.GPU and resources.device_count > 0:
            wc.gpu_count = resources.device_count
        wc.capacity_type = resources.capacity_type
        # Advertised scheduling CPU capacity; the worker reports this instead of
        # the probed host count so operators can over-commit CPU via config.
        wc.cpu_millicores = resources.cpu_millicores

    if group_config.worker is not None:
        for k, v in group_config.worker.attributes.items():
            wc.worker_attributes[k] = v
        if group_config.worker.cache_dir:
            wc.cache_dir = group_config.worker.cache_dir

    template = group_config.slice_template
    region = _region_from_template(template) if template is not None else None
    if region and not wc.worker_attributes.get(WellKnownAttribute.REGION):
        wc.worker_attributes[WellKnownAttribute.REGION] = region

    zone = _zone_from_template(template) if template is not None else None
    if zone and not wc.worker_attributes.get(WellKnownAttribute.ZONE):
        wc.worker_attributes[WellKnownAttribute.ZONE] = zone

    if group_config.name:
        wc.worker_attributes["scale-group"] = group_config.name

    return wc


def _zones_from_config(config: ScaleGroupConfig) -> list[str]:
    """Extract zones from ScaleGroupConfig's slice_template.

    Raises ValueError for GCP configs with no zones, since reconcile and
    list_slices would silently do nothing.
    """
    if config.slice_template is None or config.slice_template.gcp is None:
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
    if idle_threshold is not None and state.lifecycle == SliceLifecycleState.READY and state.quiet_since is not None:
        idle_duration = Duration.from_ms(Timestamp.now().epoch_ms() - state.quiet_since.epoch_ms())
        is_idle = idle_duration >= idle_threshold

    # The dashboard renders "idle for {now - last_active}" on idle slices.
    # For an active slice the value is unused, so report `now`; for an idle
    # slice, report the transition time (= quiet_since).
    last_active_ts = state.quiet_since if state.quiet_since is not None else Timestamp.now()

    return vm_pb2.SliceInfo(
        slice_id=state.handle.slice_id,
        scale_group=state.handle.scale_group,
        state=state.lifecycle.value,
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
        last_active=timestamp_to_proto(last_active_ts),
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
        config: ScaleGroupConfig,
        platform: WorkerInfraProvider,
        label_prefix: str = "iris",
        idle_threshold: Duration = DEFAULT_IDLE_THRESHOLD,
        quota_timeout: Duration = DEFAULT_QUOTA_TIMEOUT,
        scale_up_rate_limit: int = DEFAULT_SCALE_UP_RATE_LIMIT,
        scale_down_rate_limit: int = DEFAULT_SCALE_DOWN_RATE_LIMIT,
        short_lived_threshold: Duration = DEFAULT_SHORT_LIVED_THRESHOLD,
        detector: BackoffDetector | None = None,
    ):
        self._config = config
        self._platform = platform
        self._label_prefix = label_prefix
        self._labels = Labels(label_prefix)
        self._slices: dict[str, SliceState] = {}
        self._pending_scale_ups: int = 0
        self._slices_lock = threading.Lock()

        # Demand tracking (simple current/peak, no history)
        self._current_demand: int = 0
        self._peak_demand: int = 0

        self._idle_threshold = idle_threshold

        # Informational timestamps surfaced in the status proto.
        self._last_scale_up: Timestamp = Timestamp.from_ms(0)
        self._last_scale_down: Timestamp = Timestamp.from_ms(0)

        # The detector owns the scale-up bucket and the quota-exhaustion gate;
        # it is the single source of truth for "can this group scale up now?".
        scale_up_bucket = TokenBucket(capacity=scale_up_rate_limit, refill_period=Duration.from_minutes(1))
        self._detector = (
            detector
            if detector is not None
            else BackoffDetector(
                group_name=config.name,
                scale_up_bucket=scale_up_bucket,
                base_scale_up_per_minute=scale_up_rate_limit,
                short_lived_threshold=short_lived_threshold,
                quota_block_duration=quota_timeout,
            )
        )

        # Scale-down rate limiter, owned by the group (not health-modulated).
        self._scale_down_bucket = TokenBucket(capacity=scale_down_rate_limit, refill_period=Duration.from_minutes(1))

    def persistable_state(self) -> tuple[list[SlicePersist], GroupPersist]:
        """Snapshot this group's tracked slices and scale timestamps for persistence.

        Returns the per-slice rows plus the single group row the controller
        mirrors into the DB. In-flight scale-ups (no SliceState yet) are not
        included: they have no slice_id/handle to persist and become rows once
        ``complete_scale_up`` lands them.
        """
        with self._slices_lock:
            slices = [
                SlicePersist(
                    slice_id=slice_id,
                    scale_group=self.name,
                    lifecycle=state.lifecycle.value,
                    worker_ids=list(state.worker_ids),
                    created_at_ms=state.handle.created_at.epoch_ms(),
                    error_message=state.error_message or "",
                )
                for slice_id, state in self._slices.items()
            ]
        group = GroupPersist(
            name=self.name,
            last_scale_up_ms=self._last_scale_up.epoch_ms(),
            last_scale_down_ms=self._last_scale_down.epoch_ms(),
        )
        return slices, group

    @property
    def platform(self) -> WorkerInfraProvider:
        """Worker infrastructure provider for this scale group."""
        return self._platform

    @property
    def label_prefix(self) -> str:
        """Label prefix used for slice labels."""
        return self._label_prefix

    @property
    def config(self) -> ScaleGroupConfig:
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
    def resources(self) -> ScaleGroupResources | None:
        """Per-VM resource capacity for this scale group."""
        return self._config.resources

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
        if template is None:
            return None
        return slice_template_region(template)

    @property
    def zone(self) -> str | None:
        """Zone derived from the slice template."""
        template = self._config.slice_template
        if template is None:
            return None
        if template.gcp is not None and template.gcp.zone:
            return template.gcp.zone
        if template.coreweave is not None and template.coreweave.region:
            return template.coreweave.region
        return None

    @property
    def device_type(self) -> DeviceType:
        """Accelerator device type (TPU/GPU/CPU) for this scale group."""
        if self._config.resources is None:
            return DeviceType.CPU
        accel = self._config.resources.device_type
        if accel == AcceleratorType.GPU:
            return DeviceType.GPU
        if accel == AcceleratorType.TPU:
            return DeviceType.TPU
        return DeviceType.CPU

    @property
    def accelerator_variant(self) -> str:
        """Accelerator variant (e.g. ``v6e``); ``""`` when unset or CPU."""
        if self._config.resources is not None:
            return self._config.resources.device_variant
        return ""

    @property
    def current_demand(self) -> int:
        """Current demand level."""
        return self._current_demand

    @property
    def peak_demand(self) -> int:
        """Peak demand seen."""
        return self._peak_demand

    @property
    def detector(self) -> BackoffDetector:
        """AIMD health detector for this group."""
        return self._detector

    def health(self, timestamp: Timestamp | None = None) -> float:
        """Current AIMD health score in ``[probe_floor, 1.0]``."""
        return self._detector.health(timestamp or Timestamp.now())

    def health_label(self, timestamp: Timestamp | None = None) -> GroupHealth:
        """Display label (HEALTHY / SUSPECT / CHURNING / HOSTILE)."""
        return self._detector.health_label(timestamp or Timestamp.now())

    def _failure_count_for_status(self, now: Timestamp) -> int:
        """Approximate failure count for the proto ``consecutive_failures`` field.

        Inverts the AIMD decay: ``health = decay^n`` → ``n = log(health) / log(decay)``.
        """
        score = self._detector.health(now)
        decay = self._detector.decay
        if score >= 1.0 or decay >= 1.0 or decay <= 0:
            return 0
        return max(1, round(math.log(score) / math.log(decay)))

    def begin_scale_up(self, timestamp: Timestamp | None = None) -> None:
        """Mark that a scale-up is in progress.

        Increments the pending counter, which is included in slice_count()
        and slice_state_counts(REQUESTING) to prevent over-provisioning.
        """
        timestamp = timestamp or Timestamp.now()
        with self._slices_lock:
            self._pending_scale_ups += 1
        self._last_scale_up = timestamp

    def complete_scale_up(self, handle: SliceHandle, timestamp: Timestamp | None = None) -> None:
        """Record a successful scale-up: track the slice, register it with the detector,
        and clear any active quota block."""
        timestamp = timestamp or Timestamp.now()
        with self._slices_lock:
            self._pending_scale_ups = max(0, self._pending_scale_ups - 1)
            state = SliceState(handle=handle)
            self._slices[handle.slice_id] = state
        self._detector.record_created(handle.slice_id, handle.created_at)
        self._detector.clear_quota_block()

    def cancel_scale_up(self) -> None:
        """Record a failed scale-up: decrement the pending counter."""
        with self._slices_lock:
            self._pending_scale_ups = max(0, self._pending_scale_ups - 1)

    def mark_slice_ready(
        self,
        slice_id: str,
        worker_ids: list[str],
        worker_urls: dict[str, str] | None = None,
        timestamp: Timestamp | None = None,
    ) -> None:
        """Mark a slice READY with its worker IDs. ``quiet_since`` is left None so the next
        autoscaler tick decides idle/active afresh.

        ``worker_urls`` (worker_id -> http://host:port URL) seeds the slice
        health probe so it doesn't need to call ``handle.describe()`` on the
        first tick. Optional: tests that don't exercise the probe path can omit
        it, and the probe will lazy-fetch from the handle.
        """
        del timestamp
        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is not None:
                state.lifecycle = SliceLifecycleState.READY
                state.worker_ids = worker_ids
                state.worker_urls = dict(worker_urls or {})
                state.ping_failures = {}
                state.quiet_since = None
                state.unknown_since = None  # resolved: reset the unresolvable-timeout clock
        if state is not None:
            logger.info(
                "slice ready group=%s slice=%s n_workers=%d worker_ids=%s",
                self._config.name,
                slice_id,
                len(worker_ids),
                worker_ids,
            )

    def mark_slice_failed(self, slice_id: str, error_message: str = "") -> None:
        """Mark a slice as FAILED. Called when bootstrap fails."""
        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is not None:
                state.lifecycle = SliceLifecycleState.FAILED
                state.error_message = error_message
                registered = list(state.worker_ids)
            else:
                registered = []
        if state is not None:
            logger.warning(
                "slice failed group=%s slice=%s n_registered=%d registered=%s error=%s",
                self._config.name,
                slice_id,
                len(registered),
                registered,
                error_message,
            )

    def note_slice_unknown(self, slice_id: str, timestamp: Timestamp) -> Duration:
        """Record an UNKNOWN describe and return how long the slice has been continuously UNKNOWN.

        The first UNKNOWN observation stamps ``unknown_since``; it is cleared the
        moment the slice resolves (``mark_slice_ready``). Callers compare the
        returned duration against the unresolvable-timeout so that one transient
        UNKNOWN — common for a long-running or freshly-adopted slice — never
        terminates it, while a slice genuinely stuck UNKNOWN still fails.
        """
        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is None:
                return Duration.from_ms(0)
            if state.unknown_since is None:
                state.unknown_since = timestamp
            return Duration.from_ms(timestamp.epoch_ms() - state.unknown_since.epoch_ms())

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

    def scale_up(
        self,
        tags: dict[str, str] | None = None,
        timestamp: Timestamp | None = None,
        worker_config: WorkerConfig | None = None,
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
        chaos_raise("vm.create")
        template = self._config.slice_template if self._config.slice_template is not None else SliceConfig()
        slice_config = prepare_slice_config(
            template,
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
            slice_config.coreweave.instance_type if slice_config.coreweave is not None else "n/a",
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
            self.terminate_slice_handle(handle, context="cleaning up anyway")

    def detach_slice(self, slice_id: str, timestamp: Timestamp | None = None) -> SliceHandle | None:
        """Remove a slice from tracking and persistence without terminating it."""
        timestamp = timestamp or Timestamp.now()
        with self._slices_lock:
            state = self._slices.pop(slice_id, None)
        if state is None:
            return None
        self._last_scale_down = timestamp
        return state.handle

    def terminate_slice_handle(self, handle: SliceHandle, *, context: str) -> None:
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

    def slices_needing_describe(self) -> list[tuple[str, SliceHandle]]:
        """Snapshot slice handles the caller must ``describe()`` this tick.

        Returns not-yet-ready slices (BOOTING/INITIALIZING), plus READY slices
        the autoscaler tracks no workers for. A READY slice with empty
        ``worker_ids`` never resolved its membership (e.g. adopted from a
        checkpoint that recorded none); re-describing lets ``refresh`` repopulate
        or reap it instead of leaving it stuck DEGRADED.
        """
        with self._slices_lock:
            return [
                (slice_id, state.handle)
                for slice_id, state in self._slices.items()
                if state.lifecycle in (SliceLifecycleState.BOOTING, SliceLifecycleState.INITIALIZING)
                or (state.lifecycle == SliceLifecycleState.READY and not state.worker_ids)
            ]

    def slice_count(self) -> int:
        """Total number of slices including in-flight scale-ups."""
        with self._slices_lock:
            return len(self._slices) + self._pending_scale_ups

    def ready_slice_count(self) -> int:
        """Count of slices where all VMs are ready."""
        with self._slices_lock:
            return sum(1 for s in self._slices.values() if s.lifecycle == SliceLifecycleState.READY)

    def ready_slice_probe_targets(self) -> list[tuple[str, SliceHandle, dict[str, str]]]:
        """Snapshot READY slices for the health probe.

        Returns ``(slice_id, handle, worker_urls)`` tuples, where ``worker_urls``
        is a copy. URLs may be empty (e.g. on a controller restart for a
        checkpointed READY slice) — callers should lazy-fetch via
        ``handle.describe()`` and publish back via :meth:`set_worker_urls`.
        """
        with self._slices_lock:
            return [
                (slice_id, state.handle, dict(state.worker_urls))
                for slice_id, state in self._slices.items()
                if state.lifecycle == SliceLifecycleState.READY
            ]

    def set_worker_urls(self, slice_id: str, worker_urls: dict[str, str]) -> None:
        """Publish freshly-resolved worker URLs (typically from handle.describe()) back to state."""
        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is not None:
                state.worker_urls = dict(worker_urls)

    def record_health_probe_result(self, slice_id: str, worker_id: str, healthy: bool) -> int:
        """Update the per-worker ping_failures counter and return the new count.

        Returns 0 on success (counter reset) and the incremented count on
        failure. A counter that reaches CONSECUTIVE_FAILURE_THRESHOLD signals the
        slice should be terminated (the runtime owns that decision).
        """
        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is None:
                return 0
            # Reaching here means the slice had a probeable worker, so it is not
            # a zero-worker zombie regardless of the probe outcome.
            state.no_worker_probes = 0
            if healthy:
                state.ping_failures.pop(worker_id, None)
                return 0
            state.ping_failures[worker_id] = state.ping_failures.get(worker_id, 0) + 1
            return state.ping_failures[worker_id]

    def record_slice_no_workers(self, slice_id: str) -> int:
        """Increment and return the consecutive count of probes that found no workers.

        A READY slice whose cloud allocation reports zero workers cannot be
        /health-probed at all, so the per-worker ping counters never see it.
        Tracking the streak lets the runtime tear the slice down after
        CONSECUTIVE_FAILURE_THRESHOLD sustained observations -- mirroring the
        per-worker zombie path -- while tolerating a single transient describe().
        """
        with self._slices_lock:
            state = self._slices.get(slice_id)
            if state is None:
                return 0
            state.no_worker_probes += 1
            return state.no_worker_probes

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

    def update_slice_activity(self, worker_status_map: WorkerStatusMap, timestamp: Timestamp) -> None:
        """Stamp the active→idle / idle→active transition for each slice.

        Pure in-memory state transition: a slice flips its ``quiet_since`` to
        ``timestamp`` the first tick all of its workers are idle, and back to
        ``None`` whenever any worker has a running task. Eligibility for
        scaledown is then ``now - quiet_since >= idle_threshold``.
        """
        with self._slices_lock:
            for state in self._slices.values():
                if self._slice_has_active_workers(state, worker_status_map):
                    state.quiet_since = None
                elif state.quiet_since is None:
                    state.quiet_since = timestamp

    def _slice_has_active_workers(self, state: SliceState, worker_status_map: WorkerStatusMap) -> bool:
        """Check if any worker in a slice has running tasks."""
        for worker_id in state.worker_ids:
            status = worker_status_map.get(worker_id)
            if status is not None and not status.is_idle:
                return True
        return False

    def is_slice_eligible_for_scaledown(self, slice_id: str, timestamp: Timestamp) -> bool:
        """Check if a specific slice has been idle long enough to scale down.

        Eligible if:
        - Slice not tracked -> eligible
        - quiet_since is None (currently active or never observed) -> not eligible
        - OR idle for at least idle_threshold since quiet_since
        """
        with self._slices_lock:
            state = self._slices.get(slice_id)
        if state is None:
            return True
        if state.quiet_since is None:
            return False
        idle_duration = Duration.from_ms(timestamp.epoch_ms() - state.quiet_since.epoch_ms())
        return idle_duration >= self._idle_threshold

    def get_idle_slices(self, timestamp: Timestamp) -> list[SliceState]:
        """Get all slice states eligible for scaledown, sorted by idle time (longest first)."""
        with self._slices_lock:
            snapshot = list(self._slices.items())
        eligible: list[tuple[SliceState, int]] = []
        for slice_id, state in snapshot:
            if state.lifecycle != SliceLifecycleState.READY:
                continue
            if not self.is_slice_eligible_for_scaledown(slice_id, timestamp):
                continue
            assert state.quiet_since is not None  # implied by eligibility
            eligible.append((state, state.quiet_since.epoch_ms()))
        eligible.sort(key=lambda x: x[1])
        return [s[0] for s in eligible]

    def scale_down_if_idle(
        self,
        worker_status_map: WorkerStatusMap,
        target_capacity: int,
        timestamp: Timestamp,
    ) -> list[SliceHandle]:
        """Terminate idle slices over ``target_capacity``, rate-limited by the scale-down bucket.

        Args:
            worker_status_map: Map of worker_id to worker status.
            target_capacity: Target number of slices (typically
                ``min(demand + buffer_slices, max_slices)``).
            timestamp: Current timestamp for idle calculation.

        Returns:
            List of terminated slice handles (may be empty).
        """
        # Update activity tracking
        self.update_slice_activity(worker_status_map, timestamp)

        # Use ready + pending for capacity check to prevent churn during boot
        counts = self.slice_state_counts()
        ready = counts[SliceLifecycleState.READY]
        pending = counts[SliceLifecycleState.BOOTING] + counts[SliceLifecycleState.INITIALIZING]

        # Don't scale down if total capacity (ready + pending) is at or below target
        if ready + pending <= target_capacity:
            return []

        # Don't scale down ready slices if we're still waiting for pending
        if ready <= target_capacity:
            return []

        terminated: list[SliceHandle] = []

        # Find idle slices and verify they're still idle before termination
        idle_slices = self.get_idle_slices(timestamp)
        for slice_state in idle_slices:
            # Stop once we've scaled down to the target
            if ready - len(terminated) <= target_capacity:
                break

            # Verify idle before acquiring a rate-limit token so that
            # failed verifications don't waste tokens.
            if not self._verify_slice_idle(slice_state, worker_status_map):
                continue

            if not self.acquire_scale_down_token(timestamp):
                if terminated:
                    logger.info(
                        "Scale group %s: scale down rate-limited after %d terminations",
                        self.name,
                        len(terminated),
                    )
                break

            with self._slices_lock:
                state = self._slices.get(slice_state.handle.slice_id)
            quiet_since = state.quiet_since if state is not None else None
            idle_ms = timestamp.epoch_ms() - quiet_since.epoch_ms() if quiet_since is not None else 0
            logger.info(
                "Scale group %s: scaling down slice %s " "(idle for %dms, ready=%d/%d, pending=%d, target=%d)",
                self.name,
                slice_state.handle.slice_id,
                idle_ms,
                ready,
                self.num_vms,
                pending,
                target_capacity,
            )
            self.scale_down(slice_state.handle.slice_id, timestamp)
            terminated.append(slice_state.handle)

        return terminated

    def _verify_slice_idle(self, state: SliceState, worker_status_map: WorkerStatusMap) -> bool:
        """Verify every known worker in a slice is an idle spare before termination.

        Gates on ``is_idle_spare`` (idle AND schedulable), not bare ``is_idle``: a
        slice holding a ``DEGRADED`` worker is reported quiet (it runs no tasks) but
        is NOT reclaimable spare — the scheduler cannot place onto it, so reclaiming
        it as free capacity is exactly the autoscaler/scheduler disagreement we are
        fixing. Such a slice is retained here and torn down by the health threshold
        path instead.

        Requires at least one known worker. If no workers are known at all (none in
        worker_status_map), returns False -- the slice may still be booting. Zombie
        slices whose workers disappeared are reaped elsewhere: a dead worker process
        trips the heartbeat timeout (if a worker row exists) or the per-worker /health
        probe, and a slice whose backing allocation vanished entirely (no worker rows,
        nothing to probe) trips the no-worker counter in Autoscaler.probe_health.
        """
        has_known_worker = False
        for worker_id in state.worker_ids:
            status = worker_status_map.get(worker_id)
            if status is None:
                continue
            has_known_worker = True
            if not status.is_idle_spare:
                return False
        return has_known_worker

    def can_scale_up(self, timestamp: Timestamp | None = None) -> bool:
        """False if the detector blocks scale-up (quota) or the group is at ``max_slices``."""
        timestamp = timestamp or Timestamp.now()
        if not self._detector.can_scale_up(timestamp):
            return False
        with self._slices_lock:
            count = len(self._slices) + self._pending_scale_ups
        if count >= self._config.max_slices:
            return False
        return True

    def try_acquire_scale_up(self, timestamp: Timestamp | None = None) -> bool:
        """Acquire a scale-up token from the detector's health-modulated bucket."""
        return self._detector.try_acquire_scale_up(timestamp or Timestamp.now())

    def acquire_scale_down_token(self, timestamp: Timestamp | None = None) -> bool:
        """Try to acquire a scale-down rate limit token. Returns False if rate-limited."""
        return self._scale_down_bucket.try_acquire(now=timestamp)

    def record_quota_exceeded(self, reason: str, timestamp: Timestamp | None = None) -> None:
        """Record a quota exhaustion event, blocking scale-up via the detector."""
        self._detector.record_quota_exceeded(reason, timestamp or Timestamp.now())

    def record_slice_preempted(self, slice_id: str, timestamp: Timestamp | None = None) -> None:
        """Record a preemption of a previously-created slice."""
        self._detector.record_terminated(slice_id, SliceFate.PREEMPTED, timestamp or Timestamp.now())

    def record_slice_boot_failed(self, slice_id: str, timestamp: Timestamp | None = None) -> None:
        """Record that a slice failed to reach READY (boot failure or unresolvable timeout)."""
        self._detector.record_terminated(slice_id, SliceFate.BOOT_FAILED, timestamp or Timestamp.now())

    def record_create_failed(self, timestamp: Timestamp | None = None) -> None:
        """Record a CreateSlice RPC failure where no slice handle was returned."""
        self._detector.record_create_failed(timestamp or Timestamp.now())

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

        group_type = self.device_type
        if group_type != device_type:
            return False

        if device_variants is None:
            return True
        group_variant = self._config.resources.device_variant if self._config.resources is not None else ""
        return group_variant.lower() in {v.lower() for v in device_variants}

    def to_attributes(self) -> dict[str, AttributeValue]:
        """Express this group's routing properties as worker-style attributes.

        Enables the same evaluate_constraint + ConstraintIndex infrastructure
        used for worker matching to also work for scaling group routing.
        """
        attrs: dict[str, AttributeValue] = {}
        attrs[WellKnownAttribute.DEVICE_TYPE] = AttributeValue(self.device_type.value)
        if self._config.resources is not None and self._config.resources.device_variant:
            attrs[WellKnownAttribute.DEVICE_VARIANT] = AttributeValue(self._config.resources.device_variant.lower())
        if self._config.resources is not None:
            is_preemptible = self._config.resources.capacity_type == CapacityType.PREEMPTIBLE
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

    def availability(self, timestamp: Timestamp | None = None) -> AvailabilityState:
        """Compute current availability state for waterfall routing.

        Priority: QUOTA_EXCEEDED > BACKOFF (HOSTILE churn) > REQUESTING > AT_MAX_SLICES > AVAILABLE
        """
        timestamp = timestamp or Timestamp.now()

        quota_until = self._detector.quota_deadline(timestamp)
        if quota_until is not None:
            return AvailabilityState(
                GroupAvailability.QUOTA_EXCEEDED,
                self._detector.block_reason(timestamp) or "",
                quota_until,
            )

        # HOSTILE → BACKOFF so waterfall routing prefers a healthier group.
        if self._detector.health_label(timestamp) == GroupHealth.HOSTILE:
            score = self._detector.health(timestamp)
            return AvailabilityState(
                GroupAvailability.BACKOFF,
                f"degraded (health={score:.2f})",
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

    def find_slice_for_worker(self, worker_id: str) -> str | None:
        """Find slice_id containing a worker with the given ID."""
        with self._slices_lock:
            snapshot = list(self._slices.items())
        for slice_id, state in snapshot:
            if worker_id in state.worker_ids:
                return slice_id
        return None

    def get_slice_worker_ids(self, slice_id: str) -> list[str]:
        """Get all worker IDs for a slice. Returns empty list if not found."""
        with self._slices_lock:
            state = self._slices.get(slice_id)
        if state is None:
            return []
        return list(state.worker_ids)

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

    def restore_from_snapshot(
        self,
        slices: dict[str, SliceState],
        last_scale_up: Timestamp,
        last_scale_down: Timestamp,
    ) -> None:
        """Restore slices and scale timestamps from a snapshot.

        The in-memory detector starts fresh on each controller boot; restored
        slices are registered with their original ``created_at`` so age-based
        termination classification stays accurate.
        """
        with self._slices_lock:
            self._slices = slices
        self._last_scale_up = last_scale_up
        self._last_scale_down = last_scale_down
        for slice_id, state in slices.items():
            self._detector.record_created(slice_id, state.handle.created_at)

    def to_status(self) -> vm_pb2.ScaleGroupStatus:
        """Build a ScaleGroupStatus proto for the status API."""
        with self._slices_lock:
            snapshot = list(self._slices.values())
        now = Timestamp.now()
        availability = self.availability(now)
        blocked_until = availability.until if availability.until is not None else Timestamp.from_ms(0)
        counts = self.slice_state_counts()

        resources = self._config.resources
        status = vm_pb2.ScaleGroupStatus(
            name=self.name,
            device_type=accelerator_type_to_string(resources.device_type) if resources is not None else "",
            device_variant=resources.device_variant if resources is not None else "",
            quota_pool=self._config.quota_pool,
            allocation_tier=self._config.allocation_tier,
            region=self.region or "",
            current_demand=self._current_demand,
            peak_demand=self._peak_demand,
            backoff_until=timestamp_to_proto(Timestamp.from_ms(0)),
            consecutive_failures=self._failure_count_for_status(now),
            last_scale_up=timestamp_to_proto(self._last_scale_up),
            last_scale_down=timestamp_to_proto(self._last_scale_down),
            availability_status=availability.status.value,
            availability_reason=availability.reason,
            blocked_until=timestamp_to_proto(blocked_until),
            scale_up_cooldown_until=timestamp_to_proto(Timestamp.from_ms(0)),
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
    error_message: str = ""


@dataclass
class GroupSnapshot:
    """Lightweight record of a persisted scaling group, read from the DB."""

    name: str
    slices: list[SliceSnapshot] = field(default_factory=list)
    last_scale_up_ms: int = 0
    last_scale_down_ms: int = 0


@dataclass
class ScalingGroupRestoreResult:
    """Result of restoring a single scaling group from checkpoint metadata."""

    slices: dict[str, SliceState] = field(default_factory=dict)
    discarded_slice_ids: list[str] = field(default_factory=list)
    adopted_count: int = 0
    last_scale_up: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    last_scale_down: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))


def restore_scaling_group(
    group_snapshot: GroupSnapshot,
    cloud_handles: list[SliceHandle],
) -> ScalingGroupRestoreResult:
    """Reconcile checkpointed group slices against pre-fetched cloud handles."""
    cloud_by_id: dict[str, SliceHandle] = {h.slice_id: h for h in cloud_handles}
    checkpoint_slices = {s.slice_id: s for s in group_snapshot.slices}

    result = ScalingGroupRestoreResult()

    for slice_id, slice_snap in checkpoint_slices.items():
        cloud_handle = cloud_by_id.get(slice_id)
        if cloud_handle is None:
            logger.info("Scaling group %s: discarding slice %s (missing from cloud)", group_snapshot.name, slice_id)
            result.discarded_slice_ids.append(slice_id)
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

        # quiet_since defaults to None: the next tick decides idle/active
        # from live worker state, granting a full idle_threshold grace
        # period after restart.
        result.slices[slice_id] = SliceState(
            handle=cloud_handle,
            lifecycle=lifecycle,
            worker_ids=list(slice_snap.worker_ids),
            error_message=slice_snap.error_message,
        )

    for slice_id, cloud_handle in cloud_by_id.items():
        if slice_id in checkpoint_slices:
            continue
        logger.info("Scaling group %s: adopting unknown cloud slice %s as BOOTING", group_snapshot.name, slice_id)
        result.slices[slice_id] = SliceState(handle=cloud_handle, lifecycle=SliceLifecycleState.BOOTING)
        result.adopted_count += 1

    if group_snapshot.last_scale_up_ms > 0:
        result.last_scale_up = Timestamp.from_ms(group_snapshot.last_scale_up_ms)
    if group_snapshot.last_scale_down_ms > 0:
        result.last_scale_down = Timestamp.from_ms(group_snapshot.last_scale_down_ms)

    logger.info(
        "Restored scaling group %s: %d slices (%d discarded, %d adopted)",
        group_snapshot.name,
        len(result.slices),
        len(result.discarded_slice_ids),
        result.adopted_count,
    )
    return result
