# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AIMD health tracker for the per-group scale-up token bucket.

A per-group ``health: float`` in ``[probe_floor, 1.0]`` drives the bucket's
refill rate.

- Each failure event applies ``health *= decay_per_failure`` (multiplicative
  decrease, clamped at ``probe_floor``).
- Between events, ``health`` accrues additively so it climbs from
  ``probe_floor`` to ``1.0`` over ``recovery_duration``.
- ``try_acquire_scale_up`` sets ``bucket.refill_rate = base_rate * health``
  and then takes a token, and each failure also caps the bucket inventory
  to ``capacity * health`` so an accumulated burst can't bypass the new
  rate.

The probe floor keeps a degraded zone probing at a non-zero rate so it can
produce non-failure samples and recover.

Quota-exhaustion is a separate categorical block (``record_quota_exceeded``)
that does not feed the health score.

In-flight ``created_at`` is tracked so a terminated slice's age determines
whether the termination counts as a failure. State is in-memory; on
controller restart ``health`` resets to ``1.0``.
"""

from __future__ import annotations

import threading
from enum import StrEnum

from rigging.timing import Deadline, Duration, Timestamp, TokenBucket

# Time to recover from probe_floor back to 1.0 under no failures.
DEFAULT_RECOVERY_DURATION = Duration.from_hours(1)

# Multiplicative decay applied per failure. 0.7 → health drops to 70% after
# one failure, 34% after three, 17% after five.
DEFAULT_DECAY_PER_FAILURE = 0.7

# Minimum effective scale-up rate in attempts/min, applied even when health
# bottoms out so the group keeps producing non-failure samples.
DEFAULT_PROBE_FLOOR_PER_MINUTE = 0.5

# Slices younger than this at termination count as failures. Sized so that a
# slice has time to boot, advance, and checkpoint before it is treated as a
# normal preemption: GCP routinely returns spot capacity that lives 10-25 min
# before reclaim (see issue #5976), and a 5-min threshold mostly catches only
# boot-time failures and lets the longer churn pass through unnoticed.
DEFAULT_SHORT_LIVED_THRESHOLD = Duration.from_minutes(30)

# How long a quota-exhausted error blocks scale-up before we attempt again.
DEFAULT_QUOTA_BLOCK_DURATION = Duration.from_minutes(5)

# Display-only label thresholds on the health score.
LABEL_HEALTHY = 0.8
LABEL_SUSPECT = 0.5
LABEL_CHURNING = 0.2


class SliceFate(StrEnum):
    """How a recorded slice's life ended."""

    PREEMPTED = "preempted"
    BOOT_FAILED = "boot_failed"


class GroupHealth(StrEnum):
    """Display label derived from the current health score. Display only — the
    detector throttles on the raw float, not on these bands."""

    HEALTHY = "healthy"
    SUSPECT = "suspect"
    CHURNING = "churning"
    HOSTILE = "hostile"


class BackoffDetector:
    """Per-group AIMD health tracker. Thread-safe."""

    def __init__(
        self,
        group_name: str,
        scale_up_bucket: TokenBucket,
        base_scale_up_per_minute: float,
        recovery_duration: Duration = DEFAULT_RECOVERY_DURATION,
        decay_per_failure: float = DEFAULT_DECAY_PER_FAILURE,
        probe_floor_per_minute: float = DEFAULT_PROBE_FLOOR_PER_MINUTE,
        short_lived_threshold: Duration = DEFAULT_SHORT_LIVED_THRESHOLD,
        quota_block_duration: Duration = DEFAULT_QUOTA_BLOCK_DURATION,
    ):
        if base_scale_up_per_minute <= 0:
            raise ValueError(f"base_scale_up_per_minute must be > 0, got {base_scale_up_per_minute}")
        if not 0 < decay_per_failure < 1:
            raise ValueError(f"decay_per_failure must be in (0, 1), got {decay_per_failure}")
        if probe_floor_per_minute <= 0:
            raise ValueError(f"probe_floor_per_minute must be > 0, got {probe_floor_per_minute}")
        if probe_floor_per_minute >= base_scale_up_per_minute:
            raise ValueError(
                f"probe_floor_per_minute ({probe_floor_per_minute}) must be < "
                f"base_scale_up_per_minute ({base_scale_up_per_minute})"
            )

        self._group_name = group_name
        self._scale_up_bucket = scale_up_bucket
        self._base_refill_per_second = base_scale_up_per_minute / 60.0
        self._floor = probe_floor_per_minute / base_scale_up_per_minute
        self._decay = decay_per_failure
        self._recovery_per_second = (1.0 - self._floor) / recovery_duration.to_seconds()
        self._short_lived_threshold = short_lived_threshold
        self._quota_block_duration = quota_block_duration

        self._health: float = 1.0
        self._last_tick: Timestamp | None = None

        # created_at by slice_id; used only to classify a termination by age.
        self._inflight: dict[str, Timestamp] = {}

        # Categorical quota block, separate from the AIMD state.
        self._quota_until: Deadline | None = None
        self._quota_reason: str = ""

        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Event hooks
    # -----------------------------------------------------------------------

    def record_created(self, slice_id: str, created_at: Timestamp) -> None:
        """Record a successful slice create. Idempotent."""
        with self._lock:
            self._inflight.setdefault(slice_id, created_at)

    def record_terminated(self, slice_id: str, fate: SliceFate, terminated_at: Timestamp) -> None:
        """Record the termination of a previously-created slice.

        Decays the score on ``BOOT_FAILED`` at any age, or on ``PREEMPTED``
        if the slice was younger than ``short_lived_threshold``.
        """
        with self._lock:
            created_at = self._inflight.pop(slice_id, terminated_at)
            age_ms = terminated_at.epoch_ms() - created_at.epoch_ms()
            self._apply_recovery(terminated_at)
            if fate == SliceFate.BOOT_FAILED:
                self._apply_decay(terminated_at)
            elif fate == SliceFate.PREEMPTED and age_ms < self._short_lived_threshold.to_ms():
                self._apply_decay(terminated_at)

    def record_create_failed(self, ts: Timestamp) -> None:
        """Record a CreateSlice RPC error. Always decays the score."""
        with self._lock:
            self._apply_recovery(ts)
            self._apply_decay(ts)

    def record_quota_exceeded(self, reason: str, ts: Timestamp) -> None:
        """Set a fixed-duration quota block. Does not feed the AIMD score."""
        with self._lock:
            self._quota_until = Deadline.after(ts, self._quota_block_duration)
            self._quota_reason = reason

    def clear_quota_block(self) -> None:
        """Clear the quota-block deadline."""
        with self._lock:
            self._quota_until = None
            self._quota_reason = ""

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def health(self, now: Timestamp) -> float:
        """Health score in ``[probe_floor, 1.0]`` after applying recovery up to ``now``."""
        with self._lock:
            self._apply_recovery(now)
            return self._health

    def health_label(self, now: Timestamp) -> GroupHealth:
        """Display label derived from the current health score."""
        h = self.health(now)
        if h >= LABEL_HEALTHY:
            return GroupHealth.HEALTHY
        if h >= LABEL_SUSPECT:
            return GroupHealth.SUSPECT
        if h >= LABEL_CHURNING:
            return GroupHealth.CHURNING
        return GroupHealth.HOSTILE

    def can_scale_up(self, now: Timestamp) -> bool:
        """False only when inside the quota-block window."""
        deadline, _ = self._snapshot_quota()
        if deadline is not None and not deadline.expired(now=now):
            return False
        return True

    def try_acquire_scale_up(self, now: Timestamp) -> bool:
        """Apply recovery, set ``bucket.refill_rate = base * health``, then take a token."""
        if not self.can_scale_up(now):
            return False
        with self._lock:
            self._apply_recovery(now)
            self._scale_up_bucket.refill_rate = self._base_refill_per_second * self._health
        return self._scale_up_bucket.try_acquire(now=now)

    def block_reason(self, now: Timestamp) -> str | None:
        """Reason scale-up is hard-blocked (quota only), or ``None``."""
        deadline, reason = self._snapshot_quota()
        if deadline is not None and not deadline.expired(now=now):
            return f"quota exceeded: {reason}" if reason else "quota exceeded"
        return None

    def quota_deadline(self, now: Timestamp) -> Timestamp | None:
        deadline, _ = self._snapshot_quota()
        if deadline is None or deadline.expired(now=now):
            return None
        return deadline.as_timestamp()

    def status_label(self, now: Timestamp) -> str:
        """``"<label> health=<score>"`` (e.g. ``"churning health=0.34"``) for dashboards/logs."""
        h = self.health(now)
        return f"{self.health_label(now).value} health={h:.2f}"

    # -----------------------------------------------------------------------
    # Internals (caller holds _lock)
    # -----------------------------------------------------------------------

    def _apply_recovery(self, now: Timestamp) -> None:
        """Accrue recovery from the last tick to ``now``, clamped at 1.0."""
        if self._last_tick is None:
            self._last_tick = now
            return
        elapsed_ms = now.epoch_ms() - self._last_tick.epoch_ms()
        if elapsed_ms <= 0:
            return
        self._health = min(1.0, self._health + self._recovery_per_second * (elapsed_ms / 1000.0))
        self._last_tick = now

    def _apply_decay(self, now: Timestamp) -> None:
        """Multiply health by ``decay`` (clamped at the floor) and cap bucket inventory at ``capacity * health``."""
        self._health = max(self._floor, self._health * self._decay)
        self._scale_up_bucket.cap_tokens(self._scale_up_bucket.capacity * self._health, now=now)

    def _snapshot_quota(self) -> tuple[Deadline | None, str]:
        with self._lock:
            return self._quota_until, self._quota_reason
