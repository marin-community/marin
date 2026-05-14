# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AIMD health tracker for the per-group scale-up token bucket.

A single per-group ``health: float`` in ``[probe_floor, 1.0]`` drives the
bucket's refill rate. State:

- Starts at ``1.0`` (fully healthy).
- On every failure event: ``health *= decay_per_failure`` (multiplicative
  decrease, clamped at ``probe_floor``).
- On every query/event: ``health += recovery_per_second * dt`` (additive
  increase, clamped at ``1.0``). ``recovery_per_second`` is chosen so the
  score climbs from ``probe_floor`` to ``1.0`` over ``recovery_duration``.

This is AIMD — the same shape TCP congestion control uses. One isolated
failure throttles the group for a short time; sustained failures push it
to the floor; sustained success lets it climb back smoothly.

The bucket's ``refill_rate`` is set to ``base_rate * health`` on every
``try_acquire_scale_up`` call. Cadence — not batch size — is what gets
throttled. Each failure also caps the bucket's current token inventory
to ``capacity * health``, so a previously-healthy bucket that suddenly
degrades can't burst its accumulated tokens before the lowered refill
rate takes effect.

The probe floor exists so a fully degraded zone still attempts at some low
rate — without it, zero attempts means zero samples and the score could
never climb back.

Quota-exhaustion is a separate hard block (``record_quota_exceeded``).
Quota is categorical ("GCP said no") and recovers on GCP's timescale, not
ours, so it bypasses the health-score machinery entirely.

In-flight ``created_at`` is tracked so a terminated slice can be classified
short-lived vs. long-lived by age. Only short-lived deaths (or BOOT_FAILED
at any age) feed the decay; long-lived preemptions are normal preemptible
behaviour and do nothing.

State is in-memory; on controller restart ``health`` resets to ``1.0``.
"""

from __future__ import annotations

import threading
from enum import StrEnum

from rigging.timing import Deadline, Duration, Timestamp, TokenBucket

# Time to recover from probe_floor back to 1.0 under no failures.
DEFAULT_RECOVERY_DURATION = Duration.from_hours(1)

# Multiplicative decay applied per failure. 0.7 → 1 failure leaves us at 70%,
# 3 failures at 34%, 5 failures at 17%. Aggressive but not crippling.
DEFAULT_DECAY_PER_FAILURE = 0.7

# Floor on the effective scale-up rate, in attempts per minute. Even a fully
# degraded zone keeps probing at this rate so it can produce non-failure
# outcomes and let the score climb back. At base 16/min, 0.5/min = 3% floor.
DEFAULT_PROBE_FLOOR_PER_MINUTE = 0.5

# Slices younger than this at termination count as failures; older ones are
# normal preemptible behaviour.
DEFAULT_SHORT_LIVED_THRESHOLD = Duration.from_minutes(5)

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
    """Display label derived from the current health score.

    Informational only — the detector throttles on the raw float, not on
    these bands.
    """

    HEALTHY = "healthy"
    SUSPECT = "suspect"
    CHURNING = "churning"
    HOSTILE = "hostile"


class BackoffDetector:
    """Per-group AIMD health tracker.

    Thread-safe. Failure events decay the score multiplicatively; recovery
    accrues additively as wall time advances. ``try_acquire_scale_up``
    folds the recovery, updates the bucket's refill rate from the current
    score, and takes a token.
    """

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

        # In-flight slice tracking for age classification at termination time.
        # Does not contribute to the score; just needed so a long-lived slice's
        # eventual preemption can be recognised as not-a-failure.
        self._inflight: dict[str, Timestamp] = {}

        # Quota gate — separate from the AIMD state.
        self._quota_until: Deadline | None = None
        self._quota_reason: str = ""

        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Event hooks
    # -----------------------------------------------------------------------

    def record_created(self, slice_id: str, created_at: Timestamp) -> None:
        """Record a successful slice create. Tracks ``created_at`` for later
        age-at-death classification. Idempotent (keeps original timestamp)."""
        with self._lock:
            self._inflight.setdefault(slice_id, created_at)

    def record_terminated(self, slice_id: str, fate: SliceFate, terminated_at: Timestamp) -> None:
        """Record the end of a previously-created slice.

        Counts as a failure (decays the score) when:
        - ``fate == BOOT_FAILED`` (always — slice never produced useful work).
        - ``fate == PREEMPTED`` and age < ``short_lived_threshold``.

        Long-lived preemptions are silent — the slice did useful work first.
        If the slice was never seen by :meth:`record_created` (e.g. controller
        restart), ``created_at`` defaults to ``terminated_at`` → age 0 →
        short-lived → one decay applied.
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
        """Record a CreateSlice RPC error (no slice handle was returned).

        Always a failure event — equivalent to a zero-age BOOT_FAILED.
        """
        with self._lock:
            self._apply_recovery(ts)
            self._apply_decay(ts)

    def record_quota_exceeded(self, reason: str, ts: Timestamp) -> None:
        """Record a quota error from GCP. Sets a fixed-duration hard block.

        Quota is categorical, not probabilistic — it does not feed the AIMD
        score, just sets ``_quota_until``.
        """
        with self._lock:
            self._quota_until = Deadline.after(ts, self._quota_block_duration)
            self._quota_reason = reason

    def clear_quota_block(self) -> None:
        """Clear the quota-block deadline (called on a proven successful create)."""
        with self._lock:
            self._quota_until = None
            self._quota_reason = ""

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def health(self, now: Timestamp) -> float:
        """Current health score in ``[probe_floor, 1.0]`` after recovery to now."""
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
        """False only when inside the quota-block window.

        The health score throttles via the bucket's refill rate; it never
        produces a hard block. Quota is the only categorical gate.
        """
        deadline, _ = self._snapshot_quota()
        if deadline is not None and not deadline.expired(now=now):
            return False
        return True

    def try_acquire_scale_up(self, now: Timestamp) -> bool:
        """Apply recovery, set the bucket refill rate to ``base * health``, then acquire.

        Returns False on quota block or empty bucket.
        """
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
        """Human-readable status string for dashboards/logs.

        Format: ``"<label> health=<score>"``, e.g. ``"churning health=0.34"``.
        """
        h = self.health(now)
        return f"{self.health_label(now).value} health={h:.2f}"

    # -----------------------------------------------------------------------
    # Internals (caller holds _lock)
    # -----------------------------------------------------------------------

    def _apply_recovery(self, now: Timestamp) -> None:
        """Accrue recovery from the last tick to ``now``."""
        if self._last_tick is None:
            self._last_tick = now
            return
        elapsed_ms = now.epoch_ms() - self._last_tick.epoch_ms()
        if elapsed_ms <= 0:
            return
        self._health = min(1.0, self._health + self._recovery_per_second * (elapsed_ms / 1000.0))
        self._last_tick = now

    def _apply_decay(self, now: Timestamp) -> None:
        """Apply one multiplicative decay and drain bucket inventory to match.

        Health drops to ``health * decay`` (clamped at the probe floor). The
        bucket's accumulated tokens are then capped at ``capacity * health``,
        so a previously-healthy group that suddenly degrades can't burst
        through a full bucket before the new (lower) refill rate matters.
        """
        self._health = max(self._floor, self._health * self._decay)
        self._scale_up_bucket.cap_tokens(self._scale_up_bucket.capacity * self._health, now=now)

    def _snapshot_quota(self) -> tuple[Deadline | None, str]:
        with self._lock:
            return self._quota_until, self._quota_reason
