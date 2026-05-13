# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory churn detector for autoscaler scale groups.

Replaces the older exponential-backoff-on-failure design. The old design
treated every short-lived slice death as a fault that earned the group an
escalating timeout, which produced a sawtooth: capacity drained while the
group sat in backoff, then burst on recovery. A preemption isn't a fault
the autoscaler should be punished for — GCP took the slice and there's
nothing we did wrong. But repeated short-lived deaths *do* signal that
sending more user work into the zone will keep getting it preempted, so
we still want to stop scaling up there.

This detector tracks the rate of recent short-lived slice deaths over a
sliding window and exposes a graduated health signal:

    HEALTHY  ( <20% churn)     full scale-up rate
    SUSPECT  (20-50% churn)    half scale-up rate
    CHURNING (50-80% churn)    one-tenth scale-up rate, block scale-down
    HOSTILE  (>=80% churn)     halt scale-up entirely, block scale-down

A slice that survives past ``long_lived_threshold`` counts as a positive
sample, so a healthy steady-state group whose slices are all alive reads
as LONG_LIVED across the window and produces ``churn_rate = 0`` -- one
unlucky preemption against many healthy slices does not move the needle.

Quota-exhaustion gating lives inside this class (``_quota_until``,
``record_quota_exceeded``, ``clear_quota_block``). Quota is a categorical
"GCP said no", not churn, so it does not contribute to the churn rate
and has its own fixed-duration block.

The detector is in-memory only: on controller restart, the window starts
empty and is repopulated as slices live and die. A short period of "no
data" after restart is fine — the autoscaler defaults to HEALTHY when
there are fewer than ``min_samples`` classified outcomes, so it behaves
exactly like a freshly-deployed cluster until enough samples accumulate.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from enum import StrEnum

from rigging.timing import Deadline, Duration, Timestamp

DEFAULT_WINDOW = Duration.from_minutes(30)
DEFAULT_SHORT_LIVED_THRESHOLD = Duration.from_minutes(5)
DEFAULT_LONG_LIVED_THRESHOLD = Duration.from_minutes(10)
DEFAULT_MIN_SAMPLES = 3
DEFAULT_QUOTA_BLOCK_DURATION = Duration.from_minutes(5)

# Health thresholds on churn_rate (fraction of recent classified outcomes
# that were short-lived deaths).
SUSPECT_CHURN_RATE = 0.2
CHURNING_CHURN_RATE = 0.5
HOSTILE_CHURN_RATE = 0.8


class SliceFate(StrEnum):
    """How a slice's life ended (or was classified)."""

    # The slice was created, reached READY, then was terminated by GCP
    # (preemption) or otherwise died after running. Whether it counts as
    # a failure depends on age at death — handled internally by the
    # detector's classifier.
    PREEMPTED = "preempted"

    # The slice never reached READY (CreateSlice RPC error, bootstrap
    # FAILED, or UNKNOWN past timeout). Always counts as a failure
    # regardless of age, since no useful work could have happened.
    BOOT_FAILED = "boot_failed"

    # Synthetic — assigned at classification time to outcomes that have
    # demonstrably survived past ``long_lived_threshold``. Never passed
    # in to ``record_terminated``.
    LONG_LIVED = "long_lived"


class GroupHealth(StrEnum):
    """Graduated health response derived from churn rate."""

    HEALTHY = "healthy"
    SUSPECT = "suspect"
    CHURNING = "churning"
    HOSTILE = "hostile"


_RATE_MULTIPLIERS: dict[GroupHealth, float] = {
    GroupHealth.HEALTHY: 1.0,
    GroupHealth.SUSPECT: 0.5,
    GroupHealth.CHURNING: 0.1,
    GroupHealth.HOSTILE: 0.0,
}


@dataclass
class SliceOutcome:
    """One observed slice lifecycle, used as a sample in the churn window."""

    slice_id: str
    created_at: Timestamp
    fate: SliceFate | None = None
    fate_at: Timestamp | None = None


class BackoffDetector:
    """Per-group sliding-window churn tracker.

    Thread-safe. All event hooks (``record_created``, ``record_terminated``,
    ``record_create_failed``) and all queries (``churn_rate``, ``health``,
    ``can_scale_up``, etc.) are safe to call from any thread.

    Failure modes treated as "bad" outcomes for churn-rate purposes:

    - ``BOOT_FAILED`` at any age (the slice never produced useful work).
    - ``PREEMPTED`` with ``fate_at - created_at < short_lived_threshold``
      (the slice was up too briefly to be worth feeding more work to).

    Failure modes treated as "good" outcomes:

    - ``PREEMPTED`` with age >= ``short_lived_threshold`` (the slice ran
      long enough to be useful; preemption was just bad luck).
    - In-flight outcomes whose age >= ``long_lived_threshold`` (slice is
      demonstrably surviving, even though we haven't seen termination).

    In-flight outcomes younger than ``long_lived_threshold`` are
    "pending" — they don't contribute to the rate either way.
    """

    def __init__(
        self,
        group_name: str,
        window: Duration = DEFAULT_WINDOW,
        short_lived_threshold: Duration = DEFAULT_SHORT_LIVED_THRESHOLD,
        long_lived_threshold: Duration = DEFAULT_LONG_LIVED_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        quota_block_duration: Duration = DEFAULT_QUOTA_BLOCK_DURATION,
    ):
        if long_lived_threshold.to_ms() < short_lived_threshold.to_ms():
            raise ValueError(
                f"long_lived_threshold ({long_lived_threshold}) must be >= "
                f"short_lived_threshold ({short_lived_threshold})"
            )
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")

        self._group_name = group_name
        self._window = window
        self._short_lived_threshold = short_lived_threshold
        self._long_lived_threshold = long_lived_threshold
        self._min_samples = min_samples
        self._quota_block_duration = quota_block_duration
        self._outcomes: dict[str, SliceOutcome] = {}
        self._synthetic_seq = itertools.count()
        # Quota-exhausted gate: a GCP "no quota" signal blocks scale-up for a
        # fixed window. Quota events do not contribute to the churn rate —
        # quota is a categorical capacity-allocation failure, not a sign that
        # work we send into the zone will be reclaimed.
        self._quota_until: Deadline | None = None
        self._quota_reason: str = ""
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Event hooks
    # -----------------------------------------------------------------------

    def record_created(self, slice_id: str, created_at: Timestamp) -> None:
        """Record that a slice was successfully created.

        Called when the platform's ``CreateSlice`` returns a usable handle.
        Pairs with a subsequent ``record_terminated`` for the same slice.
        Safe to call multiple times for the same slice (no-op after first).
        """
        with self._lock:
            if slice_id in self._outcomes:
                return
            self._outcomes[slice_id] = SliceOutcome(slice_id=slice_id, created_at=created_at)
            self._gc(created_at)

    def record_terminated(self, slice_id: str, fate: SliceFate, terminated_at: Timestamp) -> None:
        """Record that a previously-created slice has ended.

        ``fate`` must be ``PREEMPTED`` or ``BOOT_FAILED``; ``LONG_LIVED``
        is synthetic and only emitted by the classifier.

        If the slice was never seen by ``record_created`` (controller
        restart between create and terminate, etc.), a synthetic outcome
        is created with ``created_at = terminated_at`` so the event still
        contributes to the churn rate as a worst-case sample.
        """
        if fate == SliceFate.LONG_LIVED:
            raise ValueError("LONG_LIVED is synthetic and cannot be recorded directly")
        with self._lock:
            outcome = self._outcomes.get(slice_id)
            if outcome is None:
                outcome = SliceOutcome(slice_id=slice_id, created_at=terminated_at)
                self._outcomes[slice_id] = outcome
            outcome.fate = fate
            outcome.fate_at = terminated_at
            self._gc(terminated_at)

    def record_create_failed(self, ts: Timestamp) -> None:
        """Record a CreateSlice RPC failure (no slice handle was returned).

        Equivalent to a BOOT_FAILED termination of a zero-age slice, but
        keyed by a synthetic id so it doesn't collide with real slice_ids.
        Counts toward churn like any other bad outcome.
        """
        with self._lock:
            synthetic_id = f"__create_failed__{ts.epoch_ms()}_{next(self._synthetic_seq)}"
            self._outcomes[synthetic_id] = SliceOutcome(
                slice_id=synthetic_id,
                created_at=ts,
                fate=SliceFate.BOOT_FAILED,
                fate_at=ts,
            )
            self._gc(ts)

    def record_quota_exceeded(self, reason: str, ts: Timestamp) -> None:
        """Record that GCP rejected a scale-up with a quota error.

        Sets a fixed-duration block on scale-up. Does **not** contribute to
        churn: quota is a categorical allocation failure (we don't have
        capacity rights), not a sign that work we send into the zone gets
        reclaimed mid-flight.
        """
        with self._lock:
            self._quota_until = Deadline.after(ts, self._quota_block_duration)
            self._quota_reason = reason

    def clear_quota_block(self) -> None:
        """Clear the quota-block deadline.

        Called when a successful scale-up proves GCP is currently allowing
        creates in this zone, so the prior "no quota" snapshot is stale.
        """
        with self._lock:
            self._quota_until = None
            self._quota_reason = ""

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def churn_rate(self, now: Timestamp) -> float:
        """Fraction of recent classified outcomes that were failures.

        Returns 0.0 when there are fewer than ``min_samples`` classified
        outcomes — a fresh group is HEALTHY by default, not HOSTILE.
        """
        with self._lock:
            self._gc(now)
            classified = [
                fate for outcome in self._outcomes.values() if (fate := self._classify(outcome, now)) is not None
            ]
        if len(classified) < self._min_samples:
            return 0.0
        bad = sum(1 for fate in classified if fate != SliceFate.LONG_LIVED)
        return bad / len(classified)

    def health(self, now: Timestamp) -> GroupHealth:
        rate = self.churn_rate(now)
        if rate >= HOSTILE_CHURN_RATE:
            return GroupHealth.HOSTILE
        if rate >= CHURNING_CHURN_RATE:
            return GroupHealth.CHURNING
        if rate >= SUSPECT_CHURN_RATE:
            return GroupHealth.SUSPECT
        return GroupHealth.HEALTHY

    def can_scale_up(self, now: Timestamp) -> bool:
        """False when the group is HOSTILE or inside the quota block window."""
        quota_deadline, _ = self._snapshot_quota()
        if quota_deadline is not None and not quota_deadline.expired(now=now):
            return False
        return self.health(now) != GroupHealth.HOSTILE

    def scale_up_rate_multiplier(self, now: Timestamp) -> float:
        """Multiplier applied to the desired scale-up batch size."""
        quota_deadline, _ = self._snapshot_quota()
        if quota_deadline is not None and not quota_deadline.expired(now=now):
            return 0.0
        return _RATE_MULTIPLIERS[self.health(now)]

    def should_block_scale_down(self, now: Timestamp) -> bool:
        """During CHURNING/HOSTILE, surviving slices are precious — don't reap them.

        Quota does not block scale-down: an active quota window only means we
        can't grow, not that the slices we have are precious.
        """
        return self.health(now) in (GroupHealth.CHURNING, GroupHealth.HOSTILE)

    def block_reason(self, now: Timestamp) -> str | None:
        """Human-readable reason scale-up is currently blocked, or ``None``."""
        quota_deadline, quota_reason = self._snapshot_quota()
        if quota_deadline is not None and not quota_deadline.expired(now=now):
            return f"quota exceeded: {quota_reason}" if quota_reason else "quota exceeded"
        if self.health(now) == GroupHealth.HOSTILE:
            return f"churn rate {self.churn_rate(now):.0%}"
        return None

    def quota_deadline(self, now: Timestamp) -> Timestamp | None:
        """Active quota-block expiry, or ``None`` if quota isn't currently blocking."""
        deadline, _ = self._snapshot_quota()
        if deadline is None or deadline.expired(now=now):
            return None
        return deadline.as_timestamp()

    def recent_failure_count(self, now: Timestamp) -> int:
        """Count of recent classified-as-failure outcomes — for status display."""
        with self._lock:
            self._gc(now)
            classified = [
                fate for outcome in self._outcomes.values() if (fate := self._classify(outcome, now)) is not None
            ]
        return sum(1 for fate in classified if fate != SliceFate.LONG_LIVED)

    def _snapshot_quota(self) -> tuple[Deadline | None, str]:
        """Snapshot (deadline, reason) under the lock for race-free reads."""
        with self._lock:
            return self._quota_until, self._quota_reason

    # -----------------------------------------------------------------------
    # Internals (caller must hold _lock for _gc; _classify is pure)
    # -----------------------------------------------------------------------

    def _gc(self, now: Timestamp) -> None:
        """Drop resolved outcomes whose ``fate_at`` has aged out of the window.

        In-flight outcomes are intentionally **not** evicted: a slice that
        legitimately runs longer than the window must still be recognised when
        it eventually terminates, otherwise a normal preemption of a long-lived
        survivor would be mis-classified as a short-lived failure (the missing
        ``record_created`` would cause :meth:`record_terminated` to fabricate
        ``created_at = terminated_at``, age=0). The no-ghost guarantee from
        the runtime ensures every in-flight outcome eventually resolves, at
        which point it ages out via its ``fate_at``.
        """
        cutoff_ms = now.epoch_ms() - self._window.to_ms()
        stale = [
            slice_id
            for slice_id, outcome in self._outcomes.items()
            if outcome.fate_at is not None and outcome.fate_at.epoch_ms() < cutoff_ms
        ]
        for slice_id in stale:
            del self._outcomes[slice_id]

    def _classify(self, outcome: SliceOutcome, now: Timestamp) -> SliceFate | None:
        """Classify a single outcome into PREEMPTED / BOOT_FAILED / LONG_LIVED / None.

        - BOOT_FAILED is always a failure.
        - PREEMPTED with age < short_lived_threshold is a failure (PREEMPTED).
        - PREEMPTED with age >= short_lived_threshold reclassifies as LONG_LIVED.
        - In-flight outcomes with age >= long_lived_threshold count as LONG_LIVED
          (the slice has demonstrably survived even though we haven't seen
          termination yet).
        - In-flight outcomes younger than long_lived_threshold are pending and
          contribute nothing to the rate.
        """
        if outcome.fate == SliceFate.BOOT_FAILED:
            return SliceFate.BOOT_FAILED
        if outcome.fate == SliceFate.PREEMPTED:
            assert outcome.fate_at is not None
            age_ms = outcome.fate_at.epoch_ms() - outcome.created_at.epoch_ms()
            if age_ms < self._short_lived_threshold.to_ms():
                return SliceFate.PREEMPTED
            return SliceFate.LONG_LIVED
        # In-flight (fate is None)
        age_ms = now.epoch_ms() - outcome.created_at.epoch_ms()
        if age_ms >= self._long_lived_threshold.to_ms():
            return SliceFate.LONG_LIVED
        return None
