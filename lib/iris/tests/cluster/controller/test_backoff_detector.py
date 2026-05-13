# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AIMD backoff detector."""

from __future__ import annotations

import math

import pytest
from iris.cluster.controller.autoscaler.backoff_detector import (
    DEFAULT_DECAY_PER_FAILURE,
    DEFAULT_PROBE_FLOOR_PER_MINUTE,
    DEFAULT_RECOVERY_DURATION,
    BackoffDetector,
    GroupHealth,
    SliceFate,
)
from rigging.timing import Duration, Timestamp, TokenBucket

BASE_RATE_PER_MIN = 16
BASE_RATE_PER_SEC = BASE_RATE_PER_MIN / 60.0


def ts(seconds: float) -> Timestamp:
    return Timestamp.from_ms(int(seconds * 1000))


def make_detector(
    *,
    group_name: str = "test_group",
    short_lived_threshold: Duration = Duration.from_minutes(5),
    quota_block_duration: Duration = Duration.from_minutes(5),
    base_rate_per_min: float = BASE_RATE_PER_MIN,
    recovery_duration: Duration = DEFAULT_RECOVERY_DURATION,
    decay_per_failure: float = DEFAULT_DECAY_PER_FAILURE,
    probe_floor_per_minute: float = DEFAULT_PROBE_FLOOR_PER_MINUTE,
) -> BackoffDetector:
    bucket = TokenBucket(capacity=int(base_rate_per_min), refill_period=Duration.from_minutes(1))
    return BackoffDetector(
        group_name=group_name,
        scale_up_bucket=bucket,
        base_scale_up_per_minute=base_rate_per_min,
        recovery_duration=recovery_duration,
        decay_per_failure=decay_per_failure,
        probe_floor_per_minute=probe_floor_per_minute,
        short_lived_threshold=short_lived_threshold,
        quota_block_duration=quota_block_duration,
    )


@pytest.fixture
def detector() -> BackoffDetector:
    return make_detector()


def test_fresh_detector_is_healthy(detector: BackoffDetector) -> None:
    now = ts(1000)
    assert detector.health(now) == 1.0
    assert detector.health_label(now) == GroupHealth.HEALTHY
    assert detector.can_scale_up(now)
    assert detector.try_acquire_scale_up(now)


def test_single_failure_decays_health_to_decay_factor(detector: BackoffDetector) -> None:
    detector.record_create_failed(ts(1000))
    # Same timestamp → no recovery applied, just the decay.
    assert detector.health(ts(1000)) == pytest.approx(DEFAULT_DECAY_PER_FAILURE, abs=1e-9)


def test_n_failures_decay_geometrically(detector: BackoffDetector) -> None:
    for _ in range(3):
        detector.record_create_failed(ts(1000))
    assert detector.health(ts(1000)) == pytest.approx(DEFAULT_DECAY_PER_FAILURE**3, abs=1e-9)


def test_boot_failed_termination_is_a_failure(detector: BackoffDetector) -> None:
    detector.record_created("s1", ts(1000))
    detector.record_terminated("s1", SliceFate.BOOT_FAILED, ts(1100))
    # Health decayed once (recovery over 100s in lock is negligible vs the decay).
    assert detector.health(ts(1100)) < 1.0


def test_short_lived_preemption_is_a_failure(detector: BackoffDetector) -> None:
    """A slice that dies before short_lived_threshold counts as a failure."""
    detector.record_created("s1", ts(1000))
    detector.record_terminated("s1", SliceFate.PREEMPTED, ts(1060))
    assert detector.health(ts(1060)) < 1.0


def test_long_lived_preemption_is_not_a_failure(detector: BackoffDetector) -> None:
    """Slice that survived past short_lived_threshold then got preempted is normal behaviour."""
    detector.record_created("s1", ts(1000))
    detector.record_terminated("s1", SliceFate.PREEMPTED, ts(1360))
    # No decay applied → still healthy.
    assert detector.health(ts(1360)) == 1.0
    assert detector.health_label(ts(1360)) == GroupHealth.HEALTHY


def test_long_lived_survivor_termination_classifies_correctly(detector: BackoffDetector) -> None:
    """A slice that lived 45 minutes still classifies as long-lived on terminate."""
    detector.record_created("survivor", ts(0))
    detector.record_terminated("survivor", SliceFate.PREEMPTED, ts(45 * 60))
    assert detector.health(ts(45 * 60 + 1)) == 1.0


def test_unknown_slice_termination_counted_as_short_lived(detector: BackoffDetector) -> None:
    """If we never saw record_created, default created_at = terminated_at → age 0 → short-lived."""
    detector.record_terminated("ghost", SliceFate.PREEMPTED, ts(1000))
    assert detector.health(ts(1000)) < 1.0


def test_record_created_is_idempotent(detector: BackoffDetector) -> None:
    """A second record_created for the same slice keeps the original created_at."""
    detector.record_created("s1", ts(1000))
    detector.record_created("s1", ts(2000))  # second call ignored
    # Terminate at t=1360 — original created_at=1000 → age 360s > 300s → long-lived.
    detector.record_terminated("s1", SliceFate.PREEMPTED, ts(1360))
    assert detector.health(ts(1400)) == 1.0


def test_recovery_climbs_back_to_full_after_one_hour() -> None:
    """Default recovery_duration=1h drives a floor-pinned score back to 1.0."""
    detector = make_detector(recovery_duration=Duration.from_hours(1))
    # Drive way past the floor.
    for _ in range(20):
        detector.record_create_failed(ts(0))
    # 1 hour + 1s later, fully recovered (clamped at 1.0).
    assert detector.health(ts(3601)) == 1.0
    assert detector.health_label(ts(3601)) == GroupHealth.HEALTHY


def test_recovery_is_partial_within_recovery_duration() -> None:
    """Inside recovery_duration, health is strictly between floor and 1.0."""
    detector = make_detector(recovery_duration=Duration.from_hours(1))
    # Drive to the floor.
    for _ in range(20):
        detector.record_create_failed(ts(0))
    floor = DEFAULT_PROBE_FLOOR_PER_MINUTE / BASE_RATE_PER_MIN
    # 30 minutes — half the recovery window. Gap closes by ~half.
    score = detector.health(ts(1800))
    assert floor < score < 1.0
    # Roughly the midpoint of [floor, 1.0].
    assert score == pytest.approx(floor + (1.0 - floor) * 0.5, abs=0.01)


def test_quota_blocks_scale_up_for_block_duration() -> None:
    detector = make_detector(quota_block_duration=Duration.from_minutes(5))
    detector.record_quota_exceeded("over v6e-256 quota", ts(1000))
    inside = ts(1000 + 60)
    assert not detector.can_scale_up(inside)
    assert not detector.try_acquire_scale_up(inside)
    assert detector.block_reason(inside) == "quota exceeded: over v6e-256 quota"
    assert detector.quota_deadline(inside) is not None
    outside = ts(1000 + 5 * 60 + 1)
    assert detector.can_scale_up(outside)
    assert detector.try_acquire_scale_up(outside)
    assert detector.block_reason(outside) is None
    assert detector.quota_deadline(outside) is None


def test_quota_does_not_decay_health() -> None:
    detector = make_detector()
    detector.record_quota_exceeded("no quota", ts(1000))
    assert detector.health(ts(1001)) == 1.0
    assert detector.health_label(ts(1001)) == GroupHealth.HEALTHY


def test_quota_block_does_not_touch_bucket() -> None:
    bucket = TokenBucket(capacity=BASE_RATE_PER_MIN, refill_period=Duration.from_minutes(1))
    detector = BackoffDetector(
        group_name="g",
        scale_up_bucket=bucket,
        base_scale_up_per_minute=BASE_RATE_PER_MIN,
    )
    detector.record_quota_exceeded("nope", ts(1000))
    tokens_before = bucket.available
    assert not detector.try_acquire_scale_up(ts(1060))
    assert bucket.available == tokens_before


def test_base_rate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="base_scale_up_per_minute"):
        make_detector(base_rate_per_min=0)


def test_decay_must_be_in_open_unit_interval() -> None:
    with pytest.raises(ValueError, match="decay_per_failure"):
        make_detector(decay_per_failure=0)
    with pytest.raises(ValueError, match="decay_per_failure"):
        make_detector(decay_per_failure=1.0)


def test_probe_floor_must_be_below_base_rate() -> None:
    with pytest.raises(ValueError, match="probe_floor_per_minute"):
        make_detector(probe_floor_per_minute=BASE_RATE_PER_MIN)


def test_refill_rate_tracks_health() -> None:
    """try_acquire_scale_up sets refill_rate = base_per_second * health."""
    bucket = TokenBucket(capacity=BASE_RATE_PER_MIN, refill_period=Duration.from_minutes(1))
    detector = BackoffDetector(
        group_name="g",
        scale_up_bucket=bucket,
        base_scale_up_per_minute=BASE_RATE_PER_MIN,
    )
    detector.record_create_failed(ts(1000))
    # Acquire at the same timestamp — recovery is zero, so health == 0.7.
    assert detector.try_acquire_scale_up(ts(1000))
    assert bucket.refill_rate == pytest.approx(BASE_RATE_PER_SEC * DEFAULT_DECAY_PER_FAILURE, abs=1e-6)


def test_probe_floor_clamps_health_above_zero() -> None:
    """Even with many failures, health never drops below the probe floor."""
    bucket = TokenBucket(capacity=BASE_RATE_PER_MIN, refill_period=Duration.from_minutes(1))
    floor_per_min = 0.5
    detector = BackoffDetector(
        group_name="g",
        scale_up_bucket=bucket,
        base_scale_up_per_minute=BASE_RATE_PER_MIN,
        probe_floor_per_minute=floor_per_min,
    )
    for _ in range(50):
        detector.record_create_failed(ts(1000))
    floor = floor_per_min / BASE_RATE_PER_MIN
    assert detector.health(ts(1000)) == pytest.approx(floor, abs=1e-9)
    assert detector.try_acquire_scale_up(ts(1000))
    assert bucket.refill_rate == pytest.approx(BASE_RATE_PER_SEC * floor, abs=1e-6)


@pytest.mark.parametrize(
    "failures,expected_label",
    [
        (0, GroupHealth.HEALTHY),  # 1.0
        (1, GroupHealth.SUSPECT),  # 0.7
        (2, GroupHealth.CHURNING),  # 0.49
        (4, GroupHealth.CHURNING),  # 0.24
        (5, GroupHealth.HOSTILE),  # 0.168
        (10, GroupHealth.HOSTILE),
    ],
)
def test_health_label_bands(failures: int, expected_label: GroupHealth) -> None:
    """Display label thresholds: HEALTHY ≥ 0.8, SUSPECT ≥ 0.5, CHURNING ≥ 0.2, else HOSTILE."""
    detector = make_detector()
    for _ in range(failures):
        detector.record_create_failed(ts(1000))
    assert detector.health_label(ts(1000)) == expected_label


def test_status_label_format() -> None:
    detector = make_detector()
    detector.record_create_failed(ts(1000))
    # health == 0.7 → "suspect health=0.70"
    assert detector.status_label(ts(1000)) == "suspect health=0.70"


def test_continuous_recovery_drives_health_back_up() -> None:
    """Without further failures, health monotonically climbs back toward 1.0."""
    detector = make_detector(recovery_duration=Duration.from_hours(1))
    for _ in range(5):  # → HOSTILE
        detector.record_create_failed(ts(0))
    early = detector.health(ts(60))
    later = detector.health(ts(1800))
    much_later = detector.health(ts(3700))
    assert early < later < much_later
    assert much_later == 1.0


def test_health_never_exceeds_one() -> None:
    detector = make_detector()
    # Many quiet ticks shouldn't push score over 1.0.
    detector.health(ts(0))
    assert detector.health(ts(10**8)) == 1.0


def test_recovery_rate_matches_recovery_duration() -> None:
    """Internal: recovery_per_second exactly bridges floor → 1.0 in recovery_duration."""
    detector = make_detector(recovery_duration=Duration.from_hours(1))
    floor = DEFAULT_PROBE_FLOOR_PER_MINUTE / BASE_RATE_PER_MIN
    expected = (1.0 - floor) / Duration.from_hours(1).to_seconds()
    assert detector._recovery_per_second == pytest.approx(expected, abs=1e-12)
    # And the inverse — number of failures needed to floor — matches log math.
    n_to_floor = math.ceil(math.log(floor) / math.log(DEFAULT_DECAY_PER_FAILURE))
    assert n_to_floor >= 5  # default tuning: ~10 failures to floor
