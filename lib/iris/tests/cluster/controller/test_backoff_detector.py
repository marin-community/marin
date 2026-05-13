# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the in-memory churn detector."""

from __future__ import annotations

import pytest
from iris.cluster.controller.autoscaler.backoff_detector import (
    BackoffDetector,
    GroupHealth,
    SliceFate,
)
from rigging.timing import Duration, Timestamp


def ts(seconds: float) -> Timestamp:
    """Test-fixture helper: build a Timestamp from a seconds-since-epoch number."""
    return Timestamp.from_ms(int(seconds * 1000))


@pytest.fixture
def detector() -> BackoffDetector:
    return BackoffDetector(
        group_name="test_group",
        window=Duration.from_minutes(30),
        short_lived_threshold=Duration.from_minutes(5),
        long_lived_threshold=Duration.from_minutes(10),
        min_samples=3,
    )


def test_empty_detector_is_healthy(detector: BackoffDetector) -> None:
    now = ts(1000)
    assert detector.churn_rate(now) == 0.0
    assert detector.health(now) == GroupHealth.HEALTHY
    assert detector.can_scale_up(now)
    assert detector.scale_up_rate_multiplier(now) == 1.0
    assert not detector.should_block_scale_down(now)
    assert detector.recent_failure_count(now) == 0


def test_below_min_samples_stays_healthy(detector: BackoffDetector) -> None:
    """Even all-failure samples don't trigger backoff below min_samples (3)."""
    detector.record_create_failed(ts(1000))
    detector.record_create_failed(ts(1001))
    assert detector.health(ts(1002)) == GroupHealth.HEALTHY
    assert detector.churn_rate(ts(1002)) == 0.0


def test_three_create_failures_hits_hostile(detector: BackoffDetector) -> None:
    detector.record_create_failed(ts(1000))
    detector.record_create_failed(ts(1001))
    detector.record_create_failed(ts(1002))
    assert detector.churn_rate(ts(1003)) == 1.0
    assert detector.health(ts(1003)) == GroupHealth.HOSTILE
    assert not detector.can_scale_up(ts(1003))
    assert detector.scale_up_rate_multiplier(ts(1003)) == 0.0
    assert detector.should_block_scale_down(ts(1003))


def test_preempted_short_lived_counts_as_failure(detector: BackoffDetector) -> None:
    """PREEMPTED with age < short_lived_threshold = failure."""
    for i in range(3):
        slice_id = f"s{i}"
        detector.record_created(slice_id, ts(1000 + i))
        # Dies 60s later — well under the 5-min short_lived threshold.
        detector.record_terminated(slice_id, SliceFate.PREEMPTED, ts(1060 + i))
    assert detector.churn_rate(ts(1100)) == 1.0
    assert detector.health(ts(1100)) == GroupHealth.HOSTILE


def test_preempted_long_lived_counts_as_success(detector: BackoffDetector) -> None:
    """PREEMPTED with age >= short_lived_threshold = positive sample (survived the test)."""
    for i in range(3):
        slice_id = f"s{i}"
        detector.record_created(slice_id, ts(1000 + i))
        # Dies 6 min later (360s) — past the 5-min threshold.
        detector.record_terminated(slice_id, SliceFate.PREEMPTED, ts(1360 + i))
    assert detector.churn_rate(ts(1400)) == 0.0
    assert detector.health(ts(1400)) == GroupHealth.HEALTHY


def test_boot_failed_always_counts_as_failure(detector: BackoffDetector) -> None:
    """BOOT_FAILED is failure regardless of age (slice never produced work)."""
    for i in range(3):
        slice_id = f"s{i}"
        detector.record_created(slice_id, ts(1000))
        # "Age" up to 10 min in record_terminated — still BOOT_FAILED.
        detector.record_terminated(slice_id, SliceFate.BOOT_FAILED, ts(1600))
    assert detector.churn_rate(ts(1700)) == 1.0
    assert detector.health(ts(1700)) == GroupHealth.HOSTILE


def test_inflight_long_enough_counts_as_long_lived(detector: BackoffDetector) -> None:
    """In-flight slice aged past long_lived_threshold should count as positive."""
    for i in range(3):
        detector.record_created(f"s{i}", ts(1000 + i))
    # 10 minutes later — past long_lived_threshold.
    now = ts(1610)
    assert detector.churn_rate(now) == 0.0
    assert detector.health(now) == GroupHealth.HEALTHY


def test_inflight_too_young_is_pending(detector: BackoffDetector) -> None:
    """In-flight slice younger than long_lived_threshold doesn't contribute."""
    for i in range(3):
        detector.record_created(f"s{i}", ts(1000 + i))
    # 30 seconds later — way too young.
    now = ts(1030)
    # All three samples are pending → classified count is 0 → below min_samples → HEALTHY.
    assert detector.churn_rate(now) == 0.0
    assert detector.health(now) == GroupHealth.HEALTHY


def test_mixed_outcomes_compute_rate(detector: BackoffDetector) -> None:
    """A mix of 2 failures + 3 successes should yield 2/5 = 40% churn → SUSPECT."""
    # 2 short-lived preemptions
    for i in range(2):
        slice_id = f"bad{i}"
        detector.record_created(slice_id, ts(1000 + i))
        detector.record_terminated(slice_id, SliceFate.PREEMPTED, ts(1060 + i))
    # 3 long-lived preemptions (survived past 5 min)
    for i in range(3):
        slice_id = f"good{i}"
        detector.record_created(slice_id, ts(1000 + i))
        detector.record_terminated(slice_id, SliceFate.PREEMPTED, ts(1500 + i))
    now = ts(1600)
    assert detector.churn_rate(now) == pytest.approx(2 / 5)
    assert detector.health(now) == GroupHealth.SUSPECT
    assert detector.scale_up_rate_multiplier(now) == 0.5


def test_churning_threshold(detector: BackoffDetector) -> None:
    """5 failures + 5 successes = 50% → CHURNING (boundary)."""
    for i in range(5):
        slice_id = f"bad{i}"
        detector.record_created(slice_id, ts(1000 + i))
        detector.record_terminated(slice_id, SliceFate.PREEMPTED, ts(1060 + i))
    for i in range(5):
        slice_id = f"good{i}"
        detector.record_created(slice_id, ts(1000 + i))
        detector.record_terminated(slice_id, SliceFate.PREEMPTED, ts(1500 + i))
    now = ts(1600)
    assert detector.churn_rate(now) == 0.5
    assert detector.health(now) == GroupHealth.CHURNING
    assert detector.scale_up_rate_multiplier(now) == 0.1
    assert detector.can_scale_up(now)  # CHURNING still scales up, just slowly
    assert detector.should_block_scale_down(now)


def test_window_drops_old_outcomes(detector: BackoffDetector) -> None:
    """Outcomes older than the 30-min window should be evicted."""
    for i in range(3):
        slice_id = f"old{i}"
        detector.record_created(slice_id, ts(0))
        detector.record_terminated(slice_id, SliceFate.PREEMPTED, ts(60))
    # 31 minutes later — past the 30-min window from fate_at.
    now = ts(60 + 31 * 60)
    assert detector.churn_rate(now) == 0.0
    assert detector.recent_failure_count(now) == 0


def test_outcome_aging_uses_fate_at(detector: BackoffDetector) -> None:
    """An old creation but recent fate_at should still be in-window."""
    detector.record_created("s1", ts(0))
    # Slice runs for 25 minutes, then gets preempted (short-lived case from
    # the detector's perspective: age 25min, but PREEMPTED long-lived wins).
    detector.record_terminated("s1", SliceFate.PREEMPTED, ts(25 * 60))
    # 4 minutes after termination — well within the 30-min window from fate_at.
    now = ts(25 * 60 + 4 * 60)
    detector.record_create_failed(now)
    detector.record_create_failed(now)
    # Now we have: 1 LONG_LIVED + 2 BOOT_FAILED = 2/3 → CHURNING.
    assert detector.churn_rate(now) == pytest.approx(2 / 3)
    assert detector.health(now) == GroupHealth.CHURNING


def test_long_lived_inflight_outcome_is_retained(detector: BackoffDetector) -> None:
    """In-flight outcomes are kept past the window so the slice can be
    correctly classified when it eventually terminates.

    If we evicted them, a normal preemption of a long-lived survivor would
    look like an "unknown slice died" (no record_created found), get a
    synthetic ``created_at = terminated_at``, and be mis-classified as a
    short-lived failure — inflating churn on healthy steady-state groups.
    """
    detector.record_created("alive", ts(0))
    now = ts(31 * 60)
    detector.record_create_failed(now)
    detector.record_create_failed(now)
    detector.record_create_failed(now)
    # 1 LONG_LIVED (in-flight, aged past long_lived_threshold) + 3 BOOT_FAILED
    # → 3/4 = 75% → CHURNING (not HOSTILE).
    assert detector.churn_rate(now) == pytest.approx(3 / 4)
    assert detector.health(now) == GroupHealth.CHURNING


def test_long_lived_survivor_preemption_is_not_short_lived(
    detector: BackoffDetector,
) -> None:
    """Regression: a slice that survived past the window and is then preempted
    must still classify as LONG_LIVED, not as a fabricated short-lived event."""
    detector.record_created("survivor", ts(0))
    # Slice quietly survives for 45 minutes (past the 30-min window).
    # Then GCP preempts it. With the old GC, the slice would have been evicted
    # at age=30min and the termination record would fabricate created_at=now,
    # classifying as PREEMPTED short-lived. With the corrected GC, the
    # in-flight outcome is retained and the termination records age=45min
    # → LONG_LIVED.
    detector.record_terminated("survivor", SliceFate.PREEMPTED, ts(45 * 60))
    now = ts(45 * 60 + 1)
    # Need min_samples=3 to compute a rate; add two BOOT_FAILED.
    detector.record_create_failed(now)
    detector.record_create_failed(now)
    # 1 LONG_LIVED + 2 BOOT_FAILED → 2/3 churn → CHURNING.
    # Critically, the survivor was NOT counted as a failure.
    assert detector.churn_rate(now) == pytest.approx(2 / 3)


def test_record_terminated_long_lived_raises(detector: BackoffDetector) -> None:
    with pytest.raises(ValueError, match="synthetic"):
        detector.record_terminated("s1", SliceFate.LONG_LIVED, ts(1000))


def test_record_created_is_idempotent(detector: BackoffDetector) -> None:
    detector.record_created("s1", ts(1000))
    detector.record_created("s1", ts(2000))  # second call ignored
    detector.record_terminated("s1", SliceFate.PREEMPTED, ts(2060))
    # The first record_created at ts=1000 stuck, so age = 2060 - 1000 = 1060s > 300s.
    # Classifies as LONG_LIVED.
    detector.record_create_failed(ts(2070))
    detector.record_create_failed(ts(2070))
    # 1 LONG_LIVED + 2 BOOT_FAILED = 2/3 churn → CHURNING.
    assert detector.churn_rate(ts(2080)) == pytest.approx(2 / 3)


def test_record_terminated_without_record_created(detector: BackoffDetector) -> None:
    """Surprise termination (no prior record_created) should still count as a failure."""
    detector.record_terminated("ghost1", SliceFate.PREEMPTED, ts(1000))
    detector.record_terminated("ghost2", SliceFate.PREEMPTED, ts(1001))
    detector.record_terminated("ghost3", SliceFate.PREEMPTED, ts(1002))
    # created_at == terminated_at → age 0 → PREEMPTED short-lived → all bad.
    assert detector.churn_rate(ts(1003)) == 1.0
    assert detector.health(ts(1003)) == GroupHealth.HOSTILE


def test_long_lived_threshold_below_short_lived_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="long_lived_threshold"):
        BackoffDetector(
            "g",
            short_lived_threshold=Duration.from_minutes(10),
            long_lived_threshold=Duration.from_minutes(5),
        )


def test_min_samples_below_one_rejected() -> None:
    with pytest.raises(ValueError, match="min_samples"):
        BackoffDetector("g", min_samples=0)


def test_quota_blocks_scale_up_for_block_duration() -> None:
    detector = BackoffDetector(
        "g",
        quota_block_duration=Duration.from_minutes(5),
    )
    detector.record_quota_exceeded("over v6e-256 quota", ts(1000))
    # Inside the block window
    inside = ts(1000 + 60)
    assert not detector.can_scale_up(inside)
    assert detector.scale_up_rate_multiplier(inside) == 0.0
    assert detector.block_reason(inside) == "quota exceeded: over v6e-256 quota"
    assert detector.quota_deadline(inside) is not None
    # Past the block window
    outside = ts(1000 + 5 * 60 + 1)
    assert detector.can_scale_up(outside)
    assert detector.scale_up_rate_multiplier(outside) == 1.0
    assert detector.block_reason(outside) is None
    assert detector.quota_deadline(outside) is None


def test_quota_does_not_contribute_to_churn() -> None:
    detector = BackoffDetector("g", min_samples=1)
    detector.record_quota_exceeded("no quota", ts(1000))
    # Recording quota should not affect churn rate.
    assert detector.churn_rate(ts(1001)) == 0.0
    assert detector.health(ts(1001)) == GroupHealth.HEALTHY


def test_quota_blocks_even_when_churn_is_healthy() -> None:
    detector = BackoffDetector("g")
    detector.record_quota_exceeded("no quota", ts(1000))
    inside = ts(1000 + 60)
    assert detector.health(inside) == GroupHealth.HEALTHY
    assert not detector.can_scale_up(inside)  # quota still blocks


def test_quota_does_not_block_scale_down() -> None:
    detector = BackoffDetector("g")
    detector.record_quota_exceeded("no quota", ts(1000))
    assert not detector.should_block_scale_down(ts(1060))


def test_block_reason_prefers_quota_over_churn() -> None:
    detector = BackoffDetector("g", min_samples=3)
    for i in range(3):
        detector.record_create_failed(ts(1000 + i))
    detector.record_quota_exceeded("over quota", ts(1010))
    inside = ts(1020)
    # HOSTILE churn AND quota — quota reason wins for the reporting string.
    assert detector.health(inside) == GroupHealth.HOSTILE
    assert detector.block_reason(inside) == "quota exceeded: over quota"


@pytest.mark.parametrize(
    "rate,expected_health,expected_multiplier",
    [
        (0.0, GroupHealth.HEALTHY, 1.0),
        (0.19, GroupHealth.HEALTHY, 1.0),
        (0.20, GroupHealth.SUSPECT, 0.5),
        (0.49, GroupHealth.SUSPECT, 0.5),
        (0.50, GroupHealth.CHURNING, 0.1),
        (0.79, GroupHealth.CHURNING, 0.1),
        (0.80, GroupHealth.HOSTILE, 0.0),
        (1.00, GroupHealth.HOSTILE, 0.0),
    ],
)
def test_health_thresholds(rate: float, expected_health: GroupHealth, expected_multiplier: float) -> None:
    """Synthesize outcomes to hit specific churn_rate values and verify mapping."""
    # Need at least min_samples=3, and rate = bad/total. Pick total=100 so we can
    # round-trip arbitrary rates to integer counts.
    detector = BackoffDetector("g", min_samples=3)
    total = 100
    bad = round(rate * total)
    good = total - bad
    base_created = ts(0)
    base_died_short = ts(60)  # 60s < short_lived 5min → counts as failure
    base_died_long = ts(60 + 6 * 60)  # 6min+60s > short_lived → counts as success
    # Choose ``now`` just past the latest fate_at so nothing has aged out of
    # the 30-min window.
    now = ts(base_died_long.epoch_ms() // 1000 + 10)
    for i in range(bad):
        slice_id = f"bad{i}"
        detector.record_created(slice_id, base_created)
        detector.record_terminated(slice_id, SliceFate.PREEMPTED, base_died_short)
    for i in range(good):
        slice_id = f"good{i}"
        detector.record_created(slice_id, base_created)
        detector.record_terminated(slice_id, SliceFate.PREEMPTED, base_died_long)
    assert detector.churn_rate(now) == pytest.approx(rate, abs=0.005)
    assert detector.health(now) == expected_health
    assert detector.scale_up_rate_multiplier(now) == expected_multiplier
