# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from iris.time_utils import Deadline, Duration, RateLimiter, Timestamp


def test_deadline_expires():
    """Deadline with short timeout expires after sleeping."""
    deadline = Deadline.from_seconds(0.1)
    time.sleep(0.15)
    assert deadline.expired()


def test_deadline_raise_if_expired():
    """Expired deadline raises TimeoutError."""
    deadline = Deadline.from_seconds(0.05)
    time.sleep(0.1)
    with pytest.raises(TimeoutError, match="Test timeout"):
        deadline.raise_if_expired("Test timeout")


def test_deadline_from_now_with_duration():
    """Deadline.from_now works with Duration."""
    deadline = Deadline.from_now(Duration.from_ms(100))
    assert not deadline.expired()
    time.sleep(0.15)
    assert deadline.expired()


def test_timestamp_proto_roundtrip():
    """Timestamp proto serialization handles edge cases correctly."""
    # Normal case
    original = Timestamp.now()
    proto = original.to_proto()
    restored = Timestamp.from_proto(proto)
    assert original == restored

    # Epoch zero (1970-01-01)
    epoch_zero = Timestamp.from_ms(0)
    proto_zero = epoch_zero.to_proto()
    restored_zero = Timestamp.from_proto(proto_zero)
    assert epoch_zero == restored_zero
    assert restored_zero.epoch_ms() == 0

    # Negative timestamp (before epoch)
    negative = Timestamp.from_ms(-1000)
    proto_neg = negative.to_proto()
    restored_neg = Timestamp.from_proto(proto_neg)
    assert negative == restored_neg
    assert restored_neg.epoch_ms() == -1000

    # Very large timestamp (year 2100+)
    large = Timestamp.from_ms(4102444800000)  # 2100-01-01
    proto_large = large.to_proto()
    restored_large = Timestamp.from_proto(proto_large)
    assert large == restored_large


def test_duration_proto_roundtrip():
    """Duration proto serialization handles edge cases correctly."""
    # Normal case
    original = Duration.from_seconds(5.5)
    proto = original.to_proto()
    restored = Duration.from_proto(proto)
    assert original == restored

    # Zero duration
    zero = Duration.from_ms(0)
    proto_zero = zero.to_proto()
    restored_zero = Duration.from_proto(proto_zero)
    assert zero == restored_zero
    assert restored_zero.to_ms() == 0

    # Negative duration (valid for time offsets/deltas)
    negative = Duration.from_ms(-5000)
    proto_neg = negative.to_proto()
    restored_neg = Duration.from_proto(proto_neg)
    assert negative == restored_neg
    assert restored_neg.to_ms() == -5000

    # Very large duration (days)
    large = Duration.from_hours(24 * 365)  # 1 year
    proto_large = large.to_proto()
    restored_large = Duration.from_proto(proto_large)
    assert large == restored_large


def test_rate_limiter_throttles():
    """RateLimiter prevents running too frequently."""
    limiter = RateLimiter(interval_seconds=0.2)
    assert limiter.should_run()
    assert not limiter.should_run()
    time.sleep(0.25)
    assert limiter.should_run()


# Additional integration tests from test_time_integration.py


def test_timestamp_uses_wall_clock():
    """Timestamp uses wall-clock time and advances correctly."""
    ts1 = Timestamp.now()
    time.sleep(0.02)  # 20ms
    ts2 = Timestamp.now()

    assert ts2.after(ts1)
    assert ts1.before(ts2)
    assert ts2 > ts1

    # Age should be at least 20ms
    age_ms = ts1.age_ms()
    assert age_ms >= 20


def test_deadline_uses_monotonic_time():
    """Deadline uses monotonic time (immune to clock changes)."""
    # Create a deadline 100ms in the future
    deadline = Deadline.from_ms(100)

    # Should not be expired immediately
    assert not deadline.expired()

    # Sleep for 50ms - still not expired
    time.sleep(0.05)
    assert not deadline.expired()

    # Sleep for another 60ms - should be expired now (total 110ms)
    time.sleep(0.06)
    assert deadline.expired()


def test_deadline_remaining_time():
    """Deadline correctly reports remaining time."""
    deadline = Deadline.from_seconds(1.0)

    # Initially should have close to 1 second remaining
    remaining = deadline.remaining_seconds()
    assert 0.95 < remaining <= 1.0

    # After sleeping, remaining time should decrease
    time.sleep(0.1)
    remaining_after = deadline.remaining_seconds()
    assert remaining_after < remaining
    assert 0.85 < remaining_after < 0.95


def test_deadline_timeout_integration():
    """Deadline can be used to enforce timeouts in operations."""
    deadline = Deadline.from_seconds(0.2)  # 200ms timeout
    iterations = 0

    while not deadline.expired():
        iterations += 1
        time.sleep(0.05)  # 50ms per iteration

    # Should have run ~4 iterations (200ms / 50ms)
    assert 3 <= iterations <= 5


def test_deadline_remaining_prevents_overshoot():
    """Using deadline.remaining_seconds() prevents sleeping past deadline."""
    deadline = Deadline.from_seconds(0.15)  # 150ms deadline

    # First sleep - full 100ms
    sleep_time = min(0.1, deadline.remaining_seconds())
    time.sleep(sleep_time)
    assert not deadline.expired()

    # Second sleep - should only sleep ~50ms to not exceed deadline
    sleep_time = min(0.1, deadline.remaining_seconds())
    time.sleep(sleep_time)

    # Should be very close to deadline or just expired
    remaining = deadline.remaining_seconds()
    assert remaining <= 0.02  # Within 20ms of deadline


def test_zero_duration():
    """Zero duration is handled correctly."""
    zero = Duration.from_ms(0)

    assert zero.to_ms() == 0
    assert zero.to_seconds() == 0.0

    # Proto roundtrip
    proto = zero.to_proto()
    restored = Duration.from_proto(proto)
    assert restored.to_ms() == 0


def test_zero_deadline_expires_immediately():
    """Deadline with zero timeout expires immediately."""
    deadline = Deadline.from_ms(0)
    assert deadline.expired()
    assert deadline.remaining_ms() == 0
    assert deadline.remaining_seconds() == 0.0


def test_expired_deadline_remaining_is_zero():
    """Expired deadline returns zero for remaining time."""
    deadline = Deadline.from_ms(10)
    time.sleep(0.02)  # 20ms - definitely expired

    assert deadline.expired()
    assert deadline.remaining_ms() == 0
    assert deadline.remaining_seconds() == 0.0
