# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time

import pytest

from iris.time_utils import Deadline, Duration, ExponentialBackoff, RateLimiter, Timestamp, TokenBucket, log_time


def _records_for(caplog, logger_name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == logger_name]


def test_log_time_logs_elapsed(caplog):
    """log_time context manager logs the elapsed time at the specified level."""
    with caplog.at_level(logging.INFO, logger="iris.time_utils"):
        with log_time("test-op"):
            time.sleep(0.05)

    records = _records_for(caplog, "iris.time_utils")
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "test-op took" in records[0].message


def test_log_time_custom_level(caplog):
    """log_time respects the custom log level."""
    with caplog.at_level(logging.DEBUG, logger="iris.time_utils"):
        with log_time("debug-op", level=logging.DEBUG):
            pass

    records = _records_for(caplog, "iris.time_utils")
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG


def test_deadline_expires():
    """Deadline with short timeout expires after the duration elapses."""
    t0 = Timestamp.from_ms(1_000_000)
    deadline = Deadline.after(t0, Duration.from_ms(100))
    assert not deadline.expired(now=t0)
    assert not deadline.expired(now=t0.add_ms(99))
    assert deadline.expired(now=t0.add_ms(100))
    assert deadline.expired(now=t0.add_ms(200))


def test_deadline_raise_if_expired():
    """Expired deadline raises TimeoutError."""
    t0 = Timestamp.from_ms(1_000_000)
    deadline = Deadline.after(t0, Duration.from_ms(50))
    deadline.raise_if_expired("Should not fire", now=t0)
    with pytest.raises(TimeoutError, match="Test timeout"):
        deadline.raise_if_expired("Test timeout", now=t0.add_ms(100))


def test_deadline_from_now_with_duration():
    """Deadline.after works with Duration."""
    t0 = Timestamp.from_ms(1_000_000)
    deadline = Deadline.after(t0, Duration.from_ms(100))
    assert not deadline.expired(now=t0)
    assert not deadline.expired(now=t0.add_ms(50))
    assert deadline.expired(now=t0.add_ms(150))


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
    t0 = 1000.0
    assert limiter.should_run(now=t0)
    assert not limiter.should_run(now=t0 + 0.1)
    assert limiter.should_run(now=t0 + 0.25)


def test_rate_limiter_wait_blocks_for_interval():
    """wait() blocks until the interval elapses, then returns True."""
    limiter = RateLimiter(interval_seconds=0.1)
    limiter.mark_run()

    start = time.monotonic()
    result = limiter.wait()
    elapsed = time.monotonic() - start

    assert result is True
    assert elapsed >= 0.09  # Slept for approximately the interval


def test_rate_limiter_wait_returns_immediately_when_past_due():
    """wait() returns immediately if more than the interval has already elapsed."""
    limiter = RateLimiter(interval_seconds=0.05)
    limiter.mark_run()
    time.sleep(0.1)

    start = time.monotonic()
    result = limiter.wait()
    elapsed = time.monotonic() - start

    assert result is True
    assert elapsed < 0.02  # Should not have slept


def test_rate_limiter_wait_returns_immediately_on_first_call():
    """wait() returns immediately when no previous run has been recorded."""
    limiter = RateLimiter(interval_seconds=1.0)

    start = time.monotonic()
    result = limiter.wait()
    elapsed = time.monotonic() - start

    assert result is True
    assert elapsed < 0.02


def test_rate_limiter_wait_cancelled_by_event():
    """wait() returns False when the cancel event is set."""
    limiter = RateLimiter(interval_seconds=10.0)
    limiter.mark_run()
    cancel = threading.Event()

    def set_cancel():
        time.sleep(0.05)
        cancel.set()

    threading.Thread(target=set_cancel).start()

    start = time.monotonic()
    result = limiter.wait(cancel=cancel)
    elapsed = time.monotonic() - start

    assert result is False
    assert elapsed < 1.0  # Woke up well before the 10s interval


def test_rate_limiter_mark_run_resets_interval():
    """mark_run() sets the reference point for time_until_next()."""
    limiter = RateLimiter(interval_seconds=0.2)
    assert limiter.time_until_next() == 0.0  # No previous run

    t0 = 1000.0
    limiter.mark_run(now=t0)
    assert limiter.time_until_next(now=t0) == 0.2
    assert limiter.time_until_next(now=t0 + 0.05) == pytest.approx(0.15)
    assert limiter.time_until_next(now=t0 + 0.2) == 0.0


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


def test_deadline_expiry_progression():
    """Deadline transitions from not-expired to expired as time advances."""
    t0 = Timestamp.from_ms(1_000_000)
    deadline = Deadline.after(t0, Duration.from_ms(100))

    assert not deadline.expired(now=t0)
    assert not deadline.expired(now=t0.add_ms(50))
    assert deadline.expired(now=t0.add_ms(110))


def test_deadline_remaining_time():
    """Deadline correctly reports remaining time."""
    t0 = Timestamp.from_ms(1_000_000)
    deadline = Deadline.after(t0, Duration.from_seconds(1))

    assert deadline.remaining_seconds(now=t0) == 1.0
    assert deadline.remaining_ms(now=t0) == 1000

    assert deadline.remaining_seconds(now=t0.add_ms(100)) == pytest.approx(0.9)
    assert deadline.remaining_ms(now=t0.add_ms(100)) == 900

    # After expiry, remaining is zero
    assert deadline.remaining_seconds(now=t0.add_ms(1500)) == 0.0
    assert deadline.remaining_ms(now=t0.add_ms(1500)) == 0


def test_deadline_timeout_integration():
    """Deadline can be used to enforce timeouts in a loop."""
    t0 = Timestamp.from_ms(1_000_000)
    deadline = Deadline.after(t0, Duration.from_ms(200))
    iterations = 0
    now = t0

    while not deadline.expired(now=now):
        iterations += 1
        now = now.add_ms(50)

    assert iterations == 4


def test_deadline_remaining_prevents_overshoot():
    """Using deadline.remaining_seconds() prevents sleeping past deadline."""
    t0 = Timestamp.from_ms(1_000_000)
    deadline = Deadline.after(t0, Duration.from_ms(150))

    now = t0
    # First step: min(100ms, remaining) = 100ms
    step = min(100, deadline.remaining_ms(now=now))
    assert step == 100
    now = now.add_ms(step)
    assert not deadline.expired(now=now)

    # Second step: min(100ms, remaining=50ms) = 50ms
    step = min(100, deadline.remaining_ms(now=now))
    assert step == 50
    now = now.add_ms(step)

    assert deadline.expired(now=now)


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
    t0 = Timestamp.from_ms(1_000_000)
    deadline = Deadline.after(t0, Duration.from_ms(0))
    assert deadline.expired(now=t0)
    assert deadline.remaining_ms(now=t0) == 0
    assert deadline.remaining_seconds(now=t0) == 0.0


def test_expired_deadline_remaining_is_zero():
    """Expired deadline returns zero for remaining time."""
    t0 = Timestamp.from_ms(1_000_000)
    deadline = Deadline.after(t0, Duration.from_ms(10))

    now = t0.add_ms(20)
    assert deadline.expired(now=now)
    assert deadline.remaining_ms(now=now) == 0
    assert deadline.remaining_seconds(now=now) == 0.0


def test_exponential_backoff_does_not_overflow_at_high_attempt_counts():
    """Backoff returns the maximum interval instead of overflowing when attempt count is large.

    Reproduces the bug where idle Zephyr workers poll for ~30 min without a
    task, accumulating ~1800 attempts. With default params (factor=1.5),
    1.5**1755 exceeds float64 max and raises OverflowError.
    """
    backoff = ExponentialBackoff(initial=0.05, maximum=1.0, jitter=0.0)

    # Simulate 2000 consecutive polls with no reset — well past the overflow
    # threshold of ~1755 attempts for factor=1.5.
    backoff._attempt = 2000
    interval = backoff.next_interval()
    assert interval == 1.0

    # Also verify at an extreme count
    backoff._attempt = 100_000
    interval = backoff.next_interval()
    assert interval == 1.0


# --- TokenBucket tests ---


def test_token_bucket_allows_burst_up_to_capacity():
    """TokenBucket allows exactly `capacity` acquires, then rejects."""
    bucket = TokenBucket(capacity=5, refill_period=Duration.from_minutes(1))
    now = Timestamp.from_ms(1_000_000)
    for _ in range(5):
        assert bucket.try_acquire(now=now)
    assert not bucket.try_acquire(now=now)


def test_token_bucket_refills_over_time():
    """After draining, tokens replenish proportionally to elapsed time."""
    bucket = TokenBucket(capacity=5, refill_period=Duration.from_minutes(1))
    t0 = Timestamp.from_ms(1_000_000)

    # Drain all tokens
    for _ in range(5):
        bucket.try_acquire(now=t0)
    assert not bucket.try_acquire(now=t0)

    # After 30 seconds, ~2.5 tokens should have refilled (capacity=5, period=60s)
    t1 = t0.add_ms(30_000)
    assert bucket.try_acquire(now=t1)
    assert bucket.try_acquire(now=t1)
    # Third should fail (only ~2.5 refilled, 2 consumed)
    assert not bucket.try_acquire(now=t1)

    # After another 30 seconds (60s total from t1's last_refill), full capacity minus
    # what was consumed at t1. Tokens: 0.5 + 30s * (5/60) = 0.5 + 2.5 = 3.0
    t2 = t1.add_ms(30_000)
    for _ in range(3):
        assert bucket.try_acquire(now=t2)
    assert not bucket.try_acquire(now=t2)


def test_token_bucket_respects_timestamp():
    """Deterministic behavior using explicit now= parameter."""
    bucket = TokenBucket(capacity=2, refill_period=Duration.from_seconds(10))
    t0 = Timestamp.from_ms(5_000_000)

    assert bucket.try_acquire(now=t0)
    assert bucket.try_acquire(now=t0)
    assert not bucket.try_acquire(now=t0)

    # 5 seconds later: 1 token refilled (2 per 10s = 0.2/s * 5s = 1)
    t1 = t0.add_ms(5_000)
    assert bucket.try_acquire(now=t1)
    assert not bucket.try_acquire(now=t1)


def test_token_bucket_thread_safe():
    """Concurrent acquire from multiple threads doesn't over-allocate."""
    bucket = TokenBucket(capacity=100, refill_period=Duration.from_hours(1))
    successes = [0]
    lock = threading.Lock()

    def acquire_many():
        local_count = 0
        now = Timestamp.from_ms(1_000_000)
        for _ in range(50):
            if bucket.try_acquire(now=now):
                local_count += 1
        with lock:
            successes[0] += local_count

    threads = [threading.Thread(target=acquire_many) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 100 tokens should have been acquired across all threads
    assert successes[0] == 100


def test_token_bucket_acquire_multiple():
    """try_acquire(n=...) consumes multiple tokens at once."""
    bucket = TokenBucket(capacity=10, refill_period=Duration.from_minutes(1))
    now = Timestamp.from_ms(1_000_000)

    assert bucket.try_acquire(n=7, now=now)
    assert bucket.try_acquire(n=3, now=now)
    assert not bucket.try_acquire(n=1, now=now)
