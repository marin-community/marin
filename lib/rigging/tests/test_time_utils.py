# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest

from rigging.time_utils import Deadline, Duration, ExponentialBackoff, RateLimiter, Timestamp, TokenBucket


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




def test_rate_limiter_throttles():
    """RateLimiter prevents running too frequently."""
    limiter = RateLimiter(interval_seconds=0.2)
    assert limiter.should_run()
    assert not limiter.should_run()
    time.sleep(0.25)
    assert limiter.should_run()


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

    limiter.mark_run()
    remaining = limiter.time_until_next()
    assert 0.15 < remaining <= 0.2


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
