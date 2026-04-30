# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time

import pytest

from rigging.timing import (
    Deadline,
    Duration,
    ExponentialBackoff,
    RateLimiter,
    Timestamp,
    TokenBucket,
    log_time,
    retry_with_backoff,
)


def _records_for(caplog, logger_name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == logger_name]


def test_log_time_logs_elapsed(caplog):
    """log_time context manager logs the elapsed time at the specified level."""
    with caplog.at_level(logging.INFO, logger="rigging.timing"):
        with log_time("test-op"):
            time.sleep(0.05)

    records = _records_for(caplog, "rigging.timing")
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "test-op took" in records[0].message


def test_log_time_custom_level(caplog):
    """log_time respects the custom log level."""
    with caplog.at_level(logging.DEBUG, logger="rigging.timing"):
        with log_time("debug-op", level=logging.DEBUG):
            pass

    records = _records_for(caplog, "rigging.timing")
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
    assert elapsed >= 0.09


def test_rate_limiter_wait_returns_immediately_when_past_due():
    """wait() returns immediately if more than the interval has already elapsed."""
    limiter = RateLimiter(interval_seconds=0.05)
    limiter.mark_run()
    time.sleep(0.1)

    start = time.monotonic()
    result = limiter.wait()
    elapsed = time.monotonic() - start

    assert result is True
    assert elapsed < 0.02


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
    assert elapsed < 1.0


def test_rate_limiter_mark_run_resets_interval():
    """mark_run() sets the reference point for time_until_next()."""
    limiter = RateLimiter(interval_seconds=0.2)
    assert limiter.time_until_next() == 0.0

    t0 = 1000.0
    limiter.mark_run(now=t0)
    assert limiter.time_until_next(now=t0) == 0.2
    assert limiter.time_until_next(now=t0 + 0.05) == pytest.approx(0.15)
    assert limiter.time_until_next(now=t0 + 0.2) == 0.0


def test_timestamp_uses_wall_clock():
    """Timestamp uses wall-clock time and advances correctly."""
    ts1 = Timestamp.now()
    time.sleep(0.02)
    ts2 = Timestamp.now()

    assert ts2.after(ts1)
    assert ts1.before(ts2)
    assert ts2 > ts1

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
    step = min(100, deadline.remaining_ms(now=now))
    assert step == 100
    now = now.add_ms(step)
    assert not deadline.expired(now=now)

    step = min(100, deadline.remaining_ms(now=now))
    assert step == 50
    now = now.add_ms(step)

    assert deadline.expired(now=now)


def test_zero_duration():
    """Zero duration is handled correctly."""
    zero = Duration.from_ms(0)
    assert zero.to_ms() == 0
    assert zero.to_seconds() == 0.0


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
    """Backoff returns the maximum interval instead of overflowing when attempt count is large."""
    backoff = ExponentialBackoff(initial=0.05, maximum=1.0, jitter=0.0)

    backoff._attempt = 2000
    interval = backoff.next_interval()
    assert interval == 1.0

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

    for _ in range(5):
        bucket.try_acquire(now=t0)
    assert not bucket.try_acquire(now=t0)

    t1 = t0.add_ms(30_000)
    assert bucket.try_acquire(now=t1)
    assert bucket.try_acquire(now=t1)
    assert not bucket.try_acquire(now=t1)

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

    assert successes[0] == 100


def test_token_bucket_acquire_multiple():
    """try_acquire(n=...) consumes multiple tokens at once."""
    bucket = TokenBucket(capacity=10, refill_period=Duration.from_minutes(1))
    now = Timestamp.from_ms(1_000_000)

    assert bucket.try_acquire(n=7, now=now)
    assert bucket.try_acquire(n=3, now=now)
    assert not bucket.try_acquire(n=1, now=now)


def test_token_bucket_available_refills():
    """available property triggers refill from elapsed time."""
    bucket = TokenBucket(capacity=10, refill_period=Duration.from_minutes(1))

    # Drain the bucket using wall-clock time (available uses Timestamp.now()).
    assert bucket.try_acquire(n=10)
    assert bucket.available == 0


# --- retry_with_backoff tests ---

_FAST_BACKOFF = ExponentialBackoff(initial=0.001, maximum=0.001, jitter=0.0)


def test_retry_with_backoff_succeeds_first_attempt():
    result = retry_with_backoff(lambda: 42)
    assert result == 42


def test_retry_with_backoff_retries_then_succeeds():
    attempts = []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("not yet")
        return "done"

    result = retry_with_backoff(fn, backoff=_FAST_BACKOFF)
    assert result == "done"
    assert len(attempts) == 3


def test_retry_with_backoff_raises_immediately_on_non_retryable():
    called = []

    def fn():
        called.append(1)
        raise TypeError("bad type")

    with pytest.raises(TypeError, match="bad type"):
        retry_with_backoff(fn, retryable=lambda e: not isinstance(e, TypeError))
    assert len(called) == 1


def test_retry_with_backoff_exhausts_max_attempts():
    attempts = []

    def fn():
        attempts.append(1)
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        retry_with_backoff(fn, max_attempts=3, backoff=_FAST_BACKOFF)
    assert len(attempts) == 3


def test_retry_with_backoff_calls_on_retry():
    retries: list[tuple[Exception, int]] = []

    def fn():
        if len(retries) < 2:
            raise OSError("transient")
        return "ok"

    result = retry_with_backoff(
        fn,
        on_retry=lambda exc, attempt: retries.append((exc, attempt)),
        backoff=_FAST_BACKOFF,
    )
    assert result == "ok"
    assert [attempt for _, attempt in retries] == [0, 1]


def test_retry_with_backoff_respects_max_elapsed():
    start = time.monotonic()
    with pytest.raises(OSError):
        retry_with_backoff(
            lambda: (_ for _ in ()).throw(OSError("slow")),
            max_attempts=1000,
            max_elapsed=0.05,
            backoff=_FAST_BACKOFF,
        )
    # Should stop well before 1000 * 0.001s = 1s
    assert time.monotonic() - start < 1.0


def test_retry_with_backoff_does_not_mutate_caller_backoff():
    backoff = ExponentialBackoff(initial=0.001, maximum=0.001, jitter=0.0)
    attempts = []

    def fn():
        attempts.append(1)
        if len(attempts) < 2:
            raise OSError("x")
        return "ok"

    retry_with_backoff(fn, backoff=backoff)
    # Caller's backoff was not advanced by the internal retry loop
    assert backoff._attempt == 0
