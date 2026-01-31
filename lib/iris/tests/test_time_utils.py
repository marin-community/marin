# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import pytest

from iris.time_utils import Deadline, Duration, RateLimiter, Timer, Timestamp, now_ms


class TestDeadline:
    def test_from_seconds_not_expired(self):
        deadline = Deadline.from_seconds(1.0)
        assert not deadline.expired()

    def test_from_seconds_expires(self):
        deadline = Deadline.from_seconds(0.1)
        time.sleep(0.15)
        assert deadline.expired()

    def test_from_ms_not_expired(self):
        deadline = Deadline.from_ms(1000)
        assert not deadline.expired()

    def test_from_ms_expires(self):
        deadline = Deadline.from_ms(100)
        time.sleep(0.15)
        assert deadline.expired()

    def test_raise_if_expired_not_expired(self):
        deadline = Deadline.from_seconds(1.0)
        deadline.raise_if_expired("Should not raise")

    def test_raise_if_expired_raises(self):
        deadline = Deadline.from_seconds(0.05)
        time.sleep(0.1)
        with pytest.raises(TimeoutError, match="Test timeout"):
            deadline.raise_if_expired("Test timeout")

    def test_remaining_seconds_decreases(self):
        deadline = Deadline.from_seconds(1.0)
        remaining1 = deadline.remaining_seconds()
        assert 0.9 < remaining1 <= 1.0
        time.sleep(0.1)
        remaining2 = deadline.remaining_seconds()
        assert remaining2 < remaining1

    def test_remaining_seconds_zero_when_expired(self):
        deadline = Deadline.from_seconds(0.05)
        time.sleep(0.1)
        assert deadline.remaining_seconds() == 0.0

    def test_remaining_ms_decreases(self):
        deadline = Deadline.from_ms(1000)
        remaining1 = deadline.remaining_ms()
        assert 900 < remaining1 <= 1000
        time.sleep(0.1)
        remaining2 = deadline.remaining_ms()
        assert remaining2 < remaining1

    def test_remaining_ms_zero_when_expired(self):
        deadline = Deadline.from_ms(50)
        time.sleep(0.1)
        assert deadline.remaining_ms() == 0

    def test_zero_timeout(self):
        deadline = Deadline.from_seconds(0.0)
        assert deadline.expired()

    def test_negative_timeout(self):
        deadline = Deadline.from_seconds(-1.0)
        assert deadline.expired()


class TestTimestamp:
    def test_now_creates_timestamp(self):
        ts = Timestamp.now()
        assert isinstance(ts.epoch_ms(), int)
        assert ts.epoch_ms() > 0

    def test_from_ms(self):
        epoch_ms = 1234567890123
        ts = Timestamp.from_ms(epoch_ms)
        assert ts.epoch_ms() == epoch_ms

    def test_from_seconds(self):
        epoch_seconds = 1234567890.123
        ts = Timestamp.from_seconds(epoch_seconds)
        assert ts.epoch_ms() == 1234567890123

    def test_epoch_ms(self):
        epoch_ms = 1234567890123
        ts = Timestamp.from_ms(epoch_ms)
        assert ts.epoch_ms() == epoch_ms

    def test_epoch_seconds(self):
        epoch_ms = 1234567890123
        ts = Timestamp.from_ms(epoch_ms)
        assert ts.epoch_seconds() == 1234567890.123

    def test_as_formatted_date(self):
        # 2009-02-13T23:31:30.123Z
        epoch_ms = 1234567890123
        ts = Timestamp.from_ms(epoch_ms)
        formatted = ts.as_formatted_date()
        assert "2009-02-13" in formatted
        assert "23:31:30" in formatted

    def test_to_proto(self):
        epoch_ms = 1234567890123
        ts = Timestamp.from_ms(epoch_ms)
        assert ts.to_proto() == epoch_ms

    def test_age_ms_increases(self):
        ts = Timestamp.now()
        time.sleep(0.1)
        age = ts.age_ms()
        assert age >= 100

    def test_age_ms_zero_for_current(self):
        ts = Timestamp.from_ms(now_ms())
        age = ts.age_ms()
        assert -10 < age < 10

    def test_now_vs_now_ms_consistent(self):
        before = now_ms()
        ts = Timestamp.now()
        after = now_ms()
        assert before <= ts.epoch_ms() <= after

    def test_conversion_roundtrip_ms(self):
        original_ms = 1234567890123
        ts = Timestamp.from_ms(original_ms)
        assert ts.epoch_ms() == original_ms

    def test_conversion_roundtrip_seconds(self):
        original_seconds = 1234567890.123
        ts = Timestamp.from_seconds(original_seconds)
        # Allow small rounding error
        assert abs(ts.epoch_seconds() - original_seconds) < 0.001

    def test_from_proto(self):
        proto_ts = 1234567890123
        ts = Timestamp.from_proto(proto_ts)
        assert ts.epoch_ms() == proto_ts

    def test_add_ms(self):
        ts = Timestamp.from_ms(1000)
        future = ts.add_ms(500)
        assert future.epoch_ms() == 1500

    def test_add_ms_negative(self):
        ts = Timestamp.from_ms(1000)
        past = ts.add_ms(-500)
        assert past.epoch_ms() == 500

    def test_add_duration(self):
        ts = Timestamp.from_ms(1000)
        duration = Duration.from_seconds(2.0)
        future = ts.add(duration)
        assert future.epoch_ms() == 3000

    def test_before(self):
        ts1 = Timestamp.from_ms(1000)
        ts2 = Timestamp.from_ms(2000)
        assert ts1.before(ts2)
        assert not ts2.before(ts1)
        assert not ts1.before(ts1)

    def test_after(self):
        ts1 = Timestamp.from_ms(1000)
        ts2 = Timestamp.from_ms(2000)
        assert ts2.after(ts1)
        assert not ts1.after(ts2)
        assert not ts1.after(ts1)

    def test_equality(self):
        ts1 = Timestamp.from_ms(1000)
        ts2 = Timestamp.from_ms(1000)
        ts3 = Timestamp.from_ms(2000)
        assert ts1 == ts2
        assert ts1 != ts3

    def test_comparison_operators(self):
        ts1 = Timestamp.from_ms(1000)
        ts2 = Timestamp.from_ms(2000)
        assert ts1 < ts2
        assert ts1 <= ts2
        assert ts2 > ts1
        assert ts2 >= ts1
        assert ts1 <= ts1
        assert ts1 >= ts1


class TestDuration:
    def test_from_seconds(self):
        duration = Duration.from_seconds(1.5)
        assert duration.to_ms() == 1500

    def test_from_ms(self):
        duration = Duration.from_ms(2000)
        assert duration.to_ms() == 2000

    def test_from_minutes(self):
        duration = Duration.from_minutes(2)
        assert duration.to_ms() == 120000
        assert duration.to_seconds() == 120.0

    def test_from_hours(self):
        duration = Duration.from_hours(1)
        assert duration.to_ms() == 3600000
        assert duration.to_seconds() == 3600.0

    def test_to_seconds(self):
        duration = Duration.from_ms(1500)
        assert duration.to_seconds() == 1.5

    def test_to_ms(self):
        duration = Duration.from_seconds(2.5)
        assert duration.to_ms() == 2500

    def test_add(self):
        d1 = Duration.from_seconds(1.0)
        d2 = Duration.from_seconds(2.0)
        d3 = d1 + d2
        assert d3.to_seconds() == 3.0

    def test_multiply(self):
        duration = Duration.from_seconds(2.0)
        doubled = duration * 2.0
        assert doubled.to_seconds() == 4.0

    def test_multiply_fractional(self):
        duration = Duration.from_seconds(10.0)
        halved = duration * 0.5
        assert halved.to_seconds() == 5.0

    def test_comparison_operators(self):
        d1 = Duration.from_seconds(1.0)
        d2 = Duration.from_seconds(2.0)
        assert d1 < d2
        assert d1 <= d2
        assert d2 > d1
        assert d2 >= d1
        assert d1 == Duration.from_ms(1000)
        assert d1 != d2


class TestTimer:
    def test_elapsed_seconds_increases(self):
        timer = Timer()
        time.sleep(0.1)
        elapsed = timer.elapsed_seconds()
        assert 0.09 < elapsed < 0.2

    def test_elapsed_ms_increases(self):
        timer = Timer()
        time.sleep(0.1)
        elapsed = timer.elapsed_ms()
        assert 90 < elapsed < 200

    def test_reset(self):
        timer = Timer()
        time.sleep(0.1)
        timer.reset()
        elapsed = timer.elapsed_seconds()
        assert elapsed < 0.05

    def test_multiple_reads(self):
        timer = Timer()
        time.sleep(0.05)
        e1 = timer.elapsed_ms()
        time.sleep(0.05)
        e2 = timer.elapsed_ms()
        assert e2 > e1


class TestRateLimiter:
    def test_should_run_initially_true(self):
        limiter = RateLimiter(interval_seconds=1.0)
        assert limiter.should_run()

    def test_should_run_rate_limited(self):
        limiter = RateLimiter(interval_seconds=0.2)
        assert limiter.should_run()
        assert not limiter.should_run()

    def test_should_run_after_interval(self):
        limiter = RateLimiter(interval_seconds=0.1)
        assert limiter.should_run()
        time.sleep(0.15)
        assert limiter.should_run()

    def test_time_until_next(self):
        limiter = RateLimiter(interval_seconds=1.0)
        assert limiter.time_until_next() == 0.0
        limiter.should_run()
        remaining = limiter.time_until_next()
        assert 0.9 < remaining <= 1.0

    def test_reset(self):
        limiter = RateLimiter(interval_seconds=1.0)
        limiter.should_run()
        assert not limiter.should_run()
        limiter.reset()
        assert limiter.should_run()

    def test_multiple_intervals(self):
        limiter = RateLimiter(interval_seconds=0.05)
        count = 0
        for _ in range(100):
            if limiter.should_run():
                count += 1
            time.sleep(0.01)
        # Should run roughly every 50ms, so in 1000ms we expect ~20 runs
        assert 15 < count < 25
