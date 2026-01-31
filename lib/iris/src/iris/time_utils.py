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


import random
import time
from collections.abc import Callable
from datetime import datetime, timezone


def now_ms() -> int:
    """Return current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


class Deadline:
    """Represents a point in time after which an operation should timeout.

    Uses monotonic time internally to avoid issues with system clock adjustments.

    Example:
        deadline = Deadline.from_seconds(30.0)
        while not deadline.expired():
            if try_operation():
                break
            time.sleep(0.1)
        deadline.raise_if_expired("Operation timed out")
    """

    def __init__(self, deadline_monotonic: float):
        """Create deadline from monotonic time value.

        Args:
            deadline_monotonic: Absolute deadline in time.monotonic() units
        """
        self._deadline = deadline_monotonic

    @classmethod
    def from_seconds(cls, timeout_seconds: float) -> "Deadline":
        """Create deadline from seconds in the future.

        Args:
            timeout_seconds: Timeout duration in seconds

        Returns:
            Deadline instance
        """
        return cls(time.monotonic() + timeout_seconds)

    @classmethod
    def from_ms(cls, timeout_ms: int) -> "Deadline":
        """Create deadline from milliseconds in the future.

        Args:
            timeout_ms: Timeout duration in milliseconds

        Returns:
            Deadline instance
        """
        return cls(time.monotonic() + timeout_ms / 1000.0)

    def expired(self) -> bool:
        """Check if deadline has passed.

        Returns:
            True if current time is past the deadline
        """
        return time.monotonic() >= self._deadline

    def raise_if_expired(self, message: str = "Deadline exceeded") -> None:
        """Raise TimeoutError if deadline has passed.

        Args:
            message: Error message to include in TimeoutError

        Raises:
            TimeoutError: If deadline has expired
        """
        if self.expired():
            raise TimeoutError(message)

    def remaining_ms(self) -> int:
        """Get remaining milliseconds until deadline.

        Returns:
            Milliseconds remaining (0 if expired)
        """
        remaining_seconds = self._deadline - time.monotonic()
        return max(0, int(remaining_seconds * 1000))

    def remaining_seconds(self) -> float:
        """Get remaining seconds until deadline.

        Returns:
            Seconds remaining (0.0 if expired)
        """
        return max(0.0, self._deadline - time.monotonic())


class Duration:
    """Represents a duration/interval of time.

    Separate from Deadline (point in time for timeouts) and Timestamp (epoch-based).
    Used for time intervals, configuration values, and arithmetic.

    Example:
        timeout = Duration.from_seconds(30.0)
        timeout_ms = timeout.to_ms()

        threshold = Duration.from_minutes(5)
        if task_duration > threshold:
            ...
    """

    def __init__(self, milliseconds: int):
        """Create duration from milliseconds.

        Args:
            milliseconds: Duration in milliseconds
        """
        self._ms = milliseconds

    @classmethod
    def from_seconds(cls, seconds: float) -> "Duration":
        """Create duration from seconds.

        Args:
            seconds: Duration in seconds

        Returns:
            Duration instance
        """
        return cls(int(seconds * 1000))

    @classmethod
    def from_ms(cls, milliseconds: int) -> "Duration":
        """Create duration from milliseconds.

        Args:
            milliseconds: Duration in milliseconds

        Returns:
            Duration instance
        """
        return cls(milliseconds)

    @classmethod
    def from_minutes(cls, minutes: int) -> "Duration":
        """Create duration from minutes.

        Args:
            minutes: Duration in minutes

        Returns:
            Duration instance
        """
        return cls(minutes * 60 * 1000)

    @classmethod
    def from_hours(cls, hours: int) -> "Duration":
        """Create duration from hours.

        Args:
            hours: Duration in hours

        Returns:
            Duration instance
        """
        return cls(hours * 60 * 60 * 1000)

    def to_seconds(self) -> float:
        """Convert to seconds.

        Returns:
            Duration in seconds
        """
        return self._ms / 1000.0

    def to_ms(self) -> int:
        """Convert to milliseconds.

        Returns:
            Duration in milliseconds
        """
        return self._ms

    def __add__(self, other: "Duration") -> "Duration":
        """Add two durations.

        Args:
            other: Duration to add

        Returns:
            New duration representing the sum
        """
        return Duration(self._ms + other._ms)

    def __mul__(self, factor: float) -> "Duration":
        """Multiply duration by a factor.

        Args:
            factor: Multiplication factor

        Returns:
            New duration multiplied by factor
        """
        return Duration(int(self._ms * factor))

    def __lt__(self, other: "Duration") -> bool:
        """Compare if this duration is less than another."""
        return self._ms < other._ms

    def __le__(self, other: "Duration") -> bool:
        """Compare if this duration is less than or equal to another."""
        return self._ms <= other._ms

    def __gt__(self, other: "Duration") -> bool:
        """Compare if this duration is greater than another."""
        return self._ms > other._ms

    def __ge__(self, other: "Duration") -> bool:
        """Compare if this duration is greater than or equal to another."""
        return self._ms >= other._ms

    def __eq__(self, other: object) -> bool:
        """Compare if this duration equals another."""
        if not isinstance(other, Duration):
            return NotImplemented
        return self._ms == other._ms


class Timestamp:
    """Represents a point in time in milliseconds since epoch.

    Provides consistent conversions between different time representations
    and formatting utilities.

    Example:
        created_at = Timestamp.now()
        # ... later ...
        age_ms = created_at.age_ms()
        formatted = created_at.as_formatted_date()

        # Arithmetic
        future = created_at.add_ms(5000)
        if Timestamp.now().after(future):
            ...
    """

    def __init__(self, epoch_ms: int):
        """Create timestamp from milliseconds since epoch.

        Args:
            epoch_ms: Milliseconds since Unix epoch
        """
        self._epoch_ms = epoch_ms

    @classmethod
    def now(cls) -> "Timestamp":
        """Create timestamp for current time.

        Returns:
            Timestamp instance for current time
        """
        return cls(now_ms())

    @classmethod
    def from_ms(cls, epoch_ms: int) -> "Timestamp":
        """Create timestamp from milliseconds since epoch.

        Args:
            epoch_ms: Milliseconds since Unix epoch

        Returns:
            Timestamp instance
        """
        return cls(epoch_ms)

    @classmethod
    def from_seconds(cls, epoch_seconds: float) -> "Timestamp":
        """Create timestamp from seconds since epoch.

        Args:
            epoch_seconds: Seconds since Unix epoch

        Returns:
            Timestamp instance
        """
        return cls(int(epoch_seconds * 1000))

    @classmethod
    def from_proto(cls, proto_timestamp_ms: int) -> "Timestamp":
        """Create timestamp from proto timestamp field.

        Args:
            proto_timestamp_ms: Proto timestamp in milliseconds

        Returns:
            Timestamp instance
        """
        return cls(proto_timestamp_ms)

    def epoch_ms(self) -> int:
        """Get milliseconds since epoch.

        Returns:
            Milliseconds since Unix epoch
        """
        return self._epoch_ms

    def epoch_seconds(self) -> float:
        """Get seconds since epoch.

        Returns:
            Seconds since Unix epoch
        """
        return self._epoch_ms / 1000.0

    def as_formatted_date(self) -> str:
        """Format as ISO 8601 string in UTC.

        Returns:
            ISO 8601 formatted timestamp string
        """
        dt = datetime.fromtimestamp(self.epoch_seconds(), tz=timezone.utc)
        return dt.isoformat()

    def to_proto(self) -> int:
        """Convert to proto timestamp (milliseconds).

        Returns:
            Milliseconds since Unix epoch (for proto messages)
        """
        return self._epoch_ms

    def age_ms(self) -> int:
        """Get age of this timestamp in milliseconds.

        Returns:
            Milliseconds elapsed since this timestamp
        """
        return now_ms() - self._epoch_ms

    def add_ms(self, milliseconds: int) -> "Timestamp":
        """Return new timestamp offset by milliseconds.

        Args:
            milliseconds: Milliseconds to add (can be negative)

        Returns:
            New timestamp offset by the given milliseconds
        """
        return Timestamp(self._epoch_ms + milliseconds)

    def add(self, duration: Duration) -> "Timestamp":
        """Return new timestamp offset by duration.

        Args:
            duration: Duration to add

        Returns:
            New timestamp offset by the duration
        """
        return Timestamp(self._epoch_ms + duration.to_ms())

    def before(self, other: "Timestamp") -> bool:
        """Check if this timestamp is before another.

        Args:
            other: Timestamp to compare against

        Returns:
            True if this timestamp is before the other
        """
        return self._epoch_ms < other._epoch_ms

    def after(self, other: "Timestamp") -> bool:
        """Check if this timestamp is after another.

        Args:
            other: Timestamp to compare against

        Returns:
            True if this timestamp is after the other
        """
        return self._epoch_ms > other._epoch_ms

    def __eq__(self, other: object) -> bool:
        """Compare if this timestamp equals another."""
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self._epoch_ms == other._epoch_ms

    def __lt__(self, other: "Timestamp") -> bool:
        """Compare if this timestamp is less than another."""
        return self._epoch_ms < other._epoch_ms

    def __le__(self, other: "Timestamp") -> bool:
        """Compare if this timestamp is less than or equal to another."""
        return self._epoch_ms <= other._epoch_ms

    def __gt__(self, other: "Timestamp") -> bool:
        """Compare if this timestamp is greater than another."""
        return self._epoch_ms > other._epoch_ms

    def __ge__(self, other: "Timestamp") -> bool:
        """Compare if this timestamp is greater than or equal to another."""
        return self._epoch_ms >= other._epoch_ms


class Timer:
    """Simple timer for measuring elapsed time using monotonic clock.

    Uses monotonic time to avoid issues with system clock adjustments.

    Example:
        timer = Timer()
        do_work()
        elapsed_ms = timer.elapsed_ms()
        print(f"Work took {elapsed_ms}ms")
    """

    def __init__(self):
        """Create and start timer."""
        self._start = time.monotonic()

    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            Seconds elapsed since timer creation or last reset
        """
        return time.monotonic() - self._start

    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds.

        Returns:
            Milliseconds elapsed since timer creation or last reset
        """
        return int(self.elapsed_seconds() * 1000)

    def reset(self) -> None:
        """Reset timer to current time."""
        self._start = time.monotonic()


class RateLimiter:
    """Rate limiter using monotonic time.

    Ensures operations don't run more frequently than a specified interval.

    Example:
        limiter = RateLimiter(interval_seconds=1.0)
        while True:
            if limiter.should_run():
                expensive_operation()
            time.sleep(0.1)
    """

    def __init__(self, interval_seconds: float):
        """Create rate limiter with specified interval.

        Args:
            interval_seconds: Minimum seconds between operations
        """
        self._interval = interval_seconds
        self._last_run: float | None = None

    def should_run(self) -> bool:
        """Check if enough time has passed since last run.

        Automatically updates last run time if returning True.

        Returns:
            True if operation should run, False if rate limited
        """
        now = time.monotonic()
        if self._last_run is None or (now - self._last_run >= self._interval):
            self._last_run = now
            return True
        return False

    def time_until_next(self) -> float:
        """Get seconds until next allowed run.

        Returns:
            Seconds until rate limit resets (0.0 if can run now)
        """
        if self._last_run is None:
            return 0.0
        elapsed = time.monotonic() - self._last_run
        return max(0.0, self._interval - elapsed)

    def reset(self) -> None:
        """Reset rate limiter to allow immediate run."""
        self._last_run = None


class ExponentialBackoff:
    """Exponential backoff with jitter for polling/retry loops.

    Example - manual loop:
        backoff = ExponentialBackoff(initial=0.1, maximum=2.0)
        while not done:
            try_operation()
            time.sleep(backoff.next_interval())

    Example - wait for condition:
        ExponentialBackoff().wait_until(lambda: server.ready, timeout=30.0)

    Example - wait with custom backoff:
        ExponentialBackoff(initial=0.1, maximum=5.0).wait_until_or_raise(
            lambda: connection.established,
            timeout=60.0,
            error_message="Connection failed",
        )
    """

    def __init__(
        self,
        initial: float = 0.05,
        maximum: float = 1.0,
        factor: float = 1.5,
        jitter: float = 0.1,
    ):
        if initial <= 0:
            raise ValueError("initial must be positive")
        if maximum < initial:
            raise ValueError("maximum must be >= initial")
        if factor < 1.0:
            raise ValueError("factor must be >= 1.0")
        if not 0 <= jitter < 1.0:
            raise ValueError("jitter must be in [0, 1)")

        self._initial = initial
        self._maximum = maximum
        self._factor = factor
        self._jitter = jitter
        self._attempt = 0

    def next_interval(self) -> float:
        interval = self._initial * (self._factor**self._attempt)
        interval = min(interval, self._maximum)

        if self._jitter > 0:
            jitter = interval * self._jitter * (2 * random.random() - 1)
            interval = max(0.001, interval + jitter)

        self._attempt += 1
        return interval

    def reset(self) -> None:
        self._attempt = 0

    def wait_until(self, condition: Callable[[], bool], timeout: float) -> bool:
        """Wait for a condition to become true with exponential backoff.

        Args:
            condition: Callable that returns True when the wait should end
            timeout: Maximum time to wait in seconds

        Returns:
            True if condition was met, False if timeout expired
        """
        start = time.monotonic()

        while True:
            if condition():
                return True

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                return False

            interval = self.next_interval()
            remaining = timeout - elapsed
            interval = min(interval, remaining)

            if interval > 0:
                time.sleep(interval)

    def wait_until_or_raise(
        self,
        condition: Callable[[], bool],
        timeout: float,
        error_message: str,
    ) -> None:
        if not self.wait_until(condition, timeout):
            raise TimeoutError(error_message)
