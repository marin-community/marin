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

from iris.rpc import time_pb2


def _now_ms() -> int:
    """Return current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


class Deadline:
    """Represents a point in time after which an operation should timeout.

    Supports two modes:
    - Monotonic mode (from_seconds/from_ms/from_now): uses monotonic clock,
      immune to system clock adjustments. Cannot be serialized to Timestamp.
    - Timestamp mode (after): uses epoch-based Timestamp for deterministic
      testing and proto serialization.

    Example (monotonic):
        deadline = Deadline.from_seconds(30.0)
        while not deadline.expired():
            if try_operation():
                break
            time.sleep(0.1)
        deadline.raise_if_expired("Operation timed out")

    Example (timestamp):
        deadline = Deadline.after(timestamp, Duration.from_seconds(30))
        if deadline.expired(now=current_ts):
            ...
    """

    def __init__(self, deadline_monotonic: float):
        self._deadline = deadline_monotonic
        self._timestamp: Timestamp | None = None

    @classmethod
    def from_seconds(cls, timeout_seconds: float) -> "Deadline":
        """Create deadline from seconds in the future."""
        return cls(time.monotonic() + timeout_seconds)

    @classmethod
    def from_ms(cls, timeout_ms: int) -> "Deadline":
        """Create deadline from milliseconds in the future."""
        return cls(time.monotonic() + timeout_ms / 1000.0)

    @classmethod
    def from_now(cls, duration: "Duration") -> "Deadline":
        """Create deadline from a Duration offset from now."""
        return cls(time.monotonic() + duration.to_seconds())

    @classmethod
    def after(cls, base_ts: "Timestamp", duration: "Duration") -> "Deadline":
        """Create a timestamp-based deadline: base_ts + duration.

        Unlike monotonic deadlines, this supports deterministic testing
        via expired(now=...) and serialization via as_timestamp().
        """
        target = base_ts.add(duration)
        d = cls.__new__(cls)
        d._deadline = 0.0  # unused in timestamp mode
        d._timestamp = target
        return d

    def expired(self, now: "Timestamp | None" = None) -> bool:
        """Check if deadline has passed.

        Args:
            now: If provided, compare against this timestamp (timestamp mode).
                 If None and this is a monotonic deadline, uses monotonic clock.
                 If None and this is a timestamp deadline, uses Timestamp.now().
        """
        if self._timestamp is not None:
            if now is None:
                now = Timestamp.now()
            return now._epoch_ms >= self._timestamp._epoch_ms
        return time.monotonic() >= self._deadline

    def as_timestamp(self) -> "Timestamp":
        """Return the deadline as a Timestamp. Only valid for timestamp-mode deadlines."""
        if self._timestamp is None:
            raise ValueError(
                "as_timestamp() is only supported for timestamp-based deadlines (created via Deadline.after)"
            )
        return self._timestamp

    def raise_if_expired(self, message: str = "Deadline exceeded") -> None:
        """Raise TimeoutError if deadline has passed."""
        if self.expired():
            raise TimeoutError(message)

    def remaining_ms(self) -> int:
        """Get remaining milliseconds until deadline (0 if expired)."""
        remaining_seconds = self._deadline - time.monotonic()
        return max(0, int(remaining_seconds * 1000))

    def remaining_seconds(self) -> float:
        """Get remaining seconds until deadline (0.0 if expired)."""
        return max(0.0, self._deadline - time.monotonic())

    def __repr__(self) -> str:
        if self._timestamp is not None:
            return f"Deadline(until={self._timestamp})"
        return f"Deadline(remaining={self.remaining_seconds():.3f}s)"


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
        self._ms = milliseconds

    @classmethod
    def from_seconds(cls, seconds: float) -> "Duration":
        """Create duration from seconds."""
        return cls(int(seconds * 1000))

    @classmethod
    def from_ms(cls, milliseconds: int) -> "Duration":
        """Create duration from milliseconds."""
        return cls(milliseconds)

    @classmethod
    def from_minutes(cls, minutes: int) -> "Duration":
        """Create duration from minutes."""
        return cls(minutes * 60 * 1000)

    @classmethod
    def from_hours(cls, hours: int) -> "Duration":
        """Create duration from hours."""
        return cls(hours * 60 * 60 * 1000)

    def to_seconds(self) -> float:
        """Convert to seconds."""
        return self._ms / 1000.0

    def to_ms(self) -> int:
        """Convert to milliseconds."""
        return self._ms

    @classmethod
    def from_proto(cls, proto: "time_pb2.Duration") -> "Duration":
        """Create from proto Duration message."""
        return cls(proto.milliseconds)

    def to_proto(self) -> "time_pb2.Duration":
        """Convert to proto Duration message."""
        return time_pb2.Duration(milliseconds=self._ms)

    def __add__(self, other: "Duration") -> "Duration":
        return Duration(self._ms + other._ms)

    def __mul__(self, factor: float) -> "Duration":
        return Duration(int(self._ms * factor))

    def __lt__(self, other: "Duration") -> bool:
        return self._ms < other._ms

    def __le__(self, other: "Duration") -> bool:
        return self._ms <= other._ms

    def __gt__(self, other: "Duration") -> bool:
        return self._ms > other._ms

    def __ge__(self, other: "Duration") -> bool:
        return self._ms >= other._ms

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Duration):
            return NotImplemented
        return self._ms == other._ms

    def __hash__(self) -> int:
        return hash(self._ms)

    def __repr__(self) -> str:
        return f"Duration({self.to_seconds():.3f}s)"


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
        self._epoch_ms = epoch_ms

    @classmethod
    def now(cls) -> "Timestamp":
        """Create timestamp for current time."""
        return cls(_now_ms())

    @classmethod
    def from_ms(cls, epoch_ms: int) -> "Timestamp":
        """Create timestamp from milliseconds since epoch."""
        return cls(epoch_ms)

    @classmethod
    def from_seconds(cls, epoch_seconds: float) -> "Timestamp":
        """Create timestamp from seconds since epoch."""
        return cls(int(epoch_seconds * 1000))

    @classmethod
    def from_proto(cls, proto: "time_pb2.Timestamp") -> "Timestamp":
        """Create from proto Timestamp message."""
        return cls(proto.epoch_ms)

    def epoch_ms(self) -> int:
        """Get milliseconds since epoch."""
        return self._epoch_ms

    def epoch_seconds(self) -> float:
        """Get seconds since epoch."""
        return self._epoch_ms / 1000.0

    def as_formatted_date(self) -> str:
        """Format as ISO 8601 string in UTC."""
        dt = datetime.fromtimestamp(self.epoch_seconds(), tz=timezone.utc)
        return dt.isoformat()

    def as_short_time(self) -> str:
        """Format as HH:MM:SS for log lines."""
        dt = datetime.fromtimestamp(self.epoch_seconds(), tz=timezone.utc)
        return dt.strftime("%H:%M:%S")

    def to_proto(self) -> "time_pb2.Timestamp":
        """Convert to proto Timestamp message."""
        return time_pb2.Timestamp(epoch_ms=self._epoch_ms)

    def age_ms(self) -> int:
        """Get age of this timestamp in milliseconds."""
        return _now_ms() - self._epoch_ms

    def add_ms(self, milliseconds: int) -> "Timestamp":
        """Return new timestamp offset by milliseconds (may be negative)."""
        return Timestamp(self._epoch_ms + milliseconds)

    def add(self, duration: Duration) -> "Timestamp":
        """Return new timestamp offset by duration."""
        return Timestamp(self._epoch_ms + duration.to_ms())

    def before(self, other: "Timestamp") -> bool:
        """Check if this timestamp is before another."""
        return self._epoch_ms < other._epoch_ms

    def after(self, other: "Timestamp") -> bool:
        """Check if this timestamp is after another."""
        return self._epoch_ms > other._epoch_ms

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self._epoch_ms == other._epoch_ms

    def __hash__(self) -> int:
        return hash(self._epoch_ms)

    def __lt__(self, other: "Timestamp") -> bool:
        return self._epoch_ms < other._epoch_ms

    def __le__(self, other: "Timestamp") -> bool:
        return self._epoch_ms <= other._epoch_ms

    def __gt__(self, other: "Timestamp") -> bool:
        return self._epoch_ms > other._epoch_ms

    def __ge__(self, other: "Timestamp") -> bool:
        return self._epoch_ms >= other._epoch_ms

    def __repr__(self) -> str:
        return f"Timestamp({self.as_formatted_date()})"


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
        self._start = time.monotonic()

    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.monotonic() - self._start

    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int(self.elapsed_seconds() * 1000)

    def reset(self) -> None:
        """Reset timer to current time."""
        self._start = time.monotonic()

    def __repr__(self) -> str:
        return f"Timer(elapsed={self.elapsed_seconds():.3f}s)"


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
        self._interval = interval_seconds
        self._last_run: float | None = None

    def should_run(self) -> bool:
        """Check if enough time has passed; updates last run time if True."""
        now = time.monotonic()
        if self._last_run is None or (now - self._last_run >= self._interval):
            self._last_run = now
            return True
        return False

    def time_until_next(self) -> float:
        """Get seconds until next allowed run (0.0 if can run now)."""
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
        ExponentialBackoff().wait_until(lambda: server.ready, timeout=Duration.from_seconds(30.0))

    Example - wait with custom backoff:
        ExponentialBackoff(initial=0.1, maximum=5.0).wait_until_or_raise(
            lambda: connection.established,
            timeout=Duration.from_seconds(60.0),
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

    def wait_until(self, condition: Callable[[], bool], timeout: Duration) -> bool:
        """Wait for a condition to become true with exponential backoff.

        Args:
            condition: Callable that returns True when the wait should end
            timeout: Maximum duration to wait

        Returns:
            True if condition was met, False if timeout expired
        """
        deadline = Deadline.from_now(timeout)

        while True:
            if condition():
                return True

            if deadline.expired():
                return False

            interval = self.next_interval()
            remaining = deadline.remaining_seconds()
            interval = min(interval, remaining)

            if interval > 0:
                time.sleep(interval)

    def wait_until_or_raise(
        self,
        condition: Callable[[], bool],
        timeout: Duration,
        error_message: str,
    ) -> None:
        if not self.wait_until(condition, timeout):
            raise TimeoutError(error_message)
