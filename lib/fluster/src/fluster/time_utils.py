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

"""Time utilities for polling and waiting with exponential backoff."""

import random
import time
from collections.abc import Callable


class ExponentialBackoff:
    """Exponential backoff with jitter for polling/retry loops.

    Each call to `next_interval()` returns an increasing interval up to max_interval.
    Call `reset()` to start over (e.g., after success).

    Example:
        backoff = ExponentialBackoff(initial=0.1, maximum=2.0)
        while not done:
            try_operation()
            time.sleep(backoff.next_interval())
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
        """Return the next backoff interval and increment internal counter."""
        interval = self._initial * (self._factor**self._attempt)
        interval = min(interval, self._maximum)

        if self._jitter > 0:
            jitter = interval * self._jitter * (2 * random.random() - 1)
            interval = max(0.001, interval + jitter)

        self._attempt += 1
        return interval

    def reset(self) -> None:
        """Reset the backoff counter to start from initial interval."""
        self._attempt = 0


def wait_until(
    condition: Callable[[], bool],
    timeout: float,
    initial_interval: float = 0.05,
    max_interval: float = 1.0,
    backoff_factor: float = 1.5,
    jitter_factor: float = 0.1,
) -> bool:
    """Wait for a condition to become true with exponential backoff.

    Polls the condition function with increasing intervals until it returns True
    or the timeout is reached. Starts with fast polling and slows down over time.

    Args:
        condition: Callable that returns True when the wait should end
        timeout: Maximum time to wait in seconds
        initial_interval: Starting poll interval in seconds
        max_interval: Maximum poll interval in seconds
        backoff_factor: Multiplier for interval each iteration
        jitter_factor: Random jitter as fraction of interval

    Returns:
        True if condition was met, False if timeout expired
    """
    backoff = ExponentialBackoff(
        initial=initial_interval,
        maximum=max_interval,
        factor=backoff_factor,
        jitter=jitter_factor,
    )

    start = time.monotonic()

    while True:
        if condition():
            return True

        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            return False

        interval = backoff.next_interval()
        remaining = timeout - elapsed
        interval = min(interval, remaining)

        if interval > 0:
            time.sleep(interval)


def wait_until_with_exception(
    condition: Callable[[], bool],
    timeout: float,
    error_message: str,
    initial_interval: float = 0.05,
    max_interval: float = 1.0,
    backoff_factor: float = 1.5,
) -> None:
    """Wait for condition, raising TimeoutError if not met.

    Same as wait_until but raises instead of returning False.
    """
    if not wait_until(
        condition,
        timeout,
        initial_interval=initial_interval,
        max_interval=max_interval,
        backoff_factor=backoff_factor,
    ):
        raise TimeoutError(error_message)
