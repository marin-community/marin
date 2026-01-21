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


def now_ms() -> int:
    """Return current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


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
