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

"""Virtual time for chaos tests - makes time.sleep() controllable."""
import time
import threading
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class SleepEvent:
    wake_time: float
    event: threading.Event = field(compare=False)


class VirtualClock:
    """Thread-safe virtual clock that can be ticked forward for testing.

    Key insight: Lock protects shared state (current_time, sleepers queue).
    Event.wait() happens outside the lock. tick() wakes threads by setting events.
    """

    def __init__(self):
        self._current_time = 0.0
        self._sleepers = []  # min-heap of SleepEvent
        self._lock = threading.Lock()

    def time(self):
        """Returns current virtual time."""
        with self._lock:
            return self._current_time

    def sleep(self, seconds):
        """Blocks thread until chronos.tick() advances past wake time."""
        if seconds <= 0:
            return

        event = threading.Event()
        with self._lock:
            wake_time = self._current_time + seconds
            heapq.heappush(self._sleepers, SleepEvent(wake_time, event))

        # Block WITHOUT holding lock - will be woken by tick()
        event.wait()

    def tick(self, duration=None):
        """Advance time and wake sleeping threads.

        If duration=None, advances to next sleeper's wake time.
        If duration specified, advances by that amount.
        """
        with self._lock:
            if duration is None:
                if not self._sleepers:
                    return  # No sleepers, nothing to do
                target_time = self._sleepers[0].wake_time
            else:
                target_time = self._current_time + duration

            # Wake all threads whose wake_time <= target_time
            while self._sleepers and self._sleepers[0].wake_time <= target_time:
                sleeper = heapq.heappop(self._sleepers)
                self._current_time = sleeper.wake_time
                sleeper.event.set()  # Wake the thread

            # Advance to target even if no sleepers woke
            self._current_time = max(self._current_time, target_time)

    def tick_until_idle(self, max_iterations=1000):
        """Keep ticking until no new sleepers appear.

        Useful for advancing through all pending work. Adds small real delays
        to let threads run between ticks.
        """
        for _ in range(max_iterations):
            with self._lock:
                num_sleepers = len(self._sleepers)

            if num_sleepers == 0:
                return  # All done

            self.tick()  # Wake next sleeper
            time.sleep(0.001)  # Small real sleep to let thread run

        raise TimeoutError(f"tick_until_idle exceeded {max_iterations} iterations")
