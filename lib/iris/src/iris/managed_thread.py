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

"""ManagedThread and ThreadRegistry for structured thread lifecycle management.

ManagedThread wraps threading.Thread with an integrated stop event, ensuring
threads are non-daemon and can be cleanly shut down. ThreadRegistry tracks
multiple ManagedThreads for bulk shutdown.

Components use the global registry via get_thread_registry(). Tests swap in
a fresh registry via set_thread_registry() and call shutdown() in teardown.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ManagedThread:
    """Non-daemon thread with integrated shutdown event.

    Target callable must accept threading.Event as its first argument.
    The event is set when stop() is called, signaling the thread to exit.
    """

    def __init__(self, target: Callable[..., Any], *, name: str | None = None, args: tuple = ()):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=target,
            args=(self._stop_event, *args),
            daemon=False,
            name=name,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def name(self) -> str | None:
        return self._thread.name


class ThreadRegistry:
    """Tracks ManagedThreads for bulk shutdown."""

    def __init__(self) -> None:
        self._threads: list[ManagedThread] = []
        self._lock = threading.Lock()

    def spawn(self, target: Callable[..., Any], *, name: str | None = None, args: tuple = ()) -> ManagedThread:
        """Create, register, and start a ManagedThread."""
        thread = ManagedThread(target=target, name=name, args=args)
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return thread

    def shutdown(self, timeout: float = 10.0) -> list[str]:
        """Stop all threads, join with timeout. Returns names of stuck threads.

        Signals all threads to stop first, then joins each with the remaining
        timeout budget (fast-exiting threads don't waste budget).
        """
        with self._lock:
            threads = list(self._threads)

        for thread in threads:
            thread.stop()

        deadline = time.monotonic() + timeout
        stuck: list[str] = []
        for thread in threads:
            remaining = max(0, deadline - time.monotonic())
            thread.join(timeout=remaining)
            if thread.is_alive:
                stuck.append(thread.name or "<unnamed>")
                logger.warning("Thread %s did not exit within timeout", thread.name)

        return stuck

    def __enter__(self) -> "ThreadRegistry":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_global_registry = ThreadRegistry()
_registry_lock = threading.Lock()


def get_thread_registry() -> ThreadRegistry:
    """Return the process-wide ThreadRegistry."""
    return _global_registry


def set_thread_registry(registry: ThreadRegistry) -> ThreadRegistry:
    """Replace the global registry. Returns the previous one (for restore)."""
    global _global_registry
    with _registry_lock:
        old = _global_registry
        _global_registry = registry
    return old
