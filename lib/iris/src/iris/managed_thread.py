"""ManagedThread and ThreadRegistry for structured thread lifecycle management.

ManagedThread wraps threading.Thread with an integrated stop event, ensuring
threads are non-daemon and can be cleanly shut down. ThreadRegistry tracks
multiple ManagedThreads for bulk shutdown.
"""

import logging
import threading
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
        """Signal the thread to stop by setting the stop event."""
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

        Signals all threads to stop first, then joins each with a share of the
        total timeout budget.
        """
        with self._lock:
            threads = list(self._threads)

        # Signal all threads to stop
        for thread in threads:
            thread.stop()

        # Join all with per-thread share of timeout
        per_thread_timeout = timeout / max(len(threads), 1)
        stuck: list[str] = []
        for thread in threads:
            thread.join(timeout=per_thread_timeout)
            if thread.is_alive:
                stuck.append(thread.name or "<unnamed>")
                logger.warning("Thread %s did not exit within %.1fs", thread.name, per_thread_timeout)

        return stuck

    def __enter__(self) -> "ThreadRegistry":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
