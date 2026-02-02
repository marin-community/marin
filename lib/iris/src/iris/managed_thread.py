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
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class ManagedThread:
    """Non-daemon thread with integrated shutdown event.

    Target callable must accept threading.Event as its first argument.
    The event is set when stop() is called, signaling the thread to exit.
    """

    def __init__(self, target: Callable[..., Any], *, name: str | None = None, args: tuple = ()):
        self._stop_event = threading.Event()

        def _safe_target(*a: Any) -> None:
            try:
                target(*a)
            except Exception:
                logger.exception("ManagedThread %s crashed", name or "<unnamed>")
                raise

        self._thread = threading.Thread(
            target=_safe_target,
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

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop all threads and block until they exit.

        Signals all threads to stop, then joins each. If timeout is None
        (the default), blocks indefinitely — rely on pytest-timeout to
        catch hangs. If timeout is set, uses a deadline-based budget.
        """
        with self._lock:
            threads = list(self._threads)

        for thread in threads:
            thread.stop()

        if timeout is None:
            for thread in threads:
                thread.join()
        else:
            deadline = time.monotonic() + timeout
            for thread in threads:
                remaining = max(0, deadline - time.monotonic())
                thread.join(timeout=remaining)

    def __enter__(self) -> "ThreadRegistry":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


class ThreadContainer:
    """Component-scoped thread group with bulk stop/join.

    Threads are created and tracked locally so the owning component can
    shut them all down in one call. Supports hierarchical composition
    via create_child().
    """

    def __init__(self, name: str = "root") -> None:
        self._name = name
        self._threads: list[ManagedThread] = []
        self._children: list[ThreadContainer] = []
        self._executors: list[ThreadPoolExecutor] = []
        self._lock = threading.Lock()

    def spawn(self, target: Callable[..., Any], *, name: str | None = None, args: tuple = ()) -> ManagedThread:
        thread = ManagedThread(target=target, name=name, args=args)
        self._threads.append(thread)
        thread.start()
        return thread

    def spawn_server(self, server: Any, *, name: str) -> ManagedThread:
        def _run(stop_event: threading.Event) -> None:
            server_done = threading.Event()

            def _watch_stop() -> None:
                while not server_done.is_set():
                    if stop_event.wait(timeout=0.25):
                        server.should_exit = True
                        return

            watcher = threading.Thread(target=_watch_stop, name=f"{name}-stop-watcher")
            watcher.start()
            try:
                server.run()
            finally:
                server_done.set()
                watcher.join()

        return self.spawn(target=_run, name=name)

    def spawn_executor(self, max_workers: int, prefix: str) -> ThreadPoolExecutor:
        """Create a ThreadPoolExecutor that will be shut down when this container stops."""
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=prefix)
        with self._lock:
            self._executors.append(executor)
        return executor

    def create_child(self, name: str) -> "ThreadContainer":
        """Create a child container whose lifecycle is bound to this parent."""
        child = ThreadContainer(name=name)
        with self._lock:
            self._children.append(child)
        return child

    def alive_threads(self) -> list[ManagedThread]:
        """Return threads that are still alive, including those in child containers."""
        alive = [t for t in self._threads if t.is_alive]
        for child in self._children:
            alive.extend(child.alive_threads())
        return alive

    def wait(self) -> None:
        """Block until all threads have exited."""
        for child in self._children:
            child.wait()
        for thread in self._threads:
            thread.join()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop children first, then executors, then own threads."""
        for child in self._children:
            child.stop(timeout=timeout)

        for executor in self._executors:
            executor.shutdown(wait=True)

        for thread in self._threads:
            thread.stop()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            remaining = max(0, deadline - time.monotonic())
            thread.join(timeout=remaining)


def spawn_server(server: Any, *, name: str) -> ManagedThread:
    """Spawn a uvicorn Server as a managed thread in the global registry.

    Bridges the ManagedThread stop_event to server.should_exit so the server
    shuts down cleanly when the registry signals stop.
    """

    def _run(stop_event: threading.Event) -> None:
        # Poll stop_event in a local non-daemon thread that exits once
        # either the stop_event fires or the server finishes on its own.
        server_done = threading.Event()

        def _watch_stop() -> None:
            # Wait for whichever comes first: stop requested or server exited.
            while not server_done.is_set():
                if stop_event.wait(timeout=0.25):
                    server.should_exit = True
                    return

        watcher = threading.Thread(target=_watch_stop, name=f"{name}-stop-watcher")
        watcher.start()
        try:
            server.run()
        finally:
            server_done.set()
            watcher.join()

    return get_thread_registry().spawn(target=_run, name=name)


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
