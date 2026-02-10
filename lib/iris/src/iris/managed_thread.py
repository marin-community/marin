# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ManagedThread and ThreadContainer for structured thread lifecycle management.

ManagedThread wraps threading.Thread with an integrated stop event, ensuring
threads are non-daemon and can be cleanly shut down. ThreadContainer provides
component-scoped thread groups with hierarchical composition.

A contextvar-based default container is available via get_thread_container()
for components that don't want to pass containers explicitly. Use
thread_container_scope() in tests for isolation.

Thread Safety Best Practices
=============================

All managed threads MUST:
1. Accept threading.Event as first parameter (the stop_event)
2. Check stop_event regularly in loops (recommended: every 0.1-1 second)
3. Exit promptly when stop_event is set (within ~1 second)
4. Use stop_event.wait(timeout) instead of time.sleep() for delays

Example thread pattern:
    def worker_loop(stop_event: threading.Event, config: Config) -> None:
        '''Worker that processes items until stopped.'''
        while not stop_event.is_set():
            item = queue.get(timeout=1.0)
            if item:
                process(item)
            # Check stop event after each item, or use wait for delays:
            # stop_event.wait(timeout=1.0)  # Sleep but check stop_event

NEVER write loops that ignore stop_event:
    BAD:
        def worker(stop_event: threading.Event):
            while True:  # Never checks stop_event!
                time.sleep(10)

    GOOD:
        def worker(stop_event: threading.Event):
            while not stop_event.is_set():
                stop_event.wait(timeout=10)  # Checks stop_event every 10s
"""

import contextlib
import logging
import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any

from iris.time_utils import Deadline, Duration

logger = logging.getLogger(__name__)


class ManagedThread:
    """Non-daemon thread with integrated shutdown event.

    Target callable must accept threading.Event as its first argument.
    The event is set when stop() is called, signaling the thread to exit.

    Optionally, an on_stop callback can be provided to perform cleanup actions
    when stop() is called. This is useful for blocking operations that need
    active intervention to exit (e.g., killing containers, setting server flags).

    Threads automatically remove themselves from their owning container on
    completion to prevent accumulation of completed threads.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        *,
        name: str | None = None,
        args: tuple = (),
        on_stop: Callable[[], None] | None = None,
        _container: "ThreadContainer | None" = None,
    ):
        self._stop_event = threading.Event()
        self._on_stop = on_stop
        self._container = _container

        def _safe_target(*a: Any) -> None:
            # Create watcher thread if on_stop callback is provided
            watcher = None
            on_stop = self._on_stop
            if on_stop is not None:

                def _watch_stop() -> None:
                    self._stop_event.wait()
                    assert on_stop is not None
                    try:
                        on_stop()
                    except Exception:
                        logger.exception("on_stop callback failed for %s", name or "<unnamed>")

                watcher = threading.Thread(
                    target=_watch_stop,
                    name=f"{name}-on-stop" if name else "on-stop",
                    daemon=True,
                )
                watcher.start()

            try:
                target(*a)
            except Exception:
                logger.exception("ManagedThread %s crashed", name or "<unnamed>")
                raise
            finally:
                if watcher:
                    watcher.join(timeout=1.0)
                    if watcher.is_alive():
                        logger.warning("on_stop callback for %s did not complete", name)

                # Remove self from container when thread completes
                if self._container is not None:
                    self._container.remove(self)

        self._thread = threading.Thread(
            target=_safe_target,
            args=(self._stop_event, *args),
            daemon=False,
            name=name,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop (but don't wait for it to exit).

        To wait for the thread to exit, call join() separately. This allows
        ThreadContainer to manage multiple thread timeouts globally.
        """
        self._stop_event.set()
        logger.debug("Signaled thread %s to stop", self._thread.name)

    def join(self, timeout: Duration | None = None) -> None:
        self._thread.join(timeout=timeout.to_seconds() if timeout is not None else None)

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def name(self) -> str | None:
        return self._thread.name


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

    def spawn(
        self,
        target: Callable[..., Any],
        *,
        name: str | None = None,
        args: tuple = (),
        on_stop: Callable[[], None] | None = None,
    ) -> ManagedThread:
        thread = ManagedThread(target=target, name=name, args=args, on_stop=on_stop, _container=self)
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return thread

    def spawn_server(self, server: Any, *, name: str) -> ManagedThread:
        """Spawn a server (like uvicorn.Server) with automatic stop_event bridging.

        When stop() is called, server.should_exit is set to True, causing server.run()
        to exit cleanly.

        Args:
            server: Server instance with should_exit attribute and run() method
            name: Name for the managed thread
        """

        def _run(stop_event: threading.Event) -> None:
            logger.debug("Running server %s (%s)", name, server)
            server.run()
            logger.debug("Server %s exited", name)

        def _stop_server() -> None:
            logger.debug("Signaling server %s to exit", name)
            server.should_exit = True

        return self.spawn(target=_run, name=name, on_stop=_stop_server)

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

    def remove(self, thread: ManagedThread) -> None:
        """Remove a thread from this container.

        Called automatically when threads complete. Thread-safe.
        """
        with self._lock:
            try:
                self._threads.remove(thread)
            except ValueError:
                # Already removed, that's fine
                pass

    @property
    def is_alive(self) -> bool:
        """True if any thread in this container or its children is still running."""
        with self._lock:
            threads = list(self._threads)
            children = list(self._children)
        return any(t.is_alive for t in threads) or any(c.is_alive for c in children)

    def alive_threads(self) -> list[ManagedThread]:
        """Return threads that are still alive, including those in child containers."""
        with self._lock:
            threads = list(self._threads)
            children = list(self._children)

        alive = [t for t in threads if t.is_alive]
        for child in children:
            alive.extend(child.alive_threads())
        return alive

    def wait(self) -> None:
        """Block until all threads have exited."""
        with self._lock:
            children = list(self._children)
            threads = list(self._threads)

        for child in children:
            child.wait()
        for thread in threads:
            thread.join()

    def stop(self, timeout: Duration = Duration.from_seconds(5.0)) -> None:
        """Stop children first, then own threads, then executors.

        Threads are stopped before executors because threads may submit work
        to executors; signaling threads to exit first avoids
        'cannot schedule new futures after shutdown' errors.
        """
        # Take snapshot of children, threads, and executors under lock
        with self._lock:
            children = list(self._children)
            threads = list(self._threads)
            executors = list(self._executors)

        # Stop children first (recursive)
        for child in children:
            child.stop(timeout=timeout)

        # Signal all threads to stop
        for thread in threads:
            thread.stop()

        # Wait for threads to exit with shared deadline
        deadline = Deadline.from_now(timeout)
        for thread in threads:
            remaining = deadline.remaining_seconds()
            thread.join(timeout=Duration.from_seconds(remaining))

        # Warn about threads that didn't exit
        for thread in threads:
            if thread.is_alive:
                logger.warning("Thread %s did not exit within %s", thread.name, timeout)

        # Shutdown executors last
        for executor in executors:
            executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Contextvar-based default container
# ---------------------------------------------------------------------------

_current_container: ContextVar[ThreadContainer | None] = ContextVar("_current_container", default=None)


def get_thread_container() -> ThreadContainer:
    """Return the current thread container, creating a default if none is set.

    In production code this returns a process-wide default container.
    In tests, use thread_container_scope() to get an isolated container
    that is automatically cleaned up.
    """
    container = _current_container.get()
    if container is not None:
        return container

    container = ThreadContainer(name="default")
    _current_container.set(container)
    return container


@contextlib.contextmanager
def thread_container_scope(name: str = "test") -> Generator[ThreadContainer, None, None]:
    """Context manager that installs a fresh ThreadContainer for the duration of the block.

    On exit, all threads in the container are stopped and the previous
    container (if any) is restored. This is the primary mechanism for
    test isolation: each test gets its own container and does not leak
    threads into the next test.
    """
    previous = _current_container.get()
    container = ThreadContainer(name=name)
    _current_container.set(container)
    try:
        yield container
    finally:
        container.stop()
        _current_container.set(previous)
