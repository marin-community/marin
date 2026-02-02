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

"""ManagedThread and ThreadContainer for structured thread lifecycle management.

ManagedThread wraps threading.Thread with an integrated stop event, ensuring
threads are non-daemon and can be cleanly shut down. ThreadContainer provides
component-scoped thread groups with hierarchical composition.
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
        print("STOPPED THREAD", self._thread.name)
        self._thread.join()
        print("JOINED THREAD", self._thread.name)

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
            import sys

            print("RUNNING SERVER", server, name, file=sys.stderr)

            def _watch_stop() -> None:
                while not stop_event.is_set():
                    time.sleep(0.25)

                print("STOPPING SERVER", server, name, file=sys.stderr)
                server.should_exit = True
                print("STOPPED SERVER", server, name, file=sys.stderr)

            watcher = threading.Thread(target=_watch_stop, name=f"{name}-stop-watcher")
            watcher.start()
            server.run()
            print("SERVER DONE", server, name, file=sys.stderr)
            assert server.should_exit
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

    @property
    def is_alive(self) -> bool:
        """True if any thread in this container or its children is still running."""
        return any(t.is_alive for t in self._threads) or any(c.is_alive for c in self._children)

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
        """Stop children first, then own threads, then executors.

        Threads are stopped before executors because threads may submit work
        to executors; signaling threads to exit first avoids
        'cannot schedule new futures after shutdown' errors.
        """
        for child in self._children:
            child.stop(timeout=timeout)

        for thread in self._threads:
            thread.stop()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            remaining = max(0, deadline - time.monotonic())
            thread.join(timeout=remaining)

        for thread in self._threads:
            if thread.is_alive:
                logger.warning("Thread %s did not exit within %s seconds", thread.name, timeout)

        for executor in self._executors:
            executor.shutdown(wait=True)
