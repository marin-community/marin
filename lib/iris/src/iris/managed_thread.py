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
threads are non-daemon and can be cleanly shut down. ThreadContainer groups
ManagedThreads at the component level for bulk shutdown.

Components create their own ThreadContainer instances. Tests ensure proper
cleanup by calling component.stop() methods which cascade to ThreadContainer.stop().
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


class ThreadContainer:
    """Component-scoped thread group with bulk stop/join.

    Threads are created, tracked, and managed entirely within this container.
    The owning component calls stop() to signal all threads and wait for exit.
    """

    def __init__(self) -> None:
        self._threads: list[ManagedThread] = []

    def spawn(self, target: Callable[..., Any], *, name: str | None = None, args: tuple = ()) -> ManagedThread:
        """Create, track, and start a ManagedThread."""
        thread = ManagedThread(target=target, name=name, args=args)
        self._threads.append(thread)
        thread.start()
        return thread

    def spawn_server(self, server: Any, *, name: str) -> ManagedThread:
        thread = spawn_server(server, name=name)
        self._threads.append(thread)
        return thread

    def wait(self) -> None:
        """Block until all threads have exited."""
        for thread in self._threads:
            thread.join()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal all threads to stop, then join with a shared deadline."""
        for thread in self._threads:
            thread.stop()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            remaining = max(0, deadline - time.monotonic())
            thread.join(timeout=remaining)


def _stop_event_to_server(stop_event: threading.Event, server: Any) -> None:
    """Bridge a stop_event to a uvicorn Server's should_exit flag.

    Spawns a daemon thread that waits for stop_event and sets server.should_exit.
    """

    def _watch() -> None:
        stop_event.wait()
        server.should_exit = True

    threading.Thread(target=_watch, daemon=True, name="stop-event-bridge").start()


def spawn_server(server: Any, *, name: str) -> ManagedThread:
    """Create a ManagedThread for a uvicorn Server.

    Bridges the ManagedThread stop_event to server.should_exit so the server
    shuts down cleanly when stop() is called.

    Note: The caller is responsible for tracking and stopping the returned thread.
    """

    def _run(stop_event: threading.Event) -> None:
        _stop_event_to_server(stop_event, server)
        server.run()

    thread = ManagedThread(target=_run, name=name)
    thread.start()
    return thread
