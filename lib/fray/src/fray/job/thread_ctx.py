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

"""Thread-based execution context for parallel computing.

Each task spawns a new thread rather than using a thread pool.
"""

import inspect
import threading
from collections.abc import Callable
from typing import Any, Literal

from fray.job.context import ActorHandle, ActorMethod


class ThreadFuture:
    """Future that wraps a thread, providing result() and done() interface."""

    def __init__(self):
        self._result: Any = None
        self._exception: Exception | None = None
        self._completed = threading.Event()

    def set_result(self, result: Any) -> None:
        self._result = result
        self._completed.set()

    def set_exception(self, exc: Exception) -> None:
        self._exception = exc
        self._completed.set()

    def result(self, timeout: float | None = None) -> Any:
        self._completed.wait(timeout=timeout)
        if self._exception is not None:
            raise self._exception
        return self._result

    def done(self) -> bool:
        return self._completed.is_set()


class GeneratorFuture:
    """Wrapper for a ThreadFuture that yields items, making it iterable."""

    def __init__(self, future: ThreadFuture):
        self._future = future
        self._iterator: Any = None

    def result(self) -> Any:
        return self._future.result()

    def done(self) -> bool:
        return self._future.done()

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.result())
        return next(self._iterator)


class ThreadActorMethod(ActorMethod):
    """Method wrapper for ThreadContext actors - executes in a new thread with lock."""

    def __init__(self, method: Callable, lock: threading.Lock, context: "ThreadContext"):
        self._method = method
        self._lock = lock
        self._context = context

    def remote(self, *args, **kwargs) -> ThreadFuture:
        future = ThreadFuture()

        def locked_call():
            try:
                with self._lock:
                    result = self._method(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        thread = threading.Thread(target=locked_call)
        thread.start()

        with self._context._threads_lock:
            self._context._threads.append(thread)

        return future


class ThreadActorHandle(ActorHandle):
    """Actor handle for ThreadContext - methods execute in new threads with serialization lock."""

    def __init__(self, instance: Any, lock: threading.Lock, context: "ThreadContext"):
        self._instance = instance
        self._lock = lock
        self._context = context

    def __getattr__(self, method_name: str) -> ThreadActorMethod:
        method = getattr(self._instance, method_name)
        return ThreadActorMethod(method, self._lock, self._context)


class ThreadContext:
    """Execution context using individual threads for parallel execution.

    Each run() call spawns a new thread rather than using a thread pool.
    """

    def __init__(self, max_workers: int = 1):
        """Initialize thread context.

        Args:
            max_workers: Ignored, kept for API compatibility
        """
        self._actors: dict[str, Any] = {}
        self._actor_locks: dict[str, threading.Lock] = {}
        self._actors_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()

    def put(self, obj: Any) -> Any:
        """Identity operation - in-process, no serialization needed."""
        return obj

    def get(self, ref: Any) -> Any:
        """Get result from ThreadFuture or GeneratorFuture."""
        if isinstance(ref, (ThreadFuture, GeneratorFuture)):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> ThreadFuture | GeneratorFuture:
        """Spawn a new thread to execute function."""
        future = ThreadFuture()

        if inspect.isgeneratorfunction(fn):

            def wrapper():
                try:
                    future.set_result(list(fn(*args)))
                except Exception as e:
                    future.set_exception(e)

            thread = threading.Thread(target=wrapper)
            thread.start()

            with self._threads_lock:
                self._threads.append(thread)

            return GeneratorFuture(future)
        else:

            def wrapper():
                try:
                    future.set_result(fn(*args))
                except Exception as e:
                    future.set_exception(e)

            thread = threading.Thread(target=wrapper)
            thread.start()

            with self._threads_lock:
                self._threads.append(thread)

            return future

    def wait(self, futures: list[ThreadFuture | GeneratorFuture], num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete."""
        if num_returns >= len(futures):
            for f in futures:
                f.result()
            return futures, []

        ready: list[ThreadFuture | GeneratorFuture] = []
        pending = list(futures)

        while len(ready) < num_returns and pending:
            for i, f in enumerate(pending):
                if f.done():
                    ready.append(pending.pop(i))
                    break
            else:
                threading.Event().wait(0.001)

        return ready, pending

    def create_actor(
        self,
        actor_class: type,
        *args,
        name: str | None = None,
        get_if_exists: bool = False,
        lifetime: Literal["non_detached", "detached"] = "non_detached",
        preemptible: bool = True,
        **kwargs,
    ) -> ThreadActorHandle:
        """Create an actor (stateful service) within the thread context."""
        with self._actors_lock:
            if name is not None and name in self._actors:
                if get_if_exists:
                    return ThreadActorHandle(self._actors[name], self._actor_locks[name], self)
                else:
                    raise ValueError(f"Actor {name} already exists")

            instance = actor_class(*args, **kwargs)
            actor_lock = threading.Lock()

            if name is not None:
                self._actors[name] = instance
                self._actor_locks[name] = actor_lock

            return ThreadActorHandle(instance, actor_lock, self)
