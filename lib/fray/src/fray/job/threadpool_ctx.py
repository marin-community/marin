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

"""Thread pool execution context for parallel computing."""

import inspect
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Literal

__all__ = ["ThreadContext"]


class _GeneratorFuture:
    """Wrapper for a Future that yields items, making it iterable."""

    def __init__(self, future: Future):
        self._future = future
        self._iterator = None

    def result(self) -> Any:
        return self._future.result()

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.result())
        return next(self._iterator)


class _ThreadActorHandle:
    """Actor handle for ThreadContext - serializes method calls with a lock."""

    def __init__(self, instance: Any, lock: threading.Lock, executor: ThreadPoolExecutor):
        self._instance = instance
        self._lock = lock
        self._executor = executor

    def __getattr__(self, method_name: str):
        method = getattr(self._instance, method_name)
        return _ThreadActorMethod(method, self._lock, self._executor)


class _ThreadActorMethod:
    """Method wrapper for ThreadContext actors - submits to executor with lock."""

    def __init__(self, method: Callable, lock: threading.Lock, executor: ThreadPoolExecutor):
        self._method = method
        self._lock = lock
        self._executor = executor

    def remote(self, *args, **kwargs) -> Future:
        def locked_call():
            with self._lock:
                return self._method(*args, **kwargs)

        return self._executor.submit(locked_call)


class ThreadContext:
    """Execution context using ThreadPoolExecutor for parallel execution."""

    def __init__(self, max_workers: int):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._actors: dict[str, Any] = {}
        self._actor_locks: dict[str, threading.Lock] = {}
        self._actors_lock = threading.Lock()

    def put(self, obj: Any) -> Any:
        """Not supported."""
        raise NotImplementedError("ThreadContext does not support .put()")

    def get(self, ref: Any) -> Any:
        """Get result, handling _GeneratorFuture, Future objects and plain values."""
        if isinstance(ref, _GeneratorFuture):
            return ref.result()
        if isinstance(ref, Future):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> Future | _GeneratorFuture:
        """Submit function to thread pool, returning _GeneratorFuture for generator functions."""
        if inspect.isgeneratorfunction(fn):
            future = self.executor.submit(lambda: list(fn(*args)))
            return _GeneratorFuture(future)
        else:
            return self.executor.submit(fn, *args)

    def wait(self, futures: list[Future | _GeneratorFuture], num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete, unwrapping _GeneratorFuture objects."""
        raw_to_wrapped = {}
        raw_futures = []
        for f in futures:
            if isinstance(f, _GeneratorFuture):
                raw_to_wrapped[f._future] = f
                raw_futures.append(f._future)
            else:
                raw_to_wrapped[f] = f
                raw_futures.append(f)

        if num_returns >= len(raw_futures):
            done, pending = wait(raw_futures, return_when="ALL_COMPLETED")
            return [raw_to_wrapped[f] for f in done], [raw_to_wrapped[f] for f in pending]

        done_set = set()
        pending_set = set(raw_futures)

        while len(done_set) < num_returns:
            done, pending = wait(pending_set, return_when="FIRST_COMPLETED")
            done_set.update(done)
            pending_set = pending

        done_list = list(done_set)[:num_returns]
        pending_list = list(done_set)[num_returns:] + list(pending_set)

        return [raw_to_wrapped[f] for f in done_list], [raw_to_wrapped[f] for f in pending_list]

    def create_actor(
        self,
        actor_class: type,
        *args,
        name: str | None = None,
        get_if_exists: bool = False,
        lifetime: Literal["non_detached", "detached"] = "non_detached",
        preemptible: bool = True,
        **kwargs,
    ) -> _ThreadActorHandle:
        with self._actors_lock:
            if name is not None and name in self._actors:
                if get_if_exists:
                    return _ThreadActorHandle(self._actors[name], self._actor_locks[name], self.executor)
                else:
                    raise ValueError(f"Actor {name} already exists")

            instance = actor_class(*args, **kwargs)
            actor_lock = threading.Lock()

            if name is not None:
                self._actors[name] = instance
                self._actor_locks[name] = actor_lock

            return _ThreadActorHandle(instance, actor_lock, self.executor)
