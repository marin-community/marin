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

from fray.job.context import ThreadActorHandle


class GeneratorFuture:
    """Wrapper for a Future that yields items, making it iterable.

    This wraps a Future whose result is a list/iterable, allowing
    iteration via __next__ while maintaining the Future interface
    for unwrapping in get() and wait().
    """

    def __init__(self, future: Future):
        self._future = future
        self._iterator = None

    def result(self) -> Any:
        """Get the underlying result from the future."""
        return self._future.result()

    def __next__(self):
        """Iterate through the future's result."""
        if self._iterator is None:
            self._iterator = iter(self.result())
        return next(self._iterator)


class ThreadContext:
    """Execution context using ThreadPoolExecutor for parallel execution."""

    def __init__(self, max_workers: int):
        """Initialize thread pool context.

        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._actors: dict[str, Any] = {}  # name -> instance
        self._actor_locks: dict[str, threading.Lock] = {}  # per-actor locks
        self._actors_lock = threading.Lock()  # protects _actors dict

    def put(self, obj: Any) -> Any:
        """Identity operation - in-process, no serialization needed."""
        return obj

    def get(self, ref: Any) -> Any:
        """Get result, handling GeneratorFuture, Future objects and plain values."""
        if isinstance(ref, GeneratorFuture):
            return ref.result()
        if isinstance(ref, Future):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> Future | GeneratorFuture:
        """Submit function to thread pool, returning GeneratorFuture for generator functions."""
        if inspect.isgeneratorfunction(fn):
            future = self.executor.submit(lambda: list(fn(*args)))
            return GeneratorFuture(future)
        else:
            return self.executor.submit(fn, *args)

    def wait(self, futures: list[Future | GeneratorFuture], num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete, unwrapping GeneratorFuture objects."""
        # Create mapping from raw futures to original (possibly wrapped) futures
        raw_to_wrapped = {}
        raw_futures = []
        for f in futures:
            if isinstance(f, GeneratorFuture):
                raw_to_wrapped[f._future] = f
                raw_futures.append(f._future)
            else:
                raw_to_wrapped[f] = f
                raw_futures.append(f)

        if num_returns >= len(raw_futures):
            # Wait for all
            done, pending = wait(raw_futures, return_when="ALL_COMPLETED")
            return [raw_to_wrapped[f] for f in done], [raw_to_wrapped[f] for f in pending]

        # Wait until at least num_returns are complete
        done_set = set()
        pending_set = set(raw_futures)

        while len(done_set) < num_returns:
            done, pending = wait(pending_set, return_when="FIRST_COMPLETED")
            done_set.update(done)
            pending_set = pending

        # Split into ready and pending based on num_returns
        done_list = list(done_set)[:num_returns]
        pending_list = list(done_set)[num_returns:] + list(pending_set)

        # Map back to wrapped futures
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
    ) -> ThreadActorHandle:
        """Create an actor (stateful service) within the thread pool context.

        Args:
            actor_class: The class to instantiate as an actor (not decorated)
            *args: Positional arguments for actor __init__
            name: Optional name for actor discovery/reuse across workers
            get_if_exists: If True and named actor exists, return existing instance
            lifetime: "non_detached" (dies with context) or "detached" (survives job)
            preemptible: Ignored for thread pool context
            **kwargs: Keyword arguments for actor __init__

        Returns:
            ThreadActorHandle wrapping the actor instance
        """
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
