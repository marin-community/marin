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

"""Execution contexts for distributed and parallel computing.

This provides object storage and task management functions for use within a job.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Literal, Protocol

try:
    import ray
except ImportError:
    ray = None


class ExecutionContext(Protocol):
    """Protocol for execution contexts that abstract put/get/run/wait primitives.

    This allows different backends (Ray, ThreadPool, Sync) to share the same
    execution logic while using different execution strategies.
    """

    def put(self, obj: Any) -> Any:
        """Store an object and return a reference to it.

        Args:
            obj: Object to store

        Returns:
            Reference to the stored object (type depends on context)
        """
        ...

    def get(self, ref: Any) -> Any:
        """Retrieve an object from its reference.

        Args:
            ref: Reference to retrieve

        Returns:
            The stored object
        """
        ...

    def run(self, fn: Callable, *args) -> Any:
        """Execute a function with arguments and return a future.

        Args:
            fn: Function to execute
            *args: Arguments to pass to function

        Returns:
            Future representing the execution (type depends on context)
        """
        ...

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete.

        Args:
            futures: List of futures to wait on
            num_returns: Number of futures to wait for

        Returns:
            Tuple of (ready_futures, pending_futures)
        """
        ...


class _ImmediateFuture:
    """Wrapper for immediately available results to match Future interface."""

    def __init__(self, result: Any):
        self._result = result

    def result(self) -> Any:
        return self._result


class SyncContext:
    """Execution context for synchronous (single-threaded) execution."""

    def put(self, obj: Any) -> Any:
        """Identity operation - no serialization needed."""
        return obj

    def get(self, ref: Any) -> Any:
        """Get result, unwrapping _ImmediateFuture if needed."""
        if isinstance(ref, _ImmediateFuture):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> _ImmediateFuture | Generator[_ImmediateFuture, None, None]:
        """Execute function immediately and wrap result."""
        result = fn(*args)
        if hasattr(result, "__iter__") and hasattr(result, "__next__"):
            for item in result:
                yield _ImmediateFuture(item)
        else:
            return _ImmediateFuture(result)

    def wait(self, futures: list[_ImmediateFuture], num_returns: int = 1) -> tuple[list, list]:
        """All futures are immediately ready."""
        return futures[:num_returns], futures[num_returns:]


class GeneratorFuture(Future):
    def __init__(self, future: Future):
        super().__init__()
        self.set_result(future.result())
        self.iterator = iter(self.result())

    def __next__(self) -> Any:
        return next(self.iterator)


class ThreadContext:
    """Execution context using ThreadPoolExecutor for parallel execution."""

    def __init__(self, max_workers: int):
        """Initialize thread pool context.

        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def put(self, obj: Any) -> Any:
        """Identity operation - in-process, no serialization needed."""
        return obj

    def get(self, ref: Any) -> Any:
        """Get result, handling both Future objects and plain values."""
        if isinstance(ref, Future):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> Future | Generator[Future, None, None]:
        """Submit function to thread pool, streaming results for generator functions."""
        # is `fn` a generator function?
        print(f"fn: {fn}, {hasattr(fn, '__iter__')}, {hasattr(fn, '__next__')}")
        if inspect.isgeneratorfunction(fn):
            return GeneratorFuture(self.executor.submit(lambda: list(fn(*args))))
        else:
            return self.executor.submit(fn, *args)

    def wait(self, futures: list[Future], num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete."""
        if num_returns >= len(futures):
            # Wait for all
            done, pending = wait(futures, return_when="ALL_COMPLETED")
            return list(done), list(pending)

        # Wait until at least num_returns are complete
        done_set = set()
        pending_set = set(futures)

        while len(done_set) < num_returns:
            done, pending = wait(pending_set, return_when="FIRST_COMPLETED")
            done_set.update(done)
            pending_set = pending

        # Split into ready and pending based on num_returns
        done_list = list(done_set)[:num_returns]
        pending_list = list(done_set)[num_returns:] + list(pending_set)

        return done_list, pending_list


class RayContext:
    """Execution context using Ray for distributed execution."""

    def __init__(self, ray_options: dict | None = None):
        """Initialize Ray context.

        Args:
            ray_options: Options to pass to ray.remote() (e.g., memory, num_cpus, num_gpus)
        """
        self.ray_options = ray_options or {}

    def put(self, obj: Any):
<<<<<<< HEAD
        """Store object on a worker node."""
        # Msgpack drops dataclass types on decode; fall back to Ray's native
        # serialization when a dataclass is present anywhere in the payload.
        if _contains_dataclass(obj):
            return ray.put(obj)
        return ray.put(msgpack_encode(obj))
||||||| parent of 748611c76 (Switch to pure ray.put in zephyr/fray, remove manual serialization)
        """Store object on a worker node."""
        return ray.put(msgpack_encode(obj))
=======
        """Store object in Ray's object store."""
        return ray.put(obj)
>>>>>>> 748611c76 (Switch to pure ray.put in zephyr/fray, remove manual serialization)

    def get(self, ref):
        """Retrieve an object from Ray's object store."""
        return ray.get(ref)

    def run(self, fn: Callable, *args):
        """Execute function remotely with configured Ray options.

        Uses SPREAD scheduling strategy to avoid running on head node.
        """
        if self.ray_options:
            remote_fn = ray.remote(**self.ray_options)(fn)
        else:
            remote_fn = ray.remote(fn)

        return remote_fn.options(scheduling_strategy="SPREAD").remote(*args)

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        """Wait for Ray futures to complete."""
        ready, pending = ray.wait(futures, num_returns=num_returns)
        return list(ready), list(pending)


@dataclass
class ContextConfig:
    """Configuration for execution context creation.

    Attributes:
        context_type: Type of context (ray, threadpool, sync, or auto)
        max_workers: Maximum number of worker threads (threadpool only)
        memory: Memory requirement per task in bytes (ray only)
        num_cpus: Number of CPUs per task (ray only)
        num_gpus: Number of GPUs per task (ray only)
        ray_options: Additional Ray remote options (ray only)
    """

    context_type: Literal["ray", "threadpool", "sync", "auto"]
    max_workers: int = 1
    memory: int | None = None
    num_cpus: float | None = None
    num_gpus: float | None = None
    ray_options: dict = field(default_factory=dict)


def auto_detect_context_type() -> Literal["ray", "threadpool"]:
    """Automatically detect the best available context type.

    Returns:
        "ray" if Ray is installed and initialized, "threadpool" otherwise
    """
    try:
        if ray.is_initialized():
            return "ray"
    except ImportError:
        pass

    return "threadpool"


def create_context(
    context_type: Literal["ray", "threadpool", "sync", "auto"] = "auto",
    max_workers: int = 1,
    memory: int | None = None,
    num_cpus: float | None = None,
    num_gpus: float | None = None,
    **ray_options,
) -> ExecutionContext:
    """Create execution context from configuration.

    Args:
        context_type: Type of context (ray, threadpool, sync, or auto).
            Use "auto" to auto-detect based on environment (default).
        max_workers: Maximum number of worker threads (threadpool only)
        memory: Memory requirement per task in bytes (ray only)
        num_cpus: Number of CPUs per task (ray only)
        num_gpus: Number of GPUs per task (ray only)
        **ray_options: Additional Ray remote options

    Returns:
        ExecutionContext instance

    Raises:
        ImportError: If ray context requested but ray not installed
        ValueError: If context_type is invalid

    Examples:
        >>> context = create_context("sync")
        >>> context = create_context("threadpool", max_workers=4)
        >>> context = create_context("ray", memory=2*1024**3, num_cpus=2)
        >>> context = create_context("auto")  # Auto-detect ray or threadpool
    """
    if context_type == "auto":
        context_type = auto_detect_context_type()

    if context_type == "sync":
        return SyncContext()

    elif context_type == "threadpool":
        import os

        workers = min(max_workers, os.cpu_count() or 1)
        return ThreadContext(max_workers=workers)

    elif context_type == "ray":
        # Build ray_options dict from parameters
        options = {}
        if memory is not None:
            options["memory"] = memory
        if num_cpus is not None:
            options["num_cpus"] = num_cpus
        if num_gpus is not None:
            options["num_gpus"] = num_gpus
        options.update(ray_options)

        return RayContext(ray_options=options)

    else:
        raise ValueError(f"Unknown context type: {context_type}. Supported: 'ray', 'threadpool', 'sync'")
