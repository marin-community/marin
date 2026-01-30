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

"""Execution contexts for Zephyr backends.

Provides the put/get/run/wait primitives that Backend uses to dispatch
work. These replace the fray.job.JobContext dependency with lightweight
implementations that live in Zephyr itself.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class BackendContext(Protocol):
    """Protocol for execution contexts providing put/get/run/wait primitives.

    This is the interface that Zephyr's Backend class uses to dispatch work.
    Implementations range from synchronous in-process execution to distributed
    actor-based execution.
    """

    def put(self, obj: Any) -> Any:
        """Store an object and return a reference to it."""
        ...

    def get(self, ref: Any) -> Any:
        """Retrieve an object from its reference."""
        ...

    def run(self, fn: Callable, *args, name: str | None = None) -> Any:
        """Execute a function, returning a future/generator proxy."""
        ...

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete. Returns (ready, pending)."""
        ...


# ---------------------------------------------------------------------------
# Immediate future for synchronous execution
# ---------------------------------------------------------------------------


class _ImmediateFuture:
    """Wrapper for immediately available results to match Future interface."""

    def __init__(self, result: Any):
        self._result = result
        self._iterator: Iterator[Any] | None = None

    def result(self) -> Any:
        return self._result

    def __next__(self):
        if self._iterator is None:
            if not inspect.isgenerator(self._result):
                raise StopIteration
            self._iterator = iter(self._result)
        return next(self._iterator)


class _GeneratorFuture:
    """Wraps a Future whose result is a list, making it iterable via __next__."""

    def __init__(self, future: Future):
        self._future = future
        self._iterator: Iterator[Any] | None = None

    def result(self) -> Any:
        return self._future.result()

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.result())
        return next(self._iterator)


# ---------------------------------------------------------------------------
# SyncBackendContext
# ---------------------------------------------------------------------------


class SyncBackendContext:
    """Synchronous (single-threaded) execution context."""

    def put(self, obj: Any) -> Any:
        return obj

    def get(self, ref: Any) -> Any:
        if isinstance(ref, _ImmediateFuture):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args, name: str | None = None) -> _ImmediateFuture:
        result = fn(*args)
        return _ImmediateFuture(result)

    def wait(self, futures: list[_ImmediateFuture], num_returns: int = 1) -> tuple[list, list]:
        return futures[:num_returns], futures[num_returns:]


# ---------------------------------------------------------------------------
# ThreadBackendContext
# ---------------------------------------------------------------------------


class ThreadBackendContext:
    """Thread pool execution context."""

    def __init__(self, max_workers: int):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def put(self, obj: Any) -> Any:
        return obj

    def get(self, ref: Any) -> Any:
        if isinstance(ref, _GeneratorFuture):
            return ref.result()
        if isinstance(ref, Future):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args, name: str | None = None) -> Future | _GeneratorFuture:
        if inspect.isgeneratorfunction(fn):
            future = self.executor.submit(lambda: list(fn(*args)))
            return _GeneratorFuture(future)
        else:
            return self.executor.submit(fn, *args)

    def wait(self, futures: list[Future | _GeneratorFuture], num_returns: int = 1) -> tuple[list, list]:
        raw_to_wrapped: dict[Future, Future | _GeneratorFuture] = {}
        raw_futures: list[Future] = []
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

        done_set: set[Future] = set()
        pending_set = set(raw_futures)

        while len(done_set) < num_returns:
            done, pending_set = wait(pending_set, return_when="FIRST_COMPLETED")
            done_set.update(done)

        done_list = list(done_set)[:num_returns]
        pending_list = list(done_set)[num_returns:] + list(pending_set)

        return [raw_to_wrapped[f] for f in done_list], [raw_to_wrapped[f] for f in pending_list]


# ---------------------------------------------------------------------------
# RayBackendContext
# ---------------------------------------------------------------------------


class RayBackendContext:
    """Ray-based distributed execution context."""

    def __init__(self, ray_options: dict | None = None):
        self.ray_options = ray_options or {}

    def put(self, obj: Any):
        import ray

        return ray.put(obj)

    def get(self, ref):
        import ray

        return ray.get(ref)

    def run(self, fn: Callable, *args, name: str | None = None):
        import ray

        if self.ray_options:
            remote_fn = ray.remote(**self.ray_options)(fn)
        else:
            remote_fn = ray.remote(max_retries=100)(fn)

        options: dict[str, Any] = {"scheduling_strategy": "SPREAD"}
        if name:
            options["name"] = name
        return remote_fn.options(**options).remote(*args)

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        import ray

        ready, pending = ray.wait(futures, num_returns=num_returns, fetch_local=False)
        return list(ready), list(pending)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend_context(
    context_type: str = "auto",
    *,
    max_workers: int = 1,
    **ray_options,
) -> BackendContext:
    """Create a BackendContext.

    Args:
        context_type: One of "sync", "threadpool", "ray", or "auto".
        max_workers: Max worker threads (threadpool only).
        **ray_options: Additional Ray remote options.
    """
    if context_type == "auto":
        import ray

        if ray.is_initialized():
            cluster_spec = os.environ.get("FRAY_CLUSTER_SPEC", "")
            if cluster_spec.startswith("local"):
                context_type = "threadpool"
            else:
                context_type = "ray"
        else:
            context_type = "threadpool"

    if context_type == "sync":
        return SyncBackendContext()
    elif context_type == "threadpool":
        workers = min(max_workers, os.cpu_count() or 1)
        return ThreadBackendContext(max_workers=workers)
    elif context_type == "ray":
        return RayBackendContext(ray_options=ray_options)
    else:
        raise ValueError(f"Unknown context type: {context_type}. Supported: 'ray', 'threadpool', 'sync'")


# ---------------------------------------------------------------------------
# Default context management
# ---------------------------------------------------------------------------

_default_backend_context: ContextVar[BackendContext | None] = ContextVar("zephyr_backend_context", default=None)


@contextmanager
def default_backend_context(ctx: BackendContext):
    """Set the default backend context for the duration of a with-block."""
    old = _default_backend_context.get()
    _default_backend_context.set(ctx)
    try:
        yield ctx
    finally:
        _default_backend_context.set(old)


def get_default_backend_context() -> BackendContext:
    """Get the current default backend context, creating one with 'auto' if unset."""
    ctx = _default_backend_context.get()
    if ctx is None:
        ctx = create_backend_context("auto")
    return ctx
