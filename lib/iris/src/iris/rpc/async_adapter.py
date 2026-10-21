# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose synchronous RPC service methods to async Connect applications.

Calls await the service result without blocking the event loop. Selected methods
can have separate concurrency limits. Existing coroutine methods and
interceptors retain their own async behavior.
"""

import asyncio
import contextvars
import functools
import inspect
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from connectrpc.code import Code
from connectrpc.errors import ConnectError


class BoundedThreadExecutor:
    """Run blocking RPCs with a fixed number of workers and queued calls."""

    def __init__(self, *, max_workers: int, max_pending: int, thread_name_prefix: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._slots = threading.BoundedSemaphore(max_workers + max_pending)

    async def run(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self._slots.acquire(blocking=False):
            raise ConnectError(Code.RESOURCE_EXHAUSTED, "Blocking RPC capacity exhausted")

        context = contextvars.copy_context()
        try:
            future = self._executor.submit(context.run, functools.partial(function, *args, **kwargs))
        except RuntimeError:
            self._slots.release()
            raise
        # The asyncio wrapper can be cancelled while its worker is still running.
        # Release capacity only when the underlying concurrent future finishes.
        future.add_done_callback(lambda _future: self._slots.release())
        return await asyncio.wrap_future(future)

    def shutdown(self) -> None:
        """Stop accepting work and cancel queued calls without waiting for workers."""
        self._executor.shutdown(wait=False, cancel_futures=True)


class AsyncServiceAdapter:
    """Wraps a sync service so it satisfies an async-method Protocol."""

    __slots__ = ("_impl", "_isolated_methods")

    def __init__(self, impl: Any, *, isolated_methods: Mapping[str, BoundedThreadExecutor] | None = None) -> None:
        self._impl = impl
        self._isolated_methods = isolated_methods or {}

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._impl, name)
        if name.startswith("_") or not callable(attr):
            return attr
        if inspect.iscoroutinefunction(attr):
            return attr

        @functools.wraps(attr)
        async def _threaded_call(*args: Any, **kwargs: Any) -> Any:
            executor = self._isolated_methods.get(name)
            if executor is not None:
                return await executor.run(attr, *args, **kwargs)
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _threaded_call
