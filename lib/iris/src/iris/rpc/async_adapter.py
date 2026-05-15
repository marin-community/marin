# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt a sync RPC service to the async surface expected by
``connectrpc.server.ConnectASGIApplication``.

The ASGI application invokes ``await endpoint.function(...)`` for every
unary RPC, so each handler must be a coroutine function.
``AsyncServiceAdapter`` exposes a sync service's methods as async:

- methods marked with :func:`on_loop` are wrapped in an ``async def`` that
  calls the sync body inline on the event loop. Use this only for short,
  non-blocking handlers (in-memory dicts, very cheap SQL reads) — a long
  handler running inline blocks every other RPC.
- every other sync method gets an ``asyncio.to_thread`` wrapper.
- methods that are already coroutine functions pass through untouched.

Interceptors are not adapted here — each interceptor that participates in
an ASGI chain implements ``async intercept_unary`` directly.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_ON_LOOP_ATTR = "__iris_rpc_on_loop__"


def on_loop(fn: F) -> F:
    """Mark a sync RPC handler as safe to run directly on the event loop.

    The handler must be short and non-blocking. A handler that blocks the
    loop for tens of milliseconds will queue every other RPC behind it.
    """
    setattr(fn, _ON_LOOP_ATTR, True)
    return fn


def _is_on_loop(fn: Callable[..., Any]) -> bool:
    return getattr(fn, _ON_LOOP_ATTR, False)


class AsyncServiceAdapter:
    """Wraps a sync service so it satisfies an async-method Protocol."""

    __slots__ = ("_impl",)

    def __init__(self, impl: Any) -> None:
        self._impl = impl

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._impl, name)
        if name.startswith("_") or not callable(attr):
            return attr
        if inspect.iscoroutinefunction(attr):
            return attr
        if _is_on_loop(attr):

            @functools.wraps(attr)
            async def _on_loop_call(*args: Any, **kwargs: Any) -> Any:
                return attr(*args, **kwargs)

            return _on_loop_call

        @functools.wraps(attr)
        async def _threaded_call(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _threaded_call
