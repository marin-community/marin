# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt a sync RPC service to the async surface expected by
``connectrpc.server.ConnectASGIApplication``.

The ASGI application invokes ``await endpoint.function(...)`` for every
unary RPC, so each handler must be a coroutine function.
``AsyncServiceAdapter`` exposes a sync service's methods as async by
wrapping each sync method in ``asyncio.to_thread``. Methods that are
already coroutine functions pass through untouched.

Interceptors are not adapted here — each interceptor that participates in
an ASGI chain implements ``async intercept_unary`` directly.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any


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

        @functools.wraps(attr)
        async def _threaded_call(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _threaded_call
