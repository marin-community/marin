# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bridge between sync and async code using a dedicated event loop thread.
"""

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncBridge:
    """
    Run async functions from synchronous code using a dedicated event loop thread.

    Example:
        bridge = AsyncBridge()
        bridge.start()

        # From sync code, call async function
        result = bridge.run(some_async_function(arg1, arg2))

        bridge.stop()
    """

    _loop: asyncio.AbstractEventLoop | None
    _thread: threading.Thread | None

    def __init__(self):
        self._loop = None
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._started = threading.Event()

    def start(self):
        """Start the event loop in a background thread."""
        if self._thread is None:
            raise RuntimeError("AsyncBridge already stopped.")
        self._thread.start()

        # Wait for loop to start
        self._started.wait()

    def stop(self):
        """Stop the event loop and join the thread."""
        if self._loop is None:
            return

        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        Run an async function from sync code and wait for result.

        Args:
            coro: Coroutine to run (e.g., async_func(args))

        Returns:
            Result of the coroutine
        """
        if self._loop is None:
            raise RuntimeError("AsyncBridge not started. Call start() first.")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        """
        Submit an async function without waiting for result.

        Args:
            coro: Coroutine to run

        Returns:
            Future that can be awaited or checked later
        """
        if self._loop is None:
            raise RuntimeError("AsyncBridge not started. Call start() first.")

        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run_event_loop(self):
        """Run the event loop in this thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Signal that loop is ready
        self._started.set()

        # Run forever until stopped
        self._loop.run_forever()

        # Cleanup
        self._loop.close()
