# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib


class HostByteBudget:
    """Bound in-flight host bytes for tasks running on one event loop."""

    def __init__(self, limit_bytes: int):
        if limit_bytes <= 0:
            raise ValueError("host byte budget must be positive")
        self._limit = limit_bytes
        self._in_flight = 0
        self._peak = 0
        self._released = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def peak_bytes(self) -> int:
        return self._peak

    async def acquire(self, num_bytes: int) -> None:
        # A chunk larger than the budget proceeds alone.
        self._loop = asyncio.get_running_loop()
        while self._in_flight and self._in_flight + num_bytes > self._limit:
            self._released.clear()
            await self._released.wait()
        self._in_flight += num_bytes
        self._peak = max(self._peak, self._in_flight)

    def release(self, num_bytes: int) -> None:
        """Release capacity from the event loop or a completion thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._release_on_loop, num_bytes)

    def _release_on_loop(self, num_bytes: int) -> None:
        self._in_flight -= num_bytes
        self._released.set()

    @contextlib.asynccontextmanager
    async def reserve(self, num_bytes: int):
        await self.acquire(num_bytes)
        try:
            yield
        finally:
            self.release(num_bytes)
