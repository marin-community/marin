# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent asyncio subprocess pipes, with owned, cancellation-safe teardown."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

logger = logging.getLogger(__name__)
_CLEANUPS: set[asyncio.Task[None]] = set()


class ClosableResource(Protocol):
    async def close(self) -> None: ...


async def finish_owned[T](task: asyncio.Task[T]) -> T:
    """Finish an owned operation despite repeated cancellation of its caller.

    The caller must preserve and re-raise its originating error or cancellation.
    Errors from the owned operation still propagate for explicit handling there.
    """
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                break
    return task.result()


def own_cleanup(cleanup: Awaitable[None]) -> asyncio.Task[None]:
    """Keep cleanup alive even if its caller is cancelled repeatedly."""
    task = asyncio.create_task(cleanup)
    _CLEANUPS.add(task)

    def finished(task: asyncio.Task[None]) -> None:
        _CLEANUPS.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Process cleanup failed (%s)", type(error).__name__)

    task.add_done_callback(finished)
    return task


async def await_cleanup(task: asyncio.Task[None]) -> None:
    """Finish cleanup before propagating its caller's original cancellation."""
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        try:
            await finish_owned(task)
        except BaseException as error:
            cancellation.add_note(f"Cleanup failed ({type(error).__name__})")
        raise


async def acquire_owned[T: ClosableResource](acquisition: asyncio.Task[T]) -> T:
    """Own an acquisition, closing any late resource after interruption."""
    try:
        return await asyncio.shield(acquisition)
    except BaseException as error:
        try:
            session = await finish_owned(acquisition)
            await finish_owned(own_cleanup(session.close()))
        except BaseException as cleanup_error:
            if cleanup_error is not error:
                error.add_note(f"Process acquisition cleanup failed ({type(cleanup_error).__name__})")
        raise


class SubprocessSession:
    """A subprocess whose stdout is consumed incrementally by its owner."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        stop: Callable[[], Awaitable[None]],
        stderr_limit: int = 65536,
    ):
        self.process = process
        self._stop = stop
        self._stderr = bytearray()
        self._stderr_limit = stderr_limit
        self._read_lock = asyncio.Lock()
        self._stderr_reader = asyncio.create_task(self._drain_stderr())
        self._closing: asyncio.Task[None] | None = None

    @property
    def stderr(self) -> bytes:
        return bytes(self._stderr)

    async def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        while data := await self.process.stderr.read(65536):
            self._stderr.extend(data[: max(0, self._stderr_limit - len(self._stderr))])

    async def read(self, max_bytes: int = 65536) -> bytes:
        if max_bytes <= 0:
            raise ValueError("Process reads require a positive byte limit")
        assert self.process.stdout is not None
        async with self._read_lock:
            return await self.process.stdout.read(max_bytes)

    async def write(self, data: bytes) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def wait(self) -> int:
        return await self.process.wait()

    async def close(self) -> None:
        if self._closing is None:
            self._closing = own_cleanup(self._close())
        await await_cleanup(self._closing)

    async def _close(self) -> None:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.close()

        async def drain() -> None:
            while await self.read():
                pass

        reader = asyncio.create_task(drain())
        try:
            try:
                await self._stop()
            finally:
                if self.process.returncode is None:
                    self.process.kill()
                await self.process.wait()
                await reader
                await self._stderr_reader
        finally:
            reader.cancel()
            self._stderr_reader.cancel()
            await asyncio.gather(reader, self._stderr_reader, return_exceptions=True)
