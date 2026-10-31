# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent binary process streams with owned, cancellation-safe teardown."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from shellbox.machine import MachineTerminated

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


class ByteReader(Protocol):
    async def read(self, max_bytes: int, /) -> bytes: ...


class BoundedByteBuffer:
    """Adapt pushed transport output to a backpressured binary reader."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Process buffers require positive capacity")
        self._capacity = capacity
        self._data = bytearray()
        self._condition = asyncio.Condition()
        self._finished = False
        self._error: BaseException | None = None

    async def feed(self, data: bytes) -> None:
        offset = 0
        async with self._condition:
            while offset < len(data):
                await self._condition.wait_for(lambda: len(self._data) < self._capacity or self._finished)
                if self._finished:
                    return
                size = min(len(data) - offset, self._capacity - len(self._data))
                self._data.extend(data[offset : offset + size])
                offset += size
                self._condition.notify_all()

    async def read(self, max_bytes: int) -> bytes:
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._data) or self._finished)
            data = bytes(self._data[:max_bytes])
            del self._data[:max_bytes]
            self._condition.notify_all()
        if not data and self._error is not None:
            raise self._error
        return data

    async def finish(self, error: BaseException | None = None) -> None:
        """Release readers and producers, delivering buffered bytes before failure."""
        async with self._condition:
            self._finished = True
            if error is not None:
                self._error = error
            self._condition.notify_all()


class StreamProcessSession:
    """Process-session semantics independent of pipe, SDK, or socket transport.

    The transport supplies one reader and write, wait, and stop operations.
    Stopping must release pending reads and writes and complete the wait operation.
    Raw pipe readers keep their existing buffering; push transports can use
    ``BoundedByteBuffer``. Framing belongs to the transport adapter.
    """

    def __init__(
        self,
        stdout: ByteReader,
        *,
        write: Callable[[bytes], Awaitable[None]],
        wait: Callable[[], Awaitable[int]],
        stop: Callable[[], Awaitable[None]],
        stderr: ByteReader | None = None,
        stderr_limit: int = 65536,
    ):
        self._stdout = stdout
        self._write = write
        self._wait = wait
        self._stop = stop
        self._stderr = bytearray()
        self._stderr_limit = stderr_limit
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._stderr_reader = asyncio.create_task(self._drain_stderr(stderr)) if stderr is not None else None
        self._closing: asyncio.Task[None] | None = None

    @property
    def stderr(self) -> bytes:
        return bytes(self._stderr)

    def capture_stderr(self, data: bytes) -> None:
        """Retain bounded diagnostics from a transport's independently drained stderr."""
        self._stderr.extend(data[: max(0, self._stderr_limit - len(self._stderr))])

    async def _drain_stderr(self, stderr: ByteReader) -> None:
        while data := await stderr.read(65536):
            self.capture_stderr(data)

    async def read(self, max_bytes: int = 65536) -> bytes:
        if max_bytes <= 0:
            raise ValueError("Process reads require a positive byte limit")
        async with self._read_lock:
            return await self._stdout.read(max_bytes)

    async def write(self, data: bytes) -> None:
        async with self._write_lock:
            if self._closing is not None:
                raise MachineTerminated("Process session is closed")
            await self._write(data)

    async def wait(self) -> int:
        return await self._wait()

    async def close(self) -> None:
        if self._closing is None:
            self._closing = own_cleanup(self._close())
        await await_cleanup(self._closing)

    async def _close(self) -> None:
        try:
            await self._stop()
        finally:
            if self._stderr_reader is not None:
                try:
                    await self._stderr_reader
                finally:
                    self._stderr_reader.cancel()
                    await asyncio.gather(self._stderr_reader, return_exceptions=True)


class SubprocessSession(StreamProcessSession):
    """Use native subprocess pipes without adding another stdout buffer."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        stop: Callable[[], Awaitable[None]],
        stderr_limit: int = 65536,
    ):
        assert process.stdout is not None and process.stderr is not None
        self.process = process
        self._stop_process = stop
        super().__init__(
            process.stdout,
            write=self._write_pipe,
            wait=process.wait,
            stop=self._stop_subprocess,
            stderr=process.stderr,
            stderr_limit=stderr_limit,
        )

    async def _write_pipe(self, data: bytes) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def drain_output(self) -> None:
        """Discard stdout and wait for independently drained stderr to reach EOF."""
        while await self.read():
            pass
        assert self._stderr_reader is not None
        await asyncio.shield(self._stderr_reader)

    async def _stop_subprocess(self) -> None:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.close()
        reader = asyncio.create_task(self.drain_output())
        try:
            try:
                await self._stop_process()
            finally:
                if self.process.returncode is None:
                    self.process.kill()
                await self.process.wait()
                await reader
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
