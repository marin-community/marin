# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded binary pipes over one Daytona v2 process."""

import asyncio
import base64
import signal
from collections.abc import Awaitable, Callable

from daytona import DaytonaConnectionError, DaytonaNotFoundError
from daytona.v2 import AsyncProcessHandle, AsyncProcessService, WaitResult
from rigging.timing import ExponentialBackoff

from shellbox.backends.docker.machine import INTERRUPT_TIMEOUT
from shellbox.machine import MachineTerminated
from shellbox.process_session import BoundedByteBuffer, StreamProcessSession, own_cleanup

# The log stream keeps flowing while the reader pauses, so the provider's bounded per-process
# log is never left to evict unread frames and, should HTTP/2 come back, a paused stream cannot
# stall the sandbox's other requests. The session fails once this much stdout is unread.
OUTPUT_BUFFER_BYTES = 64 * 1024 * 1024
STDIN_CHUNK_BYTES = 64 * 1024
# A followed log can end before its eof; it resumes from the last cursor this many times in a row.
STREAM_RESUMES = 5


def decode_frame_data(data: str) -> bytes:
    """Decode a base64 log frame.

    The API sends URL-safe base64 without padding, although the SDK documents
    padded standard base64; this accepts both.
    """
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def exit_status(result: WaitResult) -> int:
    """Return the exit code, or the negative signal number like ``subprocess``."""
    if result.reason == "exited" and result.exit_code is not None:
        return result.exit_code
    if result.reason == "signaled" and result.signal is not None:
        return -signal.Signals[f"SIG{result.signal}"].value
    raise RuntimeError(f"Daytona process ended without an exit status: {result}")


class DaytonaProcessSession(StreamProcessSession):
    """A provider process whose tag owns every process it starts."""

    def __init__(
        self,
        process: AsyncProcessService,
        name: str,
        *,
        stderr_limit: int,
        release: Callable[["DaytonaProcessSession"], None],
        failed_cleanup: Callable[[], Awaitable[None]],
    ):
        self._process = process
        self.name = name
        self._release = release
        self._failed_cleanup = failed_cleanup
        self._handle: AsyncProcessHandle | None = None
        self._output = BoundedByteBuffer(OUTPUT_BUFFER_BYTES)
        self._finished = asyncio.Event()
        self._code: int | None = None
        self._error: BaseException | None = None
        self._logs: asyncio.Task[None] | None = None
        super().__init__(
            self._output,
            write=self._send,
            wait=self._wait_provider,
            stop=self._stop_provider,
            stderr_limit=stderr_limit,
        )

    async def start(self, argv: tuple[str, ...], cwd: str | None, environment: dict[str, str]) -> None:
        async with asyncio.timeout(INTERRUPT_TIMEOUT):
            self._handle = await self._process.start(argv=argv, cwd=cwd, env=environment, stdin="open", tag=self.name)
        self._logs = asyncio.create_task(self._follow())

    async def _follow(self) -> None:
        assert self._handle is not None
        cursor: str | None = None
        resumes = 0
        backoff = ExponentialBackoff(initial=0.5, maximum=4, factor=2, jitter=0.5)
        try:
            while True:
                try:
                    async for event in self._handle.stream_logs(cursor=cursor, encoding="base64"):
                        if event.type == "log":
                            data = decode_frame_data(event.frame.data)
                            if event.frame.channel == "stdout":
                                await self._output.append(data)
                            else:
                                self.capture_stderr(data)
                        elif event.type == "warning":
                            raise RuntimeError(f"Daytona dropped process output: {event.message}")
                        elif event.type == "state":
                            self._code = exit_status(event.result)
                        # Advance only past delivered events, so a resumed stream replays nothing twice.
                        cursor, resumes = event.cursor, 0
                    break
                except DaytonaConnectionError:
                    resumes += 1
                    if resumes > STREAM_RESUMES:
                        raise
                    await asyncio.sleep(backoff.next_interval())
            if self._code is None:
                raise RuntimeError("Daytona process stream ended without an exit status")
        except asyncio.CancelledError:
            if self._closing is None:
                self._error = MachineTerminated("Daytona process stream was interrupted")
        except Exception as error:
            self._error = error
        finally:
            self._finished.set()
            await self._output.finish(self._error)
            if self._error is not None and self._closing is None:
                own_cleanup(self.close())

    async def _send(self, data: bytes) -> None:
        assert self._handle is not None
        for offset in range(0, len(data), STDIN_CHUNK_BYTES):
            if self._closing is not None or self._finished.is_set():
                raise MachineTerminated("Daytona process session is closed")
            async with asyncio.timeout(INTERRUPT_TIMEOUT):
                await self._handle.write_stdin(data[offset : offset + STDIN_CHUNK_BYTES])

    async def _wait_provider(self) -> int:
        await self._finished.wait()
        if self._error is not None:
            raise self._error
        if self._code is None:
            raise MachineTerminated("Daytona process session was closed")
        return self._code

    async def _stop_provider(self) -> None:
        self._release(self)
        await self._output.finish()
        try:
            async with asyncio.timeout(INTERRUPT_TIMEOUT):
                # The tag also reaches a process whose start request never answered, and every
                # descendant it started, including detached ones.
                await self._process.kill_all(tag=self.name, timeout=INTERRUPT_TIMEOUT / 2)
                if self._handle is not None:
                    try:
                        await self._handle.delete()
                    except DaytonaNotFoundError:
                        pass
        except BaseException:
            await self._failed_cleanup()
            raise
        finally:
            if self._logs is not None:
                self._logs.cancel()
                await asyncio.gather(self._logs, return_exceptions=True)
            self._finished.set()
            await self._output.finish(self._error)
