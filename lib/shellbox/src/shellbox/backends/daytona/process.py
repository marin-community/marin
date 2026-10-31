# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded binary pipes over one Daytona command-log websocket."""

import asyncio
import base64
import json
import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path

from daytona import DaytonaNotFoundError, SessionExecuteRequest
from daytona._async.process import AsyncProcess

from shellbox.backends.daytona import bridge
from shellbox.backends.docker.machine import INTERRUPT_TIMEOUT
from shellbox.machine import MachineTerminated
from shellbox.process_session import finish_owned, own_cleanup

BRIDGE_SOURCE = Path(bridge.__file__).read_text()
BUFFER_BYTES = bridge.CHUNK_BYTES * 2


class DaytonaProcessSession:
    """A command whose provider session owns all of its processes."""

    def __init__(
        self,
        process: AsyncProcess,
        name: str,
        *,
        stderr_limit: int,
        release: Callable[["DaytonaProcessSession"], None],
        failed_cleanup: Callable[[], Awaitable[None]],
    ):
        self._process = process
        self.name = name
        self._stderr_limit = stderr_limit
        self._release = release
        self._failed_cleanup = failed_cleanup
        self._command: str | None = None
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._stdout_line = bytearray()
        self._stderr_line = bytearray()
        self._condition = asyncio.Condition()
        self._write_lock = asyncio.Lock()
        self._finished = asyncio.Event()
        self._code: int | None = None
        self._error: BaseException | None = None
        self._logs: asyncio.Task[None] | None = None
        self._closing: asyncio.Task[None] | None = None

    @property
    def stderr(self) -> bytes:
        return bytes(self._stderr)

    async def start(self, argv: tuple[str, ...], cwd: str | None, environment: dict[str, str]) -> None:
        config = base64.b64encode(json.dumps({"argv": argv, "cwd": cwd, "environment": environment}).encode()).decode()
        async with asyncio.timeout(INTERRUPT_TIMEOUT):
            await self._process.create_session(self.name, request_timeout=INTERRUPT_TIMEOUT)
            result = await self._process.execute_session_command(
                self.name,
                SessionExecuteRequest(
                    command=shlex.join(("python3", "-I", "-u", "-c", BRIDGE_SOURCE, config)),
                    run_async=True,
                    suppress_input_echo=True,
                ),
                timeout=INTERRUPT_TIMEOUT,
            )
        self._command = result.cmd_id
        self._logs = asyncio.create_task(self._follow())

    async def _follow(self) -> None:
        assert self._command is not None
        try:
            await self._process.get_session_command_logs_async(
                self.name, self._command, self._receive_stdout, self._receive_stderr
            )
            if self._stdout_line or self._stderr_line or self._code is None:
                raise RuntimeError("Daytona process stream ended without a complete exit frame")
        except asyncio.CancelledError:
            if self._closing is None:
                self._error = MachineTerminated("Daytona process stream was interrupted")
        except Exception as error:
            self._error = error
        finally:
            self._finished.set()
            async with self._condition:
                self._condition.notify_all()
            if self._error is not None and self._closing is None:
                own_cleanup(self.close())

    async def _frames(self, text: str, pending: bytearray, receive: Callable[[bytes], Awaitable[None]]) -> None:
        for offset in range(0, len(text), bridge.FRAME_BYTES):
            pending.extend(text[offset : offset + bridge.FRAME_BYTES].encode("ascii"))
            while (end := pending.find(b"\n")) >= 0:
                if end > bridge.FRAME_BYTES:
                    raise ValueError("Daytona process frame exceeds its limit")
                frame = bytes(pending[:end])
                del pending[: end + 1]
                await receive(frame)
            if len(pending) > bridge.FRAME_BYTES:
                raise ValueError("Daytona process frame exceeds its limit")

    async def _receive_stdout(self, text: str) -> None:
        await self._frames(text, self._stdout_line, self._stdout_frame)

    async def _stdout_frame(self, frame: bytes) -> None:
        if frame.startswith(b"X"):
            if self._code is not None:
                raise ValueError("Duplicate Daytona process exit frame")
            self._code = int(frame[1:])
            return
        if not frame.startswith(b"D") or self._code is not None:
            raise ValueError("Invalid Daytona process output frame")
        data = base64.b64decode(frame[1:], validate=True)
        if len(data) > bridge.CHUNK_BYTES:
            raise ValueError("Daytona process output exceeds its limit")
        async with self._condition:
            await self._condition.wait_for(
                lambda: len(self._stdout) + len(data) <= BUFFER_BYTES or self._closing is not None
            )
            if self._closing is None:
                self._stdout.extend(data)
                self._condition.notify_all()

    async def _receive_stderr(self, text: str) -> None:
        await self._frames(text, self._stderr_line, self._stderr_frame)

    async def _stderr_frame(self, frame: bytes) -> None:
        # Startup errors from the image shell/Python precede bridge framing.
        data = base64.b64decode(frame[1:], validate=True) if frame.startswith(b"E") else frame + b"\n"
        if len(data) > bridge.CHUNK_BYTES:
            raise ValueError("Daytona process stderr exceeds its limit")
        self._stderr.extend(data[: max(0, self._stderr_limit - len(self._stderr))])

    async def read(self, max_bytes: int = 65536) -> bytes:
        if max_bytes <= 0:
            raise ValueError("Process reads require a positive byte limit")
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._stdout) or self._finished.is_set())
            data = bytes(self._stdout[:max_bytes])
            del self._stdout[:max_bytes]
            self._condition.notify_all()
        if not data and self._error is not None:
            raise self._error
        return data

    async def write(self, data: bytes) -> None:
        assert self._command is not None
        async with self._write_lock:
            for offset in range(0, len(data), bridge.CHUNK_BYTES):
                if self._closing is not None or self._finished.is_set():
                    raise MachineTerminated("Daytona process session is closed")
                text = base64.b64encode(data[offset : offset + bridge.CHUNK_BYTES]).decode() + "\n"
                await self._process.send_session_command_input(
                    self.name, self._command, text, request_timeout=INTERRUPT_TIMEOUT
                )

    async def wait(self) -> int:
        await self._finished.wait()
        if self._error is not None:
            raise self._error
        if self._code is None:
            raise MachineTerminated("Daytona process session was closed")
        return self._code

    async def close(self) -> None:
        if self._closing is None:
            self._closing = own_cleanup(self._close())
        try:
            await asyncio.shield(self._closing)
        except asyncio.CancelledError as error:
            try:
                await finish_owned(self._closing)
            except Exception as cleanup_error:
                error.add_note(f"Daytona process cleanup failed ({type(cleanup_error).__name__})")
            raise

    async def _close(self) -> None:
        self._release(self)
        async with self._condition:
            self._condition.notify_all()
        try:
            async with asyncio.timeout(INTERRUPT_TIMEOUT):
                try:
                    await self._process.delete_session(self.name, request_timeout=INTERRUPT_TIMEOUT)
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
            async with self._condition:
                self._condition.notify_all()
