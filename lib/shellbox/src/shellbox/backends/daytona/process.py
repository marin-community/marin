# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded binary pipes over one Daytona command-log websocket."""

import asyncio
import base64
import json
import shlex
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

from daytona import DaytonaNotFoundError, SessionExecuteRequest
from daytona._async.process import AsyncProcess

from shellbox import process_framing
from shellbox.backends.daytona import bridge
from shellbox.backends.docker.machine import INTERRUPT_TIMEOUT
from shellbox.machine import MachineTerminated
from shellbox.process_framing import (
    CHUNK_BYTES,
    EXIT,
    PAD,
    STDERR,
    STDIN,
    STDOUT,
    FrameDecoder,
    decode_frame,
    encode_frame,
)
from shellbox.process_session import BoundedByteBuffer, StreamProcessSession, own_cleanup

# The provider image need not have Shellbox installed. Supply the shared codec
# explicitly before executing the bridge's normal, importable source.
BRIDGE_SOURCE = (
    "import sys, types\n"
    "_package = types.ModuleType('shellbox')\n"
    "_package.__path__ = []\n"
    "sys.modules['shellbox'] = _package\n"
    "_framing = types.ModuleType('shellbox.process_framing')\n"
    f"exec({Path(process_framing.__file__).read_text()!r}, _framing.__dict__)\n"
    "sys.modules['shellbox.process_framing'] = _framing\n" + Path(bridge.__file__).read_text()
)

# Daytona's SDK can pass bytes of its stdout/stderr stream markers through as
# output when a marker ends a websocket message. Frames never contain them.
STREAM_MARKER_BYTES = b"\x01\x02"


class DaytonaProcessSession(StreamProcessSession):
    """A command whose provider session owns all of its processes.

    ``provider_failures`` translates the provider's errors as the machine's commands do, so a lost
    connection raises ``ConnectionError`` and a stopped sandbox ``MachineTerminated``.
    """

    def __init__(
        self,
        process: AsyncProcess,
        name: str,
        *,
        stderr_limit: int,
        release: Callable[["DaytonaProcessSession"], None],
        failed_cleanup: Callable[[], Awaitable[None]],
        provider_failures: Callable[[], AbstractAsyncContextManager[None]],
    ):
        self._process = process
        self._provider_failures = provider_failures
        self.name = name
        self._release = release
        self._failed_cleanup = failed_cleanup
        self._command: str | None = None
        self._output = BoundedByteBuffer(CHUNK_BYTES * 2)
        self._stdout_frames = FrameDecoder()
        self._stderr_lines = FrameDecoder()
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
        config = base64.b64encode(json.dumps({"argv": argv, "cwd": cwd, "environment": environment}).encode()).decode()
        async with asyncio.timeout(INTERRUPT_TIMEOUT), self._provider_failures():
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
            async with self._provider_failures():
                await self._process.get_session_command_logs_async(
                    self.name, self._command, self._receive_stdout, self._receive_stderr
                )
            self._stdout_frames.finish()
            self._stderr_lines.finish()
            if self._code is None:
                raise RuntimeError("Daytona process stream ended without a complete exit frame")
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

    async def _receive_stdout(self, text: str) -> None:
        for frame in self._stdout_frames.feed(text.encode("ascii").translate(None, STREAM_MARKER_BYTES)):
            kind, data = decode_frame(frame)
            if kind == PAD:
                continue
            if self._code is not None:
                raise ValueError("Daytona process output followed its exit frame")
            if kind == EXIT:
                self._code = int(data)
            elif kind == STDOUT:
                await self._output.feed(data)
            elif kind == STDERR:
                self.capture_stderr(data)
            else:
                raise ValueError("Invalid Daytona process output frame")

    async def _receive_stderr(self, text: str) -> None:
        # Only the image shell and Python write here, before the bridge starts.
        for line in self._stderr_lines.feed(text.encode("ascii")):
            self.capture_stderr(line + b"\n")

    async def _send(self, data: bytes) -> None:
        assert self._command is not None
        for offset in range(0, len(data), CHUNK_BYTES):
            if self._closing is not None or self._finished.is_set():
                raise MachineTerminated("Daytona process session is closed")
            text = encode_frame(STDIN, data[offset : offset + CHUNK_BYTES]).decode("ascii")
            async with self._provider_failures():
                await self._process.send_session_command_input(
                    self.name, self._command, text, request_timeout=INTERRUPT_TIMEOUT
                )

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
            await self._output.finish(self._error)
