# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded process pipes over a direct, authenticated primary-container socket."""

import asyncio
import json
import struct
from collections.abc import Callable

from shellbox.backends.iris.process_guest import (
    CHUNK_BYTES,
    CLOSE,
    EOF,
    ERROR,
    EXIT,
    OPEN,
    READY,
    STDERR,
    STDIN,
    STDOUT,
    WRITTEN,
    authenticate_client,
    read_frame,
    write_frame,
)
from shellbox.machine import Command, MachineTerminated
from shellbox.process_session import await_cleanup, own_cleanup

BUFFER_CHUNKS = 16
CONNECT_TIMEOUT = 30
CLOSE_TIMEOUT = 10


class IrisProcessSession:
    """One process with bounded stdout and independently drained stderr."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        stderr_limit: int,
        on_close: Callable[["IrisProcessSession"], None],
    ):
        self._reader = reader
        self._writer = writer
        self._stderr_limit = stderr_limit
        self._on_close = on_close
        self._stderr = bytearray()
        self._stdout: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=BUFFER_CHUNKS)
        self._remaining = b""
        self._reading = asyncio.Lock()
        self._writing = asyncio.Lock()
        self._input_written = asyncio.Event()
        self._acknowledged = False
        self._exit_code: int | None = None
        self._error: Exception | None = None
        self._closing: asyncio.Task[None] | None = None
        self._receiver = asyncio.create_task(self._receive())

    @classmethod
    async def connect(
        cls,
        host: str,
        port: int,
        capability: bytes,
        command: Command,
        *,
        on_close: Callable[["IrisProcessSession"], None],
    ) -> "IrisProcessSession":
        async with asyncio.timeout(CONNECT_TIMEOUT):
            reader, writer = await asyncio.open_connection(host, port, limit=CHUNK_BYTES)
            try:
                await authenticate_client(reader, writer, capability)
                await write_frame(
                    writer,
                    OPEN,
                    json.dumps({"argv": command.argv, "cwd": command.cwd, "env": command.env}).encode(),
                )
                kind, data = await read_frame(reader)
                if kind != READY or data:
                    raise RuntimeError("Iris process did not acknowledge startup")
                return cls(reader, writer, stderr_limit=command.output_limit_bytes, on_close=on_close)
            except BaseException as error:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError) as cleanup_error:
                    error.add_note(f"Iris socket cleanup failed ({type(cleanup_error).__name__})")
                raise

    @property
    def stderr(self) -> bytes:
        return bytes(self._stderr)

    async def _receive(self) -> None:
        try:
            while True:
                kind, data = await read_frame(self._reader)
                if kind == STDERR:
                    self._stderr.extend(data[: max(0, self._stderr_limit - len(self._stderr))])
                elif kind == STDOUT:
                    if data:
                        await self._stdout.put(data)
                elif kind == WRITTEN and not data:
                    self._acknowledged = True
                    self._input_written.set()
                elif kind == EXIT:
                    self._exit_code = struct.unpack("!i", data)[0]
                    return
                elif kind == ERROR:
                    raise RuntimeError("Iris process transport failed")
                else:
                    raise RuntimeError("Unexpected Iris process output frame")
        except Exception as error:
            self._error = error
        finally:
            self._input_written.set()
            try:
                self._stdout.put_nowait(None)
            except asyncio.QueueFull:
                # Readers drain queued bytes before observing receiver completion.
                pass

    def _completed(self) -> bytes:
        if self._error is not None:
            raise self._error
        if self._exit_code is None:
            raise MachineTerminated("Iris process session is closed")
        return b""

    async def read(self, max_bytes: int = CHUNK_BYTES) -> bytes:
        if max_bytes <= 0:
            raise ValueError("Process reads require a positive byte limit")
        async with self._reading:
            if not self._remaining:
                if self._stdout.empty() and self._receiver.done():
                    return self._completed()
                self._remaining = await self._stdout.get() or b""
                if not self._remaining:
                    return self._completed()
            chunk, self._remaining = self._remaining[:max_bytes], self._remaining[max_bytes:]
            return chunk

    async def write(self, data: bytes) -> None:
        async with self._writing:
            if self._closing is not None or self._receiver.done():
                raise MachineTerminated("Iris process session is closed")
            for offset in range(0, len(data), CHUNK_BYTES):
                self._acknowledged = False
                self._input_written.clear()
                await write_frame(self._writer, STDIN, data[offset : offset + CHUNK_BYTES])
                await self._input_written.wait()
                if not self._acknowledged:
                    self._completed()
                    raise MachineTerminated("Iris process exited before accepting stdin")

    async def write_eof(self) -> None:
        """Finish the stdin of a finite command without closing its output pipes."""
        async with self._writing:
            await write_frame(self._writer, EOF)

    async def wait(self) -> int:
        await asyncio.shield(self._receiver)
        self._completed()
        assert self._exit_code is not None
        return self._exit_code

    async def close(self) -> None:
        if self._closing is None:
            self._closing = own_cleanup(self._close())
        await await_cleanup(self._closing)

    async def _close(self) -> None:
        self._receiver.cancel()
        await asyncio.gather(self._receiver, return_exceptions=True)
        try:
            if self._exit_code is None:
                # Releasing stdout backpressure lets the guest acknowledge process
                # group reclamation even when the caller stopped reading output.
                async with asyncio.timeout(CLOSE_TIMEOUT):
                    async with self._writing:
                        await write_frame(self._writer, CLOSE)
                    while True:
                        kind, data = await read_frame(self._reader)
                        if kind == EXIT:
                            self._exit_code = struct.unpack("!i", data)[0]
                            break
                        if kind == ERROR:
                            raise RuntimeError("Iris process cleanup failed")
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            # A disconnected broker reclaims the process, or its Pod has exited.
            pass
        finally:
            self._writer.close()
            try:
                try:
                    await self._writer.wait_closed()
                except (ConnectionError, OSError):
                    pass
            finally:
                self._on_close(self)
