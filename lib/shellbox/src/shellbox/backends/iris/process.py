# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded asynchronous process pipes through an Iris exec gateway."""

import asyncio
from collections.abc import Callable

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.rpc import exec_gateway_pb2
from iris.rpc.exec_gateway_connect import ExecGatewayServiceClient

from shellbox.machine import MachineTerminated
from shellbox.process_session import await_cleanup, own_cleanup

CHUNK_BYTES = 65536
BUFFER_CHUNKS = 16
RECEIVE_WAIT_MS = 20000
RPC_TIMEOUT_MS = 30000


class IrisProcessSession:
    """One gateway process, with bounded stdout and independently drained stderr."""

    def __init__(
        self,
        client: ExecGatewayServiceClient,
        grant: str,
        session_id: str,
        *,
        stderr_limit: int,
        on_close: Callable[["IrisProcessSession"], None],
    ):
        self._client = client
        self._grant = grant
        self._session_id = session_id
        self._stderr_limit = stderr_limit
        self._on_close = on_close
        self._stderr = bytearray()
        self._stdout: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=BUFFER_CHUNKS)
        self._remaining = b""
        self._reading = asyncio.Lock()
        self._writing = asyncio.Lock()
        self._exit_code: int | None = None
        self._error: Exception | None = None
        self._closing: asyncio.Task[None] | None = None
        self._receiver = asyncio.create_task(self._receive())

    @property
    def stderr(self) -> bytes:
        return bytes(self._stderr)

    async def _receive(self) -> None:
        try:
            while True:
                response = await self._client.receive(
                    exec_gateway_pb2.ReceiveRequest(
                        grant=self._grant, session_id=self._session_id, max_bytes=CHUNK_BYTES, wait_ms=RECEIVE_WAIT_MS
                    ),
                    timeout_ms=RPC_TIMEOUT_MS,
                )
                if response.error:
                    raise RuntimeError("Iris process transport failed")
                if len(response.stdout) + len(response.stderr) > CHUNK_BYTES:
                    raise RuntimeError("Iris process output chunk exceeds its limit")
                self._stderr.extend(response.stderr[: max(0, self._stderr_limit - len(self._stderr))])
                if response.stdout:
                    await self._stdout.put(response.stdout)
                if response.exited:
                    self._exit_code = response.exit_code
                    return
        except Exception as error:
            self._error = error
        finally:
            try:
                self._stdout.put_nowait(None)
            except asyncio.QueueFull:
                # A reader drains queued data before observing receiver completion.
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
                await self._client.send(
                    exec_gateway_pb2.SendRequest(
                        grant=self._grant, session_id=self._session_id, stdin=data[offset : offset + CHUNK_BYTES]
                    ),
                    timeout_ms=RPC_TIMEOUT_MS,
                )

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
            try:
                await self._client.close_session(
                    exec_gateway_pb2.CloseRequest(grant=self._grant, session_id=self._session_id),
                    timeout_ms=RPC_TIMEOUT_MS,
                )
            except ConnectError as error:
                if error.code != Code.NOT_FOUND:
                    raise
        finally:
            try:
                await self._client.close()
            finally:
                self._on_close(self)
