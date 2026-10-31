# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone primary-container process broker; requires only Python's standard library.

The driver supplies a fresh capability at Pod startup. It grants access only to
processes in this container, never to controller, storage, or provider authority.
"""

import asyncio
import hashlib
import hmac
import json
import os
import signal
import struct
import sys
from collections.abc import Awaitable, Callable

CHUNK_BYTES = 65536
PORT = 10003
MAGIC = b"shellbox-process-v1\0"
HEADER = struct.Struct("!cI")
OPEN, READY, STDIN, EOF, CLOSE, WRITTEN = b"O", b"R", b"I", b"F", b"C", b"W"
STDOUT, STDERR, EXIT, ERROR = b"o", b"e", b"X", b"!"
HANDSHAKE_TIMEOUT = 10
MAX_CONNECTIONS = 32


class ProcessReader(asyncio.StreamReader):
    """Expose transport EOF even while a child's stdin applies backpressure."""

    def __init__(self):
        super().__init__(limit=CHUNK_BYTES)
        self.disconnected = asyncio.Event()

    def feed_eof(self) -> None:
        super().feed_eof()
        self.disconnected.set()

    def set_exception(self, exc: BaseException) -> None:
        super().set_exception(exc)
        self.disconnected.set()


async def start_server(
    connected: Callable[[ProcessReader, asyncio.StreamWriter], Awaitable[None] | None], host: str, port: int
) -> asyncio.AbstractServer:
    loop = asyncio.get_running_loop()
    return await loop.create_server(lambda: asyncio.StreamReaderProtocol(ProcessReader(), connected), host, port)


def proof(capability: bytes, role: bytes, server_nonce: bytes, client_nonce: bytes) -> bytes:
    return hmac.digest(capability, MAGIC + role + server_nonce + client_nonce, hashlib.sha256)


async def read_frame(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    kind, length = HEADER.unpack(await reader.readexactly(HEADER.size))
    if length > CHUNK_BYTES:
        raise ValueError("Process frame exceeds its byte limit")
    return kind, await reader.readexactly(length)


async def write_frame(writer: asyncio.StreamWriter, kind: bytes, data: bytes = b"") -> None:
    if len(data) > CHUNK_BYTES:
        raise ValueError("Process frame exceeds its byte limit")
    writer.write(HEADER.pack(kind, len(data)) + data)
    await writer.drain()


async def authenticate_server(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, capability: bytes) -> None:
    async with asyncio.timeout(HANDSHAKE_TIMEOUT):
        nonce = os.urandom(32)
        writer.write(MAGIC + nonce)
        await writer.drain()
        client_nonce, received = await reader.readexactly(32), await reader.readexactly(32)
        if not hmac.compare_digest(received, proof(capability, b"client", nonce, client_nonce)):
            raise PermissionError("Process channel authentication failed")
        writer.write(proof(capability, b"server", nonce, client_nonce))
        await writer.drain()


async def authenticate_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, capability: bytes) -> None:
    async with asyncio.timeout(HANDSHAKE_TIMEOUT):
        if await reader.readexactly(len(MAGIC)) != MAGIC:
            raise PermissionError("Unknown process channel protocol")
        server_nonce, nonce = await reader.readexactly(32), os.urandom(32)
        writer.write(nonce + proof(capability, b"client", server_nonce, nonce))
        await writer.drain()
        if not hmac.compare_digest(await reader.readexactly(32), proof(capability, b"server", server_nonce, nonce)):
            raise PermissionError("Process channel authentication failed")


async def serve_process(reader: ProcessReader, writer: asyncio.StreamWriter, capability: bytes) -> None:
    process = None
    tasks = []
    finished = False
    writing = asyncio.Lock()

    async def send(kind: bytes, data: bytes = b"") -> None:
        async with writing:
            await write_frame(writer, kind, data)

    async def pump(stream: asyncio.StreamReader, kind: bytes) -> None:
        while data := await stream.read(CHUNK_BYTES):
            await send(kind, data)

    async def input_frames() -> None:
        assert process is not None and process.stdin is not None
        receiving = asyncio.create_task(read_frame(reader))
        draining: asyncio.Task[None] | None = None
        try:
            while True:
                if draining is not None:
                    done, _ = await asyncio.wait((receiving, draining), return_when=asyncio.FIRST_COMPLETED)
                    if draining in done:
                        draining.result()
                        await send(WRITTEN)
                        draining = None
                kind, data = await receiving
                if kind == CLOSE and not data:
                    return
                if draining is not None:
                    raise ValueError("Expected stdin acknowledgment before another input frame")
                if kind == EOF and not data:
                    process.stdin.close()
                elif kind == STDIN:
                    process.stdin.write(data)
                    # Keep reading CLOSE while stdin is blocked. Acknowledging
                    # only drained chunks bounds the sender to one pending chunk.
                    draining = asyncio.create_task(process.stdin.drain())
                else:
                    raise ValueError("Unexpected process input frame")
                receiving = asyncio.create_task(read_frame(reader))
        finally:
            pending = [receiving] if draining is None else [receiving, draining]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def output_frames() -> None:
        assert process is not None and process.stdout is not None and process.stderr is not None
        await asyncio.gather(pump(process.stdout, STDOUT), pump(process.stderr, STDERR), process.wait())

    try:
        await authenticate_server(reader, writer, capability)
        async with asyncio.timeout(HANDSHAKE_TIMEOUT):
            kind, data = await read_frame(reader)
            if kind != OPEN:
                raise ValueError("Expected process open frame")
            command = json.loads(data)
            process = await asyncio.create_subprocess_exec(
                *command["argv"],
                cwd=command["cwd"],
                env={**os.environ, **command["env"]},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=CHUNK_BYTES,
            )
        await send(READY)
        incoming, outgoing = asyncio.create_task(input_frames()), asyncio.create_task(output_frames())
        disconnected = asyncio.create_task(reader.disconnected.wait())
        tasks = [incoming, outgoing, disconnected]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if disconnected in done:
            raise ConnectionError("Process driver disconnected")
        if incoming in done:
            # Disconnect and explicit close reclaim the entire command process group.
            # A successful foreground command may intentionally leave a service behind.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await outgoing
            incoming.result()
        else:
            outgoing.result()
        await send(EXIT, struct.pack("!i", process.returncode))
        finished = True
    except (ConnectionError, asyncio.IncompleteReadError, BrokenPipeError):
        pass
    except Exception as error:
        # Do not expose command contents or the capability in diagnostics.
        try:
            await send(ERROR, type(error).__name__.encode())
        except (ConnectionError, OSError):
            pass
    finally:
        if process is not None and not finished:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def serve(capability: bytes, host: str, port: int, workdir: str) -> None:
    os.makedirs(workdir, exist_ok=True)
    connections: set[asyncio.Task] = set()

    def accepted(reader: ProcessReader, writer: asyncio.StreamWriter) -> None:
        if len(connections) >= MAX_CONNECTIONS:
            writer.close()
            return
        task = asyncio.create_task(serve_process(reader, writer, capability))
        connections.add(task)
        task.add_done_callback(connections.discard)

    server = await start_server(accepted, host, port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve(bytes.fromhex(sys.argv[1]), "0.0.0.0", int(sys.argv[2]), sys.argv[3]))
