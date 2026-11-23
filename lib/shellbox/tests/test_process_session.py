# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import sys

import pytest
from shellbox.process_session import BoundedByteBuffer, SubprocessSession


def test_process_buffer_delivers_buffered_bytes_before_failure():
    async def scenario():
        buffer = BoundedByteBuffer(256)
        payload = bytes(range(256))
        await buffer.append(payload)
        await buffer.finish(ConnectionError("Transport disconnected"))
        received = bytearray()
        for _ in range(len(payload)):
            received.extend(await buffer.read(1))
        assert received == payload
        with pytest.raises(ConnectionError):
            await buffer.read(1)

    asyncio.run(scenario())


def test_process_buffer_rejects_output_beyond_its_capacity_without_waiting():
    async def scenario():
        buffer = BoundedByteBuffer(3)
        await buffer.append(b"abc")
        with pytest.raises(BufferError):
            await asyncio.wait_for(buffer.append(b"d"), 5)
        assert await buffer.read(10) == b"abc"

    asyncio.run(scenario())


def test_subprocess_session_cancelled_close_reaps_child_with_blocked_stdin_and_stdout():
    async def scenario():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import os, signal; os.write(1, b'R'); os.write(1, b'x' * 1048576); signal.pause()",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stopping, release = asyncio.Event(), asyncio.Event()

        async def stop():
            stopping.set()
            await release.wait()
            os.killpg(process.pid, signal.SIGKILL)

        session = SubprocessSession(process, stop=stop)
        writing = closing = None
        try:
            assert await session.read(1) == b"R"
            writing = asyncio.create_task(session.write(b"i" * 1048576))
            closing = asyncio.create_task(session.close())
            await asyncio.wait_for(stopping.wait(), 5)
            assert not writing.done()
            closing.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(closing, 5)
            with pytest.raises((BrokenPipeError, ConnectionResetError)):
                await writing
            assert process.returncode == -signal.SIGKILL
            assert await session.read() == b""
        finally:
            release.set()
            await session.close()
            pending = [task for task in (writing, closing) if task is not None]
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(scenario())
