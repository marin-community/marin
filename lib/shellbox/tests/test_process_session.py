# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import sys

import pytest
from shellbox.process_framing import CHUNK_BYTES, STDIN, STDOUT, FrameDecoder, decode_frame, encode_frame
from shellbox.process_session import BoundedByteBuffer, SubprocessSession


@pytest.mark.parametrize("fragment_bytes", [1, 337, 200000])
def test_process_text_frames_preserve_binary_payload_across_transport_boundaries(fragment_bytes):
    messages = [(STDIN, b"\0\xff\n"), (STDOUT, bytes(range(256)) * (CHUNK_BYTES // 256))]
    wire = b"".join(encode_frame(kind, data) for kind, data in messages)
    decoder = FrameDecoder()
    received = []
    for offset in range(0, len(wire), fragment_bytes):
        received.extend(decode_frame(frame) for frame in decoder.feed(wire[offset : offset + fragment_bytes]))
    decoder.finish()
    assert received == messages


def test_process_buffer_applies_backpressure_and_delivers_bytes_before_failure():
    async def scenario():
        buffer = BoundedByteBuffer(3)
        started = asyncio.Event()
        payload = bytes(range(256))

        async def produce():
            started.set()
            await buffer.feed(payload)
            await buffer.finish(ConnectionError("Transport disconnected"))

        producer = asyncio.create_task(produce())
        await started.wait()
        assert not producer.done()
        received = bytearray()
        for _ in range(len(payload)):
            received.extend(await buffer.read(1))
        assert received == payload
        with pytest.raises(ConnectionError):
            await buffer.read(1)
        await producer

    asyncio.run(scenario())


def test_process_buffer_close_releases_backpressured_producer():
    async def scenario():
        buffer = BoundedByteBuffer(3)
        started = asyncio.Event()

        async def produce():
            started.set()
            await buffer.feed(b"abcdef")

        producer = asyncio.create_task(produce())
        await started.wait()
        assert not producer.done()
        await buffer.finish()
        await asyncio.wait_for(producer, 5)
        assert await buffer.read(10) == b"abc"
        assert await buffer.read(10) == b""

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
