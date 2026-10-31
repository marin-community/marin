# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real subprocess execution behind the gateway's external transport boundary."""

import asyncio
import os
import signal

from iris.cluster.exec_gateway.auth import ExecPodTarget
from iris.cluster.exec_gateway.service import MAX_CHUNK_BYTES, ExecOutput, ExecTransport


class ProcessExecConnector:
    """Exercise gateway lifetime and binary I/O without Kubernetes."""

    def __init__(self):
        self.processes: list[asyncio.subprocess.Process] = []
        self.targets: list[ExecPodTarget] = []

    async def open(self, target: ExecPodTarget, command: tuple[str, ...]) -> ExecTransport:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self.processes.append(process)
        self.targets.append(target)
        return _ProcessExecTransport(process)

    async def close(self) -> None:
        for process in self.processes:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()


class _ProcessExecTransport:
    def __init__(self, process: asyncio.subprocess.Process):
        self._process = process
        self._output: asyncio.Queue[ExecOutput] = asyncio.Queue(maxsize=4)
        assert process.stdout is not None and process.stderr is not None
        self._readers = [
            asyncio.create_task(self._drain(process.stdout, False)),
            asyncio.create_task(self._drain(process.stderr, True)),
        ]
        self._completion = asyncio.create_task(self._complete())

    async def _drain(self, stream: asyncio.StreamReader, stderr: bool) -> None:
        while chunk := await stream.read(MAX_CHUNK_BYTES):
            await self._output.put(ExecOutput(stderr=chunk) if stderr else ExecOutput(stdout=chunk))

    async def _complete(self) -> None:
        code = await self._process.wait()
        await asyncio.gather(*self._readers)
        await self._output.put(ExecOutput(exit_code=code))

    async def send(self, data: bytes, *, eof: bool) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(data)
        await self._process.stdin.drain()
        if eof:
            self._process.stdin.close()

    async def receive(self) -> ExecOutput:
        return await self._output.get()

    async def close(self) -> None:
        for task in [*self._readers, self._completion]:
            task.cancel()
        await asyncio.gather(*self._readers, self._completion, return_exceptions=True)
        if self._process.returncode is None:
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                # The process may have exited before asyncio reaped its status.
                pass
        # Drain pipe buffers after cancelling readers: wait() alone can hang
        # while a backpressured subprocess pipe is paused.
        await self._process.communicate()


class PausedCloseConnector(ProcessExecConnector):
    """Expose a delayed external transport close for cancellation regressions."""

    def __init__(self):
        super().__init__()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def open(self, target: ExecPodTarget, command: tuple[str, ...]) -> ExecTransport:
        return _PausedCloseTransport(await super().open(target, command), self.close_started, self.release_close)


class _PausedCloseTransport:
    def __init__(self, transport: ExecTransport, started: asyncio.Event, release: asyncio.Event):
        self._transport = transport
        self._started = started
        self._release = release

    async def receive(self) -> ExecOutput:
        return await self._transport.receive()

    async def send(self, data: bytes, *, eof: bool) -> None:
        await self._transport.send(data, eof=eof)

    async def close(self) -> None:
        self._started.set()
        await self._release.wait()
        await self._transport.close()
