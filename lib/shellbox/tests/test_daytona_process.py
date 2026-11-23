# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real child pipes behind Daytona's v2 process API boundary."""

import asyncio
import base64
import json
import os
import signal
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("daytona")

from daytona.v2 import ConnectionError as ProviderConnectionError
from daytona.v2 import WaitResult
from shellbox.backends.daytona.machine import DaytonaMachine
from shellbox.backends.daytona.process import decode_frame_data
from shellbox.http_service import HttpResponse, serve_http
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec

# The sandbox's own environment; the host's variables never reach it.
SANDBOX_PATH = f"{Path(sys.executable).parent}{os.pathsep}{os.defpath}"


def wire_frame(data: bytes) -> str:
    """Encode a log frame as the API sends it: URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def wait_result(returncode: int) -> WaitResult:
    if returncode < 0:
        return WaitResult(reason="signaled", signal=signal.Signals(-returncode).name.removeprefix("SIG"))
    return WaitResult(reason="exited", exit_code=returncode)


class ProviderProcess:
    """One real child whose pipes are recorded as a replayable v2 process log."""

    def __init__(self, child: asyncio.subprocess.Process):
        self.child = child
        self.events: list[SimpleNamespace] = []
        self.changed = asyncio.Condition()
        self.finished = False
        self.pump = asyncio.create_task(self._pump())

    async def _append(self, kind: str, **fields) -> None:
        async with self.changed:
            self.events.append(SimpleNamespace(type=kind, cursor=str(len(self.events)), **fields))
            self.changed.notify_all()

    async def _read(self, pipe, channel: str) -> None:
        while data := await pipe.read(337):
            await self._append("log", frame=SimpleNamespace(channel=channel, data=wire_frame(data)))

    async def _pump(self) -> None:
        await asyncio.gather(self._read(self.child.stdout, "stdout"), self._read(self.child.stderr, "stderr"))
        await self._append("state", result=wait_result(await self.child.wait()))
        async with self.changed:
            self.finished = True
            self.changed.notify_all()

    async def kill(self) -> None:
        if self.child.returncode is None:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await self.child.wait()
        await self.pump


class ProcessHandle:
    def __init__(self, api: "ProcessAPI", process: ProviderProcess, tag: str):
        self.api = api
        self.process = process
        self.tag = tag

    async def stream_logs(self, *, cursor=None, encoding):
        assert encoding == "base64"
        index = 0 if cursor is None else int(cursor) + 1
        while True:
            async with self.process.changed:
                await self.process.changed.wait_for(
                    lambda position=index: position < len(self.process.events) or self.process.finished
                )
                if index >= len(self.process.events):
                    break
                event = self.process.events[index]
            if self.api.cut_at == index:
                self.api.cut_at = None
                raise ProviderConnectionError("process log stream ended before eof")
            if self.api.warning_at == index:
                yield SimpleNamespace(
                    type="warning", message="frames were evicted", cursor=event.cursor, first_available_cursor="9"
                )
            if not (self.api.drop_state and event.type == "state"):
                yield event
            index += 1
        yield SimpleNamespace(type="eof", cursor=str(index))

    async def write_stdin(self, data: bytes) -> None:
        self.process.child.stdin.write(data)
        await self.process.child.stdin.drain()

    async def delete(self) -> None:
        self.api.deleted.append(self.tag)


class ProcessAPI:
    """The v2 process service of one sandbox, backed by real child processes."""

    def __init__(self):
        self.processes: dict[str, ProviderProcess] = {}
        self.children = []
        self.started = []
        self.killed = []
        self.deleted = []
        self.hold_start = False
        self.accepted = asyncio.Event()
        self.release = asyncio.Event()
        self.killing = asyncio.Event()
        self.kill_release = asyncio.Event()
        self.kill_release.set()
        self.cut_at: int | None = None
        self.warning_at: int | None = None
        self.drop_state = False

    async def start(self, *, argv, cwd, env, stdin, tag):
        assert stdin == "open"
        self.started.append(tag)
        child = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env={"PATH": SANDBOX_PATH, "HOME": "/tmp", **env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self.children.append(child)
        process = ProviderProcess(child)
        self.processes[tag] = process
        if self.hold_start:
            # The provider started the process, but its answer has not arrived yet.
            self.accepted.set()
            await self.release.wait()
        return ProcessHandle(self, process, tag)

    async def kill_all(self, *, tag, timeout):
        self.killing.set()
        await self.kill_release.wait()
        self.killed.append(tag)
        process = self.processes.pop(tag, None)
        if process is not None:
            await process.kill()
        return []

    async def close(self):
        self.release.set()
        self.kill_release.set()
        for tag in tuple(self.processes):
            await self.kill_all(tag=tag, timeout=10)


@pytest.fixture
def session_machine(tmp_path):
    api = ProcessAPI()
    resources = AsyncExitStack()
    resources.push_async_callback(api.close)
    machine = DaytonaMachine(
        SimpleNamespace(process=api),
        MachineSpec(RegistryImage("fixture"), workdir=str(tmp_path), env={"TAG": "spec"}),
        resources,
    )
    return machine, api


async def output(session):
    result = bytearray()
    while chunk := await session.read(7919):
        result.extend(chunk)
    return bytes(result)


def test_daytona_log_frames_decode_with_or_without_padding() -> None:
    data = b"\xfb\xff\x00binary"
    # The API sends URL-safe frames without padding; the SDK documents standard padded base64.
    assert decode_frame_data(wire_frame(data)) == data
    assert decode_frame_data(base64.b64encode(data).decode()) == data


def test_daytona_process_preserves_binary_pipes_defaults_and_bounded_stderr(session_machine, tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-only-canary")
    directory = tmp_path / "command-directory"
    directory.mkdir()
    data = bytes(range(256)) * 1024
    script = """import json, os, sys
sys.stderr.buffer.write(bytes(range(256)) * 1024)
sys.stderr.buffer.flush()
print(json.dumps([os.getcwd(), os.environ['TAG'], os.environ.get('AWS_SECRET_ACCESS_KEY')]), flush=True)
data = sys.stdin.buffer.read(int(sys.argv[1]))
sys.stdout.buffer.write(data[::-1])
sys.stdout.buffer.flush()
sys.exit(7)
"""

    async def scenario():
        machine, api = session_machine
        try:
            session = await machine.open_process(
                Command(
                    (sys.executable, "-c", script, str(len(data))),
                    cwd=str(directory),
                    env={"TAG": "command"},
                    output_limit_bytes=37,
                )
            )
            received = asyncio.create_task(output(session))
            await session.write(data)
            header, body = (await asyncio.wait_for(received, 10)).split(b"\n", 1)
            assert json.loads(header) == [str(directory.resolve()), "command", None]
            assert body == data[::-1]
            assert await session.wait() == 7
            assert session.stderr == bytes(range(37))
            await session.close()
            assert api.killed == api.started and not api.processes
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_daytona_process_resumes_a_stream_cut_short_from_its_last_cursor(session_machine):
    # Enough output for several log frames, so the cut falls inside the stream.
    script = "for index in range(50):\n    print(str(index) * 100, flush=True)\n"

    async def scenario():
        machine, api = session_machine
        api.cut_at = 5
        try:
            session = await machine.open_process(Command((sys.executable, "-c", script)))
            received = await asyncio.wait_for(output(session), 10)
            assert received == "".join(f"{str(index) * 100}\n" for index in range(50)).encode()
            assert await session.wait() == 0
            assert api.cut_at is None
            await session.close()
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_daytona_process_reports_a_signal_as_a_negative_exit_status(session_machine):
    async def scenario():
        machine, _ = session_machine
        try:
            session = await machine.open_process(
                Command((sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"))
            )
            assert await asyncio.wait_for(session.wait(), 10) == -signal.SIGTERM
            await session.close()
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["evicted_output", "missing_exit_status"])
def test_daytona_process_fails_an_incomplete_provider_stream_and_stops_child(session_machine, failure):
    async def scenario():
        machine, api = session_machine
        if failure == "evicted_output":
            api.warning_at = 0
        else:
            api.drop_state = True
        try:
            script = "import signal; print('ready', flush=True); signal.pause()"
            if failure == "missing_exit_status":
                script = "print('done')"
            session = await machine.open_process(Command((sys.executable, "-c", script)))
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(output(session), 10)
            await session.close()
            assert not api.processes
            assert all(child.returncode is not None for child in api.children)
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_daytona_process_drains_the_provider_stream_while_the_reader_is_idle(session_machine):
    # The provider multiplexes the log stream with stdin writes on one HTTP/2 connection,
    # so a stream that waits for the reader would stall the whole sandbox connection.
    script = "import sys; sys.stdout.buffer.write(b'x' * (1 << 20))"

    async def scenario():
        machine, _ = session_machine
        try:
            session = await machine.open_process(Command((sys.executable, "-c", script)))
            assert await asyncio.wait_for(session.wait(), 10) == 0
            assert await output(session) == b"x" * (1 << 20)
            await session.close()
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_daytona_process_fails_once_unread_output_exceeds_its_buffer(session_machine, monkeypatch):
    monkeypatch.setattr("shellbox.backends.daytona.process.OUTPUT_BUFFER_BYTES", 4096)

    async def scenario():
        machine, api = session_machine
        try:
            session = await machine.open_process(
                Command(
                    (
                        sys.executable,
                        "-c",
                        "import signal, sys; sys.stdout.write('x' * 65536); sys.stdout.flush(); signal.pause()",
                    )
                )
            )
            with pytest.raises(BufferError):
                await asyncio.wait_for(session.wait(), 10)
            await session.close()
            assert all(child.returncode is not None for child in api.children)
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_daytona_http_capability_uses_one_persistent_provider_process(session_machine, unused_tcp_port):
    async def scenario():
        machine, api = session_machine
        blocked, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            assert request.method == "POST"
            if request.path == "/blocked":
                blocked.set()
                await release.wait()
            else:
                assert request.path == "/binary"
            return HttpResponse(201, request.body[::-1], "application/octet-stream")

        try:
            async with (
                asyncio.TaskGroup() as service_tasks,
                serve_http(machine, handler, task_group=service_tasks, port=unused_tcp_port) as endpoint,
            ):
                async with httpx.AsyncClient(trust_env=False) as client:
                    slow = asyncio.create_task(client.post(endpoint + "/blocked", content=b"slow"))
                    try:
                        await asyncio.wait_for(blocked.wait(), 5)
                        for body in (b"\0\xff\xfe", bytes(range(256)) * 1024):
                            response = await client.post(endpoint + "/binary", content=body)
                            assert (response.status_code, response.content) == (201, body[::-1])
                        release.set()
                        assert (await slow).content == b"wols"
                    finally:
                        release.set()
                        if not slow.done():
                            slow.cancel()
                        await asyncio.gather(slow, return_exceptions=True)
            assert len(api.started) == 1
            assert not api.processes
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_daytona_process_cancelled_acquisition_kills_the_accepted_process_by_tag(session_machine):
    async def scenario():
        machine, api = session_machine
        api.hold_start = True
        api.kill_release.clear()
        opening = asyncio.create_task(
            machine.open_process(Command((sys.executable, "-c", "import signal; signal.pause()")))
        )
        try:
            await asyncio.wait_for(api.accepted.wait(), 5)
            opening.cancel()
            api.release.set()
            await asyncio.wait_for(api.killing.wait(), 5)
            opening.cancel()
            api.kill_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(opening, 5)
            # The start request never answered, yet its tag reaches the process.
            assert api.killed == api.started
            assert not api.processes
            assert all(child.returncode is not None for child in api.children)
        finally:
            api.release.set()
            api.kill_release.set()
            await asyncio.gather(opening, return_exceptions=True)
            await machine.close()

    asyncio.run(scenario())


def test_daytona_process_cancelled_close_still_terminates_real_child(session_machine):
    async def scenario():
        machine, api = session_machine
        try:
            session = await machine.open_process(Command((sys.executable, "-c", "import signal; signal.pause()")))
            api.kill_release.clear()
            closing = asyncio.create_task(session.close())
            await asyncio.wait_for(api.killing.wait(), 5)
            closing.cancel()
            api.kill_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(closing, 5)
            assert not api.processes
            assert all(child.returncode is not None for child in api.children)
        finally:
            api.kill_release.set()
            await machine.close()

    asyncio.run(scenario())


def test_daytona_machine_close_reclaims_active_process_and_releases_reader(session_machine):
    async def scenario():
        machine, api = session_machine
        session = await machine.open_process(Command((sys.executable, "-c", "import signal; signal.pause()")))
        reading = asyncio.create_task(session.read())
        try:
            await machine.close()
            assert await asyncio.wait_for(reading, 5) == b""
            assert not api.processes
            assert all(child.returncode is not None for child in api.children)
        finally:
            await machine.close()

    asyncio.run(scenario())
