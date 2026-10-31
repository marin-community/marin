# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real child pipes behind Daytona's text-only session API boundary."""

import asyncio
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

from daytona import DaytonaNotFoundError
from shellbox.backends.daytona.machine import DaytonaMachine
from shellbox.http_service import HttpResponse, serve_http
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec


class SessionAPI:
    def __init__(self):
        self.sessions = {}
        self.children = []
        self.requests = []
        self.hold = None
        self.accepted = asyncio.Event()
        self.release = asyncio.Event()
        self.deleting = asyncio.Event()
        self.delete_release = asyncio.Event()
        self.delete_release.set()
        self.corrupt_output = None

    async def create_session(self, name, request_timeout):
        self.sessions[name] = None
        if self.hold == "create":
            self.accepted.set()
            await self.release.wait()

    async def execute_session_command(self, name, request, timeout):
        self.requests.append(request)
        process = await asyncio.create_subprocess_shell(
            request.command,
            env={"PATH": os.defpath, "HOME": "/tmp"},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self.sessions[name] = process
        self.children.append(process)
        if self.hold == "execute":
            self.accepted.set()
            await self.release.wait()
        return SimpleNamespace(cmd_id="command")

    async def get_session_command_logs_async(self, name, command, stdout, stderr):
        process = self.sessions[name]
        if self.corrupt_output is not None:
            await stdout(self.corrupt_output)
            return

        async def pump(pipe, callback):
            while data := await pipe.read(337):
                await callback(data.decode("ascii"))

        await asyncio.gather(pump(process.stdout, stdout), pump(process.stderr, stderr))
        await process.wait()

    async def send_session_command_input(self, name, command, data, request_timeout):
        process = self.sessions[name]
        process.stdin.write(data.encode("ascii"))
        await process.stdin.drain()

    async def delete_session(self, name, request_timeout):
        self.deleting.set()
        await self.delete_release.wait()
        if name not in self.sessions:
            raise DaytonaNotFoundError("Session is absent", status_code=404)
        process = self.sessions.pop(name)
        if process is None:
            return
        # The provider API owns every process in its session, including child
        # process groups. Reproduce that ownership with real Linux processes.
        pending, descendants = [process.pid], []
        while pending:
            pid = pending.pop()
            try:
                # Freeze before enumerating: the shell/bridge may otherwise fork
                # between this snapshot and its kill, retaining the output pipes.
                os.kill(pid, signal.SIGSTOP)
                while True:
                    state = next(
                        row.split()[1]
                        for row in Path(f"/proc/{pid}/status").read_text().splitlines()
                        if row.startswith("State:")
                    )
                    if state in {"T", "t", "Z", "X"}:
                        break
                    # Yield until the kernel has delivered SIGSTOP, rather than
                    # guessing a duration during which the child can still fork.
                    await asyncio.sleep(0)
                descendants.append(pid)
                pending.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
            except (FileNotFoundError, ProcessLookupError):
                pass
        for pid in reversed(descendants):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.wait()

    async def close(self):
        self.release.set()
        self.delete_release.set()
        for name in tuple(self.sessions):
            await self.delete_session(name, 10)


@pytest.fixture
def session_machine(tmp_path):
    api = SessionAPI()
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


def test_daytona_process_preserves_binary_pipes_defaults_and_bounded_stderr(session_machine, tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-only-canary")
    directory = tmp_path / "command-directory"
    directory.mkdir()
    data = bytes(range(256)) * 8192
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
            assert json.loads(header) == [str(directory), "command", None]
            assert body == data[::-1]
            assert await session.wait() == 7
            assert session.stderr == bytes(range(37))
            await session.close()
            assert not api.sessions
            assert len(api.requests) == 1 and api.requests[0].suppress_input_echo is True
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_daytona_http_capability_uses_one_persistent_provider_session(session_machine, unused_tcp_port):
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
            async with serve_http(machine, handler, port=unused_tcp_port) as endpoint:
                async with httpx.AsyncClient() as client:
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
            assert len(api.requests) == 1
            assert not api.sessions
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["create", "execute"])
def test_daytona_process_cancelled_acquisition_reclaims_accepted_session(session_machine, stage):
    async def scenario():
        machine, api = session_machine
        api.hold = stage
        api.delete_release.clear()
        opening = asyncio.create_task(
            machine.open_process(Command((sys.executable, "-c", "import signal; signal.pause()")))
        )
        try:
            await asyncio.wait_for(api.accepted.wait(), 5)
            opening.cancel()
            api.release.set()
            await asyncio.wait_for(api.deleting.wait(), 5)
            opening.cancel()
            api.delete_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(opening, 5)
            assert not api.sessions
            assert all(child.returncode is not None for child in api.children)
        finally:
            api.release.set()
            api.delete_release.set()
            await asyncio.gather(opening, return_exceptions=True)
            await machine.close()

    asyncio.run(scenario())


def test_daytona_process_cancelled_close_still_terminates_real_child(session_machine):
    async def scenario():
        machine, api = session_machine
        try:
            session = await machine.open_process(Command((sys.executable, "-c", "import signal; signal.pause()")))
            api.delete_release.clear()
            closing = asyncio.create_task(session.close())
            await asyncio.wait_for(api.deleting.wait(), 5)
            closing.cancel()
            api.delete_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(closing, 5)
            assert not api.sessions
            assert all(child.returncode is not None for child in api.children)
        finally:
            api.delete_release.set()
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
            assert not api.sessions
            assert all(child.returncode is not None for child in api.children)
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("wire", ["D!\n", "D" + "A" * 200000, ""])
def test_daytona_process_rejects_corrupt_or_truncated_provider_stream_and_stops_child(session_machine, wire):
    async def scenario():
        machine, api = session_machine
        api.corrupt_output = wire
        try:
            session = await machine.open_process(Command((sys.executable, "-c", "import signal; signal.pause()")))
            with pytest.raises((ValueError, RuntimeError)):
                await asyncio.wait_for(session.read(), 5)
            await session.close()
            assert not api.sessions
            assert all(child.returncode is not None for child in api.children)
        finally:
            await machine.close()

    asyncio.run(scenario())
