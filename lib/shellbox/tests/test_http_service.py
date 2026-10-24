# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import socket
from pathlib import Path

import httpx
from shellbox.http_service import HttpRequest, HttpResponse, serve_http
from shellbox.machine import Command, ExitReason, Result


class ProcessMachine:
    """Exercise the exec/transfer boundary with local child processes."""

    def __init__(self, root: Path):
        self.root = root

    def path(self, value: str) -> str:
        if value.startswith("/tmp/shellbox-http-"):
            return str(self.root / Path(value).name)
        return value

    async def run(self, command: Command) -> Result:
        process = await asyncio.create_subprocess_exec(
            *(self.path(arg) for arg in command.argv),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ["PATH"]},
        )
        stdout, stderr = await process.communicate(command.stdin)
        return Result(process.returncode, stdout, stderr, False, False, ExitReason.EXITED)

    async def upload(self, source: Path, target: str) -> None:
        shutil.copyfile(source, self.path(target))

    async def download(self, source: str, target: Path) -> None:
        shutil.copyfile(self.path(source), target)

    async def close(self) -> None:
        pass


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_http_capability_serves_concurrent_requests_and_hides_provider_errors(tmp_path):
    async def check():
        blocked = asyncio.Event()
        release = asyncio.Event()

        async def handler(request: HttpRequest) -> HttpResponse:
            if request.path == "/slow":
                blocked.set()
                await release.wait()
                return HttpResponse(200, b'{"done":true}')
            if request.path == "/broken":
                raise RuntimeError("provider rejected secret-host-key")
            body = json.loads(request.body)
            return HttpResponse(201, json.dumps({"answer": body["number"] + 1}).encode())

        machine = ProcessMachine(tmp_path)
        async with serve_http(machine, handler, port=available_port()) as url:
            async with httpx.AsyncClient(trust_env=False) as client:
                slow = asyncio.create_task(client.post(url + "/slow"))
                await asyncio.wait_for(blocked.wait(), 5)
                answer = await client.post(url + "/answer", json={"number": 41})
                assert answer.status_code == 201
                assert answer.json() == {"answer": 42}
                broken = await client.post(url + "/broken")
                assert broken.status_code == 502
                assert "secret-host-key" not in broken.text
                release.set()
                assert (await slow).json() == {"done": True}
        assert list(tmp_path.iterdir()) == []

    asyncio.run(check())


def test_http_capability_close_cancels_pending_host_work(tmp_path):
    async def check():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def handler(request: HttpRequest) -> HttpResponse:
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
            raise AssertionError("unreachable")

        async with httpx.AsyncClient(trust_env=False) as client:
            async with serve_http(ProcessMachine(tmp_path), handler, port=available_port()) as url:
                request = asyncio.create_task(client.post(url + "/wait"))
                await asyncio.wait_for(entered.wait(), 5)
            assert cancelled.is_set()
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        assert list(tmp_path.iterdir()) == []

    asyncio.run(check())
