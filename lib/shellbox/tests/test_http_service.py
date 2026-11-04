# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import signal
import socket
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import shellbox.http_service as http_service
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.backends.gvisor.machine import GvisorMachineFactory
from shellbox.http_service import MAX_BODY, HttpRequest, HttpResponse, serve_http
from shellbox.machine import Command, DockerImage, MachineSpec, NetworkPolicy, Result
from shellbox.process_session import SubprocessSession

GUEST_HTTP_REQUEST = """
import http.client, json, sys, urllib.parse
url = urllib.parse.urlsplit(sys.argv[1])
connection = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
try:
    connection.request('POST', url.path, body=sys.stdin.buffer.read(),
                       headers={'Authorization': 'Bearer guest-controlled-key'})
    response = connection.getresponse()
    print(json.dumps({'status': response.status, 'body': response.read().decode()}))
finally:
    connection.close()
"""


class ProcessMachine:
    """Exercise the exec/transfer boundary with local child processes."""

    def __init__(self):
        self.processes = []

    async def open_process(self, command: Command) -> SubprocessSession:
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ["PATH"]},
            start_new_session=True,
        )
        self.processes.append(process)

        async def stop():
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

        return SubprocessSession(process, stop=stop)

    async def run(self, command: Command) -> Result:
        raise AssertionError("HTTP relaying must not execute per-request or idle commands")

    async def upload(self, source: Path, target: str) -> None:
        raise AssertionError("HTTP relaying must not upload helper files")

    async def download(self, source: str, target: Path) -> None:
        raise AssertionError("HTTP relaying must not download helper files")

    async def close(self) -> None:
        pass


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_http_capability_serves_concurrent_requests_and_hides_provider_errors(caplog):
    async def check():
        blocked = asyncio.Event()
        release = asyncio.Event()

        async def handler(request: HttpRequest) -> HttpResponse:
            if request.path == "/echo":
                return HttpResponse(200, request.body, "application/octet-stream")
            if request.path == "/slow":
                blocked.set()
                await release.wait()
                return HttpResponse(200, b'{"done":true}')
            if request.path == "/broken":
                raise RuntimeError("provider rejected secret-host-key")
            if request.path == "/invalid-response":
                return HttpResponse(200, b"{}", "application/json\r\nsecret-host-key")
            body = json.loads(request.body)
            return HttpResponse(201, json.dumps({"answer": body["number"] + 1}).encode())

        machine = ProcessMachine()
        async with serve_http(machine, handler, port=available_port()) as url:
            async with httpx.AsyncClient(trust_env=False) as client:
                slow = asyncio.create_task(client.post(url + "/slow", timeout=30))
                await asyncio.wait_for(blocked.wait(), 5)
                answer = await client.post(url + "/answer", json={"number": 41})
                assert answer.status_code == 201
                assert answer.json() == {"answer": 42}
                for path in ("/broken", "/invalid-response"):
                    broken = await client.post(url + path)
                    assert broken.status_code == 502
                    assert broken.json() == {"error": "host service failed"}
                assert "secret-host-key" not in caplog.text
                body = bytes(range(256)) * (MAX_BODY // 256)
                echoed = await client.post(url + "/echo", content=body)
                assert echoed.status_code == 200 and echoed.content == body
                release.set()
                assert (await slow).json() == {"done": True}
        assert len(machine.processes) == 1
        assert machine.processes[0].returncode is not None

    asyncio.run(check())


@pytest.mark.parametrize("body", ["!!!", False])
def test_http_capability_rejects_malformed_guest_body_before_host_dispatch(body):
    frame = json.dumps({"type": "request", "id": "a" * 32, "method": "POST", "path": "/score", "body": body})

    class MalformedFrameMachine(ProcessMachine):
        async def open_process(self, command):
            script = (
                "import sys\n"
                'print(\'{"type":"ready"}\', flush=True)\n'
                f"print({frame!r}, flush=True)\n"
                "sys.stdin.buffer.read()\n"
            )
            return await super().open_process(replace(command, argv=("python3", "-u", "-c", script)))

    async def check():
        machine = MalformedFrameMachine()
        dispatched = []

        async def handler(request):
            dispatched.append(request)
            return HttpResponse(200, b"{}")

        with pytest.raises(ValueError):
            async with serve_http(machine, handler, port=available_port()):
                await asyncio.wait_for(asyncio.Event().wait(), 5)
        assert dispatched == []
        assert machine.processes[0].returncode is not None

    asyncio.run(check())


def test_http_capability_helper_exit_interrupts_idle_context_and_reaps_process():
    async def check():
        machine = ProcessMachine()

        async def handler(request):
            return HttpResponse(200, b"{}")

        with pytest.raises(RuntimeError, match="process ended"):
            async with serve_http(machine, handler, port=available_port()):
                machine.processes[0].kill()
                await asyncio.wait_for(asyncio.Event().wait(), 5)
        assert machine.processes[0].returncode is not None

    asyncio.run(check())


def test_http_capability_preserves_body_error_when_helper_close_fails():
    class BrokenCloseMachine(ProcessMachine):
        async def open_process(self, command):
            session = await super().open_process(command)
            close = session.close

            async def broken_close():
                await close()
                raise RuntimeError("cleanup failed")

            session.close = broken_close
            return session

    async def check():
        machine = BrokenCloseMachine()

        async def handler(request):
            return HttpResponse(200, b"{}")

        with pytest.raises(ValueError, match="body failure"):
            async with serve_http(machine, handler, port=available_port()):
                raise ValueError("body failure")
        assert machine.processes[0].returncode is not None

    asyncio.run(check())


def test_http_capability_repeated_cancellation_finishes_owned_process_cleanup():
    async def check():
        closing, release, entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class DelayedCloseMachine(ProcessMachine):
            async def open_process(self, command):
                session = await super().open_process(command)
                close = session.close

                async def delayed_close():
                    closing.set()
                    await release.wait()
                    await close()

                session.close = delayed_close
                return session

        machine = DelayedCloseMachine()

        async def handler(request):
            return HttpResponse(200, b"{}")

        async def serve():
            async with serve_http(machine, handler, port=available_port()):
                entered.set()
                await asyncio.Event().wait()

        operation = asyncio.create_task(serve())
        await asyncio.wait_for(entered.wait(), 5)
        operation.cancel("original cancellation")
        await asyncio.wait_for(closing.wait(), 5)
        operation.cancel("second cancellation")
        # Let the second cancellation reach the shielded cleanup wait.
        await asyncio.sleep(0)
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await operation
        assert caught.value.args == ("original cancellation",)
        assert machine.processes[0].returncode is not None

    asyncio.run(check())


def test_http_capability_rejects_oversized_guest_frames_and_reaps_process():
    class OversizedFrameMachine(ProcessMachine):
        async def open_process(self, command):
            script = (
                'import os\nos.write(1, b\'{"type":"ready"}\\n\')\n'
                f"os.write(1, b'x' * {http_service.WIRE_LIMIT + 1})\n"
            )
            return await super().open_process(replace(command, argv=("python3", "-u", "-c", script)))

    async def check():
        machine = OversizedFrameMachine()

        async def handler(request):
            raise AssertionError("Invalid guest frames must not reach the host handler")

        with pytest.raises(ValueError, match="exceeds its limit"):
            async with serve_http(machine, handler, port=available_port()):
                await asyncio.wait_for(asyncio.Event().wait(), 5)
        assert machine.processes[0].returncode is not None

    asyncio.run(check())


def test_http_capability_bounds_close_when_host_handler_defers_cancellation(monkeypatch):
    monkeypatch.setattr(http_service, "CLEANUP_TIMEOUT", 0.05)

    async def check():
        entered, cancelled, release, finished = (asyncio.Event() for _ in range(4))
        machine = ProcessMachine()

        async def handler(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            finally:
                finished.set()
            return HttpResponse(200, b"{}")

        request = None
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                async with asyncio.timeout(5):
                    async with serve_http(machine, handler, port=available_port()) as url:
                        request = asyncio.create_task(client.post(url + "/wait"))
                        await entered.wait()
                assert cancelled.is_set()
                assert machine.processes[0].returncode is not None
                release.set()
                await asyncio.wait_for(finished.wait(), 5)
                await asyncio.gather(request, return_exceptions=True)
        finally:
            release.set()
            if request is not None:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)

    asyncio.run(check())


def test_http_capability_close_cancels_pending_host_work():
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
            async with serve_http(ProcessMachine(), handler, port=available_port()) as url:
                request = asyncio.create_task(client.post(url + "/wait"))
                await asyncio.wait_for(entered.wait(), 5)
            assert cancelled.is_set()
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)

    asyncio.run(check())


@pytest.mark.parametrize(
    "factory_type",
    [
        pytest.param(DockerMachineFactory, marks=pytest.mark.docker),
        pytest.param(GvisorMachineFactory, marks=pytest.mark.manual),
    ],
)
def test_denied_container_relay_limits_host_authority_and_reclaims_listener(tmp_path, monkeypatch, factory_type):
    credential = tmp_path / "provider-credentials.pem"
    credential.write_text("host-only-credential-canary")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-storage-canary")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-model-canary")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credential))

    async def check(host_port):
        machine = await factory_type().create(MachineSpec(DockerImage("python:3.12-slim"), network=NetworkPolicy.DENY))
        pending = []
        blocked, release = asyncio.Event(), asyncio.Event()
        waiting, cancelled = asyncio.Event(), asyncio.Event()
        provider_calls = []

        def provider(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://provider.test/v1/score"
            assert request.headers["authorization"] == "Bearer explicit-host-model-key"
            provider_calls.append(json.loads(request.content))
            return httpx.Response(200, json={"answer": 42})

        async def request(url, body=b""):
            result = await machine.run(Command(("python3", "-c", GUEST_HTTP_REQUEST, url), stdin=body, timeout=15))
            assert result.exit_code == 0, result.stderr.decode(errors="replace")
            return json.loads(result.stdout)

        try:
            isolation = await machine.run(
                Command(
                    (
                        "python3",
                        "-c",
                        "import json,os,pathlib,socket,sys\n"
                        "try:\n socket.create_connection(('127.0.0.1',int(sys.argv[1])),timeout=1).close()\n"
                        "except OSError:\n connected=False\nelse:\n connected=True\n"
                        "print(json.dumps({'connected':connected,"
                        "'interfaces':[name for _,name in socket.if_nameindex() if name != 'lo'],'credentials':"
                        "[key for key in ('AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','OPENAI_API_KEY',"
                        "'GOOGLE_APPLICATION_CREDENTIALS','RIGGING_SECRETS_DIR') if key in os.environ],"
                        "'files':[path for path in ('/root/.aws/credentials',"
                        "'/root/.config/gcloud/application_default_credentials.json','/run/secrets',sys.argv[2]) "
                        "if pathlib.Path(path).exists()]}))",
                        str(host_port),
                        str(credential),
                    ),
                    timeout=5,
                )
            )
            assert isolation.exit_code == 0, isolation.stderr.decode(errors="replace")
            assert json.loads(isolation.stdout) == {
                "connected": False,
                "interfaces": [],
                "credentials": [],
                "files": [],
            }

            async with httpx.AsyncClient(transport=httpx.MockTransport(provider), trust_env=False) as upstream:

                async def handler(message: HttpRequest) -> HttpResponse:
                    if message.method != "POST":
                        return HttpResponse(403, b'{"error":"denied"}')
                    if message.path == "/slow":
                        blocked.set()
                        await release.wait()
                        return HttpResponse(200, b'{"done":true}')
                    if message.path == "/score":
                        response = await upstream.post(
                            "https://provider.test/v1/score",
                            json=json.loads(message.body),
                            headers={"Authorization": "Bearer explicit-host-model-key"},
                        )
                        return HttpResponse(response.status_code, response.content)
                    if message.path == "/broken":
                        raise RuntimeError("provider rejected explicit-host-model-key")
                    if message.path == "/oversized":
                        return HttpResponse(200, b"x" * (MAX_BODY + 1))
                    if message.path == "/wait":
                        waiting.set()
                        try:
                            await asyncio.Future()
                        finally:
                            cancelled.set()
                    return HttpResponse(403, b'{"error":"denied"}')

                async with serve_http(machine, handler, port=8787) as url:
                    slow = asyncio.create_task(request(url + "/slow"))
                    pending.append(slow)
                    await asyncio.wait_for(blocked.wait(), 10)
                    answer = await request(url + "/score", b'{"number":41}')
                    assert answer["status"] == 200 and json.loads(answer["body"]) == {"answer": 42}
                    release.set()
                    assert json.loads((await slow)["body"]) == {"done": True}
                    assert (await request(url + "/admin"))["status"] == 403
                    assert (await request(url + "/score", b"x" * (MAX_BODY + 1)))["status"] == 400
                    for path in ("/broken", "/oversized"):
                        failed = await request(url + path)
                        assert failed["status"] == 502
                        assert "explicit-host-model-key" not in failed["body"]
                    assert provider_calls == [{"number": 41}]
                    pending.append(asyncio.create_task(request(url + "/wait")))
                    await asyncio.wait_for(waiting.wait(), 10)
                assert cancelled.is_set()
                for operation in pending:
                    operation.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

            stopped = await machine.run(
                Command(
                    (
                        "python3",
                        "-c",
                        "import glob,json,socket\n"
                        "try:\n socket.create_connection(('127.0.0.1',8787),timeout=1).close()\n"
                        "except OSError:\n connected=False\nelse:\n connected=True\n"
                        "print(json.dumps({'connected':connected,'files':glob.glob('/tmp/shellbox-http-*.py')}))",
                    ),
                    timeout=5,
                )
            )
            assert stopped.exit_code == 0
            assert json.loads(stopped.stdout) == {"connected": False, "files": []}
        finally:
            release.set()
            for operation in pending:
                operation.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await machine.close()
        removed = await docker("inspect", machine.name)
        assert removed.exit_code != 0

    with socket.create_server(("127.0.0.1", 0)) as listener:
        asyncio.run(check(listener.getsockname()[1]))
