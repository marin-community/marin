# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris process pipes over a real loopback gateway and subprocess transport."""

import asyncio
import os
import signal
import socket
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("iris")

import uvicorn
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.exec_gateway.auth import ExecPodTarget, exec_audience, mint_exec_grant
from iris.cluster.exec_gateway.main import gateway_application
from iris.cluster.exec_gateway.service import ExecGatewayService
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, exec_gateway_pb2
from iris.rpc.exec_gateway_connect import ExecGatewayServiceClient
from iris.testing.exec_gateway import ProcessExecConnector
from rigging.token_authority import JwksVerifier, JwtSigner, generate_ed25519_keypair, signing_key_from_private_pem
from shellbox.backends.iris import machine as iris_backend
from shellbox.backends.iris.machine import IrisMachine
from shellbox.http_service import HttpRequest, HttpResponse, serve_http
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, MachineTerminated


class GatewayController:
    """The controller authorizes execution; guest data bypasses this boundary."""

    def __init__(self, signer: JwtSigner, address: str):
        self.signer = signer
        self.address = address
        self.opened: list[controller_pb2.Controller.OpenExecSessionRequest] = []
        self.admitted: threading.Event | None = None
        self.release: threading.Event | None = None

    def open_exec_session(self, request, *, timeout_ms):
        del timeout_ms
        self.opened.append(request)
        if self.admitted is not None:
            self.admitted.set()
        if self.release is not None:
            assert self.release.wait(10)
        target = ExecPodTarget("iris", "pod", "pod-uid", request.container_name or "task", "attempt-one")
        token = mint_exec_grant(
            self.signer,
            gateway_id="gateway",
            grant_id=uuid.uuid4().hex,
            subject="test-user",
            target=target,
            command=list(request.command),
        )
        return controller_pb2.Controller.OpenExecSessionResponse(
            gateway_address=self.address, grant=token, attempt_uid=target.attempt_uid
        )

    def exec_in_container(self, *args, **kwargs):
        raise AssertionError("Persistent relay data must not use controller exec")


class Endpoint:
    def close(self):
        pass


class Client:
    def shutdown(self):
        pass


class Job:
    def __init__(self, connector: ProcessExecConnector):
        self.cancelled = False
        self.connector = connector

    def cancel(self):
        self.cancelled = True
        for process in self.connector.processes:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@asynccontextmanager
async def gateway_machine(tmp_path: Path):
    keypair = generate_ed25519_keypair()
    signer = JwtSigner(signing_key_from_private_pem(keypair.private_pem), issuer="test-controller")
    verifier = JwksVerifier(
        issuers={"test-controller": [keypair.public_pem]},
        expected_audiences=frozenset({exec_audience("gateway")}),
        leeway_seconds=0,
    )
    connector = ProcessExecConnector()
    service = ExecGatewayService(verifier, connector)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        controller = GatewayController(signer, f"http://127.0.0.1:{listener.getsockname()[1]}")
        job = Job(connector)
        machine = IrisMachine(
            Endpoint(),
            Client(),
            controller,
            job,
            SimpleNamespace(task_id=JobName.from_wire("/test-user/shellbox/0")),
            MachineSpec(RegistryImage("fixture"), workdir=str(tmp_path)),
            container_name="runtime-api",
        )
        application = gateway_application(service)
        async with application.router.lifespan_context(application):
            server = uvicorn.Server(uvicorn.Config(application, log_level="error", lifespan="off", log_config=None))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                yield machine, controller, connector
            finally:
                await machine.close()
                server.should_exit = True
                await serving


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_iris_http_relay_streams_large_binary_frames_concurrently_and_reclaims_pending_work(tmp_path):
    async def check():
        entered, release, waiting, cancelled = (asyncio.Event() for _ in range(4))

        async def handler(request: HttpRequest) -> HttpResponse:
            if request.path == "/slow":
                entered.set()
                await release.wait()
                return HttpResponse(200, b"slow finished")
            if request.path == "/wait":
                waiting.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return HttpResponse(201, request.body[::-1], "application/octet-stream")

        async with gateway_machine(tmp_path) as (machine, controller, connector):
            async with httpx.AsyncClient(trust_env=False) as client:
                pending = []
                try:
                    async with serve_http(machine, handler, port=available_port()) as url:
                        slow = asyncio.create_task(client.post(url + "/slow", timeout=20))
                        pending.append(slow)
                        await asyncio.wait_for(entered.wait(), 5)
                        payload = bytes(range(256)) * 8192
                        result = await client.post(url + "/binary", content=payload, timeout=20)
                        assert result.status_code == 201 and result.content == payload[::-1]
                        release.set()
                        assert (await slow).content == b"slow finished"
                        pending.append(asyncio.create_task(client.post(url + "/wait", timeout=20)))
                        await asyncio.wait_for(waiting.wait(), 5)
                    assert cancelled.is_set()
                    assert len(controller.opened) == 1
                    assert connector.targets[0].container_name == "runtime-api"
                    assert len(connector.processes) == 1 and connector.processes[0].returncode is not None
                finally:
                    release.set()
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(check())


def test_iris_process_chunks_binary_stdin_and_stdout_and_drains_bounded_stderr(tmp_path):
    payload = bytes(range(256)) * 8192
    script = (
        "import os,sys\n"
        "assert os.getcwd() == os.environ['EXPECTED_CWD']\n"
        "sys.stderr.buffer.write(b'x' * 2097152); sys.stderr.buffer.flush()\n"
        "data = sys.stdin.buffer.read(2097152)\n"
        "sys.stdout.buffer.write(data[::-1]); sys.stdout.buffer.flush(); sys.exit(7)\n"
    )

    async def check():
        async with gateway_machine(tmp_path) as (machine, controller, connector):
            session = await machine.open_process(
                Command(("python3", "-u", "-c", script), env={"EXPECTED_CWD": str(tmp_path)}, output_limit_bytes=32)
            )
            received = bytearray()
            async with asyncio.timeout(20):
                await session.write(payload)
                while chunk := await session.read(16384):
                    received.extend(chunk)
                assert await session.wait() == 7
            assert received == payload[::-1]
            assert session.stderr == b"x" * 32
            await session.close()
            assert len(controller.opened) == 1
            assert connector.processes[0].returncode == 7

    asyncio.run(check())


def test_iris_process_cancelled_admission_waits_for_ownership_and_reclaims_process(tmp_path):
    async def check():
        async with gateway_machine(tmp_path) as (machine, controller, connector):
            controller.admitted, controller.release = threading.Event(), threading.Event()
            opening = asyncio.create_task(machine.open_process(Command(("sh", "-c", "read line"))))
            try:
                assert await asyncio.to_thread(controller.admitted.wait, 5)
                opening.cancel("original cancellation")
                # Let cancellation enter ownership recovery before interrupting it again.
                await asyncio.sleep(0)
                opening.cancel("second cancellation")
                await asyncio.sleep(0)
                assert not opening.done()
                controller.release.set()
                with pytest.raises(asyncio.CancelledError) as caught:
                    await asyncio.wait_for(opening, 5)
                assert caught.value.args == ("original cancellation",)
                assert len(connector.processes) == 1 and connector.processes[0].returncode is not None
                assert len(controller.opened) == 1
            finally:
                controller.release.set()
                await asyncio.gather(opening, return_exceptions=True)

    asyncio.run(check())


def test_iris_machine_close_reclaims_every_open_process_and_rejects_more_writes(tmp_path):
    async def check():
        async with gateway_machine(tmp_path) as (machine, controller, connector):
            first = await machine.open_process(Command(("cat",)))
            second = await machine.open_process(Command(("cat",)))
            await first.write(b"first process\n")
            await second.write(b"second process\n")
            assert await first.read() == b"first process\n"
            assert await second.read() == b"second process\n"
            await asyncio.gather(machine.close(), machine.close())
            assert len(controller.opened) == 2
            assert all(process.returncode is not None for process in connector.processes)
            for session in (first, second):
                with pytest.raises(MachineTerminated):
                    await session.write(b"late")

    asyncio.run(check())


def test_iris_process_lost_open_response_terminates_sandbox_without_repeating_exec(tmp_path, monkeypatch):
    class LostOpenClient(ExecGatewayServiceClient):
        async def open(self, request, **kwargs):
            opened = await super().open(request, **kwargs)
            # The command has started before its Open response is lost.
            await self.receive(
                exec_gateway_pb2.ReceiveRequest(
                    grant=request.grant, session_id=opened.session_id, max_bytes=65536, wait_ms=20000
                )
            )
            raise ConnectError(Code.UNAVAILABLE, "Open response lost")

    monkeypatch.setattr(iris_backend, "ExecGatewayServiceClient", LostOpenClient)

    async def check():
        async with gateway_machine(tmp_path) as (machine, controller, connector):
            with pytest.raises(ConnectError) as caught:
                await machine.open_process(Command(("sh", "-c", "printf ready; read line")))
            assert caught.value.code == Code.UNAVAILABLE
            assert machine.job.cancelled
            assert len(controller.opened) == len(connector.processes) == 1
            assert await asyncio.wait_for(connector.processes[0].wait(), 5) != 0

    asyncio.run(check())
