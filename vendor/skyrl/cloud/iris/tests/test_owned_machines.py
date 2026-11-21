"""Rollout ownership must preserve persistent machine protocols."""

import asyncio
import socket

import httpx
import pytest
from shellbox.http_service import HttpResponse, serve_http
from shellbox.image import RegistryImage
from shellbox.machine import Backend, Command, MachineSpec
from shellbox.process_session import SubprocessSession
from skyrl_train.rollouts.machines import OwnedMachineFactory


class PipeMachine:
    def __init__(self):
        self.processes = []

    async def run(self, command):
        raise AssertionError("The relay must use persistent pipes")

    async def upload(self, source, target):
        raise AssertionError("The relay must not upload its transport")

    async def download(self, source, target):
        raise AssertionError("Readiness does not download response bodies")

    async def open_process(self, command: Command):
        process = await asyncio.create_subprocess_exec(
            *command.argv, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )

        async def stop():
            if process.returncode is None:
                process.kill()
            await process.wait()

        session = SubprocessSession(process, stop=stop)
        self.processes.append(session)
        return session

    async def close(self):
        await asyncio.gather(*(process.close() for process in self.processes))


class PipeFactory:
    backend = Backend.LOCAL

    def __init__(self, machine):
        self.machine = machine
        self.closed = False

    def check_environment(self, requirements, spec):
        pass

    async def create(self, spec):
        return self.machine

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_rollout_owned_factory_keeps_private_http_relay_process_support():
    native = PipeMachine()
    native_factory = PipeFactory(native)
    factory = OwnedMachineFactory(native_factory, "local")
    machine = await factory.create(MachineSpec(RegistryImage("fixture@sha256:" + "a" * 64)))

    async def handler(_request):
        return HttpResponse(status=200, body=b"owned relay")

    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    try:
        async with asyncio.TaskGroup() as group:
            async with serve_http(machine, handler, port=port, task_group=group) as address:
                async with httpx.AsyncClient(trust_env=False) as client:
                    response = await client.get(address)
                    assert response.status_code == 200 and response.content == b"owned relay"
    finally:
        await factory.close(asyncio.get_running_loop().time() + 5)
    assert len(native.processes) == 1 and native.processes[0].process.returncode is not None
    assert native_factory.closed
