# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the machine transport over real Actor RPC with owned host processes."""

import asyncio
import shutil
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.actor.resolver import FixedResolver
from iris.actor.server import ActorServer
from shellbox.backends.iris.machine_service import MachineService, MachineServiceContract, RemoteMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    Backend,
    Command,
    CommandSemantics,
    ExecutionRequirements,
    ExitReason,
    MachineSpec,
    Result,
)
from shellbox.process_session import StreamProcessSession

IMAGE = "test-machine@sha256:" + "0" * 64
SPEC = MachineSpec(RegistryImage(IMAGE), cpus=1, memory_mb=256)


@dataclass
class HostScope:
    machine: "HostMachine"
    commands: set[asyncio.Task] = field(default_factory=set)

    async def run(self, command: Command) -> Result:
        task = asyncio.current_task()
        assert task is not None
        self.commands.add(task)
        try:
            return await self.machine.run(command)
        finally:
            self.commands.discard(task)

    async def terminate(self) -> None:
        commands = tuple(self.commands)
        for task in commands:
            task.cancel()
        await asyncio.gather(*commands, return_exceptions=True)


@dataclass
class HostMachine:
    root: Path
    closed: threading.Event = field(default_factory=threading.Event)
    command_started: threading.Event = field(default_factory=threading.Event)
    processes: list[StreamProcessSession] = field(default_factory=list)
    invocations: int = 0

    async def run(self, command: Command) -> Result:
        self.invocations += 1
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            cwd=self.root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.command_started.set()
        try:
            async with asyncio.timeout(command.timeout):
                stdout, stderr = await process.communicate(command.stdin)
            limit = command.output_limit_bytes
            return Result(
                process.returncode,
                stdout[:limit],
                stderr[:limit],
                len(stdout) > limit,
                len(stderr) > limit,
                ExitReason.EXITED,
            )
        except TimeoutError:
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def upload(self, source: Path, target: str) -> None:
        destination = self.root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    async def download(self, source: str, target: Path) -> None:
        origin = self.root / source
        if origin.is_dir():
            shutil.copytree(origin, target, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(origin, target)

    async def open_process(self, command: Command) -> StreamProcessSession:
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            cwd=self.root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None and process.stdout is not None

        async def write(data: bytes) -> None:
            assert process.stdin is not None
            process.stdin.write(data)
            await process.stdin.drain()

        async def stop() -> None:
            if process.returncode is None:
                process.kill()
            await process.wait()

        session = StreamProcessSession(process.stdout, write=write, wait=process.wait, stop=stop, stderr=process.stderr)
        self.processes.append(session)
        return session

    async def open_process_scope(self) -> HostScope:
        return HostScope(self)

    async def close(self) -> None:
        await asyncio.gather(*(process.close() for process in self.processes))
        shutil.rmtree(self.root)
        self.closed.set()


class HostFactory:
    backend = Backend.LOCAL

    def __init__(self, root: Path):
        self.root = root
        self.machines: list[HostMachine] = []
        self.creation_started = threading.Event()
        self.creation_release = threading.Event()
        self.creation_release.set()

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        pass

    async def create(self, spec: MachineSpec) -> HostMachine:
        self.creation_started.set()
        try:
            await asyncio.to_thread(self.creation_release.wait)
        except asyncio.CancelledError:
            # Model an acquisition that returns its resource after cancellation.
            await asyncio.to_thread(self.creation_release.wait)
        root = self.root / uuid4().hex
        root.mkdir()
        machine = HostMachine(root)
        self.machines.append(machine)
        return machine


@pytest.fixture
def service(tmp_path):
    native = HostFactory(tmp_path)
    contract = MachineServiceContract(
        Backend.LOCAL,
        CommandSemantics.LINUX_PROCESS,
        LINUX_PROCESS_CAPABILITIES | {"process_scope"},
        (IMAGE,),
        1,
        256,
        "US-EAST-08A",
        "x86_64",
    )
    host = MachineService(native, contract, slots=2)
    server = ActorServer(host="127.0.0.1", port=0)
    server.register("machines", host)
    port = server.serve_background()
    resolver = FixedResolver({"machines": f"http://127.0.0.1:{port}"})
    try:
        yield RemoteMachineFactory(resolver, "machines", rpc_timeout=10), native, resolver
    finally:
        native.creation_release.set()
        server.stop()
        host.close()


@pytest.mark.asyncio
async def test_binary_commands_files_and_clients_remain_isolated(service, tmp_path):
    first, native, resolver = service
    second = RemoteMachineFactory(resolver, "machines", rpc_timeout=10)
    left, right = await asyncio.gather(first.create(SPEC), second.create(SPEC))
    payload = bytes(range(256)) * 32768
    echoed = await left.run(
        Command(
            (sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"),
            stdin=payload,
            output_limit_bytes=len(payload),
        )
    )
    assert echoed.stdout == payload and echoed.exit_code == 0 and not echoed.stdout_truncated
    directory = tmp_path / "upload"
    directory.mkdir()
    (directory / "binary").write_bytes(payload)
    await left.upload(directory, "data")
    downloaded = tmp_path / "download"
    await left.download("data", downloaded)
    assert (downloaded / "binary").read_bytes() == payload
    absent = await right.run(Command((sys.executable, "-c", "from pathlib import Path; print(Path('data').exists())")))
    assert absent.stdout == b"False\n"
    await asyncio.gather(first.close(), second.close())
    assert all(machine.closed.is_set() for machine in native.machines)


@pytest.mark.asyncio
async def test_persistent_python_and_binary_process_pipes(service):
    factory, _, _ = service
    machine = await factory.create(SPEC)
    program = (
        "import sys; n=40\nfor line in sys.stdin.buffer:\n"
        " n+=1; sys.stdout.buffer.write(bytes([0,255])+str(n).encode()+b'\\n'); sys.stdout.buffer.flush()"
    )
    process = await machine.open_process(Command((sys.executable, "-u", "-c", program)))
    await process.write(b"first\n")
    assert await process.read() == b"\x00\xff41\n"
    await process.write(b"second\n")
    assert await process.read() == b"\x00\xff42\n"
    await factory.close()


@pytest.mark.asyncio
async def test_scope_termination_cancels_its_command_and_preserves_machine(service):
    factory, native, _ = service
    machine = await factory.create(SPEC)
    scope = await machine.open_process_scope()
    running = asyncio.create_task(scope.run(Command((sys.executable, "-c", "import time; time.sleep(3600)"))))
    assert await asyncio.to_thread(native.machines[0].command_started.wait, 5)
    await scope.terminate()
    with pytest.raises(asyncio.CancelledError):
        await running
    result = await machine.run(Command((sys.executable, "-c", "print('still alive')")))
    assert result.stdout == b"still alive\n"
    await factory.close()


@pytest.mark.asyncio
async def test_interrupted_creation_closes_the_late_machine(service):
    factory, native, _ = service
    native.creation_release.clear()
    creating = asyncio.create_task(factory.create(SPEC))
    assert await asyncio.to_thread(native.creation_started.wait, 5)
    creating.cancel()
    native.creation_release.set()
    with pytest.raises(asyncio.CancelledError):
        await creating
    assert len(native.machines) == 1 and native.machines[0].closed.is_set()
    await factory.close()


@pytest.mark.asyncio
async def test_lost_start_response_does_not_replay_a_command(service, monkeypatch):
    factory, native, _ = service
    machine = await factory.create(SPEC)
    start = factory.client.start

    def lost_response(*arguments):
        start(*arguments)
        assert native.machines[0].command_started.wait(5)
        raise ConnectError(Code.UNAVAILABLE, "response lost after command started")

    with monkeypatch.context() as patch:
        patch.setattr(factory.client, "start", lost_response)
        with pytest.raises(ConnectError, match="response lost"):
            await machine.run(Command((sys.executable, "-c", "import time; time.sleep(3600)")))
    assert native.machines[0].invocations == 1
    result = await machine.run(Command((sys.executable, "-c", "print('recovered')")))
    assert result.stdout == b"recovered\n"
    await factory.close()


@pytest.mark.asyncio
async def test_failed_teardown_remains_owned_for_explicit_cleanup(service, monkeypatch):
    factory, native, _ = service
    machine = await factory.create(SPEC)
    selected = native.machines[0]

    async def failed_close():
        raise RuntimeError("teardown failed")

    with monkeypatch.context() as patch:
        patch.setattr(selected, "close", failed_close)
        with pytest.raises(RuntimeError, match="teardown failed"):
            await machine.close()
    await machine.close()
    assert selected.closed.is_set()
    await factory.close()
