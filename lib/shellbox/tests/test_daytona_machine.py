# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Daytona adapter against a local process and filesystem fake."""

import asyncio
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona import CreateSandboxFromSnapshotParams, DaytonaNotFoundError
from daytona_api_client_async import SnapshotState
from shellbox.backends.daytona.machine import DaytonaMachineFactory, DaytonaNetworkMode, DaytonaNetworkPolicy
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import Command, MachineSpec


class LocalFiles:
    async def upload_file_stream(self, data: bytes, target: str) -> None:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def download_file(self, source: str) -> bytes:
        return Path(source).read_bytes()


class LocalProcess:
    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout=None):
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return SimpleNamespace(exit_code=process.returncode, result=stdout.decode(errors="replace"))


class LocalSnapshots:
    def __init__(self):
        self.snapshots = {}

    async def get(self, name):
        if name not in self.snapshots:
            raise DaytonaNotFoundError("Snapshot does not exist", status_code=404)
        return self.snapshots[name]

    async def create(self, params):
        snapshot = SimpleNamespace(name=params.name, state=SnapshotState.ACTIVE, params=params)
        self.snapshots[params.name] = snapshot
        return snapshot


class LocalDaytona:
    def __init__(self):
        self.sandbox = SimpleNamespace(fs=LocalFiles(), process=LocalProcess())
        self.deleted = False
        self.closed = False
        self.params = None
        self.timeout = None
        self.snapshot = LocalSnapshots()

    async def create(self, params, *, timeout):
        assert isinstance(params, CreateSandboxFromSnapshotParams)
        await self.snapshot.get(params.snapshot)
        self.params = params
        self.timeout = timeout
        return self.sandbox

    async def delete(self, sandbox):
        assert sandbox is self.sandbox
        self.deleted = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.closed = True


@pytest.mark.parametrize("policy", [None, DaytonaNetworkPolicy(DaytonaNetworkMode.DOMAIN_ALLOW_LIST, "example.org")])
def test_daytona_binary_command_and_files(tmp_path: Path, policy) -> None:
    async def scenario() -> None:
        client = LocalDaytona()
        workdir = tmp_path / "work"
        machine = await DaytonaMachineFactory(lambda: client, network_policy=policy).create(
            MachineSpec(
                source=RegistryImage("ubuntu:24.04"),
                workdir=str(workdir),
                cpus=2,
                memory_mb=1500,
                storage_mb=1025,
                env={"TASK_VALUE": "daytona"},
                startup_timeout=900,
            )
        )
        if policy is None:
            assert client.params.network_block_all is True
            assert client.params.domain_allow_list is None
        else:
            assert client.params.network_block_all is None
            assert client.params.domain_allow_list == "example.org"
        assert client.params.ttl_minutes == 360
        resources = (await client.snapshot.get(client.params.snapshot)).params.resources
        assert (resources.cpu, resources.memory, resources.disk) == (2, 2, 2)
        assert client.params.os_user == "root"
        assert client.timeout == 900
        try:
            environment = await machine.run(Command(("sh", "-c", 'printf "%s" "$TASK_VALUE"'), user="0"))
            assert environment.stdout == b"daytona"
            result = await machine.run(
                Command(
                    ("/bin/sh", "-c", "cat; printf '\\000\\377' >&2"),
                    stdin=b"abc\x00\xff",
                    output_limit_bytes=4,
                )
            )
            assert result.exit_code == 0
            assert result.stdout == b"abc\x00"
            assert result.stdout_truncated
            assert result.stderr == b"\x00\xff"

            source = tmp_path / "input.bin"
            source.write_bytes(b"\x00\xffpayload")
            await machine.upload(source, str(workdir / "data.bin"))
            downloaded = tmp_path / "downloaded.bin"
            await machine.download(str(workdir / "data.bin"), downloaded)
            assert downloaded.read_bytes() == source.read_bytes()
        finally:
            await machine.close()
        assert client.deleted
        assert client.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("mode, exit_code", [(0o755, 0), (0o640, 126)])
def test_daytona_upload_preserves_script_permissions(tmp_path: Path, mode: int, exit_code: int) -> None:
    async def scenario() -> None:
        source = tmp_path / "grader.sh"
        source.write_text("#!/bin/sh\nprintf '1.0\\n'\n")
        source.chmod(mode)
        target = tmp_path / "remote" / "private grader.sh"
        machine = await DaytonaMachineFactory(LocalDaytona).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(target.parent))
        )
        try:
            await machine.upload(source, str(target))
            assert stat.S_IMODE(target.stat().st_mode) == mode
            result = await machine.run(Command((str(target),)))
            assert result.exit_code == exit_code
            assert result.stdout == (b"1.0\n" if exit_code == 0 else b"")
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_daytona_factory_owns_clients_across_worker_event_loops(tmp_path: Path) -> None:
    clients = []

    class LoopClient(LocalDaytona):
        def __init__(self):
            super().__init__()
            self.loop = asyncio.get_running_loop()
            clients.append(self)

        async def create(self, params, *, timeout):
            assert asyncio.get_running_loop() is self.loop
            if params.env_vars.get("FAIL_CREATE"):
                raise ConnectionError("Sandbox creation failed")
            return await super().create(params, timeout=timeout)

    factory = DaytonaMachineFactory(LoopClient)

    async def scenario(index):
        spec = MachineSpec(
            source=RegistryImage("ubuntu:24.04"),
            workdir=str(tmp_path / str(index)),
            env={"FAIL_CREATE": "1"} if index == 2 else {},
        )
        if index == 2:
            with pytest.raises(ConnectionError):
                await factory.create(spec)
            return
        machine = await factory.create(spec)
        try:
            result = await machine.run(Command(("printf", str(index))))
            assert result.stdout == str(index).encode()
        finally:
            await machine.close()

    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = [executor.submit(asyncio.run, scenario(index)) for index in range(3)]
        for result in pending:
            result.result()
    assert all(client.closed for client in clients)
    assert sum(client.deleted for client in clients) == 2


def test_daytona_reuses_snapshot_until_build_inputs_or_resources_change(tmp_path: Path) -> None:
    snapshots = LocalSnapshots()
    clients = []

    def client_factory():
        client = LocalDaytona()
        client.snapshot = snapshots
        clients.append(client)
        return client

    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:24.04\nCOPY input /input\n")
    source = context / "input"
    source.write_text("first")
    spec = MachineSpec(
        source=DockerfileSource(context, dockerfile), workdir=str(tmp_path / "work"), cpus=1, memory_mb=1024
    )
    factory = DaytonaMachineFactory(client_factory)

    async def scenario():
        for current in (spec, spec, replace(spec, memory_mb=2048)):
            machine = await factory.create(current)
            await machine.close()
        source.write_text("second")
        machine = await factory.create(spec)
        await machine.close()

    asyncio.run(scenario())
    names = [client.params.snapshot for client in clients]
    assert names[0] == names[1]
    assert len(set(names)) == 3
    assert all(client.deleted and client.closed for client in clients)


def test_daytona_failed_snapshot_closes_client_without_starting_sandbox() -> None:
    client = LocalDaytona()

    class FailedSnapshots(LocalSnapshots):
        async def create(self, params):
            snapshot = await super().create(params)
            snapshot.state = SnapshotState.BUILD_FAILED
            snapshot.error_reason = "COPY input was missing"
            return snapshot

    client.snapshot = FailedSnapshots()
    with pytest.raises(RuntimeError, match="COPY input was missing"):
        asyncio.run(DaytonaMachineFactory(lambda: client).create(MachineSpec(source=RegistryImage("ubuntu:24.04"))))
    assert client.params is None
    assert client.closed
