# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker file transfer through the command boundary."""

import asyncio
import shutil
from pathlib import Path

import pytest
from shellbox.backends.docker.machine import DockerCommandResult, DockerMachine, DockerMachineFactory
from shellbox.machine import Command, DockerImage, MachineSpec


def test_directory_transfer_preserves_contents_without_an_extra_directory(tmp_path, monkeypatch):
    container = tmp_path / "container"
    container.mkdir()

    def path(value):
        return container / value.split(":", 1)[1].lstrip("/") if ":" in value else Path(value)

    async def docker(*args, **kwargs):
        if args[0] == "exec":
            assert args[1:3] == ("--user", "0")
            assert args[4:6] == ("mkdir", "-p")
            (container / args[6].lstrip("/")).mkdir(parents=True, exist_ok=True)
        else:
            assert args[0] == "cp"
            source, destination = path(args[1]), path(args[2])
            if source.is_dir():
                if destination.is_dir() and not args[1].endswith("/."):
                    destination /= source.name
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
        return DockerCommandResult(0, b"", b"")

    monkeypatch.setattr("shellbox.backends.docker.machine.docker", docker)
    source = tmp_path / "host-artifacts"
    (source / "nested").mkdir(parents=True)
    (source / "nested/answer").write_bytes(b"\x00\xff")
    (source / "nested/answer").chmod(0o755)
    machine = DockerMachine("fixture", MachineSpec(DockerImage("fixture")))
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()

    async def transfer():
        await machine.upload(source, "/logs/artifacts")
        await machine.download("/logs/artifacts", downloaded)

    asyncio.run(transfer())
    assert (container / "logs/artifacts/nested/answer").read_bytes() == b"\x00\xff"
    assert (downloaded / "nested/answer").read_bytes() == b"\x00\xff"
    assert (downloaded / "nested/answer").stat().st_mode & 0o777 == 0o755
    assert sorted(path.name for path in downloaded.iterdir()) == ["nested"]


def test_docker_wire_preserves_resource_limits_and_per_command_users(monkeypatch):
    requests = []

    async def docker(*args, **kwargs):
        requests.append(args)
        return DockerCommandResult(0, b"", b"")

    monkeypatch.setattr("shellbox.backends.docker.machine.docker", docker)

    async def scenario():
        machine = await DockerMachineFactory().create(
            MachineSpec(DockerImage("fixture"), workdir="", cpus=2, memory_mb=1536, storage_mb=1024, gpus=1)
        )
        try:
            await machine.run(Command(("id", "-u"), user="1001"))
            await machine.run(Command(("id", "-u"), user="1002"))
            await machine.run(Command(("pwd",)))
        finally:
            await machine.close()

    asyncio.run(scenario())
    launch = requests[0]
    limits = {flag: launch[launch.index(flag) + 1] for flag in ("--cpus", "--memory", "--storage-opt", "--gpus")}
    assert limits == {"--cpus": "2", "--memory": "1536m", "--storage-opt": "size=1024M", "--gpus": "1"}
    assert requests[1][requests[1].index("--user") + 1] == "1001"
    assert requests[2][requests[2].index("--user") + 1] == "1002"
    assert "--user" not in requests[3]
    assert "-w" not in requests[3]
    assert requests[-1][0:2] == ("rm", "-f")


def test_cancelled_docker_start_removes_a_container_before_returning(monkeypatch):
    containers = set()

    async def scenario():
        started = asyncio.Event()

        async def docker(*args, **kwargs):
            if args[0] == "run":
                containers.add(args[args.index("--name") + 1])
                started.set()
                await asyncio.Future()
            if args[:2] == ("rm", "-f"):
                containers.remove(args[2])
            return DockerCommandResult(0, b"", b"")

        monkeypatch.setattr("shellbox.backends.docker.machine.docker", docker)
        pending = asyncio.create_task(DockerMachineFactory().create(MachineSpec(DockerImage("fixture"))))
        await asyncio.wait_for(started.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert containers == set()

    asyncio.run(scenario())
