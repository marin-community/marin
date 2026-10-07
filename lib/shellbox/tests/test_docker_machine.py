# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker file transfer through the command boundary."""

import asyncio
import shutil
from pathlib import Path

import pytest
from shellbox.backends.docker.machine import DockerCommandResult, DockerMachine, DockerMachineFactory, docker
from shellbox.machine import Command, DockerImage, ExitReason, MachineSpec


@pytest.mark.skipif(shutil.which("setsid") is None, reason="The command boundary needs a host setsid executable")
def test_docker_command_preserves_stdin_and_exit_status_when_exec_is_a_group_leader(monkeypatch):
    async def docker_exec(*args, stdin=b"", timeout=None):
        process = await asyncio.create_subprocess_exec(
            *args[args.index("shellbox-test-container") + 1 :],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout=timeout)
        return DockerCommandResult(process.returncode, stdout, stderr)

    monkeypatch.setattr("shellbox.backends.docker.machine.docker", docker_exec)
    machine = DockerMachine("shellbox-test-container", MachineSpec(DockerImage("fixture")))
    result = asyncio.run(
        machine.run(
            Command(("sh", "-c", 'read -r value; printf "%s\\n" "$value"; exit 23'), stdin=b"answer\n", timeout=5)
        )
    )
    assert (result.exit_code, result.stdout, result.reason) == (23, b"answer\n", ExitReason.EXITED)


def test_directory_transfer_preserves_contents_without_an_extra_directory(tmp_path, monkeypatch):
    container = tmp_path / "container"
    container.mkdir()

    def path(value):
        return container / value.split(":", 1)[1].lstrip("/") if ":" in value else Path(value)

    async def docker(*args, **_kwargs):
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

    async def docker(*args, **_kwargs):
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
    assert requests[2][requests[2].index("--user") + 1] == "1001"
    assert requests[3][requests[3].index("--user") + 1] == "1002"
    assert "--user" not in requests[4]
    assert "-w" not in requests[4]
    assert requests[-1][0:2] == ("rm", "-f")


def test_cancelled_docker_start_removes_a_container_before_returning(monkeypatch):
    containers = set()

    async def scenario():
        started = asyncio.Event()

        async def docker(*args, **_kwargs):
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


@pytest.mark.docker
@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_interrupted_docker_commands_preserve_files_and_stop_descendants(interruption):
    async def scenario():
        machine = await DockerMachineFactory().create(MachineSpec(DockerImage("busybox:1.36"), workdir="/tmp"))
        try:
            await machine.run(Command(("sh", "-c", "echo 12 > answer")))
            await machine.run(Command(("sh", "-c", "sleep 7200 >/dev/null 2>&1 & echo $! > service.pid")))
            async with asyncio.TaskGroup() as commands:
                pending = commands.create_task(
                    machine.run(
                        Command(
                            ("sh", "-c", "sleep 3600 & echo $! > child.pid; wait"),
                            timeout=5 if interruption == "timeout" else None,
                        )
                    )
                )
                async with asyncio.timeout(10):
                    while True:
                        observed = await machine.run(Command(("cat", "child.pid")))
                        if observed.exit_code == 0:
                            break
                child = int(observed.stdout)
                if interruption == "cancel":
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                else:
                    result = await pending
                    assert result.reason == ExitReason.TIMED_OUT
            answer = await machine.run(Command(("cat", "answer")))
            assert (answer.exit_code, answer.stdout) == (0, b"12\n")
            stopped = await machine.run(
                Command(
                    (
                        "sh",
                        "-c",
                        'if [ -f "/proc/$1/stat" ]; then read -r pid comm state rest < "/proc/$1/stat"; '
                        'test "$state" = Z; fi',
                        "child-state",
                        str(child),
                    )
                )
            )
            assert stopped.exit_code == 0
            service = await machine.run(Command(("sh", "-c", 'kill -0 "$(cat service.pid)"')))
            assert service.exit_code == 0
        finally:
            await machine.close()
        inspected = await docker("inspect", machine.name)
        assert inspected.exit_code != 0

    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_interrupted_docker_command_without_a_pid_disposes_the_container(monkeypatch, interruption):
    containers = set()

    async def scenario():
        started = asyncio.Event()

        async def docker(*args, **_kwargs):
            if args[0] == "run":
                containers.add(args[args.index("--name") + 1])
            elif args[:2] == ("exec", "-i"):
                started.set()
                if interruption == "timeout":
                    raise TimeoutError("Docker exec startup timed out")
                await asyncio.Future()
            elif args[:3] == ("exec", "--user", "0"):
                return DockerCommandResult(1, b"", b"Command PID is not available")
            elif args[:2] == ("rm", "-f"):
                containers.remove(args[2])
            return DockerCommandResult(0, b"", b"")

        monkeypatch.setattr("shellbox.backends.docker.machine.docker", docker)
        machine = await DockerMachineFactory().create(MachineSpec(DockerImage("fixture")))
        pending = asyncio.create_task(machine.run(Command(("true",))))
        await asyncio.wait_for(started.wait(), timeout=5)
        if interruption == "cancel":
            pending.cancel()
        with pytest.raises(asyncio.CancelledError if interruption == "cancel" else TimeoutError):
            await pending
        assert containers == set()
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))

    asyncio.run(scenario())
