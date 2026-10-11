# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker file transfer through the command boundary."""

import asyncio
import os
import shutil
import signal
import sys
import tracemalloc
from pathlib import Path

import pytest
from shellbox.backends.docker.machine import DockerCommandResult, DockerMachine, DockerMachineFactory, docker
from shellbox.machine import Command, DockerImage, ExitReason, MachineSpec


@pytest.mark.parametrize("output_limit", [0, 1024])
def test_docker_candidate_output_is_drained_with_bounded_memory(monkeypatch, output_limit):
    create_process = asyncio.create_subprocess_exec
    script = (
        "import os, sys\nos.write(1, b'SHELLBOX_PGID:42\\n')\nassert len(sys.stdin.buffer.read()) == 196608\n"
        "for _ in range(512):\n os.write(1, b'x' * 65536)\n os.write(2, b'y' * 65536)\n"
    )

    async def local_process(*args, **kwargs):
        assert args[0] == "docker"
        return await create_process(sys.executable, "-c", script, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_process)
    machine = DockerMachine("fixture", MachineSpec(DockerImage("fixture")))
    tracemalloc.start()
    try:
        result = asyncio.run(
            machine.run(Command(("candidate",), stdin=b"abc" * 65536, output_limit_bytes=output_limit, timeout=10))
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (result.exit_code, result.reason) == (0, ExitReason.EXITED)
    assert (result.stdout, result.stderr) == (b"x" * output_limit, b"y" * output_limit)
    assert result.stdout_truncated and result.stderr_truncated
    assert peak < 8 * 1024**2


@pytest.mark.skipif(shutil.which("setsid") is None, reason="The command boundary needs a host setsid executable")
def test_docker_command_preserves_stdin_and_exit_status_when_exec_is_a_group_leader(monkeypatch):
    create_process = asyncio.create_subprocess_exec

    async def local_process(*args, **kwargs):
        return await create_process(
            *args[args.index("shellbox-test-container") + 1 :],
            start_new_session=True,
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_process)
    machine = DockerMachine("shellbox-test-container", MachineSpec(DockerImage("fixture")))
    payload = b"SHELLBOX_PGID:7\nanswer\x00\xff"
    result = asyncio.run(
        machine.run(
            Command(
                (sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); sys.exit(23)"),
                stdin=payload,
                timeout=5,
            )
        )
    )
    assert (result.exit_code, result.stdout, result.reason) == (23, payload, ExitReason.EXITED)


@pytest.mark.skipif(shutil.which("setsid") is None, reason="The command boundary needs a host setsid executable")
@pytest.mark.parametrize("candidate_state", ["running", "completed"])
def test_docker_deadline_stops_running_work_and_preserves_completed_work(tmp_path, monkeypatch, candidate_state):
    create_process = asyncio.create_subprocess_exec
    executions = []

    async def remote_exec(*args, timeout=None, **kwargs):
        # The guest continues after the client deadline, as it does with a remote Docker daemon.
        execution = asyncio.create_task(docker(*args, timeout=None, **kwargs))
        executions.append(execution)
        result = await asyncio.wait_for(asyncio.shield(execution), timeout=timeout)
        if "completed-at-deadline" in args:
            raise TimeoutError("Docker exec response arrived after the command deadline")
        return result

    async def local_process(*args, **kwargs):
        if args[1:3] == ("rm", "-f"):
            child_path = tmp_path / "child.pid"
            if child_path.exists():
                try:
                    os.killpg(os.getpgid(int(child_path.read_text())), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return await create_process("true", **kwargs)
        argv = args[args.index("shellbox-test-container") + 1 :]
        return await create_process(*argv, start_new_session=True, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_process)
    monkeypatch.setattr("shellbox.backends.docker.machine.docker", remote_exec)

    async def scenario():
        machine = DockerMachine("shellbox-test-container", MachineSpec(DockerImage("fixture"), workdir=str(tmp_path)))
        try:
            argv = ("printf", "completed-at-deadline")
            if candidate_state != "completed":
                argv = (
                    "sh",
                    "-c",
                    f"sleep 3600 & echo $! > {tmp_path}/child.pid; wait",
                )
            result = await machine.run(Command(argv, timeout=0.5))
            assert result.reason is ExitReason.TIMED_OUT
            followup = await machine.run(Command(("printf", "ready")))
            assert (followup.exit_code, followup.stdout) == (0, b"ready")
            if candidate_state != "completed":
                child = int((tmp_path / "child.pid").read_text())
                stopped = await machine.run(
                    Command(
                        (
                            "sh",
                            "-c",
                            'if [ -f "/proc/$1/stat" ]; then read pid comm state rest < "/proc/$1/stat"; '
                            'test "$state" = Z; fi',
                            "child-state",
                            str(child),
                        )
                    )
                )
                assert stopped.exit_code == 0
        finally:
            await machine.close()
            await asyncio.gather(*executions)

    asyncio.run(scenario())


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
@pytest.mark.parametrize("user", [None, "12345"])
def test_interrupted_docker_commands_preserve_files_and_stop_descendants(interruption, user):
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
                            user=user,
                            timeout=5 if interruption == "timeout" else None,
                        )
                    )
                )
                async with asyncio.timeout(10):
                    while True:
                        observed = await machine.run(Command(("cat", "child.pid")))
                        # The shell creates child.pid before echo writes the pid into it.
                        if observed.exit_code == 0 and observed.stdout.strip():
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


@pytest.mark.docker
def test_docker_commands_run_in_a_working_directory_the_image_lacks():
    async def scenario():
        machine = await DockerMachineFactory().create(MachineSpec(DockerImage("busybox:1.36"), workdir="/work/nested"))
        try:
            return await machine.run(Command(("pwd",)))
        finally:
            await machine.close()

    result = asyncio.run(scenario())
    assert (result.exit_code, result.stdout) == (0, b"/work/nested\n")


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_interrupted_docker_start_without_a_process_group_disposes_the_container(monkeypatch, interruption):
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


@pytest.mark.parametrize("probe", ["live", "provider_error", "timeout"])
@pytest.mark.parametrize("stop_exit", [0, 1])
def test_docker_stop_disposes_a_live_or_unverified_process_group(monkeypatch, probe, stop_exit):
    containers = set()

    async def provider(*args, **kwargs):
        if args[0] == "run":
            containers.add(args[args.index("--name") + 1])
        elif args[:2] == ("exec", "-i"):
            kwargs["process_group"].set_result(42)
            raise TimeoutError("Docker exec command deadline expired")
        elif "stop-command" in args:
            return DockerCommandResult(stop_exit, b"", b"Cannot stop the command group")
        elif "probe-command" in args:
            if probe == "timeout":
                raise TimeoutError("Provider probe timed out")
            return DockerCommandResult(0 if probe == "live" else 125, b"", b"Provider probe failed")
        elif args[:2] == ("rm", "-f"):
            containers.remove(args[2])
        return DockerCommandResult(0, b"", b"")

    monkeypatch.setattr("shellbox.backends.docker.machine.docker", provider)
    monkeypatch.setattr("shellbox.backends.docker.machine.INTERRUPT_TIMEOUT", 0.05)

    async def scenario():
        machine = await DockerMachineFactory().create(MachineSpec(DockerImage("fixture")))
        with pytest.raises(TimeoutError):
            await machine.run(Command(("candidate",), user="12345"))
        assert not containers
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))

    asyncio.run(scenario())


@pytest.mark.parametrize("exit_code,prefix", [(126, b"\x00\xffstart error\r\n"), (127, b"\x00\xffstart error")])
@pytest.mark.parametrize("output_limit", [0, 1024])
def test_docker_exec_start_failure_preserves_output_and_the_next_command(monkeypatch, exit_code, prefix, output_limit):
    create_process = asyncio.create_subprocess_exec
    script = (
        f"import os, sys\nos.write(1, {prefix!r})\n"
        "for _ in range(512):\n os.write(1, b'x' * 65536)\n os.write(2, b'y' * 65536)\n"
        f"sys.exit({exit_code})\n"
    )

    async def local_process(*args, **kwargs):
        if args[1:3] == ("rm", "-f"):
            return await create_process("true", **kwargs)
        if "startup-failure" in args:
            return await create_process(sys.executable, "-c", script, **kwargs)
        return await create_process(*args[args.index("exec-fixture") + 1 :], start_new_session=True, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_process)

    async def scenario():
        machine = DockerMachine("exec-fixture", MachineSpec(DockerImage("fixture")))
        try:
            tracemalloc.start()
            try:
                result = await machine.run(Command(("startup-failure",), output_limit_bytes=output_limit, timeout=10))
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert (result.exit_code, result.reason) == (exit_code, ExitReason.EXITED)
            assert result.stdout == (prefix + b"x" * output_limit)[:output_limit]
            assert result.stderr == b"y" * output_limit
            assert result.stdout_truncated and result.stderr_truncated
            assert peak < 8 * 1024**2
            following = await machine.run(Command(("printf", "ready")))
            assert (following.exit_code, following.stdout, following.reason) == (0, b"ready", ExitReason.EXITED)
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.docker
def test_docker_removed_workdir_is_an_exited_result_and_the_machine_stays_reusable():
    async def scenario():
        workdir = "/tmp/command-workdir"
        machine = await DockerMachineFactory().create(MachineSpec(DockerImage("busybox:1.36"), workdir=workdir))
        try:
            prepared = await machine.run(Command(("mkdir", "-p", workdir), cwd="/tmp"))
            assert prepared.exit_code == 0
            removed = await machine.run(Command(("rm", "-rf", workdir)))
            assert removed.exit_code == 0
            failed = await machine.run(Command(("true",)))
            assert failed.reason is ExitReason.EXITED
            assert failed.exit_code is not None and failed.exit_code != 0
            following = await machine.run(Command(("printf", "ready"), cwd="/tmp"))
            assert (following.exit_code, following.stdout) == (0, b"ready")
            restored = await machine.run(Command(("mkdir", "-p", workdir), cwd="/tmp"))
            assert restored.exit_code == 0
            reused = await machine.run(Command(("printf", "ready")))
            assert (reused.exit_code, reused.stdout) == (0, b"ready")
        finally:
            await machine.close()

    asyncio.run(scenario())
