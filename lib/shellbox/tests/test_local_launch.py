# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent launcher behavior at the namespace and exec boundaries."""

import asyncio
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from shellbox.backends.local import machine as local
from shellbox.machine import Command, HostImage, MachineSpec

# Inject the EPERM a launcher sees without SYS_ADMIN. Brokered commands must
# avoid setns; privileged external launchers still use it.
DENIED_NAMESPACE_ENTRY = """
import os
def denied_setns(descriptor, flags):
    raise PermissionError(1, "x" * 65536)
os.setns = denied_setns
"""


@pytest.fixture
def factory() -> local.LocalMachineFactory:
    try:
        return local.LocalMachineFactory()
    except local.SandboxUnavailable as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("profile", [None, "CONTAINER_PROFILE_DEFAULT", "CONTAINER_PROFILE_RESTRICTED"])
def test_local_factory_in_iris_task_rejects_unsupported_profile(monkeypatch, profile):
    monkeypatch.setenv("IRIS_TASK_ID", "/job/task-0")
    if profile is None:
        monkeypatch.delenv("IRIS_CONTAINER_PROFILE", raising=False)
    else:
        monkeypatch.setenv("IRIS_CONTAINER_PROFILE", profile)
    with pytest.raises(local.ContainerProfileRequired, match="CONTAINER_PROFILE_BUBBLEWRAP"):
        local.LocalMachineFactory()


@pytest.mark.skipif(os.geteuid() != 0, reason="Only root can switch the bwrap keeper to uid 65534")
@pytest.mark.parametrize("system_directories", list(local.SystemDirectories))
def test_iris_bubblewrap_factory_writes_to_workdir_as_sandbox_user(monkeypatch, system_directories):
    monkeypatch.setenv("IRIS_TASK_ID", "/job/task-0")
    monkeypatch.setenv("IRIS_CONTAINER_PROFILE", local.IRIS_BUBBLEWRAP_PROFILE)
    monkeypatch.setattr(local, "LAUNCH_SOURCE", DENIED_NAMESPACE_ENTRY + local.LAUNCH_SOURCE)
    probe_entered = False
    control = local.LocalMachine._control

    async def observed_control(self, *args, **kwargs):
        nonlocal probe_entered
        probe_entered = True
        return await control(self, *args, **kwargs)

    monkeypatch.setattr(local.LocalMachine, "_control", observed_control)
    try:
        factory = local.LocalMachineFactory(system_directories=system_directories)
    except local.SandboxUnavailable as error:
        if probe_entered:
            raise
        pytest.skip(str(error))
    assert probe_entered

    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/app"))
        try:
            result = await machine.run(Command(("/bin/sh", "-c", "id -u; echo written > result; cat result")))
            assert result.exit_code == 0, result.stderr.decode()
            assert result.stdout == f"{local.BUBBLEWRAP_UID}\nwritten\n".encode()
            for user in ("root", "0"):
                with pytest.raises(local.UnsupportedMachineSpec):
                    await machine.run(Command(("id", "-u"), user=user))
            if system_directories is local.SystemDirectories.COPY_ON_WRITE:
                script = "echo overlaid > /usr/local/shellbox-overlay; cat /usr/local/shellbox-overlay"
                overlay = await machine.run(Command(("/bin/sh", "-c", script)))
                assert overlay.exit_code == 0, overlay.stderr.decode()
                assert overlay.stdout == b"overlaid\n"
        finally:
            await machine.close()

    asyncio.run(check())


@pytest.mark.skipif(os.geteuid() != 0, reason="The privileged Iris path requires a root task")
def test_iris_privileged_factory_keeps_root_command_identity(monkeypatch):
    monkeypatch.setenv("IRIS_TASK_ID", "/job/task-0")
    monkeypatch.setenv("IRIS_CONTAINER_PROFILE", local.IRIS_PRIVILEGED_PROFILE)
    try:
        factory = local.LocalMachineFactory()
    except local.SandboxUnavailable as error:
        pytest.skip(str(error))

    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/app"))
        try:
            result = await machine.run(Command(("id", "-u"), user="root"))
            assert result.exit_code == 0, result.stderr.decode()
            assert result.stdout == b"0\n"
        finally:
            await machine.close()

    asyncio.run(check())


@pytest.mark.skipif(os.geteuid() != 0, reason="Only root can switch the bwrap launcher to an unprivileged uid")
def test_keeper_launcher_drops_to_non_root_before_exec(tmp_path):
    configuration = tmp_path / "launcher.json"
    configuration.write_text(
        json.dumps(
            {
                "namespaces": [],
                "program": [sys.executable, "-c", "import os; print(os.geteuid())"],
                "program_user": [local.BUBBLEWRAP_UID, local.BUBBLEWRAP_GID],
                "cgroup": None,
            }
        )
    )
    with configuration.open("rb") as source:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", local.LAUNCH_SOURCE, str(source.fileno())],
            pass_fds=(source.fileno(),),
            capture_output=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.strip() == str(local.BUBBLEWRAP_UID).encode()


@pytest.mark.parametrize("cgroup", [False, True])
def test_keeper_broker_launches_command_from_inside_its_namespace(cgroup):
    with tempfile.TemporaryDirectory(prefix="shellbox-broker-") as directory:
        socket_path = Path(directory) / "broker.sock"
        membership = Path(directory) / "cgroup.procs"
        if cgroup:
            membership.touch()
        broker = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", local.LAUNCH_SOURCE, "--broker", str(socket_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            assert broker.stdout is not None
            assert select.select([broker.stdout], [], [], 5)[0]
            assert broker.stdout.read(1) == b"R"
            assert broker.stdin is not None
            broker.stdin.write(b"shellbox-ready\n")
            broker.stdin.flush()
            with tempfile.TemporaryFile() as configuration:
                configuration.write(
                    json.dumps(
                        {
                            "broker": str(socket_path),
                            "broker_pid": broker.pid,
                            "cgroup": str(membership) if cgroup else None,
                            "limits": [],
                            "script": local.RUN_SCRIPT,
                            "cwd": "/",
                            "argv": ["/bin/sh", "-c", "printf '%s' $$" if cgroup else "printf brokered"],
                            "environment": local.CONTROL_ENVIRONMENT,
                        }
                    ).encode()
                )
                configuration.seek(0)
                result = subprocess.run(
                    [sys.executable, "-I", "-S", "-c", local.LAUNCH_SOURCE, "--relay", str(configuration.fileno())],
                    pass_fds=(configuration.fileno(),),
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            if result.returncode != 0:
                broker.kill()
                broker.wait(timeout=5)
                pytest.fail(result.stderr.decode() + broker.stderr.read().decode())
            if cgroup:
                assert result.stdout == membership.read_bytes()
            else:
                assert result.stdout == b"brokered"
        finally:
            broker.kill()
            broker.wait(timeout=5)


def test_keeper_broker_launches_companion_with_pinned_source_fd():
    with tempfile.TemporaryDirectory(prefix="shellbox-broker-") as directory:
        socket_path = Path(directory) / "broker.sock"
        source = Path(directory) / "source"
        source.write_text("companion")
        broker = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", local.LAUNCH_SOURCE, "--broker", str(socket_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            assert broker.stdout is not None
            assert select.select([broker.stdout], [], [], 5)[0]
            assert broker.stdout.read(1) == b"R"
            assert broker.stdin is not None
            broker.stdin.write(b"shellbox-ready\n")
            broker.stdin.flush()
            with tempfile.TemporaryFile() as configuration, tempfile.TemporaryFile() as information:
                with source.open("rb") as pinned:
                    host_pid_read, host_pid_write = os.pipe()
                    configuration.write(
                        json.dumps(
                            {
                                "broker": str(socket_path),
                                "broker_pid": broker.pid,
                                "cgroup": None,
                                "program": [
                                    sys.executable,
                                    "-c",
                                    "import json,os,sys; from pathlib import Path; "
                                    "os.write(int(sys.argv[2]), json.dumps({"
                                    "'child-pid': os.getpid(), 'pid-namespace': os.stat('/proc/self/ns/pid').st_ino"
                                    "}).encode()); sys.stdin.read(1); print(Path(sys.argv[1]).read_text())",
                                    "unused",
                                    "unused",
                                ],
                                "program_fds": [(3, 6, True), (4, 4, False)],
                                "environment": local.CONTROL_ENVIRONMENT,
                            }
                        ).encode()
                    )
                    configuration.seek(0)
                    try:
                        relay = subprocess.Popen(
                            [
                                sys.executable,
                                "-I",
                                "-S",
                                "-c",
                                local.LAUNCH_SOURCE,
                                "--relay",
                                str(configuration.fileno()),
                                str(host_pid_write),
                                str(information.fileno()),
                                str(pinned.fileno()),
                                str(pinned.fileno()),
                            ],
                            pass_fds=(configuration.fileno(), host_pid_write, information.fileno(), pinned.fileno()),
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                        os.close(host_pid_write)
                        assert select.select([host_pid_read], [], [], 5)[0]
                        assert int(os.read(host_pid_read, 20)) > 0
                        stdout, stderr = relay.communicate(input=b"x", timeout=5)
                        assert relay.returncode == 0, stderr.decode()
                    finally:
                        os.close(host_pid_read)
            assert stdout == b"companion\n"
        finally:
            broker.kill()
            broker.wait(timeout=5)


def test_factory_rejects_namespace_entry_failure_and_removes_probe_files(factory, monkeypatch, tmp_path):
    original_mkdtemp = tempfile.mkdtemp

    def probe_directory(*args, **kwargs):
        return original_mkdtemp(*args, **{**kwargs, "dir": tmp_path})

    monkeypatch.setattr(tempfile, "mkdtemp", probe_directory)
    monkeypatch.setattr(local, "LAUNCH_SOURCE", DENIED_NAMESPACE_ENTRY + local.LAUNCH_SOURCE)
    with pytest.raises(local.SandboxUnavailable):
        local.LocalMachineFactory(bwrap=factory.bwrap)
    assert list(tmp_path.iterdir()) == []


def test_factory_reports_keeper_exec_failure(factory, tmp_path):
    wrapper = tmp_path / "bwrap"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "arguments = ['/shellbox-missing-shell' if arg == '/bin/sh' else arg for arg in sys.argv[1:]]\n"
        f"os.execv({str(factory.bwrap)!r}, [{str(factory.bwrap)!r}, *arguments])\n"
    )
    wrapper.chmod(0o755)
    with pytest.raises(local.SandboxUnavailable, match="shellbox-missing-shell"):
        local.LocalMachineFactory(bwrap=wrapper)


def test_namespace_setup_failure_is_infrastructure_error_and_machine_remains_usable(factory, monkeypatch):
    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/"))
        try:
            with monkeypatch.context() as patch:
                patch.setattr(local, "LAUNCH_SOURCE", DENIED_NAMESPACE_ENTRY + local.LAUNCH_SOURCE)
                with pytest.raises(RuntimeError):
                    await machine.run(Command(("true",), timeout=5))
            result = await machine.run(Command(("/bin/sh", "-c", "exit 1")))
            assert result.exit_code == 1
        finally:
            await machine.close()

    asyncio.run(check())


def test_guest_exec_inherits_no_private_launcher_descriptors(factory):
    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/"))
        try:
            # Globbing can list its own directory FD before closing it. Check
            # each proc symlink after expansion to count only still-open FDs.
            result = await machine.run(
                Command(
                    (
                        "/bin/sh",
                        "-c",
                        'for fd in /proc/$$/fd/*; do test ! -L "$fd" || printf "%s\\n" "${fd##*/}"; done',
                    )
                )
            )
            assert result.exit_code == 0
            assert set(result.stdout.splitlines()) == {b"0", b"1", b"2"}
        finally:
            await machine.close()

    asyncio.run(check())


def test_process_session_keeps_pipes_without_exposing_private_descriptors(factory):
    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/"))
        session = await machine.open_process(
            Command(
                (
                    "/bin/sh",
                    "-c",
                    'for fd in /proc/$$/fd/*; do test ! -L "$fd" || printf "%s " "${fd##*/}"; done; '
                    'printf "\\n"; while IFS= read -r line; do printf "reply:%s\\n" "$line"; done',
                )
            )
        )
        try:
            async with asyncio.timeout(5):
                descriptors = bytearray()
                while not descriptors.endswith(b"\n"):
                    descriptors.extend(await session.read(1))
                assert set(bytes(descriptors).split()) == {b"0", b"1", b"2"}
                for value in (b"first", b"second"):
                    await session.write(value + b"\n")
                    response = bytearray()
                    while not response.endswith(b"\n"):
                        response.extend(await session.read(1))
                    assert response == b"reply:" + value + b"\n"
            assert (await machine.run(Command(("true",), timeout=5))).exit_code == 0
        finally:
            await session.close()
            await machine.close()
        assert await session.wait() != 0

    asyncio.run(check())


def test_cancelled_process_acquisition_reclaims_child_after_repeated_cancellation(factory, monkeypatch):
    create_process = asyncio.create_subprocess_exec

    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/"))
        acquired, release = asyncio.Event(), asyncio.Event()
        processes = []

        async def held_process(*args, **kwargs):
            process = await create_process(*args, **kwargs)
            processes.append(process)
            acquired.set()
            await release.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", held_process)
        acquisition = asyncio.create_task(machine.open_process(Command(("/bin/sh", "-c", "read line"))))
        try:
            await asyncio.wait_for(acquired.wait(), 5)
            acquisition.cancel("original cancellation")
            # Enter the acquisition's teardown before delivering cancellation again.
            await asyncio.sleep(0)
            acquisition.cancel("second cancellation")
            await asyncio.sleep(0)
            assert not acquisition.done()
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(acquisition, 5)
            assert caught.value.args == ("original cancellation",)
            assert processes[0].returncode is not None
        finally:
            release.set()
            await asyncio.gather(acquisition, return_exceptions=True)
            await machine.close()

    asyncio.run(check())


def test_commands_join_the_keeper_cgroup_namespace(factory):
    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/"))
        try:
            result = await machine.run(
                Command(("/bin/sh", "-c", "readlink /proc/self/ns/cgroup; readlink /proc/1/ns/cgroup"))
            )
            assert result.exit_code == 0
            command, keeper = result.stdout.splitlines()
            assert command == keeper
        finally:
            await machine.close()

    asyncio.run(check())


def test_cancelled_startup_reclaims_an_init_at_the_external_barrier_after_launcher_exit(factory, monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    create_subprocess = asyncio.create_subprocess_exec

    async def check():
        spawned, release = asyncio.Event(), asyncio.Event()
        processes = []
        init_descriptors = []

        async def hold_real_process(*args, **kwargs):
            process = await create_subprocess(*args, **kwargs)
            processes.append(process)
            children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
            # Reproduce the observed external-barrier deadlock after namespace
            # setup, before the caller has received the subprocess handle.
            async with asyncio.timeout(5):
                while True:
                    pids = children.read_text().split()
                    if pids:
                        [pid] = pids
                        wait = Path(f"/proc/{pid}/wchan").read_text().strip()
                        if wait in {"anon_pipe_read", "pipe_read"}:
                            break
                    await asyncio.sleep(0)
            init_descriptors.append(os.pidfd_open(int(pid)))
            spawned.set()
            await release.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", hold_real_process)
        creation = asyncio.create_task(factory.create(MachineSpec(HostImage(), workdir="/")))
        try:
            await asyncio.wait_for(spawned.wait(), 5)
            creation.cancel()
            # Let startup handle cancellation before the subprocess boundary returns.
            await asyncio.sleep(0)
            processes[0].kill()
            async with asyncio.timeout(5):
                while processes[0].returncode is None:
                    await asyncio.sleep(0)
            release.set()
            done, _ = await asyncio.wait((creation,), timeout=5)
            assert creation in done
            # The already-reaped launcher cannot supply its pidfd to allocation.
            with pytest.raises(ProcessLookupError):
                await creation
            poller = select.poll()
            poller.register(init_descriptors[0], select.POLLIN)
            assert await asyncio.to_thread(poller.poll, 5000)
            assert not tuple(tmp_path.iterdir())
        finally:
            release.set()
            for descriptor in init_descriptors:
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.close(descriptor)
            await asyncio.gather(creation, return_exceptions=True)

    asyncio.run(check())
