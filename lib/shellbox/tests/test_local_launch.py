# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent launcher behavior at the namespace and exec boundaries."""

import asyncio
import os
import select
import signal
import tempfile
from pathlib import Path

import pytest
from shellbox.backends.local import machine as local
from shellbox.machine import Command, HostImage, MachineSpec

# Inject a failing kernel boundary in the separately executed launcher. Keeper
# creation needs no setns; entering the already-created machine does.
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


def test_factory_rejects_namespace_entry_failure_and_removes_probe_files(factory, monkeypatch, tmp_path):
    original_mkdtemp = tempfile.mkdtemp

    def probe_directory(*args, **kwargs):
        return original_mkdtemp(*args, **{**kwargs, "dir": tmp_path})

    monkeypatch.setattr(tempfile, "mkdtemp", probe_directory)
    monkeypatch.setattr(local, "LAUNCH_SOURCE", DENIED_NAMESPACE_ENTRY + local.LAUNCH_SOURCE)
    with pytest.raises(local.SandboxUnavailable):
        local.LocalMachineFactory(bwrap=factory.bwrap)
    assert list(tmp_path.iterdir()) == []


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
