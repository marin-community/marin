# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local machine startup, interruption, and process cleanup."""

import asyncio
import shutil
import signal
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from shellbox.backends.local.runtime import LocalMachine
from shellbox.machine import (
    Command,
    MachineSpec,
    MachineTerminated,
    NetworkPolicy,
)
from shellbox.task import ContainerSpec, TaskSpec
from shellbox.testing.local_machine_support import SPEC, make_factory, on_machine

ROOT_DELIVERY_QUIESCE_PROGRAM = """\
import os, signal

mine = set()
pid = os.getpid()
while pid > 1:
    mine.add(pid)
    try:
        with open(f"/proc/{pid}/stat") as stream:
            fields = stream.read().rsplit(") ", 1)[1].split()
        pid = int(fields[1])
    except (OSError, IndexError, ValueError):
        break
killed = []
for entry in os.scandir("/proc"):
    if not entry.name.isdecimal():
        continue
    victim = int(entry.name)
    if victim in mine or victim == 1:
        continue
    try:
        with open(f"/proc/{victim}/stat") as stream:
            fields = stream.read().rsplit(") ", 1)[1].split()
        state = fields[0]
    except (OSError, IndexError, ValueError):
        continue
    if state.startswith("Z"):
        continue
    try:
        os.kill(victim, signal.SIGKILL)
        killed.append(victim)
    except ProcessLookupError:
        continue
print(len(killed))
"""


ROOT_DELIVERY_QUIESCE_SCRIPT = """\
set -eu
self=$$
ancestry=" $self "
pid=$self
while [ "$pid" -gt 1 ]; do
  line=$(cat "/proc/$pid/stat" 2>/dev/null) || break
  rest=${line##*) }
  state=${rest%% *}
  ppid=${rest#* }
  ppid=${ppid%% *}
  case $ppid in ''|*[!0-9]*) break;; esac
  ancestry="$ancestry $ppid "
  pid=$ppid
done
killed=0
for entry in /proc/[0-9]*; do
  victim=${entry#/proc/}
  case $ancestry in *" $victim "*) continue;; esac
  [ "$victim" -gt 1 ] || continue
  line=$(cat "/proc/$victim/stat" 2>/dev/null) || continue
  rest=${line##*) }
  case ${rest%% *} in Z*) continue;; esac
  if kill -9 "$victim" 2>/dev/null; then
    killed=$((killed + 1))
  elif [ ! -d "/proc/$victim" ]; then
    continue
  else
    echo "quiesce kill failed for $victim" >&2
    exit 1
  fi
done
echo "$killed"
"""


@pytest.mark.parametrize("members", [1, 2], ids=["machine", "task"])
@pytest.mark.parametrize("retained_stream", ["stdout", "stderr"])
def test_process_session_close_reclaims_detached_command_and_its_task(undelegated_factory, members, retained_stream):
    async def scenario():
        spec = TaskSpec(
            tuple(
                ContainerSpec(
                    f"member-{index}",
                    replace(SPEC, network=NetworkPolicy.DENY),
                    entrypoint=("sleep", "3600") if index else None,
                )
                for index in range(members)
            ),
            primary="member-0",
        )
        task = await undelegated_factory.create_task(spec)
        machine = cast(LocalMachine, task.primary)
        roots = [cast(LocalMachine, member).root for member in task.containers.values()]
        session = None
        try:
            redirect = "2>/dev/null" if retained_stream == "stdout" else ">/dev/null"
            session = await machine.open_process(
                Command(("setsid", "sh", "-c", f"echo ready; exec sleep 3600 {redirect}"))
            )
            assert await asyncio.wait_for(session.read(6), 5) == b"ready\n"
            await asyncio.wait_for(session.close(), 5)
            assert session.process.returncode is not None
            assert not any(root.exists() for root in roots)
            for member in task.containers.values():
                with pytest.raises(MachineTerminated):
                    await member.run(Command(("true",)))
        finally:
            await task.close()
            if session is not None:
                await session.close()

    asyncio.run(scenario())


def test_machine_close_drains_unconsumed_process_session_output(factory):
    async def scenario(machine):
        session = await machine.open_process(
            Command(
                (
                    "sh",
                    "-c",
                    "mkfifo /app/output-ready; echo ready; head -c 131073 /dev/zero; "
                    "echo buffered >/app/output-ready; head -c 1048576 /dev/zero; sleep 3600",
                )
            )
        )
        try:
            assert await asyncio.wait_for(session.read(6), 5) == b"ready\n"
            # The FIFO confirms stdout exceeds asyncio's pipe-reader limit
            # before teardown; the remaining output stays unread by the caller.
            buffered = await machine.run(Command(("cat", "/app/output-ready"), timeout=5))
            assert (buffered.exit_code, buffered.stdout) == (0, b"buffered\n")
            await asyncio.wait_for(machine.close(), 5)
            assert session.process.returncode is not None
            assert await session.read() == b""
            assert not machine.root.exists()
        finally:
            await machine.close()
            await session.close()

    on_machine(factory, SPEC, scenario)


def test_losing_the_host_launcher_reclaims_its_still_running_guest(factory):
    async def scenario(machine):
        # The guest keeper still holds stderr open after this host-side fault.
        signal.pidfd_send_signal(cast(int, machine._keeper_pidfd), signal.SIGKILL)
        with pytest.raises(MachineTerminated):
            await asyncio.wait_for(machine.run(Command(("sleep", "3600"))), 5)
        await machine.close()
        assert not machine.root.exists()

    on_machine(factory, SPEC, scenario)


def test_the_guest_init_keeps_the_keeper_payload_running(factory):
    """The machine's lifecycle sleeper runs as the guest init's child.

    The init is PID 1 of the guest PID namespace, which quiesces that spare
    PID 1 cannot kill, and it respawns the sleeper whenever it dies. The
    sleeper's exact argv stays visible in the namespace because graders also
    spare machines by it.
    """
    find_sleeper = """
import os

for entry in os.scandir("/proc"):
    if not entry.name.isdecimal():
        continue
    try:
        with open(f"/proc/{entry.name}/cmdline", "rb") as stream:
            argv = stream.read().split(b"\\0")
        with open(f"/proc/{entry.name}/stat") as stream:
            fields = stream.read().rsplit(") ", 1)[1].split()
    except OSError:
        continue
    if tuple(argv[:2]) == (b"sleep", b"2147483647"):
        print(int(fields[1]))
"""

    async def scenario(machine):
        async with asyncio.timeout(5):
            while True:
                result = await machine.run(Command(("python3", "-c", find_sleeper)))
                assert result.exit_code == 0, result.stderr
                parents = result.stdout.split()
                if parents:
                    assert parents == [b"1"], f"the sleeper's parent is not the guest init: {parents}"
                    return
                await asyncio.sleep(0.05)

    on_machine(factory, SPEC, scenario)


def test_the_guest_init_reaps_an_orphaned_background_process(factory):
    """A guest process orphaned by its command's exit is collected when it dies.

    Guest code observes a dead daemon through ``kill -0``, ``/proc`` and
    ``pgrep``, as on the Docker backend's ``--init``: a PID-file restart or a
    "service stopped" check fails while the zombie is unreaped. Only the guest
    init can collect the orphan, since reparenting aims at PID 1.
    """

    async def scenario(machine):
        stray = await machine.run(
            Command(("/bin/sh", "-c", "sleep 3600 & printf '%s\\n' \"$!\" > /app/orphan-pid; exit 0"))
        )
        assert stray.exit_code == 0, stray.stderr
        pid = (await machine.run(Command(("cat", "/app/orphan-pid")))).stdout.strip().decode()
        assert pid.isdigit(), pid
        killed = await machine.run(Command(("kill", "-9", pid)))
        assert killed.exit_code == 0, killed.stderr
        # The daemon was orphaned when its command's shell exited, so its zombie
        # belongs to the init. The init must reap it promptly.
        async with asyncio.timeout(5):
            while (await machine.run(Command(("test", "-d", f"/proc/{pid}")))).exit_code == 0:
                await asyncio.sleep(0.05)

    on_machine(factory, SPEC, scenario)


@pytest.mark.parametrize(
    "quiesce",
    [("python3", "-I", "-c", ROOT_DELIVERY_QUIESCE_PROGRAM), ("bash", "-c", ROOT_DELIVERY_QUIESCE_SCRIPT)],
    ids=["python", "shell"],
)
def test_machine_survives_the_root_delivery_quiesce(factory, quiesce):
    """Kill a stray with the quiesce and require the machine to stay open."""

    async def scenario(machine):
        stray = await machine.run(Command(("/bin/sh", "-c", "sleep 3600 & printf '%s\\n' \"$!\" > /app/stray-pid")))
        assert stray.exit_code == 0, stray.stderr
        pid = (await machine.run(Command(("cat", "/app/stray-pid")))).stdout.strip().decode()
        result = await machine.run(Command(quiesce))
        assert result.exit_code == 0, (result.stdout, result.stderr)
        # The quiesce must have killed the stray, and the init must have reaped
        # the orphan: guest code reads a dead process through /proc, so the
        # machine's later commands must not see it still there.
        async with asyncio.timeout(5):
            while (await machine.run(Command(("test", "-d", f"/proc/{pid}")))).exit_code == 0:
                await asyncio.sleep(0.05)
        assert (await machine.run(Command(("printf", "alive")))).stdout == b"alive"

    on_machine(factory, SPEC, scenario)


def test_cancelled_creation_reclaims_a_process_before_its_handle_is_returned(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    create_subprocess = asyncio.create_subprocess_exec

    async def scenario():
        spawned, release = asyncio.Event(), asyncio.Event()
        processes = []

        async def hold_real_process(*args, **kwargs):
            process = await create_subprocess(*args, **kwargs)
            processes.append(process)
            spawned.set()
            await release.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", hold_real_process)
        creation = asyncio.create_task(factory.create(SPEC))
        try:
            await asyncio.wait_for(spawned.wait(), 5)
            roots = tuple(tmp_path.iterdir())
            assert roots
            creation.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(creation, 10)
            assert all(process.returncode is not None for process in processes)
            assert not any(root.exists() for root in roots)
        finally:
            release.set()
            if not creation.done():
                creation.cancel()
            await asyncio.gather(creation, return_exceptions=True)

    asyncio.run(scenario())


def test_repeated_image_creation_cancellation_waits_for_external_unpacker_and_removes_bundle(tmp_path, oci_image):
    image = oci_image((), {})
    unpacker = tmp_path / "umoci"
    factory = make_factory(umoci=unpacker)

    async def scenario():
        connected = asyncio.get_running_loop().create_future()

        def accept(reader, writer):
            connected.set_result((reader, writer))

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        unpacker.write_text(
            f"#!{sys.executable}\n"
            "import json, pathlib, socket, sys\n"
            f"connection = socket.create_connection(('127.0.0.1', {port}))\n"
            "bundle = pathlib.Path(sys.argv[-1])\n"
            "connection.sendall(str(bundle).encode() + b'\\n')\n"
            "assert connection.makefile('rb').readline() == b'finish\\n'\n"
            "(bundle / 'rootfs').mkdir(parents=True)\n"
            "(bundle / 'rootfs' / 'late-write').write_text('completed')\n"
            "(bundle / 'umoci.json').write_text(json.dumps(\n"
            f"    {{'from_descriptor_path': {{'descriptor_walk': [{{'digest': {image.manifest_digest!r}}}]}}}}))\n"
            "connection.sendall(b'written\\n')\n"
        )
        unpacker.chmod(0o755)
        creation = asyncio.create_task(factory.create(MachineSpec(image)))
        reader = writer = None
        parent = None
        released = False
        try:
            reader, writer = await asyncio.wait_for(connected, 5)
            parent = Path((await reader.readline()).decode().strip()).parent
            for message in ("original cancellation", "second cancellation"):
                creation.cancel(message)
                # Deliver both cancellations while the external writer remains
                # blocked, without releasing it between cleanup checkpoints.
                checkpoint = asyncio.Event()
                asyncio.get_running_loop().call_soon(checkpoint.set)
                await checkpoint.wait()
            assert not creation.done()
            assert parent.exists()
            writer.write(b"finish\n")
            released = True
            await writer.drain()
            assert await asyncio.wait_for(reader.readline(), 5) == b"written\n"
            with pytest.raises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(creation, 5)
            assert caught.value.args == ("original cancellation",)
            assert not parent.exists()
        finally:
            if writer is not None and not released:
                writer.write(b"finish\n")
                await writer.drain()
            if reader is not None:
                await reader.read()
            await asyncio.gather(creation, return_exceptions=True)
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            server.close()
            await server.wait_closed()
            if parent is not None and parent.exists():
                shutil.rmtree(parent)

    asyncio.run(scenario())
