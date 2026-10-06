# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guest command deadlines through a local serial transport."""

import asyncio
import importlib.resources
import os
import sys
from pathlib import Path

import pytest
from shellbox.backends.qemu.machine import Acceleration, QemuMachine
from shellbox.machine import Command, ExitReason, MachineSpec, QemuBundle


async def local_guest(tmp_path: Path, upload_fd: int | None = None, env: dict[str, str] | None = None) -> QemuMachine:
    # The real guest loop uses host applets and pipes in place of guest devices.
    busybox = tmp_path / "busybox"
    busybox.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = kill ]; then\n'
        "  shift\n"
        "  signal=$1\n"
        "  shift\n"
        '  exec /bin/kill "$signal" -- "$@"\n'
        "fi\n"
        'exec "$@"\n'
    )
    busybox.chmod(0o755)
    source = importlib.resources.files("shellbox.backends.qemu").joinpath("guest/init").read_text()
    source = source[source.index("while IFS=") :]
    source = source.replace("/harbor/busybox", str(busybox)).replace("/tmp/harbor-", str(tmp_path / "harbor-"))
    replacements = f"s|/harbor/busybox|{busybox}|g"
    if upload_fd is not None:
        replacements += f";s|/dev/vport0p2|/proc/self/fd/{upload_fd}|g"
    anchor = f"      $busybox chmod 600 {tmp_path / 'harbor-input'}"
    source = source.replace(
        anchor,
        f"      $busybox sed -i '{replacements}' {tmp_path / 'harbor-command.sh'}\n" + anchor,
    )
    source = source.replace(" < /dev/ttyS0", "").replace(" > /dev/ttyS0", " >&1")
    init = tmp_path / "init"
    init.write_text(f"busybox={busybox}\n" + source)
    machine = QemuMachine(MachineSpec(QemuBundle(tmp_path), workdir=str(tmp_path)), Acceleration.TCG)
    machine.process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        str(init),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        pass_fds=() if upload_fd is None else (upload_fd,),
        env=env,
    )
    return machine


@pytest.mark.parametrize("output_limit", [3, 1024])
def test_qemu_command_timeout_preserves_output_workspace_and_later_commands(tmp_path, output_limit):
    async def scenario():
        machine = await local_guest(tmp_path)
        try:
            initial = await machine.run(Command(("sh", "-c", "echo unchanged > answer")))
            assert initial.reason == ExitReason.EXITED
            timed_out = await machine.run(
                Command(
                    ("sh", "-c", "printf partial; printf problem >&2; (sleep 0.4; echo damaged > answer) & wait"),
                    timeout=0.1,
                    output_limit_bytes=output_limit,
                )
            )
            assert timed_out.reason == ExitReason.TIMED_OUT
            assert timed_out.exit_code is None
            assert timed_out.stdout == b"partial"[:output_limit]
            assert timed_out.stderr == b"problem"[:output_limit]
            assert timed_out.stdout_truncated == (output_limit < 7)
            assert timed_out.stderr_truncated == (output_limit < 7)
            # The later command spans the descendant's intended write deadline.
            grade = await machine.run(Command(("sh", "-c", "sleep 0.5; cat answer"), timeout=2))
            assert grade.reason == ExitReason.EXITED
            assert grade.exit_code == 0
            assert grade.stdout == b"unchanged\n"
            final = await machine.run(Command(("sh", "-c", "printf next; exit 7")))
            assert (final.reason, final.exit_code, final.stdout) == (ExitReason.EXITED, 7, b"next")
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_qemu_completed_command_keeps_output_transport_active(tmp_path, monkeypatch):
    monkeypatch.setattr("shellbox.backends.qemu.machine.GUEST_RESPONSE_TIMEOUT", 0.2)
    guest = """
import asyncio
import sys

async def main():
    print('READY', flush=True)
    for line in sys.stdin:
        if line.strip() == 'END':
            print('RESULT|0', flush=True)
            for _ in range(5):
                # Serial output delay is the test input, not a readiness check.
                await asyncio.sleep(0.1)
                print('OUT|eA==', flush=True)
            print('ENDRESULT', flush=True)

asyncio.run(main())
"""

    async def scenario():
        machine = QemuMachine(MachineSpec(QemuBundle(tmp_path)), Acceleration.TCG)
        machine.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            guest,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert await asyncio.wait_for(machine.process.stdout.readline(), timeout=5) == b"READY\n"
            result = await machine.run(Command(("true",), timeout=0.01))
            assert (result.reason, result.exit_code, result.stdout) == (ExitReason.EXITED, 0, b"xxxxx")
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_qemu_unresponsive_guest_raises_infrastructure_error(tmp_path, monkeypatch):
    monkeypatch.setattr("shellbox.backends.qemu.machine.GUEST_RESPONSE_TIMEOUT", 0.01)

    async def scenario():
        machine = QemuMachine(MachineSpec(QemuBundle(tmp_path)), Acceleration.TCG)
        # This serial endpoint echoes requests but never returns a command result.
        machine.process = await asyncio.create_subprocess_exec(
            "/bin/cat",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            with pytest.raises(TimeoutError):
                await machine.run(Command(("true",), timeout=0.01))
            with pytest.raises(RuntimeError, match="not running"):
                await machine.run(Command(("true",)))
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_qemu_close_drains_unread_guest_output(tmp_path):
    async def scenario():
        machine = QemuMachine(MachineSpec(QemuBundle(tmp_path)), Acceleration.TCG)
        # Exceed the 128 KiB reader high-water mark so unread output pauses the pipe.
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'x' * 131073); sys.stdout.flush(); "
            "print('READY', file=sys.stderr, flush=True); sys.stdin.read()",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        machine.process = process
        try:
            assert await asyncio.wait_for(process.stderr.readline(), timeout=5) == b"READY\n"
            await asyncio.wait_for(machine.close(), timeout=10)
            assert process.returncode is not None
        finally:
            if process.returncode is None:
                process.kill()
            await asyncio.wait_for(process.communicate(), timeout=5)

    asyncio.run(scenario())


def test_binary_upload_preserves_bytes_modes_and_shell_after_preflight_failure(tmp_path):
    async def scenario():
        read_fd, write_fd = os.pipe()
        try:
            machine = await local_guest(tmp_path, upload_fd=read_fd)
        finally:
            os.close(read_fd)
        loop = asyncio.get_running_loop()
        protocol = asyncio.StreamReaderProtocol(asyncio.StreamReader())
        transport, _ = await loop.connect_write_pipe(lambda: protocol, os.fdopen(write_fd, "wb", buffering=0))
        machine._upload_writer = asyncio.StreamWriter(transport, protocol, None, loop)
        source = tmp_path / "source"
        source.write_bytes(bytes(range(256)) * 8192)
        source.chmod(0o751)
        target = tmp_path / "new-parent" / "binary ' quote"
        try:
            mask = await machine.run(Command(("sh", "-c", "umask")))
            await machine.upload(source, str(target))
            assert target.read_bytes() == source.read_bytes()
            assert target.stat().st_mode & 0o777 == 0o751
            assert target.parent.stat().st_mode & 0o777 == 0o777 & ~int(mask.stdout.strip(), 8)
            for payload in (b"", b"\x00\xff\r\nEND\x04" * 10000):
                source.write_bytes(payload)
                await machine.upload(source, str(target))
                assert target.read_bytes() == payload
            blocked = tmp_path / "blocked"
            blocked.write_bytes(b"regular file")
            with pytest.raises(RuntimeError, match="preparation failed"):
                await machine.upload(source, str(blocked / "target"))
            result = await machine.run(Command(("sh", "-c", "printf after-preflight")))
            assert (result.exit_code, result.stdout) == (0, b"after-preflight")
            assert blocked.read_bytes() == b"regular file"
        finally:
            await machine.close()

    asyncio.run(scenario())
