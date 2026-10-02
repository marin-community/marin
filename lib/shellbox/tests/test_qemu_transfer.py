# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""QEMU serial file transfers against a simulated guest filesystem."""

import asyncio
import base64

import shellsim
from shellbox.backends.qemu.machine import Acceleration, QemuMachine
from shellbox.machine import Command, MachineSpec, QemuBundle


class SimulatedGuestProcess:
    def __init__(self):
        self.stdin = self
        self.stdout = asyncio.StreamReader()
        self.returncode = None
        self.pending = bytearray()
        self.environment = shellsim.Environment()
        self.environment.mkdir("/bin", parents=True)
        self.environment.write_file("/bin/sh", b'#!/bin/sh\nexec sh "$@"\n', mode=0o755)
        self.environment.mkdir("/harbor", parents=True)
        self.environment.mkdir("/guest workspace", parents=True)
        self.environment.write_file("/harbor/busybox", b'#!/bin/sh\n"$@"\n', mode=0o755)

    def write(self, data):
        self.pending.extend(data)

    async def drain(self):
        frames = self.pending.splitlines()
        self.pending.clear()
        assert frames[0] == b"BEGIN"
        assert frames[-1] == b"END"
        assert all(frame.startswith(b"DATA|") for frame in frames[1:-1])
        script = base64.b64decode(b"".join(frame[5:] for frame in frames[1:-1]), validate=True).decode()
        result = self.environment.run(script)
        self.stdout.feed_data(f"RESULT|{result.returncode}\n".encode())
        for label, output in ((b"OUT|", result.stdout), (b"ERR|", result.stderr)):
            if output:
                self.stdout.feed_data(label + base64.b64encode(output) + b"\n")
        self.stdout.feed_data(b"ENDRESULT\n")


def test_uploaded_file_preserves_binary_content_and_selected_mode(tmp_path, monkeypatch):
    source = tmp_path / "source"
    payload = b"\x00\xff\r\n" * 1024
    source.write_bytes(payload)
    source.chmod(0o751)
    script = tmp_path / "script"
    script.write_bytes(b'#!/bin/sh\ncat "/guest workspace/binary"; printf "%s" "$TASK_VALUE"; printf "diagnostic" >&2\n')
    script.chmod(0o751)

    async def transfer():
        process = SimulatedGuestProcess()
        machine = QemuMachine(
            MachineSpec(QemuBundle(tmp_path), workdir="/guest workspace", env={"TASK_VALUE": "configured"}),
            Acceleration.TCG,
        )
        monkeypatch.setattr(machine, "process", process)
        await machine.upload(source, "/guest workspace/binary")
        await machine.upload(script, "/guest workspace/script")
        permissions = await machine.run(Command(("stat", "-c", "%a", "/guest workspace/binary")))
        assert permissions.exit_code == 0
        assert permissions.stdout.strip() == b"751"
        result = await machine.run(Command(("/guest workspace/script",)))
        assert result.exit_code == 0
        assert result.stdout == payload + b"configured"
        assert result.stderr == b"diagnostic"

    asyncio.run(transfer())
