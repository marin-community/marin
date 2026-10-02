# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""QEMU transfer scripts executed at their command boundary without a guest."""

import asyncio
import subprocess

from shellbox.backends.qemu.machine import Acceleration, QemuMachine
from shellbox.machine import ExitReason, MachineSpec, QemuBundle, Result


def test_uploaded_file_preserves_binary_content_and_selected_mode(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"\x00\xff\r\n")
    source.chmod(0o751)
    target = tmp_path / "guest workspace" / "script"

    async def local_command(_machine, command):
        script = command.argv[2].replace("/harbor/busybox ", "")
        process = subprocess.run((command.argv[0], command.argv[1], script), capture_output=True, check=False)
        return Result(process.returncode, process.stdout, process.stderr, False, False, ExitReason.EXITED)

    monkeypatch.setattr(QemuMachine, "run", local_command)
    machine = QemuMachine(MachineSpec(QemuBundle(tmp_path)), Acceleration.TCG)
    asyncio.run(machine.upload(source, str(target)))
    assert target.read_bytes() == b"\x00\xff\r\n"
    assert target.stat().st_mode & 0o777 == 0o751
