# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep bounded guest boot evidence when an emulator stops producing output."""

import asyncio
import os
import random
import sys

import pytest
from shellbox.backends.qemu.machine import Acceleration, QemuMachine, QemuMachineFactory
from shellbox.machine import MachineSpec, QemuBundle


@pytest.mark.parametrize(
    "payload", [b"case-input\x00\xff\n" * 100_000, random.Random(42).randbytes(131072)], ids=["compressible", "binary"]
)
def test_file_upload_preserves_bytes_through_serial_guest_protocol(tmp_path, payload):
    uploaded, wire_bytes = asyncio.run(upload_serial(tmp_path, payload))
    assert uploaded == payload
    if len(payload) > 1_000_000:
        assert wire_bytes < len(payload) // 20


def test_directory_upload_preserves_nested_binary_files(tmp_path):
    payload = random.Random(7).randbytes(131072)
    uploaded, _ = asyncio.run(upload_serial(tmp_path, payload, "nested/data.bin"))
    assert uploaded == payload


async def upload_serial(tmp_path, payload, source_relative=""):
    guest = tmp_path / "guest.py"
    guest.write_text(
        "import base64, subprocess, sys\n"
        "from pathlib import Path\n"
        "data = bytearray()\n"
        "total = 0\n"
        "for line in sys.stdin.buffer:\n"
        "    if line == b'BEGIN\\n':\n"
        "        data.clear()\n"
        "    elif line.startswith(b'DATA|'):\n"
        "        data.extend(line[5:].strip())\n"
        "    elif line == b'END\\n':\n"
        "        total += len(data)\n"
        "        Path(sys.argv[1]).write_text(str(total))\n"
        "        script = base64.b64decode(data).decode().replace('/harbor/busybox ', '')\n"
        "        result = subprocess.run(['/bin/sh'], input=script.encode(), capture_output=True)\n"
        "        print('RESULT|' + str(result.returncode), flush=True)\n"
        "        if result.stderr:\n"
        "            print('ERR|' + base64.b64encode(result.stderr).decode(), flush=True)\n"
        "        print('ENDRESULT', flush=True)\n"
    )
    source, target = tmp_path / "source", tmp_path / "quoted ' directory" / "target.bin"
    source_file = source / source_relative if source_relative else source
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(payload)
    machine = QemuMachine(MachineSpec(QemuBundle(tmp_path), workdir=str(tmp_path)), Acceleration.TCG)
    # The external guest fixture consumes the real serial framing and executes
    # the upload script with host equivalents of the guest's BusyBox tools.
    machine.process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(guest),
        str(tmp_path / "wire-bytes"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await machine.upload(source, str(target.relative_to(tmp_path)))
        uploaded = target / source_relative if source_relative else target
        return uploaded.read_bytes(), int((tmp_path / "wire-bytes").read_text())
    finally:
        await machine.close()


def test_upload_deadline_covers_guest_not_reading_serial_input(tmp_path, monkeypatch):
    monkeypatch.setattr("shellbox.backends.qemu.machine.DEFAULT_COMMAND_TIMEOUT", 0.1)
    returncode = asyncio.run(stalled_upload(tmp_path))
    assert returncode is not None


async def stalled_upload(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(random.Random(3).randbytes(131072))
    machine = QemuMachine(MachineSpec(QemuBundle(tmp_path)), Acceleration.TCG)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    machine.process = process
    try:
        with pytest.raises(RuntimeError, match="QEMU upload timed_out"):
            await machine.upload(source, "/tmp/target.bin")
        return process.returncode
    finally:
        await machine.close()


def test_stalled_guest_reports_boot_tail_and_cleans_up_process(tmp_path, monkeypatch):
    # Exercise the real subprocess pipes and timeout with a stalled emulator.
    executable = tmp_path / "qemu-system-x86_64"
    executable.write_text(
        "#!/bin/sh\n"
        f"echo $$ > '{tmp_path / 'pid'}'\n"
        "printf '%s\\n' 'early-output-discarded'\n"
        f"printf '%s\\n' '{'x' * 6000}'\n"
        "printf '%s\\n' 'mounting-rootfs-stalled'\n"
        "exec sleep 60\n"
    )
    executable.chmod(0o755)
    (tmp_path / "vmlinuz").touch()
    (tmp_path / "initramfs.cpio.gz").touch()
    monkeypatch.setattr("shellbox.backends.qemu.machine.BOOT_IDLE_TIMEOUT", 0.25)

    with pytest.raises(TimeoutError) as failure:
        asyncio.run(QemuMachineFactory(Acceleration.TCG).create(MachineSpec(QemuBundle(tmp_path))))

    detail = str(failure.value)
    assert "mounting-rootfs-stalled" in detail
    assert "early-output-discarded" not in detail
    assert "boot_elapsed=" in detail and "idle_elapsed=" in detail
    assert len(detail) < 4500
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)
