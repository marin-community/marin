# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check persistent Bash state, input, job control, and reset in one QEMU guest."""

import argparse
import asyncio
from pathlib import Path

from shellbox.backends.qemu.machine import Acceleration, QemuMachineFactory
from shellbox.machine import MachineSpec, QemuBundle, ShellStatus


async def main(bundle: Path) -> None:
    machine = await QemuMachineFactory(Acceleration.TCG).create(MachineSpec(QemuBundle(bundle)))
    try:
        shell = await machine.open_shell()
        for command in ("cd /tmp", "export HARBOR_VALUE=hello", "harbor_function() { printf '%s' \"$HARBOR_VALUE\"; }"):
            result = await shell.execute(command, wait=5)
            assert result.status is ShellStatus.COMPLETED and result.exit_code == 0, result
        result = await shell.execute("printf '%s:' \"$PWD\"; harbor_function", wait=5)
        assert b"/tmp:hello" in result.output and result.exit_code == 0, result

        result = await shell.execute("read -r value; printf 'read=%s' \"$value\"", wait=0.2)
        assert result.status is ShellStatus.RUNNING, result
        await shell.write(b"input\n")
        result = await shell.read(wait=5)
        assert b"read=input" in result.output and result.exit_code == 0, result

        result = await shell.execute("sleep 30", wait=0.2)
        assert result.status is ShellStatus.RUNNING, result
        await shell.interrupt()
        result = await shell.read(wait=5)
        assert result.status is ShellStatus.COMPLETED and result.exit_code == 130, result
        result = await shell.execute("sleep 20 & jobs", wait=5)
        assert result.status is ShellStatus.COMPLETED and b"Running" in result.output, result

        result = await shell.execute("printf '%04096d' 0", wait=5, output_limit_bytes=128)
        assert result.status is ShellStatus.COMPLETED and result.truncated and len(result.output) == 128, result
        result = await shell.execute("exit", wait=5)
        assert result.status is ShellStatus.RESET, result
        result = await shell.execute("pwd", wait=5)
        assert result.status is ShellStatus.COMPLETED and b"/workspace" in result.output, result
        print("Persistent Bash, input, interrupt, jobs, bounded output, and reset passed")
    finally:
        await machine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    arguments = parser.parse_args()
    asyncio.run(main(arguments.bundle))
