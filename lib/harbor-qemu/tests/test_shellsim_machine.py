# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior checks for the simulated shell and verifier boundary."""

import asyncio
from pathlib import Path

from harbor_qemu.machine import Command, MachineSpec, ShellSimBuiltins, ShellStatus
from harbor_qemu.shellsim_machine import ShellSimMachineFactory


def test_shell_state_and_one_shot_command() -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(ShellSimBuiltins()))
        try:
            shell = await machine.open_shell()
            first = await shell.execute(
                'cd /tmp; export ANSWER=hello; answer() { echo "$ANSWER"; }; echo file >/workspace/result'
            )
            assert first.status is ShellStatus.COMPLETED
            second = await shell.execute("pwd; answer")
            assert second.output == b"/tmp\nhello\n"

            verifier = await machine.run(Command(("sh", "-c", "pwd; cat /workspace/result")))
            assert verifier.stdout == b"/workspace\nfile\n"
            after = await shell.execute("pwd; answer")
            assert after.output == b"/tmp\nhello\n"
        finally:
            await machine.close()

    asyncio.run(check())


def test_file_transfer_and_bounded_output(tmp_path: Path) -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(ShellSimBuiltins()))
        try:
            source = tmp_path / "source"
            source.mkdir()
            (source / "data.txt").write_text("payload")
            await machine.upload(source, "/workspace/imported")
            shell = await machine.open_shell()
            result = await shell.execute("cat /workspace/imported/data.txt; printf 'x'", output_limit_bytes=4)
            assert result.output == b"payl"
            assert result.truncated

            target = tmp_path / "downloaded"
            target.mkdir()
            await machine.download("/workspace/imported", target)
            assert (target / "data.txt").read_text() == "payload"
        finally:
            await machine.close()

    asyncio.run(check())
