# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior checks for the simulated shell and verifier boundary."""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("shellsim")

from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ExitReason, MachineSpec, PackageSpec, ShellStatus
from shellbox.transfer import download_files

UNSUPPORTED_COMMAND = "unsupported-native-command-with-a-long-name-for-output-truncation-testing"
UNSUPPORTED_OUTPUT_LIMIT_BYTES = 64


@pytest.mark.parametrize(
    "limits",
    [
        {"max_files": 1, "max_bytes": 1024, "max_file_bytes": 1024},
        {"max_files": 8, "max_bytes": 3, "max_file_bytes": 1024},
        {"max_files": 8, "max_bytes": 1024, "max_file_bytes": 1},
    ],
)
def test_download_rejects_file_and_byte_overflow(limits):
    async def check():
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        try:
            result = await machine.run(Command(("sh", "-c", "printf aa > first; printf bb > second")))
            assert result.exit_code == 0
            with pytest.raises(RuntimeError):
                await download_files(machine, ("/workspace",), timeout=10, **limits)
        finally:
            await machine.close()

    asyncio.run(check())


def test_a_command_past_its_timeout_leaves_the_machine_usable() -> None:
    """The timed-out command keeps simulating on its thread; the next command waits for it, then runs."""

    async def scenario():
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        loop = "i=0; while [ $i -lt 200000 ]; do i=$((i+1)); done"
        slow = await machine.run(Command(("sh", "-c", loop), timeout=0.05))
        after = await machine.run(Command(("echo", "after"), timeout=10))
        await machine.close()
        return slow, after

    slow, after = asyncio.run(scenario())
    assert slow.reason is ExitReason.TIMED_OUT
    assert (after.exit_code, after.stdout) == (0, b"after\n")


def test_shell_state_and_one_shot_command() -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec(), env={"INITIAL": "base"}))
        try:
            shell = await machine.open_shell()
            first = await shell.execute(
                'cd /tmp; export ANSWER=hello; answer() { echo "$ANSWER"; }; echo file >/workspace/result'
            )
            assert first.status is ShellStatus.COMPLETED
            second = await shell.execute("pwd; answer")
            assert second.output == b"/tmp\nhello\n"

            verifier = await machine.run(Command(("sh", "-c", 'pwd; cat /workspace/result; echo "$INITIAL:$ANSWER"')))
            assert verifier.stdout == b"/workspace\nfile\nbase:\n"
            after = await shell.execute("pwd; answer")
            assert after.output == b"/tmp\nhello\n"
        finally:
            await machine.close()

    asyncio.run(check())


def test_file_transfer_and_bounded_output(tmp_path: Path) -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
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


def test_run_unsupported_command_returns_failed_result(tmp_path: Path) -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        try:
            source = tmp_path / "numbers.txt"
            source.write_text("1\n2\n")
            await machine.upload(source, "/workspace/numbers.txt")

            result = await machine.run(
                Command(("sh", "-c", UNSUPPORTED_COMMAND), output_limit_bytes=UNSUPPORTED_OUTPUT_LIMIT_BYTES)
            )
            assert result.reason is ExitReason.EXITED
            assert result.exit_code == 127
            assert result.stdout == b""
            assert len(result.stderr) == UNSUPPORTED_OUTPUT_LIMIT_BYTES
            assert result.stderr_truncated

            followup = await machine.run(Command(("cat", "/workspace/numbers.txt")))
            assert (followup.exit_code, followup.stdout) == (0, b"1\n2\n")
        finally:
            await machine.close()

    asyncio.run(check())


def test_shell_session_unsupported_command_returns_failed_update(tmp_path: Path) -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        try:
            source = tmp_path / "numbers.txt"
            source.write_text("1\n2\n")
            await machine.upload(source, "/workspace/numbers.txt")
            shell = await machine.open_shell()

            update = await shell.execute(UNSUPPORTED_COMMAND, output_limit_bytes=UNSUPPORTED_OUTPUT_LIMIT_BYTES)
            assert update.status is ShellStatus.COMPLETED
            assert update.exit_code == 127
            assert len(update.output) == UNSUPPORTED_OUTPUT_LIMIT_BYTES
            assert update.truncated

            followup = await shell.execute("cat /workspace/numbers.txt")
            assert (followup.exit_code, followup.output) == (0, b"1\n2\n")
        finally:
            await machine.close()

    asyncio.run(check())
