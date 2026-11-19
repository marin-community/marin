# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource installation restores declared file metadata on every backend."""

import stat

import pytest
from shellbox.machine import Command
from taskcompendium.models import InlineFile, TaskResource

from rolloutengine.machines import _install_resources


class RecordingMachine:
    """A minimal Machine stand-in recording uploads and guest commands."""

    backend = "recording"

    def __init__(self) -> None:
        self.uploads: list[tuple[bytes, int, str]] = []
        self.commands: list[Command] = []

    async def upload(self, source, target) -> None:
        self.uploads.append((source.read_bytes(), stat.S_IMODE(source.stat().st_mode), target))

    async def run(self, command: Command):
        self.commands.append(command)

        class Result:
            exit_code = 0
            reason = None
            stdout = b""
            stderr = b""

        return Result

    async def close(self) -> None:
        pass


def _resource(mode: str | None, mtime_ns: int | None) -> TaskResource:
    return TaskResource(
        path="tests/grade.py", source=InlineFile(content_base64="cHJpbnQoJ2dyYWRlZCcp"), mode=mode, mtime_ns=mtime_ns
    )


@pytest.mark.asyncio
async def test_install_restores_declared_mode_and_mtime_in_the_guest():
    machine = RecordingMachine()
    await _install_resources(machine, (_resource("640", 1_700_000_000_123_456_789),))
    assert machine.uploads[0][2] == "/tests/grade.py"
    (command,) = machine.commands
    script = command.argv[2]
    assert command.user == "0"
    assert "chmod 640 /tests/grade.py" in script
    assert "touch -d @1700000000.123456789 /tests/grade.py" in script


@pytest.mark.asyncio
async def test_install_without_declared_metadata_issues_no_guest_commands():
    machine = RecordingMachine()
    await _install_resources(machine, (_resource(None, None),))
    assert machine.uploads[0][1] == 0o644
    assert machine.commands == []


@pytest.mark.asyncio
async def test_install_mode_only_skips_the_touch():
    machine = RecordingMachine()
    await _install_resources(machine, (_resource("0755", None),))
    (command,) = machine.commands
    assert "chmod 755" in command.argv[2]
    assert "touch" not in command.argv[2]
