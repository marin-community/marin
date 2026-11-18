# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shell adapter with scripts near Linux's argument-size limit."""

import pytest
from shellbox.machine import MachineSpec, PackageSpec

from harbor.environments.execution import execute_shellbox_command
from tests.unit.environments.transport import ProcessMachine


@pytest.mark.asyncio
async def test_long_quoted_script_preserves_output_and_exit_status(tmp_path):
    machine = ProcessMachine(
        tmp_path / "guest", MachineSpec(PackageSpec(), workdir="/workspace")
    )
    command = (
        "printf '%s' \"$TRANSPORT_VALUE\"\n"
        + "# "
        + "don't expand $(false); " * 3000
        + "\nprintf '%s' \"a'b\\$c\"\n"
        + "printf '%s' 'failure' >&2\nexit 23\n"
    )
    try:
        result = await execute_shellbox_command(
            machine,
            command,
            cwd="/workspace",
            env={"TRANSPORT_VALUE": "selected"},
            timeout_sec=10,
            user=None,
        )
        assert result.return_code == 23
        assert result.stdout == "selecteda'b$c"
        assert result.stderr == "failure"
    finally:
        await machine.close()
