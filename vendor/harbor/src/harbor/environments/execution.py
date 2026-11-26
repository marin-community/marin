# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt Shellbox command results to Harbor's environment interface."""

from harbor.environments.base import EXEC_OUTPUT_LIMIT_BYTES, ExecResult, SandboxDiskFullError
from shellbox.machine import Command, ExitReason, Machine, MachineDiskFull


async def execute_shellbox_command(
    machine: Machine,
    command: str,
    *,
    cwd: str | None,
    env: dict[str, str],
    timeout_sec: int | None,
    user: str | None,
) -> ExecResult:
    """Use Bash when available and preserve bounded command output metadata."""
    # Terminus uses Bash pipelines; minimal task images may provide only sh.
    # Pass the script once; quoting it into both branches inflates remote argv.
    shell_command = (
        'if [ -x /bin/bash ]; then exec /bin/bash -c "$1"; '
        'else exec /bin/sh -c "$1"; fi'
    )
    try:
        result = await machine.run(
            Command(
                ("/bin/sh", "-c", shell_command, "harbor-command", command),
                cwd=cwd,
                env=env,
                timeout=timeout_sec,
                user=user,
                output_limit_bytes=EXEC_OUTPUT_LIMIT_BYTES,
            )
        )
    except MachineDiskFull as error:
        raise SandboxDiskFullError(str(error)) from error
    if result.reason is ExitReason.TIMED_OUT:
        raise TimeoutError("Sandbox command timed out")
    return ExecResult(
        return_code=result.exit_code,
        stdout=result.stdout.decode(errors="replace"),
        stderr=result.stderr.decode(errors="replace"),
        stdout_truncated=result.stdout_truncated,
        stderr_truncated=result.stderr_truncated,
    )
