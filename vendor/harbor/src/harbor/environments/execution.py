# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt Shellbox command results to Harbor's environment interface."""

import shlex

from harbor.environments.base import EXEC_OUTPUT_LIMIT_BYTES, ExecResult
from shellbox.machine import Command, ExitReason, Machine


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
    shell_command = (
        "if [ -x /bin/bash ]; then set -- /bin/bash; else set -- /bin/sh; fi; "
        f'exec "$@" -c {shlex.quote(command)}'
    )
    result = await machine.run(
        Command(
            ("/bin/sh", "-c", shell_command),
            cwd=cwd,
            env=env,
            timeout=timeout_sec,
            user=user,
            output_limit_bytes=EXEC_OUTPUT_LIMIT_BYTES,
        )
    )
    if result.reason is ExitReason.TIMED_OUT:
        raise TimeoutError("Sandbox command timed out")
    return ExecResult(
        return_code=result.exit_code,
        stdout=result.stdout.decode(errors="replace"),
        stderr=result.stderr.decode(errors="replace"),
        stdout_truncated=result.stdout_truncated,
        stderr_truncated=result.stderr_truncated,
    )
