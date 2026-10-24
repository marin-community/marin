# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt Shellbox command results to Harbor's environment interface."""

from harbor.environments.base import EXEC_OUTPUT_LIMIT_BYTES, ExecResult
from shellbox.machine import Command, ExitReason, Machine


async def execute_shellbox_command(
    machine: Machine,
    argv: tuple[str, ...],
    *,
    cwd: str | None,
    env: dict[str, str],
    timeout_sec: int | None,
    user: str | None,
) -> ExecResult:
    """Execute a resolved environment command and preserve bounded output metadata."""
    result = await machine.run(
        Command(
            argv,
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
