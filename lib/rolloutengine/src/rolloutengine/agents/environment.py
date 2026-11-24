# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The task machine as the shell environment of a lifted Harbor agent.

It runs each command as Harbor's Shellbox-backed environments do: through Bash when the image has it, else
``sh``, with Harbor's output limit. Commands run as the machine's default user unless they name one.
"""

from dataclasses import dataclass
from pathlib import Path

from shellbox.machine import Command, ExitReason, Machine
from taskcompendium.importers.harbor.package import harbor_shell

EXEC_OUTPUT_LIMIT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    return_code: int


@dataclass(frozen=True)
class MachineEnvironment:
    """Run an agent's shell commands and uploads on the attempt's task machine.

    ``session_id`` identifies the attempt in error messages.
    """

    machine: Machine
    session_id: str

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
        user: str | None = None,
    ) -> ExecResult:
        """Run ``command`` and return its decoded output.

        Raises:
            TimeoutError: The command outlived ``timeout_sec``.
        """
        result = await self.machine.run(
            Command(
                harbor_shell(command),
                cwd=cwd,
                env=dict(env or {}),
                timeout=timeout_sec,
                user=user,
                output_limit_bytes=EXEC_OUTPUT_LIMIT_BYTES,
            )
        )
        if result.reason is ExitReason.TIMED_OUT:
            raise TimeoutError("Sandbox command timed out")
        assert result.exit_code is not None
        return ExecResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            return_code=result.exit_code,
        )

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        await self.machine.upload(source_path, target_path)
