# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transport-independent Linux command scopes for persistent machines."""

import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from shellbox import process_supervisor

SUPERVISOR_SOURCE = Path(process_supervisor.__file__).read_text()
CLEANUP_TIMEOUT = process_supervisor.CLEANUP_TIMEOUT + 7
TIMEOUT_MARKER = process_supervisor.TIMEOUT_MARKER
FAILURE_MARKER = process_supervisor.FAILURE_MARKER


@dataclass(frozen=True)
class ProcessScope:
    """Own one command and its descendants, including detached and orphaned tools.

    The guest needs Linux procfs and Python 3. Run cancellation through an independent
    transport. Retain the control directory until the machine closes: a late command
    submission must still see the cancellation marker.
    """

    directory: str = field(default_factory=lambda: f"/tmp/.shellbox-process-{uuid.uuid4().hex}")

    python: str = "python3"

    def command(self, argv: tuple[str, ...], timeout: float | None) -> tuple[str, ...]:
        return (
            self.python,
            "-I",
            "-S",
            "-c",
            SUPERVISOR_SOURCE,
            "run",
            self.directory,
            str(timeout) if timeout is not None else "none",
            *argv,
        )

    def cancel_command(self) -> str:
        return shlex.join((self.python, "-I", "-S", "-c", SUPERVISOR_SOURCE, "cancel", self.directory))

    def status_command(self) -> str:
        """Read the supervisor outcome independently of command output truncation."""
        return shlex.join((self.python, "-I", "-S", "-c", SUPERVISOR_SOURCE, "status", self.directory))
