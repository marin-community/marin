# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transport-independent Linux command scopes for persistent machines."""

import asyncio
import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from shellbox import process_supervisor

SUPERVISOR_SOURCE = Path(process_supervisor.__file__).read_text()
# Allow the guest's cleanup budget, its confirmation poll, and transport latency.
CLEANUP_TIMEOUT = process_supervisor.CLEANUP_TIMEOUT + process_supervisor.CANCEL_CONFIRMATION_GRACE + 5
TIMEOUT_MARKER = process_supervisor.TIMEOUT_MARKER
FAILURE_MARKER = process_supervisor.FAILURE_MARKER
PROCESS_SCOPE_START_ATTEMPTS = 1000
PROCESS_SCOPE_POLL_INTERVAL = 0.01


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


SUPERVISOR_PYTHON_VERSION = "3.12.12"


async def private_supervisor_runtime(platform: str) -> Path:
    """Fetch a host-cached Linux interpreter for images without Python.

    The caller uploads this directory under a private guest prefix. Nothing is
    installed in the task's package database or added to its PATH.
    """
    if platform not in {"linux-x86_64-gnu", "linux-aarch64-gnu", "linux-x86_64-musl", "linux-aarch64-musl"}:
        raise ValueError(f"Unsupported supervisor platform: {platform}")
    target = f"cpython-{SUPERVISOR_PYTHON_VERSION}-{platform}"
    cache = Path.home() / ".cache" / "shellbox" / "python"
    process = await asyncio.create_subprocess_exec(
        "uv",
        "python",
        "install",
        "--no-bin",
        "--install-dir",
        str(cache),
        target,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await process.communicate()
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"Cannot prepare private supervisor Python: {stderr.decode()}")
    return cache / target
