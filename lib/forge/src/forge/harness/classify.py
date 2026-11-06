# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from forge.effects.issues import Hold
from forge.harness.events import ToolReturned

# Source: Taskforge integration sandbox/host_failures.py, used by stage40 infrastructure.
HOST_FAILURES = (
    "no space left on device",
    "container exited",
    "machine exited",
    "sandbox terminated",
    "machine is closed",
    "shellsim terminated",
    "iris sandbox task failed before running",
)


async def host_failure(event: ToolReturned) -> None:
    """Hold on a lost shell host; ordinary command failures remain author feedback."""
    if event.name == "shell" and any(pattern in event.content.lower() for pattern in HOST_FAILURES):
        raise Hold(event.content)
