# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stopping a child process started in its own session (``start_new_session=True``) with its whole process group."""

import asyncio
import contextlib
import os
import signal

TERMINATE_GRACE = 10.0
"""Seconds between SIGTERM and SIGKILL when a child process group is stopped."""


async def stop_process_group(process: asyncio.subprocess.Process) -> None:
    """SIGTERM the process group of a running ``process``, then SIGKILL it after ``TERMINATE_GRACE``."""
    if process.returncode is not None:
        return
    # The group may exit between the check and either signal.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), TERMINATE_GRACE)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
