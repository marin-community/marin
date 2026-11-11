# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stopping a timed-out local command's session, without bubblewrap or cgroups."""

import asyncio
import os
import sys

from shellbox.backends.local.machine import _session_members, stop_session

# Ignores SIGTERM, then starts 20 spinning children that ignore it too; one leaves the process group
# as `timeout` does. Prints every PID once all have started.
STUBBORN = """
import os, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pids = []
for index in range(20):
    pid = os.fork()
    if pid == 0:
        if index == 0:
            os.setpgid(0, 0)
        while True:
            pass
    pids.append(pid)
print(" ".join(map(str, [os.getpid(), *pids])), flush=True)
while True:
    time.sleep(1)
"""

# Like the sandbox launcher: run argv[1] in a child, wait for it, and exit with its status.
LAUNCHER = """
import os, sys
child = os.fork()
if child == 0:
    os.execvp("sh", ["sh", "-c", sys.argv[1]])
sys.exit(os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]))
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_stop_session_kills_term_ignoring_descendants_that_left_the_process_group():
    async def scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", STUBBORN, stdout=asyncio.subprocess.PIPE, start_new_session=True
        )
        assert process.stdout is not None
        pids = [int(pid) for pid in (await process.stdout.readline()).split()]
        assert len(pids) == 21 and set(pids) <= set(_session_members(process.pid))
        assert await stop_session(process)
        assert process.returncode is not None
        await asyncio.sleep(0.2)  # Orphans are reaped by init after the session is gone.
        assert not [pid for pid in pids if _alive(pid)]
        assert not _session_members(process.pid)

    asyncio.run(scenario())


def test_stop_session_lets_a_command_handle_sigterm():
    async def scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", LAUNCHER, "trap 'exit 7' TERM; while :; do sleep 0.01; done", start_new_session=True
        )
        await asyncio.sleep(0.5)
        assert await stop_session(process)
        # The launcher was not signalled: it reaped its child and reported the child's own exit.
        assert process.returncode == 7

    asyncio.run(scenario())
