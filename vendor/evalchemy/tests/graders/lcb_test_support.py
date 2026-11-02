# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared child-process helpers for the grader and routing regression tests.

A child is considered leaked if it is still a living direct child of the test process after
the grader returns. CPython's ``multiprocessing`` spawns a ``resource_tracker`` daemon and a
``ps`` is spawned to ask, so both are excluded; the grader worker is what must be caught.
"""

import os
import subprocess


def living_children() -> set[int]:
    """Return the pids of this process's living children, excluding the tools used to ask.

    A snapshot transiently sees the ``ps`` being launched and CPython's
    ``resource_tracker`` daemon; neither is a grader worker, so they are not leaks.
    """
    out = subprocess.run(
        ["ps", "-ww", "-A", "-o", "pid=", "-o", "ppid=", "-o", "command="],
        capture_output=True,
        text=True,
    ).stdout
    mine = str(os.getpid())
    children = set()
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3 or parts[1] != mine:
            continue
        command = parts[2]
        if "resource_tracker" in command:
            continue
        if command.split(None, 1)[0].rsplit("/", 1)[-1] == "ps":
            continue
        children.add(int(parts[0]))
    return children
