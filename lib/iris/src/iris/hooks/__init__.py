# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task hooks: composable transforms over a task's run command.

A hook wraps a command so the user's process runs *under* it — a profiler
(:mod:`iris.hooks.nsys`), a multi-process GPU supervisor
(:mod:`iris.hooks.multigpu`). Everything the wrapper needs at run time
(rank selection, report upload, signal forwarding) lives in the module its
``wrap`` prepends.

iris does not inject hooks; the entrypoint it schedules is run verbatim. Callers
compose them either programmatically::

    entrypoint.command = MultiGpuHook(nproc=8).wrap(entrypoint.command)

or by writing the equivalent command by hand, since each hook's run-phase module
parses the same arguments ``wrap`` emits::

    python -m iris.hooks.multigpu_main --nproc 8 -- python train.py

Order is the nesting: a hook applied later ends up the outer wrapper.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["TaskHook"]


@runtime_checkable
class TaskHook(Protocol):
    """A transform over a task's run command."""

    def wrap(self, command: Sequence[str]) -> list[str]:
        """Return *command* wrapped to run under this hook."""
