# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task hooks: composable transforms over a task's run command.

A hook wraps a command so the user's process runs *under* it. The profiler
hook lives in :mod:`iris.hooks.nsys`; the multi-process GPU hook lives in
:mod:`iris.jax.multigpu`. Their run-phase modules handle profiling and process
supervision.

iris does not inject hooks; the entrypoint it schedules is run verbatim. Callers
compose them either programmatically::

    from iris.jax.multigpu import MultiGpuHook
    entrypoint.command = MultiGpuHook(nproc=8).wrap(entrypoint.command)

or by writing the equivalent command by hand, since each hook's run-phase module
parses the same arguments ``wrap`` emits::

    python -m iris.jax.multigpu_main --nproc 8 -- python train.py

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
