# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task hooks: pluggable transforms over a task's setup and run command.

A ``TaskHook`` contributes two things, either of which may be a no-op:

- ``setup()`` — a shell script appended to the job's build-phase setup (e.g. installing
  a profiler CLI), or ``None``.
- ``wrap(command)`` — the run command wrapped so the user's process runs *under* the
  hook (e.g. ``python -m iris.cluster.hooks.nsys_main -- <command>``), or the command
  unchanged.

Hooks are applied in order by the client, and order *is* the nesting: a hook applied
later ends up the outer wrapper. Each hook is self-contained in its own module —
``nsys`` (spec + install script + ``--profile*`` flags) with its run-phase half in
``nsys_main``, and ``multigpu`` (spec). Everything a hook needs at run time — rank
selection, output upload, signal forwarding — lives inside the module its ``wrap``
prepends; the client only knows "it wraps the command".

This package is a leaf: the hook classes take plain values, so the resource- and
spec-aware *builders* (which hooks to add, in which order) live in the client alongside
``collect_hooks``. Only ``MultiGpuHook`` is re-exported here; ``NsysHook`` and the CLI
plumbing stay in ``.nsys`` so importing the package (and hence ``iris.cluster.types``)
does not pull in ``click`` or the profiler's runtime deps.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from iris.cluster.hooks.multigpu import MultiGpuHook

__all__ = ["MultiGpuHook", "TaskHook"]


@runtime_checkable
class TaskHook(Protocol):
    """A transform over a task's setup script and run command."""

    def setup(self) -> str | None:
        """A build-phase setup script to append, or ``None`` if the hook installs nothing."""

    def wrap(self, command: Sequence[str]) -> list[str]:
        """Return *command* wrapped to run under this hook."""
