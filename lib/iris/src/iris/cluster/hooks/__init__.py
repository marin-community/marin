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
later ends up the outer wrapper. Each hook is self-contained in its own module — the
submit-side spec (plus install script, CLI flags, and any env-var contract it exposes)
alongside its run-phase ``_main`` wrapper:

- ``nsys`` / ``nsys_main`` — ``NsysHook`` + the nsight install + the ``--profile*`` flags,
  and the launch wrapper that runs the command under ``nsys profile``.
- ``multigpu`` / ``multigpu_main`` — ``MultiGpuHook`` + ``build_multigpu_hook`` + the
  ``IRIS_MULTIGPU_*`` rank-env contract, and the per-node process supervisor.

Everything a hook needs at run time — rank selection, output upload, signal forwarding —
lives inside the module its ``wrap`` prepends; the client only knows "it wraps the command".
Which hooks to add, in which order, is ``iris.client.client.collect_hooks``.

This ``__init__`` deliberately holds *only* the ``TaskHook`` protocol and imports no
submodule. ``iris.cluster.types`` imports ``TaskHook`` from here, so pulling in a hook
implementation would drag ``click`` (nsys' flags) or ``iris.cluster.types`` (multigpu's
builder) into every module that touches a job spec — and the latter is an import cycle.
Import the concrete hooks straight from their submodules instead.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["TaskHook"]


@runtime_checkable
class TaskHook(Protocol):
    """A transform over a task's setup script and run command."""

    def setup(self) -> str | None:
        """A build-phase setup script to append, or ``None`` if the hook installs nothing."""

    def wrap(self, command: Sequence[str]) -> list[str]:
        """Return *command* wrapped to run under this hook."""
