# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The crash-respawn hook: client-side spec, its CLI flags, and the respawner↔child env contract.

Everything the respawn backend contributes to *submission* lives here:

- ``RespawnHook`` — the :class:`~iris.cluster.hooks.TaskHook` that wraps the run command with
  ``iris.cluster.hooks.respawn_main``, which restarts the command in place when it dies from
  a crash signal.
- ``respawn_cli_options`` / ``build_respawn_hook`` — the ``--respawn*`` flags and the builder
  that turns them into a hook, so the CLI never constructs the hook itself.
- ``IRIS_RESPAWN_ATTEMPT_ENV`` — stamped on each child with its 0-based attempt index, so the
  workload (and its logs) can tell a respawn from a cold start.

The run-phase half is :mod:`iris.cluster.hooks.respawn_main`; it is imported only in-task
(via ``python -m``) so this module stays free of its subprocess/signal machinery.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import click

# Module path of the in-task respawner this hook prepends (``python -m <module> -- <cmd>``).
_RESPAWN_MAIN_MODULE = "iris.cluster.hooks.respawn_main"

# Stamped on each child with its 0-based attempt index. Iris-private (not a framework
# namespace) and defined here — the contract — so the producer (respawn_main) and any
# consumer cannot drift.
IRIS_RESPAWN_ATTEMPT_ENV = "IRIS_RESPAWN_ATTEMPT"


@dataclass(frozen=True)
class RespawnHook:
    """Restart the command in place when it dies from a crash signal (see ``iris.cluster.hooks.respawn_main``).

    Built for gang workloads whose processes fate-share: when one JAX task
    crashes (e.g. its coordination-service RPC drops), every process in the gang
    aborts, and an iris-level retry pays scheduling, container, and setup costs
    across the whole gang. Respawning inside the still-running task keeps the
    warm container and venv; the workload re-forms its world and resumes from
    its own checkpoints. Deterministic failures (nonzero exits) still propagate
    immediately, so iris's retry budgets remain the fallback.

    Attributes:
        max_restarts: Total in-place restarts allowed per task attempt. Exhausting
            the budget propagates the last exit, handing recovery back to iris.
    """

    max_restarts: int = 100

    def __post_init__(self) -> None:
        if self.max_restarts < 1:
            raise ValueError(f"max_restarts must be >= 1, got {self.max_restarts}")

    def setup(self) -> str | None:
        return None

    def wrap(self, command: Sequence[str]) -> list[str]:
        return [
            "python",
            "-m",
            _RESPAWN_MAIN_MODULE,
            "--max-restarts",
            str(self.max_restarts),
            "--",
            *command,
        ]


def respawn_cli_options(command):
    """Attach the ``--respawn`` flag group to a click command.

    The respawn hook owns these so the flags that configure it live beside it: the
    CLI applies this decorator and calls :func:`build_respawn_hook`, never naming a
    respawn detail itself.
    """
    options = [
        click.option(
            "--respawn",
            is_flag=True,
            default=False,
            help=(
                "Restart the command in place when it dies from a crash signal (SIGABRT/SIGSEGV/…), "
                "instead of failing the task. For gang jobs this rides out a single task's crash "
                "without an iris-level gang reschedule; nonzero exits still fail the task."
            ),
        ),
        click.option(
            "--respawn-max-restarts",
            type=click.IntRange(min=1),
            default=RespawnHook.max_restarts,
            show_default=True,
            help="Total in-place restarts allowed per task attempt before the failure propagates.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def build_respawn_hook(respawn: bool, *, max_restarts: int) -> RespawnHook | None:
    """Build the respawn hook the ``--respawn*`` flags select, or ``None`` if unrequested."""
    if not respawn:
        return None
    return RespawnHook(max_restarts=max_restarts)
