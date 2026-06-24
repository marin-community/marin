# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build-phase context for executor step construction.

``ExecutorStep`` and ``StepSpec`` must be built during an executor *build phase*,
not at module-import time. Building a step at import scope freezes the importing
environment's region-specific ``marin_prefix()`` into the pipeline, which then
trips the executor's cross-region guard when the run lands in a different region.

:func:`executor_context` marks a build phase. Constructing a step outside one
raises. Standard frozen-dataclass pickling and ``deepcopy`` bypass
``__post_init__``, so a worker unpickling a step never trips the guard; only
direct construction and ``dataclasses.replace`` do.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from collections.abc import Iterator


@dataclasses.dataclass(frozen=True)
class ExecutorContext:
    """Marks an executor build phase. Steps may only be constructed inside one."""


_active_context: contextvars.ContextVar[ExecutorContext | None] = contextvars.ContextVar(
    "marin_executor_context", default=None
)


@contextlib.contextmanager
def executor_context() -> Iterator[ExecutorContext]:
    """Open an executor build phase.

    Steps constructed inside the block are exempt from the construction guard.
    Nesting is allowed; the block restores the previous state on exit.
    """
    new_context = ExecutorContext()
    token = _active_context.set(new_context)
    try:
        yield new_context
    finally:
        _active_context.reset(token)


def current_executor_context() -> ExecutorContext | None:
    """Return the active :class:`ExecutorContext`, or ``None`` outside a build phase."""
    return _active_context.get()


def check_build_context(kind: str, name: str) -> None:
    """Raise when a step is constructed outside an :func:`executor_context`."""
    if _active_context.get() is not None:
        return

    raise RuntimeError(
        f"{kind} {name!r} was constructed outside an executor_context(). Build steps inside a "
        f"build_steps()/executor_main flow (wrapped in `with executor_context():`) rather than at "
        f"module-import scope."
    )
