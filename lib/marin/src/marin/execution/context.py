# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build-phase context for executor step construction.

``ExecutorStep`` and ``StepSpec`` are meant to be built during an executor *build
phase*, not at module-import time. Building a step at import scope freezes the
importing environment's region-specific ``marin_prefix()`` into the pipeline,
which then trips the executor's cross-region guard when the run lands in a
different region.

:func:`executor_context` marks a build phase. Constructing a step outside one
warns once per step name (or raises when ``MARIN_EXECUTOR_STRICT`` is set).
:func:`audit_construction` instead collects each contextless construction with
its full stack, so ``scripts/audit_executor_construction.py`` can enumerate the
migration surface in one import sweep.

Standard frozen-dataclass pickling and ``deepcopy`` bypass ``__post_init__``, so a
worker unpickling a step never trips the guard; only direct construction and
``dataclasses.replace`` do.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import logging
import os
import traceback
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_STRICT_ENV = "MARIN_EXECUTOR_STRICT"


@dataclasses.dataclass(frozen=True)
class ExecutorContext:
    """Marks an executor build phase. Steps may only be constructed inside one."""


@dataclasses.dataclass(frozen=True)
class ContextlessConstruction:
    """A step built outside any build phase, captured by :func:`audit_construction`."""

    kind: str
    name: str
    stack: traceback.StackSummary

    @property
    def site(self) -> traceback.FrameSummary | None:
        """The innermost non-framework frame — where the step is literally constructed."""
        for frame in reversed(self.stack):
            if "/marin/execution/" not in frame.filename and not frame.filename.startswith("<"):
                return frame
        return self.stack[-1] if self.stack else None


_active_context: contextvars.ContextVar[ExecutorContext | None] = contextvars.ContextVar(
    "marin_executor_context", default=None
)
_audit_sink: contextvars.ContextVar[list[ContextlessConstruction] | None] = contextvars.ContextVar(
    "marin_executor_audit_sink", default=None
)
# Names already warned about, so a module-level factory does not log once per step.
_warned_names: set[tuple[str, str]] = set()


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


@contextlib.contextmanager
def audit_construction() -> Iterator[list[ContextlessConstruction]]:
    """Collect contextless constructions instead of warning — for migration scoping.

    Yields a list that accumulates one :class:`ContextlessConstruction` per step
    built outside an :func:`executor_context` while the block is active.
    """
    sink: list[ContextlessConstruction] = []
    token = _audit_sink.set(sink)
    try:
        yield sink
    finally:
        _audit_sink.reset(token)


def _strict_mode() -> bool:
    return os.environ.get(_STRICT_ENV, "").strip().lower() in ("1", "true", "yes")


def check_build_context(kind: str, name: str) -> None:
    """Assert that a step is being constructed inside an :func:`executor_context`.

    A no-op inside a context. Inside an :func:`audit_construction` block the
    construction is recorded. Otherwise it warns once per step name, or raises
    ``RuntimeError`` when ``MARIN_EXECUTOR_STRICT`` is set. Construction at
    module-import scope is the classic cause: it freezes import-time constants
    (e.g. the region from ``marin_prefix()``) into the pipeline.
    """
    if _active_context.get() is not None:
        return

    sink = _audit_sink.get()
    if sink is not None:
        sink.append(ContextlessConstruction(kind=kind, name=name, stack=traceback.extract_stack()[:-1]))
        return

    if _strict_mode():
        raise RuntimeError(
            f"{kind} {name!r} was constructed outside an executor_context(). Build steps inside a "
            f"build_steps()/executor_main flow rather than at module-import scope."
        )

    key = (kind, name)
    if key not in _warned_names:
        _warned_names.add(key)
        logger.warning(
            "%s %r constructed outside an executor_context(); this will become an error. Build steps "
            "inside a build_steps()/executor_main flow, not at module-import scope. Run "
            "scripts/audit_executor_construction.py to list every site.",
            kind,
            name,
        )
