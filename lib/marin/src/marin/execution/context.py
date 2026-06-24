# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build-phase context for executor step construction.

``ExecutorStep`` and ``StepSpec`` are meant to be built during an executor
*build phase*, not at module-import time. Building a step at import scope freezes
import-environment constants into the pipeline — most damagingly the
region-specific prefix resolved by ``marin_prefix()``, which then trips the
executor's cross-region guard when the run lands in a different region.

:func:`executor_context` marks a legitimate build phase. The step constructors
call :func:`check_build_context`, which detects construction outside any context
and reports the offending ``file:line`` (raising when ``MARIN_EXECUTOR_STRICT``
is set). Standard frozen-dataclass pickling and ``deepcopy`` bypass
``__post_init__``, so a worker unpickling a step never trips the guard; only
direct construction and ``dataclasses.replace`` do.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import logging
import os
import sys
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_STRICT_ENV = "MARIN_EXECUTOR_STRICT"
_EXECUTION_PKG = os.path.dirname(os.path.abspath(__file__))


@dataclasses.dataclass(frozen=True)
class ExecutorContext:
    """Marks an executor build phase.

    Attributes:
        prefix: Optional explicit storage prefix for the build. ``None`` (the
            default) leaves step output paths prefix-relative so the
            :class:`~marin.execution.executor.Executor` anchors them under the
            run prefix — the region-portable path.
    """

    prefix: str | None = None


_active_context: contextvars.ContextVar[ExecutorContext | None] = contextvars.ContextVar(
    "marin_executor_context", default=None
)


@contextlib.contextmanager
def executor_context(prefix: str | None = None) -> Iterator[ExecutorContext]:
    """Open an executor build phase.

    Steps constructed inside the block are exempt from the
    contextless-construction guard. Nesting is allowed; the innermost context
    wins for the duration of the block.
    """
    new_context = ExecutorContext(prefix=prefix)
    token = _active_context.set(new_context)
    try:
        yield new_context
    finally:
        _active_context.reset(token)


def current_executor_context() -> ExecutorContext | None:
    """Return the active :class:`ExecutorContext`, or ``None`` outside a build phase."""
    return _active_context.get()


def _strict_mode() -> bool:
    return os.environ.get(_STRICT_ENV, "").strip().lower() in ("1", "true", "yes")


_warned_sites: set[tuple[str, int]] = set()


def _construction_site() -> tuple[str, int, str]:
    """Return ``(filename, lineno, function)`` of the nearest construction frame
    outside the execution package.

    Walks ``f_back`` directly (no source-line reads) so the contextless path
    stays cheap even when many module-level constructions fire during import.
    """
    frame = sys._getframe(2)  # skip _construction_site + check_build_context
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith("<") and not filename.startswith(_EXECUTION_PKG):
            return filename, frame.f_lineno, frame.f_code.co_name
        frame = frame.f_back
    return "<unknown>", 0, "<unknown>"


def check_build_context(kind: str, name: str) -> None:
    """Assert that a step is being constructed inside an :func:`executor_context`.

    A no-op inside a context. Outside one, warns once per construction site
    (pointing at the offending ``file:line``) or raises ``RuntimeError`` when
    ``MARIN_EXECUTOR_STRICT`` is set. Construction at module-import scope is the
    classic cause: it freezes import-time constants (e.g. the region from
    ``marin_prefix()``) into the pipeline.
    """
    if _active_context.get() is not None:
        return

    filename, lineno, function = _construction_site()
    message = (
        f"{kind} {name!r} constructed at {filename}:{lineno} (in {function}) outside an "
        f"executor_context(). Build steps inside a build_steps()/executor_main flow rather "
        f"than at module-import scope, so import-time constants (e.g. the region from "
        f"marin_prefix()) are not frozen into the pipeline."
    )
    if _strict_mode():
        raise RuntimeError(message)

    site = (filename, lineno)
    if site not in _warned_sites:
        _warned_sites.add(site)
        logger.warning(message)
