# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage for the ambient Iris context."""

from contextvars import ContextVar, Token

_current_context: ContextVar[object | None] = ContextVar("iris_context", default=None)


def current_context() -> object | None:
    """Return the explicitly scoped or cached Iris context."""
    return _current_context.get()


def has_current_context() -> bool:
    """Return whether an Iris context is already active."""
    return _current_context.get() is not None


def set_context(context: object) -> Token[object | None]:
    """Set the ambient Iris context and return its reset token."""
    return _current_context.set(context)


def reset_context(token: Token[object | None]) -> None:
    """Restore the context state represented by ``token``."""
    _current_context.reset(token)
