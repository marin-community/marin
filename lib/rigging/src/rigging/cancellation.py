# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Context-local cancellation for synchronous operations."""

import contextlib
import logging
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from threading import Event, Lock

logger = logging.getLogger(__name__)

CancellationCallback = Callable[[str], None]


class CancellationToken:
    """Send one cancellation reason to registered operation callbacks."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._reason: str | None = None
        self._callbacks: dict[int, CancellationCallback] = {}
        self._next_callback_id = 0

    @property
    def cancelled(self) -> bool:
        """Return true after cancellation."""
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """Return the first cancellation reason."""
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation and return true if cancellation occurred."""
        return self._event.wait(timeout)

    def cancel(self, reason: str) -> None:
        """Cancel once and call each registered callback."""
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
            callbacks = tuple(self._callbacks.values())
            self._callbacks.clear()

        for callback in callbacks:
            try:
                callback(reason)
            except Exception:
                logger.exception("Cancellation callback failed")

    def add_callback(self, callback: CancellationCallback) -> Callable[[], None]:
        """Register a callback and return its removal function.

        The callback runs immediately when this token is already cancelled.
        """
        with self._lock:
            reason = self._reason
            if reason is None:
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback

        if reason is not None:
            callback(reason)
            return lambda: None

        def remove_callback() -> None:
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return remove_callback


_current_cancellation_token: ContextVar[CancellationToken | None] = ContextVar(
    "current_cancellation_token", default=None
)


@contextlib.contextmanager
def cancellation_scope(token: CancellationToken) -> Iterator[None]:
    """Set a cancellation token for synchronous work in this context."""
    context_token = _current_cancellation_token.set(token)
    try:
        yield
    finally:
        _current_cancellation_token.reset(context_token)


def current_cancellation_token() -> CancellationToken | None:
    """Return the cancellation token for the current context."""
    return _current_cancellation_token.get()
