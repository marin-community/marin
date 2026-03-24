# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-defined counters for Iris tasks.

Task code can increment named counters during execution; counters are
aggregated across all tasks and exposed in ``JobStatus.counters``.

Usage::

    from iris import counters

    counters.increment("documents_processed", 100)
    counters.increment("validation_errors")

Counter values are accumulated in-memory and flushed to disk periodically
(every ``_FLUSH_INTERVAL_SECONDS``) by a background thread. The worker
monitor loop reads the file each poll cycle and forwards values through
the heartbeat. ``flush()`` is also called on clean process exit via
``atexit``.

The environment variable ``$IRIS_COUNTER_FILE`` (set by the worker when
launching a task container) controls the output path. Outside of a task
context, all calls are no-ops.
"""

import atexit
import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_counters: dict[str, int] = {}
_dirty = False
_flush_thread: threading.Thread | None = None

IRIS_COUNTER_FILE_ENV = "IRIS_COUNTER_FILE"

_FLUSH_INTERVAL_SECONDS = 5.0


def increment(name: str, value: int = 1) -> None:
    """Increment a named counter by ``value`` (default 1).

    O(1) in-memory update — no disk IO. Thread-safe, no-op outside tasks.
    """
    global _dirty
    path = _counter_file_path()
    if path is None:
        return
    with _lock:
        _counters[name] = _counters.get(name, 0) + value
        _dirty = True
        _ensure_flush_thread(path)


def flush() -> None:
    """Write accumulated counters to disk atomically.

    Called periodically by the background flush thread and on process exit.
    Safe to call from any thread.
    """
    global _dirty
    path = _counter_file_path()
    if path is None:
        return
    with _lock:
        if not _dirty:
            return
        snapshot = dict(_counters)
        _dirty = False
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot))
    tmp.rename(path)


def _ensure_flush_thread(path: Path) -> None:
    """Lazily start the background flush thread on first increment.

    Must be called while ``_lock`` is held.
    """
    global _flush_thread
    if _flush_thread is not None:
        return
    _flush_thread = threading.Thread(target=_flush_loop, args=(path,), daemon=True, name="iris-counter-flush")
    _flush_thread.start()
    atexit.register(flush)


def _flush_loop(path: Path) -> None:
    """Background loop that flushes counters to disk every ``_FLUSH_INTERVAL_SECONDS``."""
    stop = threading.Event()
    while not stop.wait(_FLUSH_INTERVAL_SECONDS):
        try:
            flush()
        except Exception:
            logger.debug("Counter flush failed", exc_info=True)


def _counter_file_path() -> Path | None:
    env = os.environ.get(IRIS_COUNTER_FILE_ENV)
    if env is None:
        return None
    return Path(env)
