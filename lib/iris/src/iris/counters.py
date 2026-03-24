# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-defined counters for Iris tasks.

Task code can increment named counters during execution; counters are
aggregated across all tasks and exposed in ``JobStatus.counters``.

Usage::

    from iris import counters

    counters.increment("documents_processed", 100)
    counters.increment("validation_errors")

Counter values are written atomically to ``$IRIS_COUNTER_FILE`` (set by
the worker when launching a task container). Outside of a task context the
calls are no-ops.
"""

import json
import os
import threading
from pathlib import Path

_lock = threading.Lock()

IRIS_COUNTER_FILE_ENV = "IRIS_COUNTER_FILE"


def increment(name: str, value: int = 1) -> None:
    """Increment a named counter by ``value`` (default 1). Thread-safe, no-op outside tasks."""
    path = _counter_file_path()
    if path is None:
        return
    with _lock:
        data: dict[str, int] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
        data[name] = data.get(name, 0) + value
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.rename(path)


def _counter_file_path() -> Path | None:
    env = os.environ.get(IRIS_COUNTER_FILE_ENV)
    if env is None:
        return None
    return Path(env)
