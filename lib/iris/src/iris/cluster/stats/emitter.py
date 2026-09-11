# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shared runner behind every periodic stats emitter."""

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class PeriodicEmitter:
    """Daemon thread that runs ``step`` every ``interval`` seconds until closed.

    The scaffold behind every periodic stats collector: the thread starts at
    construction and waits one full interval before the first step, so freshly
    constructed emitters observe a settled system and tests can drive the step
    directly (with a large ``interval``) without racing the thread. A ``step``
    that raises logs and skips the cycle — an emitter never takes down its host
    component.
    """

    def __init__(self, step: Callable[[], None], *, interval: float, name: str) -> None:
        self._step = step
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            try:
                self._step()
            except Exception:
                logger.warning("%s: emit cycle failed", self._thread.name, exc_info=True)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
