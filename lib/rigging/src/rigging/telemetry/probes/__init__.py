# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit lifecycle for Rigging's default NVIDIA and NCCL probes."""

import logging
import math
import threading
from collections.abc import Callable

from rigging.telemetry.probes import nccl, nvidia
from rigging.telemetry.probes.runner import BoundedCommandRunner
from rigging.timing import Deadline

DEFAULT_INTERVAL = 10 * 60.0
MAX_SHUTDOWN_TIMEOUT = 5.0

logger = logging.getLogger(__name__)


class ProbeSession:
    """Own the daemon threads and subprocesses for the default probes."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._runner = BoundedCommandRunner()
        self._threads = [
            threading.Thread(
                target=self._run,
                args=(name, collect),
                name=f"rigging-probe-{name}",
                daemon=True,
            )
            for name, collect in (("nvidia_smi", nvidia.collect), ("nccl_ras", nccl.collect))
        ]
        self._shutdown_lock = threading.Lock()
        self._shutdown = False
        for thread in self._threads:
            thread.start()

    def _run(self, name: str, collect: Callable[[BoundedCommandRunner], None]) -> None:
        while not self._stop.is_set():
            try:
                collect(self._runner)
            except Exception:
                logger.warning("%s telemetry probe failed", name, exc_info=True)
            if self._stop.wait(DEFAULT_INTERVAL):
                return

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop scheduling and reap active probe subprocesses within one budget."""
        try:
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("shutdown timeout must be nonnegative and finite") from None
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("shutdown timeout must be nonnegative and finite")
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
            deadline = Deadline.from_seconds(min(budget, MAX_SHUTDOWN_TIMEOUT))
            self._stop.set()
            self._runner.cancel(deadline.remaining_seconds())
            for thread in self._threads:
                thread.join(deadline.remaining_seconds())

    def __enter__(self) -> "ProbeSession":
        return self

    def __exit__(self, *_args: object) -> None:
        self.shutdown()


def start() -> ProbeSession:
    """Run the default NVIDIA and NCCL collectors until shutdown."""
    return ProbeSession()
