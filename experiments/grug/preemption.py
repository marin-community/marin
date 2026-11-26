# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinate SIGTERM shutdown at completed Grug training steps."""

import os
import signal
import threading
import time
from collections.abc import Callable
from types import FrameType
from typing import Self

import jax
import numpy as np
from iris.preemption import SHUTDOWN_MARGIN, TERMINATION_GRACE_ENV
from jax.experimental import multihost_utils


class GracefulPreemption:
    """Record SIGTERM and let all ranks stop at one completed step."""

    def __init__(self) -> None:
        self._requested_at: float | None = None
        self._stopping = False
        self._previous_handler: signal.Handlers | Callable[[int, FrameType | None], object] | None = None

    def install(self) -> None:
        """Handle SIGTERM during the training loop."""
        # Fray's in-process local backend runs callables in a thread. Only the
        # main thread can receive process signals; Iris task ranks run there.
        if threading.current_thread() is not threading.main_thread():
            return
        self._previous_handler = signal.signal(signal.SIGTERM, self._request_stop)

    def __enter__(self) -> Self:
        self.install()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.restore()

    def restore(self) -> None:
        """Restore the previous SIGTERM handler after training."""
        if self._previous_handler is not None:
            signal.signal(signal.SIGTERM, self._previous_handler)

    def _request_stop(self, _signum: int, _frame: FrameType | None) -> None:
        if self._requested_at is None:
            self._requested_at = time.monotonic()

    def should_stop(self, completed_step: int) -> bool:
        """Collect every rank's stop request and verify the completed step agrees."""
        if jax.process_count() == 1:
            self._stopping = self._requested_at is not None
            return self._stopping
        local = np.asarray([completed_step, int(self._requested_at is not None)], dtype=np.int64)
        gathered = np.asarray(multihost_utils.process_allgather(local, tiled=False)).reshape(-1, 2)
        if np.any(gathered[:, 0] != completed_step):
            raise RuntimeError(f"Grug ranks reached different completed steps: {gathered[:, 0].tolist()}")
        self._stopping = bool(np.any(gathered[:, 1]))
        return self._stopping

    def checkpoint_deadline(self) -> float:
        """On all ranks after a stop decision, compute a local monotonic deadline for saving."""
        observed_at = time.monotonic()
        requested_at = self._requested_at
        elapsed_ms = 0 if requested_at is None else max(0, int((observed_at - requested_at) * 1000))
        if jax.process_count() > 1:
            # Collect ages after the ranks agree to stop, so a signal received on a
            # fast rank still counts time spent waiting for the rest of the step.
            ages = multihost_utils.process_allgather(np.asarray(elapsed_ms, dtype=np.int32))
            elapsed_ms = int(np.asarray(ages).max())
        grace = int(os.environ.get(TERMINATION_GRACE_ENV, "0")) / 1000
        return observed_at + grace - elapsed_ms / 1000 - SHUTDOWN_MARGIN.to_seconds()

    def exit_if_preempted(self) -> None:
        """Report preemption so Iris retries the task."""
        if self._stopping or self._requested_at is not None:
            raise SystemExit(128 + signal.SIGTERM)
