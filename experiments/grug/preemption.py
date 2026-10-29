# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinate SIGTERM shutdown at completed Grug training steps."""

import signal
import threading
from types import FrameType
from typing import Callable, Self

import jax
import numpy as np
from jax.experimental import multihost_utils


class GracefulPreemption:
    """Record SIGTERM and let all ranks stop at one completed step."""

    def __init__(self) -> None:
        self._requested = False
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
        self._requested = True

    def should_stop(self, completed_step: int) -> bool:
        """Collect every rank's stop request and verify the completed step agrees."""
        if jax.process_count() == 1:
            return self._requested
        local = np.asarray([completed_step, int(self._requested)], dtype=np.int64)
        gathered = np.asarray(multihost_utils.process_allgather(local, tiled=False)).reshape(-1, 2)
        if np.any(gathered[:, 0] != completed_step):
            raise RuntimeError(f"Grug ranks reached different completed steps: {gathered[:, 0].tolist()}")
        return bool(np.any(gathered[:, 1]))
