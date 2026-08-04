# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Terminate training after a step or the surrounding pipeline stops progressing."""

import logging
import os
import threading
from dataclasses import dataclass
from datetime import timedelta
from time import monotonic
from typing import Any

from levanter.callbacks._core import Callback, ProgressEvent, StepInfo


logger = logging.getLogger(__name__)

_WATCHDOG_POLL_INTERVAL = 60.0
STALLED_TRAINING_EXIT_CODE = 124


class ProgressWatchdog(Callback[Any]):
    """Watch explicit training lifecycle events with step and process deadlines."""

    def __init__(
        self,
        *,
        step_timeout: timedelta | None,
        process_timeout: timedelta | None,
        poll_interval: float = _WATCHDOG_POLL_INTERVAL,
    ) -> None:
        if step_timeout is None and process_timeout is None:
            raise ValueError("at least one watchdog timeout must be set")
        if step_timeout is not None and step_timeout.total_seconds() <= 0:
            raise ValueError("step_timeout must be positive")
        if process_timeout is not None and process_timeout.total_seconds() <= 0:
            raise ValueError("process_timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        self._step_timeout = step_timeout.total_seconds() if step_timeout is not None else None
        self._process_timeout = process_timeout.total_seconds() if process_timeout is not None else None
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._completed_training_step = False
        self._active_step_started_at: float | None = None
        self._last_progress: tuple[ProgressEvent, float] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="levanter-progress-watchdog", daemon=True)
        self._thread.start()

    def on_step(self, info: StepInfo[Any], force: bool = False) -> None:
        del info, force

    def on_event(self, event: ProgressEvent) -> None:
        if event is ProgressEvent.TRAINING_FINISHED:
            self.stop()
            return

        now = monotonic()
        with self._lock:
            if self._stop.is_set():
                return

            if event is ProgressEvent.TRAIN_STEP_STARTED and self._completed_training_step:
                self._active_step_started_at = now
            else:
                self._active_step_started_at = None

            if event is ProgressEvent.TRAIN_STEP_FINISHED:
                self._completed_training_step = True

            if self._completed_training_step:
                self._last_progress = (event, now)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._poll_interval * 2)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval):
            now = monotonic()
            with self._lock:
                active_step_started_at = self._active_step_started_at
                last_progress = self._last_progress

            if (
                active_step_started_at is not None
                and self._step_timeout is not None
                and now - active_step_started_at >= self._step_timeout
            ):
                self._terminate(ProgressEvent.TRAIN_STEP_STARTED, now - active_step_started_at, self._step_timeout)
                return

            if last_progress is not None and self._process_timeout is not None:
                event, event_time = last_progress
                if now - event_time >= self._process_timeout:
                    self._terminate(event, now - event_time, self._process_timeout)
                    return

    def _terminate(self, event: ProgressEvent, elapsed: float, timeout: float) -> None:
        self._stop.set()
        try:
            logger.critical(
                "No progress after %s for %.1f seconds (timeout %.1f); terminating with exit code %d",
                event.value,
                elapsed,
                timeout,
                STALLED_TRAINING_EXIT_CODE,
            )
        finally:
            os._exit(STALLED_TRAINING_EXIT_CODE)


@dataclass(frozen=True)
class ProgressWatchdogConfig:
    """Configure step-local and process-wide training progress deadlines."""

    step_timeout: timedelta | None = None
    process_timeout: timedelta | None = None

    def __post_init__(self) -> None:
        if self.step_timeout is not None and self.step_timeout.total_seconds() <= 0:
            raise ValueError("step_timeout must be positive")
        if self.process_timeout is not None and self.process_timeout.total_seconds() <= 0:
            raise ValueError("process_timeout must be positive")

    def create(self) -> ProgressWatchdog | None:
        if self.step_timeout is None and self.process_timeout is None:
            return None
        return ProgressWatchdog(step_timeout=self.step_timeout, process_timeout=self.process_timeout)
