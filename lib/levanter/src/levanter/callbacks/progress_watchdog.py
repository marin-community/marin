# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Terminate training after a step or the surrounding pipeline stops progressing."""

import logging
import os
import threading
from dataclasses import dataclass
from datetime import timedelta
from time import monotonic
from typing import Any, Callable

from levanter.callbacks._core import Callback, ProgressEvent, StepInfo


logger = logging.getLogger(__name__)

_WATCHDOG_POLL_INTERVAL = 60.0
_DEFAULT_STARTUP_GRACE_PERIOD = timedelta(hours=1)
STALLED_TRAINING_EXIT_CODE = 124


@dataclass(frozen=True)
class ProgressTimeout:
    """Progress deadline that triggered watchdog termination."""

    event: ProgressEvent
    elapsed: float
    timeout: float


class ProgressWatchdog(Callback[Any]):
    """Watch explicit training lifecycle events with step and process deadlines."""

    def __init__(
        self,
        *,
        step_timeout: timedelta | None,
        process_timeout: timedelta | None,
        startup_grace_period: timedelta = _DEFAULT_STARTUP_GRACE_PERIOD,
        diagnostic: Callable[[ProgressTimeout], None] | None = None,
        diagnostic_timeout: timedelta | None = None,
        poll_interval: float = _WATCHDOG_POLL_INTERVAL,
    ) -> None:
        if step_timeout is None and process_timeout is None:
            raise ValueError("at least one watchdog timeout must be set")
        if step_timeout is not None and step_timeout.total_seconds() <= 0:
            raise ValueError("step_timeout must be positive")
        if process_timeout is not None and process_timeout.total_seconds() <= 0:
            raise ValueError("process_timeout must be positive")
        if startup_grace_period.total_seconds() < 0:
            raise ValueError("startup_grace_period must be non-negative")
        if diagnostic is not None and diagnostic_timeout is None:
            raise ValueError("diagnostic_timeout is required when diagnostic is set")
        if diagnostic_timeout is not None and diagnostic_timeout.total_seconds() <= 0:
            raise ValueError("diagnostic_timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        self._step_timeout = step_timeout.total_seconds() if step_timeout is not None else None
        self._process_timeout = process_timeout.total_seconds() if process_timeout is not None else None
        self._startup_grace_period = startup_grace_period.total_seconds()
        self._diagnostic = diagnostic
        self._diagnostic_timeout = diagnostic_timeout.total_seconds() if diagnostic_timeout is not None else None
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._completed_training_step = False
        self._training_started_at: float | None = None
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

            if event is ProgressEvent.TRAIN_STEP_STARTED and self._training_started_at is None:
                self._training_started_at = now

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
                training_started_at = self._training_started_at
                active_step_started_at = self._active_step_started_at
                last_progress = self._last_progress

            if training_started_at is None or now - training_started_at < self._startup_grace_period:
                continue

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
            if self._diagnostic is not None:
                assert self._diagnostic_timeout is not None
                diagnostic = threading.Thread(
                    target=self._run_diagnostic,
                    args=(ProgressTimeout(event, elapsed, timeout),),
                    name="levanter-progress-diagnostic",
                    daemon=True,
                )
                diagnostic.start()
                diagnostic.join(self._diagnostic_timeout)
                if diagnostic.is_alive():
                    logger.error("Progress diagnostic exceeded its %.1f second budget", self._diagnostic_timeout)
        finally:
            os._exit(STALLED_TRAINING_EXIT_CODE)

    def _run_diagnostic(self, timeout: ProgressTimeout) -> None:
        try:
            assert self._diagnostic is not None
            self._diagnostic(timeout)
        except Exception:
            logger.exception("Progress diagnostic failed")


@dataclass(frozen=True)
class ProgressWatchdogConfig:
    """Configure the startup grace period and training progress deadlines."""

    step_timeout: timedelta | None = None
    process_timeout: timedelta | None = None
    startup_grace_period: timedelta = _DEFAULT_STARTUP_GRACE_PERIOD
    diagnostic_timeout: timedelta | None = None

    def __post_init__(self) -> None:
        if self.step_timeout is not None and self.step_timeout.total_seconds() <= 0:
            raise ValueError("step_timeout must be positive")
        if self.process_timeout is not None and self.process_timeout.total_seconds() <= 0:
            raise ValueError("process_timeout must be positive")
        if self.startup_grace_period.total_seconds() < 0:
            raise ValueError("startup_grace_period must be non-negative")
        if self.diagnostic_timeout is not None and self.diagnostic_timeout.total_seconds() <= 0:
            raise ValueError("diagnostic_timeout must be positive")

    def create(
        self,
        *,
        process_index: int = 0,
        diagnostic: Callable[[ProgressTimeout], None] | None = None,
    ) -> ProgressWatchdog | None:
        if self.step_timeout is None and self.process_timeout is None:
            return None
        if self.diagnostic_timeout is not None and process_index != 0:
            return None
        if self.diagnostic_timeout is not None and diagnostic is None:
            raise ValueError("diagnostic is required when diagnostic_timeout is set")
        return ProgressWatchdog(
            step_timeout=self.step_timeout,
            process_timeout=self.process_timeout,
            startup_grace_period=self.startup_grace_period,
            diagnostic=diagnostic if self.diagnostic_timeout is not None else None,
            diagnostic_timeout=self.diagnostic_timeout,
        )
