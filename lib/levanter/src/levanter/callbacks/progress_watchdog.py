# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Terminate training after a step or the surrounding pipeline stops progressing."""

import logging
import os
import threading
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum, auto
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


class ProgressState(StrEnum):
    """Where a run sits relative to its progress deadlines."""

    STARTING = auto()
    """No step has completed yet, or the startup grace period is still open."""

    PROGRESSING = auto()
    """The current wait is inside the deadline that governs it."""

    STALLED = auto()
    """The current wait has passed its deadline. The watchdog terminates on it."""

    FINISHED = auto()
    """Training ended and the watchdog no longer holds a deadline."""


@dataclass(frozen=True)
class ProgressHealth:
    """One evaluation of the deadline that governs the current wait.

    ``elapsed`` and ``timeout`` describe whichever wait is active: startup, the
    in-flight train step, the gap since the last lifecycle event, or the startup
    grace period. ``timeout`` is ``None`` when no deadline governs that wait.
    """

    state: ProgressState
    event: ProgressEvent | None = None
    elapsed: float | None = None
    timeout: float | None = None

    @property
    def healthy(self) -> bool:
        return self.state is not ProgressState.STALLED

    def as_timeout(self) -> ProgressTimeout:
        assert self.state is ProgressState.STALLED
        assert self.event is not None and self.elapsed is not None and self.timeout is not None
        return ProgressTimeout(self.event, self.elapsed, self.timeout)


def _deadline(
    event: ProgressEvent,
    elapsed: float,
    timeout: float | None,
    *,
    within: ProgressState = ProgressState.PROGRESSING,
) -> ProgressHealth:
    """Classify one wait. ``within`` is the state to report while it is inside ``timeout``."""
    state = ProgressState.STALLED if timeout is not None and elapsed >= timeout else within
    return ProgressHealth(state, event=event, elapsed=elapsed, timeout=timeout)


class ProgressWatchdog(Callback[Any]):
    """Watch explicit training lifecycle events with step and process deadlines."""

    def __init__(
        self,
        *,
        step_timeout: timedelta | None,
        process_timeout: timedelta | None,
        startup_timeout: timedelta | None = None,
        startup_grace_period: timedelta = _DEFAULT_STARTUP_GRACE_PERIOD,
        diagnostic: Callable[[ProgressTimeout], None] | None = None,
        diagnostic_timeout: timedelta | None = None,
        poll_interval: float = _WATCHDOG_POLL_INTERVAL,
    ) -> None:
        if step_timeout is None and process_timeout is None and startup_timeout is None:
            raise ValueError("at least one watchdog timeout must be set")
        if startup_timeout is not None and startup_timeout.total_seconds() <= 0:
            raise ValueError("startup_timeout must be positive")
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

        self._startup_timeout = startup_timeout.total_seconds() if startup_timeout is not None else None
        self._step_timeout = step_timeout.total_seconds() if step_timeout is not None else None
        self._process_timeout = process_timeout.total_seconds() if process_timeout is not None else None
        self._startup_grace_period = startup_grace_period.total_seconds()
        self._diagnostic = diagnostic
        self._diagnostic_timeout = diagnostic_timeout.total_seconds() if diagnostic_timeout is not None else None
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._created_at = monotonic()
        self._terminal: ProgressHealth | None = None
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

    def health(self) -> ProgressHealth:
        """Evaluate the deadline that governs the current wait, without acting on it.

        Safe to call from another thread: it reads plain Python state the training
        thread published, and never touches JAX or a device.
        """
        return self._evaluate(monotonic())

    def _evaluate(self, now: float) -> ProgressHealth:
        with self._lock:
            # _terminate sets _stop before it runs diagnostics, so the stalled verdict has to
            # outlive it. Otherwise the process reports FINISHED for the diagnostic budget while
            # it is stalled and on its way out.
            if self._terminal is not None:
                return self._terminal
            if self._stop.is_set():
                return ProgressHealth(ProgressState.FINISHED)
            completed_training_step = self._completed_training_step
            training_started_at = self._training_started_at
            active_step_started_at = self._active_step_started_at
            last_progress = self._last_progress

        if not completed_training_step:
            # No step has completed, so the process is still restoring, building caches, or
            # compiling its first step. Elapsed time is all that bounds this: the step deadline
            # stays unarmed until a step completes, and the process deadline has no progress
            # event to measure from.
            return _deadline(
                ProgressEvent.PROCESS_STARTED,
                now - self._created_at,
                self._startup_timeout,
                within=ProgressState.STARTING,
            )

        if training_started_at is None:
            return ProgressHealth(ProgressState.STARTING, timeout=self._startup_grace_period)
        startup_elapsed = now - training_started_at
        if startup_elapsed < self._startup_grace_period:
            return ProgressHealth(ProgressState.STARTING, elapsed=startup_elapsed, timeout=self._startup_grace_period)

        # Both deadlines are evaluated, and a breach of either terminates. The step
        # deadline comes first so an in-flight step reports the deadline it races.
        deadlines: list[ProgressHealth] = []
        if active_step_started_at is not None and self._step_timeout is not None:
            deadlines.append(
                _deadline(ProgressEvent.TRAIN_STEP_STARTED, now - active_step_started_at, self._step_timeout)
            )
        # A completed step always leaves a recorded progress event behind it.
        assert last_progress is not None
        event, event_time = last_progress
        if self._process_timeout is not None:
            deadlines.append(_deadline(event, now - event_time, self._process_timeout))

        for deadline in deadlines:
            if deadline.state is ProgressState.STALLED:
                return deadline
        if deadlines:
            return deadlines[0]
        return ProgressHealth(ProgressState.PROGRESSING, event=event, elapsed=now - event_time)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval):
            health = self._evaluate(monotonic())
            if health.state is ProgressState.STALLED:
                self._terminate(health)
                return

    def _terminate(self, health: ProgressHealth) -> None:
        with self._lock:
            self._terminal = health
        self._stop.set()
        timeout = health.as_timeout()
        try:
            logger.critical(
                "No progress after %s for %.1f seconds (timeout %.1f); terminating with exit code %d",
                timeout.event.value,
                timeout.elapsed,
                timeout.timeout,
                STALLED_TRAINING_EXIT_CODE,
            )
            if self._diagnostic is not None:
                assert self._diagnostic_timeout is not None
                diagnostic = threading.Thread(
                    target=self._run_diagnostic,
                    args=(timeout,),
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
    startup_timeout: timedelta | None = None
    startup_grace_period: timedelta = _DEFAULT_STARTUP_GRACE_PERIOD
    diagnostic_timeout: timedelta | None = None

    def __post_init__(self) -> None:
        if self.step_timeout is not None and self.step_timeout.total_seconds() <= 0:
            raise ValueError("step_timeout must be positive")
        if self.startup_timeout is not None and self.startup_timeout.total_seconds() <= 0:
            raise ValueError("startup_timeout must be positive")
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
        if self.step_timeout is None and self.process_timeout is None and self.startup_timeout is None:
            return None
        runs_diagnostic = self.diagnostic_timeout is not None and process_index == 0
        if runs_diagnostic and diagnostic is None:
            raise ValueError("diagnostic is required when diagnostic_timeout is set")
        return ProgressWatchdog(
            step_timeout=self.step_timeout,
            process_timeout=self.process_timeout,
            startup_timeout=self.startup_timeout,
            startup_grace_period=self.startup_grace_period,
            diagnostic=diagnostic if runs_diagnostic else None,
            diagnostic_timeout=self.diagnostic_timeout if runs_diagnostic else None,
        )
