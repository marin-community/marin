# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tracker that exports Levanter snapshots through direct process telemetry."""

import dataclasses
import logging
import os
import re
import threading
from collections.abc import Mapping
from datetime import timedelta
from enum import IntEnum
from time import monotonic, time
from typing import Any, Optional

import numpy as np
from iris.runtime import telemetry as runtime_telemetry
from rigging import telemetry

from levanter.tracker import Tracker, TrackerConfig
from levanter.tracker.histogram import SummaryStats

logger = logging.getLogger(__name__)

_CURRENT = telemetry.snapshot_attributes("gauge", telemetry.CURRENT_SNAPSHOT)


class TrainingPhase(IntEnum):
    """Numeric training phases persisted to Finelog."""

    INITIALIZING = 0
    TRAINING = 1
    FINISHED = 2


def _metric_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _set(name: str, value: float, *, attributes: dict[str, str] = _CURRENT) -> None:
    telemetry.gauge(_metric_name(name)).set(value, attributes=attributes)


# Keep this well under the reader's enrollment window. Telemetry is best-effort and
# drops records under queue pressure, so several missed beats must not un-enroll a
# live job. The reader's window lives in infra/grafana/src/training_stalls.py.
_PHASE_HEARTBEAT_SECONDS = 60.0
_TRAINING_WATCHDOG_POLL_SECONDS = 60.0
STALLED_TRAINING_EXIT_CODE = 124


class _TrainingProgressWatchdog:
    """Terminate a process whose training loop stopped returning completed steps."""

    def __init__(self, timeout: timedelta, poll_interval: float) -> None:
        self._timeout = timeout.total_seconds()
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._last_progress: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="levanter-training-watchdog", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def record_progress(self) -> None:
        with self._lock:
            self._last_progress = monotonic()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._poll_interval * 2)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval):
            with self._lock:
                last_progress = self._last_progress
            if last_progress is None:
                continue
            stalled_for = monotonic() - last_progress
            if stalled_for < self._timeout:
                continue
            self._stop.set()
            try:
                logger.critical(
                    "Training made no progress for %.1f seconds; terminating process with exit code %d",
                    stalled_for,
                    STALLED_TRAINING_EXIT_CODE,
                )
            finally:
                os._exit(STALLED_TRAINING_EXIT_CODE)


class _PhaseHeartbeat:
    """Republishes the current training phase until the run finishes.

    Stalled-training detection finds a job by its newest `phase` row. Written only
    on transition, a job that hangs before its first step has one row, from process
    start, and the reader must scan all of history to find it. Republishing keeps
    that row recent so the reader can bound its scan.
    """

    def __init__(self, interval: float = _PHASE_HEARTBEAT_SECONDS) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._phase: TrainingPhase | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None

    def publish(self, phase: TrainingPhase) -> None:
        with self._lock:
            self._phase = phase
        _set("phase", float(phase))

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="levanter-phase-heartbeat", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread, self._thread = self._thread, None
            self._stop.set()
        if thread is not None:
            thread.join(timeout=self._interval * 2)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            with self._lock:
                phase = self._phase
            if phase is not None:
                _set("phase", float(phase))


_HEARTBEAT = _PhaseHeartbeat()


def set_training_phase(phase: TrainingPhase) -> None:
    """Publish the current training phase for stalled-training detection."""
    _HEARTBEAT.publish(phase)


def _as_scalar(value: Any) -> float | None:
    if isinstance(value, bool | str | bytes):
        return None
    try:
        array = np.asarray(value)
    except (ValueError, TypeError):
        logger.debug("value of type %s has no array form", type(value).__name__, exc_info=True)
        return None
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        return None
    return float(array)


class TelemetryTracker(Tracker):
    """Export training snapshots and optionally terminate a stalled process.

    The watchdog arms on the first scalar ``train/loss`` and resets on each
    subsequent loss. It remains unarmed during initialization and compilation.
    """

    name: str = "telemetry"

    def __init__(
        self,
        training_stall_timeout: timedelta | None = None,
        *,
        watchdog_poll_interval: float = _TRAINING_WATCHDOG_POLL_SECONDS,
    ) -> None:
        if training_stall_timeout is not None and training_stall_timeout.total_seconds() <= 0:
            raise ValueError("training_stall_timeout must be positive")
        if watchdog_poll_interval <= 0:
            raise ValueError("watchdog_poll_interval must be positive")
        self._training_watchdog = (
            _TrainingProgressWatchdog(training_stall_timeout, watchdog_poll_interval)
            if training_stall_timeout is not None
            else None
        )
        if self._training_watchdog is not None:
            self._training_watchdog.start()
        _set("progress_time_seconds", 0)
        set_training_phase(TrainingPhase.INITIALIZING)
        _HEARTBEAT.start()

    def _publish(self, metrics: Mapping[str, object]) -> None:
        for key, value in metrics.items():
            if isinstance(value, SummaryStats):
                self._publish_summary(key, value)
                continue
            scalar = _as_scalar(value)
            if scalar is not None:
                _set(key, scalar)

    def _publish_summary(self, key: str, stats: SummaryStats) -> None:
        """Export a summary's reduced moments as gauges."""
        # Histogram buckets stay out. A row per bin, per metric, per step is a row
        # count the telemetry store does not absorb — a six-layer MoE router emitted
        # 774 a step. The W&B tracker still records the full bucket shape.
        reduced = {
            "mean": stats.mean,
            "min": stats.min,
            "max": stats.max,
            "variance": stats.variance,
            "rms": stats.rms,
            "count": stats.num,
            "sum": stats.sum,
        }
        for suffix, value in reduced.items():
            scalar = _as_scalar(value)
            if scalar is not None:
                _set(f"{key}_{suffix}", scalar)

    def log_hyperparameters(self, hparams: dict[str, Any]):
        pass

    def log(self, metrics, *, step: Optional[int], commit: Optional[bool] = None):
        if step is not None:
            _set("step", float(step))
            loss = _as_scalar(metrics.get("train/loss"))
            if loss is not None:
                if self._training_watchdog is not None:
                    self._training_watchdog.record_progress()
                _set("progress_time_seconds", time())
                set_training_phase(TrainingPhase.TRAINING)
        self._publish(metrics)

    def log_summary(self, metrics: dict[str, Any]):
        self._publish(metrics)

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        pass

    def finish(self):
        if self._training_watchdog is not None:
            self._training_watchdog.stop()
        set_training_phase(TrainingPhase.FINISHED)
        # The run is over, so there is nothing left to keep enrolled. FINISHED is
        # already published above, and republishing it for the process's remaining
        # lifetime tells the reader nothing new.
        _HEARTBEAT.stop()


@TrackerConfig.register_subclass("telemetry")
@dataclasses.dataclass
class TelemetryConfig(TrackerConfig):
    """Configure direct telemetry and optional stalled-training termination."""

    training_stall_timeout: timedelta | None = None
    """After training begins, hard-exit after this long without a new loss; disabled when unset."""

    def init(self, run_id: Optional[str]) -> Tracker:
        runtime_telemetry.configure(
            "levanter",
            root_run_uid=run_id,
        )
        return TelemetryTracker(training_stall_timeout=self.training_stall_timeout)
