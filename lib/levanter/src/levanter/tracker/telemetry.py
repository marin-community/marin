# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tracker that exports Levanter snapshots through direct process telemetry."""

import dataclasses
import logging
import os
import re
import threading
from collections.abc import Mapping
from enum import IntEnum
from time import time
from typing import Any, Optional

import jax
import numpy as np
from iris.runtime import telemetry as runtime_telemetry
from rigging import telemetry
from rigging.telemetry.probes import nccl, nccl_client

from levanter.callbacks.progress_watchdog import ProgressTimeout
from levanter.tracker import BackgroundTracker, Tracker, TrackerConfig, get_tracker
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
_NCCL_RAS_POLL_SECONDS = 10 * 60.0
_NCCL_STALL_CAPTURE_SECONDS = 8.0
_STALL_TELEMETRY_FLUSH_SECONDS = 5.0


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


def _start_nccl_ras_probe() -> nccl.NcclRasSession | None:
    try:
        if not telemetry.runtime_status().configured:
            return None
        if os.environ.get(nccl_client.NCCL_RAS_ENABLE_ENV, "1") == "0":
            return None
        if jax.default_backend() != "gpu" or jax.process_index() != 0:
            return None
        return nccl.start(interval=_NCCL_RAS_POLL_SECONDS)
    except RuntimeError:
        logger.warning("could not start NCCL RAS telemetry", exc_info=True)
        return None


def capture_stall_diagnostics(timeout: ProgressTimeout) -> None:
    """Capture bounded process-zero diagnostics immediately before watchdog exit."""
    try:
        tracker = get_tracker("telemetry")
    except KeyError:
        logger.warning("No telemetry tracker is available for stalled-training diagnostics")
        return
    if isinstance(tracker, BackgroundTracker):
        tracker = tracker.wrapped
    if not isinstance(tracker, TelemetryTracker):
        logger.warning("The active telemetry tracker does not support stalled-training diagnostics")
        return
    logger.error(
        "Capturing NCCL RAS after %s made no progress for %.1f seconds",
        timeout.event.value,
        timeout.elapsed,
    )
    tracker.capture_stall_diagnostics()


class TelemetryTracker(Tracker):
    """Export training snapshots and phase transitions."""

    name: str = "telemetry"

    def __init__(self) -> None:
        self._nccl_ras_probe = _start_nccl_ras_probe()
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
                _set("progress_time_seconds", time())
                set_training_phase(TrainingPhase.TRAINING)
        self._publish(metrics)

    def log_summary(self, metrics: dict[str, Any]):
        self._publish(metrics)

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        pass

    def finish(self):
        if self._nccl_ras_probe is not None:
            try:
                self._nccl_ras_probe.shutdown()
            except Exception:
                logger.warning("could not stop NCCL RAS telemetry", exc_info=True)
        set_training_phase(TrainingPhase.FINISHED)
        # The run is over, so there is nothing left to keep enrolled. FINISHED is
        # already published above, and republishing it for the process's remaining
        # lifetime tells the reader nothing new.
        _HEARTBEAT.stop()

    def capture_stall_diagnostics(self) -> None:
        """Stop periodic RAS polling, publish a fresh stall sample, and flush telemetry."""
        if self._nccl_ras_probe is not None:
            self._nccl_ras_probe.capture_stall(_NCCL_STALL_CAPTURE_SECONDS)
        telemetry.shutdown(_STALL_TELEMETRY_FLUSH_SECONDS)


@TrackerConfig.register_subclass("telemetry")
@dataclasses.dataclass
class TelemetryConfig(TrackerConfig):
    """Configure direct training telemetry."""

    def init(self, run_id: Optional[str]) -> Tracker:
        process_index = jax.process_index()
        runtime_telemetry.configure(
            "levanter",
            root_run_uid=run_id,
            process_index=process_index,
        )
        return TelemetryTracker()
