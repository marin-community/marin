# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tracker that exports Levanter snapshots through direct process telemetry."""

import dataclasses
import logging
import re
from collections.abc import Mapping
from enum import IntEnum
from time import time
from typing import Any, Optional

import numpy as np
from iris.runtime import telemetry as runtime_telemetry
from rigging import telemetry

from levanter.tracker import Tracker, TrackerConfig
from levanter.tracker.histogram import SummaryStats

logger = logging.getLogger(__name__)

_CURRENT = telemetry.snapshot_attributes("gauge", telemetry.CURRENT_SNAPSHOT)
_CURRENT_HISTOGRAM = telemetry.snapshot_attributes("histogram", telemetry.CURRENT_SNAPSHOT)


class TrainingPhase(IntEnum):
    """Numeric training phases persisted to Finelog."""

    INITIALIZING = 0
    TRAINING = 1
    FINISHED = 2


def _metric_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _set(name: str, value: float, *, attributes: dict[str, str] = _CURRENT) -> None:
    telemetry.gauge(_metric_name(name)).set(value, attributes=attributes)


def set_training_phase(phase: TrainingPhase) -> None:
    """Publish the current training phase for stalled-training detection."""
    _set("phase", float(phase))


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
    """Exports scalar and pre-aggregated training snapshots as gauges."""

    name: str = "telemetry"

    def __init__(self) -> None:
        _set("progress_time_seconds", 0)
        set_training_phase(TrainingPhase.INITIALIZING)

    def _publish(self, metrics: Mapping[str, object]) -> None:
        for key, value in metrics.items():
            if isinstance(value, SummaryStats):
                self._publish_summary(key, value)
                continue
            scalar = _as_scalar(value)
            if scalar is not None:
                _set(key, scalar)

    def _publish_summary(self, key: str, stats: SummaryStats) -> None:
        for field in ("mean", "min", "max", "variance", "rms"):
            scalar = _as_scalar(getattr(stats, field))
            if scalar is not None:
                _set(f"{key}_{field}", scalar)
        if stats.histogram is None:
            return
        counts, limits = stats.histogram.to_numpy_histogram()
        cumulative = np.cumsum(counts)
        for index, count in enumerate(cumulative):
            _set(
                f"{key}_bucket",
                float(count),
                attributes={**_CURRENT_HISTOGRAM, "le": str(limits[index + 1])},
            )
        total = float(cumulative[-1]) if len(cumulative) else 0.0
        _set(f"{key}_bucket", total, attributes={**_CURRENT_HISTOGRAM, "le": "+Inf"})
        _set(f"{key}_count", total, attributes=_CURRENT_HISTOGRAM)
        _set(f"{key}_sum", float(stats.sum), attributes=_CURRENT_HISTOGRAM)

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
        set_training_phase(TrainingPhase.FINISHED)


@TrackerConfig.register_subclass("telemetry")
@dataclasses.dataclass
class TelemetryConfig(TrackerConfig):
    def init(self, run_id: Optional[str]) -> Tracker:
        runtime_telemetry.configure(
            "levanter",
            root_run_uid=run_id,
        )
        return TelemetryTracker()
