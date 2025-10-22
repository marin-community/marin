# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import jax

from levanter.tracker import Tracker
from levanter.tracker.histogram import Histogram
from levanter.tracker.tracker import TrackerConfig
from levanter.utils.jax_utils import jnp_to_python


logger = logging.getLogger(__name__)


def _to_jsonable(value: Any):
    """Recursively convert ``value`` to something JSON serializable."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Histogram):
        return {
            "min": jnp_to_python(value.min),
            "max": jnp_to_python(value.max),
            "num": jnp_to_python(value.num),
            "sum": jnp_to_python(value.sum),
            "sum_squares": jnp_to_python(value.sum_squares),
            "bucket_limits": jnp_to_python(value.bucket_limits),
            "bucket_counts": jnp_to_python(value.bucket_counts),
        }
    if isinstance(value, jax.Array):
        return jnp_to_python(value)
    return value


def _flatten(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        name = f"{prefix}/{k}" if prefix else k
        if isinstance(v, Mapping):
            out.update(_flatten(v, name))
        else:
            out[name] = v
    return out


class JsonLoggerTracker(Tracker):
    """Tracker that logs metrics to a Python logger as JSON lines."""

    name: str = "json_logger"

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("levanter.json_logger")
        self._last_metrics: dict[str, Any] = {}
        self._summary_metrics: dict[str, Any] = {}

    def log_hyperparameters(self, hparams: dict[str, Any]):
        record = {
            "tracker": self.name,
            "event": "hparams",
            "hparams": _to_jsonable(hparams),
        }
        self.logger.info(json.dumps(record))

    def log(self, metrics: Mapping[str, Any], *, step: Optional[int], commit: Optional[bool] = None):
        del commit
        record = {
            "tracker": self.name,
            "event": "log",
            "step": step,
            "metrics": _to_jsonable(metrics),
        }
        self.logger.info(json.dumps(record))
        if step is not None:
            self._last_metrics.update(_flatten(metrics))

    def log_summary(self, metrics: Mapping[str, Any]):
        record = {
            "tracker": self.name,
            "event": "summary",
            "metrics": _to_jsonable(metrics),
        }
        self.logger.info(json.dumps(record))
        self._summary_metrics.update(_flatten(metrics))

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        record = {
            "tracker": self.name,
            "event": "artifact",
            "path": artifact_path,
            "name": name,
            "artifact_type": type,
        }
        self.logger.info(json.dumps(record))

    def finish(self):
        summary = {**self._summary_metrics, **self._last_metrics}
        record = {
            "tracker": self.name,
            "event": "finish",
            "summary": _to_jsonable(summary),
        }
        self.logger.info(json.dumps(record))


@TrackerConfig.register_subclass("json_logger")
@dataclass
class JsonLoggerConfig(TrackerConfig):
    """Configuration for :class:`JsonLoggerTracker`."""

    logger_name: str = "levanter.json_logger"
    level: int = logging.INFO

    def init(self, run_id: Optional[str]) -> JsonLoggerTracker:
        del run_id
        log = logging.getLogger(self.logger_name)
        log.setLevel(self.level)
        return JsonLoggerTracker(log)
