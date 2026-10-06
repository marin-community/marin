# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Optional

import jax
import numpy as np
from rigging.filesystem.storage_path import StoragePath

from levanter.tracker import Tracker
from levanter.tracker.tracker import FatalTrackerError, TrackerConfig
from levanter.utils.jax_utils import jnp_to_python


def _to_jsonable(value: Any):
    """Recursively convert ``value`` to something JSON serializable."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, jax.Array):
        return jnp_to_python(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, str | int | float | bool | type(None)):
        return value

    # coerce to string as a last resort
    return str(value)


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

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        *,
        metric_destination: str | None = None,
        run_id: str | None = None,
    ):
        if metric_destination is not None and run_id is None:
            raise ValueError("Durable JSON metrics require a run ID")
        self.metric_destination = metric_destination
        self.run_id = run_id
        self._destination_prepared = False
        self.logger = logger or logging.getLogger("levanter.json_logger")
        self._last_metrics: dict[str, Any] = {}
        self._summary_metrics: dict[str, Any] = {}

    def log_hyperparameters(self, hparams: dict[str, Any]):
        record = _to_jsonable(
            {
                "tracker": self.name,
                "event": "hparams",
                "hparams": hparams,
            }
        )
        self.logger.info(json.dumps(record))

    def log(self, metrics: Mapping[str, Any], *, step: Optional[int], commit: Optional[bool] = None):
        del commit
        record = _to_jsonable(
            {
                "tracker": self.name,
                "event": "log",
                "step": step,
                "metrics": metrics,
            }
        )
        if self.metric_destination is not None and jax.process_index() == 0:
            record["run_id"] = self.run_id
            payload = json.dumps(record, sort_keys=True).encode()
            destination = StoragePath(self.metric_destination)
            try:
                if not self._destination_prepared:
                    destination.mkdirs()
                    self._destination_prepared = True
                (destination / f"step-{step}-{hashlib.sha256(payload).hexdigest()}.json").write_bytes(payload)
            except Exception as error:
                raise FatalTrackerError(f"Could not persist metric event at {destination}") from error
        self.logger.info(json.dumps(record))
        if step is not None:
            self._last_metrics.update(_flatten(metrics))

    def log_summary(self, metrics: Mapping[str, Any]):
        record = _to_jsonable(
            {
                "tracker": self.name,
                "event": "summary",
                "metrics": metrics,
            }
        )
        self.logger.info(json.dumps(record))
        self._summary_metrics.update(_flatten(metrics))

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        record = _to_jsonable(
            {
                "tracker": self.name,
                "event": "artifact",
                "path": artifact_path,
                "name": name,
                "artifact_type": type,
            }
        )
        self.logger.info(json.dumps(record))

    def finish(self):
        summary = {**self._summary_metrics, **self._last_metrics}
        record = _to_jsonable(
            {
                "tracker": self.name,
                "event": "finish",
                "summary": summary,
            }
        )
        self.logger.info(json.dumps(record))


@TrackerConfig.register_subclass("json_logger")
@dataclass
class JsonLoggerConfig(TrackerConfig):
    """Configuration for :class:`JsonLoggerTracker`."""

    logger_name: str = "levanter.json_logger"
    level: int = logging.INFO
    metric_destination: str | None = None

    def init(self, run_id: Optional[str]) -> JsonLoggerTracker:
        log = logging.getLogger(self.logger_name)
        log.setLevel(self.level)
        return JsonLoggerTracker(log, metric_destination=self.metric_destination, run_id=run_id)
