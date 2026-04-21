# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import logging
import os
from typing import Any, Mapping, Optional

import jax

from levanter.tracker.json_logger import _flatten, _to_jsonable
from levanter.tracker.tracker import NoopTracker, Tracker, TrackerConfig

logger = logging.getLogger(__name__)


class JsonFileTracker(Tracker):
    """Tracker that accumulates metrics and saves them to a JSON file on finish()."""

    name: str = "json_file"

    def __init__(self, output_path: str):
        self.output_path = output_path
        self._last_metrics: dict[str, Any] = {}
        self._summary_metrics: dict[str, Any] = {}

    def log_hyperparameters(self, hparams: dict[str, Any]):
        pass

    def log(self, metrics: Mapping[str, Any], *, step: Optional[int], commit: Optional[bool] = None):
        if step is not None:
            self._last_metrics.update(_flatten(metrics))

    def log_summary(self, metrics: Mapping[str, Any]):
        self._summary_metrics.update(_flatten(metrics))

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        pass

    def finish(self):
        import fsspec

        summary = {**self._summary_metrics, **self._last_metrics}
        output_file = os.path.join(self.output_path, "eval_results.json")
        with fsspec.open(output_file, "wt") as f:
            json.dump(_to_jsonable(summary), f, indent=2)
        logger.info(f"Saved eval results to {output_file}")


@TrackerConfig.register_subclass("json_file")
@dataclasses.dataclass
class JsonFileTrackerConfig(TrackerConfig):
    output_path: str = ""

    def init(self, run_id: Optional[str]) -> Tracker:
        if jax.process_index() != 0:
            return NoopTracker()
        return JsonFileTracker(self.output_path)
