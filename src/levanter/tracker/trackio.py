# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass
from typing import Any, Optional

import jax
import numpy as np
from draccus import field

from levanter.tracker import Tracker
from levanter.tracker.histogram import Histogram
from levanter.tracker.tracker import TrackerConfig, NoopTracker


logger = logging.getLogger(__name__)


TrackioRun = Any


class TrackioTracker(Tracker):
    """Tracker backed by [trackio](https://github.com/gradio-app/trackio)."""

    name: str = "trackio"
    run: TrackioRun

    def __init__(self, run: Optional[TrackioRun] = None):
        import trackio

        if run is None:
            logger.warning("Trackio run is not initialized. Initializing a new run.")
            self.run = trackio.init(project="levanter")
        else:
            self.run = run

    def log_hyperparameters(self, hparams: dict[str, Any]):
        self.run.config.update(_convert_value_to_loggable_rec(hparams))

    def log(self, metrics: typing.Mapping[str, Any], *, step, commit=None):
        del commit
        to_log = {}
        for k, v in metrics.items():
            if isinstance(v, Histogram):
                counts, limits = v.to_numpy_histogram()
                to_log[f"{k}/histogram"] = {
                    "counts": counts.tolist(),
                    "limits": limits.tolist(),
                    "min": v.min,
                    "max": v.max,
                    "mean": v.mean,
                    "variance": v.variance,
                }
            else:
                to_log[k] = _convert_value_to_loggable_rec(v)

        import trackio

        trackio.log(to_log, step=step)

    def log_summary(self, metrics: typing.Mapping[str, Any]):
        import trackio

        to_log = {f"summary/{k}": _convert_value_to_loggable_rec(v) for k, v in metrics.items()}
        trackio.log(to_log)

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        del name, type
        logger.warning("Trackio does not currently support artifacts. Skipping upload for %s", artifact_path)

    def finish(self):
        import trackio

        logger.info("Finishing trackio run...")
        trackio.finish()


def _convert_value_to_loggable_rec(value: Any):
    if isinstance(value, (list, tuple)):
        return [_convert_value_to_loggable_rec(v) for v in value]
    elif isinstance(value, typing.Mapping):
        return {k: _convert_value_to_loggable_rec(v) for k, v in value.items()}
    elif isinstance(value, jax.Array):
        if value.ndim == 0:
            return value.item()
        else:
            return np.array(value)
    elif isinstance(value, Histogram):
        counts, limits = value.to_numpy_histogram()
        return {"counts": counts.tolist(), "limits": limits.tolist()}
    else:
        return value


@TrackerConfig.register_subclass("trackio")
@dataclass
class TrackioConfig(TrackerConfig):
    """Configuration for Trackio."""

    project: str = "levanter"
    name: Optional[str] = None
    space_id: Optional[str] = None
    dataset_id: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)
    mode: Optional[str] = None
    """Controls how Trackio logs.

    "online" logs normally, "offline" is synonymous with online since Trackio is
    local-first, and "disabled" disables logging. If ``None`` the default is
    "online" for the primary process and "disabled" for others.
    """

    resume: str = "auto"

    def init(self, run_id: Optional[str]) -> Tracker:
        import trackio

        # Only the primary process should log by default.
        if jax.process_index() == 0:
            mode = self.mode or "online"
        else:
            mode = "disabled"

        if mode == "disabled":
            return NoopTracker()

        cfg = dict(self.config)
        if run_id is not None:
            cfg.setdefault("run_id", run_id)

        resume = self.resume
        if resume == "auto":
            resume = "allow" if run_id is not None else "never"

        r = trackio.init(
            project=self.project,
            name=self.name,
            space_id=self.space_id,
            dataset_id=self.dataset_id,
            config=cfg or None,
            resume=resume,
        )
        return TrackioTracker(r)
