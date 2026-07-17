# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tracker that mirrors training metrics onto the process's telltale page.

Trainers already log every step through the tracker interface, so mirroring there
puts live loss and throughput on the process's ``/metrics`` without touching the
training loop. Compose it with the real backend rather than replacing one::

    tracker: !composite
      - !wandb
      - !telltale

W&B remains the record of a run. This is the view of a process while it is still
running, reachable by curling the box when a run looks wedged and W&B only says
the last step was ten minutes ago.
"""

import dataclasses
import logging
import numbers
import typing
from typing import Any, Optional

from rigging import telltale

from levanter.tracker import Tracker, TrackerConfig

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "levanter"


class TelltaleTracker(Tracker):
    """Mirrors scalar metrics onto the telltale registry as gauges.

    Gauges, not counters: a tracker payload is a reading at a step (loss,
    throughput, learning rate), not a monotonic total.

    Only scalars are mirrored. Histograms, arrays and strings are dropped rather
    than flattened — a metrics page carrying a smeared array is worse than one
    that omits it, and the real tracker still receives everything.
    """

    name: str = "telltale"

    def __init__(self) -> None:
        self._step = telltale.gauge(f"{_METRIC_PREFIX}_step", "Most recent training step logged")

    def _mirror(self, metrics: typing.Mapping[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                continue
            try:
                name = telltale.metric_name(key, prefix=_METRIC_PREFIX)
                telltale.gauge(name, f"levanter metric {key}").set(float(value))
            except ValueError:
                # Two keys sanitizing to one name, or a name already taken by a
                # different metric type. Skip the sample; never fail a train step.
                logger.warning("could not mirror metric %r to telltale", key, exc_info=True)

    def log_hyperparameters(self, hparams: dict[str, Any]):
        pass

    def log(self, metrics: typing.Mapping[str, Any], *, step: Optional[int], commit: Optional[bool] = None):
        if step is not None:
            self._step.set(step)
        self._mirror(metrics)

    def log_summary(self, metrics: dict[str, Any]):
        self._mirror(metrics)

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        pass

    def finish(self):
        pass


@TrackerConfig.register_subclass("telltale")
@dataclasses.dataclass
class TelltaleConfig(TrackerConfig):
    def init(self, run_id: Optional[str]) -> Tracker:
        return TelltaleTracker()
