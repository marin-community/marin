# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tracker that mirrors training metrics onto the process's telltale page.

Compose it with the backend that keeps the record, rather than replacing one::

    tracker: !composite
      - !wandb
      - !telltale
"""

import dataclasses
import logging
import threading
import typing
from typing import Any, Optional

import numpy as np
from prometheus_client.core import HistogramMetricFamily
from rigging import telltale

from levanter.tracker import Tracker, TrackerConfig
from levanter.tracker.histogram import SummaryStats

logger = logging.getLogger(__name__)

_PREFIX = "levanter"


def _as_scalar(value: Any) -> float | None:
    """Coerce a metric value to a float, or None if it is not a real scalar.

    Training metrics arrive as 0-d arrays, not Python numbers — a `jax.Array`
    stays 0-d through `jax.device_get`, and neither it nor a 0-d `ndarray` is a
    `numbers.Real`. Testing for that ABC drops every real metric.
    """
    if isinstance(value, bool | str | bytes):
        return None
    try:
        array = np.asarray(value)
    except (ValueError, TypeError):
        # What numpy raises for something with no array form, e.g. a ragged
        # sequence. Debug, not warning: trackers are handed plenty of non-metrics.
        logger.debug("value of type %s has no array form", type(value).__name__, exc_info=True)
        return None
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
        return None
    return float(array)


class _Buckets(typing.NamedTuple):
    key: str
    bounds: list[str]
    cumulative: list[float]
    total: float


class _HistogramCollector:
    """Serves the most recent bucket counts levanter reduced for each metric.

    ``prometheus_client.Histogram`` only accepts ``observe`` and would want every
    individual value; levanter has already reduced them on device, so the buckets
    are handed over whole and emitted at scrape time instead.

    One per process, like every other metric telltale holds: two collectors
    holding the same key would each emit its family, and a repeated family is
    malformed exposition that Prometheus rejects on scrape.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, _Buckets] = {}

    def observe(self, key: str, stats: SummaryStats) -> None:
        assert stats.histogram is not None
        counts, limits = stats.histogram.to_numpy_histogram()
        # Prometheus buckets are cumulative and labelled by upper bound: bucket i
        # holds everything at or below limits[i + 1]. limits has one more entry
        # than counts, and the last bound is the observed max, so +Inf repeats the
        # final total.
        cumulative = np.cumsum(counts)
        bounds = [str(limits[i + 1]) for i in range(len(counts))] + ["+Inf"]
        totals = [float(c) for c in cumulative] + [float(cumulative[-1]) if len(cumulative) else 0.0]
        with self._lock:
            self._latest[telltale.metric_name(key, prefix=_PREFIX)] = _Buckets(key, bounds, totals, float(stats.sum))

    def collect(self) -> typing.Iterator[HistogramMetricFamily]:
        with self._lock:
            latest = dict(self._latest)
        for name, buckets in latest.items():
            # prometheus_client drops _sum and _count when the lowest bound is
            # negative, since averaging a sum of signed observations is
            # meaningless. Distributions that straddle zero are normal here
            # (gradients, activations), so let it: the +Inf bucket still carries
            # the count, and _publish_summary exports the moments as gauges.
            yield HistogramMetricFamily(
                name,
                f"levanter metric {buckets.key}",
                buckets=list(zip(buckets.bounds, buckets.cumulative, strict=True)),
                sum_value=buckets.total,
            )


_HISTOGRAMS = _HistogramCollector()


class TelltaleTracker(Tracker):
    """Publishes training metrics on this process's telltale page.

    Scalars become gauges: a tracker payload is a reading at a step, not a
    monotonic total. A ``SummaryStats`` becomes its reduced scalars plus, when it
    carries buckets, a real Prometheus histogram. Anything else is skipped — the
    backend tracker still receives everything.
    """

    name: str = "telltale"

    def __init__(self) -> None:
        self._step = telltale.gauge(f"{_PREFIX}_step", "Most recent training step logged")
        telltale.register_collector(_HISTOGRAMS)

    def _set(self, key: str, value: float) -> None:
        telltale.publish_gauge(key, value, f"levanter metric {key}", prefix=_PREFIX)

    def _publish(self, metrics: typing.Mapping[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, SummaryStats):
                self._publish_summary(key, value)
                continue
            scalar = _as_scalar(value)
            if scalar is not None:
                self._set(key, scalar)

    def _publish_summary(self, key: str, stats: SummaryStats) -> None:
        for field in ("mean", "min", "max", "variance", "rms"):
            scalar = _as_scalar(getattr(stats, field))
            if scalar is not None:
                self._set(f"{key}/{field}", scalar)
        if stats.histogram is not None:
            _HISTOGRAMS.observe(key, stats)

    def log_hyperparameters(self, hparams: dict[str, Any]):
        pass

    def log(self, metrics: typing.Mapping[str, Any], *, step: Optional[int], commit: Optional[bool] = None):
        if step is not None:
            self._step.set(step)
            loss = _as_scalar(metrics.get("train/loss"))
            telltale.set_status(f"step {step}" + (f", train/loss {loss:.4f}" if loss is not None else ""))
        self._publish(metrics)

    def log_summary(self, metrics: dict[str, Any]):
        self._publish(metrics)

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        pass

    def finish(self):
        pass


@TrackerConfig.register_subclass("telltale")
@dataclasses.dataclass
class TelltaleConfig(TrackerConfig):
    def init(self, run_id: Optional[str]) -> Tracker:
        # The metrics carry no identity; tag every persisted row (see the iris
        # telltale->finelog forwarder) with the run so dashboards can pick out one
        # training run. `run_id` is None for an unnamed run — then only `source`
        # is set.
        labels = {"source": "levanter"}
        if run_id is not None:
            labels["run"] = run_id
        telltale.set_global_labels(**labels)
        return TelltaleTracker()
