import dataclasses
import logging as pylogging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union

import jax
from dataclasses_json import dataclass_json

import levanter.tracker


# TODO: should we just make the ledger have all this?
@dataclass_json
@dataclass
class InProgressCacheMetrics:
    rows_finished: int = 0
    shards_finished: int = 0
    field_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    is_finished: bool = False


class MetricsMonitor(Protocol):
    def __call__(self, metrics: InProgressCacheMetrics):
        ...


class LoggingMetricsMonitor(MetricsMonitor):
    last_metrics: Optional[InProgressCacheMetrics]
    last_time: Optional[float]

    def __init__(self, prefix: str = "preproc", commit=False):
        """
        :param prefix:
        :param commit: Forwarded to wandb.log. Use False (default) if it's part of a simultaneous training run,
        and True if you're running standalone.
        """
        self.prefix = prefix
        self.commit = commit
        self.last_metrics = None
        self.last_time = None

    def __call__(self, metrics: InProgressCacheMetrics):
        to_log: Dict[str, Any] = {}

        to_log[f"{self.prefix}/shards"] = metrics.shards_finished
        to_log[f"{self.prefix}/rows"] = metrics.rows_finished

        for field, count in metrics.field_counts.items():
            to_log[f"{self.prefix}/{field}"] = count

        if metrics.is_finished:
            to_log[f"{self.prefix}/finished"] = 1

        self.last_metrics = metrics
        self.last_time = time.time()

        levanter.tracker.log(to_log, step=None, commit=self.commit)


class LoggerMetricsMonitor(MetricsMonitor):
    # TODO: I'd like to get the trainer pbar migrated to rich and just use rich everywhere, but until then,
    # we have separate logging
    def __init__(
        self,
        logger: Optional[Union[pylogging.Logger, str]] = None,
        level=pylogging.INFO,
        log_interval: float | int = 30.0,
    ):
        if isinstance(logger, str):
            logger = pylogging.getLogger(logger)
        self.logger = logger or pylogging.getLogger(__name__)
        self.level = level
        self.log_interval = log_interval
        self._last_log_time = time.time()

    def __call__(self, metrics: InProgressCacheMetrics):
        if jax.process_index() == 0:
            if time.time() - self._last_log_time > self.log_interval:
                self._last_log_time = time.time()

                self.logger.log(
                    self.level,
                    f" done: Shards: {metrics.shards_finished} | Docs: {metrics.rows_finished}",
                )

        if metrics.is_finished:
            self.logger.info("Cache creation finished")
