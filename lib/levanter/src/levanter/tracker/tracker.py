# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
import logging
import typing
from typing import Any, List, Optional

import draccus


logger = logging.getLogger(__name__)


class Tracker(abc.ABC):
    """
    A tracker is responsible for logging metrics, hyperparameters, and artifacts.
    Meant to be used with the [levanter.tracker.current_tracker][] context manager, but can also be used directly.

    The name is borrowed from HF Accelerate.

    Examples:
        >>> from levanter.tracker import current_tracker, log
        >>> from levanter.tracker.wandb import WandbTracker
        >>> with current_tracker(WandbTracker()):
        ...     log({"foo": 1}, step=0)
    """

    name: str

    @abc.abstractmethod
    def log_hyperparameters(self, hparams: dict[str, Any]):
        pass

    @abc.abstractmethod
    def log(self, metrics: typing.Mapping[str, typing.Any], *, step: Optional[int], commit: Optional[bool] = None):
        """
        Log metrics to the tracker. Step is always required.

        Args:
            metrics: Metrics to log
            step: Step to log at
            commit: Whether to commit the metrics. If None, uses the default for the tracker.
        """
        pass

    @abc.abstractmethod
    def log_summary(self, metrics: dict[str, Any]):
        pass

    @abc.abstractmethod
    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        pass

    @abc.abstractmethod
    def finish(self):
        """
        Finish the tracker. This is called when the tracker is no longer needed. This can, e.g.,
        force a commit of all metrics.
        """
        pass

    def __enter__(self):
        import levanter.tracker.tracker_fns as tracker_fns

        if hasattr(self, "_tracker_cm"):
            raise RuntimeError("This tracker is already set as the global tracker")
        setattr(self, "_tracker_cm", tracker_fns.current_tracker(self))
        self._tracker_cm.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not hasattr(self, "_tracker_cm"):
            raise RuntimeError("This tracker is not set as the global tracker")
        self._tracker_cm.__exit__(exc_type, exc_val, exc_tb)
        delattr(self, "_tracker_cm")


class CompositeTracker(Tracker):
    """A tracker that fans calls out to a list of trackers.

    Exceptions from any single member tracker are caught and logged so that one
    failing backend (e.g. W&B losing connectivity) doesn't take down the others.
    Wrap members in :class:`~levanter.tracker.BackgroundTracker` if you also
    want isolation from latency.
    """

    def __init__(self, loggers: List[Tracker]):
        self.loggers = loggers

    def _for_each(self, op: str, fn) -> None:
        for tracker in self.loggers:
            try:
                fn(tracker)
            except Exception:
                logger.exception(
                    "Tracker '%s' raised during %s; continuing with remaining trackers.",
                    getattr(tracker, "name", type(tracker).__name__),
                    op,
                )

    def log_hyperparameters(self, hparams: dict[str, Any]):
        self._for_each("log_hyperparameters", lambda t: t.log_hyperparameters(hparams))

    def log(self, metrics: typing.Mapping[str, Any], *, step, commit=None):
        self._for_each("log", lambda t: t.log(metrics, step=step, commit=commit))

    def log_summary(self, metrics: dict[str, Any]):
        self._for_each("log_summary", lambda t: t.log_summary(metrics))

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        self._for_each("log_artifact", lambda t: t.log_artifact(artifact_path, name=name, type=type))

    def finish(self):
        # finish() exceptions are logged and swallowed too; a tracker failing to
        # flush at the very end of a run shouldn't crash the trainer's shutdown.
        self._for_each("finish", lambda t: t.finish())


class TrackerConfig(draccus.PluginRegistry, abc.ABC):
    discover_packages_path = "levanter.tracker"

    @abc.abstractmethod
    def init(self, run_id: Optional[str]) -> Tracker:
        raise NotImplementedError

    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "wandb"


class NoopTracker(Tracker):
    name: str = "noop"

    def log_hyperparameters(self, hparams: dict[str, Any]):
        pass

    def log(self, metrics: typing.Mapping[str, Any], *, step, commit: Optional[bool] = None):
        pass

    def log_summary(self, metrics: dict[str, Any]):
        pass

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        pass

    def finish(self):
        pass


@TrackerConfig.register_subclass("noop")
@dataclasses.dataclass
class NoopConfig(TrackerConfig):
    def init(self, run_id: Optional[str]) -> Tracker:
        return NoopTracker()


class DictTracker(Tracker):
    """
    A tracker that logs to a dictionary. We mostly use this to smuggle things outside of jit
    """

    def __init__(self):
        self.metrics: dict[str, Any] = {}

    def log_hyperparameters(self, hparams: dict[str, Any]):
        self.metrics["hparams"] = hparams

    def log(self, metrics: typing.Mapping[str, Any], *, step: Optional[int], commit: Optional[bool] = None):
        if step is not None:
            self.metrics[f"step_{step}"] = metrics
        else:
            self.metrics.update(metrics)

    def log_summary(self, metrics: dict[str, Any]):
        self.metrics["summary"] = metrics

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        self.metrics["artifact"] = {"path": artifact_path, "name": name, "type": type}

    def finish(self):
        pass
