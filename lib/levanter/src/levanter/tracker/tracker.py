# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
import logging
import os
import typing
from typing import Any, List, Optional

import draccus
import jax


logger = logging.getLogger(__name__)


def artifact_size_bytes(artifact_path: str | os.PathLike[str]) -> int:
    """Return the number of regular-file bytes a tracker would upload for a path."""
    path = os.fspath(artifact_path)
    if os.path.isfile(path):
        return os.path.getsize(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    total = 0
    for directory, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(directory, filename)
            if not os.path.islink(file_path):
                total += os.path.getsize(file_path)
    return total


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

    def _prepare_log(self, metrics: typing.Mapping[str, Any]) -> Any:
        """Convert a ``log`` payload to host data, on the caller thread.

        Both ``log`` and :class:`~levanter.tracker.background.BackgroundTracker`
        call this. The default pulls ``jax.Array`` leaves to host with
        ``jax.device_get``; subclasses override to also fold in backend-specific
        conversion. Must be idempotent — the background worker re-invokes ``log``
        on the result.
        """
        return jax.device_get(metrics)

    def _prepare_summary(self, metrics: typing.Mapping[str, Any]) -> Any:
        """Producer-thread counterpart to :meth:`_prepare_log` for ``log_summary``."""
        return jax.device_get(metrics)

    def _prepare_hyperparameters(self, hparams: typing.Mapping[str, Any]) -> Any:
        """Producer-thread counterpart to :meth:`_prepare_log` for ``log_hyperparameters``."""
        return jax.device_get(hparams)

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

    @property
    def max_artifact_bytes(self) -> Optional[int]:
        """Largest artifact this tracker uploads, in bytes, or None for no limit."""
        return None

    def skip_artifact_over_limit(self, artifact_path, *, name: Optional[str] = None) -> bool:
        """Warn and return True when ``artifact_path`` exceeds ``max_artifact_bytes``.

        The caller must not upload the artifact when this returns True. A tracker
        without a limit returns False without measuring the artifact.
        """
        limit = self.max_artifact_bytes
        if limit is None:
            return False
        size = artifact_size_bytes(artifact_path)
        if size <= limit:
            return False
        display_name = name if name is not None else os.path.basename(os.fspath(artifact_path)) or "artifact"
        logger.warning(
            "Skipping %s artifact %r: %s bytes exceeds the %s-byte limit.",
            getattr(self, "name", type(self).__name__),
            display_name,
            f"{size:,}",
            f"{limit:,}",
        )
        return True

    def log_html(self, key: str, html_path, *, step: Optional[int], commit: Optional[bool] = None):
        pass

    def capture_stall_diagnostics(self) -> None:
        """Capture tracker-specific diagnostics before watchdog termination."""
        pass

    @abc.abstractmethod
    def finish(self):
        """
        Finish the tracker. This is called when the tracker is no longer needed. This can, e.g.,
        force a commit of all metrics.
        """
        pass

    def __enter__(self):
        # Deferred: tracker_fns eagerly imports the wandb/trackio backends, which import
        # `background`, which imports this module -> a genuine cycle. __enter__ only needs
        # the global-tracker accessor at call time, so import it lazily here.
        import levanter.tracker.tracker_fns as tracker_fns  # noqa: PLC0415

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

    def _for_each(self, op: str, *args, **kwargs) -> None:
        for tracker in self.loggers:
            try:
                getattr(tracker, op)(*args, **kwargs)
            except Exception:
                logger.exception(
                    "Tracker '%s' raised during %s; continuing with remaining trackers.",
                    getattr(tracker, "name", type(tracker).__name__),
                    op,
                )

    def log_hyperparameters(self, hparams: dict[str, Any]):
        self._for_each("log_hyperparameters", hparams)

    def log(self, metrics: typing.Mapping[str, Any], *, step, commit=None):
        self._for_each("log", metrics, step=step, commit=commit)

    def log_summary(self, metrics: dict[str, Any]):
        self._for_each("log_summary", metrics)

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        self._for_each("log_artifact", artifact_path, name=name, type=type)

    def log_html(self, key: str, html_path, *, step: Optional[int], commit: Optional[bool] = None):
        self._for_each("log_html", key, html_path, step=step, commit=commit)

    def capture_stall_diagnostics(self) -> None:
        self._for_each("capture_stall_diagnostics")

    def finish(self):
        # finish() exceptions are logged and swallowed too; a tracker failing to
        # flush at the very end of a run shouldn't crash the trainer's shutdown.
        self._for_each("finish")


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
