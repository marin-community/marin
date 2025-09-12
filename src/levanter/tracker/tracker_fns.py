# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import logging
import os
import tempfile
import typing
import warnings
from contextlib import AbstractContextManager
from typing import Any, Literal, Optional

import draccus
import jax
from jaxtyping import Scalar

from levanter.tracker import CompositeTracker, Tracker
from levanter.tracker.helpers import hparams_to_dict
from levanter.tracker.histogram import Histogram
from levanter.tracker.tensorboard import TensorboardTracker
from levanter.tracker.tracker import DictTracker
from levanter.tracker.wandb import WandbTracker
from levanter.utils.jax_utils import is_inside_jit


logger = logging.getLogger(__name__)

_should_use_callback = True
_global_tracker: Optional["Tracker"] = None

LoggableValue: typing.TypeAlias = Scalar | jax.Array | str | dict | Histogram


def log(metrics: typing.Mapping[str, LoggableValue | Any], *, step: Optional[int], commit: Optional[bool] = None):
    """
    Log metrics to the global tracker.

    Args:
        metrics: Metrics to log. We use LoggableValues just to give you a sense of what you can log. Backends may
            support additional types.
        step: Step to log at. If None, uses the default for the tracker.
        commit: Whether to commit the metrics. If None, uses the default for the tracker.
    """
    global _global_tracker
    if _global_tracker is None:
        raise RuntimeError("No global tracker set")

    if is_inside_jit():
        # we're inside a jit, so we need to log from the host
        if commit:
            raise ValueError("Cannot commit from inside jit")
        jit_log(metrics, step=step)
    else:
        # TODO: do we need to coerce to np here?
        _global_tracker.log(metrics, step=step, commit=commit)


# deprecated in favor of log()
def log_metrics(
    metrics: typing.Mapping[str, LoggableValue | Any], *, step: Optional[int], commit: Optional[bool] = None
):
    """
    Deprecated. Use log instead.
    """
    warnings.warn("log_metrics is deprecated in favor of log", DeprecationWarning)
    log(metrics, step=step, commit=commit)


def _do_jit_log(metrics, *, step=None):
    try:
        if _global_tracker is None:
            warnings.warn("No global tracker set")
        else:
            _global_tracker.log(metrics, step=step, commit=False)
    except Exception:
        logger.exception("Error logging metrics")


def jit_log(metrics, *, step=None):
    """
    JAX doesn't allow tracers to escape the jit boundary, so we have to be clever about how we log metrics.
    In Levanter, we enable tracking inside jit with two mechanisms.

    * The first, most performant way, is to use the levanter.tracker.defer_tracker_for_jit context manager,
      which will cause logging to go to a dictionary (that is returned by capture_logging). You can
      then return this dictionary from the JIT function and log it outside of the JIT.
    * The second way is to just use an effect callback to log the metrics to the host.

    We strongly recommend using the first method, as it is much more performant.
    """
    if _global_tracker is None:
        warnings.warn("No global tracker set")
        return
    if not _should_use_callback:
        # we're not using the callback, so we assume we're inside a defer_tracker_for_jit context manager
        # and we just return the metrics dictionary
        _global_tracker.log(metrics, step=step, commit=False)
    else:
        jax.experimental.io_callback(_do_jit_log, None, metrics=metrics, step=step)


@contextlib.contextmanager
def defer_tracker_for_jit():
    """
    Context manager to defer capturing the tracker until the end of the context.
    This is useful when you want to log metrics that are computed inside JIT (i.e. usually),
    but you want to defer the actual logging until after the JIT has completed for performance reasons.

    The usual pattern is like this:

    ```python
        @equinox.filter_jit
        def my_jit_function():
            with defer_tracker_for_jit() as metrics:
                levanter.tracker.jit_log({ "foo": 1 })
                ...

                # do some JIT work
                result = ...

            return result, metrics

        result, metrics = my_jit_function()
        levanter.tracker.log(metrics, step=0)
    ```

    Returns:
        A context manager that defers capturing the tracker until the end of the context.
        The context manager yields the metrics dictionary that metrics are logged into.
        You can log this dictionary directly to the global tracker after the context manager exits.
    """
    global _global_tracker, _should_use_callback
    old_tracker = _global_tracker
    old_should_use_callback = _should_use_callback
    _should_use_callback = False
    local_tracker = DictTracker()
    _global_tracker = local_tracker

    try:
        yield local_tracker.metrics
    finally:
        _global_tracker = old_tracker
        _should_use_callback = old_should_use_callback


def log_summary(metrics: dict[str, Any]):
    """
     Log summary metrics to the global tracker.

    Args:
         metrics: Metrics to log
    """
    global _global_tracker
    if _global_tracker is None:
        warnings.warn("No global tracker set")
        return

    _global_tracker.log_summary(metrics)


def log_hyperparameters(hparams: dict[str, Any]):
    """
     Log hyperparameters to the global tracker.

    Args:
         hparams: Hyperparameters to log
    """
    global _global_tracker
    if _global_tracker is None:
        warnings.warn("No global tracker set")
        return

    _global_tracker.log_hyperparameters(hparams)


def log_configuration(hparams: Any, config_name: Optional[str] = None):
    """
     Logs a configuration object to the global tracker. If the configuration object is a dataclass,
        it is dumped to a yaml file and logged as an artifact.

    Args:
         hparams: Hyperparameters to log
    """
    global _global_tracker
    if _global_tracker is None:
        warnings.warn("No global tracker set")
        return

    hparams_dict = hparams_to_dict(hparams)
    _global_tracker.log_hyperparameters(hparams_dict)

    if dataclasses.is_dataclass(hparams):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            try:
                with open(config_path, "w") as f:
                    draccus.dump(hparams, f, encoding="utf-8")
                    name = config_name or "config.yaml"
                    _global_tracker.log_artifact(config_path, name=name, type="config")
            except Exception:  # noqa
                logger.warning("Failed to dump config to yaml. Skipping logging as artifact.", exc_info=True)


def set_global_tracker(tracker: Tracker):
    """
    Set the global tracker. Note that setting the global tracker is not thread-safe,
    and using a tracker from multiple threads is only supported if the tracker itself is thread-safe.

    In general, it's preferred to use the context manager returned by `current_tracker` instead of this function
    except for once at the beginning of the program.

    Args:
        tracker: The tracker to set as the global tracker
        force: Whether to force setting the global tracker even if it is already set

    Examples:
        >>> from levanter.tracker import set_global_tracker, log
        >>> from levanter.tracker.wandb import WandbTracker
        >>> set_global_tracker(WandbTracker())
        >>> log({"foo": 1}, step=0)
    """
    global _global_tracker
    if _global_tracker is not None:
        warnings.warn("Global tracker is already set. Overwriting it.")
    _global_tracker = tracker


@typing.overload
def current_tracker() -> "Tracker": ...


@typing.overload
def current_tracker(tracker: "Tracker") -> typing.ContextManager:
    """Returns a context manager for setting the global tracker"""
    ...


def current_tracker(
    tracker: Optional[Tracker] = None,
) -> Tracker | typing.ContextManager:
    """
    Get or set the global tracker. Note that setting the global tracker is not thread-safe,
    and using a tracker from multiple threads is only supported if the tracker itself is thread-safe.

    Args:
      tracker: If provided, returns a context manager that sets the global tracker to the provided tracker when used.

    Returns:
        If no tracker is provided, returns the current global tracker.
        If a tracker is provided, returns a context manager that sets the global tracker to the provided tracker when used.

    Examples:
        >>> from levanter.tracker import current_tracker, log
        >>> from levanter.tracker.wandb import WandbTracker
        >>> with current_tracker(WandbTracker()):
        ...     log({"foo": 1}, step=0)
        ...     current_tracker().log({"foo": 2}, step=1)
    """
    global _global_tracker
    if tracker is None:
        if _global_tracker is None:
            raise RuntimeError("No global tracker set")
        return _global_tracker
    else:
        return _GlobalLoggerContextManager(tracker)


@typing.overload
def get_tracker(name: Literal["wandb"]) -> WandbTracker: ...


@typing.overload
def get_tracker(name: Literal["tensorboard"]) -> TensorboardTracker: ...


@typing.overload
def get_tracker(name: str) -> Tracker: ...


def get_tracker(name: str) -> Tracker:
    """
    Lookup a tracker in the current global tracker with the provided name.

    Args:
        name: Name of the tracker to lookup

    Returns:
        The tracker with the provided name

    Examples:
        >>> from levanter.tracker import get_tracker, log
        >>> from levanter.tracker.wandb import WandbTracker
        >>> with current_tracker(WandbTracker()):
        ...     log({"foo": 1}, step=0)
        ...     get_tracker("wandb").log_metrics({"foo": 2}, step=1)
    """
    tracker = current_tracker()
    if isinstance(tracker, CompositeTracker):
        for t in tracker.loggers:
            if t.name == name:
                return t
    elif tracker.name == name:
        return tracker

    raise KeyError(f"Tracker with name {name} not found")


class _GlobalLoggerContextManager(AbstractContextManager):
    def __init__(self, tracker: "Tracker"):
        self.tracker = tracker

    def __enter__(self):
        global _global_tracker
        self.old_tracker = _global_tracker
        _global_tracker = self.tracker

        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_tracker
        _global_tracker = self.old_tracker
