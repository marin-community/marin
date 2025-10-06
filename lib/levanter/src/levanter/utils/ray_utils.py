# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import logging
import logging as pylogging
import sys
from dataclasses import dataclass
from typing import Optional

import ray
import tblib
from ray.runtime_env import RuntimeEnv


@dataclass
class ExceptionInfo:
    ex: Optional[BaseException]
    tb: tblib.Traceback

    def restore(self):
        if self.ex is not None:
            exc_value = self.ex.with_traceback(self.tb.as_traceback())
            return (self.ex.__class__, exc_value, self.tb.as_traceback())
        else:
            return (Exception, Exception("Process failed with no exception"), self.tb.as_traceback())

    def reraise(self):
        if self.ex is not None:
            raise self.ex.with_traceback(self.tb.as_traceback())
        else:
            raise Exception("Process failed with no exception").with_traceback(self.tb.as_traceback())


@dataclass
class RayResources:
    """
    A dataclass that represents the resources for a ray task or actor. It's main use is to be
    fed to ray.remote() to specify the resources for a task.
    """

    num_cpus: int = 1
    num_gpus: int = 0
    resources: dict = dataclasses.field(default_factory=dict)
    runtime_env: RuntimeEnv = dataclasses.field(default_factory=RuntimeEnv)
    accelerator_type: Optional[str] = None

    def to_kwargs(self):
        """
        Returns a dictionary of kwargs that can be passed to ray.remote() to specify the resources for a task.
        """
        out = dict(
            num_cpus=self.num_cpus, num_gpus=self.num_gpus, resources=self.resources, runtime_env=self.runtime_env
        )

        if self.accelerator_type is not None:
            out["accelerator_type"] = self.accelerator_type

        return out

    def to_resource_dict(self):
        """
        Returns a dictionary of resources that describe the resources.

        Note that Ray for some reason doesn't like passing resources={"GPU": 1} to a remote so you can't use this
        for that purpose. Instead, use @ray.remote(**ray_resources.to_kwargs())
        """
        out = dict(CPU=self.num_cpus, GPU=self.num_gpus, **self.resources)
        # https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#accelerator-types
        if self.accelerator_type is not None:
            out[f"accelerator_type:{self.accelerator_type}"] = 0.001
        return out

    @staticmethod
    def from_resource_dict(resources: dict):
        return RayResources(num_cpus=resources.get("CPU", 0), num_gpus=resources.get("GPU", 0), resources=resources)


@dataclass
class RefBox:
    """Ray doesn't dereference ObjectRefs if they're nested in another object. So we use this to take advantage of that.
    https://docs.ray.io/en/latest/ray-core/objects.html#passing-object-arguments"""

    ref: ray.ObjectRef

    def get(self):
        return ray.get(self.ref)


class DoneSentinel:
    pass


DONE = DoneSentinel()


def ser_exc_info(exception=None) -> ExceptionInfo:
    if exception is None:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = tblib.Traceback(exc_traceback)
        return ExceptionInfo(exc_value, tb)
    else:
        tb = exception.__traceback__
        tb = tblib.Traceback(tb)
        return ExceptionInfo(exception, tb)


def current_actor_handle() -> ray.actor.ActorHandle:
    return ray.runtime_context.get_runtime_context().current_actor


class SnitchRecipient:
    logger: logging.Logger

    def _child_failed(self, child: ray.actor.ActorHandle | str | None, exception: ExceptionInfo):
        info = exception.restore()
        self.logger.error(f"Child {child} failed with exception {info[1]}", exc_info=info)
        exception.reraise()


@contextlib.contextmanager
def log_failures_to(parent, suppress=False):
    # parent is actorref of SnitchRecipient
    try:
        yield
    except Exception as e:
        try:
            handle = current_actor_handle()
        except RuntimeError:
            handle = ray.runtime_context.get_runtime_context().get_task_id()

        parent._child_failed.remote(handle, ser_exc_info(e))
        if not suppress:
            raise e


DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"


@ray.remote
class StopwatchActor:
    def __init__(self):
        pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        self._logger = pylogging.getLogger("StopwatchActor")
        self._times_per = {}
        self._counts_per = {}
        self._total = 0

    def measure(self, name: str, time: float):
        self._times_per[name] = self._times_per.get(name, 0) + time
        self._counts_per[name] = self._counts_per.get(name, 0) + 1
        self._total += 1

        if self._total % 1000 == 0:
            for name, time in self._times_per.items():
                self._logger.info(f"{name}: {time / self._counts_per[name]}")

    def get(self, name: str):
        return self._times_per.get(name, 0), self._counts_per.get(name, 0)

    def average(self, name: str):
        return self._times_per.get(name, 0) / self._counts_per.get(name, 1)
