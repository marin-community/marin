# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import logging
import sys
from dataclasses import dataclass

import ray
import tblib
from ray.runtime_env import RuntimeEnv


@dataclass
class ExceptionInfo:
    ex: BaseException | None
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


def ser_exc_info(exception=None) -> ExceptionInfo:
    if exception is None:
        _, exc_value, exc_traceback = sys.exc_info()
        tb = tblib.Traceback(exc_traceback)
        return ExceptionInfo(exc_value, tb)
    else:
        tb = exception.__traceback__
        tb = tblib.Traceback(tb)
        return ExceptionInfo(exception, tb)


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
    accelerator_type: str | None = None

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
