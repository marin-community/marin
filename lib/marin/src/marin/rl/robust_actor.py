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

"""
Robust Ray actor wrapper with automatic recovery on failure.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError, RayActorError

logger = logging.getLogger(__name__)


class RobustActor:
    """Wraps a Ray actor handle with automatic retry logic.

    Intercepts method calls and recreates the actor handle on death.

    Example:
        curriculum = RobustActor.create(
            Curriculum,
            actor_name="curriculum",
            actor_args=(config,)
        )

        # Use like a normal Ray actor - automatically retries on actor death
        lesson_id = ray.get(curriculum.sample_lesson.remote(seed))
    """

    def __init__(self, factory_fn: Callable, actor_name: str):
        self._factory = factory_fn
        self._actor_name = actor_name
        self._handle = factory_fn()

    @staticmethod
    def create(
        actor_class: type, actor_name: str, actor_args: tuple = (), actor_kwargs: dict | None = None, **ray_options
    ):
        """Create a robust actor instance with auto-recovery.

        Args:
            actor_class: The actor class to wrap (not decorated with @ray.remote)
            actor_name: Name for the Ray actor (used with get_if_exists)
            actor_args: Positional arguments passed to actor __init__
            actor_kwargs: Keyword arguments passed to actor __init__
            **ray_options: Additional options passed to ray.remote().options()

        Returns:
            A wrapped actor handle with automatic retry on actor death
        """
        if actor_kwargs is None:
            actor_kwargs = {}

        def factory():
            """Factory function that creates/recreates the actor."""
            actor = (
                ray.remote(actor_class)
                .options(
                    name=actor_name,
                    get_if_exists=True,
                    max_restarts=-1,
                    max_task_retries=-1,
                    enable_task_events=False,
                    **ray_options,
                )
                .remote(*actor_args, **actor_kwargs)
            )
            return actor

        return RobustActor(factory, actor_name=actor_name)

    def __getattr__(self, name: str):
        """Forward attribute access to the underlying actor handle."""
        attr = getattr(self._handle, name)

        # If it's an actor method (has .remote), wrap it
        if hasattr(attr, "remote"):
            return _MethodWrapper(attr, name, self)

        return attr

    def _recreate_handle(self):
        """Recreate the actor handle after death."""
        logger.info(f"Recreating actor handle for '{self._actor_name}' after death")
        self._handle = self._factory()


class _MethodWrapper:
    """Wraps an actor method to retry .remote() calls on actor death."""

    def __init__(self, method: Any, method_name: str, handle: RobustActor):
        self._method = method
        self._method_name = method_name
        self._handle = handle

    def remote(self, *args, **kwargs):
        """Call .remote() with retry on actor death."""
        backoff = 1.0
        attempts = 0
        max_attempts = 5
        actor_name = self._handle._actor_name
        while True:
            try:
                return self._method.remote(*args, **kwargs)
            except (ActorDiedError, ActorUnavailableError, RayActorError):
                attempts += 1
                if attempts >= max_attempts:
                    logger.error(f"Actor '{actor_name}' failed after {max_attempts} attempts to {self._method_name}")
                    raise
                logger.warning(
                    f"Actor '{actor_name}' died calling {self._method_name}. Retrying after {backoff:.1f}s..."
                )
                self._handle._recreate_handle()
                self._method = getattr(self._handle._handle, self._method_name)
                backoff *= 2
                time.sleep(backoff)

    def call(self, *args, **kwargs):
        """Call method synchronously with retry on actor death.

        This is equivalent to ray.get(method.remote(*args, **kwargs)) but with
        automatic retry on actor death.
        """
        backoff = 1.0
        attempts = 0
        max_attempts = 5
        actor_name = self._handle._actor_name

        while True:
            try:
                ref = self._method.remote(*args, **kwargs)
                return ray.get(ref)
            except (ActorDiedError, ActorUnavailableError, RayActorError):
                attempts += 1
                if attempts >= max_attempts:
                    logger.error(
                        f"Actor '{actor_name}' failed after {max_attempts} attempts to call {self._method_name}"
                    )
                    raise
                logger.warning(
                    f"Actor '{actor_name}' died during call to {self._method_name}. Retrying after {backoff:.1f}s..."
                )
                self._handle._recreate_handle()
                self._method = getattr(self._handle._handle, self._method_name)
                time.sleep(backoff)
                backoff *= 2

    def options(self, **kwargs):
        """Forward .options() calls to the underlying method."""
        # Return a new wrapper with the options applied
        return _MethodWrapper(self._method.options(**kwargs), self._method_name, self._handle)
