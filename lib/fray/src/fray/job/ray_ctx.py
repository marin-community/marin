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

"""Ray execution context for distributed computing."""

from collections.abc import Callable
from typing import Any, Literal

try:
    import ray
except ImportError:
    ray = None

__all__ = ["RayContext"]


class RayContext:
    """Execution context using Ray for distributed execution."""

    def __init__(self, ray_options: dict | None = None):
        """Initialize Ray context.

        Args:
            ray_options: Options to pass to ray.remote() (e.g., memory, num_cpus, num_gpus)
        """
        self.ray_options = ray_options or {}

    def put(self, obj: Any):
        """Not supported - use ray.put() directly if needed."""
        raise NotImplementedError("RayContext.put() has been removed - use ray.put() directly if needed")

    def get(self, ref):
        """Retrieve an object from Ray's object store."""
        return ray.get(ref)

    def run(self, fn: Callable, *args):
        """Execute function remotely with configured Ray options.

        Uses SPREAD scheduling strategy to avoid running on head node.
        """
        if self.ray_options:
            remote_fn = ray.remote(**self.ray_options)(fn)
        else:
            remote_fn = ray.remote(fn)

        return remote_fn.options(scheduling_strategy="SPREAD").remote(*args)

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        """Wait for Ray futures to complete."""
        # NOTE: fetch_local=False is paramount to avoid copying the data to the Zephyr
        # driver node, especially for the data futures.
        ready, pending = ray.wait(futures, num_returns=num_returns, fetch_local=False)
        return list(ready), list(pending)

    def create_actor(
        self,
        actor_class: type,
        *args,
        name: str | None = None,
        get_if_exists: bool = False,
        lifetime: Literal["non_detached", "detached"] = "non_detached",
        preemptible: bool = True,
        **kwargs,
    ) -> Any:
        options = {}
        if name is not None:
            options["name"] = name
        options["get_if_exists"] = get_if_exists
        options["lifetime"] = lifetime

        # run non-preemptible actors on the head node for persistence
        if not preemptible:
            options["resources"] = {"head_node": 0.0001}

        remote_class = ray.remote(actor_class)
        ray_actor = remote_class.options(**options).remote(*args, **kwargs)

        return ray_actor
