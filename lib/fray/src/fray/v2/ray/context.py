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

"""Ray-based backend context for Zephyr-style put/get/run/wait dispatch.

This module isolates all direct Ray imports so that higher-level code
(e.g. Zephyr) can use distributed execution without importing Ray itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ray


def is_ray_initialized() -> bool:
    """Check if Ray is initialized without requiring callers to import Ray."""
    return ray.is_initialized()


class RayBackendContext:
    """Ray-based distributed execution context.

    Provides put/get/run/wait primitives that Zephyr's Backend uses to dispatch
    work to Ray workers. Zephyr imports this from fray.v2.ray rather than
    importing Ray directly.
    """

    def __init__(self, ray_options: dict | None = None):
        self.ray_options = ray_options or {}

    def put(self, obj: Any):
        return ray.put(obj)

    def get(self, ref):
        return ray.get(ref)

    def run(self, fn: Callable, *args, name: str | None = None):
        if self.ray_options:
            remote_fn = ray.remote(**self.ray_options)(fn)
        else:
            remote_fn = ray.remote(max_retries=100)(fn)

        options: dict[str, Any] = {"scheduling_strategy": "SPREAD"}
        if name:
            options["name"] = name
        return remote_fn.options(**options).remote(*args)

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        ready, pending = ray.wait(futures, num_returns=num_returns, fetch_local=False)
        return list(ready), list(pending)
