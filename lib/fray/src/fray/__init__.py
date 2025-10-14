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
Fray: A Ray-like distributed execution abstraction layer.

Fray provides a context-based API for distributed task execution that can be
backed by different implementations (in-memory for testing, Ray for production,
and potentially others in the future).

Quick Start
-----------

For testing and development, the in-memory backend is automatically used:

    from fray import get_job_context

    ctx = get_job_context()

    def my_task(x):
        return x * 2

    ref = ctx.create_task(my_task, 5)
    result = ctx.get(ref)  # Returns 10

For production use with Ray or other backends, explicitly set the context:

    from fray import set_job_context
    from fray.ray_backend import RayJobContext

    ctx = RayJobContext()
    set_job_context(ctx)

    # Now all code using get_job_context() will use Ray

Key Concepts
------------

JobContext: Provides access to distributed task execution, object storage,
    and actor creation within a job. This is the primary interface for most
    code.

ClusterContext: Manages jobs on a cluster. Used to create, list, and delete
    jobs. Separates resource allocation from task execution.

RuntimeEnv: Specifies the execution environment (packages, resources, env vars)
    for a job.

Backends
--------

- in_memory (LocalJobContext): Thread-based local execution for testing
- ray (future): Production execution using Ray
"""

from fray.backend.in_memory import LocalClusterContext, LocalJobContext
from fray.cluster import ClusterContext
from fray.context import clear_job_context, get_job_context, set_job_context
from fray.job import JobContext
from fray.types import ActorOptions, Lifetime, Resource, RuntimeEnv, TaskOptions, TpuRunConfig

__version__ = "0.1.0"

__all__ = [
    "ActorOptions",
    "ClusterContext",
    "JobContext",
    "Lifetime",
    "LocalClusterContext",
    "LocalJobContext",
    "Resource",
    "RuntimeEnv",
    "TaskOptions",
    "TpuRunConfig",
    "clear_job_context",
    "get_job_context",
    "set_job_context",
]
