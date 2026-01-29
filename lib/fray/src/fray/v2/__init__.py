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

"""Fray v2 - Iris-aligned distributed computing API.

This module provides the public API for Fray v2, which is designed to work
with multiple backends (Local, Iris, Ray) using Iris-shaped semantics.

Basic usage:
    from fray.v2 import current_cluster, Entrypoint, ResourceSpec

    cluster = current_cluster()

    # Submit a job
    job = cluster.submit(
        Entrypoint.from_callable(my_function, arg1, arg2),
        name="my-job",
        resources=ResourceSpec.with_cpu(cpu=2, memory="4g"),
    )
    job.wait()

    # Use a worker pool
    with cluster.worker_pool(num_workers=4, resources=ResourceSpec()) as pool:
        futures = [pool.submit(process, item) for item in items]
        results = [f.result() for f in futures]

    # Host actors
    server = ActorServer(cluster)
    server.register("my-actor", MyActor())
    server.serve_background()

    pool = cluster.resolver().lookup("my-actor")
    pool.wait_for_size(1)
    result = pool.call().my_method(arg)
"""

# Types
from fray.v2.types import (
    Entrypoint,
    EnvironmentSpec,
    JobId,
    JobStatus,
    Namespace,
    ResourceSpec,
    namespace_from_job_id,
)

# Cluster protocol and factory
from fray.v2.cluster import (
    ActorPool,
    BroadcastResult,
    Cluster,
    Job,
    Resolver,
    WorkerPool,
    create_cluster,
    current_cluster,
    set_current_cluster,
)

# Actor system
from fray.v2.actor import ActorServer, FixedResolver

# Backends (for direct import if needed)
from fray.v2.backends import LocalCluster

# Ray backend is optional (requires ray)
try:
    from fray.v2.backends.ray import RayCluster
except ImportError:
    RayCluster = None  # type: ignore[misc,assignment]

# Iris backend is optional (requires iris)
try:
    from fray.v2.backends.iris import IrisCluster
except ImportError:
    IrisCluster = None  # type: ignore[misc,assignment]

__all__ = [
    "ActorPool",
    "ActorServer",
    "BroadcastResult",
    "Cluster",
    "Entrypoint",
    "EnvironmentSpec",
    "FixedResolver",
    "IrisCluster",
    "Job",
    "JobId",
    "JobStatus",
    "LocalCluster",
    "Namespace",
    "RayCluster",
    "Resolver",
    "ResourceSpec",
    "WorkerPool",
    "create_cluster",
    "current_cluster",
    "namespace_from_job_id",
    "set_current_cluster",
]
