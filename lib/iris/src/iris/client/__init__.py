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

"""High-level client layer with automatic namespace management and job hierarchy."""

from iris.actor.resolver import (
    FixedResolver,
    GcsResolver,
    ResolvedEndpoint,
    Resolver,
    ResolveResult,
)
from iris.client.client import (
    EndpointRegistry,
    IrisClient,
    IrisContext,
    Job,
    JobFailedError,
    LocalClientConfig,
    LogEntry,
    Task,
    get_iris_ctx,
    iris_ctx,
    iris_ctx_scope,
)
from iris.client.resolver import ClusterResolver
from iris.client.worker_pool import WorkerPool, WorkerPoolConfig

__all__ = [
    "ClusterResolver",
    "EndpointRegistry",
    "FixedResolver",
    "GcsResolver",
    "IrisClient",
    "IrisContext",
    "Job",
    "JobFailedError",
    "LocalClientConfig",
    "LogEntry",
    "ResolveResult",
    "ResolvedEndpoint",
    "Resolver",
    "Task",
    "WorkerPool",
    "WorkerPoolConfig",
    "get_iris_ctx",
    "iris_ctx",
    "iris_ctx_scope",
]
