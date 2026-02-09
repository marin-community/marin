# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
    Task,
    TaskLogEntry,
    TaskLogError,
    TaskLogsResult,
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
    "ResolveResult",
    "ResolvedEndpoint",
    "Resolver",
    "Task",
    "TaskLogEntry",
    "TaskLogError",
    "TaskLogsResult",
    "WorkerPool",
    "WorkerPoolConfig",
    "get_iris_ctx",
    "iris_ctx",
    "iris_ctx_scope",
]
