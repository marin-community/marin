# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""High-level client layer with automatic namespace management and job hierarchy."""

from iris.actor.resolver import (
    FixedResolver,
    ResolvedEndpoint,
    Resolver,
    ResolveResult,
)
from iris.client.client import (
    EndpointRegistry,
    IrisClient,
    IrisContext,
    Job,
    JobAlreadyExists,
    JobFailedError,
    LocalClientConfig,
    Task,
    TaskLogEntry,
    get_iris_ctx,
    iris_ctx,
    iris_ctx_scope,
)
from iris.client.resolver import ClusterResolver
from iris.client.worker_pool import WorkerPool, WorkerPoolConfig

# Side-effect import: registers the ``iris://`` URL scheme with
# ``rigging.resolver``. Importing ``iris.client`` is the standard activation
# path; off-cluster code that does not import iris cannot resolve ``iris://``
# (raises ``ValueError("unsupported scheme: 'iris'")``), which is the right
# failure mode.
from iris.client import resolver_plugin

__all__ = [
    "ClusterResolver",
    "EndpointRegistry",
    "FixedResolver",
    "IrisClient",
    "IrisContext",
    "Job",
    "JobAlreadyExists",
    "JobFailedError",
    "LocalClientConfig",
    "ResolveResult",
    "ResolvedEndpoint",
    "Resolver",
    "Task",
    "TaskLogEntry",
    "WorkerPool",
    "WorkerPoolConfig",
    "get_iris_ctx",
    "iris_ctx",
    "iris_ctx_scope",
]
