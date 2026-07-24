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
from iris.cluster.setup_scripts import default_setup_script, iris_runtime_setup_script

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
    "default_setup_script",
    "get_iris_ctx",
    "iris_ctx",
    "iris_ctx_scope",
    "iris_runtime_setup_script",
]
