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
    get_iris_ctx,
    iris_ctx,
    iris_ctx_scope,
)
from iris.client.job_info import JobInfo, get_job_info, resolve_job_user, set_job_info
from iris.client.protocol import ClusterClient, ResourceClientProtocol
from iris.client.remote_client import RemoteClusterClient
from iris.client.resolver import ClusterResolver
from iris.cluster.setup_scripts import (
    default_setup_script,
    iris_runtime_setup_script,
)
