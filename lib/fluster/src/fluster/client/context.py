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

"""Fluster execution context management.

This module provides a unified context for Fluster jobs, enabling transparent
local-vs-remote execution. The FlusterContext is available via fluster_ctx()
in any job code, providing access to:
- namespace: Namespace for actor isolation
- job_id: Unique job identifier
- worker_id: Worker executing the job
- client: ClusterClient for job operations
- registry: EndpointRegistry for actor registration
- ports: Allocated ports for actor servers

Example:
    # In job code:
    from fluster.client.context import fluster_ctx

    ctx = fluster_ctx()
    print(f"Running job {ctx.job_id} in namespace {ctx.namespace}")

    # Get allocated port for actor server
    port = ctx.get_port("actor")

    # Submit a sub-job
    sub_job_id = ctx.client.submit(entrypoint, "sub-job", resources)
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from fluster.cluster.client.job_info import JobInfo
from fluster.cluster.types import Namespace


@dataclass
class FlusterContext:
    """Unified execution context for Fluster.

    Available in any fluster job via `fluster_ctx()`. Contains all
    information about the current execution environment.

    The namespace is derived from the job_id: all jobs in a hierarchy share the
    same namespace (the root job's ID). This makes namespace an actor-only concept
    that doesn't need to be explicitly passed around.

    This wraps the cluster.client.job_info.JobInfo with additional smart features
    like client/registry references.

    Attributes:
        job_id: Unique identifier for this job (hierarchical: "root/parent/child")
        attempt_id: Attempt number for this job execution (0-based)
        worker_id: Identifier for the worker executing this job (may be None)
        client: ClusterClient for job operations (submit, status, wait, etc.)
        registry: EndpointRegistry for actor endpoint registration
        ports: Allocated ports by name (e.g., {"actor": 50001})
    """

    job_id: str
    attempt_id: int = 0
    worker_id: str | None = None
    client: "ClusterClient | None" = None
    registry: "EndpointRegistry | None" = None
    ports: dict[str, int] | None = None

    def __post_init__(self):
        if self.ports is None:
            object.__setattr__(self, "ports", {})

    @property
    def namespace(self) -> Namespace:
        """Namespace derived from the root job ID.

        All jobs in a hierarchy share the same namespace, enabling actors
        to be discovered across the job tree.
        """
        return Namespace.from_job_id(self.job_id)

    @property
    def parent_job_id(self) -> str | None:
        """Parent job ID, or None if this is a root job.

        For job_id "root/parent/child", returns "root/parent".
        For job_id "root", returns None.
        """
        parts = self.job_id.rsplit("/", 1)
        if len(parts) == 1:
            return None
        return parts[0]

    def get_port(self, name: str) -> int:
        """Get an allocated port by name.

        Args:
            name: Port name (e.g., "actor")

        Returns:
            Port number

        Raises:
            KeyError: If port was not allocated for this job
        """
        if name not in self.ports:
            raise KeyError(
                f"Port '{name}' not allocated. "
                f"Available ports: {list(self.ports.keys()) or 'none'}. "
                f"Did you request ports=['actor'] when submitting the job?"
            )
        return self.ports[name]

    @property
    def resolver(self) -> "Resolver":
        """Get a resolver for actor discovery.

        The resolver uses the namespace derived from this context's job ID.

        Raises:
            RuntimeError: If no client is available
        """
        if self.client is None:
            raise RuntimeError("No client available in context")
        return self.client.resolver()

    @staticmethod
    def from_job_info(
        info: JobInfo,
        client: "ClusterClient | None" = None,
        registry: "EndpointRegistry | None" = None,
    ) -> "FlusterContext":
        """Create FlusterContext from JobInfo.

        Args:
            info: JobInfo from cluster layer
            client: Optional ClusterClient instance
            registry: Optional EndpointRegistry instance

        Returns:
            FlusterContext with metadata from JobInfo
        """
        return FlusterContext(
            job_id=info.job_id,
            attempt_id=info.attempt_id,
            worker_id=info.worker_id,
            client=client,
            registry=registry,
            ports=dict(info.ports),
        )


# Module-level ContextVar for the current fluster context
_fluster_context: ContextVar[FlusterContext | None] = ContextVar(
    "fluster_context",
    default=None,
)


def fluster_ctx() -> FlusterContext:
    """Get or create FlusterContext from environment.

    On first call (when context is not set), automatically creates context
    from FLUSTER_* environment variables. This enables worker thunks to
    skip explicit context setup.

    Returns:
        Current FlusterContext
    """
    ctx = _fluster_context.get()
    if ctx is None:
        ctx = create_context_from_env()
        _fluster_context.set(ctx)
    return ctx


def get_fluster_ctx() -> FlusterContext | None:
    """Get the current FlusterContext, or None if not in a job.

    Unlike fluster_ctx(), this function does not raise if called
    outside a job context.

    Returns:
        Current FlusterContext or None
    """
    return _fluster_context.get()


@contextmanager
def fluster_ctx_scope(ctx: FlusterContext) -> Generator[FlusterContext, None, None]:
    """Set the fluster context for the duration of this scope.

    This is used internally by LocalClient and RpcClusterClient to
    inject context before executing job entrypoints.

    Args:
        ctx: Context to set for this scope

    Yields:
        The provided context

    Example:
        ctx = FlusterContext(job_id="my-namespace/job-1", worker_id="worker-1")
        with fluster_ctx_scope(ctx):
            # fluster_ctx() now returns ctx
            # ctx.namespace returns Namespace("my-namespace")
            my_job_function()
    """
    token = _fluster_context.set(ctx)
    try:
        yield ctx
    finally:
        _fluster_context.reset(token)


def create_context_from_env() -> FlusterContext:
    """Create FlusterContext from environment variables.

    Used by workers to set up context when executing job entrypoints.
    Reads:
    - FLUSTER_JOB_ID
    - FLUSTER_ATTEMPT_ID
    - FLUSTER_WORKER_ID
    - FLUSTER_CONTROLLER_ADDRESS
    - FLUSTER_BUNDLE_GCS_PATH (for sub-job workspace inheritance)
    - FLUSTER_PORT_<NAME> (e.g., FLUSTER_PORT_ACTOR -> ports["actor"])

    Returns:
        Configured FlusterContext
    """
    from fluster.cluster.client.job_info import get_job_info

    # Get job info from environment
    job_info = get_job_info()
    if job_info is None:
        # If no job info available, create minimal context
        return FlusterContext(job_id="")

    # Set up client and registry if controller address is available
    client = None
    registry = None
    if job_info.controller_address:
        from fluster.client.rpc_client import RpcClusterClient, RpcEndpointRegistry
        from fluster.rpc.cluster_connect import ControllerServiceClientSync

        bundle_gcs_path = os.environ.get("FLUSTER_BUNDLE_GCS_PATH")

        rpc_client = ControllerServiceClientSync(
            address=job_info.controller_address,
            timeout_ms=30000,
        )
        client = RpcClusterClient(
            controller_address=job_info.controller_address,
            bundle_gcs_path=bundle_gcs_path,
        )
        registry = RpcEndpointRegistry(rpc_client)

    return FlusterContext.from_job_info(job_info, client=client, registry=registry)
