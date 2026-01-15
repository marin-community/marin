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
- controller: Protocol for cluster operations
- ports: Allocated ports for actor servers

Example:
    # In job code:
    from fluster.context import fluster_ctx

    ctx = fluster_ctx()
    print(f"Running job {ctx.job_id} in namespace {ctx.namespace}")

    # Get allocated port for actor server
    port = ctx.get_port("actor")

    # Submit a sub-job
    sub_job_id = ctx.controller.submit(entrypoint, "sub-job", resources)
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fluster.actor.resolver import Resolver

from fluster.cluster.types import Namespace


@runtime_checkable
class EndpointRegistry(Protocol):
    """Protocol for registering actor endpoints.

    Implementations:
    - RpcEndpointRegistry: Registers via RPC to controller
    - LocalEndpointRegistry: In-memory registry for local execution
    """

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint for actor discovery.

        Args:
            name: Actor name for discovery
            address: Address where actor is listening (host:port)
            metadata: Optional metadata for the endpoint

        Returns:
            Unique endpoint ID for later unregistration
        """
        ...

    def unregister(self, endpoint_id: str) -> None:
        """Unregister a previously registered endpoint.

        Args:
            endpoint_id: ID returned from register()
        """
        ...


@runtime_checkable
class ClusterController(Protocol):
    """Protocol for cluster operations.

    Abstracts the interface for job management and actor discovery,
    allowing the same code to work with both LocalClient and RpcClusterClient.

    Implementations:
    - RpcClusterClient: Backed by RPC to controller (with job_id set)
    - _LocalJobControllerAdapter: In-process for local execution
    """

    def submit(
        self,
        entrypoint: Any,  # Entrypoint
        name: str,
        resources: Any,  # cluster_pb2.ResourceSpec
        environment: Any = None,  # cluster_pb2.EnvironmentConfig | None
        ports: list[str] | None = None,
    ) -> Any:  # JobId
        """Submit a job for execution.

        Namespace is inherited from the current context.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name
            resources: Resource requirements (cluster_pb2.ResourceSpec)
            environment: Environment configuration (cluster_pb2.EnvironmentConfig)
            ports: Port names to allocate (e.g., ["actor", "metrics"])

        Returns:
            Job ID for the submitted job
        """
        ...

    def status(self, job_id: Any) -> Any:  # JobId -> cluster_pb2.JobStatus
        """Get job status.

        Args:
            job_id: Job ID to query

        Returns:
            JobStatus proto with current state
        """
        ...

    def wait(
        self,
        job_id: Any,  # JobId
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> Any:  # cluster_pb2.JobStatus
        """Wait for job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        ...

    def terminate(self, job_id: Any) -> None:  # JobId
        """Terminate a running job.

        Args:
            job_id: Job ID to terminate
        """
        ...

    def resolver(self) -> Any:  # -> Resolver
        """Get a resolver for actor discovery.

        The namespace is derived from the current job context.

        Returns:
            Resolver implementation
        """
        ...

    @property
    def endpoint_registry(self) -> EndpointRegistry:
        """Get the endpoint registry for actor registration."""
        ...

    @property
    def address(self) -> str:
        """Address of the controller (for compatibility)."""
        ...


@dataclass
class FlusterContext:
    """Unified execution context for Fluster.

    Available in any fluster job via `fluster_ctx()`. Contains all
    information about the current execution environment.

    The namespace is derived from the job_id: all jobs in a hierarchy share the
    same namespace (the root job's ID). This makes namespace an actor-only concept
    that doesn't need to be explicitly passed around.

    Attributes:
        job_id: Unique identifier for this job (hierarchical: "root/parent/child")
        worker_id: Identifier for the worker executing this job (may be None)
        controller: ClusterController for job/actor operations
        ports: Allocated ports by name (e.g., {"actor": 50001})
    """

    job_id: str
    worker_id: str | None = None
    controller: ClusterController | None = None
    ports: dict[str, int] = field(default_factory=dict)

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
            RuntimeError: If no controller is available
        """
        if self.controller is None:
            raise RuntimeError("No controller available in context")
        return self.controller.resolver()


# Module-level ContextVar for the current fluster context
_fluster_context: ContextVar[FlusterContext | None] = ContextVar(
    "fluster_context",
    default=None,
)


def fluster_ctx() -> FlusterContext:
    """Get the current FlusterContext.

    Returns:
        Current FlusterContext

    Raises:
        RuntimeError: If called outside of a fluster job
    """
    ctx = _fluster_context.get()
    if ctx is None:
        raise RuntimeError(
            "fluster_ctx() called outside of a fluster job. "
            "Ensure your code is running within a job submitted via "
            "LocalClient or RpcClusterClient."
        )
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
    - FLUSTER_WORKER_ID
    - FLUSTER_CONTROLLER_ADDRESS
    - FLUSTER_BUNDLE_GCS_PATH (for sub-job workspace inheritance)
    - FLUSTER_PORT_<NAME> (e.g., FLUSTER_PORT_ACTOR -> ports["actor"])

    Returns:
        Configured FlusterContext
    """
    import os

    job_id = os.environ.get("FLUSTER_JOB_ID", "")
    worker_id = os.environ.get("FLUSTER_WORKER_ID")
    controller_address = os.environ.get("FLUSTER_CONTROLLER_ADDRESS")
    bundle_gcs_path = os.environ.get("FLUSTER_BUNDLE_GCS_PATH")

    # Read ports from FLUSTER_PORT_* env vars
    ports = {}
    for key, value in os.environ.items():
        if key.startswith("FLUSTER_PORT_"):
            port_name = key[len("FLUSTER_PORT_") :].lower()
            ports[port_name] = int(value)

    controller = None
    if controller_address:
        from fluster.cluster.client import RpcClusterClient

        controller = RpcClusterClient(
            controller_address=controller_address,
            job_id=job_id,
            bundle_gcs_path=bundle_gcs_path,
        )

    return FlusterContext(
        job_id=job_id,
        worker_id=worker_id,
        controller=controller,
        ports=ports,
    )
