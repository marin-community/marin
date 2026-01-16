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

"""High-level cluster client that wraps cluster operations with smart features.

This module provides the ClusterClient implementation that wraps the low-level
ClusterOperations protocol and adds:
- Job hierarchy (auto-prefixes job names with parent job_id)
- Namespace-aware resolver
- Context-based default parameters

This is the "smart" client layer that user code interacts with.
"""

from fluster.actor.types import Resolver
from fluster.client.context import get_fluster_ctx
from fluster.client.resolver import ClusterResolver
from fluster.cluster.client.protocols import ClusterOperations
from fluster.cluster.types import Entrypoint, JobId
from fluster.rpc import cluster_pb2


class RpcClusterClient:
    """High-level cluster client wrapping RPC-based ClusterOperations.

    This client adds "smart" features on top of the raw cluster operations:
    - Job hierarchy: auto-prefixes job names with parent job_id
    - Namespace-aware resolver
    - Context-based default parameters

    Can be used in two modes:
    1. External client (with workspace): Bundles workspace and submits jobs from outside.
    2. Inside-job client: Used by job code to submit sub-jobs.

    For the raw cluster operations without magic, use cluster.client.rpc.RpcClusterOperations.
    """

    def __init__(self, cluster: ClusterOperations):
        """Initialize high-level cluster client.

        Args:
            cluster: Low-level ClusterOperations implementation
        """
        self._cluster = cluster
        self._resolver: Resolver | None = None

    def resolver(self) -> Resolver:
        """Get a resolver for actor discovery.

        The resolver uses the namespace derived from the current job context.
        The resolver is cached for efficiency.

        Returns:
            Resolver instance scoped to current namespace
        """
        if self._resolver is None:
            # TODO: Get controller address from ClusterOperations once it's exposed
            # For now, this assumes RpcClusterOperations
            ctx = get_fluster_ctx()
            if ctx is None:
                raise RuntimeError("No FlusterContext - must be called from within a job")
            # This is a bit of a hack - we need controller_address from somewhere
            # In practice, this gets created by RpcClusterClient which has it
            raise NotImplementedError("resolver() not yet implemented for generic ClusterClient")
        return self._resolver

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job with automatic job_id hierarchy.

        The name is combined with the current job's name (if any) to form a
        hierarchical identifier. For example, if the current job is "my-exp"
        and you submit with name="worker-0", the full name becomes "my-exp/worker-0".

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])

        Returns:
            Job ID (the full hierarchical name)

        Raises:
            ValueError: If name contains '/'
        """
        # Validate name
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")

        # Get parent job ID from context
        ctx = get_fluster_ctx()
        parent_job_id = ctx.job_id if ctx else ""

        # Construct full hierarchical name
        if parent_job_id:
            full_job_id = f"{parent_job_id}/{name}"
        else:
            full_job_id = name

        # Submit via low-level cluster operations
        self._cluster.submit_job(
            job_id=full_job_id,
            entrypoint=entrypoint,
            resources=resources,
            environment=environment,
            ports=ports,
        )

        return JobId(full_job_id)

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Job ID to query

        Returns:
            JobStatus proto with current state
        """
        return self._cluster.get_job_status(job_id)

    def wait(
        self,
        job_id: JobId,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> cluster_pb2.JobStatus:
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
        return self._cluster.wait_for_job(job_id, timeout, poll_interval)

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Args:
            job_id: Job ID to terminate
        """
        self._cluster.terminate_job(job_id)
