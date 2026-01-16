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

"""RPC-based cluster client implementation.

This module provides:
- BundleCreator: Helper for creating workspace bundles
- RpcEndpointRegistry: Endpoint registry via RPC to controller
- RpcClusterClient: Full cluster client using RPC to controller
"""

import tempfile
import time
import zipfile
from pathlib import Path

import cloudpickle

from fluster.actor.types import Resolver
from fluster.client.context import get_fluster_ctx
from fluster.client.protocols import ClusterResolver, EndpointRegistry
from fluster.cluster.types import Entrypoint, JobId, Namespace
from fluster.rpc import cluster_pb2
from fluster.rpc.cluster_connect import ControllerServiceClientSync
from fluster.time_utils import ExponentialBackoff


def is_job_finished(state: int) -> bool:
    """Check if job has reached terminal state."""
    return state in (
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        cluster_pb2.JOB_STATE_UNSCHEDULABLE,
    )


class BundleCreator:
    """Helper for creating workspace bundles.

    Bundles a user's workspace directory (containing pyproject.toml, uv.lock,
    and source code) into a zip file for job execution.

    The workspace must already have fluster as a dependency in pyproject.toml.
    If uv.lock doesn't exist, it will be generated.
    """

    def __init__(self, workspace: Path):
        """Initialize bundle creator.

        Args:
            workspace: Path to workspace directory containing pyproject.toml
        """
        self._workspace = workspace

    def create_bundle(self) -> bytes:
        """Create a workspace bundle.

        Creates a zip file containing the workspace directory contents.
        Excludes common non-essential files like __pycache__, .git, etc.

        Returns:
            Bundle as bytes (zip file contents)
        """
        with tempfile.TemporaryDirectory(prefix="bundle_") as td:
            bundle_path = Path(td) / "bundle.zip"
            with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in self._workspace.rglob("*"):
                    if file.is_file() and not self._should_exclude(file):
                        zf.write(file, file.relative_to(self._workspace))
            return bundle_path.read_bytes()

    def _should_exclude(self, path: Path) -> bool:
        """Check if a file should be excluded from the bundle."""
        exclude_patterns = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "*.pyc",
            "*.egg-info",
            ".venv",
            "venv",
            "node_modules",
        }
        parts = path.relative_to(self._workspace).parts
        for part in parts:
            for pattern in exclude_patterns:
                if pattern.startswith("*"):
                    if part.endswith(pattern[1:]):
                        return True
                elif part == pattern:
                    return True
        return False


class RpcEndpointRegistry:
    """EndpointRegistry implementation that registers via RPC to controller.

    Used by ActorServer to register endpoints when running in a remote worker.
    Names are auto-prefixed with the namespace (root job ID) for isolation.

    The namespace prefix is computed dynamically from the current FlusterContext,
    so a single registry instance can be shared.
    """

    def __init__(self, client: ControllerServiceClientSync):
        """Initialize the RPC endpoint registry.

        Args:
            client: RPC client for controller communication
        """
        self._client = client

    def _get_job_context(self) -> tuple[str, str]:
        """Get job_id and namespace_prefix from current FlusterContext."""
        ctx = get_fluster_ctx()
        if ctx is None:
            raise RuntimeError("No FlusterContext - must be called from within a job")
        namespace_prefix = str(Namespace.from_job_id(ctx.job_id))
        return ctx.job_id, namespace_prefix

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint via RPC to controller.

        The name is auto-prefixed with the namespace (root job ID) for isolation.
        For example, registering "calculator" with job "abc123/worker-0" stores
        as "abc123/calculator".

        Args:
            name: Actor name for discovery (will be prefixed)
            address: Address where actor is listening (host:port)
            metadata: Optional metadata

        Returns:
            Endpoint ID assigned by controller
        """
        job_id, namespace_prefix = self._get_job_context()
        prefixed_name = f"{namespace_prefix}/{name}"
        request = cluster_pb2.Controller.RegisterEndpointRequest(
            name=prefixed_name,
            address=address,
            job_id=job_id,
            metadata=metadata or {},
        )
        response = self._client.register_endpoint(request)
        return response.endpoint_id

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint.

        This is a no-op for the RPC registry. The controller automatically
        cleans up endpoints when jobs terminate, so explicit unregistration
        is not required.

        Args:
            endpoint_id: Endpoint ID (ignored)
        """
        # No-op: controller auto-cleans endpoints on job termination
        del endpoint_id


class RpcClusterClient:
    """ClusterClient implementation using RPC to controller.

    Can be used in two modes:
    1. External client (with workspace): Bundles workspace and submits jobs from outside.
       Example: client = RpcClusterClient("http://controller:8080", workspace=Path("./my-project"))

    2. Inside-job client: Used by job code to submit sub-jobs. The job_id is automatically
       inferred from the FlusterContext.
       Example: client = RpcClusterClient("http://controller:8080", bundle_gcs_path="gs://...")
    """

    def __init__(
        self,
        controller_address: str,
        *,
        workspace: Path | None = None,
        bundle_gcs_path: str | None = None,
        timeout_ms: int = 30000,
    ):
        """Initialize RPC cluster client.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            workspace: Path to workspace directory containing pyproject.toml.
                If provided, this directory will be bundled and sent to workers.
                Required for external job submission.
            bundle_gcs_path: GCS path to workspace bundle for sub-job inheritance.
                When set, sub-jobs use this path instead of creating new bundles.
            timeout_ms: RPC timeout in milliseconds
        """
        self._address = controller_address
        self._workspace = workspace
        self._bundle_gcs_path = bundle_gcs_path
        self._timeout_ms = timeout_ms
        self._bundle_blob: bytes | None = None
        self._registry: RpcEndpointRegistry | None = None
        self._resolver: ClusterResolver | None = None
        self._client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
        )

    def _get_bundle(self) -> bytes:
        """Get workspace bundle (lazy creation with caching)."""
        if self._workspace is None:
            return b""
        if self._bundle_blob is None:
            creator = BundleCreator(self._workspace)
            self._bundle_blob = creator.create_bundle()
        return self._bundle_blob

    @property
    def endpoint_registry(self) -> EndpointRegistry:
        """Get the endpoint registry for actor registration.

        The registry looks up job_id from the FlusterContext dynamically.
        """
        if self._registry is None:
            self._registry = RpcEndpointRegistry(self._client)
        return self._registry

    @property
    def address(self) -> str:
        """Controller address."""
        return self._address

    def resolver(self) -> Resolver:
        """Get a resolver for actor discovery.

        The resolver uses the namespace derived from the current job context.
        The resolver is cached for efficiency.
        """
        if self._resolver is None:
            self._resolver = ClusterResolver(self._address)
        return self._resolver

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
    ) -> JobId:
        """Submit a job to the cluster.

        The name is combined with the current job's name (if any) to form a
        hierarchical identifier. For example, if the current job is "my-exp"
        and you submit with name="worker-0", the full name becomes "my-exp/worker-0".

        Child jobs are automatically terminated when their parent is terminated.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name (cannot contain '/')
            resources: Resource requirements
            environment: Environment configuration
            ports: Port names to allocate (e.g., ["actor", "metrics"])
            scheduling_timeout_seconds: Timeout for scheduling (0 = no timeout)

        Returns:
            Job ID (the full hierarchical name)

        Raises:
            ValueError: If name contains '/'
        """
        # Validate name
        if "/" in name:
            raise ValueError("Job name cannot contain '/'")

        serialized = cloudpickle.dumps(entrypoint)

        env_config = cluster_pb2.EnvironmentConfig(
            workspace=environment.workspace if environment else "/app",
            pip_packages=list(environment.pip_packages) if environment else [],
            env_vars=dict(environment.env_vars) if environment else {},
            extras=list(environment.extras) if environment else [],
        )

        # Get parent job ID from context
        ctx = get_fluster_ctx()
        parent_job_id = ctx.job_id if ctx else ""

        # Construct full hierarchical name
        if parent_job_id:
            full_name = f"{parent_job_id}/{name}"
        else:
            full_name = name

        # Use bundle_gcs_path if available (inside-job mode inherits parent workspace),
        # otherwise create bundle_blob from workspace
        if self._bundle_gcs_path:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=full_name,
                serialized_entrypoint=serialized,
                resources=resources,
                environment=env_config,
                bundle_gcs_path=self._bundle_gcs_path,
                ports=ports or [],
                parent_job_id=parent_job_id,
                scheduling_timeout_seconds=scheduling_timeout_seconds,
            )
        else:
            request = cluster_pb2.Controller.LaunchJobRequest(
                name=full_name,
                serialized_entrypoint=serialized,
                resources=resources,
                environment=env_config,
                bundle_blob=self._get_bundle(),
                ports=ports or [],
                parent_job_id=parent_job_id,
                scheduling_timeout_seconds=scheduling_timeout_seconds,
            )
        response = self._client.launch_job(request)
        return JobId(response.job_id)

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status."""
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self._client.get_job_status(request)
        return response.job

    def wait(
        self,
        job_id: JobId,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete with exponential backoff polling."""
        start = time.monotonic()
        backoff = ExponentialBackoff(initial=0.1, maximum=poll_interval)

        while True:
            job_info = self.status(job_id)
            if is_job_finished(job_info.state):
                return job_info

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            interval = backoff.next_interval()
            remaining = timeout - elapsed
            time.sleep(min(interval, remaining))

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job."""
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self._client.terminate_job(request)
