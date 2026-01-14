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

"""Cluster client for job management.

This module provides:
- ClusterClient: Protocol for cluster job operations
- RpcClusterClient: Default implementation using RPC to controller
- BundleCreator: Helper for creating workspace bundles
"""

import tempfile
import time
import zipfile
from pathlib import Path
from typing import Protocol

import cloudpickle

from fluster import cluster_pb2
from fluster.cluster.types import Entrypoint, JobId, is_job_finished
from fluster.cluster_connect import ControllerServiceClientSync


class ClusterClient(Protocol):
    """Protocol for cluster job operations.

    This is the interface WorkerPool and other clients use to interact
    with a cluster. Default implementation is RpcClusterClient.
    """

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        namespace: str = "<local>",
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job to the cluster.

        Args:
            entrypoint: Job entrypoint (callable + args/kwargs)
            name: Job name
            resources: Resource requirements
            environment: Environment configuration
            namespace: Namespace for actor isolation
            ports: Port names to allocate (e.g., ["actor", "metrics"])

        Returns:
            Job ID
        """
        ...

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Job ID to query

        Returns:
            JobInfo proto with current state
        """
        ...

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
            Final JobInfo

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        ...

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Args:
            job_id: Job ID to terminate
        """
        ...

    @property
    def controller_address(self) -> str:
        """Address of the cluster controller (for resolver)."""
        ...


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


class RpcClusterClient:
    """ClusterClient implementation using RPC to controller.

    Example:
        client = RpcClusterClient("http://controller:8080", workspace=Path("./my-project"))
        entrypoint = Entrypoint.from_callable(my_func, arg1, arg2)
        job_id = client.submit(entrypoint, "my-job", resources)
    """

    def __init__(
        self,
        controller_address: str,
        *,
        workspace: Path,
        timeout_ms: int = 30000,
    ):
        """Initialize RPC cluster client.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            workspace: Path to workspace directory containing pyproject.toml.
                This directory will be bundled and sent to workers.
            timeout_ms: RPC timeout in milliseconds
        """
        self._address = controller_address
        self._workspace = workspace
        self._timeout_ms = timeout_ms
        self._bundle_blob: bytes | None = None
        self._client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
        )

    def _get_bundle(self) -> bytes:
        """Get workspace bundle (lazy creation with caching)."""
        if self._bundle_blob is None:
            creator = BundleCreator(self._workspace)
            self._bundle_blob = creator.create_bundle()
        return self._bundle_blob

    @property
    def controller_address(self) -> str:
        return self._address

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        namespace: str = "<local>",
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job to the cluster."""
        serialized = cloudpickle.dumps(entrypoint)

        # Build environment with namespace
        env = dict(environment.env_vars) if environment else {}
        env["FLUSTER_NAMESPACE"] = namespace

        env_config = cluster_pb2.EnvironmentConfig(
            workspace=environment.workspace if environment else "/app",
            pip_packages=list(environment.pip_packages) if environment else [],
            env_vars=env,
            extras=list(environment.extras) if environment else [],
        )

        request = cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=serialized,
            resources=resources,
            environment=env_config,
            bundle_blob=self._get_bundle(),
            ports=ports or [],
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
        poll_interval: float = 0.5,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete."""
        start = time.time()

        while time.time() - start < timeout:
            job_info = self.status(job_id)
            if is_job_finished(job_info.state):
                return job_info
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job."""
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self._client.terminate_job(request)
