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

import shutil
import subprocess
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


MINIMAL_PYPROJECT = """\
[project]
name = "fluster-bundle"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "cloudpickle",
    "fluster",
]

[tool.uv.sources]
fluster = { path = "./fluster" }
"""


class BundleCreator:
    """Helper for creating workspace bundles.

    Creates minimal workspace bundles with pyproject.toml, uv.lock,
    and fluster source for job execution.
    """

    def __init__(self, fluster_root: Path | None = None):
        """Initialize bundle creator.

        Args:
            fluster_root: Path to fluster project root. If None, auto-detects
                from this file's location.
        """
        if fluster_root is None:
            # This file is at: lib/fluster/src/fluster/cluster/client.py
            # Fluster root is: lib/fluster/
            fluster_root = Path(__file__).parent.parent.parent.parent
        self._fluster_root = fluster_root

    def create_bundle(self, temp_dir: Path | None = None) -> bytes:
        """Create a workspace bundle.

        Creates a zip file containing:
        - pyproject.toml with fluster dependency
        - uv.lock generated from the workspace
        - fluster source code

        Args:
            temp_dir: Optional temp directory for workspace. Creates one if None.

        Returns:
            Bundle as bytes (zip file contents)
        """
        if temp_dir is None:
            with tempfile.TemporaryDirectory(prefix="bundle_") as td:
                return self._create_bundle_in_dir(Path(td))
        return self._create_bundle_in_dir(temp_dir)

    def _create_bundle_in_dir(self, temp_dir: Path) -> bytes:
        """Create bundle in the given temp directory."""
        workspace = temp_dir / "workspace"
        workspace.mkdir(exist_ok=True)

        # Write minimal pyproject.toml
        (workspace / "pyproject.toml").write_text(MINIMAL_PYPROJECT)

        # Copy fluster source
        fluster_dest = workspace / "fluster"
        fluster_dest.mkdir(exist_ok=True)
        shutil.copy2(self._fluster_root / "pyproject.toml", fluster_dest / "pyproject.toml")
        shutil.copytree(
            self._fluster_root / "src",
            fluster_dest / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )

        # Generate uv.lock
        subprocess.run(
            ["uv", "lock"],
            cwd=workspace,
            check=True,
            capture_output=True,
        )

        # Create zip bundle
        bundle_path = temp_dir / "bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in workspace.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(workspace))

        return bundle_path.read_bytes()


class RpcClusterClient:
    """ClusterClient implementation using RPC to controller."""

    def __init__(
        self,
        controller_address: str,
        bundle_blob: bytes,
        timeout_ms: int = 30000,
    ):
        """Initialize RPC cluster client.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:8080")
            bundle_blob: Workspace bundle bytes (use BundleCreator to create)
            timeout_ms: RPC timeout in milliseconds
        """
        self._address = controller_address
        self._bundle_blob = bundle_blob
        self._timeout_ms = timeout_ms
        self._client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
        )

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
            bundle_blob=self._bundle_blob,
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
