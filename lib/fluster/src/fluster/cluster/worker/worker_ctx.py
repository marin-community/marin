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

"""Context manager for manual worker testing.

Provides an easy-to-use interface for starting a worker server and
submitting jobs interactively.

Example:
    ```python
    import asyncio
    from fluster.cluster.worker.worker_ctx import WorkerContext

    async def main():
        async with WorkerContext() as worker:
            # Submit a job
            def my_job():
                print("Hello from job")
                return 42

            job_id = await worker.submit(my_job)
            print(f"Submitted job: {job_id}")

            # Wait for completion
            await worker.wait(job_id)

            # Get logs
            logs = await worker.logs(job_id)
            for log in logs:
                print(log)

            # Get status
            status = await worker.status(job_id)
            print(f"Job state: {status.state}")

    asyncio.run(main())
    ```
"""

import asyncio
import socket
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import cloudpickle
import httpx

from fluster import cluster_pb2
from fluster.cluster.types import Entrypoint
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.builder import ImageCache, VenvCache
from fluster.cluster.worker.dashboard import WorkerDashboard
from fluster.cluster.worker.docker import DockerRuntime
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.service import WorkerServiceImpl
from fluster.cluster_connect import WorkerServiceClient


class WorkerContext:
    """Context manager for running a worker server and submitting jobs.

    Automatically:
    - Starts HTTP server with Connect RPC
    - Creates temporary cache directories
    - Initializes all worker components
    - Provides simple API for job submission
    - Cleans up on exit

    Args:
        host: Bind host (default: "127.0.0.1")
        port: Bind port (default: 0 for ephemeral)
        registry: Docker registry for images (default: "localhost:5000")
        max_concurrent_jobs: Max concurrent jobs (default: 5)
        port_range: Port range for job ports (default: (30000, 40000))
        workspace: Workspace directory to bundle (default: current directory)
        cache_dir: Cache directory (default: temp directory)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        registry: str = "localhost:5000",
        max_concurrent_jobs: int = 5,
        port_range: tuple[int, int] = (30000, 40000),
        workspace: Path | None = None,
        cache_dir: Path | None = None,
    ):
        self.host = host
        self.port = port
        self.registry = registry
        self.max_concurrent_jobs = max_concurrent_jobs
        self.port_range = port_range
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self._cache_dir = cache_dir
        self._temp_dir = None

        self._server_task = None
        self._dashboard = None
        self._client = None
        self._http_client = None
        self._actual_port = None

        # Cached workspace bundle
        self._workspace_bundle = None

    async def __aenter__(self):
        """Start worker server."""
        # Setup cache directory
        if self._cache_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            cache_path = Path(self._temp_dir.name) / "cache"
        else:
            cache_path = Path(self._cache_dir)

        cache_path.mkdir(parents=True, exist_ok=True)
        uv_cache_path = cache_path / "uv"
        uv_cache_path.mkdir(exist_ok=True)

        # Initialize components
        bundle_cache = BundleCache(cache_path, max_bundles=100)
        venv_cache = VenvCache(uv_cache_path)
        image_cache = ImageCache(cache_path, registry=self.registry, max_images=50)
        runtime = DockerRuntime()
        port_allocator = PortAllocator(self.port_range)

        manager = JobManager(
            bundle_cache=bundle_cache,
            venv_cache=venv_cache,
            image_cache=image_cache,
            runtime=runtime,
            port_allocator=port_allocator,
            max_concurrent_jobs=self.max_concurrent_jobs,
        )

        service = WorkerServiceImpl(manager)

        # Find free port if needed
        if self.port == 0:
            sock = socket.socket()
            sock.bind(("", 0))
            self._actual_port = sock.getsockname()[1]
            sock.close()
        else:
            self._actual_port = self.port

        self._dashboard = WorkerDashboard(service, host=self.host, port=self._actual_port)

        # Start server in background
        self._server_task = asyncio.create_task(self._dashboard.run_async())

        # Wait for server to be ready
        await asyncio.sleep(0.5)

        # Create client
        self._http_client = httpx.AsyncClient()
        self._client = WorkerServiceClient(
            address=f"http://{self.host}:{self._actual_port}",
            session=self._http_client,
        )

        print(f"Worker started at http://{self.host}:{self._actual_port}")
        print(f"Dashboard: http://{self.host}:{self._actual_port}/")

        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Stop worker server and cleanup."""
        # Close client
        if self._http_client:
            await self._http_client.aclose()

        # Stop server gracefully
        if self._dashboard:
            await self._dashboard.shutdown()

        if self._server_task:
            try:
                # Give the server time to finish gracefully
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except asyncio.TimeoutError:
                # Force cancel if it takes too long
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass

        # Cleanup temp directory
        if self._temp_dir:
            self._temp_dir.cleanup()

    def _get_workspace_bundle(self) -> str:
        """Create workspace bundle (cached)."""
        if self._workspace_bundle is not None:
            return self._workspace_bundle

        # Create bundle from workspace
        if self._temp_dir:
            bundle_path = Path(self._temp_dir.name) / "workspace.zip"
        else:
            bundle_path = Path(tempfile.mkdtemp()) / "workspace.zip"

        with zipfile.ZipFile(bundle_path, "w") as zf:
            for file in self.workspace.rglob("*"):
                if file.is_file() and not self._should_exclude(file):
                    zf.write(file, file.relative_to(self.workspace))

        self._workspace_bundle = f"file://{bundle_path}"
        return self._workspace_bundle

    def _should_exclude(self, path: Path) -> bool:
        """Check if file should be excluded from bundle."""
        exclude_patterns = [
            ".git/",
            ".venv/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
        ]

        path_str = str(path)
        return any(pattern.rstrip("/") in path_str for pattern in exclude_patterns)

    async def submit(
        self,
        fn: Callable,
        *args,
        job_id: str | None = None,
        timeout_seconds: int | None = None,
        ports: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs,
    ) -> str:
        """Submit a job to the worker.

        Args:
            fn: Callable to execute
            *args: Positional arguments for fn
            job_id: Optional job ID (auto-generated if None)
            timeout_seconds: Job timeout in seconds
            ports: List of port names to allocate
            env_vars: Environment variables to set
            **kwargs: Keyword arguments for fn

        Returns:
            Job ID
        """
        assert self._client is not None, "Worker not started - use 'async with WorkerContext()'"

        # Create entrypoint
        entrypoint = Entrypoint(callable=fn, args=args, kwargs=kwargs)

        # Build environment config with env_vars
        env_config = cluster_pb2.EnvironmentConfig(
            workspace="/app",
            env_vars=env_vars or {},
        )

        # Build request
        request = cluster_pb2.RunJobRequest(
            job_id=job_id or "",
            serialized_entrypoint=cloudpickle.dumps(entrypoint),
            bundle_gcs_path=self._get_workspace_bundle(),
            environment=env_config,
            timeout_seconds=timeout_seconds or 0,
            ports=ports or [],
        )

        response = await self._client.run_job(request)
        return response.job_id

    async def status(self, job_id: str) -> cluster_pb2.JobStatus:
        """Get job status.

        Args:
            job_id: Job ID

        Returns:
            JobStatus proto
        """
        assert self._client is not None, "Worker not started - use 'async with WorkerContext()'"
        return await self._client.get_job_status(cluster_pb2.GetStatusRequest(job_id=job_id))

    async def wait(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> cluster_pb2.JobStatus:
        """Wait for job to complete.

        Args:
            job_id: Job ID
            timeout: Timeout in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If job doesn't complete in time
        """
        import time

        start = time.time()

        while time.time() - start < timeout:
            status = await self.status(job_id)

            if status.state in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
            ):
                return status

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

    async def logs(
        self,
        job_id: str,
        regex: str | None = None,
        tail: int | None = None,
        max_lines: int | None = None,
    ) -> list[str]:
        """Get job logs.

        Args:
            job_id: Job ID
            regex: Optional regex filter
            tail: Get last N lines
            max_lines: Maximum lines to return

        Returns:
            List of log lines
        """
        assert self._client is not None, "Worker not started - use 'async with WorkerContext()'"

        filter_proto = cluster_pb2.FetchLogsFilter()

        if regex:
            filter_proto.regex = regex
        if tail:
            filter_proto.start_line = -tail
        if max_lines:
            filter_proto.max_lines = max_lines

        response = await self._client.fetch_logs(cluster_pb2.FetchLogsRequest(job_id=job_id, filter=filter_proto))

        return [entry.data for entry in response.logs]

    async def kill(
        self,
        job_id: str,
        term_timeout_ms: int = 5000,
    ) -> None:
        """Kill a running job.

        Args:
            job_id: Job ID
            term_timeout_ms: Graceful termination timeout in milliseconds
        """
        assert self._client is not None, "Worker not started - use 'async with WorkerContext()'"

        await self._client.kill_job(
            cluster_pb2.KillJobRequest(
                job_id=job_id,
                term_timeout_ms=term_timeout_ms,
            )
        )

    async def list_jobs(self) -> list[cluster_pb2.JobStatus]:
        """List all jobs.

        Returns:
            List of JobStatus protos
        """
        assert self._client is not None, "Worker not started - use 'async with WorkerContext()'"
        response = await self._client.list_jobs(cluster_pb2.ListJobsRequest())
        return list(response.jobs)

    async def health(self) -> cluster_pb2.HealthResponse:
        """Check worker health.

        Returns:
            HealthResponse proto
        """
        assert self._client is not None, "Worker not started - use 'async with WorkerContext()'"
        return await self._client.health_check(cluster_pb2.Empty())

    @property
    def url(self) -> str:
        """Get worker URL."""
        return f"http://{self.host}:{self._actual_port}"
