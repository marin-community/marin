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

"""Unified worker managing all components and lifecycle.

The Worker class encapsulates all worker components (caches, runtime, job manager,
service, dashboard) and provides a clean interface for lifecycle management.

Example:
    config = WorkerConfig(port=8081)
    worker = Worker(config)
    worker.start()
    try:
        # Use worker
        job_id = worker.submit_job(request)
    finally:
        worker.stop()
"""

import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from fluster import cluster_pb2
from fluster.cluster.worker.builder import ImageCache, ImageProvider, VenvCache
from fluster.cluster.worker.bundle import BundleCache, BundleProvider
from fluster.cluster.worker.dashboard import WorkerDashboard
from fluster.cluster.worker.docker import ContainerRuntime, DockerRuntime
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.service import WorkerServiceImpl


@dataclass
class WorkerConfig:
    """Worker configuration.

    Args:
        host: Host to bind to (default: "127.0.0.1")
        port: Port to bind to (default: 0 for ephemeral)
        cache_dir: Cache directory for bundles and images (default: temp directory)
        registry: Docker registry for images (default: "localhost:5000")
        max_concurrent_jobs: Maximum concurrent jobs (default: 10)
        port_range: Port range for job ports (default: (30000, 40000))
        controller_address: Controller URL for endpoint registration (default: None)
        worker_id: Worker ID (default: None)
    """

    host: str = "127.0.0.1"
    port: int = 0
    cache_dir: Path | None = None
    registry: str = "localhost:5000"
    max_concurrent_jobs: int = 10
    port_range: tuple[int, int] = (30000, 40000)
    controller_address: str | None = None
    worker_id: str | None = None


class Worker:
    """Unified worker managing all components and lifecycle.

    Encapsulates:
    - BundleCache
    - VenvCache
    - ImageCache
    - DockerRuntime
    - PortAllocator
    - JobManager
    - WorkerServiceImpl
    - WorkerDashboard

    Example:
        config = WorkerConfig(port=8081)
        worker = Worker(config)
        worker.start()
        try:
            # Use worker
            job_id = worker.submit_job(request)
        finally:
            worker.stop()
    """

    def __init__(
        self,
        config: WorkerConfig,
        cache_dir: Path | None = None,
        bundle_provider: BundleProvider | None = None,
        image_provider: ImageProvider | None = None,
        container_runtime: ContainerRuntime | None = None,
    ):
        """Initialize worker components.

        Args:
            config: Worker configuration
            cache_dir: Override cache directory from config
            bundle_provider: Optional bundle provider for testing
            image_provider: Optional image provider for testing
            container_runtime: Optional container runtime for testing
        """
        self._config = config

        # Setup cache directory
        if cache_dir:
            self._cache_dir = cache_dir
            self._temp_dir = None
        elif config.cache_dir:
            self._cache_dir = config.cache_dir
            self._temp_dir = None
        else:
            # Create temporary cache
            self._temp_dir = tempfile.TemporaryDirectory(prefix="worker_cache_")
            self._cache_dir = Path(self._temp_dir.name)

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Use overrides if provided, otherwise create defaults
        self._bundle_cache = bundle_provider or BundleCache(self._cache_dir, max_bundles=100)
        self._venv_cache = VenvCache()
        self._image_cache = image_provider or ImageCache(
            self._cache_dir,
            registry=config.registry,
            max_images=50,
        )
        self._runtime = container_runtime or DockerRuntime()
        self._port_allocator = PortAllocator(config.port_range)

        self._job_manager = JobManager(
            bundle_cache=self._bundle_cache,
            venv_cache=self._venv_cache,
            image_cache=self._image_cache,
            runtime=self._runtime,
            port_allocator=self._port_allocator,
            max_concurrent_jobs=config.max_concurrent_jobs,
            controller_address=config.controller_address,
            worker_id=config.worker_id,
        )

        self._service = WorkerServiceImpl(self._job_manager)
        self._dashboard = WorkerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        self._server_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start worker server."""
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        self._server_thread.start()
        time.sleep(1.0)  # Wait for startup

    def stop(self) -> None:
        """Stop worker server and cleanup.

        Note: Cleanup of temp directory is best-effort. If jobs have created
        files in the cache, cleanup may fail. This is acceptable since the
        temp directory will be cleaned up by the OS eventually.
        """
        # Cleanup temp directory (best-effort)
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except OSError:
                # Cleanup may fail if cache has files from running jobs
                # This is acceptable - temp directories are cleaned by OS
                pass
        # Dashboard stops when thread exits (daemon)

    def _run_server(self) -> None:
        """Run worker server (blocking, for thread)."""
        try:
            uvicorn.run(
                self._dashboard._app,
                host=self._config.host,
                port=self._config.port,
                log_level="error",
            )
        except Exception as e:
            print(f"Worker server error: {e}")

    # Delegate key service methods
    def submit_job(
        self,
        request: cluster_pb2.Worker.RunJobRequest,
    ) -> cluster_pb2.Worker.RunJobResponse:
        """Submit a job."""
        return self._service.run_job(request, None)  # type: ignore[arg-type]

    def get_job_status(
        self,
        job_id: str,
    ) -> cluster_pb2.JobStatus:
        """Get job status."""
        request = cluster_pb2.Worker.GetJobStatusRequest(job_id=job_id)
        return self._service.get_job_status(request, None)  # type: ignore[arg-type]

    def list_jobs(self) -> cluster_pb2.Worker.ListJobsResponse:
        """List all jobs."""
        return self._service.list_jobs(cluster_pb2.Worker.ListJobsRequest(), None)  # type: ignore[arg-type]

    # Properties
    @property
    def job_manager(self) -> JobManager:
        """Access to job manager (for advanced usage)."""
        return self._job_manager

    @property
    def url(self) -> str:
        """Worker URL."""
        return f"http://{self._config.host}:{self._config.port}"
