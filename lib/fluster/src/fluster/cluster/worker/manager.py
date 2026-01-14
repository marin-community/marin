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

"""Port allocation and job lifecycle management for worker jobs."""

import base64
import socket
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cloudpickle

from fluster import cluster_pb2
from fluster.cluster.worker.builder import ImageProvider, VenvCache
from fluster.cluster.worker.bundle import BundleProvider
from fluster.cluster.worker.docker import ContainerConfig, ContainerRuntime
from fluster.cluster.worker.worker_types import Job, collect_workdir_size_mb


def _rewrite_address_for_container(address: str) -> str:
    """Rewrite localhost addresses to host.docker.internal for container access.

    Docker containers on Mac/Windows cannot reach host localhost directly.
    Using host.docker.internal works cross-platform when combined with
    --add-host=host.docker.internal:host-gateway on Linux.
    """
    for localhost in ("127.0.0.1", "localhost", "0.0.0.0"):
        if localhost in address:
            return address.replace(localhost, "host.docker.internal")
    return address


class PortAllocator:
    """Allocate ephemeral ports for jobs.

    Tracks allocated ports to avoid conflicts.
    Ports are released when jobs terminate.
    """

    def __init__(self, port_range: tuple[int, int] = (30000, 40000)):
        self._range = port_range
        self._allocated: set[int] = set()
        self._lock = threading.Lock()

    def allocate(self, count: int = 1) -> list[int]:
        """Allocate N unused ports."""
        with self._lock:
            ports = []
            for _ in range(count):
                port = self._find_free_port()
                self._allocated.add(port)
                ports.append(port)
            return ports

    def release(self, ports: list[int]) -> None:
        """Release allocated ports."""
        with self._lock:
            for port in ports:
                self._allocated.discard(port)

    def _find_free_port(self) -> int:
        """Find an unused port in range."""
        for port in range(self._range[0], self._range[1]):
            if port in self._allocated:
                continue
            if self._is_port_free(port):
                return port
        raise RuntimeError("No free ports available")

    def _is_port_free(self, port: int) -> bool:
        """Check if port is free on host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False


class JobManager:
    """Orchestrates job lifecycle.

    Phases:
    1. PENDING: Job submitted, waiting for resources
    2. BUILDING: Downloading bundle, building image
    3. RUNNING: Container executing
    4. SUCCEEDED/FAILED/KILLED: Terminal states

    Cleanup: Removes containers after completion. Logs are retained
    in memory (no rotation - assumes adequate disk space).
    """

    def __init__(
        self,
        bundle_cache: BundleProvider,
        venv_cache: VenvCache,
        image_cache: ImageProvider,
        runtime: ContainerRuntime,
        port_allocator: PortAllocator,
        max_concurrent_jobs: int = 10,
        controller_address: str | None = None,
        worker_id: str | None = None,
    ):
        self._bundle_cache = bundle_cache
        self._venv_cache = venv_cache
        self._image_cache = image_cache
        self._runtime = runtime
        self._port_allocator = port_allocator
        self._max_concurrent_jobs = max_concurrent_jobs
        self._controller_address = controller_address
        self._worker_id = worker_id
        self._semaphore = threading.Semaphore(max_concurrent_jobs)
        # TODO: Jobs are never removed from this dict, causing unbounded memory growth.
        # Need to implement LRU eviction for completed jobs similar to BundleCache/VenvCache.
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit_job(self, request: cluster_pb2.RunJobRequest) -> str:
        """Submit job for execution.

        Returns job_id immediately, execution happens in background.

        Fluster auto-injects system environment variables (these override user-provided values):
        - FLUSTER_JOB_ID: The job's unique identifier
        - FLUSTER_WORKER_ID: ID of the worker running this job
        - FLUSTER_CONTROLLER_ADDRESS: Controller URL for endpoint registration
        - FLUSTER_PORT_<NAME>: Allocated port numbers (e.g., FLUSTER_PORT_HTTP)
        - FRAY_PORT_MAPPING: All port mappings as "name:port,name:port"

        User-provided environment variables (including FLUSTER_NAMESPACE for actors) are
        passed through from the RunJobRequest.environment.env_vars.
        """
        job_id = request.job_id or str(uuid.uuid4())

        # Allocate requested ports
        port_names = list(request.ports)
        allocated_ports = self._port_allocator.allocate(len(port_names)) if port_names else []
        ports = dict(zip(port_names, allocated_ports, strict=True))

        # Create job working directory
        workdir = Path(tempfile.gettempdir()) / "fluster-worker" / "jobs" / job_id
        workdir.mkdir(parents=True, exist_ok=True)

        job = Job(
            job_id=job_id,
            request=request,
            status=cluster_pb2.JOB_STATE_PENDING,
            ports=ports,
            workdir=workdir,
        )

        with self._lock:
            self._jobs[job_id] = job

        # Start execution in background
        job.thread = threading.Thread(target=self._execute_job, args=(job,), daemon=True)
        job.thread.start()

        return job_id

    def _execute_job(self, job: Job) -> None:
        """Execute job through all phases with integrated stats collection."""
        import shutil

        try:
            # Acquire semaphore to limit concurrent jobs
            self._semaphore.acquire()

            # Phase 1: Download bundle
            job.transition_to(cluster_pb2.JOB_STATE_BUILDING, message="downloading bundle")
            job.started_at_ms = int(time.time() * 1000)

            bundle_path = self._bundle_cache.get_bundle(
                job.request.bundle_gcs_path,
                expected_hash=None,
            )

            # Phase 2: Build image
            job.transition_to(cluster_pb2.JOB_STATE_BUILDING, message="building image")
            job.build_started_ms = int(time.time() * 1000)
            env_config = job.request.environment
            extras = list(env_config.extras)

            # Compute deps_hash for caching
            deps_hash = self._venv_cache.compute_deps_hash(bundle_path)

            job.transition_to(cluster_pb2.JOB_STATE_BUILDING, message="populating uv cache")
            job.logs.add("build", "Building Docker image...")

            # Detect host Python version for container compatibility
            # cloudpickle serializes bytecode which is version-specific
            import sys

            py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            base_image = f"python:{py_version}-slim"

            build_result = self._image_cache.build(
                bundle_path=bundle_path,
                base_image=base_image,
                extras=extras,
                job_id=job.job_id,
                deps_hash=deps_hash,
                job_logs=job.logs,
            )

            job.build_finished_ms = int(time.time() * 1000)
            job.build_from_cache = build_result.from_cache
            job.image_tag = build_result.image_tag

            # Phase 3: Create and start container
            job.transition_to(cluster_pb2.JOB_STATE_RUNNING)

            # Deserialize entrypoint
            entrypoint = cloudpickle.loads(job.request.serialized_entrypoint)
            command = self._build_command(entrypoint)

            # Build environment from user-provided vars + EnvironmentConfig
            env = dict(env_config.env_vars)

            # Auto-inject Fluster system variables (these override user-provided values)
            env["FLUSTER_JOB_ID"] = job.job_id

            if self._worker_id:
                env["FLUSTER_WORKER_ID"] = self._worker_id

            if self._controller_address:
                env["FLUSTER_CONTROLLER_ADDRESS"] = _rewrite_address_for_container(self._controller_address)

            # Inject allocated ports
            for name, port in job.ports.items():
                env[f"FLUSTER_PORT_{name.upper()}"] = str(port)

            if job.ports:
                port_mapping = ",".join(f"{name}:{port}" for name, port in job.ports.items())
                env["FRAY_PORT_MAPPING"] = port_mapping

            config = ContainerConfig(
                image=build_result.image_tag,
                command=command,
                env=env,
                resources=job.request.resources if job.request.HasField("resources") else None,
                timeout_seconds=job.request.timeout_seconds or None,
                ports=job.ports,
            )

            # Create and start container
            container_id = self._runtime.create_container(config)
            job.container_id = container_id
            self._runtime.start_container(container_id)

            # Phase 4: Poll loop - check status and collect stats
            timeout = config.timeout_seconds
            start_time = time.time()

            while True:
                # Check if we should stop
                if job.should_stop:
                    job.transition_to(cluster_pb2.JOB_STATE_KILLED)
                    break

                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    self._runtime.kill(container_id, force=True)
                    job.transition_to(
                        cluster_pb2.JOB_STATE_FAILED,
                        error="Timeout exceeded",
                        exit_code=-1,
                    )
                    break

                # Check container status
                status = self._runtime.inspect(container_id)
                if not status.running:
                    # Container has stopped
                    if status.error:
                        job.transition_to(
                            cluster_pb2.JOB_STATE_FAILED,
                            error=status.error,
                            exit_code=status.exit_code or -1,
                        )
                    elif status.exit_code == 0:
                        job.transition_to(cluster_pb2.JOB_STATE_SUCCEEDED, exit_code=0)
                    else:
                        job.transition_to(
                            cluster_pb2.JOB_STATE_FAILED,
                            error=f"Exit code: {status.exit_code}",
                            exit_code=status.exit_code or -1,
                        )
                    break

                # Collect stats
                try:
                    stats = self._runtime.get_stats(container_id)
                    if stats.available:
                        job.current_memory_mb = stats.memory_mb
                        job.current_cpu_percent = stats.cpu_percent
                        job.process_count = stats.process_count
                        if stats.memory_mb > job.peak_memory_mb:
                            job.peak_memory_mb = stats.memory_mb

                    if job.workdir:
                        job.disk_mb = collect_workdir_size_mb(job.workdir)
                except Exception:
                    pass  # Don't fail job on stats collection errors

                # Sleep before next poll
                time.sleep(5.0)

        except Exception as e:
            job.transition_to(cluster_pb2.JOB_STATE_FAILED, error=repr(e))
        finally:
            # Release semaphore
            self._semaphore.release()

            # Cleanup: release ports, remove workdir (keep container for logs)
            if not job.cleanup_done:
                job.cleanup_done = True
                self._port_allocator.release(list(job.ports.values()))
                # Keep container around for log retrieval via docker logs
                # Remove working directory (no longer needed since logs come from Docker)
                if job.workdir and job.workdir.exists():
                    shutil.rmtree(job.workdir, ignore_errors=True)

    def _build_command(self, entrypoint) -> list[str]:
        """Build command to run entrypoint.

        Serializes only the raw callable/args/kwargs tuple rather than the Entrypoint
        dataclass to avoid requiring fluster.cluster.types in the container.
        """
        # Extract raw components to avoid serializing the Entrypoint class itself
        data = (entrypoint.callable, entrypoint.args, entrypoint.kwargs)
        serialized = cloudpickle.dumps(data)
        encoded = base64.b64encode(serialized).decode()
        cmd = (
            "import cloudpickle, base64; "
            f"fn, args, kwargs = cloudpickle.loads(base64.b64decode('{encoded}')); "
            "fn(*args, **kwargs)"
        )
        return ["python", "-c", cmd]

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        """List all jobs."""
        return list(self._jobs.values())

    def kill_job(self, job_id: str, term_timeout_ms: int = 5000) -> bool:
        """Kill a running job by setting should_stop flag.

        The poll loop in _execute_job will handle the actual termination.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        # Check if already in terminal state
        if job.status not in (
            cluster_pb2.JOB_STATE_RUNNING,
            cluster_pb2.JOB_STATE_BUILDING,
            cluster_pb2.JOB_STATE_PENDING,
        ):
            return False

        # Set flag to signal thread to stop
        job.should_stop = True

        # If container exists, try to kill it
        if job.container_id:
            try:
                # Send SIGTERM
                self._runtime.kill(job.container_id, force=False)

                # Wait for graceful shutdown
                timeout_sec = term_timeout_ms / 1000
                start_time = time.time()
                while job.status in (
                    cluster_pb2.JOB_STATE_RUNNING,
                    cluster_pb2.JOB_STATE_BUILDING,
                ):
                    if (time.time() - start_time) > timeout_sec:
                        # Force kill
                        try:
                            self._runtime.kill(job.container_id, force=True)
                        except RuntimeError:
                            pass
                        break
                    time.sleep(0.1)
            except RuntimeError:
                # Container may have already been removed or stopped
                pass

        return True

    def get_logs(self, job_id: str, start_line: int = 0) -> list[cluster_pb2.LogEntry]:
        """Get logs for a job.

        Combines build logs (from job.logs) with container logs (from Docker).

        Args:
            job_id: Job ID
            start_line: Starting line number. If negative, returns last N lines
                       (e.g., start_line=-100 returns last 100 lines for tailing).

        Returns:
            List of log entries sorted by timestamp
        """
        job = self._jobs.get(job_id)
        if not job:
            return []

        logs: list[cluster_pb2.LogEntry] = []

        # Add build logs from job.logs (these have proper timestamps)
        for log_line in job.logs.lines:
            logs.append(log_line.to_proto())

        # Fetch container stdout/stderr from Docker if container exists
        if job.container_id:
            container_logs = self._runtime.get_logs(job.container_id)
            for log_line in container_logs:
                logs.append(log_line.to_proto())

        # Sort by timestamp
        logs.sort(key=lambda x: x.timestamp_ms)

        return logs[start_line:]
