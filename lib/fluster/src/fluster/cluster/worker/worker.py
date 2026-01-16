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

"""Unified worker managing all components and lifecycle."""

import base64
import logging
import shutil
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import uvicorn

from fluster.rpc import cluster_pb2
from fluster.time_utils import ExponentialBackoff
from fluster.rpc.cluster_connect import ControllerServiceClientSync
from fluster.rpc.errors import format_exception_with_traceback
from fluster.cluster.worker.builder import ImageCache, ImageProvider, VenvCache
from fluster.cluster.worker.bundle_cache import BundleCache, BundleProvider
from fluster.cluster.worker.dashboard import WorkerDashboard
from fluster.cluster.worker.docker import ContainerConfig, ContainerRuntime, DockerRuntime
from fluster.cluster.worker.env_probe import DefaultEnvironmentProvider, EnvironmentProvider
from fluster.cluster.worker.service import WorkerServiceImpl
from fluster.cluster.worker.worker_types import Job, collect_workdir_size_mb

logger = logging.getLogger(__name__)


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
    """Allocate ephemeral ports for jobs."""

    def __init__(self, port_range: tuple[int, int] = (30000, 40000)):
        self._range = port_range
        self._allocated: set[int] = set()
        self._lock = threading.Lock()

    def allocate(self, count: int = 1) -> list[int]:
        with self._lock:
            ports = []
            for _ in range(count):
                port = self._find_free_port()
                self._allocated.add(port)
                ports.append(port)
            return ports

    def release(self, ports: list[int]) -> None:
        with self._lock:
            for port in ports:
                self._allocated.discard(port)

    def _find_free_port(self) -> int:
        for port in range(self._range[0], self._range[1]):
            if port in self._allocated:
                continue
            if self._is_port_free(port):
                return port
        logger.warning("Port allocation exhausted: no free ports in range %d-%d", self._range[0], self._range[1])
        raise RuntimeError("No free ports available")

    def _is_port_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False


@dataclass
class WorkerConfig:
    """Worker configuration."""

    host: str = "127.0.0.1"
    port: int = 0
    cache_dir: Path | None = None
    registry: str = "localhost:5000"
    max_concurrent_jobs: int = 10
    port_range: tuple[int, int] = (30000, 40000)
    controller_address: str | None = None
    worker_id: str | None = None
    poll_interval_seconds: float = 5.0


class Worker:
    """Unified worker managing all components and lifecycle."""

    def __init__(
        self,
        config: WorkerConfig,
        cache_dir: Path | None = None,
        bundle_provider: BundleProvider | None = None,
        image_provider: ImageProvider | None = None,
        container_runtime: ContainerRuntime | None = None,
        environment_provider: EnvironmentProvider | None = None,
    ):
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
        self._environment_provider = environment_provider or DefaultEnvironmentProvider()
        self._port_allocator = PortAllocator(config.port_range)

        # Job state
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(config.max_concurrent_jobs)

        self._service = WorkerServiceImpl(self)
        self._dashboard = WorkerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        self._server_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

        # Heartbeat/registration thread
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_heartbeat = threading.Event()
        self._worker_id: str | None = config.worker_id
        self._controller_client: ControllerServiceClientSync | None = None

    def start(self) -> None:
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        self._server_thread.start()

        # Wait for server startup with exponential backoff
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=5.0,
        )

        # Create controller client synchronously (before any jobs can be dispatched)
        if self._config.controller_address:
            self._controller_client = ControllerServiceClientSync(
                address=self._config.controller_address,
                timeout_ms=5000,
            )

        # Start heartbeat loop if controller configured
        if self._config.controller_address:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
            )
            self._heartbeat_thread.start()

    def stop(self) -> None:
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)

        # Kill and remove all containers
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.container_id:
                try:
                    self._runtime.kill(job.container_id, force=True)
                except RuntimeError:
                    pass
                try:
                    self._runtime.remove(job.container_id)
                except RuntimeError:
                    pass

        # Cleanup temp directory (best-effort)
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except OSError:
                pass

    def _run_server(self) -> None:
        try:
            config = uvicorn.Config(
                self._dashboard._app,
                host=self._config.host,
                port=self._config.port,
                log_level="error",
            )
            self._server = uvicorn.Server(config)
            self._server.run()
        except Exception as e:
            print(f"Worker server error: {e}")

    def _heartbeat_loop(self) -> None:
        metadata = self._environment_provider.probe()
        resources = self._environment_provider.build_resource_spec(metadata)

        # Generate worker ID if not provided
        if not self._worker_id:
            self._worker_id = f"worker-{uuid.uuid4().hex[:8]}"

        # Determine the address to advertise to the controller.
        # If host is 0.0.0.0 (bind to all interfaces), use the probed IP for external access.
        # Otherwise, use the configured host.
        address_host = self._config.host
        if address_host == "0.0.0.0":
            address_host = metadata.ip_address

        # Build registration request
        request = cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id=self._worker_id,
            address=f"{address_host}:{self._config.port}",
            resources=resources,
            metadata=metadata,
        )

        # Controller client is created in start() before this thread starts
        assert self._controller_client is not None

        # Retry registration until successful
        attempt = 0
        while not self._stop_heartbeat.is_set():
            attempt += 1
            try:
                logger.debug("Registration attempt %d for worker %s", attempt, self._worker_id)
                response = self._controller_client.register_worker(request)
                if response.accepted:
                    logger.info("Registered with controller: %s", self._worker_id)
                    break
            except Exception as e:
                logger.warning("Registration attempt %d failed, retrying in 5s: %s", attempt, e)
            self._stop_heartbeat.wait(5.0)

        # Periodic heartbeat (re-registration)
        heartbeat_interval = 10.0  # seconds
        while not self._stop_heartbeat.is_set():
            self._stop_heartbeat.wait(heartbeat_interval)
            if self._stop_heartbeat.is_set():
                break
            try:
                self._controller_client.register_worker(request)
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    def _report_job_state(self, job: "Job") -> None:
        if not self._controller_client or not self._worker_id:
            return

        try:
            request = cluster_pb2.Controller.ReportJobStateRequest(
                worker_id=self._worker_id,
                job_id=job.job_id,
                attempt_id=job.attempt_id,
                state=job.status,
                exit_code=job.exit_code or 0,
                error=job.error or "",
                finished_at_ms=job.finished_at_ms or 0,
            )
            self._controller_client.report_job_state(request)
        except Exception as e:
            logger.warning(f"Failed to report job state to controller: {e}")

    # Job management methods

    def submit_job(self, request: cluster_pb2.Worker.RunJobRequest) -> str:
        job_id = request.job_id or str(uuid.uuid4())
        attempt_id = request.attempt_id

        # Allocate requested ports
        port_names = list(request.ports)
        allocated_ports = self._port_allocator.allocate(len(port_names)) if port_names else []
        ports = dict(zip(port_names, allocated_ports, strict=True))

        # Create job working directory with attempt isolation
        # Use safe path component for hierarchical job IDs (e.g., "my-exp/worker-0" -> "my-exp__worker-0")
        safe_job_id = job_id.replace("/", "__")
        workdir = Path(tempfile.gettempdir()) / "fluster-worker" / "jobs" / f"{safe_job_id}_attempt_{attempt_id}"
        workdir.mkdir(parents=True, exist_ok=True)

        job = Job(
            job_id=job_id,
            attempt_id=attempt_id,
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
        import sys

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
            command = self._build_command(entrypoint, job.ports)

            # Build environment from user-provided vars + EnvironmentConfig
            env = dict(env_config.env_vars)

            # Auto-inject Fluster system variables (these override user-provided values)
            env["FLUSTER_JOB_ID"] = job.job_id
            env["FLUSTER_ATTEMPT_ID"] = str(job.attempt_id)

            if self._config.worker_id:
                env["FLUSTER_WORKER_ID"] = self._config.worker_id

            if self._config.controller_address:
                # Only rewrite localhost addresses for Docker containers
                if isinstance(self._runtime, DockerRuntime):
                    env["FLUSTER_CONTROLLER_ADDRESS"] = _rewrite_address_for_container(self._config.controller_address)
                else:
                    env["FLUSTER_CONTROLLER_ADDRESS"] = self._config.controller_address

            # Inject bundle path for sub-job inheritance
            if job.request.bundle_gcs_path:
                env["FLUSTER_BUNDLE_GCS_PATH"] = job.request.bundle_gcs_path

            # Inject bind host - 0.0.0.0 for Docker (so port mapping works), 127.0.0.1 otherwise
            if isinstance(self._runtime, DockerRuntime):
                env["FLUSTER_BIND_HOST"] = "0.0.0.0"
            else:
                env["FLUSTER_BIND_HOST"] = "127.0.0.1"

            # Inject allocated ports
            for name, port in job.ports.items():
                env[f"FLUSTER_PORT_{name.upper()}"] = str(port)

            config = ContainerConfig(
                image=build_result.image_tag,
                command=command,
                env=env,
                resources=job.request.resources if job.request.HasField("resources") else None,
                timeout_seconds=job.request.timeout_seconds or None,
                ports=job.ports,
                mounts=[(str(job.workdir), "/workdir", "rw")],
            )

            # Create and start container with retry on port binding failures
            container_id = None
            max_port_retries = 3
            for attempt in range(max_port_retries):
                try:
                    container_id = self._runtime.create_container(config)
                    job.container_id = container_id
                    self._runtime.start_container(container_id)
                    break
                except RuntimeError as e:
                    if "address already in use" in str(e) and attempt < max_port_retries - 1:
                        logger.warning(
                            "Port conflict for job %s, retrying with new ports (attempt %d)", job.job_id, attempt + 2
                        )
                        job.logs.add("build", f"Port conflict, retrying with new ports (attempt {attempt + 2})")
                        # Release current ports and allocate new ones
                        self._port_allocator.release(list(job.ports.values()))
                        port_names = list(job.ports.keys())
                        new_ports = self._port_allocator.allocate(len(port_names))
                        job.ports = dict(zip(port_names, new_ports, strict=True))

                        # Update config with new ports
                        config.ports = job.ports
                        for name, port in job.ports.items():
                            config.env[f"FLUSTER_PORT_{name.upper()}"] = str(port)

                        # Try to remove failed container if it was created
                        if container_id:
                            try:
                                self._runtime.remove(container_id)
                            except RuntimeError:
                                pass
                            container_id = None
                    else:
                        raise

            # container_id is guaranteed to be set here (loop breaks on success, raises on failure)
            assert container_id is not None

            # Report RUNNING state to controller so endpoints become visible
            self._report_job_state(job)

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
                    # Read result file only if container succeeded
                    if status.exit_code == 0 and job.workdir:
                        result_path = job.workdir / "_result.pkl"
                        if result_path.exists():
                            try:
                                job.result = result_path.read_bytes()
                            except Exception as e:
                                job.logs.add("error", f"Failed to read result file: {e}")

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
                time.sleep(self._config.poll_interval_seconds)

        except Exception as e:
            error_msg = format_exception_with_traceback(e)
            job.logs.add("error", f"Job failed:\n{error_msg}")
            job.transition_to(cluster_pb2.JOB_STATE_FAILED, error=error_msg)
        finally:
            # Release semaphore
            self._semaphore.release()

            # Report job state to controller if in terminal state
            if job.status in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
            ):
                self._report_job_state(job)

            # Cleanup: release ports, remove workdir (keep container for logs)
            if not job.cleanup_done:
                job.cleanup_done = True
                self._port_allocator.release(list(job.ports.values()))
                # Keep container around for log retrieval via docker logs
                # Remove working directory (no longer needed since logs come from Docker)
                if job.workdir and job.workdir.exists():
                    shutil.rmtree(job.workdir, ignore_errors=True)

    def _build_command(self, entrypoint, ports: dict[str, int]) -> list[str]:
        del ports  # Ports are passed via FLUSTER_PORT_* env vars, not serialized

        data = (entrypoint.callable, entrypoint.args, entrypoint.kwargs)
        serialized = cloudpickle.dumps(data)
        encoded = base64.b64encode(serialized).decode()

        thunk = f"""
import cloudpickle
import base64

fn, args, kwargs = cloudpickle.loads(base64.b64decode('{encoded}'))
result = fn(*args, **kwargs)

with open('/workdir/_result.pkl', 'wb') as f:
    f.write(cloudpickle.dumps(result))
"""
        return ["python", "-c", thunk]

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def kill_job(self, job_id: str, term_timeout_ms: int = 5000) -> bool:
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

                # Wait for graceful shutdown with exponential backoff
                running_states = (cluster_pb2.JOB_STATE_RUNNING, cluster_pb2.JOB_STATE_BUILDING)
                stopped = ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
                    lambda: job.status not in running_states,
                    timeout=term_timeout_ms / 1000,
                )

                # Force kill if graceful shutdown timed out
                if not stopped:
                    try:
                        self._runtime.kill(job.container_id, force=True)
                    except RuntimeError:
                        pass
            except RuntimeError:
                # Container may have already been removed or stopped
                pass

        return True

    def get_logs(self, job_id: str, start_line: int = 0) -> list[cluster_pb2.Worker.LogEntry]:
        job = self._jobs.get(job_id)
        if not job:
            return []

        logs: list[cluster_pb2.Worker.LogEntry] = []

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

    @property
    def url(self) -> str:
        return f"http://{self._config.host}:{self._config.port}"
