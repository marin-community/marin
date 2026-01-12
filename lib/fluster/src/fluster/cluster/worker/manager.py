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

import asyncio
import base64
import cloudpickle
import socket
import time
import uuid
from pathlib import Path

from fluster import cluster_pb2
from .types import Job
from .bundle import BundleCache
from .builder import VenvCache, ImageBuilder
from .runtime import DockerRuntime, ContainerConfig


class PortAllocator:
    """Allocate ephemeral ports for jobs.

    Tracks allocated ports to avoid conflicts.
    Ports are released when jobs terminate.
    """

    def __init__(self, port_range: tuple[int, int] = (30000, 40000)):
        self._range = port_range
        self._allocated: set[int] = set()
        self._lock = asyncio.Lock()

    async def allocate(self, count: int = 1) -> list[int]:
        """Allocate N unused ports."""
        async with self._lock:
            ports = []
            for _ in range(count):
                port = self._find_free_port()
                self._allocated.add(port)
                ports.append(port)
            return ports

    async def release(self, ports: list[int]) -> None:
        """Release allocated ports."""
        async with self._lock:
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
        bundle_cache: BundleCache,
        venv_cache: VenvCache,
        image_builder: ImageBuilder,
        runtime: DockerRuntime,
        port_allocator: PortAllocator,
        max_concurrent_jobs: int = 10,
    ):
        self._bundle_cache = bundle_cache
        self._venv_cache = venv_cache
        self._image_builder = image_builder
        self._runtime = runtime
        self._port_allocator = port_allocator
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def submit_job(self, request: cluster_pb2.RunJobRequest) -> str:
        """Submit job for execution.

        Returns job_id immediately, execution happens in background.

        Note: FLUSTER_* and FRAY_* environment variables are reserved and will be overwritten:
        - FLUSTER_JOB_ID: Set to the job ID
        - FLUSTER_PORT_<NAME>: Set to allocated port numbers (e.g., FLUSTER_PORT_HTTP)
        - FRAY_PORT_MAPPING: Comma-separated port mappings (e.g., "web:8080,api:8081")
        """
        job_id = request.job_id or str(uuid.uuid4())

        # Allocate requested ports
        port_names = list(request.ports)
        allocated_ports = await self._port_allocator.allocate(len(port_names)) if port_names else []
        ports = dict(zip(port_names, allocated_ports, strict=True))

        # Create job working directory
        workdir = Path(f"/tmp/fluster-worker/jobs/{job_id}")
        workdir.mkdir(parents=True, exist_ok=True)

        job = Job(
            job_id=job_id,
            request=request,
            status=cluster_pb2.JOB_STATE_PENDING,
            ports=ports,
            workdir=workdir,
        )

        async with self._lock:
            self._jobs[job_id] = job

        # Start execution in background
        job.task = asyncio.create_task(self._execute_job(job))

        return job_id

    async def _execute_job(self, job: Job) -> None:
        """Execute job through all phases."""
        async with self._semaphore:
            try:
                # Phase 1: Download bundle
                job.status = cluster_pb2.JOB_STATE_BUILDING
                job.status_message = "downloading bundle"
                job.started_at_ms = int(time.time() * 1000)

                bundle_path = await self._bundle_cache.get_bundle(
                    job.request.bundle_gcs_path,
                    expected_hash=None,
                )

                # Phase 2: Build image
                job.status_message = "building image"
                env_config = job.request.environment
                extras = list(env_config.extras)

                # Compute deps_hash for caching
                deps_hash = self._venv_cache.compute_deps_hash(bundle_path)

                job.status_message = "populating uv cache"
                build_result = await self._image_builder.build(
                    bundle_path=bundle_path,
                    base_image="python:3.11-slim",
                    extras=extras,
                    job_id=job.job_id,
                    deps_hash=deps_hash,
                )

                # Phase 3: Run container
                job.status = cluster_pb2.JOB_STATE_RUNNING
                job.status_message = ""

                # Deserialize entrypoint
                entrypoint = cloudpickle.loads(job.request.serialized_entrypoint)
                command = self._build_command(entrypoint)

                # Build environment
                env = dict(job.request.env_vars)
                env.update(dict(env_config.env_vars))
                env["FLUSTER_JOB_ID"] = job.job_id
                for name, port in job.ports.items():
                    env[f"FLUSTER_PORT_{name.upper()}"] = str(port)

                # Add FRAY_PORT_MAPPING for communicating all port mappings as a single variable
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
                    stdout_file=str(job.workdir / "STDOUT"),
                    stderr_file=str(job.workdir / "STDERR"),
                )

                result = await self._runtime.run(config)
                job.container_id = result.container_id

                # Phase 4: Complete
                job.exit_code = result.exit_code
                job.finished_at_ms = int(time.time() * 1000)
                job.status_message = ""

                if result.error:
                    job.status = cluster_pb2.JOB_STATE_FAILED
                    job.error = result.error
                elif result.exit_code == 0:
                    job.status = cluster_pb2.JOB_STATE_SUCCEEDED
                else:
                    job.status = cluster_pb2.JOB_STATE_FAILED
                    job.error = f"Exit code: {result.exit_code}"

            except asyncio.CancelledError:
                # Task was cancelled (likely from kill_job)
                job.status = cluster_pb2.JOB_STATE_KILLED
                job.status_message = ""
                job.finished_at_ms = int(time.time() * 1000)
                raise
            except Exception as e:
                job.status = cluster_pb2.JOB_STATE_FAILED
                job.status_message = ""
                job.error = repr(e)
                job.finished_at_ms = int(time.time() * 1000)
            finally:
                # Cleanup: release ports, remove container, remove workdir (only if not already cleaned up)
                if not job.cleanup_done:
                    job.cleanup_done = True
                    await self._port_allocator.release(list(job.ports.values()))
                    if job.container_id:
                        try:
                            await self._runtime.remove(job.container_id)
                        except RuntimeError:
                            # Container may have already been removed
                            pass
                    # Remove working directory
                    if job.workdir and job.workdir.exists():
                        import shutil

                        shutil.rmtree(job.workdir, ignore_errors=True)

    def _build_command(self, entrypoint) -> list[str]:
        """Build command to run entrypoint."""
        # Serialize entrypoint and run via python -c
        serialized = cloudpickle.dumps(entrypoint)
        encoded = base64.b64encode(serialized).decode()
        cmd = (
            "import cloudpickle, base64; "
            f"e = cloudpickle.loads(base64.b64decode('{encoded}')); "
            "e.callable(*e.args, **e.kwargs)"
        )
        return ["python", "-c", cmd]

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        """List all jobs."""
        return list(self._jobs.values())

    async def kill_job(self, job_id: str, term_timeout_ms: int = 5000) -> bool:
        """Kill a running job.

        Handles race conditions where job may complete between state check and kill.
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

        # Cancel the task to interrupt execution
        if job.task and not job.task.done():
            job.task.cancel()

        # If container exists, try to kill it
        if job.container_id:
            try:
                # Send SIGTERM
                await self._runtime.kill(job.container_id, force=False)

                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(
                        self._wait_for_termination(job),
                        timeout=term_timeout_ms / 1000,
                    )
                except asyncio.TimeoutError:
                    # Force kill
                    try:
                        await self._runtime.kill(job.container_id, force=True)
                    except RuntimeError:
                        # Container may have already stopped
                        pass
            except RuntimeError:
                # Container may have already been removed or stopped
                pass

        job.status = cluster_pb2.JOB_STATE_KILLED
        job.status_message = ""
        job.finished_at_ms = int(time.time() * 1000)
        return True

    async def _wait_for_termination(self, job: Job) -> None:
        """Wait until job reaches terminal state."""
        while job.status in (cluster_pb2.JOB_STATE_RUNNING, cluster_pb2.JOB_STATE_BUILDING):
            await asyncio.sleep(0.1)

    async def get_logs(self, job_id: str, start_line: int = 0) -> list[cluster_pb2.LogEntry]:
        """Get logs for a job.

        Args:
            job_id: Job ID
            start_line: Starting line number. If negative, returns last N lines
                       (e.g., start_line=-100 returns last 100 lines for tailing).

        Returns:
            List of log entries from STDOUT and STDERR files
        """
        job = self._jobs.get(job_id)
        if not job or not job.workdir:
            return []

        logs = []

        # Read from STDOUT file
        stdout_file = job.workdir / "STDOUT"
        if stdout_file.exists():
            for line in stdout_file.read_text().splitlines():
                if line:  # Skip empty lines
                    logs.append(
                        cluster_pb2.LogEntry(
                            timestamp_ms=int(time.time() * 1000),
                            source="stdout",
                            data=line,
                        )
                    )

        # Read from STDERR file
        stderr_file = job.workdir / "STDERR"
        if stderr_file.exists():
            for line in stderr_file.read_text().splitlines():
                if line:  # Skip empty lines
                    logs.append(
                        cluster_pb2.LogEntry(
                            timestamp_ms=int(time.time() * 1000),
                            source="stderr",
                            data=line,
                        )
                    )

        # Handle negative start_line (tail behavior)
        if start_line < 0:
            return logs[start_line:]
        return logs[start_line:]
