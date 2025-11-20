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

"""Local subprocess-based cluster implementation for development and testing."""

import logging
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from fray.cluster.base import Cluster, CpuConfig, EnvironmentConfig, JobId, JobInfo, JobRequest, JobStatus

logger = logging.getLogger(__name__)


class LocalCluster(Cluster):
    """Local cluster implementation using subprocess.

    Runs jobs as local subprocesses. Useful for development, testing,
    and single-machine workloads. Does not support distributed execution
    or GPU/TPU resources.

    Jobs are executed using `uv run` for workspace-based execution,
    which provides isolated dependency management similar to Ray's
    runtime environments.
    """

    def __init__(self, working_dir: Path | None = None):
        """Initialize local cluster.

        Args:
            working_dir: Directory to run jobs in (default: current directory)
        """
        self._working_dir = working_dir or Path.cwd()
        self._jobs: dict[JobId, _LocalJob] = {}

    def launch(self, request: JobRequest) -> JobId:
        """Launch a job as a local subprocess."""

        # Validate request
        if not isinstance(request.resources.device, CpuConfig):
            raise ValueError("LocalCluster only supports CPU resources")

        if request.environment is None:
            raise ValueError("LocalCluster requires environment configuration")

        job_id = JobId(str(uuid.uuid4()))

        # Build command
        cmd = self._build_command(request)
        env = self._build_environment(request.environment)

        logger.info(f"[LAUNCH] Launching job {job_id} with command: {' '.join(cmd)}")

        # Start process
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self._working_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            logger.info(f"[LAUNCH] Process started with PID: {process.pid}")
        except Exception as e:
            raise RuntimeError(f"Failed to launch job: {e}") from e

        # Track job
        local_job = _LocalJob(
            job_id=job_id,
            request=request,
            process=process,
            start_time=time.time(),
        )
        self._jobs[job_id] = local_job

        # Start log collection thread
        local_job.start_log_thread()
        logger.info(f"[LAUNCH] Log thread started for job {job_id}")

        return job_id

    def monitor(self, job_id: JobId) -> Iterator[str]:
        """Stream logs from job's stdout/stderr."""
        job = self._get_job(job_id)

        # Yield buffered logs
        while True:
            try:
                line = job.log_queue.get(timeout=0.1)
                yield line
            except Empty:
                # Check if process is done
                if job.process.poll() is not None:
                    # Drain remaining logs
                    while not job.log_queue.empty():
                        yield job.log_queue.get_nowait()
                    break

    def poll(self, job_id: JobId) -> JobInfo:
        return self._get_job(job_id).get_info()

    def terminate(self, job_id: JobId) -> None:
        job = self._get_job(job_id)
        if job.process.poll() is None:
            job.process.terminate()
            try:
                job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.process.kill()

    def list_jobs(self) -> list[JobInfo]:
        return [job.get_info() for job in self._jobs.values()]

    @contextmanager
    def connect(self) -> Iterator[None]:
        """No-op connection for local cluster."""
        yield

    def _get_job(self, job_id: JobId) -> "_LocalJob":
        if job_id not in self._jobs:
            raise KeyError(f"Job {job_id} not found")
        return self._jobs[job_id]

    def _build_command(self, request: JobRequest) -> list[str]:
        entrypoint = request.entrypoint

        if entrypoint.callable is not None:
            # Callable entrypoint: use thunk helper to convert to binary entrypoint
            from fray.fn_thunk import create_thunk_entrypoint

            # Convert callable to binary-based entrypoint
            entrypoint = create_thunk_entrypoint(entrypoint.callable, prefix=f"/tmp/{request.name}")

        # Binary entrypoint
        assert entrypoint.binary is not None, "Command-line entrypoint requires binary"

        if request.environment and request.environment.workspace:
            return ["uv", "run", entrypoint.binary, *entrypoint.args]
        else:
            raise NotImplementedError("Docker execution not yet supported in LocalCluster")

    def _build_environment(self, env_config: EnvironmentConfig) -> dict[str, str]:
        import os

        env = os.environ.copy()
        env.update(env_config.env_vars)
        return env


class _LocalJob:
    """Internal job tracking for LocalCluster."""

    def __init__(
        self,
        job_id: JobId,
        request: JobRequest,
        process: subprocess.Popen,
        start_time: float,
    ):
        self.job_id = job_id
        self.request = request
        self.process = process
        self.start_time = start_time
        self.log_queue: Queue[str] = Queue()
        self._log_thread: Thread | None = None

    def start_log_thread(self):
        """Start background thread to collect logs."""

        def collect_logs():
            logger.info(f"[LOG_THREAD] Starting log collection for job {self.job_id}")
            if self.process.stdout:
                for line in self.process.stdout:
                    logger.debug(f"[LOG_THREAD] Job {self.job_id} output: {line.rstrip()}")
                    self.log_queue.put(line.rstrip())
            logger.info(f"[LOG_THREAD] Log collection ended for job {self.job_id}")

        self._log_thread = Thread(target=collect_logs, daemon=True)
        self._log_thread.start()

    def get_info(self) -> JobInfo:
        """Get current job info.

        Returns:
            Job information with current status
        """
        returncode = self.process.poll()
        logger.info(f"[GET_INFO] Job {self.job_id} returncode: {returncode}, pid: {self.process.pid}")

        if returncode is None:
            status: JobStatus = "running"
            end_time = None
            error_message = None
        elif returncode == 0:
            status = "succeeded"
            end_time = time.time()
            error_message = None
        else:
            status = "failed"
            end_time = time.time()
            error_message = f"Process exited with code {returncode}"

        logger.info(f"[GET_INFO] Job {self.job_id} determined status: {status}")
        return JobInfo(
            job_id=self.job_id,
            status=status,
            name=self.request.name,
            start_time=self.start_time,
            end_time=end_time,
            error_message=error_message,
        )
