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

from fray.cluster.base import Cluster, CpuConfig, EnvironmentConfig, JobId, JobInfo, JobRequest, TaskStatus
from fray.fn_thunk import create_thunk_entrypoint
from fray.isolated_env import TemporaryVenv

logger = logging.getLogger(__name__)


class LocalCluster(Cluster):
    """Local cluster implementation using subprocess.

    Runs jobs as local subprocesses in isolated virtual environments. Useful for
    development, testing, and single-machine workloads. Does not support distributed
    execution or GPU/TPU resources.
    """

    def __init__(self):
        """Initialize local cluster."""
        self._jobs: dict[JobId, _LocalJob] = {}

    def _get_cluster_spec(self) -> str:
        return "local"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        """Terminate remaining jobs."""
        logger.info(f"Shutting down cluster with {len(self._jobs)} jobs")
        for job_id, job in self._jobs.items():
            try:
                job.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up job {job_id}: {e}")
        self._jobs.clear()

    def launch(self, request: JobRequest) -> JobId:
        """Launch a job as a local subprocess using TemporaryVenv."""
        if not isinstance(request.resources.device, CpuConfig):
            raise ValueError("LocalCluster only supports CPU resources")

        if request.environment is None:
            raise ValueError("LocalCluster requires environment configuration")

        job_id = JobId(str(uuid.uuid4()))
        replica_count = request.resources.replicas

        logger.info(f"Launching job {job_id} with {replica_count} replica(s)")

        # Create venv with workspace (one-time setup)
        venv = TemporaryVenv(
            workspace=request.environment.workspace,
            pip_install_args=list(request.environment.pip_packages),
            extras=list(request.environment.extras),
            prefix=f"fray-job-{job_id}-",
        )
        venv.__enter__()  # Creates venv, copies workspace, installs packages

        # Build command (simple binary execution, no uv run wrapper)
        cmd = self._build_command(request, venv)
        logger.info(f"Command: {' '.join(cmd)}")

        # Launch all replicas using shared venv
        processes = []
        try:
            for replica_id in range(replica_count):
                replica_env = self._build_replica_env(venv, request.environment, replica_id, replica_count)

                process = venv.run_async(
                    cmd,
                    env=replica_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                processes.append(process)
                logger.info(f"Replica {replica_id}/{replica_count} started (PID {process.pid})")

        except Exception as e:
            venv.__exit__(None, None, None)
            raise RuntimeError(f"Failed to launch job: {e}") from e

        local_job = _LocalJob(
            job_id=job_id,
            request=request,
            processes=processes,
            venv=venv,
            start_time=time.time(),
        )
        self._jobs[job_id] = local_job
        local_job.start_log_thread()

        return job_id

    def monitor(self, job_id: JobId) -> Iterator[str]:
        """Monitor job output from all replicas."""
        job = self._get_job(job_id)
        while True:
            try:
                line = job.log_queue.get(timeout=1)
                yield line
            except Empty:
                if all(process.poll() is not None for process in job.processes):
                    logger.info("All processes for job %s have finished.", job_id)
                    # Drain remaining logs
                    while not job.log_queue.empty():
                        yield job.log_queue.get_nowait()
                    break

    def poll(self, job_id: JobId) -> JobInfo:
        return self._get_job(job_id).get_info()

    def terminate(self, job_id: JobId) -> None:
        """Terminate all replicas of a job."""
        self._get_job(job_id).cleanup()

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

    def _build_command(self, request: JobRequest, venv) -> list[str]:
        entrypoint = request.entrypoint

        if entrypoint.callable is not None:
            # Ensure tmp directory exists for thunk files
            tmp_dir = Path(venv.workspace_path) / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            entrypoint = create_thunk_entrypoint(
                entrypoint.callable,
                prefix=f"{venv.workspace_path}/tmp/{request.name}",
                function_args=entrypoint.function_args,
            )

        assert entrypoint.binary is not None, "Entrypoint requires binary"
        return [entrypoint.binary, *entrypoint.args]

    def _build_replica_env(
        self, venv, env_config: EnvironmentConfig, replica_id: int, replica_count: int
    ) -> dict[str, str]:
        env = venv.get_env()
        env.update(
            {
                "FRAY_CLUSTER_SPEC": self._get_cluster_spec(),
                "FRAY_REPLICA_ID": str(replica_id),
                "FRAY_REPLICA_COUNT": str(replica_count),
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.update(env_config.env_vars)
        return env


class _LocalJob:
    """Internal job tracking for LocalCluster."""

    def __init__(
        self,
        job_id: JobId,
        request: JobRequest,
        processes: list[subprocess.Popen],
        venv,
        start_time: float,
    ):
        self.job_id = job_id
        self.request = request
        self.processes = processes
        self.venv = venv
        self.replica_count = len(processes)
        self.start_time = start_time
        self.log_queue: Queue[str] = Queue()
        self._log_threads: list[Thread] = []

    def cleanup(self):
        """Clean up venv and all tracked processes."""
        logger.info(f"Cleaning up job {self.job_id}")
        if self.venv is not None:
            self.venv.__exit__(None, None, None)
            self.venv = None

    def start_log_thread(self):
        """Start background threads to collect logs from all replicas."""
        thread_logger = logging.getLogger(__name__ + ".log_thread")

        def collect_logs_for_replica(replica_id: int, process: subprocess.Popen):
            thread_logger.info(f"Starting log collection for job {self.job_id} replica {replica_id}")
            try:
                for line in iter(process.stdout.readline, ""):
                    line = line.rstrip()
                    prefixed_line = f"[replica-{replica_id}] {line}"
                    thread_logger.info(f"{self.job_id}: {prefixed_line}")
                    self.log_queue.put(prefixed_line)
            except Exception as e:
                thread_logger.error(f"Error reading logs for job {self.job_id} replica {replica_id}: {e}")

        # Start one thread per replica
        for replica_id, process in enumerate(self.processes):
            thread = Thread(target=collect_logs_for_replica, args=(replica_id, process), daemon=True)
            thread.start()
            self._log_threads.append(thread)

    def get_info(self) -> JobInfo:
        returncodes = [process.poll() for process in self.processes]
        task_status = []
        for rc in returncodes:
            if rc is None:
                task_status.append(TaskStatus(status="running", error_message=""))
            elif rc != 0:
                task_status.append(TaskStatus(status="failed", error_message=f"Replica failed with exit code {rc}"))
            else:
                task_status.append(TaskStatus(status="succeeded", error_message=""))

        if any(ts.status == "running" for ts in task_status):
            # At least one replica still running
            status = "running"
            error_message = None
        elif any(ts.status == "failed" for ts in task_status):
            # At least one replica failed
            status = "failed"
            error_message = "One or more replicas failed"
        else:
            status = "succeeded"
            error_message = None

        return JobInfo(
            job_id=self.job_id,
            status=status,
            tasks=task_status,
            name=self.request.name,
            error_message=error_message,
        )
