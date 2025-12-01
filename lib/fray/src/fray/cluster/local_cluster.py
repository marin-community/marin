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
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from fray.cluster.base import Cluster, CpuConfig, EnvironmentConfig, JobId, JobInfo, JobRequest, JobStatus, TaskStatus

logger = logging.getLogger(__name__)


class LocalCluster(Cluster):
    """Local cluster implementation using subprocess.

    Runs jobs as local subprocesses. Useful for development, testing,
    and single-machine workloads. Does not support distributed execution
    or GPU/TPU resources. Jobs are run in isolated temporary directories using `uv run`.
    """

    def __init__(self, working_dir: Path | None = None):
        """Initialize local cluster.

        Args:
            working_dir: Directory to run jobs in (default: current directory)
        """
        self._working_dir = working_dir or Path(tempfile.mkdtemp(prefix="fray-local-cluster-"))
        self._jobs: dict[JobId, _LocalJob] = {}

    def _get_cluster_spec(self) -> str:
        return "local"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        """Terminate any remaining jobs forcefully."""
        logger.info(f"Shutting down cluster with {len(self._jobs)} jobs")
        for job_id, job in self._jobs.items():
            for replica_id, process in enumerate(job.processes):
                if process.poll() is None:
                    logger.info(f"Killing job {job_id} replica {replica_id} (PID {process.pid})")
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5)
                    except (ProcessLookupError, OSError):
                        # Process group already dead
                        pass
                else:
                    logger.info(f"Job {job_id} replica {replica_id} already exited with code {process.returncode}")

        self._jobs.clear()

    def launch(self, request: JobRequest) -> JobId:
        """Launch a job as a local subprocess."""

        if not isinstance(request.resources.device, CpuConfig):
            raise ValueError("LocalCluster only supports CPU resources")

        if request.environment is None:
            raise ValueError("LocalCluster requires environment configuration")

        job_id = JobId(str(uuid.uuid4()))

        cmd = self._build_command(request)
        env = self._environment_dict(request.environment)
        job_dir = tempfile.mkdtemp(dir=self._working_dir, suffix=f"fray-job-{job_id}")

        if request.environment.workspace:
            logger.info(f"Copying workspace {request.environment.workspace} to {job_dir}")

            # Get list of files from git (tracked + untracked, respecting gitignore)
            try:
                # --cached: tracked files
                # --others: untracked files
                # --exclude-standard: respect .gitignore
                git_files = subprocess.check_output(
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                    cwd=request.environment.workspace,
                    text=True,
                ).splitlines()

                for rel_path in git_files:
                    src_path = Path(request.environment.workspace) / rel_path
                    dst_path = Path(job_dir) / rel_path

                    if src_path.is_file():
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_path, dst_path)

            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to query git files, falling back to simple copy: {e}")
                shutil.copytree(request.environment.workspace, job_dir, dirs_exist_ok=True)

        replica_count = request.resources.replicas
        logger.info(f"Launching job {job_id} with {replica_count} replica(s) in {job_dir}")
        logger.info(f"Command: {' '.join(cmd)}")

        processes = []
        try:
            for replica_id in range(replica_count):
                # Add replica-specific environment variables
                replica_env = env.copy()
                replica_env["FRAY_REPLICA_ID"] = str(replica_id)
                replica_env["FRAY_REPLICA_COUNT"] = str(replica_count)

                process = subprocess.Popen(
                    cmd,
                    cwd=job_dir,
                    env=replica_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,  # Create new process group
                )
                processes.append(process)
                logger.info(f"Replica {replica_id}/{replica_count} started with PID: {process.pid}")
        except Exception as e:
            # Clean up any started processes
            for p in processes:
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
            raise RuntimeError(f"Failed to launch job: {e}") from e

        # Track job
        local_job = _LocalJob(
            job_id=job_id,
            request=request,
            processes=processes,
            start_time=time.time(),
            job_dir=job_dir,
        )
        self._jobs[job_id] = local_job

        # Start log collection threads
        local_job.start_log_thread()
        logger.info(f"Log threads started for job {job_id}")

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
        job = self._get_job(job_id)
        for replica_id, process in enumerate(job.processes):
            if process.poll() is None:
                logger.info(f"Terminating job {job_id} replica {replica_id} (PID {process.pid})")
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Replica {replica_id} did not respond to SIGTERM, sending SIGKILL")
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    # Process group already dead
                    logger.info(f"Replica {replica_id} process group already dead")
                    pass
            else:
                logger.info(f"Replica {replica_id} already exited with code {process.returncode}")

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
            from fray.fn_thunk import create_thunk_entrypoint

            entrypoint = create_thunk_entrypoint(
                entrypoint.callable, prefix=f"/tmp/{request.name}", function_args=entrypoint.function_args
            )

        assert entrypoint.binary is not None, "Command-line entrypoint requires binary"

        if request.environment and request.environment.workspace:
            cmd = ["uv", "run"]

            # set python version to current Python
            import sys

            py_version = sys.version_info
            cmd.append(f"--python={py_version.major}.{py_version.minor}")
            cmd.append("--with=setuptools")

            for pkg in request.environment.pip_packages:
                cmd.append(f"--with={pkg}")

            for group in request.environment.extras:
                cmd.append(f"--with={group}")

            cmd.append(entrypoint.binary)
            cmd.extend(entrypoint.args)
            return cmd
        else:
            raise NotImplementedError("Docker execution not yet supported in LocalCluster")

    def _environment_dict(self, env_config: EnvironmentConfig) -> dict[str, str]:
        env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "USER": os.environ["USER"],
            "FRAY_CLUSTER_SPEC": self._get_cluster_spec(),
            "PYTHONUNBUFFERED": "1",
        }
        env.update(env_config.env_vars)
        return env


class _LocalJob:
    """Internal job tracking for LocalCluster."""

    def __init__(
        self,
        job_id: JobId,
        request: JobRequest,
        processes: list[subprocess.Popen],
        start_time: float,
        job_dir: str,
    ):
        self.job_id = job_id
        self.request = request
        self.processes = processes
        self.replica_count = len(processes)
        self.start_time = start_time
        self.job_dir = job_dir
        self.log_queue: Queue[str] = Queue()
        self._log_threads: list[Thread] = []

    def start_log_thread(self):
        """Start background threads to collect logs from all replicas."""
        thread_logger = logging.getLogger(__name__ + ".log_thread")

        def collect_logs_for_replica(replica_id: int, process: subprocess.Popen):
            thread_logger.info(f"Starting log collection for job {self.job_id} replica {replica_id}")
            try:
                # Read lines until EOF (process exits and pipe closes)
                for line in iter(process.stdout.readline, ""):
                    line = line.rstrip()
                    if line:  # Only log non-empty lines
                        prefixed_line = f"[replica-{replica_id}] {line}"
                        thread_logger.info(f"{self.job_id}: {prefixed_line}")
                        self.log_queue.put(prefixed_line)
            except Exception as e:
                thread_logger.error(f"Error reading logs for job {self.job_id} replica {replica_id}: {e}")
            finally:
                thread_logger.info(f"Log collection ended for job {self.job_id} replica {replica_id}")
                if process.poll() is not None:
                    thread_logger.info(f"Replica {replica_id} ended with returncode: {process.returncode}")

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
            status = JobStatus("running")
            error_message = None
        elif any(ts.status == "failed" for ts in task_status):
            # At least one replica failed
            status = JobStatus("failed")
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
