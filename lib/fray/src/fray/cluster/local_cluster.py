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
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Literal

from fray.cluster.base import Cluster, EnvironmentConfig, JobId, JobInfo, JobRequest, TaskStatus
from fray.isolated_env import TemporaryVenv
from fray.job.context import SyncContext, fray_default_job_ctx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalClusterConfig:
    isolated_env_mode: Literal["shared", "per_job"] | None = None
    """How to run jobs:

    - None: run jobs in the local process (no isolated venv).
    - "shared": run jobs in a shared isolated venv (cached per env config).
    - "per_job": run jobs in a fresh isolated venv per job.
    """

    def __post_init__(self) -> None:
        if self.isolated_env_mode not in {None, "shared", "per_job"}:
            raise ValueError(f"Invalid isolated_env_mode={self.isolated_env_mode!r}; expected None|'shared'|'per_job'")


class FakeProcess(Thread):
    """Thread wrapper that mimics subprocess.Popen for non-isolated execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exception: BaseException | None = None

    def run(self):
        try:
            with fray_default_job_ctx(SyncContext()):
                super().run()
        except BaseException as e:
            self._exception = e
            raise

    def poll(self) -> int | None:
        if self.is_alive():
            return None
        # Return non-zero exit code if thread raised an exception
        return 1 if self._exception is not None else 0

    @property
    def exception(self) -> BaseException | None:
        return self._exception


class LocalCluster(Cluster):
    """Local cluster implementation using subprocess.

    Runs jobs as local subprocesses in isolated virtual environments. Useful for
    development, testing, and single-machine workloads. Does not support distributed
    execution or GPU/TPU resources.
    """

    def __init__(self, config: LocalClusterConfig = LocalClusterConfig()):
        """Initialize local cluster."""
        self._jobs: dict[JobId, _LocalJob] = {}
        self._shared_envs: dict[tuple[str, tuple[str, ...], tuple[str, ...]], TemporaryVenv] = {}
        self.config = config

    @staticmethod
    def from_spec(spec: dict[str, list[str]]) -> Cluster:
        logger.info(f"Creating local cluster with spec: {spec}")
        isolated_env_mode = spec.get("isolated_env_mode", [None])[0]
        if isolated_env_mode is not None:
            isolated_env_mode = isolated_env_mode.strip().lower() or None

        config = LocalClusterConfig(isolated_env_mode=isolated_env_mode)
        logger.info(f"Local cluster config: {config}")
        return LocalCluster(config=config)

    def _get_cluster_spec(self) -> str:
        if self.config.isolated_env_mode is None:
            return "local"
        return f"local?isolated_env_mode={self.config.isolated_env_mode}"

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
        for env in self._shared_envs.values():
            try:
                env.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error cleaning up shared venv: {e}")
        self._shared_envs.clear()

    def _shared_env_key(self, env_config: EnvironmentConfig) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        if not env_config.workspace:
            raise ValueError("LocalCluster isolated execution requires environment.workspace")
        return (
            str(Path(env_config.workspace).resolve()),
            tuple(env_config.pip_packages),
            tuple(env_config.extras),
        )

    def _get_or_create_shared_env(self, env_config: EnvironmentConfig) -> TemporaryVenv:
        key = self._shared_env_key(env_config)
        existing = self._shared_envs.get(key)
        if existing is not None:
            return existing

        venv = TemporaryVenv(
            workspace=env_config.workspace,
            pip_install_args=list(env_config.pip_packages),
            extras=list(env_config.extras),
            prefix="fray-shared-env-",
        )
        venv.__enter__()
        self._shared_envs[key] = venv
        return venv

    def launch(self, request: JobRequest) -> JobId:
        """Launch a job as a local subprocess."""
        if request.environment is None:
            raise ValueError("LocalCluster requires environment configuration")

        job_id = JobId(str(uuid.uuid4()))
        replica_count = request.resources.replicas

        logger.info(f"Launching job {job_id} with {replica_count} replica(s)")

        processes = []
        process_env = None
        owns_process_env = False
        if self.config.isolated_env_mode is None:
            if request.environment.env_vars:
                logger.warning(
                    "LocalCluster does not support custom environment variables in non-isolated mode, ignored."
                )

            # Run callable in parent process as a thread
            callable_ep = request.entrypoint.callable_entrypoint
            processes = []
            for _ in range(replica_count):
                logger.info(
                    f"Running callable in parent process with args={callable_ep.args}, kwargs={callable_ep.kwargs}"
                )
                process = FakeProcess(
                    target=callable_ep.callable, args=callable_ep.args, kwargs=callable_ep.kwargs, daemon=True
                )
                process.start()
                processes.append(process)
        else:
            owns_process_env = True
            if self.config.isolated_env_mode == "shared":
                process_env = self._get_or_create_shared_env(request.environment)
                owns_process_env = False
            else:
                if self.config.isolated_env_mode != "per_job":
                    raise RuntimeError(f"Unknown isolated_env_mode: {self.config.isolated_env_mode!r}")
                process_env = TemporaryVenv(
                    workspace=request.environment.workspace,
                    pip_install_args=list(request.environment.pip_packages),
                    extras=list(request.environment.extras),
                    prefix=f"fray-job-{job_id}-",
                )
                process_env.__enter__()
            cmd = self._build_command(request, process_env)
            logger.info(f"Command: {' '.join(cmd)}")
            # Launch all replicas using shared venv
            try:
                for replica_id in range(replica_count):
                    replica_env = self._build_replica_env(process_env, request.environment, replica_id, replica_count)

                    process = process_env.run_async(
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
                if owns_process_env and process_env is not None:
                    process_env.__exit__(None, None, None)
                raise RuntimeError(f"Failed to launch job: {e}") from e

        local_job = _LocalJob(
            job_id=job_id,
            request=request,
            processes=processes,
            process_env=process_env,
            owns_process_env=owns_process_env,
            start_time=time.time(),
        )
        self._jobs[job_id] = local_job
        local_job.start_log_thread()

        return job_id

    def monitor(self, job_id: JobId) -> JobInfo:
        """Monitor job output from all replicas, logging directly."""
        job = self._get_job(job_id)
        while True:
            try:
                line = job.log_queue.get(timeout=1)
                logger.info(line.rstrip())
            except Empty:
                if all(process.poll() is not None for process in job.processes):
                    logger.info("All processes for job %s have finished.", job_id)
                    while not job.log_queue.empty():
                        logger.info(job.log_queue.get_nowait().rstrip())
                    break
        return self.poll(job_id)

    def poll(self, job_id: JobId) -> JobInfo:
        return self._get_job(job_id).get_info()

    def terminate(self, job_id: JobId) -> None:
        """Terminate all replicas of a job."""
        self._get_job(job_id).cleanup()

    def list_jobs(self) -> list[JobInfo]:
        return [job.get_info() for job in self._jobs.values()]

    def _get_job(self, job_id: JobId) -> "_LocalJob":
        if job_id not in self._jobs:
            raise KeyError(f"Job {job_id} not found")
        return self._jobs[job_id]

    def _build_command(self, request: JobRequest, process_env: TemporaryVenv) -> list[str]:
        from fray.fn_thunk import create_thunk_entrypoint

        entrypoint = request.entrypoint

        if entrypoint.callable_entrypoint is not None:
            # Ensure tmp directory exists for thunk files
            tmp_dir = Path(process_env.workspace_path) / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            callable_ep = entrypoint.callable_entrypoint
            entrypoint = create_thunk_entrypoint(
                callable_ep.callable,
                prefix=f"{process_env.workspace_path}/tmp/",
                args=callable_ep.args,
                kwargs=callable_ep.kwargs,
            )

        assert entrypoint.binary_entrypoint is not None, "Entrypoint requires binary"
        return [entrypoint.binary_entrypoint.command, *entrypoint.binary_entrypoint.args]

    def _build_replica_env(
        self, process_env: TemporaryVenv, env_config: EnvironmentConfig, replica_id: int, replica_count: int
    ) -> dict[str, str]:
        env = process_env.get_env()
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

    job_id: JobId
    request: JobRequest
    processes: list[subprocess.Popen]
    process_env: TemporaryVenv | None
    owns_process_env: bool
    replica_count: int
    start_time: float
    log_queue: Queue[str]
    _log_threads: list[Thread]

    def __init__(
        self,
        job_id: JobId,
        request: JobRequest,
        processes: list[subprocess.Popen],
        process_env: TemporaryVenv | None,
        owns_process_env: bool,
        start_time: float,
    ):
        self.job_id = job_id
        self.request = request
        self.processes = processes
        self.process_env = process_env
        self.owns_process_env = owns_process_env
        self.replica_count = len(processes)
        self.start_time = start_time
        self.log_queue: Queue[str] = Queue()
        self._log_threads: list[Thread] = []

    def cleanup(self):
        """Clean up venv and all tracked processes."""
        logger.info(f"Cleaning up job {self.job_id}")
        if self.process_env is not None and self.owns_process_env:
            self.process_env.__exit__(None, None, None)
            self.process_env = None

    def start_log_thread(self):
        """Start background threads to collect logs from all replicas."""
        if self.process_env is None:
            return
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
        for i, rc in enumerate(returncodes):
            if rc is None:
                task_status.append(TaskStatus(status="running", error_message=""))
            elif rc != 0:
                # Try to get exception message from FakeProcess
                process = self.processes[i]
                if isinstance(process, FakeProcess) and process.exception is not None:
                    error_msg = f"Replica failed: {process.exception}"
                else:
                    error_msg = f"Replica failed with exit code {rc}"
                task_status.append(TaskStatus(status="failed", error_message=error_msg))
            else:
                task_status.append(TaskStatus(status="succeeded", error_message=""))

        if any(ts.status == "running" for ts in task_status):
            # At least one replica still running
            status = "running"
            error_message = None
        elif any(ts.status == "failed" for ts in task_status):
            # At least one replica failed
            status = "failed"
            # Collect error messages from failed tasks
            failed_msgs = [ts.error_message for ts in task_status if ts.status == "failed" and ts.error_message]
            error_message = "; ".join(failed_msgs) if failed_msgs else "One or more replicas failed"
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
