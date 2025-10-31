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

"""Monarch-backed ClusterContext implementation for Fray."""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fray.cluster import ClusterContext, EntryPoint, JobInfo, RuntimeEnv, TpuRunConfig
from fray.job import JobContext


@dataclass
class MonarchJobInfo:
    """Internal job tracking information."""

    job_id: str
    process: subprocess.Popen
    entrypoint: str
    working_dir: Path
    status: str  # "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED"
    submission_time: float
    start_time: float | None
    end_time: float | None


class MonarchClusterContext(ClusterContext):
    """Manual implementation of cluster-level job management for Monarch backend.

    Jobs are launched as subprocesses running Monarch controller processes.
    This provides a simple cluster interface without requiring complex
    orchestration infrastructure.

    Key design decisions:
    1. Each job is a subprocess
    2. Jobs run on local host (single-host for now)
    3. Logs captured to files
    4. Status tracked via process polling
    5. No scheduling - jobs start immediately
    """

    def __init__(self, log_dir: Path | None = None):
        """
        Initialize MonarchClusterContext.

        Args:
            log_dir: Directory for storing job logs (default: ./.fray/logs)
        """
        self._jobs: dict[str, MonarchJobInfo] = {}
        self._log_dir = log_dir or Path.cwd() / ".fray" / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Lock for thread-safe job access
        self._jobs_lock = threading.Lock()

    def create_job(self, entrypoint: EntryPoint, env: RuntimeEnv) -> str:
        """
        Launch a new job as a subprocess.

        Args:
            entrypoint: Shell command to execute
            env: Runtime environment specification

        Returns:
            Unique job ID

        Raises:
            RuntimeError: If job submission fails
        """
        # Generate unique job ID
        job_id = f"monarch-{uuid.uuid4().hex[:8]}"

        # Prepare environment variables
        job_env = os.environ.copy()
        if env.env:
            job_env.update(env.env)

        # Set up working directory
        working_dir = Path.cwd()  # RuntimeEnv doesn't have working_dir field

        # Create log files
        stdout_log = self._log_dir / f"{job_id}.stdout"
        stderr_log = self._log_dir / f"{job_id}.stderr"

        wrapped_entrypoint = entrypoint

        # if env.package_requirements:
        #     # Create a wrapper script that installs packages then runs entrypoint
        #     install_cmd = f"pip install {' '.join(env.package_requirements)}"
        #     wrapped_entrypoint = f"{install_cmd} && {entrypoint}"
        # else:
        #     wrapped_entrypoint = entrypoint

        submission_time = time.time()

        try:
            # Launch controller process
            with open(stdout_log, "w") as stdout, open(stderr_log, "w") as stderr:
                process = subprocess.Popen(
                    wrapped_entrypoint,
                    shell=True,
                    env=job_env,
                    cwd=working_dir,
                    stdout=stdout,
                    stderr=stderr,
                )

            # Track job
            with self._jobs_lock:
                self._jobs[job_id] = MonarchJobInfo(
                    job_id=job_id,
                    process=process,
                    entrypoint=entrypoint,
                    working_dir=working_dir,
                    status="RUNNING",
                    submission_time=submission_time,
                    start_time=submission_time,  # Job starts immediately
                    end_time=None,
                )

            return job_id

        except Exception as e:
            raise RuntimeError(f"Failed to submit job: {e}") from e

    def list_jobs(self) -> list[JobInfo]:
        """
        List all tracked jobs with current status.

        Returns:
            List of JobInfo objects
        """
        jobs = []

        with self._jobs_lock:
            for job_id, info in self._jobs.items():
                # Update status based on process state
                if info.status == "RUNNING":
                    returncode = info.process.poll()
                    if returncode is not None:
                        info.status = "SUCCEEDED" if returncode == 0 else "FAILED"
                        info.end_time = time.time()

                jobs.append(
                    JobInfo(
                        id=job_id,
                        status=info.status,
                        submission_time=info.submission_time,
                        start_time=info.start_time,
                        end_time=info.end_time,
                    )
                )

        return jobs

    def delete_job(self, job_id: str) -> None:
        """
        Terminate a running job.

        Args:
            job_id: Job ID to delete

        Raises:
            ValueError: If job not found
        """
        with self._jobs_lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")

            info = self._jobs[job_id]

            if info.status == "RUNNING":
                # Terminate process
                info.process.terminate()
                try:
                    info.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if terminate doesn't work
                    info.process.kill()
                    info.process.wait()

                info.status = "STOPPED"
                info.end_time = time.time()

            # Remove from tracking
            del self._jobs[job_id]

    def get_job_logs(self, job_id: str) -> tuple[str, str]:
        """
        Retrieve stdout/stderr logs for a job.

        Args:
            job_id: Job ID to get logs for

        Returns:
            Tuple of (stdout, stderr) content

        Raises:
            ValueError: If job not found
        """
        with self._jobs_lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")

        stdout_log = self._log_dir / f"{job_id}.stdout"
        stderr_log = self._log_dir / f"{job_id}.stderr"

        stdout = stdout_log.read_text() if stdout_log.exists() else ""
        stderr = stderr_log.read_text() if stderr_log.exists() else ""

        return stdout, stderr

    def run_on_tpu(
        self,
        fn: Callable[[JobContext], Any],
        config: TpuRunConfig,
        runtime_env: RuntimeEnv | None = None,
    ) -> list[Any]:
        """
        Execute function across TPU slices.

        Note: TPU support is not yet implemented for Monarch backend.

        Args:
            fn: Function to execute on each TPU host
            config: TPU configuration
            runtime_env: Runtime environment

        Raises:
            NotImplementedError: TPU support not yet available
        """
        raise NotImplementedError(
            "TPU support is not yet implemented for Monarch backend. " "Use Ray backend for TPU workloads."
        )
