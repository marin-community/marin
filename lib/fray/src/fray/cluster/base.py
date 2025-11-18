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

"""Abstract cluster interface for job scheduling."""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

from fray.cluster.types import JobId, JobInfo, JobRequest

logger = logging.getLogger(__name__)


class Cluster(ABC):
    """Abstract interface for cluster job scheduling.

    Provides methods to launch jobs, monitor their progress, and manage
    their lifecycle. Implementations include RayCluster and LocalCluster.

    The Cluster abstraction is designed to handle both:
    1. CLI-style job submissions (fire-and-forget batch jobs)
    2. ray.remote-style function execution (distributed computation)

    Any execution pattern that resembles job-level scheduling (as opposed
    to task-level execution within a running job) should use this interface.
    """

    @abstractmethod
    def launch(self, request: JobRequest) -> JobId:
        """Launch a job on the cluster.

        Args:
            request: Job specification including resources, environment, and entrypoint

        Returns:
            Unique identifier for the launched job

        Raises:
            ValueError: If the request is invalid
            RuntimeError: If job submission fails
        """
        ...

    @abstractmethod
    def monitor(self, job_id: JobId) -> Iterator[str]:
        """Stream logs from a running job.

        Yields log lines as they become available. Blocks until the job
        completes or is terminated.

        Args:
            job_id: Job identifier returned by launch()

        Yields:
            Log lines as they become available

        Raises:
            KeyError: If job_id is not found
        """
        ...

    @abstractmethod
    def poll(self, job_id: JobId) -> JobInfo:
        """Get current status of a job without blocking.

        Args:
            job_id: Job identifier

        Returns:
            Current job information including status

        Raises:
            KeyError: If job_id is not found
        """
        ...

    @abstractmethod
    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Attempts graceful termination first, then forceful kill if needed.

        Args:
            job_id: Job identifier

        Raises:
            KeyError: If job_id is not found
        """
        ...

    @abstractmethod
    def list_jobs(self) -> list[JobInfo]:
        """List all jobs managed by this cluster.

        Returns:
            List of job information for all jobs (running, completed, and failed)
        """
        ...

    def wait(self, job_id: JobId) -> JobInfo:
        """Block until the specified job completes, returning its final status."""
        # default implementation polls until job is no longer running
        logger.info(f"[WAIT] Starting wait for job {job_id}")

        while True:
            info = self.poll(job_id)
            logger.info(f"[WAIT] Job {job_id} status: {info.status}")
            if info.status in ["succeeded", "failed", "stopped"]:
                logger.info(f"[WAIT] Job {job_id} completed with status {info.status}")
                return info
            time.sleep(0.5)
