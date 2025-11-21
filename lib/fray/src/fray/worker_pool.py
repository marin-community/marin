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

"""Autoscaling worker pool for distributed job-based processing."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from fray.cluster.base import Cluster, CpuConfig, JobId, JobInfo, JobRequest, ResourceConfig, create_environment
from fray.queues.base import Queue

logger = logging.getLogger(__name__)

# Sentinel value for graceful worker shutdown (must be pickleable)
_SHUTDOWN_SENTINEL = "__FRAY_WORKER_SHUTDOWN__"


@dataclass
class WorkerPoolConfig:
    """Configuration for worker pool.

    Attributes:
        worker_impl: Class to run on workers. Must be callable after instantiation.
        num_workers: Number of workers to maintain
        resources: Resource configuration for workers (default: CPU)
        worker_args: Positional arguments for worker initialization
        worker_kwargs: Keyword arguments for worker initialization
    """

    worker_impl: type
    num_workers: int
    resources: ResourceConfig | None = None
    worker_args: tuple = field(default_factory=tuple)
    worker_kwargs: dict[str, Any] = field(default_factory=dict)


class WorkerPool:
    """Fixed-size pool of worker jobs for distributed task processing.

    The pool manages a fixed set of worker jobs. Tasks are distributed via
    provided queues.
    """

    def __init__(
        self,
        cluster: Cluster,
        config: WorkerPoolConfig,
        task_queue: Queue,
        result_queue: Queue,
    ):
        """Initialize worker pool.

        Args:
            cluster: Cluster backend to use for worker jobs
            config: Pool configuration including worker function
            task_queue: Queue for submitting tasks
            result_queue: Queue for collecting results
        """
        self._cluster = cluster
        self._config = config
        self._task_queue = task_queue
        self._result_queue = result_queue

        # Generate unique pool ID for worker naming
        self._pool_id = str(uuid.uuid4())

        # Worker tracking
        self._workers: dict[JobId, JobInfo | None] = {}
        self._metadata_lock = Lock()

        # Create initial workers
        for _ in range(self._config.num_workers):
            self._create_worker()

    def _create_worker(self) -> JobId:
        """Launch a new worker job.

        The cluster will automatically inject FRAY_CLUSTER_SPEC, allowing
        workers to call current_cluster() without manual setup.

        Returns:
            Job ID of the created worker
        """
        from fray.cluster.base import Entrypoint

        worker_impl = self._config.worker_impl
        worker_args = self._config.worker_args
        worker_kwargs = self._config.worker_kwargs
        task_queue = self._task_queue
        result_queue = self._result_queue

        def worker_closure():
            logging.basicConfig(level=logging.INFO)
            logger.info(f"Worker initialized. Queue types: {type(task_queue)}, {type(result_queue)}")

            # Initialize worker
            try:
                processor = worker_impl(*worker_args, **worker_kwargs)
                logger.info(f"Initialized worker: {worker_impl.__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize worker {worker_impl.__name__}: {e}")
                raise

            # Check if worker handles its own loop
            if hasattr(processor, "run") and callable(processor.run):
                logger.info(f"Worker {worker_impl.__name__} has 'run' method, delegating queue handling.")
                try:
                    processor.run(task_queue, result_queue)
                except Exception as e:
                    logger.error(f"Worker run loop failed: {e}")
                    raise
            else:
                # Process tasks until terminated (Legacy/Simple mode)
                while True:
                    # Try to get a task with short lease timeout for faster recovery
                    lease = task_queue.pop(lease_timeout=5.0)

                    if lease is None:
                        # No tasks available, sleep and retry
                        time.sleep(0.1)
                        continue

                    # Check for shutdown sentinel
                    if lease.item == _SHUTDOWN_SENTINEL:
                        task_queue.done(lease)
                        logger.info("Worker received shutdown signal, exiting gracefully")
                        break

                    try:
                        # Process the task using the user's worker function
                        logger.debug(f"Worker processing task: {lease.item}")
                        result = processor(lease.item)
                        logger.debug(f"Worker produced result: {result}")

                        # Publish result
                        result_queue.push(result)

                        # Mark task as complete
                        task_queue.done(lease)

                    except Exception as e:
                        logger.error(f"Error processing task: {e}")
                        # Release the lease so task can be retried
                        task_queue.release(lease)

        resources = self._config.resources or ResourceConfig(device=CpuConfig())

        request = JobRequest(
            name=f"worker-{self._pool_id[:8]}",
            entrypoint=Entrypoint(callable=worker_closure),
            resources=resources,
            environment=create_environment(),
        )

        job_id = self._cluster.launch(request)

        with self._metadata_lock:
            self._workers[job_id] = None

        logger.info(f"Created worker {job_id}, running {worker_closure} total workers: {len(self._workers)}")
        return job_id

    def submit(self, task: Any) -> None:
        """Submit a task to the worker pool.

        Args:
            task: Task data to submit
        """
        self._task_queue.push(task)

    def collect(self, num_results: int, timeout: float | None = None) -> list[Any]:
        """Collect results from the worker pool.

        Args:
            num_results: Number of results to collect
            timeout: Maximum time to wait for all results (seconds)

        Returns:
            List of results

        Raises:
            TimeoutError: If timeout is reached before all results are collected
        """
        results = []
        start_time = time.time()

        for _ in range(num_results):
            while True:
                if timeout is not None and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Timeout collecting results: got {len(results)}/{num_results}")

                lease = self._result_queue.pop()
                if lease is not None:
                    results.append(lease.item)
                    self._result_queue.done(lease)
                    break

                # No result available, sleep briefly and retry
                time.sleep(0.1)

        return results

    def num_workers(self) -> int:
        with self._metadata_lock:
            return len(self._workers)

    def shutdown(self, timeout: float = 60.0) -> None:
        """Shutdown the pool and clean up resources.

        Args:
            timeout: Maximum time to wait for threads to finish
        """
        logger.info("Shutting down worker pool")

        # Send shutdown sentinels to workers for graceful exit
        with self._metadata_lock:
            num_workers = len(self._workers)

        for _ in range(num_workers):
            self._task_queue.push(_SHUTDOWN_SENTINEL)

        # Wait for workers to finish (optional, or just kill them)
        # Since we don't have a way to wait for specific tasks easily without tracking,
        # we'll give them some time then kill.
        time.sleep(min(timeout, 5.0))

        # Terminate any remaining workers
        with self._metadata_lock:
            workers_to_kill = list(self._workers.keys())
            self._workers.clear()

        for worker_id in workers_to_kill:
            try:
                self._cluster.terminate(worker_id)
            except Exception as e:
                logger.warning(f"Failed to terminate worker {worker_id}: {e}")

        logger.info("Worker pool shutdown complete")
