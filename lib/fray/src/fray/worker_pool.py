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
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

from fray.cluster.base import Cluster, CpuConfig, JobId, JobInfo, JobRequest, ResourceConfig, create_environment
from fray.cluster.queue import Queue

logger = logging.getLogger(__name__)


@dataclass
class WorkerPoolConfig:
    """Configuration for worker pool autoscaling.

    Attributes:
        worker_func: Callable that processes individual tasks (task -> result)
        min_workers: Minimum number of workers to maintain
        max_workers: Maximum number of workers to scale up to
        scale_up_threshold: Tasks per worker to trigger scale up
        scale_down_threshold: Tasks per worker to trigger scale down
        scale_check_interval: How often to check scaling conditions (seconds)
    """

    worker_func: Callable[[Any], Any]
    min_workers: int
    max_workers: int
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.2
    scale_check_interval: float = 5.0


class WorkerPool:
    """Autoscaling pool of worker jobs for distributed task processing.

    The pool manages a dynamic set of worker jobs, automatically scaling up when
    the task queue grows and scaling down when idle. Tasks are distributed via
    queues with a background autoscaler thread.

    Unlike the Zephyr WorkerPool which uses Ray actors, this implementation uses
    the Cluster API to launch separate worker jobs. Workers consume tasks from a
    shared queue and publish results to a result queue.
    """

    def __init__(
        self,
        cluster: Cluster,
        config: WorkerPoolConfig,
    ):
        """Initialize worker pool.

        Args:
            cluster: Cluster backend to use for worker jobs
            config: Pool configuration including worker function and scaling parameters
        """
        self._cluster = cluster
        self._config = config

        # Generate unique pool ID for queue names
        self._pool_id = str(uuid.uuid4())

        # Create queues for task distribution and result collection
        task_queue_name = f"{self._pool_id}_tasks"
        result_queue_name = f"{self._pool_id}_results"
        logger.info(f"Creating queue from cluster {cluster}")
        self._task_queue: Queue = cluster.create_queue(task_queue_name)
        self._result_queue: Queue = cluster.create_queue(result_queue_name)

        # Worker tracking
        self._workers: dict[JobId, JobInfo] = {}
        self._metadata_lock = Lock()

        # Control flags
        self._shutdown_event = threading.Event()
        self._threads: list[threading.Thread] = []

        # Start background threads
        self._start_background_threads()

        # Create initial workers
        for _ in range(self._config.min_workers):
            self._create_worker()

    def _create_worker(self) -> JobId:
        """Launch a new worker job.

        The cluster will automatically inject FRAY_CLUSTER_SPEC, allowing
        workers to call current_cluster() without manual setup.

        Returns:
            Job ID of the created worker
        """
        from fray.cluster import current_cluster
        from fray.cluster.base import Entrypoint

        task_queue_name = f"{self._pool_id}_tasks"
        result_queue_name = f"{self._pool_id}_results"

        worker_func = self._config.worker_func

        def worker_closure():
            # Get queues from current cluster context
            cluster = current_cluster()
            logger.info(f"Worker initialized with cluster: {cluster}, type: {type(cluster).__name__}")

            # Log cluster details for debugging
            if hasattr(cluster, "_namespace"):
                logger.info(f"Worker cluster namespace: {cluster._namespace}")

            task_queue = cluster.create_queue(task_queue_name)
            result_queue = cluster.create_queue(result_queue_name)
            logger.info(f"Worker queues created: task={task_queue_name}, result={result_queue_name}")

            # Process tasks until terminated
            while True:
                # Try to get a task with short lease timeout for faster recovery
                lease = task_queue.pop(lease_timeout=5.0)

                if lease is None:
                    # No tasks available, sleep and retry
                    time.sleep(0.1)
                    continue

                try:
                    # Process the task using the user's worker function
                    logger.info(f"Worker processing task: {lease.item}")
                    result = worker_func(lease.item)
                    logger.info(f"Worker produced result: {result}")

                    # Publish result
                    result_queue.push(result)

                    # Mark task as complete
                    task_queue.done(lease)

                except Exception as e:
                    logger.error(f"Error processing task: {e}")
                    # Release the lease so task can be retried
                    task_queue.release(lease)

        request = JobRequest(
            name=f"worker-{self._pool_id[:8]}",
            entrypoint=Entrypoint(callable=worker_closure),
            resources=ResourceConfig(device=CpuConfig()),
            environment=create_environment(),
        )

        job_id = self._cluster.launch(request)

        with self._metadata_lock:
            # We don't have JobInfo yet, will update via poll
            self._workers[job_id] = None  # type: ignore

        logger.info(f"Created worker {job_id}, total workers: {len(self._workers)}")
        return job_id

    def _terminate_worker(self, job_id: JobId) -> None:
        """Terminate a worker job.

        Args:
            job_id: Job ID to terminate
        """
        with self._metadata_lock:
            if job_id in self._workers:
                del self._workers[job_id]

        try:
            self._cluster.terminate(job_id)
        except Exception as e:
            logger.warning(f"Failed to terminate worker {job_id}: {e}")

        logger.info(f"Terminated worker {job_id}, total workers: {len(self._workers)}")

    def _autoscale_loop(self) -> None:
        """Background thread: monitor load and scale workers up/down."""
        while not self._shutdown_event.is_set():
            time.sleep(self._config.scale_check_interval)

            if self._shutdown_event.is_set():
                break

            try:
                # Get worker count
                with self._metadata_lock:
                    num_workers = len(self._workers)

                if num_workers == 0:
                    continue

                # Calculate queue depth (pending tasks)
                queue_depth = self._task_queue.pending()
                load_ratio = queue_depth / num_workers

                # Scale up if queue is growing
                if load_ratio > self._config.scale_up_threshold and num_workers < self._config.max_workers:
                    logger.info(f"Scaling up: load_ratio={load_ratio:.2f}, workers={num_workers}")
                    self._create_worker()

                # Scale down if idle
                elif load_ratio < self._config.scale_down_threshold and num_workers > self._config.min_workers:
                    logger.info(f"Scaling down: load_ratio={load_ratio:.2f}, workers={num_workers}")
                    # Get worker to terminate
                    with self._metadata_lock:
                        if self._workers:
                            worker_to_kill = next(iter(self._workers.keys()))
                        else:
                            worker_to_kill = None

                    # Terminate outside the lock to avoid deadlock
                    if worker_to_kill is not None:
                        self._terminate_worker(worker_to_kill)

            except Exception as e:
                if not self._shutdown_event.is_set():
                    logger.error(f"Error in autoscaler loop: {e}")

    def _start_background_threads(self) -> None:
        autoscaler_thread = threading.Thread(target=self._autoscale_loop, daemon=True, name="autoscaler")
        autoscaler_thread.start()
        self._threads.append(autoscaler_thread)

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

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for threads to finish (this releases the lock)
        for thread in self._threads:
            thread.join(timeout=timeout)

        # Now safe to acquire lock and terminate workers
        with self._metadata_lock:
            workers_to_kill = list(self._workers.keys())
            self._workers.clear()

        # Terminate workers outside the lock
        for worker_id in workers_to_kill:
            try:
                self._cluster.terminate(worker_id)
            except Exception as e:
                logger.warning(f"Failed to terminate worker {worker_id}: {e}")

        logger.info("Worker pool shutdown complete")
