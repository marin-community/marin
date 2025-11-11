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

"""Autoscaling worker pool for distributed actor-based processing."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Generic, TypeVar

import ray
from ray.util.queue import Queue

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass
class WorkerPoolConfig:
    """Configuration for autoscaling worker pool.

    Attributes:
        min_workers: Minimum number of workers to maintain
        max_workers: Maximum number of workers to scale up to
        scale_up_threshold: Queue size ratio to trigger scale up (tasks/workers)
        scale_down_threshold: Queue size ratio to trigger scale down
        scale_check_interval: How often to check scaling conditions (seconds)
        actor_kwargs: Keyword arguments to pass to actor constructor
        actor_options: Ray actor options (e.g., num_cpus, resources)
    """

    min_workers: int = 1
    max_workers: int = 8
    scale_up_threshold: float = 2.0
    scale_down_threshold: float = 0.5
    scale_check_interval: float = 5.0
    actor_kwargs: dict[str, Any] | None = None
    actor_options: dict[str, Any] | None = None


class WorkerPool(Generic[T, R]):
    """Autoscaling pool of Ray actors for distributed task processing.

    The pool manages a dynamic set of Ray actors, automatically scaling up when
    the task queue grows and scaling down when idle. Tasks are distributed via
    Ray queues with background threads handling dispatch and result collection.

    Example:
        >>> @ray.remote
        ... class Worker:
        ...     def process(self, item):
        ...         return item * 2
        >>>
        >>> pool = WorkerPool(Worker, config=WorkerPoolConfig(max_workers=4))
        >>> results = pool.map([1, 2, 3, 4, 5])
        >>> list(results)  # [2, 4, 6, 8, 10]
        >>> pool.shutdown()
    """

    def __init__(
        self,
        actor_class: type,
        method_name: str = "process",
        config: WorkerPoolConfig | None = None,
        **actor_kwargs,
    ):
        """Initialize the worker pool.

        Args:
            actor_class: Ray actor class to use for workers
            method_name: Name of the method to call on actors
            config: Pool configuration (uses defaults if not provided)
            **actor_kwargs: Additional keyword arguments for actor constructor
        """
        self.actor_class = actor_class
        self.method_name = method_name
        self.config = config or WorkerPoolConfig()

        # Merge actor_kwargs from config and direct args
        self.actor_kwargs = {**(self.config.actor_kwargs or {}), **actor_kwargs}
        self.actor_options = self.config.actor_options or {}

        # Queues for task distribution
        self.task_queue: Queue = Queue()
        self.result_queue: Queue = Queue()

        # Actor management
        self.actors: list[ray.actor.ActorHandle] = []
        self.actor_futures: dict[ray.actor.ActorHandle, set[ray.ObjectRef]] = {}
        self.future_to_actor: dict[ray.ObjectRef, ray.actor.ActorHandle] = {}
        self.future_to_task: dict[ray.ObjectRef, Any] = {}
        self.metadata_lock = Lock()

        # Control flags
        self._shutdown_flag = threading.Event()
        self._threads: list[threading.Thread] = []

        # Start background threads
        self._start_background_threads()

        # Create initial workers
        for _ in range(self.config.min_workers):
            self._create_worker()

    def _create_worker(self) -> ray.actor.ActorHandle:
        """Create and register a new worker actor.

        Returns:
            The created actor handle
        """
        actor = self.actor_class.options(**self.actor_options).remote(**self.actor_kwargs)

        # Ping to ensure actor is ready
        ray.get(actor.ping.remote() if hasattr(actor, "ping") else actor.__ray_ready__.remote())

        with self.metadata_lock:
            self.actors.append(actor)
            self.actor_futures[actor] = set()

        logger.info(f"Created worker, total workers: {len(self.actors)}")
        return actor

    def _kill_worker(self, actor: ray.actor.ActorHandle) -> None:
        """Remove and kill a worker actor.

        Args:
            actor: Actor to kill
        """
        with self.metadata_lock:
            if actor in self.actors:
                self.actors.remove(actor)
                self.actor_futures.pop(actor, None)

        ray.kill(actor)
        logger.info(f"Killed worker, total workers: {len(self.actors)}")

    def _get_least_loaded_actor(self) -> ray.actor.ActorHandle | None:
        """Get the actor with the fewest pending tasks.

        Returns:
            Actor with minimum load, or None if no actors available
        """
        with self.metadata_lock:
            if not self.actors:
                return None

            # Find actor with minimum number of pending futures
            return min(self.actors, key=lambda a: len(self.actor_futures.get(a, set())))

    def _dispatch_task(self, task: T) -> None:
        """Dispatch a task to the least loaded worker.

        Args:
            task: Task to dispatch
        """
        actor = self._get_least_loaded_actor()
        if actor is None:
            logger.warning("No workers available, requeueing task")
            self.task_queue.put(task)
            return

        # Submit task to actor
        method = getattr(actor, self.method_name)
        future = method.remote(task)

        with self.metadata_lock:
            self.actor_futures[actor].add(future)
            self.future_to_actor[future] = actor
            self.future_to_task[future] = task

    def _dispatcher_loop(self) -> None:
        """Background thread: pull tasks from queue and dispatch to actors."""
        while not self._shutdown_flag.is_set():
            try:
                # Get task with timeout to allow checking shutdown flag
                task = self.task_queue.get(timeout=0.1)
                self._dispatch_task(task)
            except Exception as e:
                if not self._shutdown_flag.is_set():
                    logger.debug(f"Dispatcher queue timeout: {e}")

    def _result_collector_loop(self) -> None:
        """Background thread: collect completed futures and put results in queue."""
        while not self._shutdown_flag.is_set():
            with self.metadata_lock:
                all_futures = list(self.future_to_actor.keys())

            if not all_futures:
                time.sleep(0.1)
                continue

            # Wait for at least one future to complete
            ready_futures, _ = ray.wait(all_futures, num_returns=1, timeout=0.1)

            for future in ready_futures:
                try:
                    result = ray.get(future)
                    self.result_queue.put(result)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    # Optionally requeue failed tasks
                    with self.metadata_lock:
                        failed_task = self.future_to_task.get(future)
                    if failed_task is not None:
                        logger.info("Requeueing failed task")
                        self.task_queue.put(failed_task)

                # Clean up metadata
                with self.metadata_lock:
                    actor = self.future_to_actor.pop(future, None)
                    if actor and actor in self.actor_futures:
                        self.actor_futures[actor].discard(future)
                    self.future_to_task.pop(future, None)

    def _autoscaler_loop(self) -> None:
        """Background thread: monitor load and scale workers up/down."""
        while not self._shutdown_flag.is_set():
            time.sleep(self.config.scale_check_interval)

            num_workers = len(self.actors)
            queue_size = self.task_queue.qsize()

            if num_workers == 0:
                continue

            load_ratio = queue_size / num_workers

            # Scale up if queue is growing
            if load_ratio > self.config.scale_up_threshold and num_workers < self.config.max_workers:
                logger.info(f"Scaling up: load_ratio={load_ratio:.2f}, workers={num_workers}")
                self._create_worker()

            # Scale down if idle
            elif load_ratio < self.config.scale_down_threshold and num_workers > self.config.min_workers:
                # Only scale down if there are idle workers
                with self.metadata_lock:
                    idle_actors = [a for a in self.actors if len(self.actor_futures.get(a, set())) == 0]

                if idle_actors:
                    logger.info(f"Scaling down: load_ratio={load_ratio:.2f}, workers={num_workers}")
                    self._kill_worker(idle_actors[0])

    def _start_background_threads(self) -> None:
        """Start dispatcher, result collector, and autoscaler threads."""
        threads = [
            threading.Thread(target=self._dispatcher_loop, daemon=True, name="dispatcher"),
            threading.Thread(target=self._result_collector_loop, daemon=True, name="result_collector"),
            threading.Thread(target=self._autoscaler_loop, daemon=True, name="autoscaler"),
        ]

        for thread in threads:
            thread.start()
            self._threads.append(thread)

    def map(self, items: list[T]) -> list[R]:
        """Process a list of items through the worker pool.

        Args:
            items: Items to process

        Returns:
            List of results in the same order as input items

        Example:
            >>> results = pool.map([1, 2, 3, 4, 5])
        """
        # Submit all tasks
        for item in items:
            self.task_queue.put(item)

        # Collect all results
        results = []
        for _ in range(len(items)):
            result = self.result_queue.get()
            results.append(result)

        return results

    def shutdown(self, timeout: float = 10.0) -> None:
        """Shutdown the pool and clean up resources.

        Args:
            timeout: Maximum time to wait for threads to finish
        """
        logger.info("Shutting down worker pool")

        # Signal shutdown
        self._shutdown_flag.set()

        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=timeout)

        # Kill all actors
        with self.metadata_lock:
            actors_to_kill = list(self.actors)

        for actor in actors_to_kill:
            ray.kill(actor)

        self.actors.clear()
        self.actor_futures.clear()
        self.future_to_actor.clear()
        self.future_to_task.clear()

        logger.info("Worker pool shutdown complete")
