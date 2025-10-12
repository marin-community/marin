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

import logging
from typing import Any
import time
import ray
from threading import Thread, Lock
from multiprocessing import Event
from ray.exceptions import GetTimeoutError, RayActorError
from ray.util.queue import Queue

from marin.processing.classification.classifier import BaseClassifier

logger = logging.getLogger("ray")


class AutoscalingActorPool:
    """
    Autoscaling actor pool that manages BaseClassifier actors and distributes tasks.
    """

    NUM_ACTORS_TO_SCALE_UP = 1

    def __init__(
        self,
        actor_class: type[BaseClassifier],
        model_name_or_path: str,
        attribute_name: str,
        model_type: str,
        task_queue: Queue,
        result_queue: Queue,
        min_actors: int = 1,
        max_actors: int = 32,
        target_queue_size: int = 5,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
        scale_check_interval: float = 2,
        actor_kwargs: dict | None = None,
        actor_options: dict | None = None,
        actor_health_check_interval: float = 5.0,
        actor_health_check_timeout: float = 2.0,
        max_tasks_per_actor: int | None = 4,
    ):
        """
        Initialize the autoscaling actor pool.

        Args:
            actor_class: The Ray actor class to use
            min_actors: Minimum number of actors to maintain
            max_actors: Maximum number of actors allowed
            target_queue_size: Target queue size per actor for scaling decisions
            scale_up_threshold: Queue utilization threshold to trigger scale up
            scale_down_threshold: Queue utilization threshold to trigger scale down
            scale_check_interval: Interval in seconds between scaling checks
            actor_kwargs: Additional keyword arguments to pass to actor initialization
            actor_health_check_interval: Interval between actor health probes
            actor_health_check_timeout: Seconds to wait for a ping response before marking actor unhealthy
            max_tasks_per_actor: Maximum number of in-flight tasks allowed per actor. When reached, dispatch
                waits for capacity or new actors before assigning more work. If None, no hard cap is enforced.
        """
        self.actor_class = actor_class
        self.model_name_or_path = model_name_or_path
        self.attribute_name = attribute_name
        self.model_type = model_type
        self.min_actors = min_actors
        self.max_actors = max_actors
        self.target_queue_size = target_queue_size
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval = scale_check_interval
        self.actor_kwargs = actor_kwargs or {}
        self.task_queue = task_queue
        self.result_queue = result_queue

        # Actor management
        self.actors = []
        self.actor_futures = {}  # Track ongoing tasks per actor
        self.future_to_actor = {}
        self.future_to_task = {}
        self.actor_options = actor_options or {}
        self.actor_health_check_interval = actor_health_check_interval
        self.actor_health_check_timeout = actor_health_check_timeout
        self.max_tasks_per_actor = max_tasks_per_actor
        self.lock = Lock()

        # Statistics
        self.total_processed = 0
        self.total_submitted = 0

        # Initialize minimum number of actors
        self._initialize_actors()

        # Start autoscaling monitor, dispatcher, and result collector
        self.scaling_task = None
        self._start_autoscaling_monitor()
        self._start_dispatcher()
        self._start_result_collector()
        # self._start_actor_health_monitor()

    def _initialize_actors(self):
        """Initialize the minimum number of actors."""
        logger.info(f"Initializing {self.min_actors} actors...")
        for _ in range(self.min_actors):
            self._create_and_register_actor()
        logger.info(f"Initialized {len(self.actors)} actors")

    def _create_and_register_actor(self):
        """Create one actor and wait until it's ready before adding to pool."""
        NUM_RETRIES = 3
        for _ in range(NUM_RETRIES):
            try:
                actor = self.actor_class.options(**self.actor_options).remote(
                    self.model_name_or_path, self.attribute_name, self.model_type, **self.actor_kwargs
                )
                break
            except Exception as e:
                logger.warning(f"Actor creation failed: {e}")
                time.sleep(1)
        try:
            # Ensure the actor is scheduled and ready before exposing to dispatch
            ray.get(actor.ping.remote())
        except Exception as e:
            logger.warning(f"Actor readiness check failed: {e}")
        self.actors.append(actor)
        self.actor_futures[actor] = []

    def _start_autoscaling_monitor(self):
        """Start a local background thread for monitoring and autoscaling.

        Avoids passing the non-serializable pool object to a Ray actor.
        """
        self._monitor_stop_event: Event = Event()

        def _loop():
            while not self._monitor_stop_event.is_set():
                try:
                    time.sleep(self.scale_check_interval)
                    self._check_and_scale()
                except Exception:
                    # Best-effort; do not crash the monitor
                    pass

        self._monitor_thread: Thread = Thread(target=_loop, daemon=True)
        self._monitor_thread.start()

    def _start_dispatcher(self):
        """Start a background dispatcher that pulls tasks and dispatches work."""
        self._dispatch_stop_event: Event = Event()

        def _loop():
            while not self._dispatch_stop_event.is_set():
                if self.task_queue is None:
                    time.sleep(0.1)
                    continue

                if len(self.actors) == 0:
                    logger.debug("Dispatcher waiting for actors to come online")
                    time.sleep(0.5)
                    continue

                dispatched_any = False
                for _ in range(4):  # cap per-iteration dispatch volume
                    try:
                        task = self.task_queue.get(timeout=1)
                    except Exception:
                        break

                    self.total_submitted += 1
                    try:
                        self._dispatch_task(task)
                        dispatched_any = True
                    except RuntimeError:
                        # No actor available; put task back and wait before retrying
                        try:
                            self.task_queue.put(task)
                        except Exception:
                            logger.error("Dispatcher failed to requeue task after no actor available")
                        time.sleep(0.1)
                        break

                if not dispatched_any:
                    logger.info("No tasks dispatched - going to sleep for 10 seconds.")
                    time.sleep(10)

        self._dispatch_thread: Thread = Thread(target=_loop, daemon=True)
        self._dispatch_thread.start()

    def _start_result_collector(self):
        """Start a background thread that collects completed futures."""
        self._result_stop_event: Event = Event()

        def _loop():
            while not self._result_stop_event.is_set():
                futures_snapshot = []
                with self.lock:
                    futures_snapshot = list(self.future_to_actor.keys())

                if not futures_snapshot:
                    time.sleep(0.05)
                    continue

                ready_refs, _ = ray.wait(futures_snapshot, num_returns=len(futures_snapshot), timeout=0)

                if not ready_refs:
                    time.sleep(0.05)
                    continue

                for ref in ready_refs:
                    self._handle_completed_future(ref)

        self._result_thread: Thread = Thread(target=_loop, daemon=True)
        self._result_thread.start()

    def _handle_completed_future(self, future: ray.ObjectRef) -> None:
        task_to_retry = None
        try:
            result = ray.get(future, timeout=1)
            self.total_processed += 1

            if self.result_queue is not None:
                try:
                    self.result_queue.put(result)
                except Exception as e:
                    logger.error(f"Failed to put result in results queue! {e}")
        except (RayActorError, ray.exceptions.ActorDiedError):
            with self.lock:
                task_to_retry = self.future_to_task.get(future)
            if task_to_retry is not None and self.task_queue is not None:
                try:
                    self.task_queue.put(task_to_retry)
                except Exception:
                    logger.error("Failed to requeue task after actor failure")
        except Exception:
            logger.exception("Error retrieving future result")
        finally:
            with self.lock:
                actor = self.future_to_actor.pop(future, None)
                self.future_to_task.pop(future, None)
                if actor is not None and actor in self.actor_futures:
                    try:
                        self.actor_futures[actor].remove(future)
                    except ValueError:
                        logger.error("Failed to remove future from actor futures!")

    def _start_actor_health_monitor(self):
        """Start a background thread to verify actor liveness."""
        self._health_stop_event: Event = Event()

        def _loop():
            while not self._health_stop_event.is_set():
                with self.lock:
                    actors_snapshot = list(self.actors)
                for actor in actors_snapshot:
                    if self._health_stop_event.is_set():
                        break
                    try:
                        ping_ref = actor.ping.remote()
                        ray.get(ping_ref, timeout=self.actor_health_check_timeout)
                    except (RayActorError, GetTimeoutError):
                        self._handle_actor_failure(actor)
                    except Exception:
                        self._handle_actor_failure(actor)
                time.sleep(self.actor_health_check_interval)

        self._health_thread: Thread = Thread(target=_loop, daemon=True)
        self._health_thread.start()

    def _handle_actor_failure(self, actor):
        """Remove a failed actor and requeue its tasks."""
        tasks_to_requeue: list[Any] = []
        need_replacement = False

        with self.lock:
            if actor not in self.actors:
                return

            futures = self.actor_futures.pop(actor, [])
            try:
                self.actors.remove(actor)
            except ValueError:
                pass

            for future in futures:
                self.future_to_actor.pop(future, None)
                task = self.future_to_task.pop(future, None)
                if task is not None:
                    tasks_to_requeue.append(task)

            need_replacement = len(self.actors) < self.min_actors

        if tasks_to_requeue:
            logger.warning("Actor failure detected; requeueing %d pending tasks", len(tasks_to_requeue))
        else:
            logger.warning("Actor failure detected; no pending tasks to requeue")

        if self.task_queue is not None:
            for task in tasks_to_requeue:
                try:
                    logger.debug(
                        "Requeueing task after actor failure",
                        extra={
                            "task": task,
                            "requeue_count": len(tasks_to_requeue),
                        },
                    )
                    self.task_queue.put(task)
                except Exception:
                    logger.error("Failed to requeue task after actor failure")

        try:
            ray.kill(actor)
        except Exception:
            pass

        if need_replacement:
            try:
                self._create_and_register_actor()
            except Exception:
                logger.warning("Failed to replace actor after failure", exc_info=True)

    def _check_and_scale(self):
        """Check current load and scale actors accordingly."""
        with self.lock:
            current_actors = len(self.actors)
            try:
                pending_count = int(self.task_queue.qsize()) if self.task_queue is not None else 0
            except Exception:
                pending_count = 0

            # Calculate total active tasks
            active_tasks = sum(len(futures) for futures in self.actor_futures.values())
        total_load = pending_count + active_tasks

        # Calculate utilization
        per_actor_capacity = self.max_tasks_per_actor or self.target_queue_size
        capacity = current_actors * per_actor_capacity
        utilization = total_load / capacity if capacity > 0 else 0

        logger.info(
            f"Load check - Actors: {current_actors}, Pending: {pending_count}, "
            f"Active: {active_tasks}, Utilization: {utilization:.2%}"
        )

        # Scale up if needed
        if (
            utilization > self.scale_up_threshold and current_actors < self.max_actors
        ) or current_actors < self.min_actors:

            # If a system quickly overwhelms the scale up threshold,
            # we end up having to scale up too many actors at once.
            # Let's just scale up one at a time
            # new_actors_count = min(
            #     self.max_actors - current_actors,
            #     max(1, (total_load // self.target_queue_size) - current_actors)
            # )
            self._scale_up(self.NUM_ACTORS_TO_SCALE_UP)

        # Scale down if needed
        elif utilization < self.scale_down_threshold and current_actors > self.min_actors:
            # Only scale down idle actors
            idle_actors = [actor for actor, futures in self.actor_futures.items() if len(futures) == 0]
            if idle_actors:
                remove_count = min(len(idle_actors), current_actors - self.min_actors)
                self._scale_down(remove_count)

    def _scale_up(self, count: int):
        """Add new actors to the pool."""
        logger.info(f"Scaling up: adding {count} new actors")
        for _ in range(count):
            self._create_and_register_actor()
        logger.info(f"Pool size now: {len(self.actors)} actors")

    def _scale_down(self, count: int):
        """Remove idle actors from the pool."""
        with self.lock:
            removed = 0
            actors_to_remove = []

            for actor, futures in list(self.actor_futures.items()):
                if removed >= count:
                    break
                if len(futures) == 0 and len(self.actors) > self.min_actors:
                    actors_to_remove.append(actor)
                    removed += 1

            for actor in actors_to_remove:
                self.actors.remove(actor)
                del self.actor_futures[actor]
                ray.kill(actor)

            if removed > 0:
                logger.info(f"Scaled down: removed {removed} actors. Pool size now: {len(self.actors)}")

    def _get_least_loaded_actor(self):
        """Get the actor with the least number of pending tasks."""
        if not self.actors:
            return None

        logger.info("Trying to dispatch the task")
        with self.lock:
            alive_actors = []
            for actor in self.actors:
                ping_received = actor.ping.remote()
                logger.info(f"Pinging actor: {actor}")
                ping_received_ready, _ = ray.wait([ping_received], num_returns=1, timeout=1.0)

                # ping_received = ray.get(ping_received, timeout=self.actor_health_check_timeout)
                logger.debug("Ping wait result", extra={"actor": str(actor), "ready": bool(ping_received_ready)})
                if ping_received_ready:
                    try:
                        result = ray.get(ping_received_ready, timeout=5.0)
                        if result:
                            logger.info(f"Ping received from actor: {actor}")
                            alive_actors.append(actor)
                    except Exception as e:
                        logger.error(f"Error received trying to ping actor: {e}")

            self.actors = alive_actors
            alive_actor_futures = {}
            for actor, futures in self.actor_futures.items():
                alive_actor_futures[actor] = futures

            self.actor_futures = alive_actor_futures
            available_actors = [
                actor
                for actor in self.actors
                if self.max_tasks_per_actor is None or len(self.actor_futures.get(actor, [])) < self.max_tasks_per_actor
            ]

        if not available_actors:
            return None

        return min(available_actors, key=lambda a: len(self.actor_futures.get(a, [])))

    def _dispatch_task(self, task: list[dict[str, Any]]) -> ray.ObjectRef:
        """Dispatch a single task to an available actor."""

        actor = self._get_least_loaded_actor()
        if actor is None:
            raise RuntimeError("No actors available")

        with self.lock:
            current_load = len(self.actor_futures.get(actor, []))
            if self.max_tasks_per_actor is not None and current_load >= self.max_tasks_per_actor:
                raise RuntimeError("Actor saturated")

            future = actor.__call__.remote(task)
            self.actor_futures[actor] = [*self.actor_futures.get(actor, []), future]
            self.future_to_actor[future] = actor
            self.future_to_task[future] = task
            logger.info(f"Assigning to actor: {actor}")
            return future

    def _cleanup_completed_futures(self):
        """Remove completed futures from tracking."""
        with self.lock:
            for actor in self.actors:
                if actor in self.actor_futures:
                    # Filter out completed futures
                    pending = []
                    for future in self.actor_futures[actor]:
                        ready, _ = ray.wait([future], timeout=0)
                        if not ready:
                            pending.append(future)
                        else:
                            self.future_to_actor.pop(future, None)
                            self.future_to_task.pop(future, None)
                    self.actor_futures[actor] = pending

    def get_pool_stats(self) -> dict[str, Any]:
        """Get current pool statistics."""
        with self.lock:
            active_tasks = sum(len(futures) for futures in self.actor_futures.values())

            stats = {
                "num_actors": len(self.actors),
                "min_actors": self.min_actors,
                "max_actors": self.max_actors,
                "max_tasks_per_actor": self.max_tasks_per_actor,
                "active_tasks": active_tasks,
                "pending_tasks": self.task_queue.qsize() if self.task_queue is not None else 0,
                "total_submitted": self.total_submitted,
                "total_processed": self.total_processed,
                "actor_loads": {f"actor_{i}": len(futures) for i, futures in enumerate(self.actor_futures.values())},
            }
            return stats

    def shutdown(self):
        """Shutdown the actor pool and clean up resources."""
        logger.info("Shutting down actor pool...")

        # Stop the autoscaling monitor thread
        if hasattr(self, "_monitor_stop_event"):
            self._monitor_stop_event.set()
            if hasattr(self, "_monitor_thread"):
                self._monitor_thread.join(timeout=5)

        # Stop dispatcher thread
        if hasattr(self, "_dispatch_stop_event"):
            self._dispatch_stop_event.set()
            if hasattr(self, "_dispatch_thread"):
                self._dispatch_thread.join(timeout=5)

        if hasattr(self, "_health_stop_event"):
            self._health_stop_event.set()
            if hasattr(self, "_health_thread"):
                self._health_thread.join(timeout=5)

        # Kill all actors
        with self.lock:
            for actor in self.actors:
                ray.kill(actor)
            self.actors.clear()
            self.actor_futures.clear()

        logger.info("Actor pool shutdown complete")
