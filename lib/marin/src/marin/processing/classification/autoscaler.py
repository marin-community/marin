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
from dataclasses import dataclass
from ray.exceptions import RayActorError
from ray.util.queue import Queue

from marin.processing.classification.classifier import BaseClassifier

logger = logging.getLogger("ray")


@dataclass
class AutoscalingActorPoolConfig:
    """Config for the autoscaling actor pool."""

    min_actors: int
    max_actors: int
    scale_up_threshold: float
    scale_down_threshold: float
    scale_check_interval: float
    actor_kwargs: dict | None
    actor_options: dict | None


DEFAULT_AUTOSCALING_ACTOR_POOL_CONFIG = AutoscalingActorPoolConfig(
    min_actors=1,
    max_actors=1,  # No autoscaling since min=max
    scale_up_threshold=0.8,
    scale_down_threshold=0.2,
    scale_check_interval=1.0,
    actor_kwargs={},
    actor_options={},
)


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
        autoscaler_config: AutoscalingActorPoolConfig,
    ):
        """
        The autoscaling actor pool is repsonsible for creating, managing and scaling actors to handle tasks.
        We can think of the autoscaling actor pool as essentially the coordinator in the "MapReduce" pattern.
        It is responsible for:
        1. Accepting tasks from the client (e.g. inference.py) and dispatching these tasks
        to the available actors.
        2. Scaling up and down the number of actors based on the load.
        3. Collecting results from the actors and returning them to the client.
        4. Requeuing tasks to available actors when an actor fails.

        To communicate between the client and the autoscaling actor pool, we use a task queue and a result queue.
        To detect actor failures, we periodically ping the actors and remove the actors if we do not receive a response.
        Furthermore, if the coordinator is ray.wait'ing on a future that had an actor failure, the subsequent ray.get on
        that task will fail with ray.exceptions.ActorDiedError,
        which prompts this autoscaler to requeue the task to a new actor.

        There are three different threads that are responsible for these tasks:
        1. Autoscaling monitor thread: This thread is responsible for monitoring the load
        and scaling the number of actors.
        2. Dispatcher thread: This thread is responsible for dispatching tasks to the available actors.
        3. Result collector thread: This thread is responsible for collecting results from the actors
        and returning them to the client.

        Args:
            actor_class: The Ray actor class to use. This needs to be a subclass of AutoClassifierRayActor
            or a class that implements a ping method so that the autoscaler can check if the actor is alive.
            min_actors: Minimum number of actors to maintain
            max_actors: Maximum number of actors allowed
            scale_up_threshold: Queue utilization threshold to trigger scale up
            scale_down_threshold: Queue utilization threshold to trigger scale down
            scale_check_interval: Interval in seconds between scaling checks
            actor_kwargs: Additional keyword arguments to pass to actor initialization
            actor_health_check_interval: Interval between actor health probes
        """
        self.actor_class = actor_class
        self.model_name_or_path = model_name_or_path
        self.attribute_name = attribute_name
        self.model_type = model_type
        self.min_actors = autoscaler_config.min_actors
        self.max_actors = autoscaler_config.max_actors
        self.scale_up_threshold = autoscaler_config.scale_up_threshold
        self.scale_down_threshold = autoscaler_config.scale_down_threshold
        self.scale_check_interval = autoscaler_config.scale_check_interval
        self.actor_kwargs = autoscaler_config.actor_kwargs or {}
        self.task_queue = task_queue
        self.result_queue = result_queue

        # Actor management
        self.actors = []
        self.actor_futures = {}  # Track ongoing tasks per actor
        self.future_to_actor = {}
        self.future_to_task = {}
        self.actor_options = autoscaler_config.actor_options or {}

        self.actor_task_metadata_lock = Lock()
        """Lock to protect the actor and task metadata mappings
        (e.g. actors, actor_futures, future_to_actor, future_to_task)"""

        # Statistics
        self.total_processed = 0
        self.total_submitted = 0

        # Initialize minimum number of actors
        self._initialize_actors()

        # Start autoscaling monitor, dispatcher, and result collector
        self._monitor_stop_event: Event = Event()
        self._monitor_thread: Thread = Thread(target=self._autoscaling_monitor_loop, daemon=True)
        self._monitor_thread.start()

        self._dispatch_stop_event: Event = Event()
        self._dispatch_thread: Thread = Thread(target=self._dispatcher_loop, daemon=True)
        self._dispatch_thread.start()

        self._result_stop_event: Event = Event()
        self._result_thread: Thread = Thread(target=self._result_collector_loop, daemon=True)
        self._result_thread.start()

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

    def _autoscaling_monitor_loop(self):
        """Start a background thread that monitors the load and scales the number of actors accordingly."""
        while not self._monitor_stop_event.is_set():
            try:
                time.sleep(self.scale_check_interval)
                self._check_and_scale()
            except Exception:
                # Best-effort; do not crash the monitor
                pass

    def _dispatcher_loop(self):
        """Start a background dispatcher that pulls tasks and dispatches work."""
        while not self._dispatch_stop_event.is_set():
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
                logger.info("No tasks dispatched - going to sleep for 1 second.")
                time.sleep(1)

    def _result_collector_loop(self):
        """Start a background thread that collects completed futures."""

        while not self._result_stop_event.is_set():
            futures_snapshot = []
            with self.actor_task_metadata_lock:
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
            with self.actor_task_metadata_lock:
                task_to_retry = self.future_to_task.get(future)
            if task_to_retry is not None and self.task_queue is not None:
                try:
                    self.task_queue.put(task_to_retry)
                except Exception:
                    logger.error("Failed to requeue task after actor failure")
        except Exception:
            logger.exception("Error retrieving future result")
        finally:
            with self.actor_task_metadata_lock:
                actor = self.future_to_actor.pop(future, None)
                self.future_to_task.pop(future, None)
                if actor is not None and actor in self.actor_futures:
                    try:
                        self.actor_futures[actor].remove(future)
                    except ValueError:
                        logger.error("Failed to remove future from actor futures!")

    def _check_and_scale(self):
        """Check current load and scale actors accordingly."""
        with self.actor_task_metadata_lock:
            current_actors = len(self.actors)
            try:
                pending_count = int(self.task_queue.qsize())
            except Exception:
                pending_count = 0

            # Calculate total active tasks
            active_tasks = sum(len(futures) for futures in self.actor_futures.values())
        total_load = pending_count + active_tasks

        utilization = total_load / current_actors if current_actors > 0 else total_load

        logger.info(
            f"Load check - Actors: {current_actors}, Pending: {pending_count}, "
            f"Active: {active_tasks}, Utilization: {utilization:.2%}"
        )

        # Scale up if needed
        if (
            utilization > self.scale_up_threshold and current_actors < self.max_actors
        ) or current_actors < self.min_actors:
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
        with self.actor_task_metadata_lock:
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

        logger.debug("Trying to dispatch the task")
        with self.actor_task_metadata_lock:
            alive_actors = []
            for actor in self.actors:
                ping_received = actor.ping.remote()
                logger.debug(f"Pinging actor: {actor}")
                ping_received_ready, _ = ray.wait([ping_received], num_returns=1, timeout=1.0)

                logger.debug("Ping wait result", extra={"actor": str(actor), "ready": bool(ping_received_ready)})
                if ping_received_ready:
                    try:
                        result = ray.get(ping_received_ready, timeout=5.0)
                        if result:
                            logger.debug(f"Ping received from actor: {actor}")
                            alive_actors.append(actor)
                    except Exception as e:
                        logger.error(f"Error received trying to ping actor: {e}")

            self.actors = alive_actors
            alive_actor_futures = {}
            for actor, futures in self.actor_futures.items():
                alive_actor_futures[actor] = futures

            self.actor_futures = alive_actor_futures
            available_actors = [actor for actor, futures in self.actor_futures.items() if len(futures) == 0]

        if not available_actors:
            return None

        return min(available_actors, key=lambda a: len(self.actor_futures.get(a, [])))

    def _dispatch_task(self, task: list[dict[str, Any]]) -> ray.ObjectRef:
        """Dispatch a single task to an available actor."""

        actor = self._get_least_loaded_actor()
        if actor is None:
            raise RuntimeError("No actors available")

        with self.actor_task_metadata_lock:
            current_load = len(self.actor_futures.get(actor, []))
            if current_load >= 1:  # Don't assign to actor since it already has a task
                raise RuntimeError("Actor saturated")

            future = actor.__call__.remote(task)
            self.actor_futures[actor] = [*self.actor_futures.get(actor, []), future]
            self.future_to_actor[future] = actor
            self.future_to_task[future] = task
            logger.debug(f"Assigning to actor: {actor} for task: {task}")
            return future

    def _cleanup_completed_futures(self):
        """Remove completed futures from tracking."""
        with self.actor_task_metadata_lock:
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

    def shutdown(self):
        """Shutdown the actor pool and clean up resources."""
        logger.info("Shutting down actor pool...")

        # Stop the autoscaling monitor thread
        self._monitor_stop_event.set()
        self._monitor_thread.join(timeout=5)

        # Stop dispatcher thread
        self._dispatch_stop_event.set()
        self._dispatch_thread.join(timeout=5)

        self._result_stop_event.set()
        self._result_thread.join(timeout=5)

        # Kill all actors
        with self.actor_task_metadata_lock:
            for actor in self.actors:
                ray.kill(actor)
            self.actors.clear()
            self.actor_futures.clear()

        logger.info("Actor pool shutdown complete")
