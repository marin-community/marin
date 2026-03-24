# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import queue
import time
import uuid
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

from fray.v2 import ActorFuture, ActorHandle, ResourceConfig, current_client

from marin.processing.classification.classifier import BaseClassifier

logger = logging.getLogger(__name__)


@dataclass
class AutoscalingActorPoolConfig:
    """Config for the autoscaling actor pool."""

    min_actors: int = 1
    max_actors: int = 1
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.2
    scale_check_interval: float = 1.0
    actor_kwargs: dict | None = None
    actor_resources: ResourceConfig = field(default_factory=ResourceConfig)


DEFAULT_AUTOSCALING_ACTOR_POOL_CONFIG = AutoscalingActorPoolConfig()


class AutoscalingActorPool:
    """Autoscaling actor pool that manages BaseClassifier actors via Fray and distributes tasks.

    Coordinates batch classification work across a pool of Fray actors. Responsible for:
    1. Accepting tasks from the client (e.g. inference.py) and dispatching to available actors.
    2. Scaling up and down the number of actors based on load.
    3. Collecting results from actors and returning them to the client.
    4. Requeuing tasks to available actors when an actor fails.
    """

    NUM_ACTORS_TO_SCALE_UP = 1

    def __init__(
        self,
        actor_class: type[BaseClassifier],
        model_name_or_path: str,
        attribute_name: str,
        model_type: str,
        task_queue: queue.Queue,
        result_queue: queue.Queue,
        autoscaler_config: AutoscalingActorPoolConfig,
    ):
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
        self.actor_resources = autoscaler_config.actor_resources
        self.task_queue = task_queue
        self.result_queue = result_queue

        self.client = current_client()
        self._pool_id = uuid.uuid4().hex[:8]
        self._actor_counter = 0

        # Actor management: keyed by actor handle identity
        self.actors: list[ActorHandle] = []
        self.actor_futures: dict[int, list[ActorFuture]] = {}
        self.future_to_actor: dict[int, ActorHandle] = {}
        self.future_to_task: dict[int, Any] = {}

        self.actor_task_metadata_lock = Lock()

        # Statistics
        self.total_processed = 0
        self.total_submitted = 0

        self._initialize_actors()

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
        """Create one Fray actor and wait until it's ready before adding to pool."""
        NUM_RETRIES = 3
        actor = None
        name = f"classifier-{self._pool_id}-{self._actor_counter}"
        self._actor_counter += 1

        for attempt in range(NUM_RETRIES):
            try:
                actor = self.client.create_actor(
                    self.actor_class,
                    self.model_name_or_path,
                    self.attribute_name,
                    self.model_type,
                    name=name if attempt == 0 else f"{name}-retry{attempt}",
                    resources=self.actor_resources,
                    **self.actor_kwargs,
                )
                break
            except Exception as e:
                logger.warning(f"Actor creation failed (attempt {attempt + 1}): {e}")
                time.sleep(1)

        if actor is None:
            raise RuntimeError(f"Failed to create actor after {NUM_RETRIES} attempts")

        try:
            actor.ping.remote().result(timeout=30)
        except Exception as e:
            logger.warning(f"Actor readiness check failed: {e}")

        self.actors.append(actor)
        self.actor_futures[id(actor)] = []

    def _autoscaling_monitor_loop(self):
        """Background thread that monitors load and scales actors."""
        while not self._monitor_stop_event.is_set():
            try:
                time.sleep(self.scale_check_interval)
                self._check_and_scale()
            except Exception:
                pass

    def _dispatcher_loop(self):
        """Background dispatcher that pulls tasks and dispatches work."""
        while not self._dispatch_stop_event.is_set():
            if len(self.actors) == 0:
                logger.debug("Dispatcher waiting for actors to come online")
                time.sleep(0.5)
                continue

            dispatched_any = False
            for _ in range(4):
                try:
                    task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    break

                self.total_submitted += 1
                try:
                    self._dispatch_task(task)
                    dispatched_any = True
                except RuntimeError:
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
        """Background thread that collects completed futures."""
        while not self._result_stop_event.is_set():
            with self.actor_task_metadata_lock:
                futures_snapshot = list(self.future_to_actor.keys())

            if not futures_snapshot:
                time.sleep(0.05)
                continue

            completed_any = False
            for future_id in futures_snapshot:
                with self.actor_task_metadata_lock:
                    # Look up the actual future object from our tracking
                    future_obj = None
                    for _actor_id, futures in self.actor_futures.items():
                        for f in futures:
                            if id(f) == future_id:
                                future_obj = f
                                break
                        if future_obj is not None:
                            break

                if future_obj is None:
                    continue

                try:
                    result = future_obj.result(timeout=0)
                    self._handle_success(future_id, result)
                    completed_any = True
                except TimeoutError:
                    pass
                except Exception:
                    self._handle_failure(future_id)
                    completed_any = True

            if not completed_any:
                time.sleep(0.05)

    def _handle_success(self, future_id: int, result: Any) -> None:
        self.total_processed += 1
        if self.result_queue is not None:
            try:
                self.result_queue.put(result)
            except Exception as e:
                logger.error(f"Failed to put result in results queue! {e}")
        self._cleanup_future(future_id)

    def _handle_failure(self, future_id: int) -> None:
        task_to_retry = None
        with self.actor_task_metadata_lock:
            task_to_retry = self.future_to_task.get(future_id)
        if task_to_retry is not None and self.task_queue is not None:
            try:
                self.task_queue.put(task_to_retry)
            except Exception:
                logger.error("Failed to requeue task after actor failure")
        self._cleanup_future(future_id)

    def _cleanup_future(self, future_id: int) -> None:
        with self.actor_task_metadata_lock:
            actor = self.future_to_actor.pop(future_id, None)
            self.future_to_task.pop(future_id, None)
            if actor is not None and id(actor) in self.actor_futures:
                self.actor_futures[id(actor)] = [f for f in self.actor_futures[id(actor)] if id(f) != future_id]

    def _check_and_scale(self):
        """Check current load and scale actors accordingly."""
        with self.actor_task_metadata_lock:
            current_actors = len(self.actors)
            try:
                pending_count = self.task_queue.qsize()
            except Exception:
                pending_count = 0

            active_tasks = sum(len(futures) for futures in self.actor_futures.values())
        total_load = pending_count + active_tasks

        utilization = total_load / current_actors if current_actors > 0 else total_load

        logger.info(
            f"Load check - Actors: {current_actors}, Pending: {pending_count}, "
            f"Active: {active_tasks}, Utilization: {utilization:.2%}"
        )

        if (
            utilization > self.scale_up_threshold and current_actors < self.max_actors
        ) or current_actors < self.min_actors:
            self._scale_up(self.NUM_ACTORS_TO_SCALE_UP)

        elif utilization < self.scale_down_threshold and current_actors > self.min_actors:
            idle_actors = [actor for actor in self.actors if len(self.actor_futures.get(id(actor), [])) == 0]
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

            for actor in self.actors:
                if removed >= count:
                    break
                if len(self.actor_futures.get(id(actor), [])) == 0 and len(self.actors) > self.min_actors:
                    actors_to_remove.append(actor)
                    removed += 1

            for actor in actors_to_remove:
                self.actors.remove(actor)
                self.actor_futures.pop(id(actor), None)

            if removed > 0:
                logger.info(f"Scaled down: removed {removed} actors. Pool size now: {len(self.actors)}")

    def _get_least_loaded_actor(self) -> ActorHandle | None:
        """Get the actor with the least number of pending tasks."""
        if not self.actors:
            return None

        logger.debug("Trying to dispatch the task")
        with self.actor_task_metadata_lock:
            alive_actors = []
            for actor in self.actors:
                try:
                    result = actor.ping.remote().result(timeout=5.0)
                    if result:
                        logger.debug("Ping received from actor")
                        alive_actors.append(actor)
                except Exception as e:
                    logger.error(f"Error received trying to ping actor: {e}")

            self.actors = alive_actors
            self.actor_futures = {id(actor): self.actor_futures.get(id(actor), []) for actor in alive_actors}
            available_actors = [actor for actor in self.actors if len(self.actor_futures.get(id(actor), [])) == 0]

        if not available_actors:
            return None

        return min(available_actors, key=lambda a: len(self.actor_futures.get(id(a), [])))

    def _dispatch_task(self, task: list[dict[str, Any]]) -> ActorFuture:
        """Dispatch a single task to an available actor."""
        actor = self._get_least_loaded_actor()
        if actor is None:
            raise RuntimeError("No actors available")

        with self.actor_task_metadata_lock:
            current_load = len(self.actor_futures.get(id(actor), []))
            if current_load >= 1:
                raise RuntimeError("Actor saturated")

            future = actor.classify.remote(task)
            self.actor_futures[id(actor)] = [*self.actor_futures.get(id(actor), []), future]
            self.future_to_actor[id(future)] = actor
            self.future_to_task[id(future)] = task
            logger.debug("Assigning task to actor")
            return future

    def shutdown(self):
        """Shutdown the actor pool and clean up resources."""
        logger.info("Shutting down actor pool...")

        self._monitor_stop_event.set()
        self._monitor_thread.join(timeout=5)

        self._dispatch_stop_event.set()
        self._dispatch_thread.join(timeout=5)

        self._result_stop_event.set()
        self._result_thread.join(timeout=5)

        with self.actor_task_metadata_lock:
            self.actors.clear()
            self.actor_futures.clear()

        logger.info("Actor pool shutdown complete")
