import logging
from typing import Any, Dict, List, Optional
import time
import asyncio
import ray
from threading import Thread, Lock
from multiprocessing import Event
from ray.util.queue import Queue

from marin.processing.classification.classifier import BaseClassifier

logger = logging.getLogger(__name__)


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
        max_actors: int = 8,
        target_queue_size: int = 5,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
        scale_check_interval: float = 2,
        actor_kwargs: Optional[Dict] = None,
        actor_options: Optional[Dict] = None,
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
        """
        self.actor_class = ray.remote(actor_class)
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
        self.actor_options = actor_options or {}
        self.lock = Lock()
        
        # Statistics
        self.total_processed = 0
        self.total_submitted = 0
        
        # Initialize minimum number of actors
        self._initialize_actors()
        
        # Start autoscaling monitor and dispatcher
        self.scaling_task = None
        self._start_autoscaling_monitor()
        self._start_dispatcher()
    
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
                actor = self.actor_class.options(**self.actor_options).remote(self.model_name_or_path, self.attribute_name, self.model_type, **self.actor_kwargs)
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
        """Start a background dispatcher that pulls tasks and pushes results."""
        self._dispatch_stop_event: Event = Event()

        def _loop():
            while not self._dispatch_stop_event.is_set():
                # Pull tasks from queue (non-blocking) and dispatch
                dispatched_any = False
                futures_snapshot = []
                if self.task_queue is not None:
                    for _ in range(64):  # cap per-iteration dispatch volume
                        try:
                            task = self.task_queue.get(timeout=30)
                            self.total_submitted += 1
                        except Exception:
                            break
                        try:
                            # Dispatch single-item as list to classifier
                            actor = self._get_least_loaded_actor()
                            if actor is None:
                                # No actors yet; requeue and break
                                try:
                                    self.task_queue.put(task)
                                except Exception:
                                    pass
                                break

                            # Found an actor, dispatch the task
                            self._dispatch_task(task)
                            dispatched_any = True
                        except Exception:
                            # On dispatch failure, drop or requeue
                            try:
                                self.task_queue.put(task)
                            except Exception:
                                logger.error(f"Failed to requeue task: {task}")
                            break

                # Currently pending futures
                futures_snapshot = list(self.future_to_actor.keys())
                # Collect completed results
                if futures_snapshot:
                    try:
                        ready, _ = ray.wait(futures_snapshot, num_returns=len(futures_snapshot), timeout=0)
                    except Exception:
                        ready = []
                    for fut in ready:
                        try:
                            result = ray.get(fut)
                            self.total_processed += 1
                            if self.result_queue is not None:
                                try:
                                    # print(f"Putting result in results queue: {result}")
                                    self.result_queue.put(result)
                                except Exception:
                                    logger.error(f"Failed to put result in results queue!")
                        except Exception:
                            # Result failed; ignore or log
                            result = None
                        finally:
                            with self.lock:
                                actor = self.future_to_actor.pop(fut, None)
                                if actor is not None and actor in self.actor_futures:
                                    try:
                                        self.actor_futures[actor].remove(fut)
                                    except ValueError:
                                        logger.error(f"Failed to remove future from actor futures!")

                if not dispatched_any and not futures_snapshot:
                    time.sleep(0.05)

        self._dispatch_thread: Thread = Thread(target=_loop, daemon=True)
        self._dispatch_thread.start()

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
        capacity = current_actors * self.target_queue_size
        utilization = total_load / capacity if capacity > 0 else 0
        
        logger.info(f"Load check - Actors: {current_actors}, Pending: {pending_count}, "
                    f"Active: {active_tasks}, Utilization: {utilization:.2%}")
        
        # Scale up if needed
        if utilization > self.scale_up_threshold and current_actors < self.max_actors:

            # If a system quickly overwhelms the scale up threshold, we end up having to scale up too many actors at once.
            # Let's just scale up one at a time
            # new_actors_count = min(
            #     self.max_actors - current_actors,
            #     max(1, (total_load // self.target_queue_size) - current_actors)
            # )
            self._scale_up(self.NUM_ACTORS_TO_SCALE_UP)
        
        # Scale down if needed
        elif utilization < self.scale_down_threshold and current_actors > self.min_actors:
            # Only scale down idle actors
            idle_actors = [
                actor for actor, futures in self.actor_futures.items()
                if len(futures) == 0
            ]
            if idle_actors:
                remove_count = min(
                    len(idle_actors),
                    current_actors - self.min_actors
                )
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
        
        return min(self.actors, key=lambda a: len(self.actor_futures.get(a, [])))
    
    def _dispatch_task(self, task: List[Dict[str, Any]]) -> ray.ObjectRef:
        """Dispatch a single task to an available actor."""

        with self.lock:
            actor = self._get_least_loaded_actor()
            if actor:
                future = actor.__call__.remote(task)
                self.actor_futures[actor] = self.actor_futures.get(actor, []) + [future]
                self.future_to_actor[future] = actor
                return future
            else:
                # This shouldn't happen if min_actors > 0
                raise RuntimeError("No actors available")
    
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
                    self.actor_futures[actor] = pending
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        with self.lock:
            active_tasks = sum(len(futures) for futures in self.actor_futures.values())
            
            stats = {
                "num_actors": len(self.actors),
                "min_actors": self.min_actors,
                "max_actors": self.max_actors,
                "active_tasks": active_tasks,
                "pending_tasks": self.task_queue.qsize() if self.task_queue is not None else 0,
                "total_submitted": self.total_submitted,
                "total_processed": self.total_processed,
                "actor_loads": {
                    f"actor_{i}": len(futures)
                    for i, futures in enumerate(self.actor_futures.values())
                }
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
        
        # Kill all actors
        with self.lock:
            for actor in self.actors:
                ray.kill(actor)
            self.actors.clear()
            self.actor_futures.clear()
        
        logger.info("Actor pool shutdown complete")