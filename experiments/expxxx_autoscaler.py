import ray
import time
import asyncio
from typing import List, Dict, Any, Optional
from collections import deque
from threading import Lock, Thread, Event
import logging
from vllm import LLM
from ray.util.queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Classifier:
    """
    Example Classifier actor that processes prompts.
    Replace this with your actual classifier implementation.
    """
    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.processed_count = 0
        # Simulate model initialization
        time.sleep(0.5)
        logger.info(f"Classifier initialized with model: {model_name}")
        self.llm = LLM(model="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4", max_model_len=1024, enforce_eager=True)
    
    def __call__(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a single prompt dictionary.
        
        Args:
            prompts: List of dictionaries containing the prompt data
            
        Returns:
            Dictionary with classification results
        """
        # Simulate processing time
        time.sleep(5)
        self.processed_count += len(prompts)
        
        # Example processing - replace with actual classification logic
        result = {
            "input": prompts,
            "classification": f"processed_{prompts[0].get('id', 'unknown')}",
            "confidence": [0.95] * len(prompts),
            "model": [self.model_name] * len(prompts),
            "processed_count": [self.processed_count] * len(prompts)
        }
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from this classifier instance."""
        return {
            "model_name": self.model_name,
            "processed_count": self.processed_count
        }

    def ping(self) -> bool:
        """Lightweight readiness probe."""
        return True


class AutoscalingActorPool:
    """
    Autoscaling actor pool that manages Classifier actors and distributes tasks.
    """
    NUM_ACTORS_TO_SCALE_UP = 1
    
    def __init__(
        self,
        actor_class=Classifier,
        min_actors: int = 1,
        max_actors: int = 10,
        target_queue_size: int = 5,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
        scale_check_interval: float = 2,
        actor_kwargs: Optional[Dict] = None,
        actor_options: Optional[Dict] = None,
        task_queue=None,
        result_queue=None,
    ):
        """
        Initialize the autoscaling actor pool.
        
        Args:
            actor_class: The Ray actor class to use (default: Classifier)
            min_actors: Minimum number of actors to maintain
            max_actors: Maximum number of actors allowed
            target_queue_size: Target queue size per actor for scaling decisions
            scale_up_threshold: Queue utilization threshold to trigger scale up
            scale_down_threshold: Queue utilization threshold to trigger scale down
            scale_check_interval: Interval in seconds between scaling checks
            actor_kwargs: Additional keyword arguments to pass to actor initialization
        """
        self.actor_class = ray.remote(actor_class)
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
        actor = self.actor_class.options(**self.actor_options).remote(**self.actor_kwargs)
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
    
    def process_batch(self, tasks: List[Dict[str, Any]], timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of tasks and return results in order.
        
        Args:
            tasks: List of task dictionaries to process
            timeout: Optional timeout in seconds for processing all tasks
            
        Returns:
            List of results in the same order as input tasks
        """
        if not tasks:
            return []
        
        self.total_submitted += len(tasks)
        logger.info(f"Processing batch of {len(tasks)} tasks")
        
        # Dispatch all tasks
        # futures = []
        # for task in tasks:
        #     future = self._dispatch_task(task)
        #     futures.append(future)

        # actor = self._get_least_loaded_actor()
        
        # Wait for all results
        try:
            # if timeout:
            #     ready, not_ready = ray.wait(futures, num_returns=len(futures), timeout=timeout)
            #     if not_ready:
            #         logger.warning(f"Timeout: {len(not_ready)} tasks did not complete")
            #         # Cancel pending tasks
            #         for future in not_ready:
            #             ray.cancel(future)
            #     results = ray.get(ready)
            # else:
            # print(f"Futures: {len(futures)}")
            # future = actor.__call__.remote(tasks)
            # results = ray.get(future)

            future = self._dispatch_task(tasks)
            results = ray.get(future)
            
            self.total_processed += len(results)
            
            # Cleanup completed futures
            self._cleanup_completed_futures()
            
            # For timeout case, we need to handle partial results
            # if timeout and not_ready:
            #     # Create a mapping of future to index
            #     future_to_idx = {f: i for i, f in enumerate(futures)}
            #     ordered_results = [None] * len(tasks)
            #     for future in ready:
            #         idx = future_to_idx[future]
            #         ordered_results[idx] = ray.get(future)
            #     return ordered_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    async def process_batch_async(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Asynchronously process a batch of tasks.
        
        Args:
            tasks: List of task dictionaries to process
            
        Returns:
            List of results in the same order as input tasks
        """
        if not tasks:
            return []
        
        self.total_submitted += len(tasks)
        logger.info(f"Processing batch of {len(tasks)} tasks asynchronously")
        
        # Dispatch all tasks
        futures = []
        for task in tasks:
            future = self._dispatch_task(task)
            futures.append(future)
        
        # Wait for all results asynchronously
        results = await asyncio.gather(*[asyncio.create_task(f) for f in futures])
        
        self.total_processed += len(results)
        self._cleanup_completed_futures()
        
        return results
    
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

@ray.remote
def run_autoscaler_exp():
    task_q = Queue()
    result_q = Queue()

    pool = AutoscalingActorPool(
        actor_class=Classifier,
        min_actors=1,
        max_actors=4,
        target_queue_size=5,
        scale_up_threshold=0.7,
        scale_down_threshold=0.3,
        actor_kwargs={"model_name": "bert-base"},
        actor_options={"resources": {"TPU": 1}},
        task_queue=task_q,
        result_queue=result_q,
    )
    
    # Example: Process a batch of tasks

    num_tasks = 10
    task = [
        {"id": i, "prompt": f"Classify this text {i}", "data": f"Sample text {i}"}
        for i in range(5)
    ]
    
    # Enqueue tasks and stream results as they complete
    for _ in range(num_tasks):
        task_q.put(task)
    collected = 0
    while collected < num_tasks:
        try:
            res = result_q.get(timeout=600)
        except Exception:
            print("Timed out waiting for results")
            # break

        if res is None:
            continue
        # Print a small sample of the result
        print({k: (v[:2] if isinstance(v, list) else v) for k, v in res.items()})
        # Each result corresponds to a list of prompts processed
        collected += 1
    
    # Get pool statistics
    stats = pool.get_pool_stats()
    print(f"\nPool Statistics: {stats}")
    
    # Simulate varying load
    print("\nSimulating varying load...")

    num_repeats = 20
    for batch_num in range(3):
        batch_size = 20 * (batch_num + 1)
        task = [
            {"id": f"batch{batch_num}_item{i}", "prompt": f"Text {i}"}
            for i in range(batch_size)
        ]
        for _ in range(num_repeats):
            task_q.put(task)

        # Drain as many results as arrive within a short window
        batch_collected = 0
        # deadline = time.time() + 30
        while batch_collected < num_repeats:
            try:
                res = result_q.get(timeout=60)
                batch_collected += 1
            except Exception:
                continue
        stats = pool.get_pool_stats()
        print(f"Batch {batch_num}: Completed ~{batch_collected} tasks, Actors: {stats['num_actors']}")
        time.sleep(1)
    
    # Shutdown
    print(pool.get_pool_stats())
    pool.shutdown()

# Example usage
if __name__ == "__main__":
    # Initialize Ray
    # ray.init(ignore_reinit_error=True)
    
    # Create the autoscaling pool
    # ray.shutdown()
    ray.get(run_autoscaler_exp.remote())