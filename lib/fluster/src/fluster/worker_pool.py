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

"""WorkerPool for task dispatch.

WorkerPool provides a high-level interface for dispatching arbitrary callables
to a pool of stateless workers. Unlike ActorPool (which load-balances calls to
pre-existing actors with known methods), WorkerPool creates and manages worker
jobs that can execute any callable.

Example:
    from pathlib import Path
    from fluster.worker_pool import WorkerPool, WorkerPoolConfig
    from fluster.cluster.client import RpcClusterClient

    client = RpcClusterClient("http://controller:8080", workspace=Path("./my-project"))

    config = WorkerPoolConfig(
        num_workers=3,
        resources=cluster_pb2.ResourceSpec(cpu=1, memory="512m"),
    )

    with WorkerPool(client, config) as pool:
        # Submit tasks
        futures = [pool.submit(expensive_fn, i) for i in range(10)]
        results = [f.result() for f in futures]
"""

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from queue import Empty, Queue
from typing import Any, Generic, TypeVar

import cloudpickle

from fluster.rpc import actor_pb2, cluster_pb2
from fluster.actor import ActorServer
from fluster.actor.resolver import Resolver
from fluster.rpc.actor_connect import ActorServiceClientSync
from fluster.cluster.client import ClusterClient
from fluster.cluster.types import Entrypoint, JobId
from fluster.context import fluster_ctx

T = TypeVar("T")


class WorkerStatus(Enum):
    """Status of a worker in the pool."""

    PENDING = auto()  # Worker job launched, not yet registered
    IDLE = auto()  # Ready to accept tasks
    BUSY = auto()  # Currently executing a task
    FAILED = auto()  # Worker has failed/disconnected


class UserException(Exception):
    """Wrapper for exceptions raised by user code (not infrastructure failures)."""

    def __init__(self, inner: BaseException):
        self.inner = inner
        super().__init__(str(inner))


@dataclass
class WorkerState:
    """Client-side state for a single worker."""

    worker_id: str
    worker_name: str
    endpoint_url: str | None = None
    status: WorkerStatus = WorkerStatus.PENDING
    current_task_id: str | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0


@dataclass
class PendingTask:
    """A task waiting to be dispatched to a worker."""

    task_id: str
    serialized_fn: bytes
    serialized_args: bytes
    serialized_kwargs: bytes
    future: Future
    fn_name: str
    submitted_at: float
    retries_remaining: int = 0


@dataclass
class PoolStatus:
    """Status snapshot of the worker pool."""

    pool_id: str
    num_workers: int
    workers_idle: int
    workers_busy: int
    workers_pending: int
    workers_failed: int
    tasks_queued: int
    tasks_completed: int
    tasks_failed: int
    worker_details: list[dict]


class TaskExecutorActor:
    """Actor that executes arbitrary callables.

    This is the server-side component of WorkerPool. Each worker job runs
    one of these actors to execute tasks dispatched by the pool.

    The callable and arguments are received as cloudpickle-serialized bytes.
    The return value is returned raw - ActorServer handles serialization.
    """

    def execute(
        self,
        serialized_callable: bytes,
        serialized_args: bytes,
        serialized_kwargs: bytes,
    ) -> Any:
        """Execute a pickled callable and return the result.

        Args:
            serialized_callable: cloudpickle-serialized callable
            serialized_args: cloudpickle-serialized tuple of positional args
            serialized_kwargs: cloudpickle-serialized dict of keyword args

        Returns:
            The return value of calling fn(*args, **kwargs).
            ActorServer handles serialization of this value.

        Raises:
            Any exception raised by the callable (propagated to client).
        """
        fn = cloudpickle.loads(serialized_callable)
        args = cloudpickle.loads(serialized_args)
        kwargs = cloudpickle.loads(serialized_kwargs)
        return fn(*args, **kwargs)


def worker_job_entrypoint(pool_id: str, worker_index: int) -> None:
    """Job entrypoint that starts a TaskExecutor actor with a unique name.

    This function is called when a worker job starts. It:
    1. Gets cluster configuration from FlusterContext
    2. Starts an ActorServer with a TaskExecutorActor
    3. Registers the endpoint with the controller
    4. Runs forever, serving requests

    Each worker registers with a unique name so the client can target
    specific idle workers when dispatching tasks.

    Args:
        pool_id: Unique identifier for the worker pool
        worker_index: Index of this worker (0, 1, 2, ...)
    """
    ctx = fluster_ctx()

    # Unique name per worker
    worker_name = f"_workerpool_{pool_id}:worker-{worker_index}"

    print(f"Worker starting: pool_id={pool_id}, worker_index={worker_index}")
    print(f"Worker name: {worker_name}, job_id={ctx.job_id}")

    # Start actor server
    server = ActorServer(host="0.0.0.0")
    server.register(worker_name, TaskExecutorActor())
    actual_port = server.serve_background()

    # Register endpoint with controller
    address = f"localhost:{actual_port}"
    ctx.controller.endpoint_registry.register(worker_name, address, {"job_id": ctx.job_id})
    print(f"ActorServer started and registered on port {actual_port}")

    # Serve forever
    print("Worker ready, waiting for tasks...")
    while True:
        time.sleep(1)


class WorkerDispatcher:
    """Dispatch thread for a single worker.

    Handles endpoint discovery and task dispatch in a dedicated thread.
    State transitions:
    - PENDING: Poll resolver for endpoint registration
    - IDLE: Wait for task from queue
    - BUSY: Execute task on worker endpoint
    - FAILED: Worker has failed, thread exits

    On infrastructure failure (connection error), the task is re-queued for
    another worker if retries remain. User exceptions propagate immediately.
    """

    def __init__(
        self,
        state: WorkerState,
        task_queue: "Queue[PendingTask]",
        resolver: Resolver,
        timeout: float,
    ):
        self.state = state
        self._task_queue = task_queue
        self._resolver = resolver
        self._timeout = timeout
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the dispatch thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"dispatch-{self.state.worker_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the dispatch thread to stop."""
        self._shutdown.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the dispatch thread to finish."""
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        """Main dispatch loop."""
        while not self._shutdown.is_set():
            if self.state.status == WorkerStatus.PENDING:
                self._discover_endpoint()
                continue

            if self.state.status == WorkerStatus.FAILED:
                break

            task = self._get_task()
            if task:
                self._execute_task(task)

    def _discover_endpoint(self) -> None:
        """Poll resolver for endpoint registration."""
        result = self._resolver.resolve(self.state.worker_name)
        if not result.is_empty:
            endpoint = result.first()
            self.state.endpoint_url = endpoint.url
            self.state.status = WorkerStatus.IDLE
            print(f"Worker {self.state.worker_id} discovered at {endpoint.url}")
        else:
            time.sleep(0.1)

    def _get_task(self) -> PendingTask | None:
        """Try to get a task from the queue."""
        try:
            return self._task_queue.get(timeout=0.5)
        except Empty:
            return None

    def _execute_task(self, task: PendingTask) -> None:
        """Execute a task on the worker endpoint."""
        self.state.status = WorkerStatus.BUSY
        self.state.current_task_id = task.task_id

        try:
            result = _call_worker_endpoint(
                endpoint_url=self.state.endpoint_url,
                actor_name=self.state.worker_name,
                task=task,
                timeout=self._timeout,
            )
            task.future.set_result(result)
            self.state.tasks_completed += 1
        except UserException as e:
            task.future.set_exception(e.inner)
            self.state.tasks_failed += 1
        except Exception as e:
            if task.retries_remaining > 0:
                task.retries_remaining -= 1
                self._task_queue.put(task)
                print(
                    f"Worker {self.state.worker_id} failed, re-queuing task {task.task_id} "
                    f"({task.retries_remaining} retries left)"
                )
                self.state.status = WorkerStatus.FAILED
                self.state.current_task_id = None
                self._task_queue.task_done()
                return
            else:
                task.future.set_exception(e)
                self.state.tasks_failed += 1
        finally:
            if self.state.status == WorkerStatus.BUSY:
                self.state.status = WorkerStatus.IDLE
                self.state.current_task_id = None
                self._task_queue.task_done()


def _call_worker_endpoint(
    endpoint_url: str,
    actor_name: str,
    task: PendingTask,
    timeout: float,
) -> Any:
    """Make a direct RPC call to a specific worker endpoint."""
    client = ActorServiceClientSync(
        address=endpoint_url,
        timeout_ms=int(timeout * 1000),
    )

    call = actor_pb2.ActorCall(
        method_name="execute",
        actor_name=actor_name,
        serialized_args=cloudpickle.dumps(
            (
                task.serialized_fn,
                task.serialized_args,
                task.serialized_kwargs,
            )
        ),
        serialized_kwargs=cloudpickle.dumps({}),
    )

    resp = client.call(call)

    if resp.HasField("error"):
        if resp.error.serialized_exception:
            # User exception - wrap it so we know not to retry
            raise UserException(cloudpickle.loads(resp.error.serialized_exception))
        raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")

    return cloudpickle.loads(resp.serialized_value)


@dataclass
class WorkerPoolConfig:
    """Configuration for a WorkerPool.

    Attributes:
        num_workers: Number of worker jobs to launch
        resources: Resource requirements per worker
        environment: Optional environment configuration
        name_prefix: Prefix for worker job names
        max_retries: Number of retries for failed tasks (worker failures only)
    """

    num_workers: int
    resources: cluster_pb2.ResourceSpec
    environment: cluster_pb2.EnvironmentConfig | None = None
    name_prefix: str = "worker"
    max_retries: int = 0


@dataclass
class WorkerFuture(Generic[T]):
    """Future representing an in-flight task.

    Wraps a concurrent.futures.Future with a simpler interface.
    ActorClient handles cloudpickle deserialization, so result() returns
    the value directly.
    """

    _future: Future
    _fn_name: str

    def result(self, timeout: float | None = None) -> T:
        """Block until result is available.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            The return value of the submitted callable

        Raises:
            TimeoutError: If result not available within timeout
            Exception: Any exception raised by the callable
        """
        return self._future.result(timeout=timeout)

    def done(self) -> bool:
        """Check if the task has completed."""
        return self._future.done()

    def exception(self) -> BaseException | None:
        """Get the exception if the task failed, None otherwise."""
        if not self._future.done():
            return None
        return self._future.exception()


class WorkerPool:
    """Pool of stateless workers for task dispatch with idle worker targeting.

    WorkerPool manages a set of worker jobs that execute arbitrary callables.
    Each worker registers with a unique name, and tasks are dispatched only
    to idle workers. Tasks queue internally when all workers are busy.

    Usage:
        with WorkerPool(client, config) as pool:
            # Submit single task
            future = pool.submit(fn, arg1, arg2)
            result = future.result()

            # Map over items
            futures = pool.map(fn, items)
            results = [f.result() for f in futures]

            # Check pool status
            pool.print_status()
    """

    def __init__(
        self,
        client: ClusterClient,
        config: WorkerPoolConfig,
        timeout: float = 30.0,
        resolver: Resolver | None = None,
    ):
        """Create a worker pool.

        Args:
            client: ClusterClient for launching worker jobs
            config: Pool configuration (workers, resources, etc.)
            timeout: RPC timeout in seconds for worker calls
            resolver: Optional resolver override (for testing)
        """
        self._client = client
        self._config = config
        self._timeout = timeout
        self._pool_id = uuid.uuid4().hex[:8]

        # Worker management
        self._workers: dict[str, WorkerState] = {}
        self._job_ids: list[JobId] = []

        # Task queue and dispatch
        self._task_queue: Queue[PendingTask] = Queue()
        self._dispatchers: list[WorkerDispatcher] = []
        self._shutdown = False

        # Resolver for endpoint discovery (injectable for testing)
        self._resolver: Resolver | None = resolver

    def __enter__(self) -> "WorkerPool":
        """Start workers and wait for at least one to register."""
        self._launch_workers()
        self._wait_for_workers(min_workers=1)
        return self

    def __exit__(self, *_):
        """Shutdown all workers."""
        self.shutdown(wait=False)

    @property
    def pool_id(self) -> str:
        """Unique identifier for this pool."""
        return self._pool_id

    @property
    def size(self) -> int:
        """Number of workers that have registered (IDLE or BUSY)."""
        return sum(1 for w in self._workers.values() if w.status in (WorkerStatus.IDLE, WorkerStatus.BUSY))

    @property
    def idle_count(self) -> int:
        """Number of idle workers ready for tasks."""
        return sum(1 for w in self._workers.values() if w.status == WorkerStatus.IDLE)

    @property
    def job_ids(self) -> list[JobId]:
        """List of worker job IDs."""
        return list(self._job_ids)

    def _launch_workers(self) -> None:
        """Launch worker jobs and start dispatch threads."""
        # Create resolver for worker discovery if not injected
        # The resolver automatically derives namespace from the current FlusterContext
        if self._resolver is None:
            self._resolver = self._client.resolver()

        # Initialize worker state and launch jobs
        for i in range(self._config.num_workers):
            worker_id = f"worker-{i}"
            worker_name = f"_workerpool_{self._pool_id}:{worker_id}"
            self._workers[worker_id] = WorkerState(
                worker_id=worker_id,
                worker_name=worker_name,
                status=WorkerStatus.PENDING,
            )

            entrypoint = Entrypoint(
                callable=worker_job_entrypoint,
                args=(self._pool_id, i),
            )

            job_id = self._client.submit(
                entrypoint=entrypoint,
                name=f"{self._config.name_prefix}-{self._pool_id}-{i}",
                resources=self._config.resources,
                environment=self._config.environment,
                ports=["actor"],
            )
            self._job_ids.append(job_id)

        # Start dispatchers (one per worker)
        for worker_state in self._workers.values():
            dispatcher = WorkerDispatcher(
                state=worker_state,
                task_queue=self._task_queue,
                resolver=self._resolver,
                timeout=self._timeout,
            )
            dispatcher.start()
            self._dispatchers.append(dispatcher)

    def _wait_for_workers(
        self,
        min_workers: int = 1,
        timeout: float = 60.0,
    ) -> None:
        """Wait for workers to register.

        Args:
            min_workers: Minimum number of workers required
            timeout: Maximum time to wait in seconds

        Raises:
            TimeoutError: If min_workers not available within timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.size >= min_workers:
                return
            time.sleep(0.5)

        raise TimeoutError(f"Only {self.size} of {min_workers} workers registered within {timeout}s")

    def wait_for_workers(
        self,
        min_workers: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Wait for workers to become available.

        Args:
            min_workers: Minimum workers required (default: all workers)
            timeout: Maximum time to wait in seconds
        """
        if min_workers is None:
            min_workers = self._config.num_workers
        self._wait_for_workers(min_workers=min_workers, timeout=timeout)

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> WorkerFuture[T]:
        """Submit a task for execution.

        Tasks are queued internally and dispatched to idle workers.
        Returns immediately with a Future that resolves when the task completes.

        Args:
            fn: Callable to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future that resolves to the function's return value
        """
        if self._shutdown:
            raise RuntimeError("WorkerPool has been shutdown")

        if not self._workers:
            raise RuntimeError("No workers available")

        task = PendingTask(
            task_id=uuid.uuid4().hex[:8],
            serialized_fn=cloudpickle.dumps(fn),
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
            future=Future(),
            fn_name=getattr(fn, "__name__", "lambda"),
            submitted_at=time.monotonic(),
            retries_remaining=self._config.max_retries,
        )

        self._task_queue.put(task)
        return WorkerFuture(_future=task.future, _fn_name=task.fn_name)

    def map(
        self,
        fn: Callable[[Any], T],
        items: Sequence[Any],
    ) -> list[WorkerFuture[T]]:
        """Map a function over items in parallel.

        Args:
            fn: Function to apply to each item
            items: Items to process

        Returns:
            List of futures, one per item
        """
        return [self.submit(fn, item) for item in items]

    def status(self) -> PoolStatus:
        """Get current pool status."""
        workers_by_status = {s: 0 for s in WorkerStatus}
        total_completed = 0
        total_failed = 0
        worker_details = []

        for worker in self._workers.values():
            workers_by_status[worker.status] += 1
            total_completed += worker.tasks_completed
            total_failed += worker.tasks_failed
            worker_details.append(
                {
                    "worker_id": worker.worker_id,
                    "worker_name": worker.worker_name,
                    "status": worker.status.name,
                    "endpoint_url": worker.endpoint_url,
                    "current_task_id": worker.current_task_id,
                    "tasks_completed": worker.tasks_completed,
                    "tasks_failed": worker.tasks_failed,
                }
            )

        return PoolStatus(
            pool_id=self._pool_id,
            num_workers=len(self._workers),
            workers_idle=workers_by_status[WorkerStatus.IDLE],
            workers_busy=workers_by_status[WorkerStatus.BUSY],
            workers_pending=workers_by_status[WorkerStatus.PENDING],
            workers_failed=workers_by_status[WorkerStatus.FAILED],
            tasks_queued=self._task_queue.qsize(),
            tasks_completed=total_completed,
            tasks_failed=total_failed,
            worker_details=worker_details,
        )

    def print_status(self) -> None:
        """Print current pool status to stdout."""
        s = self.status()
        print(f"WorkerPool[{s.pool_id}]")
        print(
            f"  Workers: {s.num_workers} total "
            f"({s.workers_idle} idle, {s.workers_busy} busy, "
            f"{s.workers_pending} pending, {s.workers_failed} failed)"
        )
        print(f"  Tasks: {s.tasks_queued} queued, " f"{s.tasks_completed} completed, {s.tasks_failed} failed")
        print("  Worker details:")
        for w in s.worker_details:
            task_info = f", task={w['current_task_id']}" if w["current_task_id"] else ""
            print(
                f"    {w['worker_id']}: {w['status']}{task_info} "
                f"(done={w['tasks_completed']}, err={w['tasks_failed']})"
            )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool.

        Args:
            wait: If True, wait for pending tasks to complete before
                terminating workers
        """
        self._shutdown = True

        # Stop all dispatchers
        for dispatcher in self._dispatchers:
            dispatcher.stop()

        if wait:
            self._task_queue.join()
            for dispatcher in self._dispatchers:
                dispatcher.join(timeout=5.0)

        # Terminate worker jobs
        for job_id in self._job_ids:
            try:
                self._client.terminate(job_id)
            except Exception:
                pass
