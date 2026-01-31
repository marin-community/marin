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
    from iris.client import IrisClient, WorkerPool, WorkerPoolConfig
    from iris.cluster.types import ResourceSpec

    client = IrisClient.remote("http://controller:8080", workspace=Path("./my-project"))

    config = WorkerPoolConfig(
        num_workers=3,
        resources=ResourceSpec(cpu=1, memory="512m"),
    )

    with WorkerPool(client, config) as pool:
        # Submit tasks
        futures = [pool.submit(expensive_fn, i) for i in range(10)]
        results = [f.result() for f in futures]
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from contextvars import Context, copy_context
from dataclasses import dataclass
from enum import Enum, auto
from queue import Empty, Queue
from typing import Any, Generic, TypeVar

import cloudpickle
from connectrpc.errors import ConnectError

from iris.actor import ActorServer
from iris.actor.client import ActorClient
from iris.actor.resolver import Resolver
from iris.client.client import IrisClient, Job, iris_ctx
from iris.cluster.client import get_job_info
from iris.cluster.types import EnvironmentSpec, Entrypoint, JobId, ResourceSpec
from iris.time_utils import ExponentialBackoff

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WorkerStatus(Enum):
    """Status of a worker in the pool."""

    PENDING = auto()  # Worker job launched, not yet registered
    IDLE = auto()  # Ready to accept tasks
    BUSY = auto()  # Currently executing a task
    FAILED = auto()  # Worker has failed/disconnected


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
    """Actor that executes arbitrary callables received as cloudpickle-serialized bytes."""

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
            The return value of calling fn(*args, **kwargs)

        Raises:
            Any exception raised by the callable
        """
        fn = cloudpickle.loads(serialized_callable)
        args = cloudpickle.loads(serialized_args)
        kwargs = cloudpickle.loads(serialized_kwargs)
        return fn(*args, **kwargs)


def worker_job_entrypoint(pool_id: str) -> None:
    """Job entrypoint that starts a TaskExecutor actor.

    This function runs inside each task of the co-scheduled worker pool job.
    It uses IRIS_TASK_INDEX from the environment to determine which worker
    index this task represents.

    Args:
        pool_id: Unique identifier for the worker pool
    """
    ctx = iris_ctx()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("No job info available - must run inside an Iris job")

    task_index = job_info.task_index
    worker_name = f"_workerpool_{pool_id}:worker-{task_index}"

    logger.info("Worker starting: pool_id=%s, task_index=%d of %d", pool_id, task_index, job_info.num_tasks)
    logger.info("Worker name: %s, job_id=%s", worker_name, ctx.job_id)

    # Get the allocated port - this port is published by Docker for container access
    port = ctx.get_port("actor")

    # Start actor server on the allocated port
    server = ActorServer(host="0.0.0.0", port=port)
    server.register(worker_name, TaskExecutorActor())
    actual_port = server.serve_background()

    # Register endpoint with registry
    if ctx.registry is None:
        raise RuntimeError("No registry available - are you running in a cluster context?")
    address = f"{job_info.advertise_host}:{actual_port}"
    ctx.registry.register(worker_name, address, {"job_id": ctx.job_id})
    logger.info("ActorServer started and registered on port %d", actual_port)

    # Serve forever
    logger.info("Worker ready, waiting for tasks...")
    while True:
        time.sleep(1)


class WorkerDispatcher:
    """Dispatch thread for a single worker with automatic retry on infrastructure failures.

    State transitions:
    - PENDING: Poll resolver for endpoint registration
    - IDLE: Wait for task from queue
    - BUSY: Execute task on worker endpoint
    - FAILED: Worker has failed, thread exits
    """

    def __init__(
        self,
        state: WorkerState,
        task_queue: "Queue[PendingTask]",
        resolver: Resolver,
        timeout: float,
        context: Context | None = None,
    ):
        self.state = state
        self._task_queue = task_queue
        self._resolver = resolver
        self._timeout = timeout
        self._context = context
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._discover_backoff = ExponentialBackoff(initial=0.05, maximum=1.0)
        self._actor_client: ActorClient | None = None

    def start(self) -> None:
        if self._context is not None:
            self._thread = threading.Thread(
                target=self._context.run,
                args=(self._run,),
                daemon=True,
                name=f"dispatch-{self.state.worker_id}",
            )
        else:
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name=f"dispatch-{self.state.worker_id}",
            )
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._shutdown.is_set():
            if self.state.status == WorkerStatus.PENDING:
                self._discover_endpoint()
                continue

            if self.state.status == WorkerStatus.FAILED:
                break

            # Initialize actor client if needed (handles test cases where worker starts IDLE)
            if self._actor_client is None:
                self._actor_client = ActorClient(
                    resolver=self._resolver,
                    name=self.state.worker_name,
                    timeout=self._timeout,
                )

            task = self._get_task()
            if task:
                self._execute_task(task)

    def _discover_endpoint(self) -> None:
        result = self._resolver.resolve(self.state.worker_name)
        if not result.is_empty:
            endpoint = result.first()
            self.state.endpoint_url = endpoint.url
            self.state.status = WorkerStatus.IDLE
            self._discover_backoff.reset()
            self._actor_client = ActorClient(
                resolver=self._resolver,
                name=self.state.worker_name,
                timeout=self._timeout,
            )
            logger.info("Worker %s discovered at %s", self.state.worker_id, endpoint.url)
        else:
            time.sleep(self._discover_backoff.next_interval())

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
            if self._actor_client is None:
                raise RuntimeError(f"Worker {self.state.worker_id} has no actor client")

            result = self._actor_client.execute(
                task.serialized_fn,
                task.serialized_args,
                task.serialized_kwargs,
            )
            task.future.set_result(result)
            self.state.tasks_completed += 1
        except Exception as e:
            # ActorClient propagates user exceptions directly (no UserException wrapper needed)
            # We treat RuntimeError/TimeoutError/ConnectError as infrastructure failures eligible for retry
            is_infrastructure_failure = isinstance(e, (RuntimeError, TimeoutError, ConnectionError, ConnectError))

            if is_infrastructure_failure and task.retries_remaining > 0:
                task.retries_remaining -= 1
                self._task_queue.put(task)
                logger.exception(
                    "Worker %s failed, re-queuing task %s (%d retries left)",
                    self.state.worker_id,
                    task.task_id,
                    task.retries_remaining,
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
    resources: ResourceSpec
    environment: EnvironmentSpec | None = None
    name_prefix: str = "worker"
    max_retries: int = 0


@dataclass
class WorkerFuture(Generic[T]):
    """Future representing an in-flight task."""

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
        return self._future.done()

    def exception(self) -> BaseException | None:
        if not self._future.done():
            return None
        return self._future.exception()


class WorkerPool:
    """Pool of stateless workers for task dispatch.

    Usage:
        with WorkerPool(client, config) as pool:
            # Submit single task
            future = pool.submit(fn, arg1, arg2)
            result = future.result()

            # Map over items
            futures = pool.map(fn, items)
            results = [f.result() for f in futures]

            # Check pool status
            status = pool.status()
    """

    def __init__(
        self,
        client: IrisClient,
        config: WorkerPoolConfig,
        timeout: float = 30.0,
        resolver: Resolver | None = None,
    ):
        """Create a worker pool.

        Args:
            client: IrisClient for launching worker jobs
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
        self._job: Job | None = None

        # Task queue and dispatch
        self._task_queue: Queue[PendingTask] = Queue()
        self._dispatchers: list[WorkerDispatcher] = []
        self._shutdown = False

        # Resolver for endpoint discovery (injectable for testing)
        self._resolver: Resolver | None = resolver

    def __enter__(self) -> "WorkerPool":
        self._launch_workers()
        self.wait_for_workers()
        return self

    def __exit__(self, *_):
        self.shutdown(wait=False)

    @property
    def pool_id(self) -> str:
        return self._pool_id

    @property
    def size(self) -> int:
        return sum(1 for w in self._workers.values() if w.status in (WorkerStatus.IDLE, WorkerStatus.BUSY))

    @property
    def idle_count(self) -> int:
        return sum(1 for w in self._workers.values() if w.status == WorkerStatus.IDLE)

    @property
    def job_id(self) -> JobId | None:
        return self._job.job_id if self._job else None

    def _launch_workers(self) -> None:
        # Initialize worker state for each task we'll create
        for i in range(self._config.num_workers):
            worker_id = f"worker-{i}"
            worker_name = f"_workerpool_{self._pool_id}:{worker_id}"
            self._workers[worker_id] = WorkerState(
                worker_id=worker_id,
                worker_name=worker_name,
                status=WorkerStatus.PENDING,
            )

        # Submit ONE job with replicas=num_workers (co-scheduling)
        # Each replica becomes a task that reads its task_index from the environment
        entrypoint = Entrypoint.from_callable(worker_job_entrypoint, self._pool_id)

        job = self._client.submit(
            entrypoint=entrypoint,
            name=f"{self._config.name_prefix}-{self._pool_id}",
            resources=self._config.resources,
            environment=self._config.environment,
            ports=["actor"],
            replicas=self._config.num_workers,
        )
        self._job = job

        # Create resolver after job submission so we can use the job's namespace.
        # Workers register endpoints with namespace prefix derived from job_id.
        if self._resolver is None:
            self._resolver = self._client.resolver_for_job(self._job.job_id)

        # Start dispatchers (one per worker). Each thread needs its own context copy
        # because a Context can only be entered by one thread at a time.
        for worker_state in self._workers.values():
            ctx = copy_context()
            dispatcher = WorkerDispatcher(
                state=worker_state,
                task_queue=self._task_queue,
                resolver=self._resolver,
                timeout=self._timeout,
                context=ctx,
            )
            dispatcher.start()
            self._dispatchers.append(dispatcher)

    def wait_for_workers(
        self,
        min_workers: int | None = None,
        timeout: float = 600.0,
    ) -> None:
        """Wait for workers to become available.

        Args:
            min_workers: Minimum workers required (default: all workers)
            timeout: Maximum time to wait in seconds

        Raises:
            TimeoutError: If min_workers not available within timeout
        """
        if min_workers is None:
            min_workers = self._config.num_workers

        ExponentialBackoff(initial=0.05, maximum=1.0).wait_until_or_raise(
            lambda: self.size >= min_workers,
            timeout=timeout,
            error_message=f"Only {self.size} of {min_workers} workers registered within {timeout}s",
        )

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> WorkerFuture[T]:
        """Submit a task for execution.

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

        # Always join dispatchers to prevent orphaned threads logging after stdout/stderr closed
        # No timeout to ensure complete cleanup - if a thread hangs, pytest-timeout will handle it
        if wait:
            self._task_queue.join()
        for dispatcher in self._dispatchers:
            dispatcher.join(timeout=None)

        # Terminate worker job
        if self._job:
            try:
                self._job.terminate()
            except Exception as e:
                logger.debug("Failed to terminate worker job %s: %s", self._job.job_id, e)
