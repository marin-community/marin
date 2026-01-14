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
    from fluster.worker_pool import WorkerPool, WorkerPoolConfig
    from fluster.cluster.client import RpcClusterClient, BundleCreator

    bundle = BundleCreator().create_bundle()
    client = RpcClusterClient("http://controller:8080", bundle)

    config = WorkerPoolConfig(
        num_workers=3,
        resources=cluster_pb2.ResourceSpec(cpu=1, memory="512m"),
    )

    with WorkerPool(client, config) as pool:
        # Submit tasks
        futures = [pool.submit(expensive_fn, i) for i in range(10)]
        results = [f.result() for f in futures]
"""

import os
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import cloudpickle

from fluster import cluster_pb2
from fluster.actor import ActorServer
from fluster.actor.pool import ActorPool
from fluster.actor.resolver import ClusterResolver
from fluster.cluster.client import ClusterClient
from fluster.cluster.types import Entrypoint, JobId
from fluster.cluster_connect import ControllerServiceClientSync

T = TypeVar("T")


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


def register_endpoint(
    controller_url: str,
    name: str,
    address: str,
    job_id: str,
    namespace: str,
) -> str:
    """Register an endpoint with the cluster controller.

    Args:
        controller_url: Controller URL (e.g., "http://localhost:8080")
        name: Endpoint name for discovery
        address: Address where the endpoint is listening (host:port)
        job_id: Job ID that owns this endpoint
        namespace: Namespace for isolation

    Returns:
        Endpoint ID assigned by the controller
    """
    client = ControllerServiceClientSync(address=controller_url, timeout_ms=10000)
    request = cluster_pb2.Controller.RegisterEndpointRequest(
        name=name,
        address=address,
        job_id=job_id,
        namespace=namespace,
    )
    response = client.register_endpoint(request)
    return response.endpoint_id


def worker_job_entrypoint(pool_id: str) -> None:
    """Job entrypoint that starts a TaskExecutor actor.

    This function is called when a worker job starts. It:
    1. Reads cluster configuration from environment variables
    2. Starts an ActorServer with a TaskExecutorActor
    3. Registers the endpoint with the controller
    4. Runs forever, serving requests

    All workers register under the same name (pool_id), and the resolver
    returns all endpoints for load balancing.

    Environment variables (injected by the cluster):
        FLUSTER_JOB_ID: Unique job identifier
        FLUSTER_NAMESPACE: Namespace for actor isolation
        FLUSTER_PORT_ACTOR: Port allocated for the actor server
        FLUSTER_CONTROLLER_ADDRESS: Controller URL for registration

    Args:
        pool_id: Unique identifier for the worker pool
    """
    job_id = os.environ["FLUSTER_JOB_ID"]
    namespace = os.environ["FLUSTER_NAMESPACE"]
    port = int(os.environ["FLUSTER_PORT_ACTOR"])
    controller_url = os.environ["FLUSTER_CONTROLLER_ADDRESS"]

    print(f"Worker starting: pool_id={pool_id}, job_id={job_id}, namespace={namespace}")
    print(f"Using allocated port: {port}")

    # Start actor server
    server = ActorServer(host="0.0.0.0", port=port)
    actor_name = f"_workerpool_{pool_id}"  # All workers share same name
    server.register(actor_name, TaskExecutorActor())
    actual_port = server.serve_background()
    print(f"ActorServer started on port {actual_port}")

    # Register with controller
    # Use localhost since the port is mapped from host to container via Docker -p
    endpoint_address = f"localhost:{actual_port}"
    print(f"Registering endpoint: {actor_name} at {endpoint_address}")

    endpoint_id = register_endpoint(
        controller_url,
        actor_name,
        endpoint_address,
        job_id,
        namespace,
    )
    print(f"Endpoint registered: {endpoint_id}")

    # Serve forever
    print("Worker ready, waiting for tasks...")
    while True:
        time.sleep(1)


@dataclass
class WorkerPoolConfig:
    """Configuration for a WorkerPool.

    Attributes:
        num_workers: Number of worker jobs to launch
        resources: Resource requirements per worker
        environment: Optional environment configuration
        name_prefix: Prefix for worker job names
    """

    num_workers: int
    resources: cluster_pb2.ResourceSpec
    environment: cluster_pb2.EnvironmentConfig | None = None
    name_prefix: str = "worker"


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
    """Pool of stateless workers for task dispatch.

    WorkerPool manages a set of worker jobs that execute arbitrary callables.
    Workers are stateless - if a worker fails, tasks can be retried on any
    other worker.

    Usage:
        with WorkerPool(client, config) as pool:
            # Submit single task
            future = pool.submit(fn, arg1, arg2)
            result = future.result()

            # Map over items
            futures = pool.map(fn, items)
            results = [f.result() for f in futures]
    """

    def __init__(
        self,
        client: ClusterClient,
        config: WorkerPoolConfig,
    ):
        """Create a worker pool.

        Args:
            client: ClusterClient for launching worker jobs
            config: Pool configuration (workers, resources, etc.)
        """
        self._client = client
        self._config = config
        self._pool_id = uuid.uuid4().hex[:8]
        self._job_ids: list[JobId] = []
        self._actor_pool: ActorPool | None = None
        self._executor = ThreadPoolExecutor(max_workers=32)
        self._resolver: ClusterResolver | None = None
        self._namespace = "<local>"

    def __enter__(self) -> "WorkerPool":
        """Start workers and wait for at least one to register."""
        self._launch_workers()
        self._wait_for_workers(min_workers=1)
        return self

    def __exit__(self, *args):
        """Shutdown all workers."""
        self.shutdown(wait=False)

    @property
    def pool_id(self) -> str:
        """Unique identifier for this pool."""
        return self._pool_id

    @property
    def actor_name(self) -> str:
        """Actor name used by workers for registration."""
        return f"_workerpool_{self._pool_id}"

    @property
    def size(self) -> int:
        """Number of workers currently available."""
        if self._actor_pool is None:
            return 0
        return self._actor_pool.size

    @property
    def job_ids(self) -> list[JobId]:
        """List of worker job IDs."""
        return list(self._job_ids)

    def _launch_workers(self) -> None:
        """Launch worker jobs."""
        # Create resolver for worker discovery
        self._resolver = ClusterResolver(
            self._client.controller_address,
            namespace=self._namespace,
        )

        # Launch worker jobs
        for i in range(self._config.num_workers):
            entrypoint = Entrypoint(
                callable=worker_job_entrypoint,
                args=(self._pool_id,),
            )

            job_id = self._client.submit(
                entrypoint=entrypoint,
                name=f"{self._config.name_prefix}-{self._pool_id}-{i}",
                resources=self._config.resources,
                environment=self._config.environment,
                namespace=self._namespace,
                ports=["actor"],  # Request port allocation for ActorServer
            )
            self._job_ids.append(job_id)

        # Create actor pool for dispatching to workers
        self._actor_pool = ActorPool(self._resolver, self.actor_name)

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
        if self._actor_pool is None:
            raise RuntimeError("Workers not launched")

        start = time.time()
        while time.time() - start < timeout:
            if self._actor_pool.size >= min_workers:
                return
            time.sleep(0.5)

        raise TimeoutError(f"Only {self._actor_pool.size} of {min_workers} workers " f"registered within {timeout}s")

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

        The callable and arguments must be picklable (via cloudpickle).
        Tasks are distributed round-robin across available workers.

        Args:
            fn: Callable to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future that resolves to the function's return value
        """
        if self._actor_pool is None:
            raise RuntimeError("Workers not available")

        # Serialize callable and args
        serialized_fn = cloudpickle.dumps(fn)
        serialized_args = cloudpickle.dumps(args)
        serialized_kwargs = cloudpickle.dumps(kwargs)

        # Submit to executor - calls actor via pool
        def _call():
            return self._actor_pool.call().execute(
                serialized_fn,
                serialized_args,
                serialized_kwargs,
            )

        future = self._executor.submit(_call)
        return WorkerFuture(_future=future, _fn_name=getattr(fn, "__name__", "lambda"))

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

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool.

        Args:
            wait: If True, wait for pending tasks to complete before
                terminating workers
        """
        # Shutdown executor
        self._executor.shutdown(wait=wait)

        # Terminate worker jobs
        for job_id in self._job_ids:
            try:
                self._client.terminate(job_id)
            except Exception:
                pass  # Job may already be terminated
