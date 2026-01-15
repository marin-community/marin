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

"""Example demonstrating full cluster operation with controller and worker.

This example runs a complete mini-cluster locally:
- Controller: schedules jobs, tracks workers, serves dashboard
- Worker: executes jobs in Docker containers

Usage:
    cd lib/fluster
    uv run python examples/cluster_example.py
"""

# TODO, consider having like a post-mortem view of the cluster state
# means cluster state should be serializable, cluster dashboard would always be a mapping over the state

import socket
import tempfile
import threading
import time
import uuid
from pathlib import Path

import click
from fluster import cluster_pb2
from fluster.cluster.client import RpcClusterClient
from fluster.cluster.controller.controller import Controller, ControllerConfig, DefaultWorkerStubFactory
from fluster.cluster.types import Entrypoint
from fluster.cluster.worker.worker import Worker, WorkerConfig
from fluster.cluster_connect import ControllerServiceClientSync, WorkerServiceClientSync

# The fluster project root (lib/fluster/) - used as workspace for the example
FLUSTER_ROOT = Path(__file__).parent.parent


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class LogPoller:
    """Background thread that polls for job logs and prints them."""

    def __init__(
        self,
        job_id: str,
        worker_address: str,
        poll_interval: float = 1.0,
        prefix: str = "",
    ):
        """Initialize log poller.

        Args:
            job_id: Job ID to poll logs for
            worker_address: Worker RPC address (e.g., "http://127.0.0.1:8080")
            poll_interval: How often to poll for new logs (in seconds)
            prefix: Optional prefix to add to log lines (e.g., "[calculator] ")
        """
        self._job_id = job_id
        self._worker_address = worker_address
        self._poll_interval = poll_interval
        self._prefix = prefix
        self._last_timestamp_ms = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Start polling for logs in background thread."""
        if self._thread is not None:
            return  # Already started

        def _poll():
            client = WorkerServiceClientSync(address=self._worker_address, timeout_ms=5000)

            while not self._stop_event.is_set():
                try:
                    # Build filter with timestamp for incremental fetching
                    filter_proto = cluster_pb2.Worker.FetchLogsFilter()
                    if self._last_timestamp_ms > 0:
                        filter_proto.start_ms = self._last_timestamp_ms

                    request = cluster_pb2.Worker.FetchLogsRequest(
                        job_id=self._job_id,
                        filter=filter_proto,
                    )
                    response = client.fetch_logs(request)

                    for entry in response.logs:
                        # Update last seen timestamp
                        if entry.timestamp_ms > self._last_timestamp_ms:
                            self._last_timestamp_ms = entry.timestamp_ms

                        # Print log with prefix
                        print(f"{self._prefix}{entry.data}", flush=True)

                except Exception:
                    # Ignore errors (job may not exist yet, worker starting, etc.)
                    pass

                # Wait for next poll interval
                self._stop_event.wait(self._poll_interval)

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop polling thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None


class ExampleCluster:
    """Synchronous context manager running a controller + worker cluster.

    Provides a simple API for submitting jobs through the controller,
    which schedules them to workers.

    Example:
        with ExampleCluster() as cluster:
            job_id = cluster.submit(my_function, arg1, arg2)
            status = cluster.wait(job_id)
            logs = cluster.logs(job_id)
    """

    def __init__(
        self,
        controller_port: int = 0,
        worker_port: int = 0,
        max_concurrent_jobs: int = 3,
        registry: str = "localhost:5000",
    ):
        self._controller_port = controller_port or find_free_port()
        self._worker_port = worker_port or find_free_port()
        self._max_concurrent_jobs = max_concurrent_jobs
        self._registry = registry

        # Will be initialized in __enter__
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._bundle_dir: Path | None = None
        self._worker_id: str | None = None

        # Controller and Worker
        self._controller: Controller | None = None
        self._worker: Worker | None = None

        # RPC client for controller calls
        self._controller_client: ControllerServiceClientSync | None = None

        # Cached RpcClusterClient
        self._rpc_client: RpcClusterClient | None = None

        # Log polling
        self._log_pollers: dict[str, LogPoller] = {}

    def __enter__(self):
        """Start controller and worker."""
        self._temp_dir = tempfile.TemporaryDirectory(prefix="cluster_")
        temp_path = Path(self._temp_dir.name)
        self._bundle_dir = temp_path / "bundles"
        self._bundle_dir.mkdir()
        cache_path = temp_path / "cache"
        cache_path.mkdir()

        # --- Start Worker First (so it's ready when controller dispatches) ---
        print("Starting worker components...")
        self._worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        worker_config = WorkerConfig(
            host="127.0.0.1",
            port=self._worker_port,
            cache_dir=cache_path,
            registry=self._registry,
            max_concurrent_jobs=self._max_concurrent_jobs,
            controller_address=f"http://127.0.0.1:{self._controller_port}",
            worker_id=self._worker_id,
        )
        self._worker = Worker(worker_config, cache_dir=cache_path)
        self._worker.start()
        print(f"Worker server should be at http://127.0.0.1:{self._worker_port}")

        # --- Start Controller ---
        print("Starting controller components...")
        controller_config = ControllerConfig(
            host="127.0.0.1",
            port=self._controller_port,
            bundle_dir=self._bundle_dir,
        )
        self._controller = Controller(
            config=controller_config,
            worker_stub_factory=DefaultWorkerStubFactory(),
        )
        self._controller.start()
        print(f"Controller server should be at http://127.0.0.1:{self._controller_port}", flush=True)

        # Create RPC client
        print("Creating RPC client...", flush=True)
        self._controller_client = ControllerServiceClientSync(
            address=f"http://127.0.0.1:{self._controller_port}",
            timeout_ms=30000,
        )
        print("RPC client created", flush=True)

        # Register worker with controller
        print(f"Registering worker {self._worker_id}...", flush=True)
        self._register_worker()
        print("Worker registered", flush=True)

        print(f"Controller: http://127.0.0.1:{self._controller_port}", flush=True)
        print(f"Worker: http://127.0.0.1:{self._worker_port}", flush=True)

        print("Cluster startup complete!", flush=True)
        return self

    def start_log_polling(self, job_id: str, poll_interval: float = 1.0, prefix: str = ""):
        """Start a background thread that polls for job logs.

        Args:
            job_id: Job ID to poll logs for
            poll_interval: How often to poll for new logs (in seconds)
            prefix: Optional prefix to add to log lines (e.g., "[calculator] ")
        """
        if job_id in self._log_pollers:
            return  # Already polling

        poller = LogPoller(job_id, self.worker_url, poll_interval, prefix)
        poller.start()
        self._log_pollers[job_id] = poller

    def stop_log_polling(self, job_id: str):
        """Stop log polling for a job.

        Args:
            job_id: Job ID to stop polling for
        """
        poller = self._log_pollers.pop(job_id, None)
        if poller:
            poller.stop()

    def __exit__(self, *args):
        """Stop cluster and cleanup."""
        # Stop all log polling threads
        for job_id in list(self._log_pollers.keys()):
            self.stop_log_polling(job_id)

        if self._controller_client:
            self._controller_client.close()

        if self._controller:
            self._controller.stop()

        if self._worker:
            self._worker.stop()

        if self._temp_dir:
            self._temp_dir.cleanup()

    def _register_worker(self):
        """Register worker with controller."""
        request = cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id=self._worker_id,
            address=f"127.0.0.1:{self._worker_port}",
            resources=cluster_pb2.ResourceSpec(
                cpu=4,
                memory="16g",
            ),
        )
        self._controller_client.register_worker(request)

    def submit(
        self,
        fn,
        *args,
        name: str | None = None,
        env_vars: dict[str, str] | None = None,
        cpu: int = 1,
        memory: str = "1g",
        namespace: str | None = None,
        ports: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Submit a job to the cluster.

        Args:
            fn: Callable to execute
            *args: Positional arguments for fn
            name: Job name (defaults to function name)
            env_vars: Environment variables
            cpu: Number of CPUs to request
            memory: Memory to request (e.g., "1g", "512m")
            namespace: Namespace for actor isolation (defaults to "default")
            ports: List of port names to allocate (e.g., ["actor", "metrics"])
            **kwargs: Keyword arguments for fn

        Returns:
            Job ID
        """
        entrypoint = Entrypoint.from_callable(fn, *args, **kwargs)
        environment = cluster_pb2.EnvironmentConfig(
            workspace="/app",
            env_vars=env_vars or {},
        )
        resources = cluster_pb2.ResourceSpec(cpu=cpu, memory=memory)

        return self.get_client().submit(
            entrypoint=entrypoint,
            name=name or fn.__name__,
            resources=resources,
            environment=environment,
            namespace=namespace or "default",
            ports=ports,
        )

    def status(self, job_id: str) -> dict:
        """Get job status from controller."""
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self._controller_client.get_job_status(request)
        # Convert protobuf to dict for compatibility with existing code
        return {
            "jobId": response.job.job_id,
            "state": cluster_pb2.JobState.Name(response.job.state),
            "exitCode": response.job.exit_code,
            "error": response.job.error,
            "workerId": response.job.worker_id,
        }

    def wait(self, job_id: str, timeout: float = 300.0, poll_interval: float = 0.5) -> dict:
        """Wait for job to complete."""
        start = time.time()
        terminal_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_KILLED",
            "JOB_STATE_UNSCHEDULABLE",
        }

        while time.time() - start < timeout:
            status = self.status(job_id)
            if status["state"] in terminal_states:
                return status
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

    def logs(self, job_id: str, since_ms: int | None = None) -> list[cluster_pb2.Worker.LogEntry]:
        """Get job logs from worker.

        Args:
            job_id: Job ID
            since_ms: Only return logs after this timestamp (milliseconds since epoch).
                     If None, returns all logs.

        Returns:
            List of LogEntry protos with timestamp_ms, source, and data.
        """
        # Find the worker that has this job
        status = self.status(job_id)
        worker_id = status.get("workerId")

        if not worker_id:
            return []

        # For now, query worker directly (in a real cluster, would route through controller)
        worker_client = WorkerServiceClientSync(
            address=f"http://127.0.0.1:{self._worker_port}",
            timeout_ms=10000,
        )

        # Build filter with optional timestamp
        filter_proto = cluster_pb2.Worker.FetchLogsFilter()
        if since_ms is not None:
            filter_proto.start_ms = since_ms

        request = cluster_pb2.Worker.FetchLogsRequest(job_id=job_id, filter=filter_proto)
        response = worker_client.fetch_logs(request)
        return list(response.logs)

    def kill(self, job_id: str) -> None:
        """Kill a job via controller."""
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self._controller_client.terminate_job(request)

    @property
    def controller_url(self) -> str:
        return f"http://127.0.0.1:{self._controller_port}"

    @property
    def worker_url(self) -> str:
        return f"http://127.0.0.1:{self._worker_port}"

    def get_client(self) -> RpcClusterClient:
        """Get an RpcClusterClient for this cluster."""
        if self._rpc_client is None:
            self._rpc_client = RpcClusterClient(
                self.controller_url,
                workspace=FLUSTER_ROOT,
            )
        return self._rpc_client


# =============================================================================
# ACTOR SYSTEM EXAMPLES
# =============================================================================


def example_actor_job_workflow(cluster: ExampleCluster):
    """Demonstrate real actor job workflow with cluster integration.

    This example shows the complete end-to-end workflow:
    1. Submit a job that runs an ActorServer
    2. The job registers its actor endpoint via serve_and_register()
    3. A client uses ClusterResolver to discover and call the actor
    4. The actor can access cluster context via fluster_ctx()

    This is the recommended pattern for production actor deployments.
    """
    print("\n=== Example: Real Actor Job Workflow ===\n")

    # Step 1: Define an actor job entrypoint
    # This function will run inside a cluster job and start an ActorServer
    def actor_job_entrypoint():
        """Job entrypoint that starts an ActorServer and registers with controller."""
        import time

        from fluster.actor import ActorServer
        from fluster.context import fluster_ctx

        # Get context injected by the cluster
        ctx = fluster_ctx()
        print(f"Actor job starting: job_id={ctx.job_id}, namespace={ctx.namespace}")

        # Define our actor class inline (could also be imported)
        class Calculator:
            def __init__(self):
                self._history = []

            def add(self, a: int, b: int) -> int:
                result = a + b
                self._history.append(f"add({a}, {b}) = {result}")
                print(f"Calculator.add({a}, {b}) = {result}")
                return result

            def multiply(self, a: int, b: int) -> int:
                result = a * b
                self._history.append(f"multiply({a}, {b}) = {result}")
                print(f"Calculator.multiply({a}, {b}) = {result}")
                return result

            def get_history(self) -> list[str]:
                return self._history

        # Start the ActorServer
        server = ActorServer(host="0.0.0.0")
        server.register("calculator", Calculator())
        port = server.serve_background()

        # Register endpoint with controller
        address = f"localhost:{port}"
        ctx.controller.endpoint_registry.register("calculator", address, {"job_id": ctx.job_id})
        print(f"ActorServer started and registered on port {port}")

        # Keep the job running to serve requests
        print("Actor server ready, waiting for requests...")
        while True:
            time.sleep(1)

    # Step 2: Submit the actor job to the cluster
    # Request a port named "actor" which will be allocated and mapped by Docker
    print("Submitting actor job to cluster...")
    job_id = cluster.submit(
        actor_job_entrypoint,
        name="calculator-actor",
        cpu=1,
        memory="512m",
        namespace="default",
        ports=["actor"],
    )
    print(f"Job submitted: {job_id}")

    # Start log polling in background thread
    print("Starting log polling...")
    cluster.start_log_polling(job_id, poll_interval=1.0, prefix="[calculator] ")

    # Step 3: Wait for the job to start and endpoint to be registered
    print("Waiting for job to start...")
    max_wait = 30
    start_time = time.time()
    job_running = False

    while time.time() - start_time < max_wait:
        status = cluster.status(job_id)
        state = status.get("state", "")
        print(f"  Job state: {state}")

        if state == "JOB_STATE_RUNNING":
            job_running = True
            print("Job is running!")
            break

        if state in ["JOB_STATE_FAILED", "JOB_STATE_KILLED"]:
            print(f"Job failed: {status.get('error', 'Unknown error')}")
            logs = cluster.logs(job_id)
            if logs:
                print("Job logs:")
                for log in logs:
                    print(f"  {log}")
            return

        time.sleep(1)

    if not job_running:
        print("Job did not start in time")
        return

    # Give the actor server time to register
    print("Waiting for endpoint registration...")
    time.sleep(3)

    # Step 4: Use ClusterResolver to discover the actor
    from fluster.actor import ActorClient, ClusterResolver
    from fluster.cluster.types import Namespace

    print("\nResolving actor via ClusterResolver...")
    resolver = ClusterResolver(cluster.controller_url, namespace=Namespace("default"))
    client = ActorClient(resolver, "calculator")

    # Step 5: Call the actor methods
    print("\nCalling actor methods...")
    result1 = client.add(10, 20)
    print(f"Client received: add(10, 20) = {result1}")

    result2 = client.multiply(5, 7)
    print(f"Client received: multiply(5, 7) = {result2}")

    history = client.get_history()
    print(f"Operation history: {history}")

    print("\nActor job workflow complete!")
    print("Note: The actor job will continue running until the cluster shuts down.")


def example_worker_pool(cluster: ExampleCluster):
    """Demonstrate WorkerPool for task dispatch."""
    from fluster.worker_pool import WorkerPool, WorkerPoolConfig

    print("\n=== Example: Worker Pool ===\n")

    client = cluster.get_client()
    config = WorkerPoolConfig(
        num_workers=2,
        resources=cluster_pb2.ResourceSpec(cpu=1, memory="512m"),
    )

    def square(x):
        return x * x

    with WorkerPool(client, config) as pool:
        print(f"WorkerPool started with {pool.size} workers")
        futures = pool.map(square, [1, 2, 3, 4, 5])
        results = [f.result() for f in futures]
        print(f"Results: {results}")

    print("\nWorkerPool example complete!")


def example_basic(cluster: ExampleCluster):
    """Basic job submission through cluster."""
    print("\n=== Example: Basic Job Submission ===\n", flush=True)

    def hello():
        print("Hello from the cluster!")
        return 42

    job_id = cluster.submit(hello, name="hello-job")
    print(f"Submitted: {job_id}")

    status = cluster.wait(job_id)
    print(f"Completed with state: {status['state']}")
    if status.get("error"):
        print(f"Error: {status['error']}")
    if status.get("exitCode"):
        print(f"Exit code: {status['exitCode']}")

    logs = cluster.logs(job_id)
    if logs:
        print("Logs:")
        for log in logs:
            print(f"  {log}")


def example_with_args(cluster: ExampleCluster):
    """Job with arguments."""
    print("\n=== Example: Job With Arguments ===\n")

    def add_numbers(a, b):
        result = a + b
        print(f"{a} + {b} = {result}")
        return result

    job_id = cluster.submit(add_numbers, 10, 32, name="add-job")
    print(f"Submitted: {job_id}")

    status = cluster.wait(job_id)
    print(f"Completed: {status['state']}")

    logs = cluster.logs(job_id)
    if logs:
        print("Output:")
        for log in logs:
            print(f"  {log}")


def example_concurrent(cluster: ExampleCluster):
    """Multiple concurrent jobs."""
    print("\n=== Example: Concurrent Jobs ===\n")

    def slow_job(n):
        import time as t

        for i in range(3):
            print(f"Job {n}: iteration {i}")
            t.sleep(1)
        return n

    # Submit 3 jobs
    job_ids = []
    for i in range(3):
        job_id = cluster.submit(slow_job, i, name=f"slow-{i}")
        job_ids.append(job_id)
        print(f"Submitted job {i}: {job_id[:8]}...")

    # Wait for all
    print("\nWaiting for jobs to complete...")
    for i, job_id in enumerate(job_ids):
        status = cluster.wait(job_id)
        print(f"Job {i} ({job_id[:8]}...): {status['state']}")


def example_kill(cluster: ExampleCluster):
    """Kill a running job."""
    print("\n=== Example: Kill Job ===\n")

    def long_job():
        import time as t

        for i in range(60):
            print(f"Tick {i}")
            t.sleep(1)

    job_id = cluster.submit(long_job, name="long-job")
    print(f"Started: {job_id[:8]}...")

    # Wait for it to start running
    print("Waiting for job to start...")
    for _ in range(60):
        status = cluster.status(job_id)
        if status["state"] == "JOB_STATE_RUNNING":
            print("Job is running!")
            break
        time.sleep(0.5)
    else:
        print("Job did not start in time")
        return

    # Give it a moment to produce some output
    time.sleep(2)

    print("Killing job...")
    cluster.kill(job_id)

    status = cluster.status(job_id)
    print(f"Final state: {status['state']}")


def example_resource_serialization(cluster: ExampleCluster):
    """Demonstrate job serialization based on resource constraints.

    The worker has 4 CPUs. We submit jobs requiring 2 CPUs each,
    so only 2 can run concurrently. The rest must wait in queue.
    """
    print("\n=== Example: Resource Serialization ===\n")

    def cpu_bound_job(n):
        import time as t

        print(f"Job {n}: starting (needs 2 CPUs)")
        t.sleep(3)
        print(f"Job {n}: completed")
        return n

    # Submit 4 jobs, each requiring 2 CPUs (worker has 4 CPUs total)
    # Only 2 should run at a time
    job_ids = []
    for i in range(4):
        job_id = cluster.submit(cpu_bound_job, i, name=f"cpu-job-{i}", cpu=2, memory="1g")
        job_ids.append(job_id)
        print(f"Submitted job {i}: {job_id[:8]}... (requires 2 CPUs)")

    # Check initial states - first 2 should be running, rest pending
    time.sleep(2)
    print("\nChecking job states after 2 seconds:")
    for i, job_id in enumerate(job_ids):
        status = cluster.status(job_id)
        print(f"  Job {i}: {status['state']}")

    # Wait for all jobs to complete
    print("\nWaiting for all jobs to complete...")
    for i, job_id in enumerate(job_ids):
        status = cluster.wait(job_id)
        print(f"Job {i} ({job_id[:8]}...): {status['state']}")


def example_scheduling_timeout(cluster: ExampleCluster):
    """Demonstrate scheduling timeout for jobs that can't be scheduled.

    Submit a job requiring more resources than available. With a short
    scheduling timeout, it should become UNSCHEDULABLE.
    """
    print("\n=== Example: Scheduling Timeout ===\n")

    def impossible_job():
        print("This should never run!")
        return 0

    # Submit a job requiring 100 CPUs (worker only has 4)
    # With a 2 second scheduling timeout, it should fail quickly
    print("Submitting job requiring 100 CPUs (worker has 4)...")
    print("Setting 2 second scheduling timeout...")
    job_id = cluster.submit(
        impossible_job,
        name="impossible-job",
        cpu=100,
        memory="1g",
        scheduling_timeout_seconds=2,
    )
    print(f"Submitted: {job_id[:8]}...")

    # Wait for it to timeout
    status = cluster.wait(job_id, timeout=10.0)
    print(f"Final state: {status['state']}")
    if status.get("error"):
        print(f"Error: {status['error']}")


def example_small_job_skips_queue(cluster: ExampleCluster):
    """Demonstrate that smaller jobs can skip ahead of larger jobs.

    Submit a large job that won't fit, then a small job. The small
    job should be scheduled even though the large job is ahead in queue.
    """
    print("\n=== Example: Small Jobs Skip Large Jobs ===\n")

    def big_job():
        import time as t

        print("Big job running (this shouldn't happen immediately)")
        t.sleep(5)
        return "big"

    def small_job():
        import time as t

        print("Small job running!")
        t.sleep(1)
        return "small"

    # First submit a job that's too big to fit
    print("Submitting big job (8 CPUs, won't fit on 4-CPU worker)...")
    big_job_id = cluster.submit(big_job, name="big-job", cpu=8, memory="1g", scheduling_timeout_seconds=0)
    print(f"Big job: {big_job_id[:8]}...")

    # Then submit a small job that can run
    print("Submitting small job (1 CPU)...")
    small_job_id = cluster.submit(small_job, name="small-job", cpu=1, memory="1g")
    print(f"Small job: {small_job_id[:8]}...")

    # Small job should run even though big job is first in queue
    time.sleep(2)
    big_status = cluster.status(big_job_id)
    small_status = cluster.status(small_job_id)
    print("\nAfter 2 seconds:")
    print(f"  Big job: {big_status['state']}")
    print(f"  Small job: {small_status['state']}")

    # Wait for small job to complete
    small_result = cluster.wait(small_job_id, timeout=30.0)
    print(f"\nSmall job completed: {small_result['state']}")

    # Big job should still be pending (never scheduled)
    big_status = cluster.status(big_job_id)
    print(f"Big job still: {big_status['state']}")


@click.command()
@click.option(
    "--wait/--no-wait", default=False, help="Wait for Ctrl+C after examples complete (for dashboard exploration)"
)
@click.option(
    "--mode",
    type=click.Choice(["all", "actors", "jobs"], case_sensitive=False),
    default="all",
    help="Which examples to run: all (default), actors (actor system only), or jobs (cluster jobs only)",
)
def main(wait: bool, mode: str):
    """Run cluster and actor examples.

    This example demonstrates the full Fluster system including:
    - Cluster controller and worker for job scheduling
    - Actor system for distributed RPC between services
    - Various patterns: coordinator, pool, load-balancing, broadcast
    """
    print("=" * 60)
    print("Fluster Cluster & Actor System Example")
    print("=" * 60)

    if mode in ["all", "jobs"]:
        print("\nNote: Job examples require Docker to be running.")

    try:
        with ExampleCluster(max_concurrent_jobs=3) as cluster:
            print(f"\nController dashboard: {cluster.controller_url}", flush=True)
            print(f"Worker dashboard: {cluster.worker_url}", flush=True)
            if wait:
                print("\nPress Ctrl+C to stop.\n", flush=True)

            # Run actor examples
            if mode in ["all", "actors"]:
                print("\n" + "=" * 60)
                print("ACTOR SYSTEM EXAMPLES")
                print("=" * 60)
                example_actor_job_workflow(cluster)
                example_worker_pool(cluster)

            # Run cluster job examples
            if mode in ["all", "jobs"]:
                print("\n" + "=" * 60)
                print("CLUSTER JOB EXAMPLES")
                print("=" * 60)
                print("About to run job examples...", flush=True)
                example_basic(cluster)
                example_with_args(cluster)
                example_concurrent(cluster)
                example_kill(cluster)
                example_resource_serialization(cluster)
                example_scheduling_timeout(cluster)
                example_small_job_skips_queue(cluster)

            print("\n" + "=" * 60)
            print("All examples completed!")
            print("=" * 60)

            if wait:
                print("Dashboards still available for exploration.")
                print("Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        if mode in ["all", "jobs"]:
            print("\nMake sure Docker is running and try again.")
        raise


if __name__ == "__main__":
    main()
