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

import base64
import socket
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import click
import cloudpickle
import httpx
import uvicorn
from fluster import cluster_pb2
from fluster.cluster.controller.dashboard import ControllerDashboard
from fluster.cluster.controller.heartbeat import HeartbeatMonitor
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.state import ControllerState
from fluster.cluster.types import Entrypoint, JobId, WorkerId
from fluster.cluster.worker.builder import ImageCache, VenvCache
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.dashboard import WorkerDashboard
from fluster.cluster.worker.docker import DockerRuntime
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.service import WorkerServiceImpl


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


MINIMAL_PYPROJECT = """\
[project]
name = "fluster-example"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["cloudpickle"]
"""


def create_minimal_workspace(temp_dir: Path) -> Path:
    """Create a minimal workspace with pyproject.toml and uv.lock."""
    workspace = temp_dir / "workspace"
    workspace.mkdir(exist_ok=True)

    # Write minimal pyproject.toml
    (workspace / "pyproject.toml").write_text(MINIMAL_PYPROJECT)

    # Generate uv.lock
    import subprocess

    subprocess.run(
        ["uv", "lock"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    return workspace


def create_workspace_bundle(workspace: Path, output_path: Path) -> None:
    """Create a zip bundle from workspace directory."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in workspace.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(workspace))


class ClusterContext:
    """Synchronous context manager running a controller + worker cluster.

    Provides a simple API for submitting jobs through the controller,
    which schedules them to workers.

    Example:
        with ClusterContext() as cluster:
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
        self._workspace: Path | None = None
        self._worker_id: str | None = None

        # Controller components
        self._state: ControllerState | None = None
        self._scheduler: Scheduler | None = None
        self._heartbeat_monitor: HeartbeatMonitor | None = None
        self._controller_service: ControllerServiceImpl | None = None
        self._controller_dashboard: ControllerDashboard | None = None
        self._controller_thread: threading.Thread | None = None

        # Worker components
        self._job_manager: JobManager | None = None
        self._worker_service: WorkerServiceImpl | None = None
        self._worker_dashboard: WorkerDashboard | None = None
        self._worker_thread: threading.Thread | None = None

        # HTTP client for RPC calls
        self._client: httpx.Client | None = None

        # Cached bundle
        self._bundle_blob: bytes | None = None

    def __enter__(self):
        """Start controller and worker."""
        self._temp_dir = tempfile.TemporaryDirectory(prefix="cluster_")
        temp_path = Path(self._temp_dir.name)
        self._bundle_dir = temp_path / "bundles"
        self._bundle_dir.mkdir()
        cache_path = temp_path / "cache"
        cache_path.mkdir()

        # Create a minimal workspace with pyproject.toml and uv.lock
        print("Creating minimal workspace...", flush=True)
        self._workspace = create_minimal_workspace(temp_path)

        # --- Start Worker First (so it's ready when controller dispatches) ---
        print("Starting worker components...")
        bundle_cache = BundleCache(cache_path, max_bundles=100)
        venv_cache = VenvCache()
        image_cache = ImageCache(cache_path, registry=self._registry, max_images=50)
        runtime = DockerRuntime()
        port_allocator = PortAllocator((30000, 40000))

        self._job_manager = JobManager(
            bundle_cache=bundle_cache,
            venv_cache=venv_cache,
            image_cache=image_cache,
            runtime=runtime,
            port_allocator=port_allocator,
            max_concurrent_jobs=self._max_concurrent_jobs,
        )

        self._worker_service = WorkerServiceImpl(self._job_manager)
        self._worker_dashboard = WorkerDashboard(
            self._worker_service,
            host="127.0.0.1",
            port=self._worker_port,
        )

        # Start worker in thread
        print("Starting worker server thread...")
        self._worker_thread = threading.Thread(
            target=self._run_worker,
            daemon=True,
        )
        self._worker_thread.start()
        time.sleep(1.0)  # Wait for server startup
        print(f"Worker server should be at http://127.0.0.1:{self._worker_port}")

        # --- Start Controller ---
        print("Starting controller components...")
        self._state = ControllerState()
        self._scheduler = Scheduler(
            self._state,
            self._dispatch_job,
            interval_seconds=0.5,
        )
        self._heartbeat_monitor = HeartbeatMonitor(
            self._state,
            self._send_heartbeat,
            self._on_worker_failed,
            interval_seconds=2.0,
        )
        self._controller_service = ControllerServiceImpl(
            self._state,
            self._scheduler,
            bundle_dir=self._bundle_dir,
        )
        self._controller_dashboard = ControllerDashboard(
            self._controller_service,
            host="127.0.0.1",
            port=self._controller_port,
        )

        # Start controller in thread
        print("Starting controller server thread...", flush=True)
        self._controller_thread = threading.Thread(
            target=self._run_controller,
            daemon=True,
        )
        self._controller_thread.start()
        print("Controller thread started, waiting...", flush=True)
        time.sleep(1.0)
        print(f"Controller server should be at http://127.0.0.1:{self._controller_port}", flush=True)

        # Start scheduler and heartbeat monitor
        print("Starting scheduler and heartbeat monitor...", flush=True)
        self._scheduler.start()
        print("Scheduler started", flush=True)
        self._heartbeat_monitor.start()
        print("Heartbeat monitor started", flush=True)

        # Create HTTP client
        print("Creating HTTP client...", flush=True)
        self._client = httpx.Client(
            base_url=f"http://127.0.0.1:{self._controller_port}",
            timeout=30.0,
        )
        print("HTTP client created", flush=True)

        # Register worker with controller
        self._worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        print(f"Registering worker {self._worker_id}...", flush=True)
        self._register_worker()
        print("Worker registered", flush=True)

        print(f"Controller: http://127.0.0.1:{self._controller_port}", flush=True)
        print(f"Worker: http://127.0.0.1:{self._worker_port}", flush=True)

        print("Cluster startup complete!", flush=True)
        return self

    def __exit__(self, *args):
        """Stop cluster and cleanup."""
        if self._client:
            self._client.close()

        if self._heartbeat_monitor:
            self._heartbeat_monitor.stop()

        if self._scheduler:
            self._scheduler.stop()

        if self._temp_dir:
            self._temp_dir.cleanup()

    def _run_worker(self):
        """Run worker server (blocking, for thread)."""
        try:
            uvicorn.run(
                self._worker_dashboard._app,
                host="127.0.0.1",
                port=self._worker_port,
                log_level="error",
            )
        except Exception as e:
            print(f"Worker server error: {e}")

    def _run_controller(self):
        """Run controller server (blocking, for thread)."""
        try:
            uvicorn.run(
                self._controller_dashboard._app,
                host="127.0.0.1",
                port=self._controller_port,
                log_level="error",
            )
        except Exception as e:
            print(f"Controller server error: {e}")

    def _dispatch_job(self, job, worker) -> bool:
        """Dispatch job to worker (called by scheduler in its thread)."""
        try:
            response = httpx.post(
                f"http://{worker.address}/fluster.cluster.WorkerService/RunJob",
                headers={"Content-Type": "application/json"},
                json={
                    "jobId": str(job.job_id),
                    "serializedEntrypoint": base64.b64encode(job.request.serialized_entrypoint).decode(),
                    "environment": {
                        "workspace": job.request.environment.workspace,
                        "envVars": dict(job.request.environment.env_vars),
                    },
                    "bundleGcsPath": job.request.bundle_gcs_path,
                    "resources": {
                        "cpu": job.request.resources.cpu,
                        "memory": job.request.resources.memory,
                    },
                },
                timeout=10.0,
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Dispatch failed: {e}")
            return False

    def _send_heartbeat(self, address: str):
        """Send heartbeat to worker (called by heartbeat monitor)."""
        try:
            response = httpx.post(
                f"http://{address}/fluster.cluster.WorkerService/HealthCheck",
                headers={"Content-Type": "application/json"},
                json={},
                timeout=5.0,
            )
            if response.status_code == 200:
                # Get job statuses from worker
                jobs_response = httpx.post(
                    f"http://{address}/fluster.cluster.WorkerService/ListJobs",
                    headers={"Content-Type": "application/json"},
                    json={},
                    timeout=5.0,
                )
                if jobs_response.status_code == 200:
                    jobs_data = jobs_response.json().get("jobs", [])
                    # Convert to proto-like response
                    return cluster_pb2.HeartbeatResponse(
                        jobs=[
                            cluster_pb2.JobStatus(
                                job_id=j.get("jobId", ""),
                                state=self._parse_job_state(j.get("state", "")),
                                exit_code=j.get("exitCode", 0),
                                error=j.get("error", ""),
                            )
                            for j in jobs_data
                        ],
                        timestamp_ms=int(time.time() * 1000),
                    )
            return None
        except Exception:
            return None

    def _parse_job_state(self, state_str: str) -> int:
        """Parse job state string to proto enum."""
        state_map = {
            "JOB_STATE_PENDING": cluster_pb2.JOB_STATE_PENDING,
            "JOB_STATE_BUILDING": cluster_pb2.JOB_STATE_BUILDING,
            "JOB_STATE_RUNNING": cluster_pb2.JOB_STATE_RUNNING,
            "JOB_STATE_SUCCEEDED": cluster_pb2.JOB_STATE_SUCCEEDED,
            "JOB_STATE_FAILED": cluster_pb2.JOB_STATE_FAILED,
            "JOB_STATE_KILLED": cluster_pb2.JOB_STATE_KILLED,
            "JOB_STATE_WORKER_FAILED": cluster_pb2.JOB_STATE_WORKER_FAILED,
            "JOB_STATE_UNSCHEDULABLE": cluster_pb2.JOB_STATE_UNSCHEDULABLE,
        }
        return state_map.get(state_str, cluster_pb2.JOB_STATE_UNSPECIFIED)

    def _on_worker_failed(self, worker_id: WorkerId, job_ids: list[JobId]):
        """Handle worker failure."""
        print(f"Worker {worker_id} failed with jobs: {job_ids}")

    def _register_worker(self):
        """Register worker with controller."""
        response = self._client.post(
            "/fluster.cluster.ControllerService/RegisterWorker",
            headers={"Content-Type": "application/json"},
            json={
                "workerId": self._worker_id,
                "address": f"127.0.0.1:{self._worker_port}",
                "resources": {"cpu": 4, "memory": "16g"},
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to register worker: {response.text}")

    def _get_bundle_blob(self) -> bytes:
        """Get workspace bundle (cached)."""
        if self._bundle_blob is not None:
            return self._bundle_blob

        bundle_path = Path(self._temp_dir.name) / "workspace.zip"
        create_workspace_bundle(self._workspace, bundle_path)
        self._bundle_blob = bundle_path.read_bytes()
        return self._bundle_blob

    def submit(
        self,
        fn,
        *args,
        name: str | None = None,
        timeout_seconds: int = 0,
        env_vars: dict[str, str] | None = None,
        cpu: int = 1,
        memory: str = "1g",
        scheduling_timeout_seconds: int = 0,
        **kwargs,
    ) -> str:
        """Submit a job to the cluster.

        Args:
            fn: Callable to execute
            *args: Positional arguments for fn
            name: Job name (defaults to function name)
            timeout_seconds: Job timeout
            env_vars: Environment variables
            cpu: Number of CPUs to request
            memory: Memory to request (e.g., "1g", "512m")
            scheduling_timeout_seconds: How long to wait for scheduling before marking UNSCHEDULABLE
            **kwargs: Keyword arguments for fn

        Returns:
            Job ID
        """
        entrypoint = Entrypoint(callable=fn, args=args, kwargs=kwargs)
        serialized = cloudpickle.dumps(entrypoint)

        response = self._client.post(
            "/fluster.cluster.ControllerService/LaunchJob",
            headers={"Content-Type": "application/json"},
            json={
                "name": name or fn.__name__,
                "serializedEntrypoint": base64.b64encode(serialized).decode(),
                "resources": {"cpu": cpu, "memory": memory},
                "environment": {
                    "workspace": "/app",
                    "envVars": env_vars or {},
                },
                "bundleBlob": base64.b64encode(self._get_bundle_blob()).decode(),
                "schedulingTimeoutSeconds": scheduling_timeout_seconds,
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to submit job: {response.text}")

        return response.json()["jobId"]

    def status(self, job_id: str) -> dict:
        """Get job status from controller."""
        response = self._client.post(
            "/fluster.cluster.ControllerService/GetJobStatus",
            headers={"Content-Type": "application/json"},
            json={"jobId": job_id},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get status: {response.text}")
        return response.json()["job"]

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

    def logs(self, job_id: str) -> list[str]:
        """Get job logs from worker."""
        # Find the worker that has this job
        status = self.status(job_id)
        worker_id = status.get("workerId")

        if not worker_id:
            return []

        # For now, query worker directly (in a real cluster, would route through controller)
        response = httpx.post(
            f"http://127.0.0.1:{self._worker_port}/fluster.cluster.WorkerService/FetchLogs",
            headers={"Content-Type": "application/json"},
            json={"jobId": job_id},
            timeout=10.0,
        )

        if response.status_code != 200:
            return []

        logs_data = response.json().get("logs", [])
        return [entry.get("data", "") for entry in logs_data]

    def kill(self, job_id: str) -> None:
        """Kill a job via controller."""
        response = self._client.post(
            "/fluster.cluster.ControllerService/TerminateJob",
            headers={"Content-Type": "application/json"},
            json={"jobId": job_id},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to kill job: {response.text}")

    @property
    def controller_url(self) -> str:
        return f"http://127.0.0.1:{self._controller_port}"

    @property
    def worker_url(self) -> str:
        return f"http://127.0.0.1:{self._worker_port}"


# =============================================================================
# EXAMPLE FUNCTIONS
# =============================================================================


def example_basic(cluster: ClusterContext):
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


def example_with_args(cluster: ClusterContext):
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


def example_concurrent(cluster: ClusterContext):
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


def example_kill(cluster: ClusterContext):
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


def example_resource_serialization(cluster: ClusterContext):
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


def example_scheduling_timeout(cluster: ClusterContext):
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


def example_small_job_skips_queue(cluster: ClusterContext):
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
def main(wait: bool):
    """Run all examples with a single shared cluster."""
    print("=" * 60)
    print("Fluster Cluster Example")
    print("=" * 60)
    print("\nNote: This example requires Docker to be running.")

    try:
        with ClusterContext(max_concurrent_jobs=3) as cluster:
            print(f"\nController dashboard: {cluster.controller_url}", flush=True)
            print(f"Worker dashboard: {cluster.worker_url}", flush=True)
            if wait:
                print("\nPress Ctrl+C to stop.\n", flush=True)

            print("About to run examples...", flush=True)
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
        print("\nMake sure Docker is running and try again.")
        raise


if __name__ == "__main__":
    main()
