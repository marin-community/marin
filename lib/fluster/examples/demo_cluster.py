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

"""Demo cluster with Jupyter notebook integration.

This script boots a fluster cluster (in-process by default, no Docker required),
seeds it with quick demo jobs, and optionally launches a Jupyter notebook for
interactive exploration.

Usage:
    # Validate that the cluster works (for CI)
    cd lib/fluster
    uv run python examples/demo_cluster.py --validate-only

    # Launch interactive demo with Jupyter
    uv run python examples/demo_cluster.py

    # Use Docker instead of in-process execution
    uv run python examples/demo_cluster.py --docker
"""

import base64
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import click
import cloudpickle

from fluster.client import FlusterClient
from fluster.cluster.controller.controller import Controller, ControllerConfig, DefaultWorkerStubFactory
from fluster.cluster.types import Entrypoint
from fluster.cluster.worker.builder import BuildResult, ImageCache
from fluster.cluster.worker.bundle_cache import BundleCache
from fluster.cluster.worker.docker import ContainerConfig, ContainerStats, ContainerStatus, DockerRuntime
from fluster.cluster.worker.worker import Worker, WorkerConfig
from fluster.cluster.worker.worker_types import LogLine
from fluster.rpc import cluster_pb2
from fluster.rpc.cluster_connect import ControllerServiceClientSync, WorkerServiceClientSync

# The fluster project root (lib/fluster/) - used as workspace for the example
FLUSTER_ROOT = Path(__file__).parent.parent


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# =============================================================================
# Mock Infrastructure (from test_e2e.py for in-process execution)
# =============================================================================


@dataclass
class MockContainer:
    """Simulates a container executing a job function in-process."""

    config: ContainerConfig
    _thread: threading.Thread | None = field(default=None, repr=False)
    _running: bool = False
    _exit_code: int | None = None
    _error: str | None = None
    _logs: list[LogLine] = field(default_factory=list)
    _killed: threading.Event = field(default_factory=threading.Event)

    def start(self):
        """Execute the job function in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._execute, daemon=True)
        self._thread.start()

    def _execute(self):
        original_env = {}
        try:
            # Set environment variables from container config
            for key, value in self.config.env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

            # Extract encoded data from command (command is ['python', '-c', script])
            script = self.config.command[2]
            fn, args, kwargs = self._extract_entrypoint(script)

            # Check if killed before executing
            if self._killed.is_set():
                self._exit_code = 137
                return

            # Execute the function
            fn(*args, **kwargs)
            self._exit_code = 0

        except Exception as e:
            self._error = str(e)
            self._exit_code = 1
        finally:
            self._running = False
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def _extract_entrypoint(self, script: str):
        """Extract pickled (fn, args, kwargs) from the thunk script."""
        match = re.search(r"base64\.b64decode\('([^']+)'\)\)", script)
        if match:
            encoded = match.group(1)
            return cloudpickle.loads(base64.b64decode(encoded))
        raise ValueError("Could not extract entrypoint from command")

    def kill(self):
        self._killed.set()
        # Give thread a moment to notice
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._running:
            self._running = False
            self._exit_code = 137


class InProcessContainerRuntime:
    """Container runtime that executes jobs in-process without Docker.

    Implements the ContainerRuntime protocol for testing.
    """

    def __init__(self):
        self._containers: dict[str, MockContainer] = {}

    def create_container(self, config: ContainerConfig) -> str:
        container_id = f"mock-{uuid.uuid4().hex[:8]}"
        self._containers[container_id] = MockContainer(config=config)
        return container_id

    def start_container(self, container_id: str) -> None:
        self._containers[container_id].start()

    def inspect(self, container_id: str) -> ContainerStatus:
        c = self._containers.get(container_id)
        if not c:
            return ContainerStatus(running=False, exit_code=1, error="container not found")
        return ContainerStatus(
            running=c._running,
            exit_code=c._exit_code,
            error=c._error,
        )

    def kill(self, container_id: str, force: bool = False) -> None:
        del force  # unused
        if container_id in self._containers:
            self._containers[container_id].kill()

    def remove(self, container_id: str) -> None:
        self._containers.pop(container_id, None)

    def get_logs(self, container_id: str) -> list[LogLine]:
        c = self._containers.get(container_id)
        return c._logs if c else []

    def get_stats(self, container_id: str) -> ContainerStats:
        del container_id  # unused
        return ContainerStats(memory_mb=100, cpu_percent=10, process_count=1, available=True)


class MockBundleProvider:
    """Returns a fake bundle path without downloading."""

    def __init__(self, bundle_path: Path):
        self._bundle_path = bundle_path

    def get_bundle(self, gcs_path: str, expected_hash: str | None = None) -> Path:
        del gcs_path, expected_hash  # unused
        return self._bundle_path


class MockImageProvider:
    """Skips image building, returns a fake result."""

    def build(
        self,
        bundle_path: Path,
        base_image: str,
        extras: list[str],
        job_id: str,
        deps_hash: str,
        job_logs=None,
    ) -> BuildResult:
        del bundle_path, base_image, extras, job_id, job_logs  # unused
        return BuildResult(
            image_tag="mock:latest",
            deps_hash=deps_hash,
            build_time_ms=0,
            from_cache=True,
        )


# =============================================================================
# Log Poller (from cluster_example.py)
# =============================================================================


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

                except (ConnectionError, OSError):
                    # Expected when worker is starting or job doesn't exist yet
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


# =============================================================================
# Demo Cluster
# =============================================================================


class DemoCluster:
    """Demo cluster with Jupyter integration.

    Supports two execution modes:
    - In-process (default): Fast, no Docker required, jobs run in threads
    - Docker: Real containers, matches production behavior

    Example:
        with DemoCluster() as demo:
            results = demo.seed_cluster()
            if demo.validate_seed_results(results):
                demo.launch_jupyter()
                demo.wait_for_interrupt()
    """

    def __init__(
        self,
        use_docker: bool = False,
        max_concurrent_jobs: int = 3,
        num_workers: int = 1,
    ):
        self._use_docker = use_docker
        self._max_concurrent_jobs = max_concurrent_jobs
        self._num_workers = num_workers
        self._controller_port = find_free_port()

        # Will be initialized in __enter__
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._controller: Controller | None = None
        self._workers: list[Worker] = []
        self._worker_ids: list[str] = []
        self._worker_ports: list[int] = []
        self._controller_client: ControllerServiceClientSync | None = None
        self._rpc_client: FlusterClient | None = None

        # Jupyter integration
        self._notebook_proc: subprocess.Popen | None = None
        self._notebook_url: str | None = None

        # Log polling
        self._log_pollers: dict[str, LogPoller] = {}

    def __enter__(self):
        """Start controller and worker."""
        self._temp_dir = tempfile.TemporaryDirectory(prefix="demo_cluster_")
        temp_path = Path(self._temp_dir.name)
        bundle_dir = temp_path / "bundles"
        bundle_dir.mkdir()
        cache_path = temp_path / "cache"
        cache_path.mkdir()

        # Create fake bundle with minimal structure
        fake_bundle = temp_path / "fake_bundle"
        fake_bundle.mkdir()
        (fake_bundle / "pyproject.toml").write_text("[project]\nname = 'demo'\n")

        # Start Controller first (workers need to connect to it)
        controller_config = ControllerConfig(
            host="127.0.0.1",
            port=self._controller_port,
            bundle_dir=bundle_dir,
        )
        self._controller = Controller(
            config=controller_config,
            worker_stub_factory=DefaultWorkerStubFactory(),
        )
        self._controller.start()

        # Create RPC client
        self._controller_client = ControllerServiceClientSync(
            address=f"http://127.0.0.1:{self._controller_port}",
            timeout_ms=30000,
        )

        # Select providers based on use_docker flag
        if self._use_docker:
            bundle_provider = BundleCache(cache_path, max_bundles=10)
            image_provider = ImageCache(cache_path, registry="", max_images=10)
            container_runtime = DockerRuntime()
        else:
            bundle_provider = MockBundleProvider(fake_bundle)
            image_provider = MockImageProvider()
            container_runtime = InProcessContainerRuntime()

        # Start Workers
        for i in range(self._num_workers):
            worker_id = f"worker-{i}-{uuid.uuid4().hex[:8]}"
            worker_port = find_free_port()
            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=cache_path,
                max_concurrent_jobs=self._max_concurrent_jobs,
                controller_address=f"http://127.0.0.1:{self._controller_port}",
                worker_id=worker_id,
                poll_interval_seconds=0.1,  # Fast polling for demos
            )
            worker = Worker(
                worker_config,
                cache_dir=cache_path,
                bundle_provider=bundle_provider,
                image_provider=image_provider,
                container_runtime=container_runtime,
            )
            worker.start()
            self._workers.append(worker)
            self._worker_ids.append(worker_id)
            self._worker_ports.append(worker_port)

        # Wait for workers to register with controller. Workers send heartbeats
        # every 0.1s (poll_interval_seconds), and registration happens on first
        # heartbeat. 2s is conservative to handle slow CI environments.
        time.sleep(2.0)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop cluster and cleanup."""
        del exc_type, exc_val, exc_tb  # unused
        # Stop all log polling threads
        for job_id in list(self._log_pollers.keys()):
            self.stop_log_polling(job_id)

        # Stop Jupyter notebook
        if self._notebook_proc:
            self._notebook_proc.terminate()
            try:
                self._notebook_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._notebook_proc.kill()
            self._notebook_proc = None

        if self._rpc_client:
            self._rpc_client = None
        if self._controller_client:
            self._controller_client.close()
        for worker in self._workers:
            worker.stop()
        if self._controller:
            self._controller.stop()
        if self._temp_dir:
            self._temp_dir.cleanup()

    @property
    def controller_url(self) -> str:
        return f"http://127.0.0.1:{self._controller_port}"

    @property
    def worker_url(self) -> str:
        if self._worker_ports:
            return f"http://127.0.0.1:{self._worker_ports[0]}"
        return ""

    def get_client(self) -> FlusterClient:
        """Get a FlusterClient for this cluster."""
        if self._rpc_client is None:
            self._rpc_client = FlusterClient.remote(
                self.controller_url,
                workspace=FLUSTER_ROOT,
            )
        return self._rpc_client

    def submit(
        self,
        fn,
        *args,
        name: str | None = None,
        cpu: int = 1,
        memory: str = "1g",
        **kwargs,
    ) -> str:
        """Submit a job to the cluster."""
        entrypoint = Entrypoint.from_callable(fn, *args, **kwargs)
        environment = cluster_pb2.EnvironmentConfig(workspace="/app", env_vars={})
        resources = cluster_pb2.ResourceSpec(cpu=cpu, memory=memory)
        return self.get_client().submit(
            entrypoint=entrypoint,
            name=name or fn.__name__,
            resources=resources,
            environment=environment,
        )

    def status(self, job_id: str) -> dict:
        """Get job status from controller."""
        if self._controller_client is None:
            raise RuntimeError("Cluster not started - use 'with DemoCluster() as demo:'")
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self._controller_client.get_job_status(request)
        return {
            "jobId": response.job.job_id,
            "state": cluster_pb2.JobState.Name(response.job.state),
            "exitCode": response.job.exit_code,
            "error": response.job.error,
            "workerId": response.job.worker_id,
        }

    def wait(self, job_id: str, timeout: float = 60.0, poll_interval: float = 0.1) -> dict:
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

    def start_log_polling(self, job_id: str, poll_interval: float = 1.0, prefix: str = ""):
        """Start a background thread that polls for job logs."""
        if job_id in self._log_pollers:
            return  # Already polling

        poller = LogPoller(job_id, self.worker_url, poll_interval, prefix)
        poller.start()
        self._log_pollers[job_id] = poller

    def stop_log_polling(self, job_id: str):
        """Stop log polling for a job."""
        poller = self._log_pollers.pop(job_id, None)
        if poller:
            poller.stop()

    def seed_cluster(self) -> list[tuple[str, str]]:
        """Submit demo jobs to the cluster.

        Returns:
            List of (job_id, state) tuples for validation.
        """
        results = []

        def hello():
            print("Hello from fluster!")
            return 42

        def compute(a, b):
            result = a + b
            print(f"{a} + {b} = {result}")
            return result

        def countdown(n):
            for i in range(n, 0, -1):
                print(f"Countdown: {i}")
                time.sleep(0.3)
            print("Liftoff!")
            return "Done!"

        jobs = [
            (hello, [], {}, "demo-hello"),
            (compute, [10, 32], {}, "demo-compute"),
            (countdown, [3], {}, "demo-countdown"),
        ]

        for fn, args, kwargs, name in jobs:
            job_id = self.submit(fn, *args, name=name, **kwargs)
            status = self.wait(job_id)
            results.append((job_id, status["state"]))
            print(f"  {name}: {status['state']}")

        return results

    def validate_seed_results(self, results: list[tuple[str, str]]) -> bool:
        """Validate that seed jobs completed as expected."""
        expected = ["JOB_STATE_SUCCEEDED"] * len(results)
        actual = [r[1] for r in results]
        return actual == expected

    def launch_jupyter(self, open_browser: bool = True) -> str:
        """Start Jupyter notebook server and return URL.

        Args:
            open_browser: Whether to open the browser automatically

        Returns:
            Jupyter notebook URL
        """
        env = os.environ.copy()
        env["FLUSTER_CONTROLLER_ADDRESS"] = self.controller_url
        env["FLUSTER_WORKSPACE"] = str(FLUSTER_ROOT)

        # Find the demo notebook
        notebook_dir = FLUSTER_ROOT / "examples"
        notebook_path = notebook_dir / "demo.ipynb"

        browser_flag = [] if open_browser else ["--no-browser"]

        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "notebook",
            *browser_flag,
            f"--notebook-dir={notebook_dir}",
            str(notebook_path),
        ]

        self._notebook_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Extract Jupyter URL from output
        self._notebook_url = self._extract_jupyter_url()
        return self._notebook_url

    def _extract_jupyter_url(self, timeout: float = 30.0) -> str:
        """Extract the Jupyter URL from the notebook server output."""
        start = time.time()
        url_pattern = re.compile(r"(http://127\.0\.0\.1:\d+/\S*)")

        while time.time() - start < timeout:
            if self._notebook_proc is None or self._notebook_proc.poll() is not None:
                raise RuntimeError("Jupyter notebook process died unexpectedly")

            stdout = self._notebook_proc.stdout
            if stdout is None:
                raise RuntimeError("Jupyter notebook process has no stdout")

            line = stdout.readline()
            if line:
                match = url_pattern.search(line)
                if match:
                    return match.group(1)

        raise TimeoutError("Could not extract Jupyter URL within timeout")

    def wait_for_interrupt(self):
        """Wait for Ctrl+C, keeping the cluster running."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def run_notebook(self, notebook_path: Path | None = None) -> bool:
        """Execute the demo notebook and validate all cells succeed.

        Args:
            notebook_path: Path to notebook (default: demo.ipynb in same dir)

        Returns:
            True if all cells executed successfully, False otherwise.
        """
        import nbformat
        from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

        if notebook_path is None:
            notebook_path = Path(__file__).parent / "demo.ipynb"

        print(f"Running notebook: {notebook_path}")

        # Read the notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Set environment for the kernel (inherited by subprocess)
        os.environ["FLUSTER_CONTROLLER_ADDRESS"] = self.controller_url
        os.environ["FLUSTER_WORKSPACE"] = str(FLUSTER_ROOT)

        # Create executor
        ep = ExecutePreprocessor(
            timeout=120,
            kernel_name="python3",
        )

        try:
            # Execute the notebook
            ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
            print("All notebook cells executed successfully!")
            return True
        except CellExecutionError as e:
            print(f"Notebook execution failed: {e}")
            return False


@click.command()
@click.option("--docker", is_flag=True, help="Use Docker instead of in-process execution")
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser for Jupyter")
@click.option("--validate-only", is_flag=True, help="Run seed jobs and exit (for CI)")
@click.option("--test-notebook", is_flag=True, help="Run notebook programmatically and validate (for CI)")
def main(docker: bool, no_browser: bool, validate_only: bool, test_notebook: bool):
    """Launch demo cluster with Jupyter notebook.

    By default, runs jobs in-process (no Docker required). Use --docker for
    real container execution.
    """
    mode = "Docker" if docker else "in-process"
    print(f"Starting demo cluster ({mode} mode)...")

    with DemoCluster(use_docker=docker) as demo:
        print(f"Controller: {demo.controller_url}")
        print()
        print("Seeding cluster with demo jobs...")
        results = demo.seed_cluster()

        if not demo.validate_seed_results(results):
            print()
            print("ERROR: Seed jobs did not complete as expected!")
            for job_id, state in results:
                if state != "JOB_STATE_SUCCEEDED":
                    print(f"  {job_id}: {state}")
            sys.exit(1)

        print()
        print("All seed jobs succeeded!")

        if validate_only:
            print("Validation passed!")
            return

        if test_notebook:
            print()
            print("Testing notebook execution...")
            if demo.run_notebook():
                print("Notebook test passed!")
            else:
                print("Notebook test FAILED!")
                sys.exit(1)
            return

        print()
        print("Launching Jupyter notebook...")
        url = demo.launch_jupyter(open_browser=not no_browser)
        print(f"Notebook: {url}")
        print()
        print("Press Ctrl+C to stop.")
        demo.wait_for_interrupt()

    print("Shutting down...")


if __name__ == "__main__":
    main()
