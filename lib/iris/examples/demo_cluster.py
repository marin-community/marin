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

This script boots an iris cluster (in-process by default, no Docker required),
seeds it with quick demo jobs, and optionally launches a Jupyter notebook for
interactive exploration.

Usage:
    # Validate that the cluster works (for CI)
    cd lib/iris
    uv run python examples/demo_cluster.py --validate-only

    # Launch interactive demo with Jupyter
    uv run python examples/demo_cluster.py

    # Use Docker instead of in-process execution
    uv run python examples/demo_cluster.py --docker
"""

import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import click
from iris.client import IrisClient
from iris.cluster.client.local_client import (
    LocalEnvironmentProvider,
    _LocalBundleProvider,
    _LocalContainerRuntime,
    _LocalImageProvider,
)
from iris.cluster.controller.controller import Controller, ControllerConfig, DefaultWorkerStubFactory
from iris.cluster.types import EnvironmentSpec, Entrypoint, ResourceSpec
from iris.cluster.worker.builder import ImageCache
from iris.cluster.worker.bundle_cache import BundleCache
from iris.cluster.worker.docker import DockerRuntime
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

# The iris project root (lib/iris/) - used as workspace for the example
IRIS_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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
        num_workers: int = 1,
    ):
        self._use_docker = use_docker
        self._num_workers = num_workers
        self._controller_port = find_free_port()

        # Will be initialized in __enter__
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._controller: Controller | None = None
        self._workers: list[Worker] = []
        self._worker_ids: list[str] = []
        self._worker_ports: list[int] = []
        self._controller_client: ControllerServiceClientSync | None = None
        self._rpc_client: IrisClient | None = None

        # Jupyter integration
        self._notebook_proc: subprocess.Popen | None = None
        self._notebook_url: str | None = None

    def __enter__(self):
        """Start controller and worker."""
        # Clean up any orphaned containers from previous runs
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
            bundle_provider = BundleCache(cache_path, max_bundles=100)
            image_provider = ImageCache(cache_path, registry="", max_images=100)
            container_runtime = DockerRuntime()
            environment_provider = None  # Use default (probe real system)
        else:
            bundle_provider = _LocalBundleProvider(fake_bundle)
            image_provider = _LocalImageProvider()
            container_runtime = _LocalContainerRuntime()
            environment_provider = LocalEnvironmentProvider(cpu=1000, memory_gb=1000)

        # Start Workers
        for i in range(self._num_workers):
            worker_id = f"worker-{i}-{uuid.uuid4().hex[:8]}"
            worker_port = find_free_port()
            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=cache_path,
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
                environment_provider=environment_provider,
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

    @property
    def client(self) -> IrisClient:
        """IrisClient for this cluster."""
        if self._rpc_client is None:
            self._rpc_client = IrisClient.remote(
                self.controller_url,
                workspace=IRIS_ROOT,
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
    ):
        """Submit a job to the cluster and return Job handle."""
        entrypoint = Entrypoint.from_callable(fn, *args, **kwargs)
        environment = EnvironmentSpec(workspace="/app")
        resources = ResourceSpec(cpu=cpu, memory=memory)
        return self.client.submit(
            entrypoint=entrypoint,
            name=name or fn.__name__,
            resources=resources,
            environment=environment,
        )

    def seed_cluster(self) -> list[tuple[str, str]]:
        """Submit demo jobs to the cluster.

        Returns:
            List of (job_id, state) tuples for validation.
        """
        results = []

        def hello():
            print("Hello from iris!")
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
            job = self.submit(fn, *args, name=name, **kwargs)
            status = job.wait()
            results.append((str(job.job_id), cluster_pb2.JobState.Name(status.state)))
            print(f"  {name}: {cluster_pb2.JobState.Name(status.state)}")

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
        env["IRIS_CONTROLLER_ADDRESS"] = self.controller_url
        env["IRIS_WORKSPACE"] = str(IRIS_ROOT)

        # Find the demo notebook
        notebook_dir = IRIS_ROOT / "examples"
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
        os.environ["IRIS_CONTROLLER_ADDRESS"] = self.controller_url
        os.environ["IRIS_WORKSPACE"] = str(IRIS_ROOT)

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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(docker: bool, no_browser: bool, validate_only: bool, test_notebook: bool, verbose: bool):
    """Launch demo cluster with Jupyter notebook.

    By default, runs jobs in-process (no Docker required). Use --docker for
    real container execution.
    """
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

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
