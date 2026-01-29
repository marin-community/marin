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
interactive exploration. It can also connect to a remote controller via SSH tunnel.

Workers are created on-demand by the autoscaler when jobs are submitted. The
autoscaler manages two scale groups:
- cpu: For jobs without device requirements
- tpu_v5e_4: For TPU jobs (v5litepod-4 topology)

Usage:
    # Launch interactive demo with Jupyter
    uv run python examples/demo_cluster.py

    # Connect to remote controller via SSH tunnel
    uv run python examples/demo_cluster.py --controller-url http://localhost:10000
"""

import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import click
from iris.client import IrisClient
from iris.cluster.types import (
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    tpu_device,
)
from iris.cluster.vm.cluster_manager import ClusterManager, make_local_config
from iris.rpc import cluster_pb2, config_pb2

# The iris project root (lib/iris/) - used as workspace for the example
IRIS_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


# =============================================================================
# Demo Cluster
# =============================================================================


class DemoCluster:
    """Demo cluster with Jupyter integration.

    Supports two execution modes:
    - In-process (default): Fast, no Docker required, jobs run in threads
    - Docker: Real containers, matches production behavior
    - Remote: Connect to an existing controller (e.g., via SSH tunnel)

    Workers are created on-demand by the autoscaler. The autoscaler manages:
    - cpu: For jobs without device requirements (min_slices=1)
    - tpu_v5e_4: For TPU jobs (v5litepod-4 topology)

    Example:
        with DemoCluster() as demo:
            results = demo.seed_cluster()
            if demo.validate_seed_results(results):
                demo.launch_jupyter()
                demo.wait_for_interrupt()

        # Remote mode - connect to existing controller
        with DemoCluster(controller_url="http://localhost:10000") as demo:
            results = demo.seed_cluster()
    """

    def __init__(
        self,
        controller_url: str | None = None,
        config_path: str | None = None,
        workspace: Path | None = None,
    ):
        self._remote_url = controller_url
        self._config_path = config_path
        self._workspace = workspace or IRIS_ROOT
        self._manager: ClusterManager | None = None
        self._rpc_client: IrisClient | None = None

        # Jupyter integration
        self._notebook_proc: subprocess.Popen | None = None
        self._notebook_url: str | None = None

    def _load_or_default_config(self) -> config_pb2.IrisClusterConfig:
        """Load config from file or build a default demo config."""
        if self._config_path:
            from iris.cluster.vm.config import load_config

            return load_config(Path(self._config_path))

        # Build default demo config programmatically
        config = config_pb2.IrisClusterConfig()
        cpu_sg = config.scale_groups["cpu"]
        cpu_sg.name = "cpu"
        cpu_sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
        cpu_sg.min_slices = 0
        cpu_sg.max_slices = 4

        tpu_sg = config.scale_groups["tpu_v5e_16"]
        tpu_sg.name = "tpu_v5e_16"
        tpu_sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
        tpu_sg.accelerator_variant = "v5litepod-16"
        tpu_sg.min_slices = 0
        tpu_sg.max_slices = 4

        return config

    def __enter__(self):
        """Start controller with autoscaler (or connect to remote)."""
        if self._remote_url:
            return self

        config = self._load_or_default_config()
        config = make_local_config(config)
        self._manager = ClusterManager(config)
        self._manager.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop cluster and cleanup."""
        del exc_type, exc_val, exc_tb  # unused
        self._stop_jupyter()
        self._rpc_client = None
        if self._manager:
            self._manager.stop()

    def _stop_jupyter(self):
        if self._notebook_proc:
            self._notebook_proc.terminate()
            try:
                self._notebook_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._notebook_proc.kill()
            self._notebook_proc = None

    @property
    def controller_url(self) -> str:
        if self._remote_url:
            return self._remote_url
        if self._manager:
            url = self._manager.controller.discover()
            if url:
                return url
        raise RuntimeError("No controller URL available. Call __enter__ first.")

    @property
    def client(self) -> IrisClient:
        """IrisClient for this cluster."""
        if self._rpc_client is None:
            self._rpc_client = IrisClient.remote(
                self.controller_url,
                workspace=self._workspace,
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
        environment = EnvironmentSpec()
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

        def distributed_work():
            from iris.cluster.client import get_job_info

            info = get_job_info()
            if info is None:
                raise RuntimeError("Not running in an Iris job context")
            print(f"Task {info.task_index} of {info.num_tasks} on worker {info.worker_id}")
            return f"Task {info.task_index} done"

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

        # Coscheduled TPU job
        job = self.client.submit(
            entrypoint=Entrypoint.from_callable(distributed_work),
            name="demo-coscheduled",
            resources=ResourceSpec(
                cpu=1,
                memory="512m",
                replicas=4,
                device=tpu_device("v5litepod-16"),
            ),
            environment=EnvironmentSpec(),
            coscheduling=CoschedulingConfig(group_by="tpu-name"),
        )
        status = job.wait()
        results.append((str(job.job_id), cluster_pb2.JobState.Name(status.state)))
        print(f"  demo-coscheduled: {cluster_pb2.JobState.Name(status.state)}")

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
        env["IRIS_WORKSPACE"] = str(self._workspace)

        # Find the demo notebook (always in IRIS_ROOT, not workspace)
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
        os.environ["IRIS_WORKSPACE"] = str(self._workspace)

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
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser for Jupyter")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--controller-url",
    help="Connect to remote controller (e.g., http://localhost:10000). Skips local cluster creation.",
)
@click.option(
    "--workspace",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace directory (default: lib/iris)",
)
def main(
    no_browser: bool,
    verbose: bool,
    controller_url: str | None,
    workspace: Path | None,
):
    """Launch demo cluster with Jupyter notebook.

    Runs in-process (no Docker required), with jobs running in threads.
    Can also connect to a remote controller via SSH tunnel.

    Examples:
        # Launch interactive demo with Jupyter
        uv run python examples/demo_cluster.py

        # Connect to remote controller via SSH tunnel
        uv run python examples/demo_cluster.py --controller-url http://localhost:10000
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

    if controller_url:
        print(f"Connecting to remote controller: {controller_url}")
    else:
        print("Starting demo cluster with autoscaler...")

    with DemoCluster(controller_url=controller_url, workspace=workspace) as demo:
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

        print()
        print("Testing notebook execution...")
        if not demo.run_notebook():
            print("Notebook test FAILED!")
            sys.exit(1)
        print("Notebook test passed!")

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
