#!/usr/bin/env python3
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

"""Iris cluster autoscaling smoke test.

This script provides end-to-end validation of an Iris cluster:
1. Starts the cluster (creates controller VM)
2. Establishes SSH tunnel to controller
3. Submits TPU jobs to exercise autoscaling
4. Logs results to stdout and structured log directory
5. Cleans up on success/failure/interrupt

Usage:
    # Basic smoke test (logs to .agents/logs/smoke-test-{timestamp}/)
    uv run python scripts/smoke-test.py --config examples/eu-west4.yaml

    # With custom log directory
    uv run python scripts/smoke-test.py --config examples/eu-west4.yaml \\
        --log-dir /path/to/logs

    # Custom timeout (45 min) for slow environments
    uv run python scripts/smoke-test.py --config examples/eu-west4.yaml --timeout 2700

    # Keep cluster running on failure for debugging
    uv run python scripts/smoke-test.py --config examples/eu-west4.yaml --no-cleanup-on-failure
"""

import logging
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TextIO

import click
from google.protobuf.json_format import MessageToJson

from iris.client import IrisClient
from iris.cluster.types import (
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    tpu_device,
)
from iris.cluster.vm.cluster_manager import ClusterManager, make_local_config
from iris.cluster.vm.config import load_config
from iris.cluster.vm.debug import (
    cleanup_iris_resources,
    discover_controller_vm,
    list_docker_containers,
    list_iris_tpus,
    stream_docker_logs,
)
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.rpc.proto_utils import format_accelerator_display

IRIS_ROOT = Path(__file__).parent.parent
DEFAULT_CONTROLLER_PORT = 10000
DEFAULT_JOB_TIMEOUT = 300  # 5 minutes; TPU slices are pre-warmed by earlier tests
SCHEDULING_POLL_INTERVAL_SECONDS = 5.0
WORKER_DISCOVERY_INTERVAL_SECONDS = 10.0


@dataclass
class LogArtifact:
    path: Path
    description: str


class LogTree:
    """Tracks log artifacts created during a smoke test run."""

    def __init__(self, root: Path):
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._artifacts: list[LogArtifact] = []

    @property
    def root(self) -> Path:
        return self._root

    def get_writer(self, name: str, description: str) -> Path:
        """Register and return path for a log artifact. Creates parent dirs."""
        path = self._root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts.append(LogArtifact(path=path, description=description))
        return path

    def get_dir(self, name: str, description: str) -> Path:
        """Register and return path for a directory artifact."""
        path = self._root / name
        path.mkdir(parents=True, exist_ok=True)
        self._artifacts.append(LogArtifact(path=path, description=description))
        return path

    def summary_lines(self) -> list[str]:
        lines = ["## Artifacts", ""]
        for artifact in self._artifacts:
            rel = artifact.path.relative_to(self._root)
            if artifact.path.is_dir():
                count = sum(1 for _ in artifact.path.iterdir()) if artifact.path.exists() else 0
                status = f"{count} files"
            else:
                exists = artifact.path.exists()
                size = artifact.path.stat().st_size if exists else 0
                status = f"{size:,} bytes" if exists else "not created"
            lines.append(f"- **{artifact.description}:** `{rel}` ({status})")
        return lines


# =============================================================================
# Test Job Definitions
# =============================================================================


def _hello_tpu_job():
    """Simple job that prints and returns."""
    print("Hello from TPU!")
    return 42


def _quick_task_job(task_id: int):
    """Quick job that sleeps and returns."""
    import time as time_module

    time_module.sleep(2.0)
    print(f"Task {task_id} completed")
    return task_id


def _distributed_work_job():
    """Coscheduled job that uses job context."""
    from iris.cluster.client import get_job_info

    info = get_job_info()
    if info is None:
        raise RuntimeError("Not running in an Iris job context")
    print(f"Task {info.task_index} of {info.num_tasks} on worker {info.worker_id}")
    return f"Task {info.task_index} done"


def _jax_tpu_job():
    """Initialize JAX on TPU and run simple computation."""
    import os
    import sys

    print(f"Python: {sys.executable} {sys.version}", flush=True)
    print(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '<unset>')}", flush=True)
    print(f"PJRT_DEVICE={os.environ.get('PJRT_DEVICE', '<unset>')}", flush=True)
    print(f"TPU_NAME={os.environ.get('TPU_NAME', '<unset>')}", flush=True)
    print(f"TPU_WORKER_ID={os.environ.get('TPU_WORKER_ID', '<unset>')}", flush=True)
    print(f"TPU_WORKER_HOSTNAMES={os.environ.get('TPU_WORKER_HOSTNAMES', '<unset>')}", flush=True)
    print(f"TPU_CHIPS_PER_HOST_BOUNDS={os.environ.get('TPU_CHIPS_PER_HOST_BOUNDS', '<unset>')}", flush=True)

    # Check /dev/vfio exists
    vfio_path = "/dev/vfio"
    if os.path.exists(vfio_path):
        print(f"/dev/vfio exists, contents: {os.listdir(vfio_path)}", flush=True)
    else:
        print("/dev/vfio does NOT exist", flush=True)

    # Check jax is importable
    print("Importing jax...", flush=True)
    import jax

    print(f"JAX version: {jax.__version__}", flush=True)

    import jax.numpy as jnp

    # Verify TPU is available
    print("Calling jax.devices()...", flush=True)
    devices = jax.devices()
    tpu_devices = [d for d in devices if d.platform == "tpu"]
    if not tpu_devices:
        raise RuntimeError(f"No TPU devices found. Available: {[d.platform for d in devices]}")

    print(f"Found {len(tpu_devices)} TPU device(s): {tpu_devices}")

    # Simple computation to exercise TPU
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    result = float(y[0, 0])

    print(f"JAX TPU computation successful: {result}")
    return result


def _configure_logging():
    """Configure logging to show all iris module output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # Ensure iris modules are visible
    logging.getLogger("iris").setLevel(logging.INFO)


# =============================================================================
# Logging Infrastructure
# =============================================================================


class SmokeTestLogger:
    """Dual-output logger with timestamps and elapsed time.

    Writes both to stdout (with timestamps) and to a markdown file with
    structured sections for easy post-mortem analysis.
    """

    def __init__(self, log_tree: LogTree):
        self._start_time = time.monotonic()
        self._start_datetime = datetime.now()
        self._log_tree = log_tree
        summary_path = log_tree.get_writer("summary.md", "Execution summary")
        self._file: TextIO = open(summary_path, "w")

    def close(self):
        self._file.close()

    def write_header(self, config: "SmokeTestConfig"):
        """Write the markdown document header with run configuration."""
        self._file.write("# Smoke Test Execution Log\n\n")
        self._file.write(f"**Started:** {self._start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self._file.write(f"**Config:** `{config.config_path}`\n\n")
        self._file.write(f"**TPU type:** `{config.tpu_type}`\n\n")
        if config.prefix:
            self._file.write(f"**Prefix:** `{config.prefix}`\n\n")
        self._file.write(f"**Log directory:** `{config.log_dir}`\n\n")
        self._file.write("---\n\n")
        self._file.flush()

    def write_artifacts(self, log_tree: LogTree):
        """Write the artifacts section from the log tree."""
        self._file.write("\n---\n\n")
        for line in log_tree.summary_lines():
            self._file.write(line + "\n")
        self._file.flush()

    def log(self, message: str, level: str = "INFO"):
        now = datetime.now()
        elapsed = time.monotonic() - self._start_time
        line = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [{elapsed:8.1f}s] [{level}] {message}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def section(self, title: str):
        """Write a markdown section header."""
        self.log("")
        self._file.write(f"\n## {title}\n\n")
        self._file.flush()
        print("=" * 60, flush=True)
        print(f" {title}", flush=True)
        print("=" * 60, flush=True)
        self.log("")


class DockerLogStreamer:
    """Background thread that streams docker logs from controller or workers.

    Supports two modes:
    - "controller": Streams logs from single controller VM
    - "workers": Discovers TPU workers and streams from all
    """

    def __init__(
        self,
        mode: Literal["controller", "workers"],
        zone: str,
        project: str,
        log_tree: LogTree,
        container_name: str = "iris-worker",
        label_prefix: str = "iris",
    ):
        self._mode = mode
        self._zone = zone
        self._project = project
        self._log_tree = log_tree
        self._container_name = container_name
        self._label_prefix = label_prefix
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._streaming: set[tuple[str, str]] = set()  # (vm_name, container_name) pairs

    def start(self):
        """Start background streaming thread."""
        self._stop_event.clear()
        if self._mode == "controller":
            self._thread = threading.Thread(target=self._stream_controller, daemon=True)
        else:
            self._thread = threading.Thread(target=self._discover_and_stream_workers, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop streaming and wait for thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _stream_controller(self):
        """Stream logs from controller VM."""
        vm_name = discover_controller_vm(self._zone, self._project, self._label_prefix)
        if not vm_name:
            return
        log_file = self._log_tree.get_writer("controller-logs.txt", "Controller docker logs")
        stream_docker_logs(
            vm_name,
            self._container_name,
            self._zone,
            self._project,
            log_file,
            is_tpu=False,
            stop_event=self._stop_event,
        )

    def _start_stream(self, vm_name: str, container_name: str, is_tpu: bool):
        """Start streaming a container's logs if not already streaming."""
        key = (vm_name, container_name)
        if key in self._streaming:
            return
        self._streaming.add(key)
        log_file = self._log_tree.get_writer(
            f"workers/{vm_name}/{container_name}.txt",
            f"Worker logs: {vm_name}/{container_name}",
        )
        threading.Thread(
            target=stream_docker_logs,
            args=(vm_name, container_name, self._zone, self._project, log_file),
            kwargs={"is_tpu": is_tpu, "stop_event": self._stop_event},
            daemon=True,
        ).start()

    def _discover_and_stream_workers(self):
        """Discovery loop: find workers and stream logs from service + per-task containers."""
        while not self._stop_event.is_set():
            tpu_names = list_iris_tpus(self._zone, self._project, self._label_prefix)
            for tpu_name in tpu_names:
                # Stream the main worker service container
                self._start_stream(tpu_name, self._container_name, is_tpu=True)

                # Discover and stream per-task containers (labeled iris.managed=true)
                try:
                    task_containers = list_docker_containers(
                        tpu_name,
                        self._zone,
                        self._project,
                        label_filter="iris.managed=true",
                        is_tpu=True,
                    )
                    for container in task_containers:
                        self._start_stream(tpu_name, container, is_tpu=True)
                except Exception:
                    logging.debug("Failed to discover containers on %s", tpu_name, exc_info=True)
            self._stop_event.wait(WORKER_DISCOVERY_INTERVAL_SECONDS)


class TaskSchedulingMonitor:
    """Background thread that polls task status and logs scheduling info."""

    def __init__(self, controller_url: str, log_tree: LogTree, logger: SmokeTestLogger):
        self._client = ControllerServiceClientSync(controller_url)
        self._scheduling_dir = log_tree.get_dir("scheduling", "Task scheduling snapshots")
        self._logger = logger
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracked_jobs: set[str] = set()

    def track_job(self, job_id: str):
        """Add a job to monitor."""
        self._tracked_jobs.add(job_id)

    def start(self):
        """Start background polling thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop polling."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._client.close()

    def _poll_loop(self):
        """Poll task status every 5 seconds."""
        while not self._stop_event.is_set():
            for job_id in list(self._tracked_jobs):
                try:
                    self._poll_job_tasks(job_id)
                except Exception as e:
                    self._logger.log(f"Error polling job {job_id}: {e}", level="WARN")
            self._stop_event.wait(SCHEDULING_POLL_INTERVAL_SECONDS)

    def _poll_job_tasks(self, job_id: str):
        """Poll tasks for a specific job and log scheduling info."""
        request = cluster_pb2.Controller.ListTasksRequest(job_id=job_id)
        response = self._client.list_tasks(request)

        # Log pending tasks with reasons
        pending_tasks = [t for t in response.tasks if t.state == cluster_pb2.TASK_STATE_PENDING]

        if pending_tasks:
            for task in pending_tasks:
                if task.pending_reason:
                    self._logger.log(f"  Task {task.task_index} pending: {task.pending_reason}", level="DEBUG")

        snapshot_file = self._scheduling_dir / f"{job_id}-{int(time.time())}.json"
        with open(snapshot_file, "w") as f:
            f.write(MessageToJson(response, preserving_proto_field_name=True))


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SmokeTestConfig:
    """Configuration for the smoke test."""

    config_path: Path
    log_dir: Path
    timeout_seconds: int = 1800  # 30 min total
    job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT
    tpu_type: str = "v5litepod-16"
    cleanup_on_failure: bool = True
    clean_start: bool = True  # Delete existing resources before starting
    prefix: str | None = None  # Unique prefix for controller VM name (sets label_prefix in config)
    local: bool = False  # Run locally without GCP

    @property
    def label_prefix(self) -> str:
        return self.prefix or "iris"


# =============================================================================
# Image Building
# TODO: Refactor to use iris.build module directly instead of shelling out to CLI
# =============================================================================


def _build_and_push_image(image_type: str, region: str, project: str) -> bool:
    """Build and push a Docker image using the iris CLI.

    Args:
        image_type: "controller" or "worker"
        region: GCP Artifact Registry region
        project: GCP project ID

    Returns:
        True if build and push succeeded
    """
    cmd = [
        "uv",
        "run",
        "iris",
        "build",
        f"{image_type}-image",
        "-t",
        f"iris-{image_type}:latest",
        "--push",
        "--region",
        region,
        "--project",
        project,
    ]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# =============================================================================
# Test Result Tracking
# =============================================================================


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    message: str
    duration_seconds: float


# =============================================================================
# Main Test Runner
# =============================================================================


class SmokeTestRunner:
    """Orchestrates the smoke test lifecycle."""

    def __init__(self, config: SmokeTestConfig):
        self.config = config
        self.log_tree = LogTree(config.log_dir)
        self.logger = SmokeTestLogger(self.log_tree)
        self._manager: ClusterManager | None = None
        self._interrupted = False
        self._deadline: float | None = None
        self._results: list[TestResult] = []
        # Unique run ID to avoid job name collisions with previous runs
        self._run_id = datetime.now().strftime("%H%M%S")
        # Store zone/project for use in cleanup
        self._zone: str | None = None
        self._project: str | None = None
        # Store region/project for image building
        self._image_region: str | None = None
        self._image_project: str | None = None
        # Background monitoring threads
        self._controller_streamer: DockerLogStreamer | None = None
        self._worker_streamer: DockerLogStreamer | None = None
        self._scheduling_monitor: TaskSchedulingMonitor | None = None

    def run(self) -> bool:
        """Run the smoke test. Returns True if all tests pass."""
        # Configure logging to show subprocess output from iris modules
        _configure_logging()

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        self._deadline = time.monotonic() + self.config.timeout_seconds

        try:
            self._print_header()
            cluster_config = load_config(self.config.config_path)

            # Apply --local override if requested
            if self.config.local:
                cluster_config = make_local_config(cluster_config)

            # Apply prefix to cluster config for unique controller VM name
            if self.config.prefix:
                cluster_config.label_prefix = self.config.prefix

            manager = ClusterManager(cluster_config)
            self._manager = manager

            zone = cluster_config.zone
            project = cluster_config.project_id
            # Store for use in cleanup
            self._zone = zone
            self._project = project

            # GCP-only setup phases
            if not manager.is_local:
                # Extract region/project for image building
                self._image_region = zone.rsplit("-", 1)[0]  # "europe-west4-b" -> "europe-west4"
                self._image_project = project

                # Phase 0a: Build and push images
                self.logger.section("PHASE 0a: Building Images")
                if not self._build_images():
                    self.logger.log("Image build failed!", level="ERROR")
                    return False
                if self._interrupted or self._check_deadline():
                    return False

                # Phase 0b: Clean start (delete existing resources)
                if self.config.clean_start:
                    self.logger.section("PHASE 0b: Clean Start")
                    self._cleanup_existing(zone, project)
                    if self._interrupted or self._check_deadline():
                        return False

            # Start cluster and run tests via connect() (handles tunnel for GCP, direct for local)
            self.logger.section("Starting Cluster")
            with manager.connect() as url:
                if self._interrupted or self._check_deadline():
                    return False
                self.logger.log(f"Connected to controller at {url}")

                # GCP-only: start log streaming
                if not manager.is_local:
                    label_prefix = self.config.label_prefix
                    self._controller_streamer = DockerLogStreamer(
                        mode="controller",
                        zone=zone,
                        project=project,
                        log_tree=self.log_tree,
                        container_name="iris-controller",
                        label_prefix=label_prefix,
                    )
                    self._controller_streamer.start()
                    self.logger.log("Started controller log streaming")

                    self._worker_streamer = DockerLogStreamer(
                        mode="workers",
                        zone=zone,
                        project=project,
                        log_tree=self.log_tree,
                        container_name="iris-worker",
                        label_prefix=label_prefix,
                    )
                    self._worker_streamer.start()
                    self.logger.log("Started worker log streaming")

                # Start task scheduling monitor
                self._scheduling_monitor = TaskSchedulingMonitor(
                    controller_url=url,
                    log_tree=self.log_tree,
                    logger=self.logger,
                )
                self._scheduling_monitor.start()
                self.logger.log("Started task scheduling monitor")

                # Run tests
                self.logger.section("Running Tests")
                self._run_tests(url)

                # Results
                self.logger.section("Results Summary")
                success = self._print_results()

            return success

        except Exception as e:
            self.logger.log(f"FATAL ERROR: {e}", level="ERROR")
            return False

        finally:
            self._cleanup()
            self.logger.close()

    def _handle_interrupt(self, _signum: int, _frame: object):
        self.logger.log("Interrupted! Cleaning up...", level="WARN")
        self._interrupted = True

    def _check_deadline(self) -> bool:
        """Returns True if the global deadline has passed. Sets interrupted flag."""
        if self._deadline is not None and time.monotonic() > self._deadline:
            self.logger.log(
                f"Global timeout ({self.config.timeout_seconds}s) exceeded!",
                level="ERROR",
            )
            self._interrupted = True
            return True
        return False

    def _print_header(self):
        self.logger.write_header(self.config)
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log(" IRIS CLUSTER SMOKE TEST")
        self.logger.log("=" * 60)
        self.logger.log("")
        self.logger.log(f"Config: {self.config.config_path}")
        self.logger.log(f"Timeout: {self.config.timeout_seconds}s")
        self.logger.log(f"TPU type: {self.config.tpu_type}")
        self.logger.log(f"Log directory: {self.config.log_dir}")
        if self.config.prefix:
            self.logger.log(f"Prefix: {self.config.prefix}")

    def _cleanup_existing(self, zone: str, project: str):
        """Delete existing iris resources (controller VM, TPU slices) for clean start."""
        label_prefix = self.config.label_prefix
        self.logger.log(f"Cleaning up existing resources for prefix '{label_prefix}'...")

        deleted = cleanup_iris_resources(zone, project, label_prefix=label_prefix, dry_run=False)

        if not deleted:
            self.logger.log("  No existing resources found")
        else:
            for resource in deleted:
                self.logger.log(f"  Deleted: {resource}")
            self.logger.log("Cleanup complete")

    def _build_images(self) -> bool:
        """Build and push controller and worker images."""
        if self._image_region is None or self._image_project is None:
            self.logger.log("Image region/project not set", level="ERROR")
            return False

        region = self._image_region
        project = self._image_project

        self.logger.log(f"Building controller image (region={region}, project={project})...")
        if not _build_and_push_image("controller", region, project):
            self.logger.log("Controller image build failed!", level="ERROR")
            return False
        self.logger.log("Controller image built and pushed")

        self.logger.log(f"Building worker image (region={region}, project={project})...")
        if not _build_and_push_image("worker", region, project):
            self.logger.log("Worker image build failed!", level="ERROR")
            return False
        self.logger.log("Worker image built and pushed")

        return True

    def _run_tests(self, controller_url: str):
        """Run test jobs against the cluster."""
        client = IrisClient.remote(controller_url, workspace=IRIS_ROOT)

        # Test 1: Simple TPU job
        self.logger.log(f"[Test 1/4] Simple TPU job ({self.config.tpu_type})")
        result = self._run_simple_tpu_job(client)
        self._results.append(result)
        self._log_autoscaler_status(controller_url)

        if self._interrupted or self._check_deadline():
            return

        # Test 2: Concurrent TPU jobs
        self.logger.log(f"[Test 2/4] Concurrent TPU jobs (3x {self.config.tpu_type})")
        result = self._run_concurrent_tpu_jobs(client)
        self._results.append(result)
        self._log_autoscaler_status(controller_url)

        if self._interrupted or self._check_deadline():
            return

        # Test 3: Coscheduled multi-task job
        self.logger.log(f"[Test 3/4] Coscheduled multi-task job ({self.config.tpu_type})")
        result = self._run_coscheduled_job(client)
        self._results.append(result)
        self._log_autoscaler_status(controller_url)

        if self._interrupted:
            return

        # Test 4: JAX TPU job - validates JAX can initialize and use TPU
        self.logger.log(f"[Test 4/4] JAX TPU job ({self.config.tpu_type})")
        result = self._run_jax_tpu_job(client)
        self._results.append(result)
        self._log_autoscaler_status(controller_url)

    def _print_task_logs_on_failure(self, job, max_lines: int = 50):
        """Print task logs when a job fails, to show build errors."""
        try:
            for task in job.tasks():
                logs = task.logs(max_lines=max_lines)
                if logs:
                    self.logger.log(f"  Task {task.task_index} logs (last {len(logs)} lines):")
                    for entry in logs[-max_lines:]:
                        self.logger.log(f"    [{entry.source}] {entry.data}")
        except Exception as e:
            self.logger.log(f"  Failed to fetch task logs: {e}", level="WARN")

    def _run_job_test(
        self,
        client: IrisClient,
        test_name: str,
        entrypoint: Entrypoint,
        job_name: str,
        resources: ResourceSpec,
        coscheduling: CoschedulingConfig | None = None,
        environment: EnvironmentSpec | None = None,
        timeout: int | None = None,
    ) -> TestResult:
        """Generic job runner that handles submission, waiting, and result collection."""
        job_timeout = timeout or self.config.job_timeout_seconds
        start = time.monotonic()
        try:
            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=resources,
                environment=environment or EnvironmentSpec(),
                coscheduling=coscheduling,
            )
            self.logger.log(f"  Job submitted: {job.job_id}")

            if self._scheduling_monitor:
                self._scheduling_monitor.track_job(job.job_id)

            status = job.wait(timeout=job_timeout, raise_on_failure=False)
            duration = time.monotonic() - start

            if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                self.logger.log(f"  [PASS] Completed in {duration:.1f}s")
                return TestResult(test_name, True, f"Completed in {duration:.1f}s", duration)
            else:
                state_name = cluster_pb2.JobState.Name(status.state)
                self.logger.log(f"  [FAIL] Job ended with state {state_name}", level="ERROR")
                self._print_task_logs_on_failure(job)
                return TestResult(test_name, False, f"State: {state_name}, error: {status.error}", duration)

        except TimeoutError:
            duration = time.monotonic() - start
            self.logger.log(f"  [FAIL] Timed out after {job_timeout}s", level="ERROR")
            return TestResult(test_name, False, f"Timed out after {job_timeout}s", duration)

    def _submit_and_wait_multiple(
        self,
        client: IrisClient,
        jobs_config: list[tuple[Entrypoint, str, ResourceSpec]],
    ) -> tuple[float, list[str]]:
        """Submit multiple jobs and wait for all to complete.

        Returns:
            (duration, failed_job_descriptions)
        """
        start = time.monotonic()
        jobs = []

        for entrypoint, name, resources in jobs_config:
            job = client.submit(
                entrypoint=entrypoint,
                name=name,
                resources=resources,
                environment=EnvironmentSpec(),
            )
            jobs.append(job)
            self.logger.log(f"  Job submitted: {job.job_id}")

            if self._scheduling_monitor:
                self._scheduling_monitor.track_job(job.job_id)

        failed_jobs = []
        for job in jobs:
            status = job.wait(timeout=self.config.job_timeout_seconds, raise_on_failure=False)
            if status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
                state_name = cluster_pb2.JobState.Name(status.state)
                failed_jobs.append(f"{job.job_id}: {state_name}")

        return time.monotonic() - start, failed_jobs

    def _run_simple_tpu_job(self, client: IrisClient) -> TestResult:
        """Run a simple TPU job that just prints and returns."""
        return self._run_job_test(
            client=client,
            test_name=f"Simple TPU job ({self.config.tpu_type})",
            entrypoint=Entrypoint.from_callable(_hello_tpu_job),
            job_name=f"smoke-simple-{self._run_id}",
            resources=ResourceSpec(device=tpu_device(self.config.tpu_type)),
        )

    def _run_concurrent_tpu_jobs(self, client: IrisClient) -> TestResult:
        """Submit 3 concurrent TPU jobs to test parallel provisioning and queueing."""
        resources = ResourceSpec(device=tpu_device(self.config.tpu_type))
        jobs_config = [
            (Entrypoint.from_callable(_quick_task_job, i), f"smoke-concurrent-{self._run_id}-{i}", resources)
            for i in range(3)
        ]

        try:
            duration, failed_jobs = self._submit_and_wait_multiple(client, jobs_config)

            if not failed_jobs:
                self.logger.log(f"  [PASS] All 3 jobs completed in {duration:.1f}s")
                return TestResult("Concurrent TPU jobs (3x)", True, f"All completed in {duration:.1f}s", duration)
            else:
                self.logger.log(f"  [FAIL] Some jobs failed: {', '.join(failed_jobs)}", level="ERROR")
                return TestResult("Concurrent TPU jobs (3x)", False, f"Failed: {', '.join(failed_jobs)}", duration)

        except TimeoutError:
            self.logger.log("  [FAIL] Timed out waiting for jobs", level="ERROR")
            return TestResult(
                "Concurrent TPU jobs (3x)", False, f"Timed out after {self.config.job_timeout_seconds}s", 0.0
            )

    def _run_coscheduled_job(self, client: IrisClient) -> TestResult:
        """Run a coscheduled multi-task job on TPU workers."""
        return self._run_job_test(
            client=client,
            test_name="Coscheduled multi-task job",
            entrypoint=Entrypoint.from_callable(_distributed_work_job),
            job_name=f"smoke-coscheduled-{self._run_id}",
            resources=ResourceSpec(
                replicas=4,
                device=tpu_device(self.config.tpu_type),
            ),
            coscheduling=CoschedulingConfig(group_by="tpu-name"),
        )

    def _run_jax_tpu_job(self, client: IrisClient) -> TestResult:
        """Run a JAX TPU job that initializes JAX and exercises the TPU.

        Uses coscheduling with replicas=4 because multi-host TPU pods (e.g. v5litepod-16)
        require all hosts to run JAX simultaneously for collective initialization.
        """
        return self._run_job_test(
            client=client,
            test_name=f"JAX TPU job ({self.config.tpu_type})",
            entrypoint=Entrypoint.from_callable(_jax_tpu_job),
            job_name=f"smoke-jax-tpu-{self._run_id}",
            resources=ResourceSpec(
                replicas=4,
                device=tpu_device(self.config.tpu_type),
            ),
            environment=EnvironmentSpec(pip_packages=["jax[tpu]"]),
            coscheduling=CoschedulingConfig(group_by="tpu-name"),
            timeout=300,
        )

    def _log_autoscaler_status(self, controller_url: str):
        """Log current autoscaler state for observability."""
        rpc_client = ControllerServiceClientSync(controller_url)
        try:
            request = cluster_pb2.Controller.GetAutoscalerStatusRequest()
            response = rpc_client.get_autoscaler_status(request)

            status = response.status
            if status.current_demand:
                demand_str = ", ".join(f"{k}={v}" for k, v in status.current_demand.items())
                self.logger.log(f"  Autoscaler demand: {demand_str}")

            for group in status.groups:
                cfg = group.config
                accel = format_accelerator_display(cfg.accelerator_type, cfg.accelerator_variant)
                self.logger.log(
                    f"  Scale group {group.name}: demand={group.current_demand}, "
                    f"slices={cfg.min_slices}-{cfg.max_slices}, "
                    f"accelerator={accel}"
                )
        except Exception as e:
            self.logger.log(f"  (Could not fetch autoscaler status: {e})")
        finally:
            rpc_client.close()

    def _print_results(self) -> bool:
        """Print final results and return True if all passed."""
        all_passed = True
        total_duration = 0.0

        for result in self._results:
            status = "PASS" if result.passed else "FAIL"
            self.logger.log(f"  [{status}] {result.name}: {result.message}")
            total_duration += result.duration_seconds
            if not result.passed:
                all_passed = False

        self.logger.log("")
        passed_count = sum(1 for r in self._results if r.passed)
        total_count = len(self._results)

        if all_passed:
            self.logger.log(f"Results: {passed_count}/{total_count} tests passed in {total_duration:.1f}s")
        else:
            self.logger.log(f"Results: {passed_count}/{total_count} tests passed in {total_duration:.1f}s", level="WARN")

        self.logger.write_artifacts(self.log_tree)
        return all_passed

    def _cleanup(self):
        """Clean up cluster resources."""
        self.logger.section("CLEANUP")

        # Stop background monitoring threads
        if self._controller_streamer:
            self._controller_streamer.stop()
        if self._worker_streamer:
            self._worker_streamer.stop()
        if self._scheduling_monitor:
            self._scheduling_monitor.stop()
        self.logger.log("Stopped background monitoring")

        should_cleanup = self.config.cleanup_on_failure or all(r.passed for r in self._results)

        if not should_cleanup:
            self.logger.log("Skipping cleanup (--no-cleanup-on-failure and tests failed)")
            self.logger.log("Controller VM left running for debugging")
            return

        if self._manager:
            self.logger.log("Stopping cluster...")
            try:
                self._manager.stop()
                self.logger.log("Cluster stopped")
            except Exception as e:
                self.logger.log(f"Error stopping cluster: {e}", level="WARN")

        # Delete any remaining TPU slices and controller VM (GCP only)
        if self._zone and self._project and (not self._manager or not self._manager.is_local):
            label_prefix = self.config.label_prefix
            deleted = cleanup_iris_resources(self._zone, self._project, label_prefix=label_prefix, dry_run=False)
            for resource in deleted:
                self.logger.log(f"  Deleted: {resource}")

        self.logger.log("Done")


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to cluster config YAML (e.g., examples/eu-west4.yaml)",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    default=1800,
    help="Total timeout in seconds (default: 1800 = 30 min)",
)
@click.option(
    "--job-timeout",
    "job_timeout_seconds",
    default=DEFAULT_JOB_TIMEOUT,
    help=f"Per-job timeout in seconds (default: {DEFAULT_JOB_TIMEOUT} = 10 min)",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    help="Log directory path (default: .agents/logs/smoke-test-{timestamp})",
)
@click.option(
    "--tpu-type",
    default="v5litepod-16",
    help="TPU type for test jobs (default: v5litepod-16)",
)
@click.option(
    "--no-cleanup-on-failure",
    is_flag=True,
    help="Keep cluster running on failure for debugging",
)
@click.option(
    "--no-clean-start",
    is_flag=True,
    help="Skip deleting existing resources before starting",
)
@click.option(
    "--prefix",
    type=str,
    default=None,
    help="Unique prefix for controller VM name (e.g., 'smoke-123' creates 'iris-controller-smoke-123')",
)
@click.option(
    "--local",
    is_flag=True,
    help="Run locally without GCP (in-process controller and workers)",
)
def main(
    config_path: Path,
    timeout_seconds: int,
    job_timeout_seconds: int,
    log_dir: Path | None,
    tpu_type: str,
    no_cleanup_on_failure: bool,
    no_clean_start: bool,
    prefix: str | None,
    local: bool,
):
    """Run Iris cluster autoscaling smoke test.

    This script starts a cluster, submits TPU jobs to exercise autoscaling,
    and validates that everything works correctly. On completion (or failure),
    the cluster is cleaned up unless --no-cleanup-on-failure is specified.

    Examples:

        # Basic smoke test
        uv run python scripts/smoke-test.py --config examples/eu-west4.yaml

        # With custom log directory
        uv run python scripts/smoke-test.py --config examples/eu-west4.yaml \\
            --log-dir /path/to/logs

        # Custom timeout (45 min) for slow environments
        uv run python scripts/smoke-test.py --config examples/eu-west4.yaml --timeout 2700

        # Keep cluster running on failure for debugging
        uv run python scripts/smoke-test.py --config examples/eu-west4.yaml --no-cleanup-on-failure
    """
    # Create default log directory with timestamp if not provided
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = Path(".agents") / "logs" / f"smoke-test-{timestamp}"

    config = SmokeTestConfig(
        config_path=config_path,
        timeout_seconds=timeout_seconds,
        job_timeout_seconds=job_timeout_seconds,
        log_dir=log_dir,
        tpu_type=tpu_type,
        cleanup_on_failure=not no_cleanup_on_failure,
        clean_start=not no_clean_start,
        prefix=prefix,
        local=local,
    )

    runner = SmokeTestRunner(config)
    success = runner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
