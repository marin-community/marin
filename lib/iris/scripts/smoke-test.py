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
4. Logs results to stdout and optionally to file
5. Cleans up on success/failure/interrupt

Usage:
    # Basic smoke test
    uv run python scripts/smoke-test.py --config examples/eu-west4.yaml

    # With logging to file
    uv run python scripts/smoke-test.py --config examples/eu-west4.yaml \\
        --log-file smoke-test.log

    # Custom timeout (45 min) for slow environments
    uv run python scripts/smoke-test.py --config examples/eu-west4.yaml --timeout 2700

    # Keep cluster running on failure for debugging
    uv run python scripts/smoke-test.py --config examples/eu-west4.yaml --no-cleanup-on-failure
"""

import logging
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import click

from iris.client import IrisClient
from iris.cluster.types import (
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    tpu_device,
)
from iris.cluster.vm.config import load_config
from iris.cluster.vm.controller import ControllerProtocol, create_controller
from iris.cluster.vm.debug import (
    cleanup_iris_resources,
    collect_docker_logs,
    discover_controller_vm,
    list_iris_tpus,
)
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.rpc.proto_utils import format_accelerator_display

IRIS_ROOT = Path(__file__).parent.parent
DEFAULT_CONTROLLER_PORT = 10000
DEFAULT_JOB_TIMEOUT = 600  # 10 minutes for TPU provisioning


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
    """Dual-output logger with timestamps and elapsed time."""

    def __init__(self, log_file: Path | None = None):
        self._start_time = time.monotonic()
        self._file: TextIO | None = None
        if log_file:
            self._file = open(log_file, "w")

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def log(self, message: str, level: str = "INFO"):
        now = datetime.now()
        elapsed = time.monotonic() - self._start_time
        line = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [{elapsed:8.1f}s] [{level}] {message}"
        print(line, flush=True)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    def section(self, title: str):
        self.log("")
        self.log("=" * 60)
        self.log(f" {title}")
        self.log("=" * 60)
        self.log("")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SmokeTestConfig:
    """Configuration for the smoke test."""

    config_path: Path
    timeout_seconds: int = 1800  # 30 min total
    job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT
    log_file: Path | None = None
    tpu_type: str = "v5litepod-16"
    cleanup_on_failure: bool = True
    clean_start: bool = True  # Delete existing resources before starting
    build_images: bool = True  # Build and push images before starting
    # TODO: Extract region/project from config instead of hardcoding
    image_region: str = "europe-west4"
    image_project: str = "hai-gcp-models"


# =============================================================================
# SSH Tunnel (reuses pattern from cluster-tools.py)
# =============================================================================


def _wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> bool:
    """Wait for a port to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


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


@contextmanager
def _controller_tunnel(
    zone: str,
    project: str,
    logger: SmokeTestLogger,
    local_port: int = DEFAULT_CONTROLLER_PORT,
) -> Iterator[str]:
    """Establish SSH tunnel to controller and yield the local URL."""
    vm_name = discover_controller_vm(zone, project)
    if not vm_name:
        raise RuntimeError(f"No controller VM found in {zone}")

    logger.log(f"Establishing SSH tunnel to {vm_name}...")

    proc = subprocess.Popen(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--",
            "-L",
            f"{local_port}:localhost:{DEFAULT_CONTROLLER_PORT}",
            "-N",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        if not _wait_for_port(local_port, timeout=60):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait()
            raise RuntimeError(f"Tunnel failed to establish: {stderr}")

        logger.log(f"Tunnel ready: localhost:{local_port} -> {vm_name}:{DEFAULT_CONTROLLER_PORT}")
        yield f"http://localhost:{local_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


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
        self.logger = SmokeTestLogger(config.log_file)
        self._controller: ControllerProtocol | None = None
        self._tunnel_proc: subprocess.Popen | None = None
        self._interrupted = False
        self._results: list[TestResult] = []
        # Unique run ID to avoid job name collisions with previous runs
        self._run_id = datetime.now().strftime("%H%M%S")
        # Store zone/project for use in cleanup
        self._zone: str | None = None
        self._project: str | None = None

    def run(self) -> bool:
        """Run the smoke test. Returns True if all tests pass."""
        # Configure logging to show subprocess output from iris modules
        _configure_logging()

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        try:
            self._print_header()
            cluster_config = load_config(self.config.config_path)
            zone = cluster_config.zone
            project = cluster_config.project_id
            # Store for use in cleanup
            self._zone = zone
            self._project = project

            # Phase 0a: Build and push images
            if self.config.build_images:
                self.logger.section("PHASE 0a: Building Images")
                if not self._build_images():
                    self.logger.log("Image build failed!", level="ERROR")
                    return False
                if self._interrupted:
                    return False

            # Phase 0b: Clean start (delete existing resources)
            if self.config.clean_start:
                self.logger.section("PHASE 0b: Clean Start")
                self._cleanup_existing(zone, project)
                if self._interrupted:
                    return False

            # Phase 1: Start cluster
            self.logger.section("PHASE 1: Starting Cluster")
            self._start_cluster(cluster_config)  # Address logged, but we connect via SSH tunnel
            if self._interrupted:
                return False

            # Phase 2: SSH tunnel
            self.logger.section("PHASE 2: SSH Tunnel Setup")

            with _controller_tunnel(zone, project, self.logger) as tunnel_url:
                if self._interrupted:
                    return False

                # Phase 3: Run tests
                self.logger.section("PHASE 3: Running Tests")
                self._run_tests(tunnel_url)

                # Phase 4: Results
                self.logger.section("PHASE 4: Results Summary")
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

    def _print_header(self):
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log(" IRIS CLUSTER SMOKE TEST")
        self.logger.log("=" * 60)
        self.logger.log("")
        self.logger.log(f"Config: {self.config.config_path}")
        self.logger.log(f"Timeout: {self.config.timeout_seconds}s")
        self.logger.log(f"TPU type: {self.config.tpu_type}")
        if self.config.log_file:
            self.logger.log(f"Log file: {self.config.log_file}")

    def _cleanup_existing(self, zone: str, project: str):
        """Delete existing iris resources (controller VM, TPU slices) for clean start."""
        self.logger.log("Cleaning up existing iris resources...")

        deleted = cleanup_iris_resources(zone, project, dry_run=False)

        if not deleted:
            self.logger.log("  No existing resources found")
        else:
            for resource in deleted:
                self.logger.log(f"  Deleted: {resource}")
            self.logger.log("Cleanup complete")

    def _build_images(self) -> bool:
        """Build and push controller and worker images."""
        region = self.config.image_region
        project = self.config.image_project

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

    def _start_cluster(self, cluster_config) -> str:
        """Start the cluster using create_controller from controller.py."""
        self.logger.log("Creating controller from config...")
        self._controller = create_controller(cluster_config)

        self.logger.log("Starting controller VM (this may take a few minutes)...")
        start = time.monotonic()
        address = self._controller.start()
        elapsed = time.monotonic() - start

        self.logger.log(f"Controller started at {address} in {elapsed:.1f}s")
        return address

    def _run_tests(self, controller_url: str):
        """Run test jobs against the cluster."""
        client = IrisClient.remote(controller_url, workspace=IRIS_ROOT)

        # Test 1: Simple TPU job
        self.logger.log(f"[Test 1/3] Simple TPU job ({self.config.tpu_type})")
        result = self._run_simple_tpu_job(client)
        self._results.append(result)
        self._log_autoscaler_status(controller_url)

        if self._interrupted:
            return

        # Test 2: Concurrent TPU jobs
        self.logger.log(f"[Test 2/3] Concurrent TPU jobs (3x {self.config.tpu_type})")
        result = self._run_concurrent_tpu_jobs(client)
        self._results.append(result)
        self._log_autoscaler_status(controller_url)

        if self._interrupted:
            return

        # Test 3: Coscheduled multi-task job
        self.logger.log(f"[Test 3/3] Coscheduled multi-task job ({self.config.tpu_type})")
        result = self._run_coscheduled_job(client)
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

    def _collect_worker_logs(self, zone: str, project: str):
        """Collect docker logs from all TPU workers for post-mortem debugging."""
        self.logger.log("Collecting worker logs for debugging...")

        logs_dir = IRIS_ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)

        tpu_names = list_iris_tpus(zone, project)
        if not tpu_names:
            self.logger.log("  No TPU slices found")
            return

        for tpu_name in tpu_names:
            self.logger.log(f"  Collecting logs from {tpu_name}...")
            log_file = collect_docker_logs(
                vm_name=tpu_name,
                container_name="iris-worker",
                zone=zone,
                project=project,
                output_dir=logs_dir,
                is_tpu=True,
            )
            if log_file:
                self.logger.log(f"    Saved to {log_file}")
            else:
                self.logger.log("    Failed to collect logs", level="WARN")

    def _run_simple_tpu_job(self, client: IrisClient) -> TestResult:
        """Run a simple TPU job that just prints and returns."""

        def hello_tpu():
            print("Hello from TPU!")
            return 42

        start = time.monotonic()
        try:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(hello_tpu),
                name=f"smoke-simple-{self._run_id}",
                resources=ResourceSpec(device=tpu_device(self.config.tpu_type)),
                environment=EnvironmentSpec(workspace="/app"),
            )
            self.logger.log(f"  Job submitted: {job.job_id}")

            status = job.wait(timeout=self.config.job_timeout_seconds, raise_on_failure=False)
            duration = time.monotonic() - start

            if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                self.logger.log(f"  [PASS] Completed in {duration:.1f}s")
                return TestResult(
                    f"Simple TPU job ({self.config.tpu_type})", True, f"Completed in {duration:.1f}s", duration
                )
            else:
                state_name = cluster_pb2.JobState.Name(status.state)
                self.logger.log(f"  [FAIL] Job ended with state {state_name}", level="ERROR")
                self._print_task_logs_on_failure(job)
                return TestResult(
                    f"Simple TPU job ({self.config.tpu_type})",
                    False,
                    f"State: {state_name}, error: {status.error}",
                    duration,
                )

        except TimeoutError:
            duration = time.monotonic() - start
            self.logger.log(f"  [FAIL] Timed out after {self.config.job_timeout_seconds}s", level="ERROR")
            return TestResult(
                f"Simple TPU job ({self.config.tpu_type})",
                False,
                f"Timed out after {self.config.job_timeout_seconds}s",
                duration,
            )

    def _run_concurrent_tpu_jobs(self, client: IrisClient) -> TestResult:
        """Submit 3 concurrent TPU jobs to test parallel provisioning and queueing."""

        def quick_task(task_id: int):
            import time as time_module

            time_module.sleep(2.0)
            print(f"Task {task_id} completed")
            return task_id

        start = time.monotonic()
        try:
            jobs = []
            for i in range(3):
                job = client.submit(
                    entrypoint=Entrypoint.from_callable(quick_task, i),
                    name=f"smoke-concurrent-{self._run_id}-{i}",
                    resources=ResourceSpec(device=tpu_device(self.config.tpu_type)),
                    environment=EnvironmentSpec(workspace="/app"),
                )
                jobs.append(job)
                self.logger.log(f"  Job submitted: {job.job_id}")

            # Wait for all jobs
            failed_jobs = []
            for job in jobs:
                status = job.wait(timeout=self.config.job_timeout_seconds, raise_on_failure=False)
                if status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
                    state_name = cluster_pb2.JobState.Name(status.state)
                    failed_jobs.append(f"{job.job_id}: {state_name}")

            duration = time.monotonic() - start

            if not failed_jobs:
                self.logger.log(f"  [PASS] All 3 jobs completed in {duration:.1f}s")
                return TestResult("Concurrent TPU jobs (3x)", True, f"All completed in {duration:.1f}s", duration)
            else:
                self.logger.log(f"  [FAIL] Some jobs failed: {', '.join(failed_jobs)}", level="ERROR")
                return TestResult("Concurrent TPU jobs (3x)", False, f"Failed: {', '.join(failed_jobs)}", duration)

        except TimeoutError:
            duration = time.monotonic() - start
            self.logger.log("  [FAIL] Timed out waiting for jobs", level="ERROR")
            return TestResult(
                "Concurrent TPU jobs (3x)", False, f"Timed out after {self.config.job_timeout_seconds}s", duration
            )

    def _run_coscheduled_job(self, client: IrisClient) -> TestResult:
        """Run a coscheduled multi-task job on TPU workers."""

        def distributed_work():
            from iris.cluster.client import get_job_info

            info = get_job_info()
            if info is None:
                raise RuntimeError("Not running in an Iris job context")
            print(f"Task {info.task_index} of {info.num_tasks} on worker {info.worker_id}")
            return f"Task {info.task_index} done"

        start = time.monotonic()
        try:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(distributed_work),
                name=f"smoke-coscheduled-{self._run_id}",
                resources=ResourceSpec(
                    replicas=4,
                    device=tpu_device(self.config.tpu_type),
                ),
                environment=EnvironmentSpec(workspace="/app"),
                coscheduling=CoschedulingConfig(group_by="tpu-name"),
            )
            self.logger.log(f"  Job submitted: {job.job_id} (4 tasks)")

            status = job.wait(timeout=self.config.job_timeout_seconds, raise_on_failure=False)
            duration = time.monotonic() - start

            if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                self.logger.log(f"  [PASS] All 4 tasks completed in {duration:.1f}s")
                return TestResult(
                    "Coscheduled multi-task job", True, f"All 4 tasks completed in {duration:.1f}s", duration
                )
            else:
                state_name = cluster_pb2.JobState.Name(status.state)
                self.logger.log(f"  [FAIL] Job ended with state {state_name}", level="ERROR")
                return TestResult(
                    "Coscheduled multi-task job", False, f"State: {state_name}, error: {status.error}", duration
                )

        except TimeoutError:
            duration = time.monotonic() - start
            self.logger.log(f"  [FAIL] Timed out after {self.config.job_timeout_seconds}s", level="ERROR")
            return TestResult(
                "Coscheduled multi-task job", False, f"Timed out after {self.config.job_timeout_seconds}s", duration
            )

    def _log_autoscaler_status(self, controller_url: str):
        """Log current autoscaler state for observability."""
        try:
            rpc_client = ControllerServiceClientSync(controller_url)
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

            rpc_client.close()
        except Exception as e:
            self.logger.log(f"  (Could not fetch autoscaler status: {e})")

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

        return all_passed

    def _cleanup(self):
        """Clean up cluster resources."""
        self.logger.section("CLEANUP")

        # Collect worker logs before cleanup for post-mortem debugging
        any_failed = any(not r.passed for r in self._results)
        if any_failed and self._zone and self._project:
            try:
                self._collect_worker_logs(self._zone, self._project)
            except Exception as e:
                self.logger.log(f"Error collecting worker logs: {e}", level="WARN")

        should_cleanup = self.config.cleanup_on_failure or all(r.passed for r in self._results)

        if not should_cleanup:
            self.logger.log("Skipping cleanup (--no-cleanup-on-failure and tests failed)")
            self.logger.log("Controller VM left running for debugging")
            return

        if self._controller:
            self.logger.log("Stopping controller VM...")
            try:
                self._controller.stop()
                self.logger.log("Controller stopped")
            except Exception as e:
                self.logger.log(f"Error stopping controller: {e}", level="WARN")

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
    "--log-file",
    type=click.Path(path_type=Path),
    help="Log file path (also logs to stdout)",
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
    "--no-build-images",
    is_flag=True,
    help="Skip building and pushing Docker images",
)
def main(
    config_path: Path,
    timeout_seconds: int,
    job_timeout_seconds: int,
    log_file: Path | None,
    tpu_type: str,
    no_cleanup_on_failure: bool,
    no_clean_start: bool,
    no_build_images: bool,
):
    """Run Iris cluster autoscaling smoke test.

    This script starts a cluster, submits TPU jobs to exercise autoscaling,
    and validates that everything works correctly. On completion (or failure),
    the cluster is cleaned up unless --no-cleanup-on-failure is specified.

    Examples:

        # Basic smoke test
        uv run python scripts/smoke-test.py --config examples/eu-west4.yaml

        # With logging to file
        uv run python scripts/smoke-test.py --config examples/eu-west4.yaml \\
            --log-file smoke-test-$(date +%Y%m%d-%H%M%S).log

        # Custom timeout (45 min) for slow environments
        uv run python scripts/smoke-test.py --config examples/eu-west4.yaml --timeout 2700

        # Keep cluster running on failure for debugging
        uv run python scripts/smoke-test.py --config examples/eu-west4.yaml --no-cleanup-on-failure
    """
    config = SmokeTestConfig(
        config_path=config_path,
        timeout_seconds=timeout_seconds,
        job_timeout_seconds=job_timeout_seconds,
        log_file=log_file,
        tpu_type=tpu_type,
        cleanup_on_failure=not no_cleanup_on_failure,
        clean_start=not no_clean_start,
        build_images=not no_build_images,
    )

    runner = SmokeTestRunner(config)
    success = runner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
