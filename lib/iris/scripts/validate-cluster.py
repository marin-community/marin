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

"""Validate an Iris cluster by running test jobs.

This script connects to a running Iris controller (via SSH tunnel if needed),
submits validation jobs, and reports the results.

Usage:
    # Connect via SSH tunnel (auto-discovers controller VM)
    uv run python scripts/validate-cluster.py --zone europe-west4-b

    # Connect to a local or already-tunneled controller
    uv run python scripts/validate-cluster.py --controller-url http://localhost:10000
"""

import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import click

from iris.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import cluster_pb2

IRIS_ROOT = Path(__file__).parent.parent
DEFAULT_CONTROLLER_PORT = 10000
DEFAULT_TPU_TYPE = "v5litepod-16"
DEFAULT_TIMEOUT = 600  # 10 minutes for TPU provisioning


@dataclass
class ValidationResult:
    """Result of a single validation test."""

    name: str
    passed: bool
    message: str
    duration_seconds: float


def discover_controller_vm(zone: str, project: str) -> str | None:
    """Find the controller VM in the given zone using name-based filter.

    Uses name pattern 'iris-controller*' since the metadata-based filter
    doesn't work reliably with --zones in gcloud.
    """
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--zones={zone}",
            "--filter=name~^iris-controller",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Error listing instances: {result.stderr}", err=True)
        return None

    vm_names = result.stdout.strip().split("\n")
    vm_names = [v for v in vm_names if v]

    if not vm_names:
        return None
    if len(vm_names) > 1:
        click.echo(f"Warning: Multiple controller VMs found: {vm_names}", err=True)
    return vm_names[0]


def wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> bool:
    """Wait for a port to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


@contextmanager
def controller_tunnel(zone: str, project: str, local_port: int = DEFAULT_CONTROLLER_PORT) -> Iterator[str]:
    """Establish SSH tunnel to controller and yield the local URL."""
    vm_name = discover_controller_vm(zone, project)
    if not vm_name:
        raise click.ClickException(f"No controller VM found in {zone}")

    click.echo(f"Establishing SSH tunnel to {vm_name}...")

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
        if not wait_for_port(local_port, timeout=30):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait()
            raise click.ClickException(f"Tunnel failed to establish: {stderr}")

        click.echo(f"Tunnel established: localhost:{local_port} -> {vm_name}:{DEFAULT_CONTROLLER_PORT}")
        yield f"http://localhost:{local_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_simple_tpu_job(client: IrisClient, tpu_type: str) -> ValidationResult:
    """Test 1: Simple TPU job that returns a value.

    This test validates that the autoscaler can provision a TPU slice
    and the worker can execute a basic job.
    """
    start = time.monotonic()

    def hello():
        print("Hello from validation job!")
        return 42

    try:
        job = client.submit(
            entrypoint=Entrypoint.from_callable(hello),
            name="validate-hello",
            resources=ResourceSpec(device=tpu_device(tpu_type)),
            environment=EnvironmentSpec(workspace="/app"),
        )
        status = job.wait(timeout=DEFAULT_TIMEOUT, raise_on_failure=False)

        duration = time.monotonic() - start

        if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
            return ValidationResult(
                name=f"Simple TPU job ({tpu_type})",
                passed=True,
                message=f"Job completed successfully in {duration:.1f}s",
                duration_seconds=duration,
            )
        else:
            state_name = cluster_pb2.JobState.Name(status.state)
            return ValidationResult(
                name=f"Simple TPU job ({tpu_type})",
                passed=False,
                message=f"Job ended with state {state_name}: {status.error}",
                duration_seconds=duration,
            )
    except TimeoutError:
        return ValidationResult(
            name=f"Simple TPU job ({tpu_type})",
            passed=False,
            message=f"Job timed out after {DEFAULT_TIMEOUT}s (TPU provisioning may take 5-10 min)",
            duration_seconds=time.monotonic() - start,
        )
    except Exception as e:
        return ValidationResult(
            name=f"Simple TPU job ({tpu_type})",
            passed=False,
            message=f"Unexpected error: {e}",
            duration_seconds=time.monotonic() - start,
        )


def run_compute_job(client: IrisClient, tpu_type: str) -> ValidationResult:
    """Test 2: TPU job with arguments and return value."""
    start = time.monotonic()

    def compute(a: int, b: int) -> int:
        result = a + b
        print(f"{a} + {b} = {result}")
        return result

    try:
        job = client.submit(
            entrypoint=Entrypoint.from_callable(compute, 10, 32),
            name="validate-compute",
            resources=ResourceSpec(device=tpu_device(tpu_type)),
            environment=EnvironmentSpec(workspace="/app"),
        )
        status = job.wait(timeout=DEFAULT_TIMEOUT, raise_on_failure=False)

        duration = time.monotonic() - start

        if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
            return ValidationResult(
                name=f"Compute job with args ({tpu_type})",
                passed=True,
                message=f"Job completed successfully in {duration:.1f}s",
                duration_seconds=duration,
            )
        else:
            state_name = cluster_pb2.JobState.Name(status.state)
            return ValidationResult(
                name=f"Compute job with args ({tpu_type})",
                passed=False,
                message=f"Job ended with state {state_name}: {status.error}",
                duration_seconds=duration,
            )
    except TimeoutError:
        return ValidationResult(
            name=f"Compute job with args ({tpu_type})",
            passed=False,
            message=f"Job timed out after {DEFAULT_TIMEOUT}s",
            duration_seconds=time.monotonic() - start,
        )
    except Exception as e:
        return ValidationResult(
            name=f"Compute job with args ({tpu_type})",
            passed=False,
            message=f"Unexpected error: {e}",
            duration_seconds=time.monotonic() - start,
        )


def run_scheduler_test(client: IrisClient, tpu_type: str) -> ValidationResult:
    """Test 3: Submit multiple TPU jobs to exercise the scheduler.

    This test submits 2 concurrent jobs to verify the scheduler can
    handle multiple pending jobs and the autoscaler responds appropriately.
    """
    start = time.monotonic()

    def quick_task(task_id: int):
        import time as time_module

        time_module.sleep(1.0)
        print(f"Task {task_id} completed")
        return task_id

    try:
        jobs = []
        for i in range(2):
            job = client.submit(
                entrypoint=Entrypoint.from_callable(quick_task, i),
                name=f"validate-scheduler-{i}",
                resources=ResourceSpec(device=tpu_device(tpu_type)),
                environment=EnvironmentSpec(workspace="/app"),
            )
            jobs.append(job)

        all_succeeded = True
        failed_jobs = []
        for job in jobs:
            status = job.wait(timeout=DEFAULT_TIMEOUT, raise_on_failure=False)
            if status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
                all_succeeded = False
                state_name = cluster_pb2.JobState.Name(status.state)
                failed_jobs.append(f"{job.job_id}: {state_name}")

        duration = time.monotonic() - start

        if all_succeeded:
            return ValidationResult(
                name="Scheduler test (2 concurrent TPU jobs)",
                passed=True,
                message=f"All 2 jobs completed successfully in {duration:.1f}s",
                duration_seconds=duration,
            )
        else:
            return ValidationResult(
                name="Scheduler test (2 concurrent TPU jobs)",
                passed=False,
                message=f"Some jobs failed: {', '.join(failed_jobs)}",
                duration_seconds=duration,
            )
    except TimeoutError:
        return ValidationResult(
            name="Scheduler test (2 concurrent TPU jobs)",
            passed=False,
            message=f"Jobs timed out after {DEFAULT_TIMEOUT}s",
            duration_seconds=time.monotonic() - start,
        )
    except Exception as e:
        return ValidationResult(
            name="Scheduler test (2 concurrent TPU jobs)",
            passed=False,
            message=f"Unexpected error: {e}",
            duration_seconds=time.monotonic() - start,
        )


def print_results(results: list[ValidationResult]) -> bool:
    """Print validation results and return True if all passed."""
    click.echo()
    click.echo("=" * 60)
    click.echo("VALIDATION RESULTS")
    click.echo("=" * 60)

    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        click.echo(f"[{status}] {result.name}")
        click.echo(f"       {result.message}")
        if not result.passed:
            all_passed = False

    click.echo("=" * 60)

    total_duration = sum(r.duration_seconds for r in results)
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    if all_passed:
        click.echo(f"All {total_count} tests passed in {total_duration:.1f}s")
    else:
        click.echo(f"{passed_count}/{total_count} tests passed in {total_duration:.1f}s")

    return all_passed


@click.command()
@click.option("--zone", default="europe-west4-b", help="GCP zone to discover controller in")
@click.option("--project", default="hai-gcp-models", help="GCP project")
@click.option("--controller-url", help="Direct controller URL (skips SSH tunnel)")
@click.option("--workspace", type=click.Path(exists=True, path_type=Path), help="Workspace directory")
@click.option("--local-port", default=DEFAULT_CONTROLLER_PORT, help="Local port for SSH tunnel")
@click.option("--tpu-type", default=DEFAULT_TPU_TYPE, help="TPU type for validation jobs")
def validate(
    zone: str,
    project: str,
    controller_url: str | None,
    workspace: Path | None,
    local_port: int,
    tpu_type: str,
) -> None:
    """Run validation jobs against an Iris cluster.

    This script submits TPU test jobs to verify the cluster is functioning correctly.
    Jobs will trigger the autoscaler to provision TPU slices.

    Note: TPU provisioning can take 5-10 minutes. The default timeout is 10 minutes.

    Examples:

        # Auto-discover controller and establish SSH tunnel
        uv run python scripts/validate-cluster.py --zone europe-west4-b

        # Connect to a local or already-tunneled controller
        uv run python scripts/validate-cluster.py --controller-url http://localhost:10000

        # Use a different TPU type
        uv run python scripts/validate-cluster.py --tpu-type v5litepod-8
    """
    ws = workspace or IRIS_ROOT

    if controller_url:
        click.echo(f"Connecting to controller at {controller_url}")
        click.echo(f"Using workspace: {ws}")
        click.echo(f"TPU type: {tpu_type}")
        _run_validation(controller_url, ws, tpu_type)
    else:
        click.echo(f"Looking for controller VM in {zone}...")
        with controller_tunnel(zone, project, local_port) as url:
            click.echo(f"Using workspace: {ws}")
            click.echo(f"TPU type: {tpu_type}")
            _run_validation(url, ws, tpu_type)


def _run_validation(controller_url: str, workspace: Path, tpu_type: str) -> None:
    """Run validation tests against the controller."""
    click.echo()
    click.echo("Creating IrisClient...")
    client = IrisClient.remote(controller_url, workspace=workspace)

    click.echo("Running validation tests...")
    click.echo("Note: TPU provisioning can take 5-10 minutes per job")
    click.echo()

    results: list[ValidationResult] = []

    # Test 1: Simple TPU job
    click.echo("[1/3] Running simple TPU job (may trigger TPU provisioning)...")
    results.append(run_simple_tpu_job(client, tpu_type))

    # Test 2: TPU job with arguments
    click.echo("[2/3] Running compute job with arguments...")
    results.append(run_compute_job(client, tpu_type))

    # Test 3: Scheduler test with multiple concurrent TPU jobs
    click.echo("[3/3] Running scheduler test with 2 concurrent jobs...")
    results.append(run_scheduler_test(client, tpu_type))

    all_passed = print_results(results)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    validate()
