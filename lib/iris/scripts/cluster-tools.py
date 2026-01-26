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

"""Unified cluster tools for probing, debugging, cleaning, and validating Iris clusters.

This script provides commands to discover, probe, monitor, validate, and clean up an Iris
controller running on a GCP VM. It handles SSH tunneling transparently for RPC commands.

Usage:
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models discover
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models ssh-status
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models tunnel
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models health
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models autoscaler-status
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models list-workers
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models logs
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models bootstrap-logs
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models list-jobs
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models validate
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models cleanup
    uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models cleanup --no-dry-run
"""

import json
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from typing import TypeVar
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import click
from google.protobuf import json_format

from iris.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

IRIS_ROOT = Path(__file__).parent.parent
CONTROLLER_CONTAINER_NAME = "iris-controller"
DEFAULT_CONTROLLER_PORT = 10000
DEFAULT_TPU_TYPE = "v5litepod-16"
DEFAULT_VALIDATION_TIMEOUT = 600  # 10 minutes for TPU provisioning


# -----------------------------------------------------------------------------
# Shared utility functions
# -----------------------------------------------------------------------------


def run_gcloud(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a gcloud command and return the result."""
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, check=check)


def list_controller_vms(zone: str, project: str) -> list[str]:
    """Find iris controller VMs in the zone. Returns list of VM names."""
    result = run_gcloud(
        [
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--zones={zone}",
            "--filter=name~^iris-controller",
            "--format=value(name)",
        ]
    )
    if result.returncode != 0:
        click.echo(f"Error listing instances: {result.stderr}", err=True)
        return []

    names = result.stdout.strip().split("\n")
    return [n for n in names if n]


def discover_controller_vm(zone: str, project: str) -> str | None:
    """Find the controller VM in the given zone.

    Returns the first VM name if found, None otherwise.
    Warns if multiple controller VMs are found.
    """
    vm_names = list_controller_vms(zone, project)
    if not vm_names:
        return None
    if len(vm_names) > 1:
        click.echo(f"Warning: Multiple controller VMs found: {vm_names}", err=True)
    return vm_names[0]


def get_vm_status(vm_name: str, zone: str, project: str) -> dict | None:
    """Get detailed status of a VM."""
    result = run_gcloud(
        [
            "compute",
            "instances",
            "describe",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--format=json",
        ]
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> bool:
    """Wait for a port to become available.

    Returns True if port is ready, False if timeout.
    """
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
    """Establish SSH tunnel to controller and yield the local URL.

    Usage:
        with controller_tunnel("europe-west4-b", "hai-gcp-models") as url:
            client = ControllerServiceClientSync(url)
            client.list_jobs(cluster_pb2.Controller.ListJobsRequest())
    """
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


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a single validation test."""

    name: str
    passed: bool
    message: str
    duration_seconds: float


def _job_status_to_result(name: str, status: cluster_pb2.JobStatus, duration: float) -> ValidationResult:
    """Convert a job status to a ValidationResult."""
    if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
        return ValidationResult(name, True, f"Job completed in {duration:.1f}s", duration)
    state_name = cluster_pb2.JobState.Name(status.state)
    return ValidationResult(name, False, f"Job ended with state {state_name}: {status.error}", duration)


def _submit_simple_job(client: IrisClient, tpu_type: str) -> cluster_pb2.JobStatus:
    """Submit a simple TPU job that returns a value."""

    def hello():
        print("Hello from validation job!")
        return 42

    job = client.submit(
        entrypoint=Entrypoint.from_callable(hello),
        name="validate-hello",
        resources=ResourceSpec(device=tpu_device(tpu_type)),
        environment=EnvironmentSpec(workspace="/app"),
    )
    return job.wait(timeout=DEFAULT_VALIDATION_TIMEOUT, raise_on_failure=False)


def _submit_compute_job(client: IrisClient, tpu_type: str) -> cluster_pb2.JobStatus:
    """Submit a TPU job with arguments and return value."""

    def compute(a: int, b: int) -> int:
        result = a + b
        print(f"{a} + {b} = {result}")
        return result

    job = client.submit(
        entrypoint=Entrypoint.from_callable(compute, 10, 32),
        name="validate-compute",
        resources=ResourceSpec(device=tpu_device(tpu_type)),
        environment=EnvironmentSpec(workspace="/app"),
    )
    return job.wait(timeout=DEFAULT_VALIDATION_TIMEOUT, raise_on_failure=False)


def _submit_scheduler_jobs(client: IrisClient, tpu_type: str) -> list[tuple[str, cluster_pb2.JobStatus]]:
    """Submit multiple TPU jobs to exercise the scheduler. Returns list of (job_id, status)."""

    def quick_task(task_id: int):
        import time as time_module

        time_module.sleep(1.0)
        print(f"Task {task_id} completed")
        return task_id

    jobs = []
    for i in range(2):
        job = client.submit(
            entrypoint=Entrypoint.from_callable(quick_task, i),
            name=f"validate-scheduler-{i}",
            resources=ResourceSpec(device=tpu_device(tpu_type)),
            environment=EnvironmentSpec(workspace="/app"),
        )
        jobs.append(job)

    return [(job.job_id, job.wait(timeout=DEFAULT_VALIDATION_TIMEOUT, raise_on_failure=False)) for job in jobs]


def _scheduler_results_to_validation(
    results: list[tuple[str, cluster_pb2.JobStatus]], duration: float
) -> ValidationResult:
    """Convert scheduler job results to a ValidationResult."""
    name = "Scheduler test (2 concurrent TPU jobs)"
    failed_jobs = []
    for job_id, status in results:
        if status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
            state_name = cluster_pb2.JobState.Name(status.state)
            failed_jobs.append(f"{job_id}: {state_name}")

    if not failed_jobs:
        return ValidationResult(name, True, f"All {len(results)} jobs completed in {duration:.1f}s", duration)
    return ValidationResult(name, False, f"Some jobs failed: {', '.join(failed_jobs)}", duration)


def print_validation_results(results: list[ValidationResult]) -> bool:
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


T = TypeVar("T")


def _run_validation_test(
    name: str,
    client: IrisClient,
    tpu_type: str,
    submit_fn: Callable[[IrisClient, str], T],
    result_fn: Callable[[T, float], ValidationResult] | None = None,
) -> ValidationResult:
    """Run a single validation test, catching TimeoutError at this level."""
    start = time.monotonic()
    try:
        result = submit_fn(client, tpu_type)
        duration = time.monotonic() - start
        if result_fn:
            return result_fn(result, duration)
        # Default: result is a JobStatus
        assert isinstance(result, cluster_pb2.JobStatus)
        return _job_status_to_result(name, result, duration)
    except TimeoutError:
        duration = time.monotonic() - start
        return ValidationResult(name, False, f"Timed out after {DEFAULT_VALIDATION_TIMEOUT}s", duration)


def _run_validation(controller_url: str, workspace: Path, tpu_type: str) -> None:
    """Run validation tests against the controller."""
    click.echo()
    click.echo("Creating IrisClient...")
    client = IrisClient.remote(controller_url, workspace=workspace)

    click.echo("Running validation tests...")
    click.echo("Note: TPU provisioning can take 5-10 minutes per job")
    click.echo()

    results: list[ValidationResult] = []

    click.echo("[1/3] Running simple TPU job (may trigger TPU provisioning)...")
    results.append(_run_validation_test(f"Simple TPU job ({tpu_type})", client, tpu_type, _submit_simple_job))

    click.echo("[2/3] Running compute job with arguments...")
    results.append(_run_validation_test(f"Compute job with args ({tpu_type})", client, tpu_type, _submit_compute_job))

    click.echo("[3/3] Running scheduler test with 2 concurrent jobs...")
    results.append(
        _run_validation_test(
            "Scheduler test (2 concurrent TPU jobs)",
            client,
            tpu_type,
            _submit_scheduler_jobs,
            _scheduler_results_to_validation,
        )
    )

    all_passed = print_validation_results(results)

    if not all_passed:
        sys.exit(1)


# -----------------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------------


@click.group()
@click.option("--zone", required=True, help="GCP zone (e.g., europe-west4-b)")
@click.option("--project", required=True, help="GCP project ID")
@click.pass_context
def cli(ctx: click.Context, zone: str, project: str) -> None:
    """Probe, debug, and validate Iris clusters on GCP."""
    ctx.ensure_object(dict)
    ctx.obj["zone"] = zone
    ctx.obj["project"] = project


@cli.command()
@click.pass_context
def discover(ctx: click.Context) -> None:
    """Find controller VM and show its status."""
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    click.echo(f"Searching for controller VM in {zone}...")
    vm_name = discover_controller_vm(zone, project)

    if not vm_name:
        click.echo("No controller VM found.")
        raise SystemExit(1)

    click.echo(f"Found controller VM: {vm_name}")

    status = get_vm_status(vm_name, zone, project)
    if status:
        click.echo(f"  Status: {status.get('status', 'UNKNOWN')}")
        click.echo(f"  Machine type: {status.get('machineType', '').split('/')[-1]}")
        click.echo(f"  Zone: {status.get('zone', '').split('/')[-1]}")

        network_interfaces = status.get("networkInterfaces", [])
        if network_interfaces:
            internal_ip = network_interfaces[0].get("networkIP", "N/A")
            click.echo(f"  Internal IP: {internal_ip}")

            access_configs = network_interfaces[0].get("accessConfigs", [])
            if access_configs:
                external_ip = access_configs[0].get("natIP", "N/A")
                click.echo(f"  External IP: {external_ip}")

        creation_time = status.get("creationTimestamp", "N/A")
        click.echo(f"  Created: {creation_time}")


@cli.command("ssh-status")
@click.option("--tail", default=50, help="Number of log lines to show")
@click.pass_context
def ssh_status(ctx: click.Context, tail: int) -> None:
    """SSH into controller and check docker/container status."""
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    vm_name = discover_controller_vm(zone, project)
    if not vm_name:
        click.echo("No controller VM found.")
        raise SystemExit(1)

    click.echo(f"Connecting to {vm_name}...")
    click.echo()

    click.echo("=== Docker Containers ===")
    subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--",
            "sudo",
            "docker",
            "ps",
            "-a",
            "--format",
            "table {{.Names}}\t{{.Status}}\t{{.Ports}}",
        ]
    )

    click.echo()
    click.echo(f"=== Container Logs (last {tail} lines) ===")
    subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--",
            "sudo",
            "docker",
            "logs",
            CONTROLLER_CONTAINER_NAME,
            "--tail",
            str(tail),
        ]
    )


@cli.command()
@click.option("--local-port", default=DEFAULT_CONTROLLER_PORT, help="Local port for tunnel")
@click.pass_context
def tunnel(ctx: click.Context, local_port: int) -> None:
    """Open persistent SSH tunnel to controller port 10000.

    Blocks until Ctrl+C is pressed.
    """
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    vm_name = discover_controller_vm(zone, project)
    if not vm_name:
        click.echo("No controller VM found.")
        raise SystemExit(1)

    click.echo(f"Opening SSH tunnel to {vm_name}:{DEFAULT_CONTROLLER_PORT} on localhost:{local_port}")
    click.echo("Press Ctrl+C to close tunnel.")
    click.echo()

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
        ],
    )

    def signal_handler(signum: int, frame: object) -> None:
        click.echo("\nClosing tunnel...")
        proc.terminate()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not wait_for_port(local_port, timeout=30):
        click.echo("Failed to establish tunnel.", err=True)
        proc.terminate()
        proc.wait()
        raise SystemExit(1)

    click.echo(f"Tunnel ready: http://localhost:{local_port}")
    click.echo()

    proc.wait()


@cli.command()
@click.option("--local-port", default=DEFAULT_CONTROLLER_PORT, help="Local port for tunnel")
@click.pass_context
def health(ctx: click.Context, local_port: int) -> None:
    """Check controller health endpoint (auto-establishes tunnel)."""
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    with controller_tunnel(zone, project, local_port) as url:
        client = ControllerServiceClientSync(url)
        try:
            # Health check via RPC - if this succeeds, controller is healthy
            client.list_jobs(cluster_pb2.Controller.ListJobsRequest())
            click.echo("Health check: OK")
        except Exception as e:
            click.echo(f"Health check failed: {e}", err=True)
            raise SystemExit(1) from e


@cli.command("autoscaler-status")
@click.option("--local-port", default=DEFAULT_CONTROLLER_PORT, help="Local port for tunnel")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def autoscaler_status(ctx: click.Context, local_port: int, json_output: bool) -> None:
    """Get autoscaler status via RPC (auto-establishes tunnel)."""
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    with controller_tunnel(zone, project, local_port) as url:
        client = ControllerServiceClientSync(url)
        request = cluster_pb2.Controller.GetAutoscalerStatusRequest()

        try:
            response = client.get_autoscaler_status(request)
        except Exception as e:
            click.echo(f"RPC failed: {e}", err=True)
            raise SystemExit(1) from e

        if json_output:
            output = json_format.MessageToJson(response, preserving_proto_field_name=True)
            click.echo(output)
            return

        status = response.status
        click.echo("=== Autoscaler Status ===")
        click.echo(f"Last evaluation: {status.last_evaluation_ms}ms")

        if status.current_demand:
            click.echo()
            click.echo("Current Demand:")
            for group_name, demand in status.current_demand.items():
                click.echo(f"  {group_name}: {demand}")

        if status.groups:
            click.echo()
            click.echo("Scale Groups:")
            for group in status.groups:
                cfg = group.config
                click.echo(f"  {group.name}:")
                click.echo(f"    Accelerator: {cfg.accelerator_type}")
                click.echo(f"    Min/Max slices: {cfg.min_slices}/{cfg.max_slices}")
                click.echo(f"    Current demand: {group.current_demand}")
                click.echo(f"    Peak demand: {group.peak_demand}")
                if group.consecutive_failures > 0:
                    click.echo(f"    Consecutive failures: {group.consecutive_failures}")
                if group.backoff_until_ms > 0:
                    click.echo(f"    Backoff until: {group.backoff_until_ms}ms")

        if status.recent_actions:
            click.echo()
            click.echo("Recent Actions:")
            for action in status.recent_actions[-10:]:
                click.echo(f"  [{action.timestamp_ms}] {action.action_type} ({action.scale_group}): {action.reason}")


@cli.command("list-workers")
@click.option("--local-port", default=DEFAULT_CONTROLLER_PORT, help="Local port for tunnel")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def list_workers(ctx: click.Context, local_port: int, json_output: bool) -> None:
    """List registered workers (auto-establishes tunnel)."""
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    with controller_tunnel(zone, project, local_port) as url:
        client = ControllerServiceClientSync(url)
        request = cluster_pb2.Controller.ListWorkersRequest()

        try:
            response = client.list_workers(request)
        except Exception as e:
            click.echo(f"RPC failed: {e}", err=True)
            raise SystemExit(1) from e

        if json_output:
            output = json_format.MessageToJson(response, preserving_proto_field_name=True)
            click.echo(output)
            return

        workers = response.workers
        if not workers:
            click.echo("No workers registered.")
            return

        click.echo(f"=== Workers ({len(workers)}) ===")
        for worker in workers:
            status = "healthy" if worker.healthy else "unhealthy"
            click.echo(f"\n{worker.worker_id}:")
            click.echo(f"  Address: {worker.address}")
            click.echo(f"  Status: {status}")
            click.echo(f"  Running tasks: {len(worker.running_job_ids)}")
            if worker.running_job_ids:
                click.echo(f"    Tasks: {', '.join(worker.running_job_ids)}")
            click.echo(f"  Last heartbeat: {worker.last_heartbeat_ms}")
            if worker.consecutive_failures > 0:
                click.echo(f"  Consecutive failures: {worker.consecutive_failures}")
            if worker.metadata.hostname:
                click.echo(f"  Hostname: {worker.metadata.hostname}")
            if worker.metadata.ip_address:
                click.echo(f"  IP: {worker.metadata.ip_address}")
            if worker.metadata.tpu_name:
                click.echo(f"  TPU: {worker.metadata.tpu_name}")


@cli.command()
@click.option("--tail", default=100, help="Number of log lines to show (ignored with --follow)")
@click.option("--follow", "-f", is_flag=True, help="Stream logs in real-time")
@click.pass_context
def logs(ctx: click.Context, tail: int, follow: bool) -> None:
    """Fetch docker logs from the iris-controller container.

    Examples:
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models logs
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models logs --tail 50
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models logs --follow
    """
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    vm_name = discover_controller_vm(zone, project)
    if not vm_name:
        click.echo("No controller VM found.")
        raise SystemExit(1)

    click.echo(f"Fetching logs from {vm_name}:{CONTROLLER_CONTAINER_NAME}...")

    docker_args = ["sudo", "docker", "logs", CONTROLLER_CONTAINER_NAME]
    if follow:
        docker_args.append("--follow")
    else:
        docker_args.extend(["--tail", str(tail)])

    subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--",
            *docker_args,
        ]
    )


@cli.command("bootstrap-logs")
@click.option("--tail", default=200, help="Number of log lines to show")
@click.pass_context
def bootstrap_logs(ctx: click.Context, tail: int) -> None:
    """Fetch startup-script logs from the VM.

    Shows the output from the GCP startup script that bootstraps the controller.
    Useful for debugging VM initialization issues.

    Examples:
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models bootstrap-logs
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models bootstrap-logs --tail 500
    """
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    vm_name = discover_controller_vm(zone, project)
    if not vm_name:
        click.echo("No controller VM found.")
        raise SystemExit(1)

    click.echo(f"Fetching startup-script logs from {vm_name}...")

    subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--",
            "sudo",
            "journalctl",
            "-u",
            "google-startup-scripts.service",
            "-n",
            str(tail),
            "--no-pager",
        ]
    )


@cli.command("list-jobs")
@click.option("--local-port", default=DEFAULT_CONTROLLER_PORT, help="Local port for tunnel")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def list_jobs(ctx: click.Context, local_port: int, json_output: bool) -> None:
    """List all jobs (auto-establishes tunnel)."""
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    with controller_tunnel(zone, project, local_port) as url:
        client = ControllerServiceClientSync(url)
        request = cluster_pb2.Controller.ListJobsRequest()

        try:
            response = client.list_jobs(request)
        except Exception as e:
            click.echo(f"RPC failed: {e}", err=True)
            raise SystemExit(1) from e

        if json_output:
            output = json_format.MessageToJson(response, preserving_proto_field_name=True)
            click.echo(output)
            return

        jobs = response.jobs
        if not jobs:
            click.echo("No jobs found.")
            return

        click.echo(f"=== Jobs ({len(jobs)}) ===")
        for job in jobs:
            state_name = cluster_pb2.JobState.Name(job.state)
            click.echo(f"\n{job.job_id}:")
            click.echo(f"  State: {state_name}")
            click.echo(f"  Tasks: {job.completed_count}/{job.task_count} completed")
            if job.task_state_counts:
                counts = ", ".join(f"{k}={v}" for k, v in job.task_state_counts.items())
                click.echo(f"  Task states: {counts}")
            if job.error:
                click.echo(f"  Error: {job.error}")


@cli.command()
@click.option("--controller-url", help="Direct controller URL (skips SSH tunnel)")
@click.option("--workspace", type=click.Path(exists=True, path_type=Path), help="Workspace directory")
@click.option("--local-port", default=DEFAULT_CONTROLLER_PORT, help="Local port for SSH tunnel")
@click.option("--tpu-type", default=DEFAULT_TPU_TYPE, help="TPU type for validation jobs")
@click.pass_context
def validate(
    ctx: click.Context,
    controller_url: str | None,
    workspace: Path | None,
    local_port: int,
    tpu_type: str,
) -> None:
    """Run validation jobs against an Iris cluster.

    This command submits TPU test jobs to verify the cluster is functioning correctly.
    Jobs will trigger the autoscaler to provision TPU slices.

    Note: TPU provisioning can take 5-10 minutes. The default timeout is 10 minutes.

    Examples:

        # Auto-discover controller and establish SSH tunnel
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models validate

        # Connect to a local or already-tunneled controller
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models validate --controller-url http://localhost:10000

        # Use a different TPU type
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models \\
            validate --tpu-type v5litepod-8
    """
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]
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


# -----------------------------------------------------------------------------
# Cleanup helpers
# -----------------------------------------------------------------------------


def list_tpu_slices(zone: str, project: str) -> list[str]:
    """Find iris-managed TPU slices in the zone."""
    result = run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--project={project}",
            f"--zone={zone}",
            "--filter=name~^iris-",
            "--format=json",
        ]
    )
    if result.returncode != 0:
        click.echo(f"Warning: Failed to list TPUs: {result.stderr.strip()}", err=True)
        return []

    if not result.stdout.strip():
        return []

    try:
        tpus = json.loads(result.stdout)
    except json.JSONDecodeError:
        click.echo(f"Warning: Failed to parse TPU list: {result.stdout[:200]}", err=True)
        return []

    names = []
    for tpu in tpus:
        name = tpu.get("name", "")
        # GCP returns full resource path like 'projects/proj/locations/zone/nodes/my-tpu'
        if "/" in name:
            name = name.split("/")[-1]
        if name:
            names.append(name)
    return names


def delete_vm(name: str, zone: str, project: str) -> bool:
    """Delete a GCE VM. Returns True on success."""
    result = run_gcloud(
        [
            "compute",
            "instances",
            "delete",
            name,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ]
    )
    if result.returncode != 0:
        error = result.stderr.strip()
        if "not found" in error.lower():
            click.echo(f"  VM {name} already deleted")
            return True
        click.echo(f"  Failed to delete VM {name}: {error}", err=True)
        return False
    return True


def delete_tpu(name: str, zone: str, project: str) -> bool:
    """Delete a TPU slice. Returns True on success."""
    result = run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            name,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ]
    )
    if result.returncode != 0:
        error = result.stderr.strip()
        if "not found" in error.lower():
            click.echo(f"  TPU {name} already deleted")
            return True
        click.echo(f"  Failed to delete TPU {name}: {error}", err=True)
        return False
    return True


@cli.command()
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    help="Dry-run mode (default: True). Use --no-dry-run to actually delete.",
)
@click.pass_context
def cleanup(ctx: click.Context, dry_run: bool) -> None:
    """Clean all iris VMs and TPUs from the zone.

    Finds and deletes all iris-managed resources:
    - Controller VMs (name matches 'iris-controller*')
    - TPU slices (name matches 'iris-*')

    By default runs in dry-run mode to show what would be deleted.
    Use --no-dry-run to actually perform deletions.

    Examples:
        # Show what would be deleted (safe, no changes)
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models cleanup

        # Actually delete resources
        uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models cleanup --no-dry-run
    """
    zone = ctx.obj["zone"]
    project = ctx.obj["project"]

    click.echo(f"Scanning zone {zone} in project {project}...")
    if dry_run:
        click.echo("(DRY-RUN mode - no changes will be made)")
    click.echo()

    # Find controller VMs
    controller_vms = list_controller_vms(zone, project)
    if controller_vms:
        click.echo(f"Found {len(controller_vms)} controller VM(s):")
        for name in controller_vms:
            click.echo(f"  - {name}")
    else:
        click.echo("No controller VMs found.")
    click.echo()

    # Find TPU slices
    tpu_slices = list_tpu_slices(zone, project)
    if tpu_slices:
        click.echo(f"Found {len(tpu_slices)} TPU slice(s):")
        for name in tpu_slices:
            click.echo(f"  - {name}")
    else:
        click.echo("No TPU slices found.")
    click.echo()

    total_resources = len(controller_vms) + len(tpu_slices)
    if total_resources == 0:
        click.echo("Nothing to clean up.")
        return

    if dry_run:
        click.echo(f"Would delete {total_resources} resource(s). Use --no-dry-run to delete.")
        return

    # Perform deletions
    click.echo("Deleting resources...")
    failed = 0

    for name in controller_vms:
        click.echo(f"Deleting VM: {name}")
        if not delete_vm(name, zone, project):
            failed += 1

    for name in tpu_slices:
        click.echo(f"Deleting TPU: {name}")
        if not delete_tpu(name, zone, project):
            failed += 1

    click.echo()
    deleted = total_resources - failed
    click.echo(f"Deleted {deleted}/{total_resources} resource(s).")

    if failed > 0:
        click.echo(f"Failed to delete {failed} resource(s).", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
