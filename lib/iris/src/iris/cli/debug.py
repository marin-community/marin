# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster debug and validation commands.

These commands were previously in ``scripts/cluster-tools.py`` and are now
integrated into the main CLI as ``iris cluster debug <command>``.
"""

import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import click
from google.protobuf import json_format

from iris.cli.main import require_controller_url
from iris.time_utils import Timestamp
from iris.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.rpc.proto_utils import format_accelerator_display

# Constants
CONTROLLER_CONTAINER_NAME = "iris-controller"
DEFAULT_CONTROLLER_PORT = 10000
DEFAULT_TPU_TYPE = "v5litepod-16"
DEFAULT_VALIDATION_TIMEOUT = 600


# ─── Shared utilities ────────────────────────────────────────────────────────


def _run_gcloud(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a gcloud command and return the result."""
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, check=check)


def _list_controller_vms(zone: str, project: str) -> list[str]:
    """Find iris controller VMs in the zone."""
    result = _run_gcloud(
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


def _discover_controller(zone: str, project: str) -> str | None:
    """Find the controller VM in the given zone."""
    vm_names = _list_controller_vms(zone, project)
    if not vm_names:
        return None
    if len(vm_names) > 1:
        click.echo(f"Warning: Multiple controller VMs found: {vm_names}", err=True)
    return vm_names[0]


def _get_vm_status(vm_name: str, zone: str, project: str) -> dict | None:
    """Get detailed status of a VM."""
    result = _run_gcloud(
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


def _get_zone_project(ctx: click.Context) -> tuple[str, str]:
    """Extract zone and project from the cluster config in context."""
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required on the cluster group", err=True)
        raise SystemExit(1)
    if config.platform.WhichOneof("platform") != "gcp":
        click.echo("Error: Debug commands require a GCP platform config", err=True)
        raise SystemExit(1)
    platform = config.platform.gcp
    zone = platform.zone or (platform.default_zones[0] if platform.default_zones else "")
    project = platform.project_id
    if not zone or not project:
        click.echo("Error: Config must specify platform.gcp.project_id and zone", err=True)
        raise SystemExit(1)
    return zone, project


# ─── Validation helpers ──────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Result of a single validation test."""

    name: str
    passed: bool
    message: str
    duration_seconds: float


def _job_status_to_result(name: str, status: cluster_pb2.JobStatus, duration: float) -> ValidationResult:
    if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
        return ValidationResult(name, True, f"Job completed in {duration:.1f}s", duration)
    state_name = cluster_pb2.JobState.Name(status.state)
    return ValidationResult(name, False, f"Job ended with state {state_name}: {status.error}", duration)


T = TypeVar("T")


def _run_validation_test(
    name: str,
    client: IrisClient,
    tpu_type: str,
    submit_fn: Callable[[IrisClient, str], T],
    result_fn: Callable[[T, float], ValidationResult] | None = None,
) -> ValidationResult:
    start = time.monotonic()
    try:
        result = submit_fn(client, tpu_type)
        duration = time.monotonic() - start
        if result_fn:
            return result_fn(result, duration)
        assert isinstance(result, cluster_pb2.JobStatus)
        return _job_status_to_result(name, result, duration)
    except TimeoutError:
        duration = time.monotonic() - start
        return ValidationResult(name, False, f"Timed out after {DEFAULT_VALIDATION_TIMEOUT}s", duration)


def _submit_simple_job(client: IrisClient, tpu_type: str) -> cluster_pb2.JobStatus:
    def hello():
        print("Hello from validation job!")
        return 42

    job = client.submit(
        entrypoint=Entrypoint.from_callable(hello),
        name="validate-hello",
        resources=ResourceSpec(device=tpu_device(tpu_type)),
        environment=EnvironmentSpec(),
    )
    return job.wait(timeout=DEFAULT_VALIDATION_TIMEOUT, raise_on_failure=False)


def _submit_compute_job(client: IrisClient, tpu_type: str) -> cluster_pb2.JobStatus:
    def compute(a: int, b: int) -> int:
        result = a + b
        print(f"{a} + {b} = {result}")
        return result

    job = client.submit(
        entrypoint=Entrypoint.from_callable(compute, 10, 32),
        name="validate-compute",
        resources=ResourceSpec(device=tpu_device(tpu_type)),
        environment=EnvironmentSpec(),
    )
    return job.wait(timeout=DEFAULT_VALIDATION_TIMEOUT, raise_on_failure=False)


def _submit_scheduler_jobs(client: IrisClient, tpu_type: str) -> list[tuple[str, cluster_pb2.JobStatus]]:
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
            environment=EnvironmentSpec(),
        )
        jobs.append(job)

    return [(job.job_id, job.wait(timeout=DEFAULT_VALIDATION_TIMEOUT, raise_on_failure=False)) for job in jobs]


def _scheduler_results_to_validation(
    results: list[tuple[str, cluster_pb2.JobStatus]], duration: float
) -> ValidationResult:
    name = "Scheduler test (2 concurrent TPU jobs)"
    failed_jobs = []
    for job_id, status in results:
        if status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
            state_name = cluster_pb2.JobState.Name(status.state)
            failed_jobs.append(f"{job_id}: {state_name}")

    if not failed_jobs:
        return ValidationResult(name, True, f"All {len(results)} jobs completed in {duration:.1f}s", duration)
    return ValidationResult(name, False, f"Some jobs failed: {', '.join(failed_jobs)}", duration)


def _print_validation_results(results: list[ValidationResult]) -> bool:
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


def _run_validation(controller_url: str, workspace: Path, tpu_type: str) -> None:
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

    all_passed = _print_validation_results(results)

    if not all_passed:
        sys.exit(1)


# ─── Cleanup helpers ─────────────────────────────────────────────────────────


def _list_tpu_slices(zone: str, project: str) -> list[str]:
    result = _run_gcloud(
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
        if "/" in name:
            name = name.split("/")[-1]
        if name:
            names.append(name)
    return names


def _delete_vm(name: str, zone: str, project: str) -> bool:
    result = _run_gcloud(["compute", "instances", "delete", name, f"--project={project}", f"--zone={zone}", "--quiet"])
    if result.returncode != 0:
        error = result.stderr.strip()
        if "not found" in error.lower():
            click.echo(f"  VM {name} already deleted")
            return True
        click.echo(f"  Failed to delete VM {name}: {error}", err=True)
        return False
    return True


def _delete_tpu(name: str, zone: str, project: str) -> bool:
    result = _run_gcloud(
        ["compute", "tpus", "tpu-vm", "delete", name, f"--project={project}", f"--zone={zone}", "--quiet"]
    )
    if result.returncode != 0:
        error = result.stderr.strip()
        if "not found" in error.lower():
            click.echo(f"  TPU {name} already deleted")
            return True
        click.echo(f"  Failed to delete TPU {name}: {error}", err=True)
        return False
    return True


# ─── CLI commands ─────────────────────────────────────────────────────────────


@click.group()
@click.pass_context
def debug(ctx):
    """Cluster debugging and validation commands.

    These commands discover the controller VM via GCP, establish SSH tunnels
    transparently, and provide operational tooling.
    """
    pass


@debug.command()
@click.pass_context
def discover(ctx):
    """Find controller VM and show its status."""
    zone, project = _get_zone_project(ctx)

    click.echo(f"Searching for controller VM in {zone}...")
    vm_name = _discover_controller(zone, project)

    if not vm_name:
        click.echo("No controller VM found.")
        raise SystemExit(1)

    click.echo(f"Found controller VM: {vm_name}")

    status = _get_vm_status(vm_name, zone, project)
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


@debug.command("ssh-status")
@click.option("--tail", default=50, help="Number of log lines to show")
@click.pass_context
def ssh_status(ctx, tail: int):
    """SSH into controller and check docker/container status."""
    zone, project = _get_zone_project(ctx)

    vm_name = _discover_controller(zone, project)
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


@debug.command()
@click.pass_context
def health(ctx):
    """Check controller health endpoint."""
    controller_url = require_controller_url(ctx)
    client = ControllerServiceClientSync(controller_url)
    try:
        client.list_jobs(cluster_pb2.Controller.ListJobsRequest())
        click.echo("Health check: OK")
    except Exception as e:
        click.echo(f"Health check failed: {e}", err=True)
        raise SystemExit(1) from e


@debug.command("autoscaler-status")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def autoscaler_status(ctx, json_output: bool):
    """Get autoscaler status via RPC."""
    controller_url = require_controller_url(ctx)
    client = ControllerServiceClientSync(controller_url)
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
    last_eval_ms = Timestamp.from_proto(status.last_evaluation).epoch_ms()
    click.echo(f"Last evaluation: {Timestamp.from_ms(last_eval_ms).as_formatted_date()}")

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
            click.echo(f"    Accelerator: {format_accelerator_display(cfg.accelerator_type, cfg.accelerator_variant)}")
            click.echo(f"    Min/Max slices: {cfg.min_slices}/{cfg.max_slices}")
            click.echo(f"    Current demand: {group.current_demand}")
            click.echo(f"    Peak demand: {group.peak_demand}")
            if group.consecutive_failures > 0:
                click.echo(f"    Consecutive failures: {group.consecutive_failures}")
            backoff_ms = Timestamp.from_proto(group.backoff_until).epoch_ms()
            if backoff_ms > 0:
                click.echo(f"    Backoff until: {Timestamp.from_ms(backoff_ms).as_formatted_date()}")

    if status.recent_actions:
        click.echo()
        click.echo("Recent Actions:")
        for action in status.recent_actions[-10:]:
            action_ts = Timestamp.from_proto(action.timestamp).as_formatted_date()
            click.echo(f"  [{action_ts}] {action.action_type} ({action.scale_group}): {action.reason}")


@debug.command("list-workers")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def list_workers(ctx, json_output: bool):
    """List registered workers."""
    controller_url = require_controller_url(ctx)
    client = ControllerServiceClientSync(controller_url)
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
        heartbeat_ms = Timestamp.from_proto(worker.last_heartbeat).epoch_ms()
        click.echo(f"  Last heartbeat: {Timestamp.from_ms(heartbeat_ms).as_formatted_date()}")
        if worker.consecutive_failures > 0:
            click.echo(f"  Consecutive failures: {worker.consecutive_failures}")
        if worker.metadata.hostname:
            click.echo(f"  Hostname: {worker.metadata.hostname}")
        if worker.metadata.ip_address:
            click.echo(f"  IP: {worker.metadata.ip_address}")
        if worker.metadata.tpu_name:
            click.echo(f"  TPU: {worker.metadata.tpu_name}")


@debug.command()
@click.option("--tail", default=100, help="Number of log lines to show (ignored with --follow)")
@click.option("--follow", "-f", is_flag=True, help="Stream logs in real-time")
@click.pass_context
def logs(ctx, tail: int, follow: bool):
    """Fetch docker logs from the iris-controller container."""
    zone, project = _get_zone_project(ctx)

    vm_name = _discover_controller(zone, project)
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


@debug.command("bootstrap-logs")
@click.option("--tail", default=200, help="Number of log lines to show")
@click.pass_context
def bootstrap_logs(ctx, tail: int):
    """Fetch startup-script logs from the controller VM."""
    zone, project = _get_zone_project(ctx)

    vm_name = _discover_controller(zone, project)
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


@debug.command("list-jobs")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def list_jobs(ctx, json_output: bool):
    """List all jobs."""
    controller_url = require_controller_url(ctx)
    client = ControllerServiceClientSync(controller_url)
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


@debug.command()
@click.option("--workspace", type=click.Path(exists=True, path_type=Path), help="Workspace directory")
@click.option("--tpu-type", default=DEFAULT_TPU_TYPE, help="TPU type for validation jobs")
@click.pass_context
def validate(ctx, workspace: Path | None, tpu_type: str):
    """Run validation jobs against an Iris cluster.

    Submits TPU test jobs to verify the cluster is functioning correctly.
    """
    controller_url = require_controller_url(ctx)
    iris_root = Path(__file__).resolve().parents[2]
    ws = workspace or iris_root

    click.echo(f"Connecting to controller at {controller_url}")
    click.echo(f"Using workspace: {ws}")
    click.echo(f"TPU type: {tpu_type}")
    _run_validation(controller_url, ws, tpu_type)


@debug.command()
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    help="Dry-run mode (default: True). Use --no-dry-run to actually delete.",
)
@click.pass_context
def cleanup(ctx, dry_run: bool):
    """Clean all iris VMs and TPUs from the zone.

    By default runs in dry-run mode to show what would be deleted.
    Use --no-dry-run to actually perform deletions.
    """
    zone, project = _get_zone_project(ctx)

    click.echo(f"Scanning zone {zone} in project {project}...")
    if dry_run:
        click.echo("(DRY-RUN mode - no changes will be made)")
    click.echo()

    controller_vms = _list_controller_vms(zone, project)
    if controller_vms:
        click.echo(f"Found {len(controller_vms)} controller VM(s):")
        for name in controller_vms:
            click.echo(f"  - {name}")
    else:
        click.echo("No controller VMs found.")
    click.echo()

    tpu_slices = _list_tpu_slices(zone, project)
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

    click.echo("Deleting resources...")
    failed = 0

    for name in controller_vms:
        click.echo(f"Deleting VM: {name}")
        if not _delete_vm(name, zone, project):
            failed += 1

    for name in tpu_slices:
        click.echo(f"Deleting TPU: {name}")
        if not _delete_tpu(name, zone, project):
            failed += 1

    click.echo()
    deleted = total_resources - failed
    click.echo(f"Deleted {deleted}/{total_resources} resource(s).")

    if failed > 0:
        click.echo(f"Failed to delete {failed} resource(s).", err=True)
        sys.exit(1)
