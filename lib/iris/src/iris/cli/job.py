# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job management via command passthrough (replaces ``iris-run``).

Usage:
    iris --config cluster.yaml job run -- python train.py --epochs 10
    iris --config cluster.yaml job run --tpu v5litepod-16 -e WANDB_API_KEY $WANDB_API_KEY -- python train.py
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import click
import humanfriendly
import yaml
from google.protobuf import json_format
from tabulate import tabulate

from iris.cli.bug_report import file_github_issue, format_bug_report, gather_bug_report
from iris.cli.main import require_controller_url
from iris.client import IrisClient
from iris.client.client import Job, JobFailedError
from iris.cluster.types import (
    Constraint,
    Entrypoint,
    EnvironmentSpec,
    JobName,
    ResourceSpec,
    gpu_device,
    region_constraint,
    zone_constraint,
    tpu_device,
)
from iris.rpc import cluster_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)

_STATE_MAP: dict[str, cluster_pb2.JobState] = {
    "pending": cluster_pb2.JOB_STATE_PENDING,
    "building": cluster_pb2.JOB_STATE_BUILDING,
    "running": cluster_pb2.JOB_STATE_RUNNING,
    "succeeded": cluster_pb2.JOB_STATE_SUCCEEDED,
    "failed": cluster_pb2.JOB_STATE_FAILED,
    "killed": cluster_pb2.JOB_STATE_KILLED,
    "worker_failed": cluster_pb2.JOB_STATE_WORKER_FAILED,
    "unschedulable": cluster_pb2.JOB_STATE_UNSCHEDULABLE,
}


def _job_state_name(state: cluster_pb2.JobState) -> str:
    return cluster_pb2.JobState.Name(state).replace("JOB_STATE_", "").lower()


def _format_resources(resources: cluster_pb2.ResourceSpecProto | None) -> str:
    """Format job resources as a compact human-readable string."""
    if not resources:
        return "-"

    parts = []

    # CPU
    if resources.cpu_millicores:
        parts.append(f"{resources.cpu_millicores / 1000:g}cpu")

    # Memory
    if resources.memory_bytes:
        parts.append(humanfriendly.format_size(resources.memory_bytes, binary=True))

    # Disk
    if resources.disk_bytes:
        parts.append(f"{humanfriendly.format_size(resources.disk_bytes, binary=True)} disk")

    # Device (TPU/GPU)
    if resources.HasField("device"):
        device = resources.device
        if device.HasField("tpu"):
            parts.append(device.tpu.variant)
        elif device.HasField("gpu"):
            gpu = device.gpu
            gpu_str = f"{gpu.count}x{gpu.variant}" if gpu.variant else f"{gpu.count}gpu"
            parts.append(gpu_str)

    return ", ".join(parts) if parts else "-"


def _terminate_jobs(
    client: IrisClient,
    job_ids: tuple[str, ...],
    include_children: bool,
) -> list[JobName]:
    terminated: list[JobName] = []
    for raw in job_ids:
        name = JobName.from_wire(raw)
        if include_children:
            terminated.extend(client.terminate_prefix(name, exclude_finished=True))
        else:
            client.terminate(name)
            terminated.append(name)
    return terminated


def _print_terminated(terminated: list[JobName]) -> None:
    if terminated:
        click.echo("Terminated jobs:")
        for job_name in terminated:
            click.echo(f"  {job_name}")
    else:
        click.echo("No running jobs matched.")


def load_env_vars(env_flags: tuple[tuple[str, ...], ...] | list | None) -> dict[str, str]:
    """Load environment variables from .marin.yaml and merge with flags.

    Args:
        env_flags: Tuple/list of (KEY,) or (KEY, VALUE) tuples from Click

    Returns:
        Merged environment variables
    """
    env_vars: dict[str, str] = {}
    marin_yaml = Path(".marin.yaml")
    if marin_yaml.exists():
        with open(marin_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg.get("env"), dict):
            for k, v in cfg["env"].items():
                env_vars[str(k)] = "" if v is None else str(v)

    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        if key not in env_vars and os.environ.get(key):
            env_vars[key] = os.environ[key]

    if env_flags:
        for item in env_flags:
            if len(item) > 2:
                raise ValueError(f"Too many values for env var: {' '.join(item)}")
            if "=" in item[0]:
                raise ValueError(
                    f"Key cannot contain '=': {item[0]}\nYou probably meant to do '-e {' '.join(item[0].split('='))}'"
                )
            env_vars[item[0]] = item[1] if len(item) == 2 else ""

    return env_vars


def add_standard_env_vars(env_vars: dict[str, str]) -> dict[str, str]:
    """Add standard environment variables used by Marin jobs."""
    result = dict(env_vars)

    defaults = {
        "PYTHONPATH": ".",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "~/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
    }

    for key, value in defaults.items():
        if key not in result:
            result[key] = value

    for key in ("GCS_RESOLVE_REFRESH_SECS",):
        if key not in result and os.environ.get(key):
            result[key] = os.environ[key]

    return result


def parse_gpu_spec(spec: str) -> tuple[str, int]:
    """Parse a GPU spec string into (variant, count).

    Accepts: 'H100x8' → ("H100", 8), '8' → ("", 8), 'H100' → ("H100", 1).
    Only a trailing 'x<digits>' is treated as a count separator, so variants
    like 'rtx4090' are not misinterpreted.
    """
    if not spec:
        raise ValueError("GPU spec must not be empty")

    # Only treat 'x' as a count separator for trailing x<1-3 digits>,
    # so model names like 'rtx4090' (4+ digit suffix) aren't misinterpreted.
    m = re.fullmatch(r"(\w+)x(\d{1,3})", spec)
    if m:
        variant, count = m.group(1), int(m.group(2))
        if count <= 0:
            raise ValueError(f"GPU count must be positive, got {count}")
        return variant, count
    if spec.isdigit():
        count = int(spec)
        if count <= 0:
            raise ValueError(f"GPU count must be positive, got {count}")
        return "", count
    if not spec.isalnum():
        raise ValueError(f"Invalid GPU spec: {spec!r}")
    return spec, 1


def build_resources(
    tpu: str | None,
    gpu: str | None,
    cpu: float = 0.5,
    memory: str = "1GB",
    disk: str = "5GB",
) -> ResourceSpec:
    """Build ResourceSpec from CLI arguments."""
    spec = ResourceSpec(cpu=cpu, memory=memory, disk=disk)

    if tpu:
        spec.device = tpu_device(tpu)
    elif gpu:
        variant, count = parse_gpu_spec(gpu)
        spec.device = gpu_device(variant, count)

    return spec


def generate_job_name(command: list[str]) -> str:
    """Generate a job name from the command."""
    script_name = "job"
    for arg in command:
        path = Path(arg)
        if path.suffix == ".py":
            script_name = path.stem
            break

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"iris-run-{script_name}-{timestamp}"


def run_iris_job(
    command: list[str],
    env_vars: dict[str, str],
    controller_url: str,
    tpu: str | None = None,
    gpu: str | None = None,
    cpu: float = 0.5,
    memory: str = "1GB",
    disk: str = "5GB",
    wait: bool = True,
    job_name: str | None = None,
    replicas: int = 1,
    max_retries: int = 0,
    timeout: int = 0,
    extras: list[str] | None = None,
    include_children_logs: bool = True,
    terminate_on_exit: bool = True,
    regions: tuple[str, ...] | None = None,
    zone: str | None = None,
    user: str | None = None,
) -> int:
    """Core job submission logic.

    Args:
        controller_url: Controller URL (from parent context tunnel).
        terminate_on_exit: If True, terminate the job on any non-normal exit
            (KeyboardInterrupt, unexpected exceptions). Normal completion is unaffected.
        regions: If provided, restrict the job to workers in these regions.
        zone: If provided, restrict the job to workers in this zone.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    env_vars = add_standard_env_vars(env_vars)
    resources = build_resources(tpu, gpu, cpu=cpu, memory=memory, disk=disk)
    job_name = job_name or generate_job_name(command)
    extras = extras or []

    constraints: list[Constraint] = []
    if regions:
        constraints.append(region_constraint(list(regions)))
    if zone:
        constraints.append(zone_constraint(zone))

    logger.info(f"Submitting job: {job_name}")
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"Resources: cpu={resources.cpu:g}, memory={resources.memory}, disk={resources.disk}")
    if resources.device and resources.device.HasField("tpu"):
        logger.info(f"TPU: {resources.device.tpu.variant}")
    if resources.device and resources.device.HasField("gpu"):
        gpu_dev = resources.device.gpu
        logger.info(f"GPU: {gpu_dev.count}x {gpu_dev.variant or 'any'}")
    if regions:
        logger.info(f"Region constraint: {', '.join(regions)}")
    if zone:
        logger.info(f"Zone constraint: {zone}")

    logger.info(f"Using controller: {controller_url}")
    return _submit_and_wait_job(
        controller_url=controller_url,
        job_name=job_name,
        command=command,
        resources=resources,
        env_vars=env_vars,
        replicas=replicas,
        max_retries=max_retries,
        timeout=timeout,
        wait=wait,
        extras=extras,
        include_children_logs=include_children_logs,
        terminate_on_exit=terminate_on_exit,
        constraints=constraints or None,
        user=user,
    )


def _submit_and_wait_job(
    controller_url: str,
    job_name: str,
    command: list[str],
    resources: ResourceSpec,
    env_vars: dict[str, str],
    replicas: int,
    max_retries: int,
    timeout: int,
    wait: bool,
    extras: list[str] | None = None,
    include_children_logs: bool = True,
    terminate_on_exit: bool = True,
    constraints: list[Constraint] | None = None,
    user: str | None = None,
) -> int:
    """Submit job and optionally wait for completion.

    When terminate_on_exit is True, the job (and its children) are killed on
    any non-normal exit: KeyboardInterrupt, unexpected exceptions, etc.
    Normal completion (success or JobFailedError) does not trigger termination.
    """
    client = IrisClient.remote(controller_url, workspace=Path.cwd())
    entrypoint = Entrypoint.from_command(*command)

    job = client.submit(
        entrypoint=entrypoint,
        name=job_name,
        resources=resources,
        environment=EnvironmentSpec(env_vars=env_vars, extras=extras or []),
        constraints=constraints,
        replicas=replicas,
        max_retries_failure=max_retries,
        timeout=Duration.from_seconds(timeout) if timeout else None,
        user=user,
    )

    logger.info(f"Job submitted: {job.job_id}")

    if not wait:
        logger.info("Job submitted (not waiting for completion)")
        return 0

    logger.info("Streaming logs (Ctrl+C to kill)...")
    try:
        try:
            status = job.wait(stream_logs=True, include_children=include_children_logs, timeout=float("inf"))
            logger.info(f"Job completed with state: {status.state}")
            return 0 if status.state == cluster_pb2.JOB_STATE_SUCCEEDED else 1
        except JobFailedError as e:
            logger.error(f"Job failed: {e}")
            return 1
    except BaseException:
        if terminate_on_exit:
            logger.info(f"Terminating job {job.job_id}...")
            terminated = _terminate_jobs(client, (str(job.job_id),), include_children=True)
            for t in terminated:
                logger.info(f"  Terminated: {t}")
        if isinstance(sys.exc_info()[1], KeyboardInterrupt):
            return 130
        raise


@click.group("job")
def job() -> None:
    """Manage Iris jobs."""


@job.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    help="""Submit jobs to Iris clusters.

Examples:

  \b
  # Simple CPU job
  iris --config cluster.yaml job run -- python script.py

  \b
  # TPU job with environment variables
  iris --config cluster.yaml job run --tpu v5litepod-16 \\
    -e WANDB_API_KEY $WANDB_API_KEY -- python train.py

  \b
  # Submit and detach
  iris --config cluster.yaml job run --no-wait -- python long_job.py
""",
)
@click.option(
    "-e",
    "--env-vars",
    "env_vars",
    multiple=True,
    type=(str, str),
    help="Set environment variables for the job (KEY VALUE). Can be repeated.",
)
@click.option("--tpu", type=str, help="TPU type to request (e.g., v5litepod-16)")
@click.option("--gpu", type=str, help="GPU spec: VARIANTxCOUNT (e.g., H100x8), COUNT (e.g., 8), or VARIANT (e.g., H100)")
@click.option("--cpu", type=float, default=0.5, show_default=True, help="Number of CPUs to request")
@click.option("--memory", type=str, default="1GB", show_default=True, help="Memory size to request (e.g., 8GB, 512MB)")
@click.option(
    "--disk", type=str, default="5GB", show_default=True, help="Ephemeral disk size to request (e.g., 64GB, 1TB)"
)
@click.option("--no-wait", is_flag=True, help="Don't wait for job completion")
@click.option("--job-name", type=str, help="Custom job name (default: auto-generated)")
@click.option("--user", type=str, help="Override the user prefix for the submitted job.")
@click.option("--replicas", type=int, default=1, help="Number of tasks for gang scheduling (default: 1)")
@click.option("--max-retries", type=int, default=0, help="Max retries on failure (default: 0)")
@click.option("--timeout", type=int, default=0, show_default=True, help="Job timeout in seconds (0 = no timeout)")
@click.option("--region", multiple=True, help="Restrict to region(s) (e.g., --region us-central2). Can be repeated.")
@click.option("--zone", type=str, help="Restrict to zone (e.g., --zone us-central2-b).")
@click.option("--extra", multiple=True, help="UV extras to install (e.g., --extra cpu). Can be repeated.")
@click.option(
    "--include-children-logs/--no-include-children-logs",
    default=True,
    help="Stream logs from child jobs (nested submissions).",
)
@click.option(
    "--terminate-on-exit/--no-terminate-on-exit",
    default=True,
    help="Terminate the job if an unexpected error occurs (default: terminate).",
)
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED, required=True)
@click.pass_context
def run(
    ctx,
    env_vars: tuple[tuple[str, str], ...],
    tpu: str | None,
    gpu: str | None,
    cpu: float,
    memory: str,
    disk: str,
    no_wait: bool,
    job_name: str | None,
    user: str | None,
    replicas: int,
    max_retries: int,
    timeout: int,
    region: tuple[str, ...],
    zone: str | None,
    extra: tuple[str, ...],
    include_children_logs: bool,
    terminate_on_exit: bool,
    cmd: tuple[str, ...],
):
    """Submit jobs to Iris clusters."""
    controller_url = require_controller_url(ctx)

    command = list(cmd)
    if not command:
        raise click.UsageError("No command provided after --")

    env_vars_dict = load_env_vars(env_vars)

    try:
        exit_code = run_iris_job(
            command=command,
            env_vars=env_vars_dict,
            controller_url=controller_url,
            tpu=tpu,
            gpu=gpu,
            cpu=cpu,
            memory=memory,
            disk=disk,
            wait=not no_wait,
            job_name=job_name,
            user=user,
            replicas=replicas,
            max_retries=max_retries,
            timeout=timeout,
            extras=list(extra),
            include_children_logs=include_children_logs,
            terminate_on_exit=terminate_on_exit,
            regions=region or None,
            zone=zone,
        )
    except Exception:
        platform = ctx.obj.get("platform")
        if platform is not None:
            try:
                platform.debug_report()
            except Exception:
                logger.debug("Controller post-mortem failed", exc_info=True)
        raise
    sys.exit(exit_code)


@job.command("stop")
@click.argument("job_id", nargs=-1, required=True)
@click.option(
    "--include-children/--no-include-children",
    default=True,
    help="Terminate child jobs under the given job ID prefix (default: include).",
)
@click.pass_context
def stop(ctx, job_id: tuple[str, ...], include_children: bool) -> None:
    """Terminate one or more jobs."""
    controller_url = require_controller_url(ctx)
    client = IrisClient.remote(controller_url, workspace=Path.cwd())
    terminated = _terminate_jobs(client, job_id, include_children)
    _print_terminated(terminated)


@job.command("kill")
@click.argument("job_id", nargs=-1, required=True)
@click.option(
    "--include-children/--no-include-children",
    default=True,
    help="Terminate child jobs under the given job ID prefix (default: include).",
)
@click.pass_context
def kill(ctx, job_id: tuple[str, ...], include_children: bool) -> None:
    """Terminate one or more jobs (alias for stop)."""
    controller_url = require_controller_url(ctx)
    client = IrisClient.remote(controller_url, workspace=Path.cwd())
    terminated = _terminate_jobs(client, job_id, include_children)
    _print_terminated(terminated)


@job.command("list")
@click.option("--state", type=str, default=None, help="Filter by state (e.g., running, pending, failed)")
@click.option("--prefix", type=str, default=None, help="Filter by job name prefix")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def list_jobs(ctx, state: str | None, prefix: str | None, json_output: bool) -> None:
    """List jobs with optional filtering."""
    controller_url = require_controller_url(ctx)
    client = IrisClient.remote(controller_url, workspace=Path.cwd())

    states: list[cluster_pb2.JobState] | None = None
    if state is not None:
        state_lower = state.lower()
        if state_lower not in _STATE_MAP:
            valid = ", ".join(sorted(_STATE_MAP.keys()))
            raise click.UsageError(f"Unknown state '{state}'. Valid states: {valid}")
        states = [_STATE_MAP[state_lower]]

    prefix_name = JobName.from_wire(prefix) if prefix else None
    jobs = client.list_jobs(states=states, prefix=prefix_name)

    # Sort by submitted_at descending (most recent first)
    jobs.sort(key=lambda j: j.submitted_at.epoch_ms, reverse=True)

    if json_output:
        serialized = [json_format.MessageToDict(j, preserving_proto_field_name=True) for j in jobs]
        click.echo(json.dumps(serialized, indent=2))
        return

    if not jobs:
        click.echo("No jobs found.")
        return

    # Build table rows
    rows: list[list[str]] = []
    has_reasons = False

    for j in jobs:
        job_id = j.job_id
        state_name = _job_state_name(j.state)
        submitted = Timestamp.from_proto(j.submitted_at).as_formatted_date() if j.submitted_at.epoch_ms else "-"
        resources = _format_resources(j.resources) if j.HasField("resources") else "-"

        # Show error for failed jobs, pending_reason for pending/unschedulable
        reason = j.error or j.pending_reason or ""
        if reason:
            has_reasons = True
            # Truncate long reasons
            reason = (reason[:60] + "...") if len(reason) > 63 else reason

        rows.append([job_id, state_name, resources, submitted, reason])

    # Build headers - only include REASON column if there are any reasons
    if has_reasons:
        headers = ["JOB ID", "STATE", "RESOURCES", "SUBMITTED", "REASON"]
    else:
        headers = ["JOB ID", "STATE", "RESOURCES", "SUBMITTED"]
        rows = [row[:4] for row in rows]

    click.echo(tabulate(rows, headers=headers, tablefmt="plain"))


@job.command("logs")
@click.argument("job_id")
@click.option("--since-ms", type=int, default=None, help="Only show logs after this epoch millisecond timestamp.")
@click.option(
    "--since-seconds",
    type=int,
    default=None,
    help="Only show logs from the last N seconds.",
)
@click.option("--follow", "-f", is_flag=True, help="Stream logs continuously.")
@click.option(
    "--include-children/--no-include-children",
    default=False,
    help="Include logs from child jobs (nested submissions).",
)
@click.pass_context
def logs(
    ctx,
    job_id: str,
    since_ms: int | None,
    since_seconds: int | None,
    follow: bool,
    include_children: bool,
) -> None:
    """Stream task logs for a job using batch log fetching."""
    if since_ms is not None and since_seconds is not None:
        raise click.UsageError("Specify only one of --since-ms or --since-seconds.")

    controller_url = require_controller_url(ctx)
    client = IrisClient.remote(controller_url, workspace=Path.cwd())

    if since_seconds is not None:
        since_ms = Timestamp.now().epoch_ms() - (since_seconds * 1000)

    start_since_ms = since_ms or 0
    job_name = JobName.from_wire(job_id)

    if follow:
        job = Job(client, job_name)
        job.wait(
            stream_logs=True,
            include_children=include_children,
            timeout=float("inf"),
            raise_on_failure=False,
        )
        return

    entries = client.fetch_task_logs(
        job_name,
        include_children=include_children,
        start=Timestamp.from_ms(start_since_ms) if start_since_ms > 0 else None,
    )
    for entry in entries:
        ts = entry.timestamp.as_short_time()
        click.echo(f"[{ts}] worker={entry.worker_id} task={entry.task_id} | {entry.data}")


@job.command("bug-report")
@click.argument("job_id")
@click.option("--file-issue", is_flag=True, help="File a GitHub issue with the report")
@click.option("--repo", type=str, default=None, help="GitHub repo (default: auto-detect from git remote)")
@click.option("--tail", type=int, default=50, help="Recent log lines per task to include")
@click.option("--labels", type=str, default="bug", help="Comma-separated labels for the GitHub issue")
@click.pass_context
def bug_report(ctx, job_id: str, file_issue: bool, repo: str | None, tail: int, labels: str):
    """Generate a diagnostic bug report for a job."""
    controller_url = require_controller_url(ctx)
    report = gather_bug_report(controller_url, JobName.from_wire(job_id), tail=tail)
    markdown = format_bug_report(report)

    if file_issue:
        title = f"[Iris] Job {report.job_id} {report.state_name}: {report.error_summary}"
        url = file_github_issue(title, markdown, repo=repo, labels=labels.split(","))
        if url:
            click.echo(f"Filed issue: {url}")
        else:
            click.echo("Failed to file issue. Report printed below:\n")
            click.echo(markdown)
    else:
        click.echo(markdown)
