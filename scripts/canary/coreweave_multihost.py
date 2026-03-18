#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave multi-host GPU canary runner (2-phase).

Phase 1 (CPU): boots a CPU-only cluster, submits a lightweight training job,
and validates the full submit → schedule → run → succeed lifecycle.

Phase 2 (GPU): boots the multi-host GPU cluster and runs the real canary ferry.

Default command is `run`: runs both phases sequentially, skipping GPU if CPU
fails.

Usage:
    uv run scripts/canary/coreweave_multihost.py                # run both phases
    uv run scripts/canary/coreweave_multihost.py run             # explicit run
    uv run scripts/canary/coreweave_multihost.py cpu             # CPU phase only
    uv run scripts/canary/coreweave_multihost.py gpu             # GPU phase only (skip CPU)
    uv run scripts/canary/coreweave_multihost.py boot            # just boot GPU cluster
    uv run scripts/canary/coreweave_multihost.py teardown        # full teardown (both)
    uv run scripts/canary/coreweave_multihost.py validate        # check prereqs
    uv run scripts/canary/coreweave_multihost.py diagnostics     # dump controller/worker state
    uv run scripts/canary/coreweave_multihost.py loop            # run in a loop until success
"""

import contextlib
import datetime
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import click
import fsspec
import s3fs

from iris.cli.cluster import _build_cluster_images, _pin_latest_images
from iris.cli.job import add_standard_env_vars, build_resources
from iris.cli.main import _configure_client_s3, create_client_token_provider
from iris.client import IrisClient
from iris.cluster.config import IrisConfig
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import cluster_pb2

logger = logging.getLogger(__name__)

GPU_CONFIG = "lib/iris/examples/coreweave-canary-multihost.yaml"
CPU_CONFIG = "lib/iris/examples/coreweave-cpu.yaml"
GPU_NAMESPACE = "iris-canary-mh"
GPU_LABEL_PREFIX = "iris-canary-mh"
CPU_NAMESPACE = "iris-canary-cpu"
CPU_LABEL_PREFIX = "iris-canary-cpu"

CANARY_ENV_DEFAULTS = {
    "CANARY_ACCELERATOR": "gpu",
    "CANARY_MULTI_HOST": "true",
    "CANARY_BATCH_SIZE": "32",
    "CANARY_TARGET_TOKENS": "6553600",
    "WANDB_ENTITY": "marin-community",
    "WANDB_PROJECT": "marin",
    "MARIN_PREFIX": "s3://marin-na/marin/",
}

REQUIRED_ENV_VARS = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "WANDB_API_KEY"]

TERMINAL_STATES = frozenset(
    {
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        cluster_pb2.JOB_STATE_UNSCHEDULABLE,
    }
)

POLL_INTERVAL_SECONDS = 10


def _job_state_name(state: int) -> str:
    return cluster_pb2.JobState.Name(state).removeprefix("JOB_STATE_")


def _task_state_name(state: int) -> str:
    return cluster_pb2.TaskState.Name(state).removeprefix("TASK_STATE_")


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# kubectl helpers
# ---------------------------------------------------------------------------


def _kubeconfig_path() -> str:
    return os.environ.get("KUBECONFIG", str(Path.home() / ".kube/coreweave-iris"))


def _kubectl(namespace: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["kubectl", "--kubeconfig", _kubeconfig_path(), "-n", namespace, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _controller_deployment_exists(namespace: str) -> bool:
    result = _kubectl(
        namespace,
        "get",
        "deployment",
        "-l",
        "app=iris-controller",
        "--no-headers",
        check=False,
    )
    return bool(result.stdout.strip())


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def _generate_run_id(prefix: str = "mh-canary") -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}"


# ---------------------------------------------------------------------------
# Prereq validation
# ---------------------------------------------------------------------------


def validate_prereqs(config_path: str) -> None:
    errors: list[str] = []

    if not Path(config_path).exists():
        errors.append(f"Config file not found: {config_path}")

    kubeconfig = _kubeconfig_path()
    if not Path(kubeconfig).exists():
        errors.append(f"Kubeconfig not found: {kubeconfig}")

    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            errors.append(f"Environment variable not set: {var}")

    if errors:
        for e in errors:
            click.echo(f"  FAIL: {e}", err=True)
        raise SystemExit(1)

    result = subprocess.run(
        ["kubectl", "--kubeconfig", kubeconfig, "cluster-info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        click.echo(f"  FAIL: cannot reach CoreWeave cluster: {result.stderr.strip()}", err=True)
        raise SystemExit(1)

    IrisConfig.load(config_path)
    click.echo("All prereqs OK")


# ---------------------------------------------------------------------------
# S3 / state management
# ---------------------------------------------------------------------------


def _make_s3fs(endpoint: str) -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        endpoint_url=endpoint,
        key=os.environ.get("R2_ACCESS_KEY_ID"),
        secret=os.environ.get("R2_SECRET_ACCESS_KEY"),
        client_kwargs={"region_name": "auto"},
        skip_instance_cache=True,
    )


def clear_controller_state(config_path: str) -> None:
    iris_config = IrisConfig.load(config_path)
    _configure_client_s3(iris_config.proto)
    state_dir = iris_config.proto.storage.remote_state_dir
    if not state_dir:
        click.echo("No remote_state_dir configured, skipping state clear")
        return
    endpoint = iris_config.proto.platform.coreweave.object_storage_endpoint
    checkpoint_dir = f"{state_dir}/controller-state"
    _, path = fsspec.core.url_to_fs(checkpoint_dir)
    fs = _make_s3fs(endpoint)
    if fs.exists(path):
        fs.rm(path, recursive=True)
        click.echo(f"Cleared controller state: {checkpoint_dir}")
    else:
        click.echo(f"No controller state to clear: {checkpoint_dir}")


# ---------------------------------------------------------------------------
# Cluster lifecycle
# ---------------------------------------------------------------------------


def cold_boot(config_path: str, verbose: bool = False) -> str:
    click.echo("=== Cold boot: creating all resources ===")
    iris_config = IrisConfig.load(config_path)
    config = iris_config.proto

    _pin_latest_images(config)
    _build_cluster_images(config, verbose=verbose)

    platform = iris_config.platform()
    address = platform.start_controller(config)
    click.echo(f"Controller started at {address}")
    return address


def warm_reboot(config_path: str, namespace: str, label_prefix: str, verbose: bool = False) -> str:
    click.echo("=== Warm reboot: wiping workers, restarting controller ===")

    managed_label = f"iris-{label_prefix}-managed=true"
    _kubectl(namespace, "delete", "pods", "-l", managed_label, "--ignore-not-found", check=False)
    _kubectl(namespace, "delete", "configmaps", "-l", managed_label, "--ignore-not-found", check=False)

    clear_controller_state(config_path)

    iris_config = IrisConfig.load(config_path)
    config = iris_config.proto

    _pin_latest_images(config)
    _build_cluster_images(config, verbose=verbose)

    platform = iris_config.platform()
    address = platform.start_controller(config)
    click.echo(f"Controller restarted at {address}")
    return address


def boot_cluster(config_path: str, namespace: str, label_prefix: str, verbose: bool = False) -> str:
    if _controller_deployment_exists(namespace):
        return warm_reboot(config_path, namespace, label_prefix, verbose=verbose)
    return cold_boot(config_path, verbose=verbose)


def teardown_cluster(config_path: str, namespace: str, label_prefix: str) -> None:
    click.echo(f"=== Teardown ({namespace}) ===")
    iris_config = IrisConfig.load(config_path)
    platform = iris_config.platform()
    try:
        names = platform.stop_all(iris_config.proto)
        click.echo(f"Stopped {len(names)} resource(s)")
    except Exception as e:
        click.echo(f"cluster stop failed (continuing): {e}", err=True)
    finally:
        platform.shutdown()

    managed_label = f"iris-{label_prefix}-managed=true"
    _kubectl(namespace, "delete", "nodepool", "-l", managed_label, "--ignore-not-found", check=False)
    click.echo("Teardown complete")


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _connect_client(config_path: str):
    """Context manager yielding an IrisClient connected through a platform tunnel."""
    iris_config = IrisConfig.load(config_path)
    _configure_client_s3(iris_config.proto)

    platform = iris_config.platform()
    controller_address = iris_config.controller_address()
    if not controller_address:
        controller_address = platform.discover_controller(iris_config.proto.controller)

    token_provider = None
    if iris_config.proto.HasField("auth"):
        token_provider = create_client_token_provider(iris_config.proto.auth)

    with platform.tunnel(address=controller_address) as controller_url:
        client = IrisClient.remote(controller_url, workspace=Path.cwd(), token_provider=token_provider)
        try:
            yield client
        finally:
            client.shutdown()


def _print_task_details(client: IrisClient, job) -> None:
    """Print per-task status for multi-task jobs."""
    for task in job.tasks():
        status = task.status()
        state = _task_state_name(status.state)
        worker = status.worker_id or "(unassigned)"
        click.echo(f"  task/{task.task_index}: {state} worker={worker}")
        if status.error:
            click.echo(f"    error: {status.error}")


def _dump_job_logs(client: IrisClient, job, max_lines: int = 200) -> None:
    """Fetch and print recent logs for a failed job."""
    click.echo(f"\n=== Job logs (last {max_lines} lines) ===")
    try:
        logs = client.fetch_task_logs(
            job.job_id,
            include_children=True,
            max_lines=max_lines,
        )
        if not logs:
            click.echo("(no logs available)")
            return
        for entry in logs:
            click.echo(entry.data)
    except Exception as e:
        click.echo(f"(failed to fetch logs: {e})")


def submit_and_monitor(
    config_path: str,
    job_name: str,
    command: list[str],
    env_vars: dict[str, str],
    cpu: float,
    memory: str,
    disk: str,
    timeout_seconds: float = 3600,
    extras: list[str] | None = None,
    phase_label: str = "Job",
) -> int:
    """Submit a job to Iris and poll status with structured progress output.

    Returns 0 on success, 1 on failure.
    """
    env_vars = add_standard_env_vars(env_vars)
    resources = build_resources(tpu=None, gpu=None, cpu=cpu, memory=memory, disk=disk)
    entrypoint = Entrypoint.from_command(*command)
    environment = EnvironmentSpec(env_vars=env_vars, extras=extras or [])

    with _connect_client(config_path) as client:
        click.echo(f"[{phase_label}] Submitting: {job_name}")
        click.echo(f"[{phase_label}] Command: {' '.join(command)}")

        job = client.submit(
            entrypoint=entrypoint,
            name=job_name,
            resources=resources,
            environment=environment,
        )
        click.echo(f"[{phase_label}] Submitted: {job.job_id}")

        start = time.monotonic()
        last_state = None
        last_detail_time = 0.0
        deadline = start + timeout_seconds

        while True:
            elapsed = time.monotonic() - start
            if time.monotonic() > deadline:
                click.echo(f"[{_format_elapsed(elapsed)}] TIMEOUT after {timeout_seconds}s")
                try:
                    job.terminate()
                    click.echo(f"[{_format_elapsed(elapsed)}] Job terminated")
                except Exception:
                    pass
                _dump_job_logs(client, job)
                return 1

            try:
                status = job.status()
            except Exception as e:
                click.echo(f"[{_format_elapsed(elapsed)}] Status poll failed: {e}")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            state = status.state
            state_name = _job_state_name(state)

            if state != last_state:
                # State transition — always print
                pending_reason = ""
                if state == cluster_pb2.JOB_STATE_PENDING and status.pending_reason:
                    pending_reason = f" ({status.pending_reason})"
                elif state == cluster_pb2.JOB_STATE_UNSCHEDULABLE and status.pending_reason:
                    pending_reason = f" ({status.pending_reason})"

                error_info = ""
                if status.error:
                    error_info = f" — {status.error}"

                click.echo(f"[{_format_elapsed(elapsed)}] {state_name}{pending_reason}{error_info}")
                last_state = state

            # Periodic task detail dump (every 30s while running)
            if state == cluster_pb2.JOB_STATE_RUNNING and (elapsed - last_detail_time) > 30:
                _print_task_details(client, job)
                last_detail_time = elapsed

            if state in TERMINAL_STATES:
                break

            time.sleep(POLL_INTERVAL_SECONDS)

        elapsed = time.monotonic() - start

        if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
            click.echo(f"[{_format_elapsed(elapsed)}] {phase_label} PASSED")
            return 0

        click.echo(f"[{_format_elapsed(elapsed)}] {phase_label} FAILED ({_job_state_name(status.state)})")
        _print_task_details(client, job)
        _dump_job_logs(client, job)
        return 1


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


R2_ENDPOINT = "https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com"


def _build_secret_env() -> dict[str, str]:
    """Collect secret env vars that should be forwarded to jobs.

    Maps R2 credentials to the AWS_* env vars that fsspec/s3fs/boto3 expect,
    and sets FSSPEC_S3 so all fsspec operations hit the correct R2 endpoint.
    """
    env: dict[str, str] = {}
    for key in ["WANDB_API_KEY", "HF_TOKEN"]:
        val = os.environ.get(key, "")
        if val:
            env[key] = val

    r2_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    if r2_key and r2_secret:
        env["AWS_ACCESS_KEY_ID"] = r2_key
        env["AWS_SECRET_ACCESS_KEY"] = r2_secret
        env["AWS_ENDPOINT_URL"] = R2_ENDPOINT
        env["FSSPEC_S3"] = json.dumps({"endpoint_url": R2_ENDPOINT})

    return env


def run_cpu_phase(verbose: bool = False, timeout: float = 1200) -> int:
    """Phase 1: boot CPU cluster, run a lightweight training job."""
    click.echo("\n" + "=" * 60)
    click.echo("PHASE 1: CPU smoke test")
    click.echo("=" * 60)

    validate_prereqs(CPU_CONFIG)
    boot_cluster(CPU_CONFIG, CPU_NAMESPACE, CPU_LABEL_PREFIX, verbose=verbose)

    run_id = _generate_run_id(prefix="cpu-canary")
    env_vars = {
        "WANDB_ENTITY": "marin-community",
        "WANDB_PROJECT": "marin",
        "MARIN_PREFIX": "s3://marin-na/marin/",
        **_build_secret_env(),
    }

    exit_code = submit_and_monitor(
        config_path=CPU_CONFIG,
        job_name=f"cpu-canary-{run_id}",
        command=["python", "-m", "experiments.tutorials.train_tiny_model_cpu"],
        env_vars=env_vars,
        cpu=4,
        memory="32GB",
        disk="32GB",
        timeout_seconds=timeout,
        extras=["cpu"],
        phase_label="CPU",
    )

    if exit_code != 0:
        dump_diagnostics(CPU_NAMESPACE, CPU_LABEL_PREFIX)

    return exit_code


def run_gpu_phase(verbose: bool = False) -> int:
    """Phase 2: boot multi-host GPU cluster, run the real canary ferry."""
    click.echo("\n" + "=" * 60)
    click.echo("PHASE 2: Multi-host GPU canary")
    click.echo("=" * 60)

    validate_prereqs(GPU_CONFIG)
    boot_cluster(GPU_CONFIG, GPU_NAMESPACE, GPU_LABEL_PREFIX, verbose=verbose)

    run_id = _generate_run_id()
    env_vars: dict[str, str] = {}
    for key, default in CANARY_ENV_DEFAULTS.items():
        env_vars[key] = os.environ.get(key, default)
    env_vars["RUN_ID"] = run_id
    env_vars.update(_build_secret_env())

    exit_code = submit_and_monitor(
        config_path=GPU_CONFIG,
        job_name=f"multihost-canary-{run_id}",
        command=["python", "-m", "experiments.ferries.canary_ferry"],
        env_vars=env_vars,
        cpu=1,
        memory="16GB",
        disk="16GB",
        timeout_seconds=7200,
        extras=["cpu"],
        phase_label="GPU",
    )

    if exit_code != 0:
        dump_diagnostics(GPU_NAMESPACE, GPU_LABEL_PREFIX)

    return exit_code


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def dump_diagnostics(namespace: str, label_prefix: str) -> None:
    click.echo(f"\n=== Controller logs ({namespace}) ===")
    result = _kubectl(namespace, "logs", "-l", "app=iris-controller", "--tail=200", check=False)
    click.echo(result.stdout or "(no logs)")

    click.echo(f"\n=== Worker pods ({namespace}) ===")
    result = _kubectl(namespace, "get", "pods", "-l", f"{label_prefix}-role=worker", "-o", "wide", check=False)
    click.echo(result.stdout or "(no worker pods)")

    click.echo(f"\n=== Error events ({namespace}) ===")
    result = _kubectl(
        namespace,
        "get",
        "events",
        "--sort-by=.lastTimestamp",
        "--field-selector=type!=Normal",
        check=False,
    )
    lines = result.stdout.strip().split("\n")
    click.echo("\n".join(lines[-30:]) if lines else "(no error events)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.pass_context
def cli(ctx, verbose: bool):
    """CoreWeave canary runner (2-phase: CPU then GPU).

    Default (no subcommand): runs both phases, skipping GPU if CPU fails.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if verbose:
        logging.getLogger("iris").setLevel(logging.DEBUG)
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@cli.command()
@click.pass_context
def validate(ctx):
    """Check prerequisites for both CPU and GPU configs."""
    validate_prereqs(CPU_CONFIG)
    click.echo("CPU config OK")
    validate_prereqs(GPU_CONFIG)
    click.echo("GPU config OK")


@cli.command()
@click.pass_context
def boot(ctx):
    """Start GPU cluster (cold boot or warm reboot as needed)."""
    validate_prereqs(GPU_CONFIG)
    boot_cluster(GPU_CONFIG, GPU_NAMESPACE, GPU_LABEL_PREFIX, verbose=ctx.obj["verbose"])


@cli.command()
@click.pass_context
def teardown(ctx):
    """Full teardown: stop both CPU and GPU clusters."""
    teardown_cluster(CPU_CONFIG, CPU_NAMESPACE, CPU_LABEL_PREFIX)
    teardown_cluster(GPU_CONFIG, GPU_NAMESPACE, GPU_LABEL_PREFIX)


@cli.command()
@click.pass_context
def diagnostics(ctx):
    """Dump controller logs, worker pods, and error events for both clusters."""
    click.echo("=== CPU cluster ===")
    dump_diagnostics(CPU_NAMESPACE, CPU_LABEL_PREFIX)
    click.echo("\n=== GPU cluster ===")
    dump_diagnostics(GPU_NAMESPACE, GPU_LABEL_PREFIX)


@cli.command()
@click.option("--timeout", default=1200, help="CPU phase timeout in seconds")
@click.pass_context
def cpu(ctx, timeout: float):
    """Run CPU smoke test only (phase 1)."""
    exit_code = run_cpu_phase(verbose=ctx.obj["verbose"], timeout=timeout)
    raise SystemExit(exit_code)


@cli.command()
@click.pass_context
def gpu(ctx):
    """Run GPU canary only (phase 2), skipping CPU smoke test."""
    exit_code = run_gpu_phase(verbose=ctx.obj["verbose"])
    raise SystemExit(exit_code)


@cli.command()
@click.pass_context
def run(ctx):
    """Run both phases: CPU smoke test, then multi-host GPU canary."""
    verbose = ctx.obj["verbose"]

    exit_code = run_cpu_phase(verbose=verbose)
    if exit_code != 0:
        click.echo("\nCPU phase failed — skipping GPU phase", err=True)
        raise SystemExit(exit_code)

    exit_code = run_gpu_phase(verbose=verbose)
    if exit_code == 0:
        click.echo("\n=== Both phases PASSED ===")
    else:
        click.echo("\n=== GPU phase FAILED ===", err=True)
    raise SystemExit(exit_code)


@cli.command()
@click.option("--max-attempts", default=0, help="Max attempts (0 = unlimited)")
@click.pass_context
def loop(ctx, max_attempts: int):
    """Run both phases in a loop until success."""
    verbose = ctx.obj["verbose"]

    attempt = 0
    while max_attempts == 0 or attempt < max_attempts:
        attempt += 1
        click.echo(f"\n{'=' * 20} Attempt {attempt} {'=' * 20}")

        cpu_exit = run_cpu_phase(verbose=verbose)
        if cpu_exit != 0:
            click.echo(f"CPU phase failed on attempt {attempt}, retrying...")
            time.sleep(10)
            continue

        gpu_exit = run_gpu_phase(verbose=verbose)
        if gpu_exit == 0:
            click.echo(f"Both phases PASSED on attempt {attempt}")
            raise SystemExit(0)

        click.echo(f"GPU phase failed on attempt {attempt}, retrying...")
        time.sleep(10)

    click.echo(f"Exhausted {max_attempts} attempts", err=True)
    raise SystemExit(1)


if __name__ == "__main__":
    cli()
