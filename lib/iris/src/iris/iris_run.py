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

"""CLI for submitting jobs to Iris clusters.

Usage:
    uv run iris-run \\
        --config lib/iris/examples/eu-west4.yaml \\
        --tpu v5litepod-16 \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -- python experiments/train.py --epochs 10
"""

import getpass
import logging
import os
import sys
import time
from pathlib import Path

import click
import yaml

from iris.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.cluster.vm.debug import controller_tunnel
from iris.rpc import cluster_pb2
from iris.time_utils import Duration

logger = logging.getLogger(__name__)


def load_cluster_config(config_path: Path) -> dict:
    """Load cluster config YAML and extract connection info.

    Args:
        config_path: Path to cluster YAML file

    Returns:
        Dict with 'zone', 'project_id', and optionally 'controller_address' keys.
        For local clusters, controller_address bypasses SSH tunneling.

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required fields are missing
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty or invalid YAML in {config_path}")

    zone = data.get("zone")
    project_id = data.get("project_id", "")
    controller_address = data.get("controller_address")

    if not zone:
        raise ValueError(f"Missing 'zone' in {config_path}")

    # For remote clusters, project_id is required for SSH tunneling
    if not controller_address and not project_id:
        raise ValueError(f"Missing 'project_id' in {config_path} (required for remote clusters)")

    return {"zone": zone, "project_id": project_id, "controller_address": controller_address}


def load_env_vars(env_flags: tuple[tuple[str, ...], ...] | None) -> dict[str, str]:
    """Load environment variables from .marin.yaml and merge with flags.

    Args:
        env_flags: Tuple of (KEY,) or (KEY, VALUE) tuples from Click

    Returns:
        Merged environment variables

    Raises:
        ValueError: If key contains '=' or other validation fails
    """
    # 1. Load from .marin.yaml
    env_vars = {}
    marin_yaml = Path(".marin.yaml")
    if marin_yaml.exists():
        with open(marin_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg.get("env"), dict):
            for k, v in cfg["env"].items():
                env_vars[str(k)] = "" if v is None else str(v)

    # 2. Auto-include tokens from environment
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        if key not in env_vars and os.environ.get(key):
            env_vars[key] = os.environ[key]

    # 3. Merge flags
    if env_flags:
        for item in env_flags:
            if len(item) > 2:
                raise ValueError(f"Too many values for env var: {' '.join(item)}")
            if "=" in item[0]:
                raise ValueError(
                    f"Key cannot contain '=': {item[0]}\n"
                    f"You probably meant to do '-e {' '.join(item[0].split('='))}'"
                )
            env_vars[item[0]] = item[1] if len(item) == 2 else ""

    return env_vars


def add_standard_env_vars(env_vars: dict[str, str]) -> dict[str, str]:
    """Add standard environment variables used by Marin jobs.

    Args:
        env_vars: Base environment variables

    Returns:
        New dict with standard variables added (not overriding existing)
    """
    # Copy input dict to avoid mutation
    result = dict(env_vars)

    defaults = {
        "PYTHONPATH": ".",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "~/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }

    # Add defaults without overriding user-provided values
    for key, value in defaults.items():
        if key not in result:
            result[key] = value

    # Pass through specific env vars if set
    for key in ("GCS_RESOLVE_REFRESH_SECS",):
        if key not in result and os.environ.get(key):
            result[key] = os.environ[key]

    return result


def build_resources(
    tpu: str | None,
    gpu: int | None,
    cpu: int | None,
    memory: str | None,
) -> ResourceSpec:
    """Build ResourceSpec from CLI arguments.

    Args:
        tpu: TPU type (e.g., "v5litepod-16")
        gpu: Number of GPUs
        cpu: Number of CPUs
        memory: Memory size (e.g., "8GB")

    Returns:
        ResourceSpec with specified resources

    Raises:
        ValueError: If GPU is requested (not yet supported)
    """
    spec = ResourceSpec(
        cpu=cpu or 1,
        memory=memory or "2GB",
    )

    if tpu:
        spec.device = tpu_device(tpu)
    elif gpu:
        raise ValueError("GPU support not yet implemented in Iris")

    return spec


def generate_job_name(command: list[str]) -> str:
    """Generate a job name from the command.

    Args:
        command: Command argv

    Returns:
        Job name with timestamp
    """
    # Extract script name if it's a Python script
    script_name = "job"
    for arg in command:
        path = Path(arg)
        if path.suffix == ".py":
            script_name = path.stem
            break

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    username = getpass.getuser()
    return f"iris-run-{username}-{script_name}-{timestamp}"


def run_iris_job(
    config_path: Path,
    command: list[str],
    env_vars: dict[str, str],
    tpu: str | None = None,
    gpu: int | None = None,
    cpu: int | None = None,
    memory: str | None = None,
    wait: bool = True,
    job_name: str | None = None,
    replicas: int = 1,
    max_retries: int = 0,
    timeout: int = 0,
    extras: list[str] | None = None,
) -> int:
    """Core job submission logic (testable without CLI).

    Args:
        config_path: Path to cluster config YAML
        command: Command to run (argv list)
        env_vars: Environment variables for the job
        tpu: TPU type to request (e.g., v5litepod-16)
        gpu: Number of GPUs to request
        cpu: Number of CPUs to request
        memory: Memory size to request (e.g., "8GB")
        wait: Whether to wait for job completion
        job_name: Custom job name (auto-generated if None)
        replicas: Number of tasks for gang scheduling
        max_retries: Max retries on failure
        timeout: Job timeout in seconds (0 = no timeout)

    Returns:
        Exit code: 0 for success, 1 for failure

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or GPU is requested
        Various exceptions from IrisClient if submission fails
    """
    config = load_cluster_config(config_path)
    env_vars = add_standard_env_vars(env_vars)
    resources = build_resources(tpu, gpu, cpu, memory)
    job_name = job_name or generate_job_name(command)
    extras = extras or []

    logger.info(f"Submitting job: {job_name}")
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"Resources: cpu={resources.cpu}, memory={resources.memory}")
    if resources.device and resources.device.HasField("tpu"):
        logger.info(f"TPU: {resources.device.tpu.variant}")

    # Check if this is a local cluster with direct controller access
    if config["controller_address"]:
        # Local cluster - connect directly without SSH tunnel
        controller_url = config["controller_address"]
        logger.info(f"Connecting directly to controller: {controller_url}")
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
        )
    else:
        # Remote cluster - use SSH tunnel
        with controller_tunnel(
            zone=config["zone"],
            project=config["project_id"],
            tunnel_logger=logger,
        ) as controller_url:
            logger.info(f"Connected to controller: {controller_url}")
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
) -> int:
    """Submit job and optionally wait for completion.

    Args:
        controller_url: Controller URL
        job_name: Name for the job
        command: Command to run
        resources: Resource spec
        env_vars: Environment variables
        replicas: Number of replicas
        max_retries: Max retry attempts
        timeout: Job timeout in seconds
        wait: Whether to wait for completion

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    client = IrisClient.remote(controller_url, workspace=Path.cwd())
    entrypoint = Entrypoint.from_command(*command)

    job = client.submit(
        entrypoint=entrypoint,
        name=job_name,
        resources=resources,
        environment=EnvironmentSpec(env_vars=env_vars, extras=extras or []),
        replicas=replicas,
        max_retries_failure=max_retries,
        timeout=Duration.from_seconds(timeout) if timeout else None,
    )

    logger.info(f"Job submitted: {job.job_id}")

    if wait:
        logger.info("Streaming logs (Ctrl+C to detach)...")
        try:
            from iris.client.client import JobFailedError

            try:
                status = job.wait(stream_logs=True, timeout=float("inf"))
                logger.info(f"Job completed with state: {status.state}")
                return 0 if status.state == cluster_pb2.JOB_STATE_SUCCEEDED else 1
            except JobFailedError as e:
                logger.info(f"Job failed with state: {e.status.state}")
                return 1
        except KeyboardInterrupt:
            logger.info("Detached from job (job continues running)")
            return 0
    else:
        logger.info("Job submitted (not waiting for completion)")
        return 0


@click.command(
    context_settings={"ignore_unknown_options": True},
    help="""Submit jobs to Iris clusters.

Examples:

  \b
  # Simple CPU job
  iris-run --config cluster.yaml -- python script.py

  \b
  # TPU job with environment variables
  iris-run --config cluster.yaml --tpu v5litepod-16 \\
    -e WANDB_API_KEY $WANDB_API_KEY -- python train.py

  \b
  # Submit and detach
  iris-run --config cluster.yaml --no-wait -- python long_job.py
""",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to Iris cluster config YAML",
)
@click.option(
    "-e",
    "--env-vars",
    "env_vars",
    multiple=True,
    type=(str, str),
    help="Set environment variables for the job (KEY VALUE). Can be repeated.",
)
@click.option(
    "--tpu",
    type=str,
    help="TPU type to request (e.g., v5litepod-16)",
)
@click.option(
    "--gpu",
    type=int,
    help="Number of GPUs to request",
)
@click.option(
    "--cpu",
    type=int,
    help="Number of CPUs to request (default: 1)",
)
@click.option(
    "--memory",
    type=str,
    help="Memory size to request (e.g., 8GB, 512MB; default: 2GB)",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="Don't wait for job completion",
)
@click.option(
    "--job-name",
    type=str,
    help="Custom job name (default: auto-generated)",
)
@click.option(
    "--replicas",
    type=int,
    default=1,
    help="Number of tasks for gang scheduling (default: 1)",
)
@click.option(
    "--max-retries",
    type=int,
    default=0,
    help="Max retries on failure (default: 0)",
)
@click.option(
    "--timeout",
    type=int,
    default=0,
    help="Job timeout in seconds (default: 0 = no timeout)",
)
@click.option(
    "--extra",
    multiple=True,
    help="UV extras to install (e.g., --extra cpu). Can be repeated.",
)
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED, required=True)
def main(
    config: Path,
    env_vars: tuple[tuple[str, str], ...],
    tpu: str | None,
    gpu: int | None,
    cpu: int | None,
    memory: str | None,
    no_wait: bool,
    job_name: str | None,
    replicas: int,
    max_retries: int,
    timeout: int,
    extra: tuple[str, ...],
    cmd: tuple[str, ...],
):
    """Submit jobs to Iris clusters."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    command = list(cmd)
    if not command:
        raise click.UsageError("No command provided after --")

    # Load env vars - let exceptions propagate
    env_vars_dict = load_env_vars(env_vars)

    # Call core logic - let exceptions propagate
    exit_code = run_iris_job(
        config_path=config,
        command=command,
        env_vars=env_vars_dict,
        tpu=tpu,
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        wait=not no_wait,
        job_name=job_name,
        replicas=replicas,
        max_retries=max_retries,
        timeout=timeout,
        extras=list(extra),
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
