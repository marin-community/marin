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
    uv run lib/iris/scripts/iris_run.py \\
        --config lib/iris/examples/eu-west4.yaml \\
        --tpu v5litepod-16 \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -- python experiments/train.py --epochs 10
"""

import argparse
import getpass
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from iris.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.cluster.vm.debug import controller_tunnel
from iris.rpc import cluster_pb2

logger = logging.getLogger(__name__)


def load_cluster_config(config_path: Path) -> dict:
    """Load cluster config YAML and extract zone/project_id.

    Args:
        config_path: Path to cluster YAML file

    Returns:
        Dict with 'zone' and 'project_id' keys

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required fields are missing
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty or invalid YAML in {config_path}")

    zone = data.get("zone")
    project_id = data.get("project_id")

    if not zone:
        raise ValueError(f"Missing 'zone' in {config_path}")
    if not project_id:
        raise ValueError(f"Missing 'project_id' in {config_path}")

    return {"zone": zone, "project_id": project_id}


def load_env_vars(env_flags: list[list[str]] | None) -> dict[str, str]:
    """Load environment variables from .marin.yaml and merge with flags.

    Args:
        env_flags: List of [KEY, VALUE] or [KEY] pairs from argparse

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


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Submit jobs to Iris clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple CPU job
  iris_run.py --config cluster.yaml -- python script.py

  # TPU job with environment variables
  iris_run.py --config cluster.yaml --tpu v5litepod-16 \\
    -e WANDB_API_KEY $WANDB_API_KEY -- python train.py

  # Submit and detach
  iris_run.py --config cluster.yaml --no-wait -- python long_job.py
        """,
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to Iris cluster config YAML",
    )
    parser.add_argument(
        "--env_vars",
        "-e",
        action="append",
        nargs="+",
        metavar=("KEY", "VALUE"),
        help="Set environment variables for the job. If only KEY is provided, VALUE is set to empty string.",
    )
    parser.add_argument(
        "--tpu",
        type=str,
        help="TPU type to request (e.g., v5litepod-16)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="Number of GPUs to request",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help="Number of CPUs to request (default: 1)",
    )
    parser.add_argument(
        "--memory",
        type=str,
        help="Memory size to request (e.g., 8GB, 512MB; default: 2GB)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for job completion",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        help="Custom job name (default: auto-generated)",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="Number of tasks for gang scheduling (default: 1)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Max retries on failure (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Job timeout in seconds (default: 0 = no timeout)",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run (must start with --)",
    )

    args = parser.parse_args()

    # Validate and strip -- from command
    if not args.cmd:
        logger.error("No command provided. Command must start with --")
        sys.exit(1)

    if args.cmd[0] == "--":
        command = args.cmd[1:]
    else:
        logger.error("Command must start with --")
        sys.exit(1)

    if not command:
        logger.error("No command provided after --")
        sys.exit(1)

    # Load cluster config
    try:
        config = load_cluster_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load cluster config: {e}")
        sys.exit(1)

    # Load environment variables
    try:
        env_vars = load_env_vars(args.env_vars)
        env_vars = add_standard_env_vars(env_vars)
    except ValueError as e:
        logger.error(f"Failed to parse environment variables: {e}")
        sys.exit(1)

    # Build resources
    resources = build_resources(
        tpu=args.tpu,
        gpu=args.gpu,
        cpu=args.cpu,
        memory=args.memory,
    )

    # Generate job name
    job_name = args.job_name or generate_job_name(command)

    logger.info(f"Submitting job: {job_name}")
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"Resources: cpu={resources.cpu}, memory={resources.memory}")
    if resources.device and resources.device.HasField("tpu"):
        logger.info(f"TPU: {resources.device.tpu.variant}")

    # Establish tunnel and submit job
    try:
        with controller_tunnel(
            zone=config["zone"],
            project=config["project_id"],
            tunnel_logger=logger,
        ) as controller_url:
            logger.info(f"Connected to controller: {controller_url}")

            client = IrisClient.remote(controller_url, workspace=Path.cwd())

            # Build entrypoint from command
            entrypoint = Entrypoint.from_command(*command)

            # Submit job
            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=resources,
                environment=EnvironmentSpec(env_vars=env_vars),
                replicas=args.replicas,
                max_retries_failure=args.max_retries,
                timeout_seconds=args.timeout,
            )

            logger.info(f"Job submitted: {job.job_id}")

            if not args.no_wait:
                logger.info("Streaming logs (Ctrl+C to detach)...")
                try:
                    status = job.wait(stream_logs=True, timeout=float("inf"))
                    logger.info(f"Job completed with state: {status.state}")
                    if status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
                        sys.exit(1)
                except KeyboardInterrupt:
                    logger.info("Detached from job (job continues running)")
            else:
                logger.info("Job submitted (not waiting for completion)")

    except Exception as e:
        logger.error(f"Failed to submit or run job: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
