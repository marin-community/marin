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

"""Command-line interface for zephyr launcher."""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click
from fray.v2.client import current_client
from fray.v2.local_backend import LocalClient
from fray.v2.types import ResourceConfig

from zephyr.execution import ZephyrContext, _default_zephyr_context

logger = logging.getLogger(__name__)

# Silence noisy httpx logs (HTTP Request: POST ...)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class CliConfig:
    memory: str | None = None
    num_cpus: float | None = None
    num_gpus: float | None = None
    max_parallelism: int = 100
    backend: str = "ray"
    cluster: str | None = None
    entry_point: str = "main"
    dry_run: bool = False

    ray_options: dict = field(default_factory=dict)


def validate_backend_config(config: CliConfig) -> None:
    """Validate backend configuration consistency.

    Raises:
        click.UsageError: If invalid backend configuration
    """
    # Iris backend requires cluster submission
    if config.backend == "iris" and not config.cluster:
        raise click.UsageError(
            "--backend=iris requires --cluster with path to Iris cluster config YAML\n"
            "Example: --backend=iris --cluster=lib/iris/examples/eu-west4.yaml"
        )

    # Ray backend with cluster mode
    if config.backend == "ray" and config.cluster:
        # Valid - this is Ray cluster submission mode
        pass

    # Local mode cannot have cluster flag
    if config.backend == "local" and config.cluster:
        raise click.UsageError(
            "--backend=local cannot be used with --cluster (cluster submission not supported in local mode)"
        )


def run_local(
    config: CliConfig,
    script_path: str,
    script_args: list[str],
    entry_point: str,
) -> None:
    """Run script locally with configured backend.

    Args:
        config: Backend configuration
        script_path: Path to user's Python script
        script_args: Arguments to pass to user's script
        entry_point: Name of entry point function (default: "main")

    Raises:
        SystemExit: If script execution fails
    """
    resources = ResourceConfig()
    if config.memory:
        resources = dataclasses.replace(resources, ram=config.memory)
    if config.num_cpus is not None:
        resources = dataclasses.replace(resources, cpu=int(config.num_cpus))
    # Note: num_gpus would require DeviceConfig, skipping for now

    client = current_client()
    logger.info("Zephyr using fray client: %s", type(client).__name__)

    # For distributed backends (iris, ray), chunk storage must be shared (GCS).
    # Local /tmp won't work when coordinator and workers run in separate containers.
    chunk_storage_prefix = None
    is_distributed = not isinstance(client, LocalClient)
    if is_distributed:
        marin_prefix = os.environ.get("MARIN_PREFIX", "")
        if not marin_prefix:
            marin_prefix = "gs://marin-us-central2/scratch"
            logger.warning(
                "MARIN_PREFIX not set for distributed backend; using default %s",
                marin_prefix,
            )
        chunk_storage_prefix = f"{marin_prefix}/tmp/zephyr"

    ctx = ZephyrContext(
        client=client,
        num_workers=config.max_parallelism,
        resources=resources,
        chunk_storage_prefix=chunk_storage_prefix,
    )
    _default_zephyr_context.set(ctx)
    sys.argv = [script_path, *script_args]

    script_path_obj = Path(script_path).resolve()

    # Derive module name from path for Ray compatibility
    # e.g., lib/marin/src/marin/foo/bar.py -> marin.foo.bar
    module_name = None
    try:
        relative_path = script_path_obj.relative_to(Path.cwd())
        # Try to find the module name by looking for src/ in path
        parts = relative_path.parts
        if "src" in parts or "experiments" in parts:
            src_idx = parts.index("src") if "src" in parts else parts.index("experiments")
            module_parts = parts[src_idx + 1 :]
            if module_parts and module_parts[-1].endswith(".py"):
                module_parts = [*list(module_parts[:-1]), module_parts[-1][:-3]]
                module_name = ".".join(module_parts)
    except (ValueError, IndexError):
        pass

    if not module_name:
        # Fallback to file name without extension
        module_name = script_path_obj.stem
        logging.warning(
            f"Could not derive proper module name for {script_path}, using fallback: {module_name}. "
            "Functions may not pickle correctly for Ray workers."
        )

    spec = importlib.util.spec_from_file_location(module_name, script_path_obj)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except SystemExit:
        raise
    except Exception:
        raise

    if not hasattr(module, entry_point):
        raise AttributeError(
            f"Script {script_path} does not have entry point '{entry_point}'. "
            f"Available: {[name for name in dir(module) if not name.startswith('_')]}"
        )

    main_fn = getattr(module, entry_point)
    main_fn()


def run_ray_cluster(
    config: CliConfig,
    cluster: str,
    script_path: str,
    script_args: list[str],
    entry_point: str,
) -> None:
    """Submit script to Ray cluster via ray_run.py.

    Args:
        config: Backend configuration
        cluster: Cluster name or config file path
        script_path: Path to user's Python script
        script_args: Arguments to pass to user's script
        entry_point: Name of entry point function

    Raises:
        SystemExit: With ray_run.py's exit code
    """
    ray_cmd = ["uv", "run", "python", "-m", "marin.run.ray_run", "--cluster", cluster, "--auto-stop"]

    # Add resource specs as entrypoint-* args
    if config.memory:
        import humanfriendly

        memory_bytes = humanfriendly.parse_size(config.memory, binary=True)
        ray_cmd += ["--entrypoint-memory", str(memory_bytes)]
    if config.num_cpus:
        ray_cmd += ["--entrypoint-num-cpus", str(config.num_cpus)]
    if config.num_gpus:
        ray_cmd += ["--entrypoint-num-gpus", str(config.num_gpus)]

    entrypoint = [
        "python",
        "-m",
        "zephyr.cli",
        "--max-parallelism",
        "--backend=local",
        str(config.max_parallelism),
    ]

    if config.memory:
        entrypoint += ["--memory", config.memory]
    if config.num_cpus:
        entrypoint += ["--num-cpus", str(config.num_cpus)]
    if config.num_gpus:
        entrypoint += ["--num-gpus", str(config.num_gpus)]
    if entry_point != "main":
        entrypoint += ["--entry-point", entry_point]

    entrypoint += [script_path, *script_args]

    ray_cmd += ["--", *entrypoint]

    # Run ray_run.py, forward exit code
    result = subprocess.run(ray_cmd)
    sys.exit(result.returncode)


def run_iris_cluster(
    config: CliConfig,
    cluster_config_path: str,
    script_path: str,
    script_args: list[str],
    entry_point: str,
) -> None:
    """Submit script to Iris cluster.

    Args:
        config: Backend configuration
        cluster_config_path: Path to Iris cluster config YAML
        script_path: Path to user's Python script
        script_args: Arguments to pass to user's script
        entry_point: Name of entry point function

    Raises:
        SystemExit: With iris_run.py's exit code
    """
    iris_cmd = [
        "uv",
        "run",
        "iris",
        "--config",
        cluster_config_path,
        "run",
    ]

    # Install CPU-only PyTorch on Iris workers (no GPU/TPU accelerators available for CPU tasks)
    # "marin:cpu" tells the builder to pass --package marin --extra cpu to uv sync
    iris_cmd += ["--extra", "marin:cpu"]

    # Add resource specs
    # Note: iris_run uses --cpu and --gpu (not --num-cpus/--num-gpus)
    if config.memory:
        iris_cmd += ["--memory", config.memory]
    if config.num_cpus:
        iris_cmd += ["--cpu", str(int(config.num_cpus))]
    if config.num_gpus:
        iris_cmd += ["--gpu", str(int(config.num_gpus))]

    # Build entrypoint command to run Zephyr CLI on the cluster
    entrypoint = [
        "python",
        "-m",
        "zephyr.cli",
        "--backend=local",  # Inside cluster, use the default context from Iris
        "--max-parallelism",
        str(config.max_parallelism),
    ]

    if config.memory:
        entrypoint += ["--memory", config.memory]
    if config.num_cpus:
        entrypoint += ["--num-cpus", str(config.num_cpus)]
    if config.num_gpus:
        entrypoint += ["--num-gpus", str(config.num_gpus)]
    if entry_point != "main":
        entrypoint += ["--entry-point", entry_point]

    entrypoint += [script_path, *script_args]

    # Append command after --
    iris_cmd += ["--", *entrypoint]

    # Run iris_run.py, forward exit code
    result = subprocess.run(iris_cmd)
    sys.exit(result.returncode)


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    help="zephyr launcher: Execute data processing pipelines with configurable backends",
    epilog="""
Examples:

  # Run locally
  zephyr --max-parallelism=100 script.py --input=data.jsonl

  # Submit to Ray cluster (backward compatible - backend defaults to ray)
  zephyr --cluster=us-central2 --memory=2GB script.py --input=data.jsonl

  # Submit to Iris cluster
  zephyr --backend=iris --cluster=lib/iris/examples/eu-west4.yaml --memory=2GB script.py

  # Dry-run to show optimization plan
  zephyr --dry-run script.py --input=data.jsonl
""",
)
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--backend",
    type=click.Choice(["local", "ray", "iris"]),
    default="ray",
    help="Backend to use: local (in-process), ray (Ray cluster), iris (Iris cluster). Default: ray",
)
@click.option("--max-parallelism", type=int, default=100, help="Maximum concurrent tasks (default: 100)")
@click.option("--memory", type=str, help="Memory per task (e.g., '2GB', '512MB')")
@click.option("--num-cpus", type=float, help="Number of CPUs per task")
@click.option("--num-gpus", type=float, help="Number of GPUs per task")
@click.option(
    "--cluster",
    type=str,
    help="For Ray: cluster name/region (e.g., 'us-central2'). For Iris: path to cluster config YAML",
)
@click.option("--entry-point", type=str, default="main", help="Entry point function name (default: 'main')")
@click.option("--dry-run", is_flag=True, help="Show optimization plan without executing")
@click.pass_context
def main(
    ctx: click.Context,
    script: str,
    backend: str,
    max_parallelism: int,
    memory: str | None,
    num_cpus: float | None,
    num_gpus: float | None,
    cluster: str | None,
    entry_point: str,
    dry_run: bool,
) -> None:
    """Execute data processing pipeline script with configurable backend."""
    script_args = ctx.args
    script_path = Path(script).resolve()

    # Build config
    config = CliConfig(
        backend=backend,
        max_parallelism=max_parallelism,
        dry_run=dry_run,
        memory=memory,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        cluster=cluster,
        entry_point=entry_point,
    )

    # Validate backend configuration
    validate_backend_config(config)

    # Determine execution mode based on backend and cluster flag
    if config.cluster:
        relative_script_path = script_path.relative_to(Path.cwd())
        if config.backend == "iris":
            # Iris cluster submission mode
            run_iris_cluster(config, config.cluster, str(relative_script_path), script_args, entry_point)
        else:
            # Ray cluster submission mode (default)
            run_ray_cluster(config, config.cluster, str(relative_script_path), script_args, entry_point)
    else:
        # Local mode - runs script locally with LocalClient
        run_local(config, str(script_path), script_args, entry_point)


if __name__ == "__main__":
    main()
