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

"""Command-line interface for ml-flow launcher."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import click

from zephyr import set_flow_backend
from zephyr.backend_factory import BackendConfig, create_backend, parse_memory


def run_local(
    config: BackendConfig,
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
    backend = create_backend(
        backend_type=config.backend_type,
        max_parallelism=config.max_parallelism,
        memory=config.memory,
        num_cpus=config.num_cpus,
        num_gpus=config.num_gpus,
        dry_run=config.dry_run,
        **config.ray_options,
    )
    set_flow_backend(backend)
    sys.argv = [script_path, *script_args]

    # Import script dynamically
    script_path_obj = Path(script_path).resolve()

    # Derive module name from path for Ray compatibility
    # e.g., lib/marin/src/marin/foo/bar.py -> marin.foo.bar
    module_name = None
    try:
        relative_path = script_path_obj.relative_to(Path.cwd())
        # Try to find the module name by looking for src/ in path
        parts = relative_path.parts
        if "src" in parts:
            src_idx = parts.index("src")
            module_parts = parts[src_idx + 1 :]
            if module_parts and module_parts[-1].endswith(".py"):
                module_parts = [*list(module_parts[:-1]), module_parts[-1][:-3]]
                module_name = ".".join(module_parts)
    except (ValueError, IndexError):
        pass

    if not module_name:
        # Fallback to file name without extension
        module_name = script_path_obj.stem
        import logging

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


def run_cluster(
    config: BackendConfig,
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
        memory_bytes = parse_memory(config.memory)
        ray_cmd += ["--entrypoint-memory", str(memory_bytes)]
    if config.num_cpus:
        ray_cmd += ["--entrypoint-num-cpus", str(config.num_cpus)]
    if config.num_gpus:
        ray_cmd += ["--entrypoint-num-gpus", str(config.num_gpus)]

    # Construct the entrypoint command (recursive call without --cluster)
    entrypoint = [
        "uv",
        "run",
        "mlflow",
        "--backend",
        config.backend_type,
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

    # Pass entrypoint command to ray_run (must be prefixed with --)
    ray_cmd += ["--", *entrypoint]

    # Run ray_run.py, forward exit code
    result = subprocess.run(ray_cmd)
    sys.exit(result.returncode)


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    help="ml-flow launcher: Execute data processing pipelines with configurable backends",
    epilog="""
Examples:

  # Run locally with sync backend

  mlflow --backend=sync script.py --input=data.jsonl

  # Run locally with Ray backend

  mlflow --backend=ray --max-parallelism=100 --memory=2GB script.py --input=data.jsonl

  # Submit to Ray cluster

  mlflow --backend=ray --cluster=us-central2 --memory=2GB script.py --input=data.jsonl

  # Dry-run to show optimization plan

  mlflow --backend=ray --dry-run script.py --input=data.jsonl
""",
)
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.option("--backend", type=click.Choice(["ray", "threadpool", "sync"]), default="threadpool", help="Backend type")
@click.option("--max-parallelism", type=int, default=100, help="Maximum concurrent tasks (default: 100)")
@click.option("--memory", type=str, help="Memory per task (e.g., '2GB', '512MB')")
@click.option("--num-cpus", type=float, help="Number of CPUs per task")
@click.option("--num-gpus", type=float, help="Number of GPUs per task")
@click.option("--cluster", type=str, help="Cluster name or config file for Ray submission (enables cluster mode)")
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
    # Get script arguments from extra args
    script_args = ctx.args

    # Resolve script path
    script_path = Path(script).resolve()

    # Build backend config
    config = BackendConfig(
        backend_type=backend,
        max_parallelism=max_parallelism,
        memory=memory,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        dry_run=dry_run,
    )

    # Execute based on mode
    if cluster:
        # Cluster mode: submit via ray_run.py
        # Convert to relative path for cluster execution
        try:
            relative_script_path = script_path.relative_to(Path.cwd())
            run_cluster(config, cluster, str(relative_script_path), script_args, entry_point)
        except ValueError:
            # Script is outside cwd, use absolute path
            run_cluster(config, cluster, str(script_path), script_args, entry_point)
    else:
        # Local mode: run directly
        run_local(config, str(script_path), script_args, entry_point)


if __name__ == "__main__":
    main()
