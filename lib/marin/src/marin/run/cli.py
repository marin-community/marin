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

"""Command-line interface for marin cluster job management."""

import subprocess
import sys

import click


@click.group()
def main():
    """Marin cluster job management."""
    pass


@main.command()
@click.argument("script")
@click.option("--cluster", required=True, help="Cluster name or config path")
@click.option("--extra", default="", help="Dependency extras (e.g., 'cpu,tpu')")
@click.option("--infra-dir", help="Path to infra/ directory with cluster configs")
@click.option("-e", "--env", "env_vars", multiple=True, help="Environment variable KEY=VALUE")
@click.option("--no-wait", is_flag=True, help="Don't wait for job completion")
@click.option("--auto-stop", is_flag=True, help="Stop job on exit")
@click.option("--tpu", help="TPU type to reserve (e.g., 'v4-8')")
@click.option("--entrypoint-num-cpus", type=float, help="Number of CPUs to reserve for entrypoint")
@click.option("--entrypoint-num-gpus", type=float, help="Number of GPUs to reserve for entrypoint")
@click.option("--entrypoint-memory", type=int, help="Memory in bytes to reserve for entrypoint")
@click.option("--submission-id", help="Custom submission ID for the job")
@click.argument("script_args", nargs=-1)
def submit(
    script,
    cluster,
    extra,
    infra_dir,
    env_vars,
    no_wait,
    auto_stop,
    tpu,
    entrypoint_num_cpus,
    entrypoint_num_gpus,
    entrypoint_memory,
    submission_id,
    script_args,
):
    """Submit a script to Ray cluster.

    Submit SCRIPT to a Ray cluster with optional dependencies and environment variables.

    Example:

        marin submit speedrun.py --cluster=us-central2 --extra=cpu -- --epochs=10
    """
    ray_cmd = ["uv", "run", "python", "-m", "marin.run.ray_run"]

    ray_cmd.extend(["--cluster", cluster])

    if extra:
        ray_cmd.extend(["--extra", extra])

    if infra_dir:
        ray_cmd.extend(["--infra-dir", infra_dir])

    if no_wait:
        ray_cmd.append("--no_wait")

    if auto_stop:
        ray_cmd.append("--auto-stop")

    if tpu:
        ray_cmd.extend(["--tpu", tpu])

    if entrypoint_num_cpus is not None:
        ray_cmd.extend(["--entrypoint-num-cpus", str(entrypoint_num_cpus)])

    if entrypoint_num_gpus is not None:
        ray_cmd.extend(["--entrypoint-num-gpus", str(entrypoint_num_gpus)])

    if entrypoint_memory is not None:
        ray_cmd.extend(["--entrypoint-memory", str(entrypoint_memory)])

    if submission_id:
        ray_cmd.extend(["--submission-id", submission_id])

    for env_var in env_vars:
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            ray_cmd.extend(["-e", key, value])
        else:
            ray_cmd.extend(["-e", env_var])

    ray_cmd.append("--")
    ray_cmd.append("python")
    ray_cmd.append(script)
    ray_cmd.extend(script_args)

    result = subprocess.run(ray_cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
