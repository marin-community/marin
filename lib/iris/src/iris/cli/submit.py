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

"""Job submission CLI commands.

Contains both the ``iris submit`` command (Python-callable submission) and the
``iris-run`` entry point (command-passthrough submission), sharing core logic.
"""

import importlib.util
import sys
import traceback
from pathlib import Path

import click

from iris.client import IrisClient
from iris.cluster.types import Entrypoint
from iris.cluster.vm.config import load_config
from iris.cluster.vm.debug import controller_tunnel
from iris.rpc import cluster_pb2


@click.command("submit")
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--controller-url", help="Direct controller URL (e.g., http://localhost:10000)")
@click.option("--config", "config_file", type=click.Path(exists=True), help="Cluster config file for auto-tunneling")
@click.option("--name", help="Job name (defaults to script filename)")
@click.option("--timeout", default=600, type=int, help="Job timeout in seconds")
@click.option("--workspace", type=click.Path(exists=True), help="Workspace directory (defaults to script parent)")
@click.argument("script_args", nargs=-1)
@click.pass_context
def submit(
    ctx,
    script_path: str,
    controller_url: str | None,
    config_file: str | None,
    name: str | None,
    timeout: int,
    workspace: str | None,
    script_args: tuple[str, ...],
):
    """Submit a Python script as a job to the Iris cluster.

    The script must define a main() function that will be executed as the job entrypoint.

    Examples:
        iris submit script.py --controller-url http://localhost:10000
        iris submit script.py --config examples/eu-west4.yaml
    """
    if controller_url and config_file:
        click.echo("Error: --controller-url and --config are mutually exclusive", err=True)
        raise SystemExit(1)
    if not controller_url and not config_file:
        click.echo("Error: Either --controller-url or --config is required", err=True)
        raise SystemExit(1)

    script_path_obj = Path(script_path).resolve()
    script_name = script_path_obj.stem

    spec = importlib.util.spec_from_file_location(script_name, script_path_obj)
    if not spec or not spec.loader:
        click.echo(f"Error: Failed to load script: {script_path}", err=True)
        raise SystemExit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        click.echo(f"Error loading script: {e}", err=True)
        if ctx.obj and ctx.obj.get("traceback"):
            traceback.print_exc()
        raise SystemExit(1)  # noqa: B904

    if not hasattr(module, "main"):
        click.echo(f"Error: Script {script_path} must define a main() function", err=True)
        raise SystemExit(1)

    main_func = module.main

    if workspace:
        workspace_path = Path(workspace).resolve()
    else:
        workspace_path = script_path_obj.parent

    job_name = name or script_name
    args_list = list(script_args)

    def _submit_and_wait(url: str):
        click.echo(f"Connecting to controller at {url}")
        click.echo(f"Workspace: {workspace_path}")
        click.echo(f"Job name: {job_name}")
        if args_list:
            click.echo(f"Arguments: {args_list}")
        click.echo()

        try:
            client = IrisClient.remote(url, workspace=workspace_path)
        except Exception as e:
            click.echo(f"Error connecting to controller: {e}", err=True)
            if ctx.obj and ctx.obj.get("traceback"):
                traceback.print_exc()
            raise SystemExit(1)  # noqa: B904

        try:
            entrypoint = Entrypoint.from_callable(main_func, *args_list)
        except Exception as e:
            click.echo(f"Error creating entrypoint: {e}", err=True)
            if ctx.obj and ctx.obj.get("traceback"):
                traceback.print_exc()
            raise SystemExit(1)  # noqa: B904

        click.echo("Submitting job...")
        try:
            job = client.submit(entrypoint=entrypoint, name=job_name)
            click.echo(f"Job submitted: {job.job_id}")
            click.echo()
        except Exception as e:
            click.echo(f"Error submitting job: {e}", err=True)
            if ctx.obj and ctx.obj.get("traceback"):
                traceback.print_exc()
            raise SystemExit(1)  # noqa: B904

        click.echo("Streaming logs...")
        click.echo("-" * 80)
        try:
            status = job.wait(timeout=timeout, stream_logs=True, raise_on_failure=False)
        except TimeoutError:
            click.echo(f"\nError: Job timed out after {timeout} seconds", err=True)
            raise SystemExit(1)  # noqa: B904
        except Exception as e:
            click.echo(f"\nError waiting for job: {e}", err=True)
            if ctx.obj and ctx.obj.get("traceback"):
                traceback.print_exc()
            raise SystemExit(1)  # noqa: B904

        click.echo("-" * 80)
        state_name = cluster_pb2.JobState.Name(status.state)
        click.echo(f"Job {job.job_id}: {state_name}")

        if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
            click.echo("Job completed successfully")
        else:
            click.echo(f"Job failed: {status.error}", err=True)
            raise SystemExit(1)

    if controller_url:
        _submit_and_wait(controller_url)
    else:
        assert config_file is not None
        config = load_config(Path(config_file))

        zone = config.zone or "us-central1-a"
        project = config.project_id or ""

        if not project:
            click.echo("Error: Config file must specify project_id", err=True)
            raise SystemExit(1)

        click.echo(f"Establishing SSH tunnel to controller in {zone}...")
        try:
            with controller_tunnel(zone, project) as url:
                _submit_and_wait(url)
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)  # noqa: B904
