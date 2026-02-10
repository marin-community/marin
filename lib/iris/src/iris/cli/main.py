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

"""Top-level Iris CLI entry point.

Defines the ``iris`` Click group and registers all subcommands.
"""

import logging as _logging_module
import sys

import click

from iris.logging import configure_logging as _configure_logging

logger = _logging_module.getLogger(__name__)


def require_controller_url(ctx: click.Context) -> str:
    """Get controller_url from context or raise a descriptive error.

    Use this in commands that require a connection to the controller.
    Provides clear error messages based on whether --config was provided.
    """
    controller_url = ctx.obj.get("controller_url") if ctx.obj else None
    if controller_url:
        return controller_url

    config_file = ctx.obj.get("config_file") if ctx.obj else None
    if config_file:
        raise click.ClickException(
            f"Could not connect to controller (config: {config_file}). "
            "Check that the controller is running and reachable."
        )
    raise click.ClickException("Either --controller-url or --config is required")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--traceback", "show_traceback", is_flag=True, help="Show full stack traces on errors")
@click.option("--controller-url", help="Controller URL (e.g., http://localhost:10000)")
@click.option("--config", "config_file", type=click.Path(exists=True), help="Cluster config file")
@click.pass_context
def iris(ctx, verbose: bool, show_traceback: bool, controller_url: str | None, config_file: str | None):
    """Iris cluster management."""
    ctx.ensure_object(dict)
    ctx.obj["traceback"] = show_traceback

    if verbose:
        _configure_logging(level=_logging_module.DEBUG)
    else:
        _configure_logging(level=_logging_module.INFO)

    # Validate mutually exclusive options
    if controller_url and config_file:
        raise click.UsageError("Cannot specify both --controller-url and --config")

    # Skip expensive operations when showing help or doing shell completion
    if ctx.resilient_parsing or "--help" in sys.argv or "-h" in sys.argv:
        return

    # Load config if provided
    if config_file:
        from iris.cluster.vm.config import IrisConfig

        iris_config = IrisConfig.load(config_file)
        ctx.obj["config"] = iris_config.proto
        ctx.obj["config_file"] = config_file

    # Establish controller URL (either direct or via tunnel)
    if controller_url:
        ctx.obj["controller_url"] = controller_url
    elif config_file:
        # Establish controller URL from config using Platform abstraction.
        iris_config = IrisConfig(ctx.obj["config"])
        platform = iris_config.platform()

        if iris_config.proto.controller.WhichOneof("controller") == "local":
            from iris.cluster.vm.cluster_manager import ClusterManager

            manager = ClusterManager(iris_config.proto)
            controller_address = manager.start()
            ctx.call_on_close(manager.stop)
        else:
            controller_address = iris_config.controller_address()

        # Establish tunnel with 5-second timeout and keep it alive for command duration
        try:
            logger.info("Establishing tunnel to controller...")
            tunnel_cm = platform.tunnel(
                controller_address=controller_address,
                timeout=5.0,
                tunnel_logger=logger,
            )
            tunnel_url = tunnel_cm.__enter__()
            ctx.obj["controller_url"] = tunnel_url
            # Clean up tunnel when context closes
            ctx.call_on_close(lambda: tunnel_cm.__exit__(None, None, None))
        except Exception as e:
            # If tunnel fails (e.g., no controller VM), continue without controller_url
            # Commands that need it will fail with clear error
            logger.warning("Could not establish controller tunnel (timeout=5s): %s", e)


# Register subcommand groups — imported at module level to ensure they are
# always available when the ``iris`` group is used.
from iris.cli.build import build  # noqa: E402
from iris.cli.cluster import cluster  # noqa: E402
from iris.cli.rpc import register_rpc_commands  # noqa: E402
from iris.cli.job import job  # noqa: E402

iris.add_command(cluster)
iris.add_command(build)
iris.add_command(job)
register_rpc_commands(iris)
