# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level Iris CLI entry point.

Defines the ``iris`` Click group and registers all subcommands.
"""

import logging as _logging_module
import sys

import click

from iris.logging import configure_logging

logger = _logging_module.getLogger(__name__)


def require_controller_url(ctx: click.Context) -> str:
    """Get controller_url from context, establishing a tunnel lazily if needed.

    On first call with a --config, this establishes the tunnel to the controller
    and caches the result. Subsequent calls return the cached URL.
    Commands that don't call this (e.g. ``cluster start``) never pay tunnel cost.
    """
    controller_url = ctx.obj.get("controller_url") if ctx.obj else None
    if controller_url:
        return controller_url

    # Lazy tunnel establishment from config
    config = ctx.obj.get("config") if ctx.obj else None
    if config:
        from iris.cluster.config import IrisConfig

        iris_config = IrisConfig(config)
        platform = iris_config.platform()

        if iris_config.proto.controller.WhichOneof("controller") == "local":
            from iris.cluster.controller.local import LocalController

            controller = LocalController(iris_config.proto)
            controller_address = controller.start()
            ctx.call_on_close(controller.stop)
        else:
            controller_address = iris_config.controller_address()

        # Establish tunnel and keep it alive for command duration
        logger.info("Establishing tunnel to controller...")
        tunnel_cm = platform.tunnel(address=controller_address)
        tunnel_url = tunnel_cm.__enter__()
        ctx.obj["controller_url"] = tunnel_url
        # Clean up tunnel when context closes
        ctx.call_on_close(lambda: tunnel_cm.__exit__(None, None, None))
        return tunnel_url

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
        configure_logging(level=_logging_module.DEBUG)
    else:
        configure_logging(level=_logging_module.INFO)

    # Validate mutually exclusive options
    if controller_url and config_file:
        raise click.UsageError("Cannot specify both --controller-url and --config")

    # Skip expensive operations when showing help or doing shell completion.
    # Only check for help flags before "--" to avoid matching help flags
    # intended for the user's command (e.g., "job run -- python script.py --help").
    argv_before_separator = sys.argv[: sys.argv.index("--")] if "--" in sys.argv else sys.argv
    if ctx.resilient_parsing or "--help" in argv_before_separator or "-h" in argv_before_separator:
        return

    # Load config if provided
    if config_file:
        from iris.cluster.config import IrisConfig

        iris_config = IrisConfig.load(config_file)
        ctx.obj["config"] = iris_config.proto
        ctx.obj["config_file"] = config_file

    # Store direct controller URL; tunnel from config is established lazily
    # in require_controller_url() so commands like ``cluster start`` don't block.
    if controller_url:
        ctx.obj["controller_url"] = controller_url


# Register subcommand groups — imported at module level to ensure they are
# always available when the ``iris`` group is used.
from iris.cli.build import build  # noqa: E402
from iris.cli.cluster import cluster  # noqa: E402
from iris.cli.job import job  # noqa: E402
from iris.cli.rpc import register_rpc_commands  # noqa: E402

iris.add_command(cluster)
iris.add_command(build)
iris.add_command(job)
register_rpc_commands(iris)
