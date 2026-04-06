# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task operations CLI.

Usage:
    iris --config cluster.yaml task exec /user/job/0 -- bash -c "ls /app"
"""

import logging
import sys

import click

from iris.cli.main import require_controller_url, rpc_client
from iris.rpc import controller_pb2
from iris.rpc.auth import TokenProvider

logger = logging.getLogger(__name__)


@click.group()
def task():
    """Task operations."""
    pass


@task.command("exec")
@click.argument("task_id")
@click.argument("command", nargs=-1, required=True)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=int,
    default=60,
    help="Command timeout in seconds (default: 60, -1 for no timeout)",
)
@click.pass_context
def task_exec(ctx, task_id: str, command: tuple[str, ...], timeout_seconds: int):
    """Execute a command in a running task's container.

    Works across platforms: docker exec on Docker, kubectl exec on K8s.

    Examples:

      iris task exec /user/job/0 -- bash -c "ls /app"

      iris task exec /user/job/0 --timeout 300 -- cat /proc/1/status
    """
    controller_url = require_controller_url(ctx)
    token_provider: TokenProvider | None = ctx.obj.get("token_provider")

    with rpc_client(controller_url, token_provider) as client:
        request = controller_pb2.Controller.ExecInContainerRequest(
            task_id=task_id,
            command=list(command),
            timeout_seconds=timeout_seconds,
        )
        response = client.exec_in_container(request)

    if response.error:
        click.echo(f"Error: {response.error}", err=True)
        sys.exit(1)

    if response.stdout:
        click.echo(response.stdout, nl=False)
    if response.stderr:
        click.echo(response.stderr, nl=False, err=True)

    sys.exit(response.exit_code)
