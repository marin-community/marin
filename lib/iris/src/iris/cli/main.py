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

import logging

import click

from iris.logging import configure_logging as _configure_logging


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--traceback", "show_traceback", is_flag=True, help="Show full stack traces on errors")
@click.pass_context
def iris(ctx, verbose: bool, show_traceback: bool):
    """Iris cluster management."""
    ctx.ensure_object(dict)
    ctx.obj["traceback"] = show_traceback

    if verbose:
        _configure_logging(level=logging.DEBUG)
    else:
        _configure_logging(level=logging.INFO)


# Register subcommand groups — imported at module level to ensure they are
# always available when the ``iris`` group is used.
from iris.cli.build import build  # noqa: E402
from iris.cli.cluster import cluster  # noqa: E402
from iris.cli.rpc import register_rpc_commands  # noqa: E402
from iris.cli.run import run  # noqa: E402
from iris.cli.submit import submit  # noqa: E402

iris.add_command(cluster)
iris.add_command(build)
iris.add_command(run)
iris.add_command(submit)
register_rpc_commands(iris)
