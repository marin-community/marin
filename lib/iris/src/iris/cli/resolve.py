# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``iris resolve`` — resolve a service URL to ``host:port``.

Useful for verifying scheme handlers (``iris://``, ``gcp://``) and the
off-cluster :func:`rigging.proxy.proxy_stack` flow end-to-end.

Examples::

    iris resolve gcp://log-server
    iris resolve iris://marin?endpoint=/system/log-server --proxy
"""

import logging

import click

import iris.client  # noqa: F401  -- registers iris://
from rigging.proxy import proxy_stack
from rigging.resolver import resolve as resolve_url


@click.command("resolve")
@click.argument("url")
@click.option(
    "--proxy/--no-proxy",
    default=False,
    help="Wrap resolution in proxy_stack so unreachable internal addresses are tunneled.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Log INFO-level diagnostics from the resolver and tunnel layer.",
)
def resolve_cmd(url: str, proxy: bool, verbose: bool) -> None:
    """Resolve URL to host:port. With --proxy, opens tunnels for unreachable internal addresses."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if proxy:
        with proxy_stack():
            host, port = resolve_url(url)
    else:
        host, port = resolve_url(url)
    click.echo(f"{host}:{port}")
