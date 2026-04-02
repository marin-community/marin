# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI for calling actor methods through the controller proxy."""

import json

import click

from iris.actor import ActorClient, ProxyResolver
from iris.actor.resolver import ResolveResult
from iris.cli.main import require_controller_url
from iris.rpc.auth import GcpAccessTokenProvider, TokenProvider


class _AuthProxyResolver(ProxyResolver):
    """ProxyResolver that injects a bearer token for dashboard auth."""

    def __init__(self, controller_url: str, token_provider: TokenProvider):
        super().__init__(controller_url)
        self._token_provider = token_provider

    def resolve(self, name: str) -> ResolveResult:
        result = super().resolve(name)
        token = self._token_provider.get_token()
        if token:
            for ep in result.endpoints:
                ep.metadata["authorization"] = f"Bearer {token}"
        return result


@click.group()
def actor():
    """Interact with actors via the controller proxy."""
    pass


@actor.command()
@click.argument("endpoint")
@click.argument("method")
@click.argument("kwargs", required=False, default=None)
@click.option("--timeout", type=float, default=30.0, help="RPC timeout in seconds")
@click.pass_context
def call(ctx: click.Context, endpoint: str, method: str, kwargs: str | None, timeout: float):
    """Call an actor method through the controller proxy.

    ENDPOINT is the full actor name as registered in the endpoint registry
    (e.g. /user/job/coordinator/actor-0).

    METHOD is the method name to call (e.g. get_counters).

    KWARGS is an optional JSON object of keyword arguments (e.g. '{"worker_id": "w-3"}').
    """
    controller_url = require_controller_url(ctx)
    tp = ctx.obj.get("token_provider") if ctx.obj else None
    if tp is None:
        tp = GcpAccessTokenProvider()

    resolver = _AuthProxyResolver(controller_url, tp)
    client = ActorClient(resolver, endpoint, call_timeout=timeout)

    parsed_kwargs = json.loads(kwargs) if kwargs else {}
    rpc_method = getattr(client, method)
    result = rpc_method(**parsed_kwargs)

    try:
        click.echo(json.dumps(result, indent=2, default=str))
    except (TypeError, ValueError):
        click.echo(repr(result))
