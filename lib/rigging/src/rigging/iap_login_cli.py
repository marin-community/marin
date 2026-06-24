# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manage IAP desktop-OAuth credentials shared across Marin client tools.

The login command runs the browser consent flow (or a headless URL-paste flow
when no browser is available) and caches the refresh token. The print-token
command silently mints an OIDC ID token for command-line HTTP clients.
"""

import webbrowser

import click

from rigging.auth import IapLoginRequired
from rigging.iap_login import (
    MARIN_DESKTOP_OAUTH_CLIENT,
    credentials_path,
    desktop_login,
    provider_for,
    read_desktop_client,
)


@click.group("marin-login")
def main() -> None:
    """Manage cached credentials for IAP-fronted Marin endpoints."""


@main.command("login")
@click.argument("name")
@click.option(
    "--client-secrets",
    "client_secrets",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Override the built-in Marin desktop OAuth client with a downloaded client-secret JSON.",
)
def login(name: str, client_secrets: str | None) -> None:
    """Authenticate to an IAP-fronted Marin cluster and cache the refresh token.

    NAME is the logical cluster/endpoint that tools look up (e.g. 'marin'). Uses
    the built-in Marin desktop OAuth client unless --client-secrets overrides it.
    Opens a browser if one is available; otherwise prints a URL to paste back
    (works over SSH).
    """
    # Open a browser when one is reachable; otherwise fall back to the URL-paste
    # flow so the login works over SSH / on a headless box.
    try:
        webbrowser.get()
        headless = False
    except webbrowser.Error:
        headless = True

    client = read_desktop_client(client_secrets) if client_secrets else MARIN_DESKTOP_OAUTH_CLIENT
    desktop_login(name, client, headless=headless)
    click.echo(f"Cached IAP credentials for '{name}' at {credentials_path(name)}")


@main.command("print-token")
@click.argument("name")
def print_token(name: str) -> None:
    """Print an IAP ID token minted from cached credentials for NAME."""
    try:
        token = provider_for(name).get_token()
    except IapLoginRequired as exc:
        raise click.ClickException(str(exc)) from exc
    if token is None:
        raise click.ClickException(f"IAP token provider returned no token for {name!r}")
    click.echo(token)


if __name__ == "__main__":
    main()
