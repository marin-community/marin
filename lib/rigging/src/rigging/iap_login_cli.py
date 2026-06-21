# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``marin-login`` — one IAP desktop-OAuth login, shared across Marin client tools.

Runs the browser consent flow (or a headless URL-paste flow when no browser is
available) for an IAP-fronted cluster and caches the refresh token so any tool —
``finelog query``, the iris CLI, a one-off script — can silently re-mint the
OIDC ID token IAP requires.
"""

import webbrowser

import click

from rigging.iap_login import credentials_path, desktop_login


@click.command("marin-login")
@click.argument("name")
@click.option(
    "--client-secrets",
    "client_secrets",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the Google desktop OAuth client secret JSON.",
)
def main(name: str, client_secrets: str) -> None:
    """Authenticate to an IAP-fronted Marin cluster and cache the refresh token.

    NAME is the logical cluster/endpoint that tools look up (e.g. 'marin').
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

    desktop_login(name, client_secrets, headless=headless)
    click.echo(f"Cached IAP credentials for '{name}' at {credentials_path(name)}")


if __name__ == "__main__":
    main()
