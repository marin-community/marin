# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``marin-login`` — one IAP desktop-OAuth login, shared across Marin client tools.

Runs the browser consent flow (or a headless URL-paste flow when no browser is
available) for an IAP-fronted cluster and caches the refresh token so any tool —
``finelog query``, the iris CLI, a one-off script — can silently re-mint the
OIDC ID token IAP requires.
"""

import argparse
import sys
import webbrowser

from rigging.iap_login import credentials_path, desktop_login


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="marin-login",
        description="Authenticate to an IAP-fronted Marin cluster and cache the refresh token.",
    )
    parser.add_argument("name", help="Logical cluster/endpoint name (e.g. 'marin') that tools look up.")
    parser.add_argument(
        "--client-secrets",
        required=True,
        help="Path to the Google desktop OAuth client secret JSON.",
    )
    args = parser.parse_args(argv)

    # Open a browser when one is reachable; otherwise fall back to the URL-paste
    # flow so the login works over SSH / on a headless box.
    try:
        webbrowser.get()
        headless = False
    except webbrowser.Error:
        headless = True

    desktop_login(args.name, args.client_secrets, headless=headless)
    print(f"Cached IAP credentials for '{args.name}' at {credentials_path(args.name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
