# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent IAP desktop-OAuth credentials, shared across Marin client tools.

A human authenticates to an IAP-fronted Marin endpoint (the Iris controller and
anything it proxies, e.g. the finelog log server) once in the browser; this
module caches the resulting desktop-OAuth refresh token under
``~/.config/marin/iap/<name>.json`` so any tool — finelog, a one-off script —
can silently re-mint the OIDC ID token IAP requires without another browser
round-trip. The short-lived ID token is never stored, only the long-lived
refresh token plus the desktop client id/secret needed to use it (the desktop
secret is non-confidential per RFC 8252 §8.5).

``<name>`` is the logical endpoint key the user passes to both login and use
(typically the cluster name, e.g. ``marin``), so one login serves every tool
that talks to that cluster's IAP edge.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rigging.auth import IapLoginRequired, IapRefreshTokenProvider, run_iap_desktop_login


@dataclass(frozen=True)
class IapCredentials:
    """Cached desktop-OAuth material for one IAP-fronted endpoint."""

    client_id: str
    client_secret: str
    refresh_token: str


def credentials_path(name: str) -> Path:
    """Path to the cached credentials for the logical endpoint ``name``."""
    return Path.home() / ".config" / "marin" / "iap" / f"{name}.json"


def load_iap_credentials(name: str) -> IapCredentials | None:
    """Load cached credentials for ``name``, or None if the user has not logged in."""
    path = credentials_path(name)
    if not path.is_file():
        return None
    return IapCredentials(**json.loads(path.read_text()))


def save_iap_credentials(name: str, credentials: IapCredentials) -> Path:
    """Persist ``credentials`` for ``name`` with owner-only permissions."""
    path = credentials_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(credentials), indent=2))
    path.chmod(0o600)
    return path


def desktop_login(name: str, client_secrets_path: str, *, headless: bool = False) -> IapCredentials:
    """Run the browser OAuth flow for ``name`` and cache the refresh token.

    Reads the Google *desktop* OAuth client secret JSON at ``client_secrets_path``
    (downloaded from the Cloud Console), runs the consent flow (or prints a URL to
    paste back when ``headless``), and stores the credentials for later silent use
    via :func:`provider_for`. Returns the freshly cached credentials.
    """
    with open(client_secrets_path) as f:
        installed = json.load(f).get("installed")
    if installed is None:
        raise ValueError(f"{client_secrets_path}: expected a desktop ('installed') OAuth client secret")
    client_id = installed["client_id"]
    client_secret = installed["client_secret"]

    _, refresh_token = run_iap_desktop_login(client_id, client_secret, headless=headless)
    credentials = IapCredentials(client_id=client_id, client_secret=client_secret, refresh_token=refresh_token)
    save_iap_credentials(name, credentials)
    return credentials


def _login_hint(name: str) -> str:
    """The canonical 'run marin-login' remedy for endpoint ``name``."""
    return f"run `marin-login login {name} --client-secrets <desktop.json>` to authenticate"


def provider_for(name: str) -> IapRefreshTokenProvider:
    """Build an :class:`~rigging.auth.IapRefreshTokenProvider` from cached credentials.

    Raises:
        IapLoginRequired: if the user has not run the desktop login for ``name``.
    """
    credentials = load_iap_credentials(name)
    if credentials is None:
        raise IapLoginRequired(
            f"no cached IAP credentials for {name!r} at {credentials_path(name)}; {_login_hint(name)}"
        )
    return IapRefreshTokenProvider(
        credentials.client_id,
        credentials.client_secret,
        credentials.refresh_token,
        login_hint=_login_hint(name),
    )
