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
class OAuthClient:
    """A Google OAuth client identity (client id + secret)."""

    client_id: str
    client_secret: str


# The Marin desktop ("installed") OAuth client that drives the IAP browser-login
# flow. For an installed app the "client secret" is not confidential — it is part
# of the app's public identity, not a credential (RFC 8252 §8.5) — so it ships in
# source: `marin-login login <cluster>` then needs no Console download. It only
# names the app to Google's OAuth endpoint; IAP still authorizes each user
# individually against its per-backend allowlist.
MARIN_DESKTOP_OAUTH_CLIENT = OAuthClient(
    client_id="748532799086-qf8m6mvovtdmd71npm07gk1ohijsr3q5.apps.googleusercontent.com",
    client_secret="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",
)


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


def read_desktop_client(path: str) -> OAuthClient:
    """Read a Google *desktop* ('installed') OAuth client secret JSON from ``path``."""
    with open(path) as f:
        installed = json.load(f).get("installed")
    if installed is None:
        raise ValueError(f"{path}: expected a desktop ('installed') OAuth client secret")
    return OAuthClient(installed["client_id"], installed["client_secret"])


def desktop_login(
    name: str, client: OAuthClient = MARIN_DESKTOP_OAUTH_CLIENT, *, headless: bool = False
) -> IapCredentials:
    """Run the browser OAuth flow for ``name`` and cache the refresh token.

    Uses the built-in :data:`MARIN_DESKTOP_OAUTH_CLIENT` unless ``client`` overrides
    it (load one from a Console download with :func:`read_desktop_client`). Runs the
    consent flow (or prints a URL to paste back when ``headless``) and stores the
    credentials for later silent use via :func:`provider_for`. Returns the freshly
    cached credentials.
    """
    _, refresh_token = run_iap_desktop_login(client.client_id, client.client_secret, headless=headless)
    credentials = IapCredentials(
        client_id=client.client_id, client_secret=client.client_secret, refresh_token=refresh_token
    )
    save_iap_credentials(name, credentials)
    return credentials


def _login_hint(name: str) -> str:
    """The canonical 'run marin-login' remedy for endpoint ``name``."""
    return f"run `marin-login login {name}` to authenticate"


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
