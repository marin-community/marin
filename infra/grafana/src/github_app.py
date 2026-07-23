# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub App authentication for the bridge.

The GitHub panels read a public repo's commit history and Actions runs, but GitHub
gates the GraphQL build query behind auth even for public repos. Rather than a
static personal token that silently expires — the failure that took the commit
panel down — the bridge authenticates as the "Marin Ops Agent" GitHub App: it
signs a short JWT with the app's private key, exchanges it for an installation
access token scoped read-only to the one repo, and refreshes it before expiry.

The private key is the only long-lived secret; the installation tokens it mints
last an hour and roll themselves, so there is nothing to rotate operationally.
"""

import logging
import time
from collections.abc import Iterable
from datetime import datetime

import httpx
import jwt
from config import GITHUB_API_BASE, GithubAppCredentials
from errors import UpstreamError

logger = logging.getLogger(__name__)

# The installation token is attenuated to these read-only permissions. The app may
# hold broader (write) grants for other automations; the bridge asks only for what
# its panels read — contents+metadata for the commit history, checks+statuses for
# the statusCheckRollup, and actions for the ferry/nightly run lists.
_TOKEN_PERMISSIONS = {
    "metadata": "read",
    "contents": "read",
    "checks": "read",
    "statuses": "read",
    "actions": "read",
}

# Mint a fresh token this many seconds before the stated expiry, so a request never
# rides a token that expires mid-flight.
_EXPIRY_SKEW = 300

# App JWTs may live at most 10 minutes; 9 leaves headroom, and the 30s backdate
# absorbs clock skew between the bridge and GitHub.
_JWT_LIFETIME = 540
_JWT_BACKDATE = 30


class GithubAppAuth(httpx.Auth):
    """An httpx auth flow that mints, caches, and refreshes an installation token."""

    # auth_flow reads the token response body to cache the token and its expiry.
    requires_response_body = True

    def __init__(self, credentials: GithubAppCredentials, repositories: Iterable[str]) -> None:
        self._credentials = credentials
        # Scope the token to the repos the panels read (ferries/builds plus every
        # nightly lane repo). An installation token is per-owner, so they must share
        # one owner — the installation account; the request takes bare repo names.
        owner_and_names = [repo.split("/", 1) for repo in repositories]
        owners = {owner for owner, _ in owner_and_names}
        if len(owners) != 1:
            raise ValueError(f"installation token repositories must share one owner; got {sorted(owners)}")
        self._repositories = sorted(name for _, name in owner_and_names)
        self._token: str | None = None
        self._expires_at = 0.0

    def auth_flow(self, request: httpx.Request):
        # No lock: a concurrent cold cache mints twice at worst, and both tokens are
        # valid. This runs about once a minute, so the contention window is tiny.
        if self._token is None or time.time() >= self._expires_at - _EXPIRY_SKEW:
            try:
                # Yield the token request so it rides the caller's client transport
                # (httpx's "auth flow yields a request"), avoiding a second client.
                response = yield self._token_request()
            except httpx.TransportError as err:
                raise UpstreamError("github", f"installation token unreachable ({err})", status_code=504) from err
            if response.status_code != 201:
                raise UpstreamError(
                    "github",
                    f"installation token request returned {response.status_code}: {response.text}",
                    status_code=502,
                )
            body = response.json()
            self._token, self._expires_at = body["token"], _parse_expiry(body["expires_at"])
            logger.info("minted github installation token, expires %s", body["expires_at"])
        request.headers["authorization"] = f"Bearer {self._token}"
        yield request

    def _token_request(self) -> httpx.Request:
        return httpx.Request(
            "POST",
            f"{GITHUB_API_BASE}/app/installations/{self._credentials.installation_id}/access_tokens",
            headers={"authorization": f"Bearer {self._app_jwt()}", "accept": "application/vnd.github+json"},
            json={"repositories": self._repositories, "permissions": _TOKEN_PERMISSIONS},
        )

    def _app_jwt(self) -> str:
        now = int(time.time())
        payload = {"iat": now - _JWT_BACKDATE, "exp": now + _JWT_LIFETIME, "iss": self._credentials.app_id}
        return jwt.encode(payload, self._credentials.private_key, algorithm="RS256")


def _parse_expiry(value: str) -> float:
    """Parse GitHub's ISO-8601 `expires_at` to epoch seconds."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
