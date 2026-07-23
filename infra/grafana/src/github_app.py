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
from dataclasses import dataclass
from datetime import datetime

import httpx
import jwt
from errors import UpstreamError

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"

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


@dataclass(frozen=True)
class GithubAppCredentials:
    """The "Marin Ops Agent" app identity and its installation on the served repo."""

    app_id: str
    installation_id: str
    private_key: str  # PEM


class GithubAppAuth(httpx.Auth):
    """An httpx auth flow that mints, caches, and refreshes an installation token.

    It yields the token request through the caller's client (httpx's "auth flow
    yields a request" pattern) rather than opening a second client.
    """

    # auth_flow reads the token response body to cache the token and its expiry.
    requires_response_body = True

    def __init__(self, credentials: GithubAppCredentials, repository: str) -> None:
        self._credentials = credentials
        # Scope the token to the single repo the panels read; repository is "owner/name".
        self._repository = repository.split("/")[1]
        self._token: str | None = None
        self._expires_at = 0.0

    def auth_flow(self, request: httpx.Request):
        # No lock: a concurrent cold cache mints twice at worst, and both tokens are
        # valid. This runs about once a minute, so the contention window is tiny.
        if self._token is None or time.time() >= self._expires_at - _EXPIRY_SKEW:
            try:
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
            f"{_GITHUB_API}/app/installations/{self._credentials.installation_id}/access_tokens",
            headers={"authorization": f"Bearer {self._app_jwt()}", "accept": "application/vnd.github+json"},
            json={"repositories": [self._repository], "permissions": _TOKEN_PERMISSIONS},
        )

    def _app_jwt(self) -> str:
        now = int(time.time())
        payload = {"iat": now - _JWT_BACKDATE, "exp": now + _JWT_LIFETIME, "iss": self._credentials.app_id}
        return jwt.encode(payload, self._credentials.private_key, algorithm="RS256")


def _parse_expiry(value: str) -> float:
    """Parse GitHub's ISO-8601 `expires_at` to epoch seconds."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
