# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub App authentication for the bridge.

GitHub gates the GraphQL build query behind auth even for public repos. Instead of
a static personal token that silently expires — the failure that took the commit
panel down — the bridge authenticates as the "Marin Ops Agent" App: it signs a JWT
with the app's private key, looks up its installation, and mints a read-only
installation token, refreshing it before expiry. The private key is the only
long-lived secret; the tokens it mints last an hour and roll themselves.
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

# Read-only permissions the token is attenuated to, whatever the app itself holds:
# contents+metadata for the commit history, checks+statuses for the
# statusCheckRollup, actions for the ferry/nightly run lists.
_TOKEN_PERMISSIONS = {
    "metadata": "read",
    "contents": "read",
    "checks": "read",
    "statuses": "read",
    "actions": "read",
}

# Refresh a token this many seconds before its stated expiry.
_EXPIRY_SKEW = 300

# App JWTs may live at most 10 minutes; the backdate absorbs bridge/GitHub skew.
_JWT_LIFETIME = 540
_JWT_BACKDATE = 30


class GithubAppAuth(httpx.Auth):
    """An httpx auth flow that mints, caches, and refreshes an installation token."""

    # auth_flow reads the response bodies to cache the installation id and token.
    requires_response_body = True

    def __init__(self, credentials: GithubAppCredentials, repositories: Iterable[str]) -> None:
        self._credentials = credentials
        # An installation token is per-owner and takes bare repo names, so the repos
        # the panels read (ferries/builds plus every nightly lane) must share one owner.
        owner_and_names = [repo.split("/", 1) for repo in repositories]
        owners = {owner for owner, _ in owner_and_names}
        if len(owners) != 1:
            raise ValueError(f"installation token repositories must share one owner; got {sorted(owners)}")
        self._owner = owners.pop()
        self._repositories = sorted(name for _, name in owner_and_names)
        self._installation_id: int | None = None
        self._token: str | None = None
        self._expires_at = 0.0

    def auth_flow(self, request: httpx.Request):
        # No lock: a concurrent cold cache mints twice at worst, both valid, and this
        # runs about once a minute. Requests are yielded through the caller's client.
        if self._token is None or time.time() >= self._expires_at - _EXPIRY_SKEW:
            jwt_token = self._app_jwt()
            try:
                if self._installation_id is None:
                    lookup = yield _app_request("GET", f"/orgs/{self._owner}/installation", jwt_token)
                    self._installation_id = _require(lookup, 200, "installation lookup")["id"]
                minted = yield _app_request(
                    "POST",
                    f"/app/installations/{self._installation_id}/access_tokens",
                    jwt_token,
                    json={"repositories": self._repositories, "permissions": _TOKEN_PERMISSIONS},
                )
            except httpx.TransportError as err:
                raise UpstreamError("github", f"github app auth unreachable ({err})", status_code=504) from err
            body = _require(minted, 201, "installation token")
            self._token, self._expires_at = body["token"], _parse_expiry(body["expires_at"])
            logger.info("minted github installation token, expires %s", body["expires_at"])
        request.headers["authorization"] = f"Bearer {self._token}"
        yield request

    def _app_jwt(self) -> str:
        now = int(time.time())
        payload = {"iat": now - _JWT_BACKDATE, "exp": now + _JWT_LIFETIME, "iss": self._credentials.client_id}
        return jwt.encode(payload, self._credentials.private_key, algorithm="RS256")


def _app_request(method: str, path: str, jwt_token: str, *, json: dict | None = None) -> httpx.Request:
    return httpx.Request(
        method,
        f"{GITHUB_API_BASE}{path}",
        headers={"authorization": f"Bearer {jwt_token}", "accept": "application/vnd.github+json"},
        json=json,
    )


def _require(response: httpx.Response, status: int, what: str) -> dict:
    if response.status_code != status:
        raise UpstreamError("github", f"{what} returned {response.status_code}: {response.text}", status_code=502)
    return response.json()


def _parse_expiry(value: str) -> float:
    """Parse GitHub's ISO-8601 `expires_at` to epoch seconds."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
