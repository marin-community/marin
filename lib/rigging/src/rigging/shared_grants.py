# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Receiving-service adapter for locally verified grants and live decisions.

Local services can keep their native authentication policy. Installing this
adapter is an explicit deployment choice; a failed remote check never supplies
an anonymous identity or switches policies.
"""

import math
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rigging.grant_auth import GrantAccess, verified_grant_token
from rigging.grants import GrantDenied
from rigging.server_auth import VerifiedIdentity
from rigging.token_authority import JwksVerifier

SHARED_GRANT_ROLE = "shared-grant"


def authority_origin(url: str) -> str:
    """Validate a configured credential destination, allowing HTTP only on loopback."""
    origin = urlsplit(url)
    if (
        not origin.hostname
        or origin.username
        or origin.password
        or origin.query
        or origin.fragment
        or origin.path not in {"", "/"}
        or not (
            origin.scheme == "https"
            or (origin.scheme == "http" and origin.hostname in {"localhost", "127.0.0.1", "::1"})
        )
    ):
        raise ValueError("Shared grants require an HTTPS origin outside loopback")
    return url.rstrip("/")


class SharedGrantUnavailable(RuntimeError):
    """No current decision is available; retry without granting authority."""


class SharedGrantConfig(BaseModel):
    """Pinned public trust and a deployment-selected authorization endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    issuer: str
    public_keys: tuple[str, ...]
    authority_url: str
    ca_bundle: str | None = None
    timeout: float = Field(gt=0, default=5)

    @model_validator(mode="after")
    def validate_trust(self):
        authority_origin(self.authority_url)
        if not self.issuer or not self.public_keys:
            raise ValueError("Shared grants require pinned issuer and keys")
        return self


class SharedGrantClient:
    """Authenticate grant callers without assigning native service privileges."""

    def __init__(self, config: SharedGrantConfig, *, audience: str):
        self.config = config
        self.audience = audience
        self._verifier = JwksVerifier(
            issuers={config.issuer: config.public_keys},
            expected_audiences=frozenset({audience}),
            leeway_seconds=0,
        )

    def access(self, token: str) -> GrantAccess:
        return verified_grant_token(self._verifier, token, audience=self.audience)

    def verify(self, token: str) -> VerifiedIdentity:
        access = self.access(token)
        return VerifiedIdentity(user_id=access.actor, role=SHARED_GRANT_ROLE)

    def authorize(self, token: str, *, action: str, resource: str) -> GrantAccess:
        """Check the server-derived operation against token and current authority."""
        access = self.access(token)
        if not access.allows(action, resource):
            raise GrantDenied("Operation is outside token bounds")
        try:
            response = requests.post(
                self.config.authority_url.rstrip("/") + "/v1/authorize",
                headers={"Authorization": f"Bearer {token}"},
                json={"audience": self.audience, "action": action, "resource": resource},
                timeout=self.config.timeout,
                verify=self.config.ca_bundle or True,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise SharedGrantUnavailable("Shared authorization is unavailable") from error
        if response.status_code in {401, 403}:
            raise GrantDenied("Current authority denied the operation")
        if response.status_code != 200:
            raise SharedGrantUnavailable("Shared authorization returned no decision")
        try:
            decision = response.json()
            expiry = decision["expires_at"]
            matches = (
                decision["grant_id"] == access.grant_id
                and decision["actor"] == access.actor
                and decision["owner"] == access.owner
                and decision["policy_revision"] == access.policy_revision
                and isinstance(expiry, (int, float))
                and not isinstance(expiry, bool)
                and math.isfinite(expiry)
                and access.expires_at <= expiry
            )
        except (ValueError, TypeError, KeyError) as error:
            raise SharedGrantUnavailable("Malformed shared authorization decision") from error
        if not matches:
            raise SharedGrantUnavailable("Shared authorization decision does not match the token")
        # A slow authority response must not extend a token's lifetime.
        self.access(token)
        return access
