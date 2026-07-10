# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report a caller's login status for a Marin cluster.

Mint-on-demand: :func:`resolve_login_status` proactively re-mints the cluster's
IAP edge token through the *same* provider a real RPC uses, then decodes it to
surface the caller's email and the token's expiry — so the status reflects the
identity requests actually present. When minting fails for any reason (an
expired/revoked login, no ambient credentials, or no network) it falls back to an
offline view built from the cached credential record, which still names the
cluster, the login method, and the email captured at last login.

This is the read side of "one login": :mod:`rigging.credentials` builds the edge
provider (no I/O) and this module executes it to report who you are.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import google.auth.jwt

from rigging.cluster_manifest import AuthProvider, ClusterAuth
from rigging.credential_store import load_credentials
from rigging.credentials import edge_provider

logger = logging.getLogger(__name__)

# Free-form credential-record annotation the login flow stores so the offline
# fallback can still show who the cached login belongs to without a network call.
EMAIL_METADATA_KEY = "email"


class LoginMethod(StrEnum):
    """How a client authenticates to a cluster."""

    NETWORK_TRUST = "network-trust"  # in-network / loopback trust; no token minted
    IAP_BROWSER = "iap-browser"  # cached desktop-OAuth login (from ``iris login``)
    IAP_SERVICE_ACCOUNT = "iap-service-account"  # ambient service-account credentials


class LoginState(StrEnum):
    """Whether the client currently holds a usable identity for a cluster."""

    ACTIVE = "active"  # a token was minted; email/expiry are live
    OFFLINE = "offline"  # a cached login is present but the mint failed
    NOT_LOGGED_IN = "not-logged-in"  # IAP cluster with no usable credentials
    NO_LOGIN_REQUIRED = "no-login-required"  # network / loopback trust admits the caller


@dataclass(frozen=True)
class LoginStatus:
    """A caller's resolved login status for one cluster.

    Attributes:
        cluster: The logical cluster name.
        state: Whether a usable identity is currently held (see :class:`LoginState`).
        method: How auth is attempted, or None when the caller is not logged in.
        endpoint: The URL the credentials authenticate to, when known.
        email: The caller's email — live from the minted token, or the value cached
            at last login when the status is offline.
        token_expiry: When the freshly minted edge token expires (``ACTIVE`` only).
            The underlying browser login (the refresh token) has no fixed expiry; it
            stays valid until revoked, and each request silently re-mints this token.
        detail: Why minting failed, for ``OFFLINE`` / ``NOT_LOGGED_IN``.
    """

    cluster: str
    state: LoginState
    method: LoginMethod | None
    endpoint: str | None
    email: str | None
    token_expiry: datetime | None
    detail: str | None


def email_from_id_token(id_token: str) -> str | None:
    """The ``email`` claim of an OIDC ID token, or None if absent/undecodable.

    The signature is not verified: the email is only displayed for the caller's own
    cached login, never used to authorize anything.
    """
    try:
        claims = google.auth.jwt.decode(id_token, verify=False)
    except ValueError:
        return None
    email = claims.get("email")
    return email if isinstance(email, str) else None


def resolve_login_status(cluster: str, auth: ClusterAuth) -> LoginStatus:
    """Resolve ``cluster``'s login status, minting a fresh edge token on demand.

    For an IAP cluster this mints through the same provider a real RPC uses and
    decodes the token for the caller's email and expiry. If minting fails for any
    reason it degrades to an offline view from the cached credential record. Non-IAP
    clusters need no login (in-network / loopback trust).
    """
    record = load_credentials(cluster)
    endpoint = record.endpoint if record is not None else (auth.iap.url if auth.iap is not None else None)

    if auth.provider is not AuthProvider.IAP or auth.iap is None:
        return LoginStatus(
            cluster=cluster,
            state=LoginState.NO_LOGIN_REQUIRED,
            method=LoginMethod.NETWORK_TRUST,
            endpoint=endpoint,
            email=None,
            token_expiry=None,
            detail=None,
        )

    has_browser_login = record is not None and record.edge_refresh_token is not None
    provider = edge_provider(cluster, auth)
    assert provider is not None  # non-None for an IAP cluster
    try:
        token = provider.get_token()  # network: re-mint the OIDC edge token
        claims = google.auth.jwt.decode(token, verify=False) if token else {}
        exp = claims.get("exp")
        email = claims.get("email")
        return LoginStatus(
            cluster=cluster,
            state=LoginState.ACTIVE,
            method=LoginMethod.IAP_BROWSER if has_browser_login else LoginMethod.IAP_SERVICE_ACCOUNT,
            endpoint=endpoint,
            email=email if isinstance(email, str) else None,
            token_expiry=datetime.fromtimestamp(exp, tz=UTC) if exp is not None else None,
            detail=None,
        )
    except Exception as exc:
        # An offline fallback is deliberately broad: the dump stays useful if the
        # mint fails for any reason (expired login, no credentials, no network).
        # The reason travels to the user in ``detail``, so this is a reported
        # degradation, not a swallowed error.
        logger.debug("Minting the edge token for cluster %r failed; offline fallback: %s", cluster, exc)
        if has_browser_login:
            return LoginStatus(
                cluster=cluster,
                state=LoginState.OFFLINE,
                method=LoginMethod.IAP_BROWSER,
                endpoint=endpoint,
                email=record.metadata.get(EMAIL_METADATA_KEY),
                token_expiry=None,
                detail=str(exc),
            )
        return LoginStatus(
            cluster=cluster,
            state=LoginState.NOT_LOGGED_IN,
            method=None,
            endpoint=endpoint,
            email=None,
            token_expiry=None,
            detail=str(exc),
        )
