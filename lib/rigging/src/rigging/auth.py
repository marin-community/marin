# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transport-generic, client-side authentication for Connect RPC.

This module owns only the *client* side of auth: acquiring a bearer token and
attaching it as a request header. It carries no knowledge of any particular
service — no JWT minting/verification, no role semantics, no token store. Those
service-specific concerns live with the service (e.g. iris).

Two token sources are provided against ambient Google credentials:
``GcpAccessTokenProvider`` mints OAuth2 *access* tokens (for Google APIs and
loopback-trust services) and ``IapUserIdTokenProvider`` mints Google-signed
OIDC *ID* tokens for a fixed audience (the IAP OAuth client id). Both cache the
token until shortly before expiry and only touch the network inside
``get_token``.

The injectors attach the token to outgoing requests: ``AuthTokenInjector`` sets
``Authorization`` (app auth) and ``ProxyAuthTokenInjector`` sets
``proxy-authorization`` (IAP edge auth). Both are Connect *metadata*
interceptors (the ``on_start`` hook), so the header rides on every RPC shape —
unary and streaming alike — for both sync and async clients.
"""

import time
from typing import Protocol

import google.auth
import google.auth.jwt
import google.auth.transport.requests
import google.oauth2.id_token

_REFRESH_MARGIN_SECONDS = 300


class TokenProvider(Protocol):
    """Provides a bearer token for outgoing requests."""

    def get_token(self) -> str | None:
        """Return a token string, or None to skip auth."""
        ...


class StaticTokenProvider:
    """Returns a fixed token. Useful for testing and worker auth."""

    def __init__(self, token: str):
        self._token = token

    def get_token(self) -> str | None:
        return self._token


class GcpAccessTokenProvider:
    """Mints OAuth2 access tokens from ambient Google credentials.

    Works for all credential types: user accounts (from ``gcloud auth
    application-default login``), service accounts, and GCE metadata. Tokens are
    cached until five minutes before expiry. Credential discovery and refresh
    happen only inside ``get_token``.
    """

    def __init__(self):
        self._creds = None
        self._cached_token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str | None:
        if self._cached_token is not None and time.monotonic() < self._expires_at:
            return self._cached_token

        if self._creds is None:
            self._creds, _ = google.auth.default()
        self._creds.refresh(google.auth.transport.requests.Request())

        self._cached_token = self._creds.token
        now_mono = time.monotonic()
        if self._creds.expiry is not None:
            self._expires_at = now_mono + (self._creds.expiry.timestamp() - time.time()) - _REFRESH_MARGIN_SECONDS
        else:
            self._expires_at = now_mono + _REFRESH_MARGIN_SECONDS

        return self._cached_token


class IapUserIdTokenProvider:
    """Mints Google-signed OIDC ID tokens for a fixed IAP audience.

    The audience is the OAuth client id of the IAP-protected resource. The token
    is fetched from ambient credentials via ``fetch_id_token`` and cached until
    five minutes before its ``exp`` claim. Credential access happens only inside
    ``get_token``.
    """

    def __init__(self, audience: str):
        self._audience = audience
        self._cached_token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str | None:
        if self._cached_token is not None and time.monotonic() < self._expires_at:
            return self._cached_token

        token = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), self._audience)

        self._cached_token = token
        claims = google.auth.jwt.decode(token, verify=False)
        now_mono = time.monotonic()
        exp = claims.get("exp")
        if exp is not None:
            self._expires_at = now_mono + (float(exp) - time.time()) - _REFRESH_MARGIN_SECONDS
        else:
            self._expires_at = now_mono + _REFRESH_MARGIN_SECONDS

        return self._cached_token


class _BearerHeaderInjector:
    """Metadata interceptor that attaches ``<header>: Bearer <token>``.

    Implemented against Connect's metadata interceptor protocol (``on_start`` /
    ``on_start_sync``) rather than the unary hooks, so ``connectrpc`` applies it
    to every RPC shape — unary, client-stream, server-stream, and bidi — for
    both sync and async clients. No header is set when the provider returns None
    (the loopback / SSH-tunnel-trust case).
    """

    _HEADER: str

    def __init__(self, provider: TokenProvider):
        self._provider = provider

    def _apply(self, ctx) -> None:
        token = self._provider.get_token()
        if token:
            ctx.request_headers()[self._HEADER] = f"Bearer {token}"

    def on_start_sync(self, ctx):
        self._apply(ctx)

    def on_end_sync(self, token, ctx, error) -> None:
        return

    async def on_start(self, ctx):
        self._apply(ctx)

    async def on_end(self, token, ctx, error) -> None:
        return


class AuthTokenInjector(_BearerHeaderInjector):
    """Attaches app auth as ``Authorization: Bearer <token>``."""

    _HEADER = "authorization"


class ProxyAuthTokenInjector(_BearerHeaderInjector):
    """Attaches the IAP edge token as ``proxy-authorization: Bearer <token>``.

    The edge token rides in ``proxy-authorization`` so the app-level
    ``Authorization`` header stays free for the service's own JWT.
    """

    _HEADER = "proxy-authorization"
