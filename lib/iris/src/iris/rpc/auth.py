# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authentication interceptor for Iris Connect RPC services.

Provides bearer token verification on the server side and token injection
on the client side. Authentication is optional: when no verifier is configured,
all requests pass through without authentication.
"""

import hashlib
import logging
from contextvars import ContextVar
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError

logger = logging.getLogger(__name__)


def hash_token(raw_token: str) -> str:
    """SHA-256 hex digest of a raw API key. Used for storage and lookup."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


# Thread-local storage for the verified user identity within an RPC handler.
# Set by AuthInterceptor before calling the handler, read by service code.
_verified_user: ContextVar[str | None] = ContextVar("verified_user", default=None)


def get_verified_user() -> str | None:
    """Return the verified user for the current RPC, or None if auth is disabled."""
    return _verified_user.get()


class TokenVerifier(Protocol):
    """Verifies a bearer token and returns the authenticated username."""

    def verify(self, token: str) -> str:
        """Verify the token and return the username (e.g. email).

        Raises:
            ValueError: If the token is invalid or expired.
        """
        ...


class StaticTokenVerifier:
    """Maps fixed tokens to usernames. Useful for testing and worker auth."""

    def __init__(self, tokens: dict[str, str]):
        """Args:
        tokens: Mapping of token string to username.
        """
        self._tokens = tokens

    def verify(self, token: str) -> str:
        user = self._tokens.get(token)
        if user is None:
            raise ValueError("Invalid token")
        return user


class GcpTokenVerifier:
    """Verifies GCP OAuth2 / OIDC identity tokens."""

    def __init__(self, audience: str):
        self._audience = audience

    def verify(self, token: str) -> str:
        from google.auth import exceptions as google_auth_exceptions
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        try:
            info = id_token.verify_oauth2_token(token, google_requests.Request(), self._audience)
        except (ValueError, google_auth_exceptions.GoogleAuthError) as exc:
            raise ValueError(f"GCP token verification failed: {exc}") from exc
        email = info.get("email")
        if not email:
            raise ValueError("Token does not contain an email claim")
        return email


class CompositeTokenVerifier:
    """Tries multiple verifiers in order, returning the first successful result."""

    def __init__(self, verifiers: list[TokenVerifier]):
        if not verifiers:
            raise ValueError("CompositeTokenVerifier requires at least one verifier")
        self._verifiers = verifiers

    def verify(self, token: str) -> str:
        errors = []
        for verifier in self._verifiers:
            try:
                return verifier.verify(token)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(f"All verifiers failed: {'; '.join(errors)}")


class AuthInterceptor:
    """Server-side Connect RPC interceptor that enforces bearer token auth.

    Reads the Authorization header, verifies the token via the configured
    verifier, and stores the verified user in a ContextVar for the service
    layer to read via get_verified_user().
    """

    def __init__(self, verifier: TokenVerifier):
        self._verifier = verifier

    def intercept_unary_sync(self, call_next, request, ctx):
        headers = ctx.request_headers()
        auth_header = headers.get("authorization", "")

        if not auth_header.startswith("Bearer "):
            raise ConnectError(Code.UNAUTHENTICATED, "Missing or malformed Authorization header")

        token = auth_header[len("Bearer ") :]
        try:
            user = self._verifier.verify(token)
        except ValueError as exc:
            raise ConnectError(Code.UNAUTHENTICATED, f"Authentication failed: {exc}") from exc

        reset_token = _verified_user.set(user)
        try:
            return call_next(request, ctx)
        finally:
            _verified_user.reset(reset_token)


class NullAuthInterceptor:
    """Interceptor for null-auth mode: all requests are treated as a synthetic admin user."""

    def __init__(self, user: str = "anonymous"):
        self._user = user

    def intercept_unary_sync(self, call_next, request, ctx):
        reset_token = _verified_user.set(self._user)
        try:
            return call_next(request, ctx)
        finally:
            _verified_user.reset(reset_token)


class AuthTokenInjector:
    """Client-side Connect RPC interceptor that attaches a bearer token to requests."""

    def __init__(self, token_provider: "TokenProvider"):
        self._provider = token_provider

    def intercept_unary_sync(self, call_next, request, ctx):
        token = self._provider.get_token()
        if token:
            ctx.request_headers()["authorization"] = f"Bearer {token}"
        return call_next(request, ctx)


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


class GcpTokenProvider:
    """Acquires GCP OIDC identity tokens for server-side verification.

    Works on GCE instances, Cloud Run, and with service account credentials.
    For user credentials (e.g. developer laptops), use CliGcpTokenProvider instead.
    """

    def __init__(self, audience: str):
        self._audience = audience

    def get_token(self) -> str | None:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        request = google_requests.Request()
        return id_token.fetch_id_token(request, self._audience)


class CliGcpTokenProvider:
    """Acquires GCP ID tokens, falling back to gcloud CLI for user credentials.

    On GCE/service accounts, uses the metadata server via google-auth SDK.
    On developer workstations with user ADC credentials, falls back to
    `gcloud auth print-identity-token` since user credentials cannot mint
    ID tokens via the SDK.
    """

    def __init__(self, audience: str):
        self._audience = audience

    def get_token(self) -> str | None:
        import subprocess

        from google.auth import exceptions as google_auth_exceptions
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        try:
            request = google_requests.Request()
            return id_token.fetch_id_token(request, self._audience)
        except (google_auth_exceptions.DefaultCredentialsError, google_auth_exceptions.GoogleAuthError):
            pass

        result = subprocess.run(
            ["gcloud", "auth", "print-identity-token", f"--audiences={self._audience}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gcloud auth failed: {result.stderr.strip()}")
        return result.stdout.strip()
