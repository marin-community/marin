# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authentication interceptor for Iris Connect RPC services.

All tokens are JWTs signed with HMAC-SHA256. Verification is a pure crypto
operation — no database hit on the hot path. User identity and role are
embedded in the JWT claims, so authorization checks read directly from the
verified token instead of querying the database.

Authentication is optional: when no verifier is configured, all requests
pass through as the anonymous admin user.
"""

import hashlib
import logging
import time
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from http.cookies import SimpleCookie
from typing import Protocol

import jwt
from connectrpc.code import Code
from connectrpc.errors import ConnectError

logger = logging.getLogger(__name__)

SESSION_COOKIE = "iris_session"

# Default TTL for JWT tokens issued by JwtTokenManager.
DEFAULT_JWT_TTL_SECONDS = 86400 * 30  # 30 days

# Minimum interval between touch_callback invocations for the same key (seconds).
_TOUCH_INTERVAL_SECONDS = 300  # 5 minutes


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    """Identity of an authenticated caller, extracted from JWT claims."""

    user_id: str
    role: str


def _extract_cookie(cookie_header: str, name: str) -> str | None:
    """Extract a named cookie value from a raw Cookie header."""
    if not cookie_header:
        return None
    try:
        cookie = SimpleCookie(cookie_header)
        morsel = cookie.get(name)
        return morsel.value if morsel else None
    except Exception:
        return None


def extract_bearer_token(headers: dict) -> str | None:
    """Extract bearer token from Authorization header or session cookie."""
    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[len("Bearer ") :]
    cookie_header = headers.get("cookie", "")
    return _extract_cookie(cookie_header, SESSION_COOKIE)


def hash_token(raw_token: str) -> str:
    """SHA-256 hex digest of a raw API key. Used for storage and lookup."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


# Per-request identity set by AuthInterceptor, read by service handlers.
_verified_identity: ContextVar[VerifiedIdentity | None] = ContextVar("verified_identity", default=None)


def get_verified_identity() -> VerifiedIdentity | None:
    """Return the verified identity for the current RPC, or None if auth is disabled."""
    return _verified_identity.get()


def get_verified_user() -> str | None:
    """Return just the user_id for the current RPC, or None."""
    identity = _verified_identity.get()
    return identity.user_id if identity is not None else None


# ---------------------------------------------------------------------------
# Centralized authorization — policy is defined here, not scattered in service
# ---------------------------------------------------------------------------


class AuthzAction(StrEnum):
    """Actions requiring authorization. Add new actions here; policy is in POLICY."""

    REGISTER_WORKER = "register_worker"
    MANAGE_OTHER_KEYS = "manage_other_keys"
    MANAGE_BUDGETS = "manage_budgets"


# Action → frozenset of roles allowed. Admin is implicitly always allowed.
POLICY: dict[AuthzAction, frozenset[str]] = {
    AuthzAction.REGISTER_WORKER: frozenset({"worker"}),
    AuthzAction.MANAGE_OTHER_KEYS: frozenset(),  # admin only
    AuthzAction.MANAGE_BUDGETS: frozenset(),  # admin only
}


def require_identity() -> VerifiedIdentity:
    """Get the verified identity for the current RPC or raise UNAUTHENTICATED."""
    identity = _verified_identity.get()
    if identity is None:
        raise ConnectError(Code.UNAUTHENTICATED, "Authentication required")
    return identity


def authorize(action: AuthzAction) -> VerifiedIdentity:
    """Require the current caller has permission for the given action.

    Admin role is always authorized. Other roles are checked against POLICY.
    """
    identity = require_identity()
    if identity.role == "admin":
        return identity
    allowed = POLICY.get(action, frozenset())
    if identity.role not in allowed:
        raise ConnectError(Code.PERMISSION_DENIED, f"{action} not allowed for role {identity.role}")
    return identity


def authorize_resource_owner(resource_owner: str) -> VerifiedIdentity:
    """Require the caller owns the resource or is admin."""
    identity = require_identity()
    if identity.role == "admin":
        return identity
    if identity.user_id != resource_owner:
        raise ConnectError(
            Code.PERMISSION_DENIED,
            f"User '{identity.user_id}' cannot access resources owned by '{resource_owner}'",
        )
    return identity


class TokenVerifier(Protocol):
    """Verifies a bearer token and returns the authenticated identity."""

    def verify(self, token: str) -> VerifiedIdentity:
        """Verify the token and return the identity.

        Raises:
            ValueError: If the token is invalid or expired.
        """
        ...


class StaticTokenVerifier:
    """Maps fixed tokens to identities. Useful for testing and login exchange."""

    def __init__(self, tokens: dict[str, str], roles: dict[str, str] | None = None):
        """Args:
        tokens: Mapping of token string to username.
        roles: Optional mapping of username to role (defaults to "user").
        """
        self._tokens = tokens
        self._roles = roles or {}

    def verify(self, token: str) -> VerifiedIdentity:
        user = self._tokens.get(token)
        if user is None:
            raise ValueError("Invalid token")
        role = self._roles.get(user, "user")
        return VerifiedIdentity(user_id=user, role=role)


class GcpAccessTokenVerifier:
    """Verifies GCP OAuth2 access tokens via Google's tokeninfo endpoint.

    Optionally checks that the user has access to a specific GCP project
    using the Cloud Resource Manager API with the user's own token.
    """

    _TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
    _PROJECT_URL_TEMPLATE = "https://cloudresourcemanager.googleapis.com/v3/projects/{}"

    def __init__(self, project_id: str | None = None):
        self._project_id = project_id

    def verify(self, token: str) -> VerifiedIdentity:
        import requests

        resp = requests.get(self._TOKENINFO_URL, params={"access_token": token}, timeout=10)
        if resp.status_code != 200:
            raise ValueError(f"Token verification failed (status {resp.status_code})")
        info = resp.json()
        email = info.get("email")
        if not email:
            raise ValueError("Token does not contain an email claim")

        if self._project_id:
            proj_resp = requests.get(
                self._PROJECT_URL_TEMPLATE.format(self._project_id),
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if proj_resp.status_code != 200:
                raise ValueError(f"User {email} does not have access to project {self._project_id}")

        return VerifiedIdentity(user_id=email, role="user")


class JwtTokenManager:
    """Creates and verifies HMAC-SHA256 JWT tokens.

    Verification is a pure crypto operation followed by an in-memory
    revocation check — no I/O on the hot path. An optional ``touch_callback``
    fires at most once per ``_TOUCH_INTERVAL_SECONDS`` per JTI, giving the
    caller a sampled hook for updating external state (e.g. ``last_used_at``
    in a database) without adding per-RPC writes.
    """

    def __init__(
        self,
        signing_key: str,
        *,
        touch_callback: Callable[[str, float], None] | None = None,
    ):
        self._signing_key = signing_key
        self._revoked_jtis: set[str] = set()
        self._touch_callback = touch_callback
        # Tracks the last wall-clock time the touch callback fired per jti.
        self._last_touched: dict[str, float] = {}

    @property
    def signing_key(self) -> str:
        """HMAC secret used to sign and verify JWTs. Do not log or serialize."""
        return self._signing_key

    def create_token(
        self,
        user_id: str,
        role: str,
        key_id: str,
        ttl_seconds: int = DEFAULT_JWT_TTL_SECONDS,
    ) -> str:
        now = time.time()
        payload = {
            "sub": user_id,
            "role": role,
            "jti": key_id,
            "iat": int(now),
            "exp": int(now + ttl_seconds),
        }
        return jwt.encode(payload, self._signing_key, algorithm="HS256")

    def verify(self, token: str) -> VerifiedIdentity:
        """Verify JWT signature and claims, check revocation.

        On success, invokes ``touch_callback`` at most once per key per
        ``_TOUCH_INTERVAL_SECONDS`` to avoid hot-path side effects.
        """
        try:
            payload = jwt.decode(token, self._signing_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError as exc:
            raise ValueError("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise ValueError(f"Invalid token: {exc}") from exc

        jti = payload.get("jti", "")
        if jti in self._revoked_jtis:
            raise ValueError("Token has been revoked")

        self._maybe_touch(jti)

        return VerifiedIdentity(
            user_id=payload["sub"],
            role=payload.get("role", "user"),
        )

    def _maybe_touch(self, jti: str) -> None:
        """Fire the touch callback if enough time has elapsed since the last fire."""
        if self._touch_callback is None or not jti:
            return
        now = time.time()
        last = self._last_touched.get(jti, 0.0)
        if now - last < _TOUCH_INTERVAL_SECONDS:
            return
        self._last_touched[jti] = now
        try:
            self._touch_callback(jti, now)
        except Exception:
            logger.debug("Touch callback failed for key %s", jti, exc_info=True)

    def revoke(self, jti: str) -> None:
        """Add a JTI to the in-memory revocation set."""
        self._revoked_jtis.add(jti)

    def add_revocations(self, jtis: Iterable[str]) -> None:
        """Replace the in-memory revocation set with the given JTIs.

        Callers are expected to load the revocation set from their backing
        store (e.g. the controller's api_keys table). Expired JWTs are
        rejected by signature verification anyway, so only active revocations
        need to be tracked here.
        """
        self._revoked_jtis = set(jtis)


class CompositeTokenVerifier:
    """Tries multiple verifiers in order, returning the first successful result."""

    def __init__(self, verifiers: list[TokenVerifier]):
        if not verifiers:
            raise ValueError("CompositeTokenVerifier requires at least one verifier")
        self._verifiers = verifiers

    def verify(self, token: str) -> VerifiedIdentity:
        errors = []
        for verifier in self._verifiers:
            try:
                return verifier.verify(token)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(f"All verifiers failed: {'; '.join(errors)}")


def resolve_auth(
    token: str | None,
    verifier: TokenVerifier,
    optional: bool,
) -> VerifiedIdentity | None:
    """Shared auth policy for gRPC interceptors and HTTP middleware.

    Returns VerifiedIdentity on success, None for anonymous passthrough.
    Raises ValueError on rejected tokens (invalid token, or missing when required).
    """
    if token is None:
        if optional:
            return None
        raise ValueError("Missing authentication")
    return verifier.verify(token)


class AuthInterceptor:
    """Server-side Connect RPC interceptor that enforces bearer token auth.

    Reads the Authorization header (or session cookie), verifies the JWT via
    the configured verifier, and stores the VerifiedIdentity in a ContextVar
    for the service layer to read via get_verified_identity().
    """

    def __init__(self, verifier: TokenVerifier):
        self._verifier = verifier

    def intercept_unary_sync(self, call_next, request, ctx):
        token = extract_bearer_token(ctx.request_headers())
        if not token:
            raise ConnectError(Code.UNAUTHENTICATED, "Missing or malformed Authorization header")

        try:
            identity = self._verifier.verify(token)
        except ValueError as exc:
            logger.warning("Authentication failed: %s", exc)
            raise ConnectError(Code.UNAUTHENTICATED, "Authentication failed") from exc

        reset_token = _verified_identity.set(identity)
        try:
            return call_next(request, ctx)
        finally:
            _verified_identity.reset(reset_token)

    def intercept_server_stream_sync(self, call_next, request, ctx):
        raise ConnectError(Code.UNIMPLEMENTED, "Streaming RPCs are not supported")

    def intercept_client_stream_sync(self, call_next, request, ctx):
        raise ConnectError(Code.UNIMPLEMENTED, "Streaming RPCs are not supported")

    def intercept_bidi_stream_sync(self, call_next, request, ctx):
        raise ConnectError(Code.UNIMPLEMENTED, "Streaming RPCs are not supported")


class NullAuthInterceptor:
    """Interceptor for null-auth mode.

    When a verifier is provided, tokens are verified if present (e.g. worker
    tokens) but unauthenticated requests fall through as the anonymous admin.
    Without a verifier, all requests are treated as anonymous admin.
    """

    def __init__(
        self,
        user: str = "anonymous",
        role: str = "admin",
        verifier: TokenVerifier | None = None,
    ):
        self._default_identity = VerifiedIdentity(user_id=user, role=role)
        self._verifier = verifier

    def intercept_unary_sync(self, call_next, request, ctx):
        identity = self._default_identity

        if self._verifier is not None:
            token = extract_bearer_token(ctx.request_headers())
            if token:
                try:
                    identity = self._verifier.verify(token)
                except ValueError:
                    pass

        reset_token = _verified_identity.set(identity)
        try:
            return call_next(request, ctx)
        finally:
            _verified_identity.reset(reset_token)


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


class GcpAccessTokenProvider:
    """Gets OAuth2 access tokens via google-auth SDK.

    Works for all credential types: user accounts (from gcloud auth
    application-default login), service accounts, and GCE metadata.
    Tokens are cached until 5 minutes before expiry.
    """

    _REFRESH_MARGIN_SECONDS = 300

    def __init__(self):
        self._creds = None
        self._cached_token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str | None:
        if self._cached_token is not None and time.monotonic() < self._expires_at:
            return self._cached_token

        import google.auth
        import google.auth.transport.requests

        if self._creds is None:
            self._creds, _ = google.auth.default()
        self._creds.refresh(google.auth.transport.requests.Request())

        self._cached_token = self._creds.token
        now_mono = time.monotonic()
        if self._creds.expiry is not None:
            self._expires_at = now_mono + (self._creds.expiry.timestamp() - time.time()) - self._REFRESH_MARGIN_SECONDS
        else:
            self._expires_at = now_mono + self._REFRESH_MARGIN_SECONDS

        return self._cached_token
