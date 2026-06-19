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

import contextlib
import ipaddress
import logging
import time
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from http.cookies import SimpleCookie
from typing import Protocol, cast

import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
import google.oauth2.id_token
import requests
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.auth.exceptions import GoogleAuthError

logger = logging.getLogger(__name__)

SESSION_COOKIE = "iris_session"

# Read-only role granted implicitly to an IAP-authenticated caller that has not
# run `iris login` (no Iris JWT). It may only call the read RPCs in
# DASHBOARD_READABLE_RPCS; see authorize_method.
DASHBOARD_ROLE = "dashboard"


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    """Identity of an authenticated caller, extracted from JWT claims."""

    user_id: str
    role: str


# Identity granted to any tokenless caller on a genuine loopback connection.
# Mirrors the null-auth default: reaching the loopback interface (SSH tunnel /
# on-host) already implies host-level trust, so the caller is the admin user.
# Jobs are still attributed per-user via the job name's owner segment.
LOOPBACK_IDENTITY = VerifiedIdentity(user_id="anonymous", role="admin")


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


# Per-request identity set by AuthInterceptor, read by service handlers.
_verified_identity: ContextVar[VerifiedIdentity | None] = ContextVar("verified_identity", default=None)


def get_verified_identity() -> VerifiedIdentity | None:
    """Return the verified identity for the current RPC, or None if auth is disabled."""
    return _verified_identity.get()


def get_verified_user() -> str | None:
    """Return just the user_id for the current RPC, or None."""
    identity = _verified_identity.get()
    return identity.user_id if identity is not None else None


@contextlib.contextmanager
def identity_scope(identity: VerifiedIdentity | None):
    """Bind ``identity`` as the verified identity for the duration of the block.

    Mirrors the ContextVar bookkeeping AuthInterceptor performs per RPC so code
    outside the interceptor (e.g. the dashboard's RPC dispatch) can establish
    the same identity for service handlers reached via get_verified_identity().
    """
    reset_token = _verified_identity.set(identity)
    try:
        yield
    finally:
        _verified_identity.reset(reset_token)


# ---------------------------------------------------------------------------
# Centralized authorization — policy is defined here, not scattered in service
# ---------------------------------------------------------------------------


class AuthzAction(StrEnum):
    """Actions requiring authorization. Add new actions here; policy is in POLICY."""

    ACT_AS_WORKER = "act_as_worker"
    MANAGE_OTHER_KEYS = "manage_other_keys"
    MANAGE_BUDGETS = "manage_budgets"


# Action → frozenset of roles allowed. Admin is implicitly always allowed.
POLICY: dict[AuthzAction, frozenset[str]] = {
    AuthzAction.ACT_AS_WORKER: frozenset({"worker"}),
    AuthzAction.MANAGE_OTHER_KEYS: frozenset(),  # admin only
    AuthzAction.MANAGE_BUDGETS: frozenset(),  # admin only
}


# RPC methods the read-only `dashboard` role may call. A default-deny allowlist:
# a dashboard caller (an IAP-authenticated browser that has not run `iris login`)
# may invoke only these read methods; everything else — job submit/terminate,
# worker registration, key/budget management, exec, profiling, raw queries — is
# denied. A newly added RPC is therefore denied to the dashboard role until it is
# explicitly listed here, which is the safe direction for a read-only tier.
DASHBOARD_READABLE_RPCS: frozenset[str] = frozenset(
    {
        # Jobs and tasks
        "GetJobStatus",
        "GetJobState",
        "ListJobs",
        "GetTaskStatus",
        "ListTasks",
        "GetProcessStatus",
        # Workers, endpoints, scheduler, autoscaler
        "ListWorkers",
        "GetWorkerStatus",
        "ListEndpoints",
        "GetAutoscalerStatus",
        "GetSchedulerState",
        "GetKubernetesClusterStatus",
        # Identity, users, budgets (read)
        "GetAuthInfo",
        "GetCurrentUser",
        "ListApiKeys",
        "ListUsers",
        "GetUserBudget",
        "ListUserBudgets",
        # RPC stats panel
        "GetRpcStats",
    }
)


def authorize_method(identity: VerifiedIdentity, method_name: str) -> None:
    """Enforce per-method access for restricted roles before dispatch.

    The ``dashboard`` role is read-only: it may call only the methods in
    ``DASHBOARD_READABLE_RPCS``. Other roles are unrestricted here — their
    mutating actions remain gated inside the handlers by ``authorize`` /
    ``authorize_resource_owner``. Raises ``PERMISSION_DENIED`` for a dashboard
    caller invoking a non-readable method.
    """
    if identity.role == DASHBOARD_ROLE and method_name not in DASHBOARD_READABLE_RPCS:
        raise ConnectError(
            Code.PERMISSION_DENIED,
            f"Read-only dashboard access cannot call {method_name}; run `iris login` to authenticate",
        )


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


class IapIdTokenVerifier:
    """Verifies a Google OIDC ID token and returns the caller's identity.

    Raises ValueError unless the token's signature and issuer are valid and its
    ``aud`` claim is one of ``audiences`` (the email is taken from the verified
    claims). Used as the login identity proof for an IAP-fronted controller;
    IAP's own IAM is the access gate, so no further project check is done here.
    """

    def __init__(self, audiences: Iterable[str]):
        self._audiences = frozenset(audiences)
        if not self._audiences:
            raise ValueError("IapIdTokenVerifier requires at least one audience")
        self._request = google.auth.transport.requests.Request()

    def verify(self, token: str) -> VerifiedIdentity:
        try:
            # audience=None: verify signature/issuer/expiry here, then check the
            # aud claim against our allow-set so multiple audiences are supported.
            payload = google.oauth2.id_token.verify_oauth2_token(token, self._request)
        except (ValueError, GoogleAuthError) as exc:
            raise ValueError(f"IAP ID token verification failed: {exc}") from exc

        aud = payload.get("aud")
        if aud not in self._audiences:
            raise ValueError(f"ID token audience {aud!r} is not an accepted IAP audience")
        email = payload.get("email")
        if not email:
            raise ValueError("ID token has no email claim (request the 'email' scope)")
        if payload.get("email_verified") is False:
            raise ValueError(f"ID token email {email} is not verified")
        return VerifiedIdentity(user_id=email, role="user")


# IAP injects this signed JWT on every request it admits; its `aud` is the
# backend-service resource path and it is signed with IAP's own (ES256) keys,
# published at the URL below.
IAP_ASSERTION_HEADER = "x-goog-iap-jwt-assertion"
_IAP_PUBLIC_KEYS_URL = "https://www.gstatic.com/iap/verify/public_key"
_IAP_CERTS_CACHE_TTL_SECONDS = 3600.0


class _CachingCertsRequest:
    """Wraps a google-auth transport ``Request`` to cache cert-endpoint GETs.

    ``google.oauth2.id_token.verify_token`` re-fetches the signing certs on every
    call. On the per-RPC assertion path that would be an HTTP round-trip per
    request; IAP's public keys rotate slowly, so the cert response is cached for
    a TTL. Only GETs are cached (the verify path issues nothing else).
    """

    def __init__(self, inner, cache_ttl_seconds: float = _IAP_CERTS_CACHE_TTL_SECONDS):
        self._inner = inner
        self._ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, object]] = {}

    def __call__(self, url, method="GET", **kwargs):
        if method != "GET":
            return self._inner(url, method=method, **kwargs)
        cached = self._cache.get(url)
        if cached is not None and time.monotonic() < cached[0]:
            return cached[1]
        response = self._inner(url, method=method, **kwargs)
        self._cache[url] = (time.monotonic() + self._ttl, response)
        return response


class IapAssertionVerifier:
    """Verifies IAP's signed ``X-Goog-IAP-JWT-Assertion`` request header.

    IAP signs a JWT asserting the authenticated identity and attaches it to every
    request it forwards. Verifying its signature and ``aud`` proves the request
    genuinely passed through IAP for *this* backend, so the asserted email can be
    trusted without an Iris JWT — an internal caller that bypasses the load
    balancer cannot forge it. Verified callers are granted the read-only
    ``dashboard`` role (the implicit-access path for browsers behind IAP).
    """

    def __init__(self, audience: str, role: str = DASHBOARD_ROLE):
        if not audience:
            raise ValueError("IapAssertionVerifier requires a signed-header audience")
        self._audience = audience
        self._role = role
        self._request = _CachingCertsRequest(google.auth.transport.requests.Request())

    def identity_from_headers(self, headers: dict) -> VerifiedIdentity | None:
        """Return the asserted identity, or None when no assertion header is present.

        Raises ValueError if the header is present but fails verification (a
        forged, stale, or wrong-audience assertion) so the caller rejects it.
        """
        assertion = headers.get(IAP_ASSERTION_HEADER)
        if not assertion:
            return None
        try:
            payload = google.oauth2.id_token.verify_token(
                assertion,
                self._request,
                audience=self._audience,
                certs_url=_IAP_PUBLIC_KEYS_URL,
            )
        except (ValueError, GoogleAuthError) as exc:
            raise ValueError(f"IAP assertion verification failed: {exc}") from exc
        email = payload.get("email")
        if not email:
            raise ValueError("IAP assertion has no email claim")
        return VerifiedIdentity(user_id=email, role=self._role)


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


def is_trusted_loopback(client_address: str | None, headers: dict) -> bool:
    """Return True if the request arrived over a genuine loopback connection.

    A connection is trusted-loopback iff its transport peer is a loopback
    address (127.0.0.0/8 or ::1) on a nonzero port *and* it carries no
    ``X-Forwarded-For`` header.

    The two conditions are individually sufficient and kept together as
    defence in depth. The controller runs uvicorn with
    ``forwarded_allow_ips="*"``, so when the client is derived from a forwarded
    header uvicorn rewrites ``scope["client"]`` to the attacker-controllable
    leftmost ``X-Forwarded-For`` entry and zeroes the port (it cannot recover
    the forwarded client's port). A public request spoofing
    ``X-Forwarded-For: 127.0.0.1`` therefore presents as ``("127.0.0.1", 0)``
    with the header present — rejected on both counts. Only a direct transport
    peer on the loopback interface (SSH tunnel / on-host) passes. See
    ``docs/auth-loopback-transition.md``.
    """
    if not client_address:
        return False
    if headers.get("x-forwarded-for"):
        return False
    host, _, port = client_address.rpartition(":")
    if not host or not port:
        return False
    try:
        if int(port) == 0:
            return False
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_auth(
    token: str | None,
    verifier: TokenVerifier,
    optional: bool,
    *,
    client_address: str | None = None,
    headers: dict | None = None,
    iap_assertion_verifier: "IapAssertionVerifier | None" = None,
) -> VerifiedIdentity | None:
    """Shared auth policy for gRPC interceptors and HTTP middleware.

    Returns VerifiedIdentity on success, None for anonymous passthrough.
    Raises ValueError on rejected tokens (invalid token, or missing when required).

    Precedence for a tokenless request:

    1. A present Iris JWT always wins (``verifier.verify``).
    2. With ``iap_assertion_verifier`` set, a verified IAP signed-header
       assertion grants the read-only ``dashboard`` identity — the implicit
       access path for an IAP-authenticated browser that has not run
       ``iris login``. A present-but-invalid assertion is rejected.
    3. A genuine loopback connection is trusted as the admin user (see
       ``is_trusted_loopback`` / ``LOOPBACK_IDENTITY``) — the SSH-tunnel path.
    4. Otherwise a missing token is allowed only when ``optional`` is set.
    """
    if token is not None:
        return verifier.verify(token)
    if iap_assertion_verifier is not None:
        iap_identity = iap_assertion_verifier.identity_from_headers(headers or {})
        if iap_identity is not None:
            return iap_identity
    if is_trusted_loopback(client_address, headers or {}):
        return LOOPBACK_IDENTITY
    if optional:
        return None
    raise ValueError("Missing authentication")


class AuthInterceptor:
    """Server-side Connect RPC interceptor that enforces bearer token auth.

    Reads the Authorization header (or session cookie), verifies the JWT via
    the configured verifier, and stores the VerifiedIdentity in a ContextVar
    for the service layer to read via get_verified_identity().
    """

    def __init__(self, verifier: TokenVerifier):
        self._verifier = verifier

    def _verify_or_raise(self, ctx) -> "VerifiedIdentity":
        token = extract_bearer_token(ctx.request_headers())
        if not token:
            raise ConnectError(Code.UNAUTHENTICATED, "Missing or malformed Authorization header")
        try:
            return self._verifier.verify(token)
        except ValueError as exc:
            logger.warning("Authentication failed: %s", exc)
            raise ConnectError(Code.UNAUTHENTICATED, "Authentication failed") from exc

    def intercept_unary_sync(self, call_next, request, ctx):
        identity = self._verify_or_raise(ctx)
        reset_token = _verified_identity.set(identity)
        try:
            return call_next(request, ctx)
        finally:
            _verified_identity.reset(reset_token)

    async def intercept_unary(self, call_next, request, ctx):
        # Token verification is pure crypto (HMAC-SHA256 for JWTs); safe to
        # run inline on the loop. ContextVar bookkeeping mirrors the sync
        # path so service handlers see the same identity regardless of
        # which dispatch surface they came in through.
        identity = self._verify_or_raise(ctx)
        reset_token = _verified_identity.set(identity)
        try:
            return await call_next(request, ctx)
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

    def _resolve_identity(self, ctx) -> "VerifiedIdentity":
        identity = self._default_identity
        if self._verifier is not None:
            token = extract_bearer_token(ctx.request_headers())
            if token:
                try:
                    identity = self._verifier.verify(token)
                except ValueError:
                    pass
        return identity

    def intercept_unary_sync(self, call_next, request, ctx):
        reset_token = _verified_identity.set(self._resolve_identity(ctx))
        try:
            return call_next(request, ctx)
        finally:
            _verified_identity.reset(reset_token)

    async def intercept_unary(self, call_next, request, ctx):
        reset_token = _verified_identity.set(self._resolve_identity(ctx))
        try:
            return await call_next(request, ctx)
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


class ProxyAuthTokenInjector:
    """Client-side interceptor that attaches the IAP OIDC ID token.

    The token is set on ``Proxy-Authorization`` (not ``Authorization``), which
    IAP consumes at the ingress, leaving ``Authorization`` free for the Iris JWT.
    """

    def __init__(self, provider: "TokenProvider"):
        self._provider = provider

    def intercept_unary_sync(self, call_next, request, ctx):
        token = self._provider.get_token()
        if token:
            ctx.request_headers()["proxy-authorization"] = f"Bearer {token}"
        return call_next(request, ctx)


def client_interceptors(
    token_provider: "TokenProvider | None",
    iap_provider: "TokenProvider | None" = None,
) -> list:
    """Build the client-side RPC interceptor chain.

    With a token provider, attach the Iris bearer token (``Authorization``).
    Without one (the SSH-tunnel case), send nothing: a loopback-trust controller
    authenticates the connection by its transport peer, and job ownership comes
    from the job name.

    With an ``iap_provider`` (an IAP-fronted cluster), additionally attach the
    OIDC ID token in ``Proxy-Authorization`` so the request passes IAP.
    """
    interceptors: list = []
    if token_provider is not None:
        interceptors.append(AuthTokenInjector(token_provider))
    if iap_provider is not None:
        interceptors.append(ProxyAuthTokenInjector(iap_provider))
    return interceptors


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


# OAuth scopes for the IAP login flow. "openid" is what makes the token endpoint
# return an OIDC ID token (the credential IAP requires); "email" puts the user's
# address in the token so the controller can attribute the identity.
IAP_LOGIN_SCOPES = ["openid", "email"]
_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
_GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"


class IapUserIdTokenProvider:
    """Returns a fresh OIDC ID token for IAP from a cached desktop-OAuth refresh token.

    IAP requires an ID token (not an access token); this re-mints it from the
    user's long-lived refresh token with no browser prompt. Obtain the initial
    refresh token once via :func:`run_iap_desktop_login`.
    """

    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self._creds = google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
            token_uri=_GOOGLE_TOKEN_URI,
            scopes=IAP_LOGIN_SCOPES,
        )

    def get_token(self) -> str | None:
        # creds.valid is False until the first refresh and once the access token
        # (minted alongside the ID token) expires; refreshing repopulates both.
        if self._creds.id_token is None or not self._creds.valid:
            self._creds.refresh(google.auth.transport.requests.Request())
        return self._creds.id_token


def run_iap_desktop_login(client_id: str, client_secret: str, *, port: int = 0) -> tuple[str, str]:
    """Run the installed-app OAuth flow in a browser and return (id_token, refresh_token).

    Opens the system browser for Google sign-in/consent and catches the redirect
    on a localhost port. Returns the freshly minted OIDC ID token and the
    long-lived refresh token to cache for silent re-minting.
    """
    # Lazy import: google-auth-oauthlib pulls in requests-oauthlib and is only
    # needed for the interactive login path, never by the controller or workers.
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: PLC0415  # optional dep: iris[iap]
    except ImportError as exc:
        raise RuntimeError(
            "IAP login requires google-auth-oauthlib; install it with `pip install marin-iris[iap]`"
        ) from exc

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": _GOOGLE_AUTH_URI,
            "token_uri": _GOOGLE_TOKEN_URI,
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, scopes=IAP_LOGIN_SCOPES)
    creds = flow.run_local_server(port=port, open_browser=True)
    if not creds.id_token:
        raise RuntimeError("OAuth flow returned no ID token (the 'openid' scope must be granted)")
    if not creds.refresh_token:
        raise RuntimeError("OAuth flow returned no refresh token (request offline access)")
    # google-auth types these as object; the guards above prove they are non-empty strings.
    return cast(str, creds.id_token), cast(str, creds.refresh_token)
