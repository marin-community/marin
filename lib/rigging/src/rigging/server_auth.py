# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Server-side authentication: verify a bearer token, bind identity, enforce a policy.

The companion to ``rigging.auth`` (which *attaches* a token on the client). This
module *verifies* one on the server and binds the resulting identity for the
request: the Google-credential verifiers (GCP access token, IAP OIDC ID token,
IAP signed-header assertion), a static-token verifier, the authenticator chain
that resolves a request to an identity, and the enforcement points a service
mounts unconditionally — ``PolicyAuthInterceptor`` for Connect RPCs and
``RouteAuthMiddleware`` for HTTP routes annotated ``@public`` / ``@requires_auth``.

Every auth mode is expressed as a chain, never as a conditional at the mount
point: an enforcing chain ends in loopback trust (``RequestAuthPolicy.enforcing``),
a null-auth chain ends in an anonymous-admin terminal
(``RequestAuthPolicy.permissive``), and the enforcement points behave correctly
under either.

It carries no service-specific policy — no token *minting*, no role semantics, no
RBAC. A service supplies those: it injects its own ``TokenVerifier`` (e.g. one
that checks JWTs it signed) and a role resolver, reads the bound identity via
``get_verified_identity``, and authorizes against its own policy.
"""

import contextlib
import ipaddress
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from http.cookies import SimpleCookie
from typing import Protocol

import google.auth.transport.requests
import google.oauth2.id_token
import requests
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.auth.exceptions import GoogleAuthError
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Match, Mount, Route
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    """Identity of an authenticated caller, extracted from JWT claims."""

    user_id: str
    role: str
    # Set only for a token scoped to a single proxy audience (e.g. one
    # endpoint's /proxy path). A scoped identity MUST NOT authorize any RPC —
    # the consuming service enforces that. None ⇒ a full identity (the default
    # for every non-scoped token and every non-JWT authenticator).
    audience: str | None = None


# Identity granted to a credentialless caller admitted by ambient trust:
# loopback / trusted-CIDR network location, or a permissive (null-auth) chain.
# Jobs are still attributed per-user via the job name's owner segment.
ANONYMOUS_ADMIN = VerifiedIdentity(user_id="anonymous", role="admin")


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


def extract_bearer_token(headers: dict, *, cookie_name: str | None = None) -> str | None:
    """Extract a bearer token from the ``Authorization`` header or a named cookie.

    The header wins; the cookie fallback is consulted only when ``cookie_name`` is
    given (a browser session a service chooses to honour for its dashboard).
    """
    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[len("Bearer ") :]
    if cookie_name is None:
        return None
    cookie_header = headers.get("cookie", "")
    return _extract_cookie(cookie_header, cookie_name)


# Per-request identity set by PolicyAuthInterceptor, read by service handlers.
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

    Mirrors the ContextVar bookkeeping PolicyAuthInterceptor performs per RPC so
    code outside the interceptor (e.g. a separate RPC dispatch surface) can
    establish the same identity for service handlers reached via
    get_verified_identity().
    """
    reset_token = _verified_identity.set(identity)
    try:
        yield
    finally:
        _verified_identity.reset(reset_token)


def require_identity() -> VerifiedIdentity:
    """Get the verified identity for the current RPC or raise UNAUTHENTICATED."""
    identity = _verified_identity.get()
    if identity is None:
        raise ConnectError(Code.UNAUTHENTICATED, "Authentication required")
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
    claims). Used as the login identity proof for an IAP-fronted service;
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
    trusted without a service JWT — an internal caller that bypasses the load
    balancer cannot forge it.

    The verified email is mapped to a role by the injected ``role_resolver`` (the
    service owns role semantics — e.g. look the email up in a user store, falling
    back to a read-only role for an unprovisioned caller).
    """

    def __init__(self, audience: str, role_resolver: Callable[[str], str]):
        if not audience:
            raise ValueError("IapAssertionVerifier requires a signed-header audience")
        self._audience = audience
        self._role_resolver = role_resolver
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
        return VerifiedIdentity(user_id=email, role=self._role_resolver(email))


def _direct_peer_ip(client_address: str | None, headers: dict) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Return the transport-peer IP of a genuine direct connection, else None.

    A connection counts as direct iff its transport peer parses as ``ip:port``
    with a nonzero port *and* the request carries no ``X-Forwarded-For``
    header. This is the shared trust gate for every network-location
    authenticator (loopback, trusted CIDR): identity may only ever be granted
    to the *socket peer*, never to a client-supplied header.

    The two conditions are individually sufficient and kept together as
    defence in depth. A uvicorn-fronted service configured with
    ``forwarded_allow_ips="*"`` rewrites ``scope["client"]`` to the
    attacker-controllable leftmost ``X-Forwarded-For`` entry and zeroes the port
    when the client is derived from a forwarded header (it cannot recover the
    forwarded client's port). A public request spoofing
    ``X-Forwarded-For: 127.0.0.1`` therefore presents as ``("127.0.0.1", 0)``
    with the header present — rejected on both counts. And any legitimate
    proxy hop (Traefik, GCLB) appends ``X-Forwarded-For``, so traffic it
    forwards can never borrow the hop's own network location.
    """
    if not client_address:
        return None
    if headers.get("x-forwarded-for"):
        return None
    host, _, port = client_address.rpartition(":")
    if not host or not port:
        return None
    try:
        if int(port) == 0:
            return None
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def is_trusted_loopback(client_address: str | None, headers: dict) -> bool:
    """Return True if the request arrived over a genuine loopback connection.

    A connection is trusted-loopback iff it is a direct connection (see
    :func:`_direct_peer_ip` for the forwarded-header trust model) whose peer
    is a loopback address (127.0.0.0/8 or ::1). Only a direct transport peer
    on the loopback interface (SSH tunnel / on-host) passes.
    """
    peer = _direct_peer_ip(client_address, headers)
    return peer is not None and peer.is_loopback


@dataclass(frozen=True, slots=True)
class AuthRequest:
    """Facts passed to each :class:`RequestAuthenticator`.

    ``token``: bearer token from ``Authorization`` / session cookie.
    ``headers``: raw request headers (IAP assertion verifier reads the signed header).
    ``client_address``: transport peer (loopback authenticator reads this).
    """

    token: str | None
    headers: dict
    client_address: str | None = None


class AuthDecision(StrEnum):
    """Outcome of a single authenticator over a request."""

    AUTHENTICATED = "authenticated"  # this authenticator owns the request
    ABSENT = "absent"  # its credential is not present — try the next authenticator
    REJECTED = "rejected"  # credential present but invalid — stop and reject the request


@dataclass(frozen=True, slots=True)
class AuthOutcome:
    decision: AuthDecision
    identity: VerifiedIdentity | None = None
    reason: str = ""


def _authenticated(identity: VerifiedIdentity) -> AuthOutcome:
    return AuthOutcome(AuthDecision.AUTHENTICATED, identity=identity)


_ABSENT = AuthOutcome(AuthDecision.ABSENT)


def _rejected(reason: str) -> AuthOutcome:
    return AuthOutcome(AuthDecision.REJECTED, reason=reason)


class RequestAuthenticator(Protocol):
    """Decides whether a request is authenticated by one identity source.

    Returns ``AUTHENTICATED`` (this source owns the request), ``ABSENT`` (its
    credential is not present — fall through to the next), or ``REJECTED`` (a
    credential is present but invalid — the request must be rejected, never
    downgraded to a weaker source).
    """

    def authenticate(self, request: AuthRequest) -> AuthOutcome: ...


@dataclass(frozen=True)
class JwtAuthenticator:
    """Authenticates a request bearing an ``Authorization`` token via a verifier.

    A present-but-invalid token is ``REJECTED`` (never falls through), preserving
    the rule that a bad credential cannot be downgraded to ambient trust.
    """

    verifier: TokenVerifier

    def authenticate(self, request: AuthRequest) -> AuthOutcome:
        if request.token is None:
            return _ABSENT
        try:
            return _authenticated(self.verifier.verify(request.token))
        except ValueError as exc:
            return _rejected(str(exc))


@dataclass(frozen=True)
class BestEffortJwtAuthenticator:
    """Attributes identity from a valid bearer token but never rejects.

    The null-auth head of a permissive chain: a valid token (e.g. a worker JWT)
    attributes the caller; an invalid or stale one falls through to the
    anonymous-admin terminal instead of failing the request.
    """

    verifier: TokenVerifier

    def authenticate(self, request: AuthRequest) -> AuthOutcome:
        if request.token is None:
            return _ABSENT
        try:
            return _authenticated(self.verifier.verify(request.token))
        except ValueError:
            return _ABSENT


class AnonymousAuthenticator:
    """Terminal authenticator that admits any request as the anonymous admin.

    Placed last it makes the whole chain permissive (null-auth, or an
    enforcing chain with ``optional=True``); earlier authenticators still
    attribute identity and — outside ``BestEffortJwtAuthenticator`` — still
    reject a presented-but-invalid credential.
    """

    def authenticate(self, request: AuthRequest) -> AuthOutcome:
        return _authenticated(ANONYMOUS_ADMIN)


@dataclass(frozen=True)
class IapAssertionAuthenticator:
    """Authenticates a tokenless request via IAP's signed-header assertion.

    Absent assertion → ``ABSENT``; a present-but-forged assertion → ``REJECTED``.
    """

    verifier: "IapAssertionVerifier"

    def authenticate(self, request: AuthRequest) -> AuthOutcome:
        try:
            identity = self.verifier.identity_from_headers(request.headers)
        except ValueError as exc:
            return _rejected(str(exc))
        return _authenticated(identity) if identity is not None else _ABSENT


class LoopbackAuthenticator:
    """Trusts a genuine loopback connection (SSH tunnel / on-host) as admin.

    A tokenless/assertionless fallback only — see :func:`is_trusted_loopback`.
    """

    def authenticate(self, request: AuthRequest) -> AuthOutcome:
        if is_trusted_loopback(request.client_address, request.headers):
            return _authenticated(ANONYMOUS_ADMIN)
        return _ABSENT


class CidrAuthenticator:
    """Trusts a direct transport peer inside a configured CIDR as admin.

    Network-location trust for a service reachable only over a private network
    (a VPC, a k8s pod network): a tokenless caller whose socket peer falls
    inside one of ``cidrs`` authenticates as :data:`ANONYMOUS_ADMIN` — the same
    convention as :class:`LoopbackAuthenticator`.

    Trust model: the CIDR check runs against the *transport peer address*
    (``AuthRequest.client_address``) and never against ``X-Forwarded-For`` or
    any other client-supplied header. Any request that carries
    ``X-Forwarded-For`` is refused CIDR trust outright (see
    :func:`_direct_peer_ip`): a forwarded request's peer is the proxy hop, so
    an in-network ingress (Traefik, a load balancer) would otherwise lend its
    own address to every anonymous internet request it forwards — and under
    uvicorn's ``forwarded_allow_ips="*"`` the scope client is itself rewritten
    from the spoofable header. Configure only ranges where holding an address
    implies operator-level trust; never an ingress hop's source ranges.
    """

    def __init__(self, cidrs: Sequence[str]):
        if not cidrs:
            raise ValueError("CidrAuthenticator requires at least one CIDR")
        # ip_network raises ValueError on a malformed CIDR (or one with host
        # bits set) — fail at construction, never silently at request time.
        self._networks = tuple(ipaddress.ip_network(cidr) for cidr in cidrs)

    def authenticate(self, request: AuthRequest) -> AuthOutcome:
        peer = _direct_peer_ip(request.client_address, request.headers)
        if peer is None:
            return _ABSENT
        # __contains__ is False (not an error) on an IPv4 peer vs IPv6 network
        # and vice versa, so mixed-family CIDR lists are fine.
        if any(peer in network for network in self._networks):
            return _authenticated(ANONYMOUS_ADMIN)
        return _ABSENT


def resolve_auth(
    request: AuthRequest,
    authenticators: Sequence[RequestAuthenticator],
) -> VerifiedIdentity:
    """Walk ``authenticators`` in order and resolve the request's identity.

    The first authenticator to return ``AUTHENTICATED`` wins; the first to return
    ``REJECTED`` stops the walk and raises (a present-but-invalid credential is
    never downgraded to a weaker source). Raises ``ValueError`` when every
    authenticator is ``ABSENT`` — a chain that admits anonymous callers ends in
    :class:`AnonymousAuthenticator` instead.
    """
    for authenticator in authenticators:
        outcome = authenticator.authenticate(request)
        if outcome.decision is AuthDecision.AUTHENTICATED:
            assert outcome.identity is not None
            return outcome.identity
        if outcome.decision is AuthDecision.REJECTED:
            raise ValueError(outcome.reason or "Authentication failed")
    raise ValueError("Missing authentication")


@dataclass(frozen=True)
class RequestAuthPolicy:
    """Server-side auth policy: an ordered authenticator chain plus a fallback verifier.

    The chain fully determines the outcome for every request, so a service
    mounts its enforcement points (:class:`PolicyAuthInterceptor`,
    :class:`RouteAuthMiddleware`) unconditionally and never branches on an
    "is auth on" flag. Build with :meth:`enforcing` or :meth:`permissive`;
    the zero-arg default is the permissive (allow-everyone) chain.

    ``verifier`` backs the token authenticator at the head of the chain and is
    also exposed for out-of-band token checks (e.g. a session-cookie exchange).
    """

    authenticators: tuple[RequestAuthenticator, ...] = (AnonymousAuthenticator(),)
    verifier: "TokenVerifier | None" = None

    @classmethod
    def enforcing(
        cls,
        *,
        verifier: "TokenVerifier | None" = None,
        iap_assertion_verifier: "IapAssertionVerifier | None" = None,
        trusted_cidrs: Sequence[str] = (),
        optional: bool = False,
    ) -> "RequestAuthPolicy":
        """Auth-enforced chain, highest-trust first: ``[Jwt?, IapAssertion?, Cidr?, Loopback]``.

        A presented service JWT wins; otherwise an IAP signed-header assertion;
        otherwise network-location trust (trusted CIDR, then loopback).
        ``optional`` appends the anonymous-admin terminal: a credentialless
        request is admitted, but a presented-and-invalid credential still rejects.
        """
        chain: list[RequestAuthenticator] = []
        if verifier is not None:
            chain.append(JwtAuthenticator(verifier))
        if iap_assertion_verifier is not None:
            chain.append(IapAssertionAuthenticator(iap_assertion_verifier))
        if trusted_cidrs:
            chain.append(CidrAuthenticator(trusted_cidrs))
        chain.append(LoopbackAuthenticator())
        if optional:
            chain.append(AnonymousAuthenticator())
        return cls(authenticators=tuple(chain), verifier=verifier)

    @classmethod
    def permissive(cls, *, verifier: "TokenVerifier | None" = None) -> "RequestAuthPolicy":
        """Null-auth chain: every request is admitted as the anonymous admin.

        A valid bearer token still attributes the caller (e.g. worker tokens);
        an invalid one is ignored rather than rejected.
        """
        chain: list[RequestAuthenticator] = []
        if verifier is not None:
            chain.append(BestEffortJwtAuthenticator(verifier))
        chain.append(AnonymousAuthenticator())
        return cls(authenticators=tuple(chain), verifier=verifier)

    @property
    def allows_anonymous(self) -> bool:
        """Whether a credentialless request is admitted (permissive or ``optional``)."""
        return isinstance(self.authenticators[-1], AnonymousAuthenticator)

    def resolve(
        self,
        token: str | None,
        *,
        client_address: str | None = None,
        headers: dict | None = None,
    ) -> VerifiedIdentity:
        """Resolve a request's identity under this policy (see :func:`resolve_auth`)."""
        return resolve_auth(
            AuthRequest(token=token, headers=headers or {}, client_address=client_address),
            self.authenticators,
        )


class PolicyAuthInterceptor:
    """Connect RPC interceptor that resolves every RPC through a :class:`RequestAuthPolicy`.

    After authentication the optional ``authorize`` hook runs the service's
    RBAC (e.g. deny endpoint-scoped tokens, restrict read-only roles) before
    the identity is bound for handlers via :func:`get_verified_identity`.
    ``unauthenticated_methods`` (e.g. a Login RPC) bypass the policy entirely.
    """

    def __init__(
        self,
        policy: RequestAuthPolicy,
        *,
        cookie_name: str | None = None,
        unauthenticated_methods: frozenset[str] = frozenset(),
        authorize: Callable[[VerifiedIdentity, str], None] | None = None,
    ):
        self._policy = policy
        self._cookie_name = cookie_name
        self._unauthenticated_methods = unauthenticated_methods
        self._authorize = authorize

    def _resolve_or_raise(self, ctx) -> VerifiedIdentity:
        headers = ctx.request_headers()
        token = extract_bearer_token(headers, cookie_name=self._cookie_name)
        try:
            return self._policy.resolve(token, client_address=ctx.client_address(), headers=headers)
        except ValueError as exc:
            if token is None:
                raise ConnectError(Code.UNAUTHENTICATED, str(exc)) from exc
            logger.warning("Authentication failed: %s", exc)
            raise ConnectError(Code.UNAUTHENTICATED, "Authentication failed") from exc

    def intercept_unary_sync(self, call_next, request, ctx):
        if ctx.method().name in self._unauthenticated_methods:
            return call_next(request, ctx)
        identity = self._resolve_or_raise(ctx)
        if self._authorize is not None:
            self._authorize(identity, ctx.method().name)
        with identity_scope(identity):
            return call_next(request, ctx)

    async def intercept_unary(self, call_next, request, ctx):
        if ctx.method().name in self._unauthenticated_methods:
            return await call_next(request, ctx)
        identity = self._resolve_or_raise(ctx)
        if self._authorize is not None:
            self._authorize(identity, ctx.method().name)
        with identity_scope(identity):
            return await call_next(request, ctx)


# ---------------------------------------------------------------------------
# Route-scoped HTTP auth: @public / @requires_auth annotations + middleware
# ---------------------------------------------------------------------------

_ROUTE_AUTH_ATTR = "_rigging_route_auth"


class _RouteAuth(StrEnum):
    """A request's auth disposition: the matched route's annotation, or a default."""

    PUBLIC = "public"
    REQUIRES_AUTH = "requires_auth"
    SKIP = "skip"  # mounts and unmatched paths — the inner app enforces its own auth / 404s
    DENY = "deny"  # unannotated route: fail closed


def public(fn: Callable) -> Callable:
    """Mark a route handler as publicly accessible (no auth required)."""
    setattr(fn, _ROUTE_AUTH_ATTR, _RouteAuth.PUBLIC)
    return fn


def requires_auth(fn: Callable) -> Callable:
    """Mark a route handler as requiring an authenticated identity."""
    setattr(fn, _ROUTE_AUTH_ATTR, _RouteAuth.REQUIRES_AUTH)
    return fn


def scope_headers(scope: Scope) -> dict[str, str]:
    """Lowercase header dict from an ASGI scope."""
    return {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}


def scope_client_address(scope: Scope) -> str | None:
    """Return the transport peer as ``host:port``, or None.

    This is uvicorn's ``scope["client"]`` — the genuine peer for a direct
    connection, or a forwarded value (port 0) when derived from
    ``X-Forwarded-For``. :func:`_direct_peer_ip` relies on that distinction.
    """
    client = scope.get("client")
    if not client:
        return None
    return f"{client[0]}:{client[1]}"


class RouteAuthMiddleware:
    """ASGI middleware that enforces per-route auth annotations against a policy.

    Looks up the matched Starlette route's handler and applies its ``@public``
    / ``@requires_auth`` annotation; an unannotated route is denied, so a new
    route must declare its auth posture. Mounts pass through — a mounted app
    (an RPC surface, static files) enforces its own auth.

    A scoped identity (``audience`` set) never passes ``@requires_auth``: such
    a token is valid only at the surface that checks its audience.
    """

    def __init__(self, app: Starlette, policy: RequestAuthPolicy, *, cookie_name: str | None = None):
        self._app = app
        self._policy = policy
        self._cookie_name = cookie_name
        self._router = app.router

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self._app(scope, receive, send)

        annotation = self._route_annotation(scope)
        if annotation in (_RouteAuth.PUBLIC, _RouteAuth.SKIP):
            return await self._app(scope, receive, send)

        if annotation is _RouteAuth.REQUIRES_AUTH:
            deny = self._authenticate(scope)
            if deny is not None:
                return await deny(scope, receive, send)
            return await self._app(scope, receive, send)

        response = JSONResponse({"error": "authentication required"}, status_code=401)
        return await response(scope, receive, send)

    def _route_annotation(self, scope: Scope) -> _RouteAuth:
        for route in self._router.routes:
            if isinstance(route, Mount):
                if route.matches(scope)[0] != Match.NONE:
                    return _RouteAuth.SKIP
                continue
            if isinstance(route, Route):
                match_result, _ = route.matches(scope)
                if match_result == Match.FULL:
                    return getattr(route.endpoint, _ROUTE_AUTH_ATTR, _RouteAuth.DENY)
        # No route matched — let the app 404.
        return _RouteAuth.SKIP

    def _authenticate(self, scope: Scope) -> JSONResponse | None:
        headers = scope_headers(scope)
        token = extract_bearer_token(headers, cookie_name=self._cookie_name)
        try:
            identity = self._policy.resolve(token, client_address=scope_client_address(scope), headers=headers)
        except ValueError:
            return JSONResponse({"error": "authentication required"}, status_code=401)
        if identity.audience is not None:
            return JSONResponse({"error": "endpoint-scoped token cannot access this route"}, status_code=403)
        return None
