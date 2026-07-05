# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Auth setup for the controller — verifier creation and JWT key management.

All service tokens are asymmetric EdDSA (Ed25519) JWTs. The controller is its
own signing authority: it mints with a per-cluster private key (sourced from a
``SecretSpec``, never stored in the DB) and verifies against the matching public
key. :class:`JwtTokenManager` is a thin *policy* wrapper over
:mod:`rigging.token_authority` — it owns role/claim semantics and the per-plane
audience discipline. Verification is fully stateless: a pure crypto check plus
the audience/scope binding, with no database access at all. Tokens are never
revoked. Authorization is config-driven: admin grants live in cluster config
(``auth.admin_users``) and are reconciled into the user store on every controller
start (see :func:`_reconcile_admin_grants`) — grants added and, crucially,
de-listed users downgraded. Deprovisioning is therefore edit-config-and-reload;
a session token already minted with the old role is not revocable but ages out at
its short :data:`SESSION_TOKEN_TTL_SECONDS`, so access is lost within that window.

Per-plane audience discipline (RFC 8725) is the load-bearing security invariant:
every minted token names exactly one ``aud`` (plane), and the control-plane
verifier *requires* its ``aud`` to be one of :data:`CONTROL_PLANE_AUDIENCES`. A
delegation (``aud="finelog"``) token — or any other foreign-plane audience —
replayed at this controller's RPC surface is therefore rejected by the verifier
before any policy runs.
"""

import dataclasses
import logging
import secrets
import time
from collections.abc import Callable, Sequence

from rigging.server_auth import (
    GcpAccessTokenVerifier,
    IapAssertionVerifier,
    IapIdTokenVerifier,
    RequestAuthPolicy,
    TokenVerifier,
    VerifiedIdentity,
)
from rigging.timing import Timestamp
from rigging.token_authority import (
    JwksVerifier,
    JwtSigner,
    generate_ed25519_keypair,
    signing_key_from_private_pem,
)

from iris.cluster.config import AuthConfig
from iris.cluster.controller import reads, writes
from iris.cluster.controller.db import ControllerDB

logger = logging.getLogger(__name__)

WORKER_USER = "system:worker"

# User/admin login sessions (aud="iris"). Short-lived: deprovisioning is bounded
# by this TTL — a downgraded/removed user keeps access only until it expires, and
# there is no refresh flow, so the client re-runs `iris login` to obtain a fresh
# token (picking up the current store role).
SESSION_TOKEN_TTL_SECONDS = 3600  # 1 hour
# Worker machine identity (aud="iris", role="worker"). This is a SHARED,
# cluster-lived credential: one token is minted per controller start and injected
# into every worker, with no refresh path, so it must outlive any single job. It
# is not revocable. KNOWN RISK (accepted for now, hardening tracked as follow-up):
# a leaked worker token lets an attacker register a rogue worker — and thus be
# dispatched tasks — fleet-wide until it expires; the only kill switch is rotating
# the cluster signing key, which re-auths every worker. Proper fixes (per-worker
# short-lived tokens, or a worker-credential rotation lever) are in the auth design
# doc's follow-ups.
WORKER_TOKEN_TTL_SECONDS = 86400 * 30  # 30 days

# Provider name when trusted_cidrs alone enables auth. No `iris login` flow:
# in-network callers get identity by location, everything else needs a token.
CIDR_PROVIDER = "cidr"

# ---------------------------------------------------------------------------
# Per-plane audience discipline (RFC 8725). Each ``aud`` names exactly one
# recipient *plane* — a bounded, static set — never a per-resource value.
# ---------------------------------------------------------------------------
# Control-plane user/worker tokens.
CONTROL_PLANE_AUDIENCE = "iris"
# Endpoint/`/proxy` tokens: a FIXED plane value (NOT the endpoint name). The
# specific endpoint rides in the ``endpoint`` claim the /proxy gate matches.
PROXY_PLANE_AUDIENCE = "iris-proxy"
# Delegation tokens the relay presents to a shared finelog.
FINELOG_AUDIENCE = "finelog"
# The control-plane verifier's fixed allowed-audience set. Endpoint names are
# dynamic and cannot be enumerated, so binding to the endpoint name moves to the
# ``endpoint`` claim; this set still rejects a replayed finelog / peer token.
CONTROL_PLANE_AUDIENCES = frozenset({CONTROL_PLANE_AUDIENCE, PROXY_PLANE_AUDIENCE})

# Issuer used when a cluster carries no ``name``. The issuer only needs to agree
# between this controller's own signer and verifier (control-plane tokens are
# verified only by the issuing controller), so an unnamed dev/local cluster is
# internally consistent under this fallback.
_DEFAULT_ISSUER = "iris"

# Role carried by an endpoint-scoped proxy token. It has zero RPC authority
# (authorize_method denies any audience-bearing identity); it exists only so the
# token has a role claim and so audit rows read sensibly.
ENDPOINT_TOKEN_ROLE = "endpoint"
# Scope claim marking a token as endpoint-scoped; verify() surfaces its bound
# endpoint as the identity's audience only when this scope is present.
ENDPOINT_TOKEN_SCOPE = "proxy"
# Role carried by a relay→finelog delegation token.
FINELOG_RELAY_ROLE = "finelog-relay"
# Role a de-listed admin is downgraded to when config reconciliation revokes their
# grant. The users CHECK constraint allows only admin/user/worker, and worker is
# reserved for the internal worker identity, so a revoked admin drops to "user".
DEPROVISIONED_ROLE = "user"
DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS = 3600  # 1 hour
MAX_ENDPOINT_TOKEN_TTL_SECONDS = 86400  # 24 hours


# ---------------------------------------------------------------------------
# JWT token manager
# ---------------------------------------------------------------------------


class JwtTokenManager:
    """Mints and verifies EdDSA service tokens — the iris *policy* over rigging.

    Wraps a :class:`rigging.token_authority.JwtSigner` (minting) and a
    control-plane :class:`rigging.token_authority.JwksVerifier` (verification,
    ``expected_audiences={"iris", "iris-proxy"}``). Every mint names exactly one
    plane's ``aud``; ``verify`` propagates the verifier's ``ValueError`` on a bad
    signature / expiry / unexpected audience, then applies the sole remaining iris
    policy — the aud↔scope binding and the endpoint-scope → identity-audience
    surfacing. Verification is fully stateless: it never touches a database and
    there is no revocation list; deprovisioning is bounded by the session TTL.
    """

    def __init__(
        self,
        signer: JwtSigner,
        verifier: JwksVerifier,
        *,
        previous_public_keys: Sequence[str] = (),
    ):
        self._signer = signer
        self._verifier = verifier
        # Retained *previous* public-key PEMs, served on JWKS during a rotation
        # overlap so verifiers accept tokens minted by the prior key.
        self._previous_public_keys: tuple[str, ...] = tuple(previous_public_keys)

    @property
    def signer(self) -> JwtSigner:
        """The signing authority. Public views only — never exposes the private key."""
        return self._signer

    def public_jwks(self) -> dict:
        """Public JWKS for ``/.well-known/jwks.json`` (current + retained-previous keys)."""
        return self._signer.public_jwks(also=self._previous_public_keys)

    def create_token(
        self,
        user_id: str,
        role: str,
        key_id: str,
        ttl_seconds: int,
    ) -> str:
        """Mint a control-plane (``aud="iris"``) user/worker token.

        ``ttl_seconds`` is required: a session token uses
        :data:`SESSION_TOKEN_TTL_SECONDS` and a worker token
        :data:`WORKER_TOKEN_TTL_SECONDS`; there is no default (an over-long token
        is not revocable, so the caller must pick the right lifetime).
        """
        return self._signer.mint(
            {"sub": user_id, "role": role, "jti": key_id},
            audience=CONTROL_PLANE_AUDIENCE,
            ttl_seconds=ttl_seconds,
        )

    def create_endpoint_token(
        self,
        endpoint_name: str,
        key_id: str,
        ttl_seconds: int = DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS,
    ) -> str:
        """Mint a scoped bearer token authorizing only ``endpoint_name``'s /proxy path.

        Carries the fixed proxy-plane ``aud="iris-proxy"``, ``scope="proxy"``, and
        an ``endpoint`` claim naming the wire name. ``verify`` surfaces that
        endpoint as the identity's audience; the /proxy gate matches it against
        the resolved route, and the RPC/HTTP auth arms treat any audience-bearing
        identity as endpoint-scoped (no RPC authority).
        """
        return self._signer.mint(
            {
                "sub": f"endpoint:{endpoint_name}",
                "role": ENDPOINT_TOKEN_ROLE,
                "scope": ENDPOINT_TOKEN_SCOPE,
                "endpoint": endpoint_name,
                "jti": key_id,
            },
            audience=PROXY_PLANE_AUDIENCE,
            ttl_seconds=ttl_seconds,
        )

    def create_delegation_token(self, subject: str, key_id: str, ttl_seconds: int) -> str:
        """Mint a relay→finelog delegation token (``aud="finelog"``, ``role="finelog-relay"``).

        Verified by a federated finelog against this controller's public key; its
        ``aud="finelog"`` is rejected by this controller's own control-plane
        verifier, so it can never be replayed at the RPC surface.
        """
        return self._signer.mint(
            {"sub": subject, "role": FINELOG_RELAY_ROLE, "jti": key_id},
            audience=FINELOG_AUDIENCE,
            ttl_seconds=ttl_seconds,
        )

    def verify(self, token: str) -> VerifiedIdentity:
        """Verify a control-plane token and apply the aud↔scope policy.

        The verifier raises ``ValueError`` on a bad signature / expiry / unknown
        issuer / an ``aud`` outside :data:`CONTROL_PLANE_AUDIENCES` (the
        cross-plane replay guard) — propagated unchanged. On success it applies
        the aud↔scope binding and surfaces an endpoint-scoped token's bound
        endpoint as the identity's audience. This is a pure function of the token:
        it performs NO database access and there is no revocation check.
        """
        claims = self._verifier.verify(token)

        # Bind the proxy plane to its scope so the per-plane discipline is enforced
        # by *audience*, not left to the scope claim alone: aud="iris-proxy" is ONLY
        # ever a well-formed endpoint token (scope="proxy" + endpoint claim), and a
        # control aud="iris" token must NOT carry a proxy scope. Reject either
        # mismatch — otherwise an aud="iris-proxy" token lacking scope would surface
        # as a full identity (a latent authz-escalation footgun).
        is_proxy_audience = claims.aud == PROXY_PLANE_AUDIENCE
        is_proxy_scope = claims.scope == ENDPOINT_TOKEN_SCOPE
        if is_proxy_audience != is_proxy_scope:
            raise ValueError("Token audience/scope mismatch (proxy plane requires proxy scope and vice versa)")
        endpoint = claims.claims.get("endpoint") if is_proxy_scope else None
        if is_proxy_scope and not endpoint:
            raise ValueError("Proxy-scoped token is missing its endpoint claim")

        return VerifiedIdentity(
            user_id=claims.sub,
            role=claims.claims.get("role", "user"),
            audience=endpoint,
        )


# ---------------------------------------------------------------------------
# Controller auth configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ControllerAuth:
    """Resolved auth configuration for a controller instance."""

    verifier: TokenVerifier | None = None
    provider: str | None = None
    worker_token: str | None = None
    login_verifier: TokenVerifier | None = None
    gcp_project_id: str | None = None
    jwt_manager: JwtTokenManager | None = None
    optional: bool = False
    # Verifies IAP's signed-header assertion to authenticate tokenless callers
    # behind IAP (only when an IAP signed_header_audience is set).
    iap_assertion_verifier: IapAssertionVerifier | None = None
    # Direct transport peers inside these CIDRs authenticate as anonymous
    # admin (network-location trust; forwarded requests never match).
    trusted_cidrs: tuple[str, ...] = ()


def request_auth_policy(auth: ControllerAuth | None) -> RequestAuthPolicy:
    """Build the request-auth policy the controller's surfaces apply.

    With no provider (null-auth) the chain is permissive — every request is
    admitted, but a worker JWT still attributes the caller.
    """
    if auth is None or auth.provider is None:
        return RequestAuthPolicy.permissive(verifier=auth.verifier if auth else None)
    return RequestAuthPolicy.enforcing(
        verifier=auth.verifier,
        iap_assertion_verifier=auth.iap_assertion_verifier,
        trusted_cidrs=auth.trusted_cidrs,
        optional=auth.optional,
    )


# How long a resolved IAP email->role mapping is cached before re-reading the
# user store. Roles change rarely (admin grants, new provisioning); a short TTL
# keeps the per-RPC assertion path off the database without making grants slow
# to take effect.
_IAP_ROLE_CACHE_TTL_SECONDS = 60.0


def _make_iap_role_resolver(db: ControllerDB, unprovisioned_role: str) -> Callable[[str], str]:
    """Return a function that maps a verified IAP email to its Iris role.

    Looks up the role from the user store; falls back to ``unprovisioned_role``
    for an email with no row. Results are cached for
    ``_IAP_ROLE_CACHE_TTL_SECONDS`` to keep the per-RPC assertion path off the
    database.
    """
    cache: dict[str, tuple[float, str]] = {}

    def resolve(email: str) -> str:
        cached = cache.get(email)
        if cached is not None and time.monotonic() < cached[0]:
            return cached[1]
        with db.read_snapshot() as tx:
            role = reads.get_user_role_or_none(tx, email)
        resolved = role if role is not None else unprovisioned_role
        # Atomic dict assignment; a benign race just recomputes the same value.
        cache[email] = (time.monotonic() + _IAP_ROLE_CACHE_TTL_SECONDS, resolved)
        return resolved

    return resolve


def _build_jwt_token_manager(
    *,
    cluster_name: str,
    signing_key_pem: str | None,
    previous_public_keys: Sequence[str],
) -> JwtTokenManager:
    """Construct the control-plane :class:`JwtTokenManager` for this controller.

    Loads the Ed25519 signing key from ``signing_key_pem``; when it is ``None``
    (no key configured — dev / null-auth), mints an EPHEMERAL in-process keypair,
    warning that tokens will not survive a restart. The verifier trusts exactly
    this controller's own public key under its own issuer, with the fixed
    control-plane audience set (the cross-plane replay guard).
    """
    issuer = cluster_name or _DEFAULT_ISSUER
    if signing_key_pem is not None:
        key = signing_key_from_private_pem(signing_key_pem)
    else:
        key = signing_key_from_private_pem(generate_ed25519_keypair().private_pem)
        logger.warning(
            "No auth.signing_key configured — minting an EPHEMERAL in-process Ed25519 keypair "
            "(issuer=%s). Tokens will NOT survive a controller restart; configure auth.signing_key "
            "for a persistent key.",
            issuer,
        )
    signer = JwtSigner(key, issuer=issuer)
    # Trust the current key plus any retained previous public keys, so control-plane
    # tokens minted under the prior key still verify during a rotation overlap (the
    # same set served on JWKS). Accepting an old key is fail-closed: it cannot mint,
    # only verify, and the overlap window is bounded by the token TTL.
    verifier = JwksVerifier(
        issuers={issuer: [key.public_pem, *previous_public_keys]},
        expected_audiences=CONTROL_PLANE_AUDIENCES,
    )
    return JwtTokenManager(signer, verifier, previous_public_keys=previous_public_keys)


def require_persistent_signing_key(auth_config: AuthConfig | None, signing_key_pem: str | None) -> None:
    """Fail fast if a *deployed* cluster with a login provider has no persistent key.

    A gcp/iap provider issues user JWTs and typically feeds finelog delegation, so
    its signing key must be stable: with an ephemeral one, every user is logged out
    on restart and an external verifier (finelog / peer) pinned to the published
    public key can no longer trust this controller — a silent trust-anchor break.
    Called at the serve entrypoint; the ephemeral fallback in
    :func:`create_controller_auth` remains fine for in-process dev (``LocalCluster``).

    CIDR-only trust is deliberately exempt: it is network-location trust (a dev /
    in-network posture, like null-auth) with no login provider, minting only
    internal worker tokens, so an ephemeral key is fine. (Federation / finelog-relay,
    if configured, also rely on a stable key — a separate concern from the provider.)
    """
    if auth_config is None or signing_key_pem is not None:
        return
    if auth_config.provider_kind() is None:
        return
    raise ValueError(
        f"a deployed {auth_config.provider_kind()} cluster requires a persistent auth.signing_key; "
        "run 'iris cluster init-keys' and set auth.signing_key to its reference "
        "(the ephemeral-key fallback is only for in-process dev / cidr / null-auth clusters)"
    )


def create_controller_auth(
    auth_config: AuthConfig | None,
    db: ControllerDB | None = None,
    *,
    cluster_name: str,
    signing_key_pem: str | None = None,
) -> ControllerAuth:
    """Build a ``ControllerAuth`` from the auth config.

    Mints EdDSA JWTs with this controller's per-cluster Ed25519 key (``iss`` =
    ``cluster_name``), loaded from ``signing_key_pem`` — resolved from a
    ``SecretSpec`` on the serve path, never stored in the DB. Verification is
    fully stateless (pure crypto + audience), so it never hits the DB. When
    ``signing_key_pem`` is ``None`` an ephemeral keypair is used — for in-process
    dev (``LocalCluster``) and null-auth; tokens do not survive a restart. A
    *deployed* authed cluster must supply a persistent key; that requirement is
    enforced at the serve entrypoint (``controller.main``), not here, so the
    in-process dev path can still run authed against an ephemeral key.

    A ``None`` config (or one with no provider selected and no trusted CIDRs)
    runs in null-auth mode. ``trusted_cidrs`` alone enables auth: identity by
    network location for direct in-network peers, tokens for everything else.
    """
    previous_public_keys = tuple(auth_config.previous_public_keys) if auth_config is not None else ()

    if auth_config is None or (auth_config.provider_kind() is None and not auth_config.trusted_cidrs):
        if db:
            now = Timestamp.now()
            with db.transaction() as _tx:
                writes.ensure_user(_tx, "anonymous", now, role="admin")
                writes.set_user_role(_tx, "anonymous", "admin")

            jwt_mgr = _build_jwt_token_manager(
                cluster_name=cluster_name,
                signing_key_pem=signing_key_pem,
                previous_public_keys=previous_public_keys,
            )

            worker_token = _create_worker_jwt(db, jwt_mgr, now)
            logger.info("Authentication disabled — null-auth mode (workers use JWT)")
            return ControllerAuth(verifier=jwt_mgr, worker_token=worker_token, jwt_manager=jwt_mgr)
        logger.info("Authentication disabled — null-auth mode, no DB")
        return ControllerAuth()

    provider = auth_config.provider_kind() or CIDR_PROVIDER
    now = Timestamp.now()

    jwt_mgr = _build_jwt_token_manager(
        cluster_name=cluster_name,
        signing_key_pem=signing_key_pem,
        previous_public_keys=previous_public_keys,
    )
    worker_token: str | None = None

    if db:
        worker_token = _create_worker_jwt(db, jwt_mgr, now)
        _reconcile_admin_grants(db, auth_config.admin_users, now)
        verifier: TokenVerifier | None = jwt_mgr
    else:
        worker_token = jwt_mgr.create_token(
            WORKER_USER, "worker", f"iris_k_worker_{secrets.token_hex(8)}", ttl_seconds=WORKER_TOKEN_TTL_SECONDS
        )
        verifier = None

    login_verifier: TokenVerifier | None = None
    gcp_project_id: str | None = None
    if provider == "gcp":
        gcp_project_id = auth_config.gcp.project_id
        if not gcp_project_id:
            raise ValueError("GCP auth config requires a project_id")
        login_verifier = GcpAccessTokenVerifier(project_id=gcp_project_id)

    # For IAP, `iris login` presents the OIDC ID token it obtained for the IAP
    # ingress; the controller verifies it (audience + signature) and mints a JWT.
    iap_assertion_verifier: IapAssertionVerifier | None = None
    if provider == "iap":
        audiences = list(auth_config.iap.audiences)
        if not audiences and not auth_config.iap.signed_header_audience:
            raise ValueError("IAP auth config requires audiences (login) and/or signed_header_audience (assertion)")
        # Assertion-only IAP (no desktop OAuth client registered) is valid:
        # browser users authenticate via the signed header; `iris login` then
        # reports UNIMPLEMENTED (no login verifier).
        if audiences:
            login_verifier = IapIdTokenVerifier(audiences)

        # When the signed-header audience is configured, a tokenless request that
        # carries a valid IAP assertion is authenticated as the asserted email,
        # resolved to its provisioned role (or read-only dashboard if not
        # provisioned). Without a DB the resolver defaults to dashboard.
        signed_header_audience = auth_config.iap.signed_header_audience
        if signed_header_audience:
            unprovisioned_role = auth_config.iap.unprovisioned_role
            role_resolver = (
                _make_iap_role_resolver(db, unprovisioned_role) if db else (lambda _email: unprovisioned_role)
            )
            iap_assertion_verifier = IapAssertionVerifier(signed_header_audience, role_resolver=role_resolver)

    optional = auth_config.optional
    # Only the CIDR *count* is logged: CodeQL's sensitive-data heuristics treat
    # any value read off auth_config as a potential secret, and the cluster
    # config file is the authoritative place to read the ranges anyway.
    logger.info(
        "Auth enabled: provider=%s, db=%s, jwt=%s, optional=%s, trusted_cidrs=%d (loopback always trusted as admin)",
        provider,
        "yes" if db else "no",
        "yes" if jwt_mgr else "no",
        optional,
        len(auth_config.trusted_cidrs),
    )
    return ControllerAuth(
        verifier=verifier,
        provider=provider,
        worker_token=worker_token,
        login_verifier=login_verifier,
        gcp_project_id=gcp_project_id,
        jwt_manager=jwt_mgr,
        optional=optional,
        iap_assertion_verifier=iap_assertion_verifier,
        trusted_cidrs=tuple(auth_config.trusted_cidrs),
    )


def _create_worker_jwt(db: ControllerDB, jwt_mgr: JwtTokenManager, now: Timestamp) -> str:
    """Generate a JWT for the worker identity on each controller start.

    A fresh ``jti`` is minted per start for log correlation only; it is never
    persisted or revocable. Old worker tokens simply age out at their TTL, so
    in-flight workers finish gracefully with their existing credentials.
    """
    key_id = f"iris_k_worker_{secrets.token_hex(8)}"
    with db.transaction() as _tx:
        writes.ensure_user(_tx, WORKER_USER, now, role="worker")
    jwt_token = jwt_mgr.create_token(WORKER_USER, "worker", key_id, ttl_seconds=WORKER_TOKEN_TTL_SECONDS)
    logger.info("New worker JWT generated (key_id=%s)", key_id)
    return jwt_token


def _reconcile_admin_grants(db: ControllerDB, admin_users: Sequence[str], now: Timestamp) -> None:
    """Make the config's ``admin_users`` the authoritative admin set.

    Cluster config is the source of truth for the admin grant, so on every
    controller start we both grant admin to each listed user AND downgrade any user
    still holding admin who is no longer listed. That downgrade is the deprovision
    path: drop a user from ``admin_users``, reload/restart the controller, and their
    stored admin role is revoked. A session token already minted with the old role
    is not revocable but ages out at its short TTL, so access is lost within one
    ``SESSION_TOKEN_TTL_SECONDS`` window. (``worker`` is left untouched — it is not
    an admin and belongs to the internal worker identity, not a config grant.)
    """
    granted = set(admin_users)
    with db.transaction() as tx:
        for user_id in granted:
            writes.ensure_user(tx, user_id, now)
            writes.set_user_role(tx, user_id, "admin")
        for user_id in reads.list_user_ids_with_role(tx, "admin"):
            if user_id not in granted:
                writes.set_user_role(tx, user_id, DEPROVISIONED_ROLE)
                logger.info("event=admin_revoked entity=%s trigger=config-reconcile", user_id)
