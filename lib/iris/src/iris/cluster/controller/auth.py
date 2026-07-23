# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Auth setup for the controller — verifier creation and JWT key management.

All service tokens are asymmetric EdDSA (Ed25519) JWTs. The controller is its
own signing authority: it mints with a per-cluster private key (sourced from a
``SecretSpec``, never stored in the DB) and verifies against the matching public
key. :class:`JwtTokenManager` is a thin policy wrapper over
:mod:`rigging.token_authority` — it owns role/claim semantics and the per-plane
audience discipline. Verification is fully stateless: a pure crypto check plus
the audience/scope binding, with no database access at all. Tokens are never
revoked. Authorization is config-driven and resolved entirely in memory: cluster
config is the sole source of truth for roles. A :class:`RolePolicy` — a frozen map
built from :class:`AuthConfig` at controller start (admins from ``auth.admin_users``,
a provider-derived default for everyone else) — answers ``role_for`` with no DB
projection and no reconciliation. IAP users hold no minted token: their role is
resolved per request from the verified assertion, so deprovisioning is
edit-config-and-reload (rebuild the map) and takes effect on the next request.

Per-plane audience discipline (RFC 8725) is the load-bearing security invariant:
every minted token names exactly one ``aud`` (plane), and the control-plane
verifier requires its ``aud`` to be one of :data:`CONTROL_PLANE_AUDIENCES`. A
foreign-plane audience replayed at this controller's RPC surface therefore never
becomes a control-plane identity. A federation (``aud="federation"``) token from a
trusted peer does authenticate, via the separate
:class:`FederationTokenVerifier` — as a ``federation-peer`` identity that
:func:`~iris.rpc.auth.authorize_method` admits only on the federation RPC subset.
"""

import dataclasses
import logging
import secrets
from collections.abc import Mapping, Sequence

from rigging.server_auth import (
    IapAssertionVerifier,
    RequestAuthPolicy,
    TokenVerifier,
    VerifiedIdentity,
)
from rigging.token_authority import (
    JwksVerifier,
    JwtSigner,
    generate_ed25519_keypair,
    signing_key_from_private_pem,
)

from iris.cluster.config import AuthConfig, PeerConfig
from iris.rpc.auth import FEDERATION_PEER_ROLE

logger = logging.getLogger(__name__)

WORKER_USER = "system:worker"

# Role for an authenticated non-admin on a gcp cluster (and the fallback default
# anywhere config carries no more specific rule). "user" is the ordinary
# job-submitting identity.
DEFAULT_USER_ROLE = "user"
# Role of the internal worker machine identity.
WORKER_ROLE = "worker"
# Role granted to a config-listed admin.
ADMIN_ROLE = "admin"

# TTL for the control-plane admin token LocalCluster mints in-process for its
# auto-login (aud="iris"). Short-lived and non-refreshable. Deployed clusters
# authenticate users via IAP and mint no user tokens, so this is dev-only.
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

# Provider name when trusted_cidrs alone enables auth: in-network callers get
# identity by location, everything else needs a token.
CIDR_PROVIDER = "cidr"

# ---------------------------------------------------------------------------
# Per-plane audience discipline (RFC 8725). Each ``aud`` names exactly one
# recipient plane — a bounded, static set — never a per-resource value.
# ---------------------------------------------------------------------------
# Control-plane user/worker tokens.
CONTROL_PLANE_AUDIENCE = "iris"
# Endpoint/`/proxy` tokens: a FIXED plane value (NOT the endpoint name). The
# specific endpoint rides in the ``endpoint`` claim the /proxy gate matches.
PROXY_PLANE_AUDIENCE = "iris-proxy"
# The control-plane verifier's fixed allowed-audience set. Endpoint names are
# dynamic and cannot be enumerated, so binding to the endpoint name moves to the
# ``endpoint`` claim; this set still rejects a replayed peer token.
CONTROL_PLANE_AUDIENCES = frozenset({CONTROL_PLANE_AUDIENCE, PROXY_PLANE_AUDIENCE})

# The control-plane issuer every cluster minted under before ``name`` existed.
#
# A control-plane token is verified only by the controller that minted it, so this
# string carries no information and an unnamed dev/local cluster stays internally
# consistent under it. It is also accepted, transitionally, by a *named* cluster's
# control-plane verifier: a worker token minted before that cluster gained its name
# carries this issuer and stays valid for WORKER_TOKEN_TTL_SECONDS. Both issuers
# resolve to this controller's own key, so accepting it admits exactly the tokens
# this controller itself minted — no widening.
_LEGACY_ISSUER = "iris"

# Role carried by an endpoint-scoped proxy token. It has zero RPC authority
# (authorize_method denies any audience-bearing identity); it exists only so the
# token has a role claim and so audit rows read sensibly.
ENDPOINT_TOKEN_ROLE = "endpoint"
# Scope claim marking a token as endpoint-scoped; verify() surfaces its bound
# endpoint as the identity's audience only when this scope is present.
ENDPOINT_TOKEN_SCOPE = "proxy"
DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS = 3600  # 1 hour
# Ceiling on a requested endpoint-token TTL. Set to a week so a long-running
# agentic datagen/eval job can hold one capability URL for its whole run instead
# of hitting 401s when a shorter token expires under a still-healthy endpoint. The
# token stays narrowly scoped — one endpoint, /proxy access only, zero RPC
# authority (see authorize_method) — so a week-long, non-revocable lifetime widens
# nothing: a leak exposes only that one endpoint until it ages out. Callers opt
# into a long TTL explicitly; the default stays short.
MAX_ENDPOINT_TOKEN_TTL_SECONDS = 86400 * 7  # 7 days

# Federation plane: the token a parent controller presents on RPCs to this cluster,
# verified against the parent's published key by a dedicated verifier. Kept OUT of
# CONTROL_PLANE_AUDIENCES so a federation bearer can never become a full RPC identity;
# authorize_method restricts FEDERATION_PEER_ROLE to the federation RPC subset.
FEDERATION_AUDIENCE = "federation"
# Short-lived and unrevocable: a fresh token is minted per outgoing RPC, so replay is
# bounded by the TTL plus the IP allowlist and the issuer/aud/requester binding.
FEDERATION_TOKEN_TTL_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# JWT token manager
# ---------------------------------------------------------------------------


class JwtTokenManager:
    """Mints and verifies EdDSA service tokens — the iris policy over rigging.

    Wraps a :class:`rigging.token_authority.JwtSigner` (minting) and a
    control-plane :class:`rigging.token_authority.JwksVerifier` (verification,
    ``expected_audiences={"iris", "iris-proxy"}``). Every mint names exactly one
    plane's ``aud``. It owns the sole iris-specific verification policy: the
    aud↔scope binding and the endpoint-scope → identity-audience surfacing (see
    :meth:`verify`).
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
        # Retained previous public-key PEMs, served on JWKS during a rotation
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

    def create_federation_token(
        self,
        requester_id: str,
        key_id: str,
        ttl_seconds: int = FEDERATION_TOKEN_TTL_SECONDS,
    ) -> str:
        """Mint the federation bearer this controller presents on outgoing peer RPCs.

        Carries ``aud="federation"``, ``role="federation-peer"``, and this cluster's id
        as ``sub``/``iss`` (the requester), which the peer verifies against this
        controller's published key. The ``aud`` sits outside every control-plane
        verifier's audience set, so the token cannot authenticate a general RPC.
        """
        return self._signer.mint(
            {"sub": requester_id, "role": FEDERATION_PEER_ROLE, "jti": key_id},
            audience=FEDERATION_AUDIENCE,
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
        # by audience, not left to the scope claim alone: aud="iris-proxy" is ONLY
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
# Federation trust — a dedicated verifier for inbound peer handoffs
# ---------------------------------------------------------------------------


class FederationTokenVerifier:
    """Verifies inbound federation tokens against the configured peer public keys.

    Issuers are the trusted peer clusters and the sole audience is ``"federation"``.
    Held separate from the control-plane verifier so a federation token cannot
    authenticate a general RPC; the ``federation-peer`` identity it yields is
    method-scoped by :func:`~iris.rpc.auth.authorize_method`.
    """

    def __init__(self, federation_peers: Mapping[str, str]):
        self._verifier = JwksVerifier(
            issuers={peer_id: [pem] for peer_id, pem in federation_peers.items()},
            expected_audiences=frozenset({FEDERATION_AUDIENCE}),
        )

    def verify(self, token: str) -> VerifiedIdentity:
        """Return a method-scoped federation-peer identity for a valid token.

        ``user_id`` is the verified requester (the peer cluster, from the token's
        ``iss``), so a peer cannot assert a requester id other than its own.
        """
        claims = self._verifier.verify(token)
        return VerifiedIdentity(user_id=claims.iss, role=FEDERATION_PEER_ROLE)


class FederationTokenProvider:
    """Mints this cluster's federation bearer for outgoing peer RPCs.

    A ``rigging.auth.TokenProvider``: each call mints a fresh short-lived
    ``aud="federation"`` token asserting this cluster as the requester, which the peer
    verifies against this cluster's published key.
    """

    def __init__(self, requester_id: str, jwt_manager: JwtTokenManager):
        self._requester_id = requester_id
        self._jwt_manager = jwt_manager

    def get_token(self) -> str | None:
        return self._jwt_manager.create_federation_token(self._requester_id, secrets.token_hex(8))


class _ControlPlaneOrFederationVerifier:
    """Routes a bearer to the control-plane verifier, falling back to federation.

    A control-plane token (``aud`` in ``{iris, iris-proxy}``) verifies via the JWT
    manager; a federation token (``aud="federation"``) is rejected there and verified
    by the federation verifier instead, yielding a method-scoped federation-peer
    identity. No token satisfies both audiences, so the fallback never crosses planes.
    """

    def __init__(self, control_plane: JwtTokenManager, federation: FederationTokenVerifier):
        self._control_plane = control_plane
        self._federation = federation

    def verify(self, token: str) -> VerifiedIdentity:
        try:
            return self._control_plane.verify(token)
        except ValueError:
            return self._federation.verify(token)


# ---------------------------------------------------------------------------
# Role policy — config-authoritative, resolved in memory
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RolePolicy:
    """The controller's config-derived role map, resolved entirely in memory.

    Built from :class:`AuthConfig` at controller start and carried on
    :class:`ControllerAuth`. Cluster config is the sole source of truth: there is
    no ``users`` table and no reconciliation, so deprovisioning is
    edit-config-and-reload (see module docstring).

    ``admins`` are the ``auth.admin_users`` entries; ``default_role`` is the role of
    an authenticated non-admin (the IAP ``unprovisioned_role`` for an iap provider,
    ``"user"`` otherwise). null-auth/cidr identities are assigned by the auth chain
    (anonymous/loopback admin) and never go through :meth:`role_for`.
    """

    admins: frozenset[str]
    default_role: str

    def role_for(self, user_id: str) -> str:
        if user_id == WORKER_USER:
            return WORKER_ROLE
        if user_id in self.admins:
            return ADMIN_ROLE
        return self.default_role


def _build_role_policy(auth_config: AuthConfig | None, provider: str | None) -> RolePolicy:
    """Build the in-memory :class:`RolePolicy` from ``auth_config``.

    ``default_role`` is the IAP ``unprovisioned_role`` for an iap provider and
    ``"user"`` otherwise (cidr, null-auth); the cidr/null-auth default is never
    consulted since those identities are assigned by the auth chain.
    """
    admins = frozenset(auth_config.admin_users) if auth_config is not None else frozenset()
    if provider == "iap" and auth_config is not None and auth_config.iap is not None:
        default_role = auth_config.iap.unprovisioned_role
    else:
        default_role = DEFAULT_USER_ROLE
    return RolePolicy(admins=admins, default_role=default_role)


# ---------------------------------------------------------------------------
# Controller auth configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ControllerAuth:
    """Resolved auth configuration for a controller instance."""

    verifier: TokenVerifier | None = None
    provider: str | None = None
    worker_token: str | None = None
    jwt_manager: JwtTokenManager | None = None
    optional: bool = False
    # Verifies IAP's signed-header assertion to authenticate tokenless callers
    # behind IAP (only when an IAP signed_header_audience is set).
    iap_assertion_verifier: IapAssertionVerifier | None = None
    # Direct transport peers inside these CIDRs authenticate as anonymous
    # admin (network-location trust; forwarded requests never match).
    trusted_cidrs: tuple[str, ...] = ()
    # Config-derived role map (admins + default role). The sole source of truth for
    # roles; rebuilt from config on each controller start.
    role_policy: RolePolicy | None = None
    # Which submitters this cluster admits via an inbound federation handoff, matched
    # against the proto's asserted submitting_user (allow-policy syntax). Empty admits
    # none (fail closed). The federation token itself only proves the requester; the
    # verifier that checks it is folded into ``verifier``.
    allowed_submitters: tuple[str, ...] = ()


def request_auth_policy(auth: ControllerAuth | None) -> RequestAuthPolicy:
    """Build the request-auth policy the controller's surfaces apply.

    Delegates to rigging's canonical stack builders (both compile the shared
    declarative :class:`~rigging.auth_config.AuthStackConfig`): null-auth (no
    provider) is the permissive chain — every request admitted, but a worker JWT
    still attributes the caller — while an IAP provider or cidr trust is the
    enforcing chain, highest-trust first (``[jwt?, iap_assertion?, cidr?,
    loopback]``, with an anonymous tail iff ``optional``). The controller's
    request verifier is always the JWT manager, so a cidr-only cluster still
    admits its worker JWTs presented from outside the trusted network.
    """
    if auth is None or auth.provider is None:
        return RequestAuthPolicy.permissive(verifier=auth.verifier if auth is not None else None)
    return RequestAuthPolicy.enforcing(
        verifier=auth.verifier,
        iap_assertion_verifier=auth.iap_assertion_verifier,
        trusted_cidrs=auth.trusted_cidrs,
        optional=auth.optional,
    )


def _build_jwt_token_manager(
    *,
    cluster_name: str,
    signing_key_pem: str | None,
    previous_public_keys: Sequence[str],
) -> JwtTokenManager:
    """Construct the control-plane :class:`JwtTokenManager` for this controller.

    Loads the Ed25519 signing key from ``signing_key_pem``; when it is ``None``
    (no key configured — dev / null-auth), mints an EPHEMERAL in-process keypair,
    warning that tokens will not survive a restart. Every token this controller mints
    is issued under ``cluster_name`` — the cluster's identity, which a federation peer
    keys its trust on. The verifier trusts this controller's own public key under that
    issuer and under :data:`_LEGACY_ISSUER`, with the fixed control-plane audience set
    (the cross-plane replay guard).
    """
    issuer = cluster_name or _LEGACY_ISSUER
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
    #
    # Both issuers map to the same key set. _LEGACY_ISSUER covers the worker tokens a
    # cluster minted before it carried a `name`, which outlive a restart by up to
    # WORKER_TOKEN_TTL_SECONDS; it never reaches the federation plane, whose verifier
    # keys each peer's own name. Drop it once no such token can still be unexpired.
    trusted_keys = [key.public_pem, *previous_public_keys]
    verifier = JwksVerifier(
        issuers={candidate: trusted_keys for candidate in (issuer, _LEGACY_ISSUER)},
        expected_audiences=CONTROL_PLANE_AUDIENCES,
    )
    return JwtTokenManager(signer, verifier, previous_public_keys=previous_public_keys)


def require_persistent_signing_key(peers: Mapping[str, PeerConfig], signing_key_pem: str | None) -> None:
    """Fail fast if a controller that calls federation peers has no persistent key.

    A federation token (``aud="federation"``) is the only token an external verifier pins
    to this controller's published public key: each peer's ``federation_peers`` map holds
    this cluster's public key to verify the bearer on an incoming ``FederationSync``. An
    ephemeral key rotates on every restart and breaks that trust anchor, so a controller
    with ``peers`` set must anchor a persistent ``auth.signing_key``. Inbound trust
    (``federation_peers``) imposes no such requirement: a cluster that only *receives*
    federated calls verifies with its peers' keys and signs nothing anyone else pins.

    The key also signs worker tokens and endpoint-scoped ``/proxy`` tokens, but the issuing
    controller verifies those itself, so an ephemeral key is fine for them — a restart just
    expires any outstanding proxy share-links early. Only a federation token is pinned by an
    external verifier, so only outgoing federation makes persistence a correctness
    requirement. IAP authenticates each user request and the controller mints no user tokens.
    So the ephemeral fallback in :func:`create_controller_auth` is fine for every cluster
    that hands off no jobs, including dev (``LocalCluster``).
    """
    if not peers or signing_key_pem is not None:
        return
    raise ValueError(
        "peers is set, so this controller calls federation peers with tokens they verify against "
        "this controller's published public key; that requires a persistent auth.signing_key. "
        "Run 'iris cluster init-keys --gcp-secret … --accessor <controller-sa>' and set "
        "auth.signing_key to the printed reference."
    )


def create_controller_auth(
    auth_config: AuthConfig | None,
    *,
    cluster_name: str,
    signing_key_pem: str | None = None,
) -> ControllerAuth:
    """Build a ``ControllerAuth`` from the auth config.

    Mints EdDSA JWTs with this controller's per-cluster Ed25519 key (``iss`` =
    ``cluster_name``), loaded from ``signing_key_pem`` — resolved from a
    ``SecretSpec`` on the serve path, never stored in the DB. When
    ``signing_key_pem`` is ``None`` an ephemeral keypair is used — for in-process
    dev (``LocalCluster``) and null-auth; tokens do not survive a restart. A
    deployed authed cluster must supply a persistent key; that requirement is
    enforced at the serve entrypoint (``controller.main``), not here, so the
    in-process dev path can still run authed against an ephemeral key.

    Roles come from an in-memory :class:`RolePolicy` built here from ``auth_config``.
    A ``None`` config (or one with no provider selected and no trusted CIDRs) runs in
    null-auth mode. ``trusted_cidrs`` alone enables auth: identity by network location
    for direct in-network peers, tokens for everything else.
    """
    previous_public_keys = tuple(auth_config.previous_public_keys) if auth_config is not None else ()
    jwt_mgr = _build_jwt_token_manager(
        cluster_name=cluster_name,
        signing_key_pem=signing_key_pem,
        previous_public_keys=previous_public_keys,
    )
    worker_token = _create_worker_jwt(jwt_mgr)

    # Inbound federation trust: a dedicated verifier over the configured peer keys.
    # When present, the request verifier accepts both control-plane tokens and (via
    # the composite) federation tokens; the federation token stays method-scoped.
    federation_peers = dict(auth_config.federation_peers) if auth_config is not None else {}
    federation_verifier = FederationTokenVerifier(federation_peers) if federation_peers else None
    allowed_submitters = tuple(auth_config.allowed_submitters) if auth_config is not None else ()
    request_verifier: TokenVerifier = (
        _ControlPlaneOrFederationVerifier(jwt_mgr, federation_verifier) if federation_verifier is not None else jwt_mgr
    )

    # Null-auth: no login-provider arm and no trusted CIDRs. The anonymous/loopback
    # admin identity is assigned by the permissive auth chain, not resolved here.
    if auth_config is None or (auth_config.provider_kind() is None and not auth_config.trusted_cidrs):
        logger.info("Authentication disabled — null-auth mode (workers use JWT)")
        return ControllerAuth(
            verifier=request_verifier,
            worker_token=worker_token,
            jwt_manager=jwt_mgr,
            role_policy=_build_role_policy(auth_config, None),
            allowed_submitters=allowed_submitters,
        )

    provider = auth_config.provider_kind() or CIDR_PROVIDER
    role_policy = _build_role_policy(auth_config, provider)

    # IAP is the sole login provider and never mints a controller token: the IAP
    # GCLB authenticates every user request at the edge and forwards a signed
    # assertion the controller verifies. A tokenless request carrying a valid IAP
    # assertion is authenticated as the asserted email, resolved to its role by the
    # in-memory RolePolicy (admins -> admin, everyone else -> the configured
    # unprovisioned_role). No DB, no cache.
    iap_assertion_verifier: IapAssertionVerifier | None = None
    if provider == "iap":
        signed_header_audience = auth_config.iap.signed_header_audience
        if not signed_header_audience:
            raise ValueError("IAP auth config requires signed_header_audience (the IAP assertion audience)")
        iap_assertion_verifier = IapAssertionVerifier(signed_header_audience, role_resolver=role_policy.role_for)

    optional = auth_config.optional
    # Only the CIDR count is logged: CodeQL's sensitive-data heuristics treat
    # any value read off auth_config as a potential secret, and the cluster
    # config file is the authoritative place to read the ranges anyway.
    logger.info(
        "Auth enabled: provider=%s, jwt=%s, optional=%s, trusted_cidrs=%d (loopback always trusted as admin)",
        provider,
        "yes" if jwt_mgr else "no",
        optional,
        len(auth_config.trusted_cidrs),
    )
    return ControllerAuth(
        verifier=request_verifier,
        provider=provider,
        worker_token=worker_token,
        jwt_manager=jwt_mgr,
        optional=optional,
        iap_assertion_verifier=iap_assertion_verifier,
        trusted_cidrs=tuple(auth_config.trusted_cidrs),
        role_policy=role_policy,
        allowed_submitters=allowed_submitters,
    )


def _create_worker_jwt(jwt_mgr: JwtTokenManager) -> str:
    """Mint the worker-identity JWT on each controller start.

    A fresh ``jti`` is minted per start for log correlation only; it is never
    persisted or revocable. The worker role is known from the :class:`RolePolicy`
    (``WORKER_USER`` -> ``worker``), so no DB row is created. Old worker tokens
    simply age out at their TTL, so in-flight workers finish gracefully with their
    existing credentials.
    """
    key_id = f"iris_k_worker_{secrets.token_hex(8)}"
    jwt_token = jwt_mgr.create_token(WORKER_USER, WORKER_ROLE, key_id, ttl_seconds=WORKER_TOKEN_TTL_SECONDS)
    logger.info("New worker JWT generated (key_id=%s)", key_id)
    return jwt_token
