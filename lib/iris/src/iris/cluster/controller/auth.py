# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Auth setup for the controller — verifier creation and JWT key management.

All tokens are JWTs signed with a persistent HMAC-SHA256 key stored in the
controller_secrets table. Verification is a pure crypto check plus an
in-memory revocation set — no per-RPC database hit.
"""

import dataclasses
import logging
import secrets
import time
from collections.abc import Callable, Sequence

import jwt
from rigging.server_auth import (
    GcpAccessTokenVerifier,
    IapAssertionVerifier,
    IapIdTokenVerifier,
    RequestAuthPolicy,
    StaticTokenVerifier,
    TokenVerifier,
    VerifiedIdentity,
)
from rigging.timing import Timestamp
from sqlalchemy import Row, delete, insert, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from iris.cluster.config import AuthConfig, StaticAuthConfig
from iris.cluster.controller import reads, writes
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import auth_api_keys_table, auth_controller_secrets_table

logger = logging.getLogger(__name__)

WORKER_USER = "system:worker"
DEFAULT_JWT_TTL_SECONDS = 86400 * 30  # 30 days

# Provider name when trusted_cidrs alone enables auth. No `iris login` flow:
# in-network callers get identity by location, everything else needs a token.
CIDR_PROVIDER = "cidr"

# Role carried by an endpoint-scoped proxy token. It has zero RPC authority
# (authorize_method denies any audience-bearing identity); it exists only so the
# token has a role claim and so audit rows read sensibly.
ENDPOINT_TOKEN_ROLE = "endpoint"
# Scope claim marking a token as endpoint-scoped; verify() surfaces its aud as
# the identity's audience only when this scope is present.
ENDPOINT_TOKEN_SCOPE = "proxy"
DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS = 3600  # 1 hour
MAX_ENDPOINT_TOKEN_TTL_SECONDS = 86400  # 24 hours


# ---------------------------------------------------------------------------
# API key CRUD — top-level functions operating on ControllerDB
# ---------------------------------------------------------------------------


def create_api_key(
    db: ControllerDB,
    key_id: str,
    key_prefix: str,
    user_id: str,
    name: str,
    now: Timestamp,
    expires_at: Timestamp | None = None,
) -> None:
    """Insert a new API key row."""
    with db.transaction() as tx:
        tx.execute(
            insert(auth_api_keys_table).values(
                key_id=key_id,
                key_prefix=key_prefix,
                user_id=user_id,
                name=name,
                created_at_ms=now,
                expires_at_ms=expires_at,
            )
        )
    logger.info(
        "event=api_key_created entity=%s trigger=- user=%s name=%s expires_at_ms=%s",
        key_id,
        user_id,
        name,
        expires_at.epoch_ms() if expires_at else "-",
    )


def touch_api_key(db: ControllerDB, key_id: str, now: Timestamp) -> None:
    """Update last_used_at timestamp."""
    with db.transaction() as tx:
        tx.execute(update(auth_api_keys_table).where(auth_api_keys_table.c.key_id == key_id).values(last_used_at_ms=now))


def revoke_api_key(db: ControllerDB, key_id: str, now: Timestamp) -> bool:
    """Revoke an API key. Returns True if key existed and was revoked."""
    with db.transaction() as tx:
        result = tx.execute(
            update(auth_api_keys_table)
            .where(
                auth_api_keys_table.c.key_id == key_id,
                auth_api_keys_table.c.revoked_at_ms.is_(None),
            )
            .values(revoked_at_ms=now)
        )
        revoked = result.rowcount > 0
    if revoked:
        logger.info("event=api_key_revoked entity=%s trigger=-", key_id)
    return revoked


def lookup_api_key_by_id(db: ControllerDB, key_id: str):
    """Find an API key by its key_id. Returns SA Row or None."""
    with db.auth_read_snapshot() as tx:
        return tx.execute(select(auth_api_keys_table).where(auth_api_keys_table.c.key_id == key_id)).first()


def list_api_keys(db: ControllerDB, user_id: str | None = None) -> Sequence[Row]:
    """List API keys, optionally filtered by user."""
    with db.auth_read_snapshot() as tx:
        stmt = select(auth_api_keys_table)
        if user_id:
            stmt = stmt.where(auth_api_keys_table.c.user_id == user_id)
        return tx.execute(stmt).all()


def revoke_login_keys_for_user(db: ControllerDB, user_id: str, now: Timestamp) -> list[str]:
    """Revoke all active login keys for a user. Returns the revoked key_ids."""
    with db.transaction() as tx:
        rows = tx.execute(
            update(auth_api_keys_table)
            .where(
                auth_api_keys_table.c.user_id == user_id,
                auth_api_keys_table.c.name.like("login-%"),
                auth_api_keys_table.c.revoked_at_ms.is_(None),
            )
            .values(revoked_at_ms=now)
            .returning(auth_api_keys_table.c.key_id)
        ).all()
    revoked_ids = [str(row.key_id) for row in rows]
    if revoked_ids:
        logger.info(
            "event=login_keys_revoked entity=%s trigger=- count=%d",
            user_id,
            len(revoked_ids),
        )
    return revoked_ids


# ---------------------------------------------------------------------------
# JWT token manager
# ---------------------------------------------------------------------------


def _get_or_create_signing_key(db: ControllerDB) -> str:
    """Load the HMAC signing key from DB, or create one on first run."""
    with db.auth_read_snapshot() as tx:
        row = tx.execute(
            select(auth_controller_secrets_table.c.value).where(auth_controller_secrets_table.c.key == "jwt_signing_key")
        ).first()
        if row is not None:
            return str(row.value)

    new_key = secrets.token_hex(32)
    now = Timestamp.now()
    with db.transaction() as tx:
        tx.execute(
            sqlite_insert(auth_controller_secrets_table)
            .values(key="jwt_signing_key", value=new_key, created_at_ms=now)
            .on_conflict_do_nothing(index_elements=["key"])
        )
    # Re-read in case of concurrent insert (INSERT OR IGNORE)
    with db.auth_read_snapshot() as tx:
        row = tx.execute(
            select(auth_controller_secrets_table.c.value).where(auth_controller_secrets_table.c.key == "jwt_signing_key")
        ).first()
        if row is None:
            raise RuntimeError("Failed to read or create JWT signing key")
        return str(row.value)


# Minimum interval between last_used_at writes for the same key (seconds).
_TOUCH_INTERVAL_SECONDS = 300  # 5 minutes


class JwtTokenManager:
    """Creates and verifies HMAC-SHA256 JWT tokens.

    Verification is a pure crypto operation followed by an in-memory
    revocation check — no DB hit on the hot path. An optional DB reference
    enables sampled last_used_at write-back (at most once per key per
    ``_TOUCH_INTERVAL_SECONDS``).
    """

    def __init__(self, signing_key: str, db: ControllerDB | None = None):
        self._signing_key = signing_key
        self._revoked_jtis: set[str] = set()
        self._db = db
        # Tracks the last wall-clock time we wrote last_used_at per jti.
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

    def create_endpoint_token(
        self,
        endpoint_name: str,
        key_id: str,
        ttl_seconds: int = DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS,
    ) -> str:
        """Mint a scoped bearer token authorizing only ``endpoint_name``'s /proxy path.

        Carries ``scope=proxy`` and ``aud=<wire name>``; ``verify`` surfaces the
        audience on the identity, and both the proxy and the RPC/HTTP auth arms
        treat an audience-bearing identity as endpoint-scoped (no RPC authority).
        """
        now = time.time()
        payload = {
            "sub": f"endpoint:{endpoint_name}",
            "role": ENDPOINT_TOKEN_ROLE,
            "aud": endpoint_name,
            "scope": ENDPOINT_TOKEN_SCOPE,
            "jti": key_id,
            "iat": int(now),
            "exp": int(now + ttl_seconds),
        }
        return jwt.encode(payload, self._signing_key, algorithm="HS256")

    def verify(self, token: str) -> VerifiedIdentity:
        """Verify JWT signature and claims, check revocation.

        On success, updates ``last_used_at`` in the DB at most once per key
        per ``_TOUCH_INTERVAL_SECONDS`` to avoid hot-path DB writes.
        """
        try:
            # verify_aud=False: this one verify() accepts both full-identity
            # tokens (no aud) and endpoint-scoped tokens (aud set). PyJWT 2.x
            # otherwise rejects any aud-bearing token when decode() gets no
            # audience= (InvalidAudienceError). Audience enforcement is ours,
            # at the proxy, against the endpoint the request names.
            payload = jwt.decode(token, self._signing_key, algorithms=["HS256"], options={"verify_aud": False})
        except jwt.ExpiredSignatureError as exc:
            raise ValueError("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise ValueError(f"Invalid token: {exc}") from exc

        jti = payload.get("jti", "")
        if jti in self._revoked_jtis:
            raise ValueError("Token has been revoked")

        self._maybe_touch(jti)

        audience = payload.get("aud") if payload.get("scope") == ENDPOINT_TOKEN_SCOPE else None
        return VerifiedIdentity(
            user_id=payload["sub"],
            role=payload.get("role", "user"),
            audience=audience,
        )

    def _maybe_touch(self, jti: str) -> None:
        """Write last_used_at to DB if enough time has elapsed since the last write."""
        if not self._db or not jti:
            return
        now = time.time()
        last = self._last_touched.get(jti, 0.0)
        if now - last < _TOUCH_INTERVAL_SECONDS:
            return
        self._last_touched[jti] = now
        try:
            touch_api_key(self._db, jti, Timestamp.from_seconds(now))
        except Exception:
            logger.debug("Failed to update last_used_at for key %s", jti, exc_info=True)

    def revoke(self, jti: str) -> None:
        """Add a JTI to the in-memory revocation set."""
        self._revoked_jtis.add(jti)

    def load_revocations(self, db: ControllerDB) -> None:
        """Load revoked key_ids from api_keys into the revocation set.

        Only loads keys that haven't expired yet — expired JWTs are rejected
        by signature verification anyway, so their JTIs don't need tracking.
        """
        now_ms = int(time.time() * 1000)
        with db.auth_read_snapshot() as tx:
            rows = tx.execute(
                select(auth_api_keys_table.c.key_id).where(
                    auth_api_keys_table.c.revoked_at_ms.is_not(None),
                    (auth_api_keys_table.c.expires_at_ms.is_(None)) | (auth_api_keys_table.c.expires_at_ms > now_ms),
                )
            ).all()
            self._revoked_jtis = {str(row.key_id) for row in rows}


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


def create_controller_auth(
    auth_config: AuthConfig | None,
    db: ControllerDB | None = None,
) -> ControllerAuth:
    """Build a ``ControllerAuth`` from the auth config.

    Signs JWTs with a persistent key in ``controller_secrets``; ``api_keys``
    rows exist for audit and revocation, but verification never hits the DB.

    A ``None`` config (or one with no provider selected and no trusted CIDRs)
    runs in null-auth mode. ``trusted_cidrs`` alone enables auth: identity by
    network location for direct in-network peers, tokens for everything else.
    """
    if auth_config is None or (auth_config.provider_kind() is None and not auth_config.trusted_cidrs):
        if db:
            now = Timestamp.now()
            with db.transaction() as _tx:
                writes.ensure_user(_tx, "anonymous", now, role="admin")
                writes.set_user_role(_tx, "anonymous", "admin")

            signing_key = _get_or_create_signing_key(db)
            jwt_mgr = JwtTokenManager(signing_key, db=db)
            jwt_mgr.load_revocations(db)

            worker_token = _create_worker_jwt(db, jwt_mgr, now)
            logger.info("Authentication disabled — null-auth mode (workers use JWT)")
            return ControllerAuth(verifier=jwt_mgr, worker_token=worker_token, jwt_manager=jwt_mgr)
        logger.info("Authentication disabled — null-auth mode, no DB")
        return ControllerAuth()

    provider = auth_config.provider_kind() or CIDR_PROVIDER
    now = Timestamp.now()

    jwt_mgr: JwtTokenManager | None = None
    worker_token: str | None = None

    if db:
        signing_key = _get_or_create_signing_key(db)
        jwt_mgr = JwtTokenManager(signing_key, db=db)
        jwt_mgr.load_revocations(db)

        if provider == "static":
            _preload_static_tokens(auth_config.static, db, now)

        worker_token = _create_worker_jwt(db, jwt_mgr, now)

        for admin_user in auth_config.admin_users:
            with db.transaction() as _tx:
                writes.ensure_user(_tx, admin_user, now)
                writes.set_user_role(_tx, admin_user, "admin")

        verifier: TokenVerifier | None = jwt_mgr
    else:
        ephemeral_key = secrets.token_hex(32)
        jwt_mgr = JwtTokenManager(ephemeral_key)
        worker_token = jwt_mgr.create_token(WORKER_USER, "worker", f"iris_k_worker_{secrets.token_hex(8)}")
        verifier = None

    login_verifier: TokenVerifier | None = None
    gcp_project_id: str | None = None
    if provider == "gcp":
        gcp_project_id = auth_config.gcp.project_id
        if not gcp_project_id:
            raise ValueError("GCP auth config requires a project_id")
        login_verifier = GcpAccessTokenVerifier(project_id=gcp_project_id)

    # For static auth, use StaticTokenVerifier as the login verifier so
    # `iris login` can exchange a raw config token for a JWT.
    if provider == "static":
        static_tokens = dict(auth_config.static.tokens)
        login_verifier = StaticTokenVerifier(static_tokens)

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


def _preload_static_tokens(
    static_config: StaticAuthConfig,
    db: ControllerDB,
    now: Timestamp,
) -> None:
    """Insert static config tokens into the api_keys table for audit.

    Verification of static tokens happens in-memory via ``StaticTokenVerifier``;
    these rows exist only so configured tokens surface in ``iris key list``.
    """
    tokens = dict(static_config.tokens)
    if not tokens:
        raise ValueError("Static auth config requires at least one token")

    with db.transaction() as tx:
        tx.execute(delete(auth_api_keys_table).where(auth_api_keys_table.c.key_id.like("iris_k_static_%")))

    for raw_token, username in tokens.items():
        with db.transaction() as _tx:
            writes.ensure_user(_tx, username, now)
        key_id = f"iris_k_static_{username}"
        create_api_key(
            db,
            key_id=key_id,
            key_prefix=raw_token[:8],
            user_id=username,
            name=f"static-config-{username}",
            now=now,
        )
    logger.info("Preloaded %d static token(s) into api_keys", len(tokens))


def _create_worker_jwt(db: ControllerDB, jwt_mgr: JwtTokenManager, now: Timestamp) -> str:
    """Generate a JWT for the worker identity on each controller start.

    Old worker tokens are not revoked so that in-flight workers can finish
    gracefully with their existing credentials.
    """
    key_id = f"iris_k_worker_{secrets.token_hex(8)}"
    with db.transaction() as _tx:
        writes.ensure_user(_tx, WORKER_USER, now, role="worker")
    create_api_key(
        db,
        key_id=key_id,
        key_prefix="jwt",
        user_id=WORKER_USER,
        name="worker-token",
        now=now,
    )
    jwt_token = jwt_mgr.create_token(WORKER_USER, "worker", key_id)
    logger.info("New worker JWT generated (key_id=%s)", key_id)
    return jwt_token
