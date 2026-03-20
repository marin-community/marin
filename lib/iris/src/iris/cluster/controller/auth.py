# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Auth setup for the controller — single source of truth for verifier creation.

All tokens are JWTs signed with a persistent HMAC-SHA256 key stored in the
controller_secrets table. Verification is a pure crypto check plus an
in-memory revocation set — no per-RPC database hit.
"""

from __future__ import annotations

import dataclasses
import logging
import secrets
import time
from collections.abc import Callable

import jwt

from iris.cluster.controller.db import API_KEYS, ApiKey, ControllerDB
from iris.rpc import config_pb2
from iris.rpc.auth import (
    GcpAccessTokenVerifier,
    StaticTokenVerifier,
    TokenVerifier,
    VerifiedIdentity,
    hash_token,
)
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)

WORKER_USER = "system:worker"
DEFAULT_JWT_TTL_SECONDS = 86400 * 30  # 30 days


# ---------------------------------------------------------------------------
# API key CRUD — top-level functions operating on ControllerDB
# ---------------------------------------------------------------------------


def create_api_key(
    db: ControllerDB,
    key_id: str,
    key_hash: str,
    key_prefix: str,
    user_id: str,
    name: str,
    now: Timestamp,
    expires_at: Timestamp | None = None,
) -> None:
    """Insert a new API key row."""
    db.execute(
        "INSERT INTO api_keys (key_id, key_hash, key_prefix, user_id, name, created_at_ms, expires_at_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (key_id, key_hash, key_prefix, user_id, name, now.epoch_ms(), expires_at.epoch_ms() if expires_at else None),
    )


def lookup_api_key_by_hash(db: ControllerDB, key_hash: str) -> ApiKey | None:
    """Find an API key by its SHA-256 hash."""
    with db.snapshot() as q:
        return q.one(API_KEYS, where=API_KEYS.c.key_hash == key_hash)


def touch_api_key(db: ControllerDB, key_id: str, now: Timestamp) -> None:
    """Update last_used_at timestamp."""
    db.execute(
        "UPDATE api_keys SET last_used_at_ms = ? WHERE key_id = ?",
        (now.epoch_ms(), key_id),
    )


def revoke_api_key(db: ControllerDB, key_id: str, now: Timestamp) -> bool:
    """Revoke an API key. Returns True if key existed and was revoked."""
    with db.transaction() as cur:
        cur.execute(
            "UPDATE api_keys SET revoked_at_ms = ? WHERE key_id = ? AND revoked_at_ms IS NULL",
            (now.epoch_ms(), key_id),
        )
        return cur._cursor.rowcount > 0


def list_api_keys(db: ControllerDB, user_id: str | None = None) -> list[ApiKey]:
    """List API keys, optionally filtered by user."""
    with db.snapshot() as q:
        if user_id:
            return q.select(API_KEYS, where=API_KEYS.c.user_id == user_id)
        return q.select(API_KEYS)


def revoke_login_keys_for_user(db: ControllerDB, user_id: str, now: Timestamp) -> list[str]:
    """Revoke all active login keys for a user. Returns list of revoked key_ids."""
    with db.snapshot() as q:
        active_login_keys = q.raw(
            "SELECT key_id FROM api_keys WHERE user_id = ? AND name LIKE 'login-%' AND revoked_at_ms IS NULL",
            (user_id,),
            decoders={"key_id": str},
        )
    revoked_ids = [row.key_id for row in active_login_keys]
    if revoked_ids:
        with db.transaction() as cur:
            cur.execute(
                "UPDATE api_keys SET revoked_at_ms = ?"
                " WHERE user_id = ? AND name LIKE 'login-%' AND revoked_at_ms IS NULL",
                (now.epoch_ms(), user_id),
            )
    return revoked_ids


# ---------------------------------------------------------------------------
# JWT token manager
# ---------------------------------------------------------------------------


def _get_or_create_signing_key(db: ControllerDB) -> str:
    """Load the HMAC signing key from DB, or create one on first run."""
    with db.snapshot() as q:
        rows = q.raw(
            "SELECT value FROM controller_secrets WHERE key = ?",
            ("jwt_signing_key",),
            decoders={"value": str},
        )
        if rows:
            return rows[0].value

    new_key = secrets.token_hex(32)
    now = Timestamp.now()
    db.execute(
        "INSERT OR IGNORE INTO controller_secrets (key, value, created_at_ms) VALUES (?, ?, ?)",
        ("jwt_signing_key", new_key, now.epoch_ms()),
    )
    # Re-read in case of concurrent insert (INSERT OR IGNORE)
    with db.snapshot() as q:
        rows = q.raw(
            "SELECT value FROM controller_secrets WHERE key = ?",
            ("jwt_signing_key",),
            decoders={"value": str},
        )
        if not rows:
            raise RuntimeError("Failed to read or create JWT signing key")
        return rows[0].value


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

    def create_token(
        self,
        user_id: str,
        role: str,
        key_id: str,
        ttl_seconds: int = DEFAULT_JWT_TTL_SECONDS,
        worker_id: str = "",
    ) -> str:
        now = time.time()
        payload = {
            "sub": user_id,
            "role": role,
            "jti": key_id,
            "iat": int(now),
            "exp": int(now + ttl_seconds),
        }
        if worker_id:
            payload["wid"] = worker_id
        return jwt.encode(payload, self._signing_key, algorithm="HS256")

    def verify(self, token: str) -> VerifiedIdentity:
        """Verify JWT signature and claims, check revocation.

        On success, updates ``last_used_at`` in the DB at most once per key
        per ``_TOUCH_INTERVAL_SECONDS`` to avoid hot-path DB writes.
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
            worker_id=payload.get("wid", ""),
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
        with db.snapshot() as q:
            rows = q.raw(
                "SELECT key_id FROM api_keys"
                " WHERE revoked_at_ms IS NOT NULL"
                " AND (expires_at_ms IS NULL OR expires_at_ms > ?)",
                (now_ms,),
                decoders={"key_id": str},
            )
            self._revoked_jtis = {row.key_id for row in rows}


# ---------------------------------------------------------------------------
# Controller auth configuration
# ---------------------------------------------------------------------------


WorkerTokenFactory = Callable[[str], str]


@dataclasses.dataclass
class ControllerAuth:
    """Resolved auth configuration for a controller instance."""

    verifier: TokenVerifier | None = None
    provider: str | None = None
    worker_token_factory: WorkerTokenFactory | None = None
    login_verifier: TokenVerifier | None = None
    gcp_project_id: str | None = None
    jwt_manager: JwtTokenManager | None = None


def create_controller_auth(
    auth_config: config_pb2.AuthConfig,
    db: ControllerDB | None = None,
) -> ControllerAuth:
    """Create auth verifier + per-worker token factory from config proto.

    All tokens are JWTs signed with a persistent key stored in
    controller_secrets. The api_keys table is retained for audit and
    revocation tracking, but verification never hits the database.

    Instead of a single shared worker token, a ``worker_token_factory``
    callable is returned that generates a unique JWT per worker_id.
    """
    if not auth_config.HasField("provider"):
        if db:
            now = Timestamp.now()
            db.ensure_user("anonymous", now, role="admin")
            db.set_user_role("anonymous", "admin")

            signing_key = _get_or_create_signing_key(db)
            jwt_mgr = JwtTokenManager(signing_key, db=db)
            jwt_mgr.load_revocations(db)

            db.ensure_user(WORKER_USER, now, role="worker")
            factory = _make_worker_token_factory(db, jwt_mgr)
            logger.info("Authentication disabled — null-auth mode (workers use per-worker JWT)")
            return ControllerAuth(verifier=jwt_mgr, worker_token_factory=factory, jwt_manager=jwt_mgr)
        logger.info("Authentication disabled — null-auth mode, no DB")
        return ControllerAuth()

    provider = auth_config.WhichOneof("provider")
    now = Timestamp.now()

    jwt_mgr: JwtTokenManager | None = None
    factory: WorkerTokenFactory | None = None

    if db:
        signing_key = _get_or_create_signing_key(db)
        jwt_mgr = JwtTokenManager(signing_key, db=db)
        jwt_mgr.load_revocations(db)

        if provider == "static":
            _preload_static_tokens(auth_config.static, db, now)

        db.ensure_user(WORKER_USER, now, role="worker")
        factory = _make_worker_token_factory(db, jwt_mgr)

        for admin_user in auth_config.admin_users:
            db.ensure_user(admin_user, now)
            db.set_user_role(admin_user, "admin")

        verifier: TokenVerifier | None = jwt_mgr
    else:
        ephemeral_key = secrets.token_hex(32)
        jwt_mgr = JwtTokenManager(ephemeral_key)
        factory = lambda wid: jwt_mgr.create_token(  # noqa: E731
            WORKER_USER, "worker", f"iris_k_worker_{secrets.token_hex(8)}", worker_id=wid
        )
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

    logger.info("Auth enabled: provider=%s, db=%s, jwt=%s", provider, "yes" if db else "no", "yes" if jwt_mgr else "no")
    return ControllerAuth(
        verifier=verifier,
        provider=provider,
        worker_token_factory=factory,
        login_verifier=login_verifier,
        gcp_project_id=gcp_project_id,
        jwt_manager=jwt_mgr,
    )


def _preload_static_tokens(
    static_config: config_pb2.StaticAuthConfig,
    db: ControllerDB,
    now: Timestamp,
) -> None:
    """Insert static config tokens into the api_keys table for audit.

    The raw token hashes are stored so that the Login RPC can verify
    static tokens during the login exchange flow.
    """
    tokens = dict(static_config.tokens)
    if not tokens:
        raise ValueError("Static auth config requires at least one token")

    db.execute("DELETE FROM api_keys WHERE key_id LIKE 'iris_k_static_%'")

    for raw_token, username in tokens.items():
        db.ensure_user(username, now)
        key_id = f"iris_k_static_{username}"
        create_api_key(
            db,
            key_id=key_id,
            key_hash=hash_token(raw_token),
            key_prefix=raw_token[:8],
            user_id=username,
            name=f"static-config-{username}",
            now=now,
        )
    logger.info("Preloaded %d static token(s) into api_keys", len(tokens))


def _make_worker_token_factory(db: ControllerDB, jwt_mgr: JwtTokenManager) -> WorkerTokenFactory:
    """Return a callable that mints a unique per-worker JWT.

    Each call creates a fresh ``api_keys`` row for audit/revocation and
    embeds the ``worker_id`` in the JWT ``wid`` claim. This enables
    per-worker revocation and identity binding.
    """

    def _create(worker_id: str) -> str:
        now = Timestamp.now()
        key_id = f"iris_k_worker_{secrets.token_hex(8)}"
        create_api_key(
            db,
            key_id=key_id,
            key_hash=f"jwt:{key_id}",
            key_prefix="jwt",
            user_id=WORKER_USER,
            name=f"worker-token:{worker_id}",
            now=now,
        )
        token = jwt_mgr.create_token(WORKER_USER, "worker", key_id, worker_id=worker_id)
        logger.info("Per-worker JWT generated (key_id=%s, worker_id=%s)", key_id, worker_id)
        return token

    return _create
