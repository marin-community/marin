# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Auth setup for the controller — single source of truth for verifier creation."""

from __future__ import annotations

import dataclasses
import logging
import secrets

from iris.cluster.controller.db import API_KEYS, ApiKey, ControllerDB
from iris.rpc import config_pb2
from iris.rpc.auth import (
    GcpAccessTokenVerifier,
    TokenVerifier,
    hash_token,
)
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)

WORKER_USER = "system:worker"
_LAST_USED_THROTTLE_MS = 60_000


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


def revoke_login_keys_for_user(db: ControllerDB, user_id: str, now: Timestamp) -> int:
    """Revoke all active login keys for a user. Returns count of revoked keys."""
    with db.transaction() as cur:
        cur.execute(
            "UPDATE api_keys SET revoked_at_ms = ? WHERE user_id = ? AND name LIKE 'login-%' AND revoked_at_ms IS NULL",
            (now.epoch_ms(), user_id),
        )
        return cur._cursor.rowcount


# ---------------------------------------------------------------------------
# Token verification backed by the api_keys table
# ---------------------------------------------------------------------------


class DbTokenVerifier:
    """Verifies bearer tokens by hashing and looking up in the api_keys table."""

    def __init__(self, db: ControllerDB):
        self._db = db

    def verify(self, token: str) -> str:
        key_hash = hash_token(token)
        api_key = lookup_api_key_by_hash(self._db, key_hash)
        if api_key is None:
            raise ValueError("Invalid API key")
        if api_key.revoked_at is not None:
            raise ValueError("API key has been revoked")
        now = Timestamp.now()
        if api_key.expires_at is not None and api_key.expires_at < now:
            raise ValueError("API key has expired")
        if api_key.last_used_at is None or (now.epoch_ms() - api_key.last_used_at.epoch_ms()) > _LAST_USED_THROTTLE_MS:
            touch_api_key(self._db, api_key.key_id, now)
        return api_key.user_id


@dataclasses.dataclass
class ControllerAuth:
    """Resolved auth configuration for a controller instance."""

    verifier: TokenVerifier | None = None
    provider: str | None = None
    worker_token: str | None = None
    login_verifier: TokenVerifier | None = None
    gcp_project_id: str | None = None


def create_controller_auth(
    auth_config: config_pb2.AuthConfig,
    db: ControllerDB | None = None,
) -> ControllerAuth:
    """Create auth verifier + worker token from config proto.

    Returns ControllerAuth with all None fields when auth is disabled.
    When auth is enabled and a DB is provided, the DB is the single source of
    truth: static config tokens are preloaded, the worker gets an API key, and
    DbTokenVerifier handles all request verification.
    """
    if not auth_config.HasField("provider"):
        if db:
            now = Timestamp.now()
            db.ensure_user("anonymous", now, role="admin")
            db.set_user_role("anonymous", "admin")
        logger.info("Authentication disabled — null-auth mode (all requests as anonymous admin)")
        return ControllerAuth()

    provider = auth_config.WhichOneof("provider")
    now = Timestamp.now()

    if db:
        if provider == "static":
            _preload_static_tokens(auth_config.static, db, now)

        worker_token = _create_worker_token(db, now)

        for admin_user in auth_config.admin_users:
            db.ensure_user(admin_user, now)
            db.set_user_role(admin_user, "admin")

        verifier: TokenVerifier | None = DbTokenVerifier(db)
    else:
        worker_token = secrets.token_urlsafe(32)
        verifier = None

    login_verifier: TokenVerifier | None = None
    gcp_project_id: str | None = None
    if provider == "gcp":
        gcp_project_id = auth_config.gcp.project_id
        if not gcp_project_id:
            raise ValueError("GCP auth config requires a project_id")
        login_verifier = GcpAccessTokenVerifier(project_id=gcp_project_id)

    logger.info("Auth enabled: provider=%s, db=%s", provider, "yes" if db else "no")
    return ControllerAuth(
        verifier=verifier,
        provider=provider,
        worker_token=worker_token,
        login_verifier=login_verifier,
        gcp_project_id=gcp_project_id,
    )


def _preload_static_tokens(
    static_config: config_pb2.StaticAuthConfig,
    db: ControllerDB,
    now: Timestamp,
) -> None:
    """Insert static config tokens into the api_keys table (idempotent)."""
    tokens = dict(static_config.tokens)
    if not tokens:
        raise ValueError("Static auth config requires at least one token")

    for raw_token, username in tokens.items():
        db.ensure_user(username, now)
        key_hash = hash_token(raw_token)
        existing = lookup_api_key_by_hash(db, key_hash)
        if existing is None:
            key_id = f"iris_k_static_{username}"
            create_api_key(
                db,
                key_id=key_id,
                key_hash=key_hash,
                key_prefix=raw_token[:8],
                user_id=username,
                name=f"static-config-{username}",
                now=now,
            )
    logger.info("Preloaded %d static token(s) into api_keys", len(tokens))


def _create_worker_token(db: ControllerDB, now: Timestamp) -> str:
    """Generate a fresh worker API key on each controller start.

    Old worker tokens are not revoked so that in-flight workers can finish
    gracefully with their existing credentials.
    """
    raw_token = secrets.token_urlsafe(32)
    key_id = f"iris_k_worker_{secrets.token_hex(8)}"
    db.ensure_user(WORKER_USER, now, role="worker")
    create_api_key(
        db,
        key_id=key_id,
        key_hash=hash_token(raw_token),
        key_prefix=raw_token[:8],
        user_id=WORKER_USER,
        name="worker-token",
        now=now,
    )
    logger.info("New worker token generated (key_id=%s)", key_id)
    return raw_token
