# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Auth setup for the controller — single source of truth for verifier creation."""

from __future__ import annotations

import dataclasses
import logging
import secrets

from iris.cluster.controller.db import ControllerDB
from iris.rpc import config_pb2
from iris.rpc.auth import (
    GcpTokenVerifier,
    TokenVerifier,
    hash_token,
)
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)

WORKER_USER = "system:worker"
_LAST_USED_THROTTLE_MS = 60_000


class DbTokenVerifier:
    """Verifies bearer tokens by hashing and looking up in the api_keys table."""

    def __init__(self, db: ControllerDB):
        self._db = db

    def verify(self, token: str) -> str:
        key_hash = hash_token(token)
        api_key = self._db.lookup_api_key_by_hash(key_hash)
        if api_key is None:
            raise ValueError("Invalid API key")
        if api_key.revoked_at is not None:
            raise ValueError("API key has been revoked")
        now = Timestamp.now()
        if api_key.expires_at is not None and api_key.expires_at < now:
            raise ValueError("API key has expired")
        if api_key.last_used_at is None or (now.epoch_ms() - api_key.last_used_at.epoch_ms()) > _LAST_USED_THROTTLE_MS:
            self._db.touch_api_key(api_key.key_id, now)
        return api_key.user_id


@dataclasses.dataclass
class ControllerAuth:
    """Resolved auth configuration for a controller instance."""

    verifier: TokenVerifier | None = None
    provider: str | None = None
    worker_token: str | None = None
    login_verifier: TokenVerifier | None = None


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

        worker_token = _ensure_worker_token(db, now)

        for admin_user in auth_config.admin_users:
            db.ensure_user(admin_user, now)
            db.set_user_role(admin_user, "admin")

        verifier: TokenVerifier | None = DbTokenVerifier(db)
    else:
        worker_token = secrets.token_urlsafe(32)
        verifier = None

    login_verifier: TokenVerifier | None = None
    if provider == "gcp":
        audience = auth_config.gcp.audience
        if not audience:
            raise ValueError("GCP auth config requires an audience")
        login_verifier = GcpTokenVerifier(audience=audience)

    logger.info("Auth enabled: provider=%s, db=%s", provider, "yes" if db else "no")
    return ControllerAuth(
        verifier=verifier,
        provider=provider,
        worker_token=worker_token,
        login_verifier=login_verifier,
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
        existing = db.lookup_api_key_by_hash(key_hash)
        if existing is None:
            key_id = f"iris_k_static_{username}"
            db.create_api_key(
                key_id=key_id,
                key_hash=key_hash,
                key_prefix=raw_token[:8],
                user_id=username,
                name=f"static-config-{username}",
                now=now,
            )
    logger.info("Preloaded %d static token(s) into api_keys", len(tokens))


def _ensure_worker_token(db: ControllerDB, now: Timestamp) -> str:
    """Ensure a worker API key exists in the DB, creating one if needed.

    The raw token is stored in controller_kv for recovery (can't reverse a hash).
    The api_keys table is the verification authority.
    """
    raw_token = db.get_kv("worker_auth_token")
    if raw_token:
        # Ensure the key is also in api_keys (idempotent after restart)
        key_hash = hash_token(raw_token)
        if db.lookup_api_key_by_hash(key_hash) is None:
            db.ensure_user(WORKER_USER, now, role="worker")
            db.create_api_key(
                key_id="iris_k_worker",
                key_hash=key_hash,
                key_prefix=raw_token[:8],
                user_id=WORKER_USER,
                name="worker-token",
                now=now,
            )
        logger.info("Reusing persisted worker token")
        return raw_token

    raw_token = secrets.token_urlsafe(32)
    db.set_kv("worker_auth_token", raw_token)
    db.ensure_user(WORKER_USER, now, role="worker")
    db.create_api_key(
        key_id="iris_k_worker",
        key_hash=hash_token(raw_token),
        key_prefix=raw_token[:8],
        user_id=WORKER_USER,
        name="worker-token",
        now=now,
    )
    logger.info("New worker token generated")
    return raw_token
