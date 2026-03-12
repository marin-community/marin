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
    CompositeTokenVerifier,
    GcpTokenVerifier,
    StaticTokenVerifier,
    TokenVerifier,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ControllerAuth:
    """Resolved auth configuration for a controller instance."""

    verifier: TokenVerifier | None = None
    provider: str | None = None
    worker_token: str | None = None


def create_controller_auth(
    auth_config: config_pb2.AuthConfig,
    db: ControllerDB | None = None,
) -> ControllerAuth:
    """Create auth verifier + worker token from config proto.

    Returns ControllerAuth with all None fields when auth is disabled.
    When auth is enabled, reuses a persisted worker token from the DB if available,
    otherwise generates a fresh one and stores it. This ensures existing workers
    survive controller restarts.
    """
    if not auth_config.HasField("provider"):
        logger.info("Authentication disabled (no auth config)")
        return ControllerAuth()

    provider = auth_config.WhichOneof("provider")
    config_verifier = _create_config_verifier(auth_config, provider)

    # Reuse persisted worker token if available so existing workers survive restarts.
    worker_token = db.get_kv("worker_auth_token") if db else None
    if worker_token:
        logger.info("Auth enabled: provider=%s, reusing persisted worker token", provider)
    else:
        worker_token = secrets.token_urlsafe(32)
        if db:
            db.set_kv("worker_auth_token", worker_token)
        logger.info("Auth enabled: provider=%s, new worker token generated", provider)

    worker_verifier = StaticTokenVerifier({worker_token: "iris-worker"})
    composite = CompositeTokenVerifier([worker_verifier, config_verifier])

    return ControllerAuth(verifier=composite, provider=provider, worker_token=worker_token)


def _create_config_verifier(auth_config: config_pb2.AuthConfig, provider: str) -> TokenVerifier:
    """Create the user-facing verifier from config."""
    if provider == "gcp":
        audience = auth_config.gcp.audience
        if not audience:
            raise ValueError("GCP auth config requires an audience")
        return GcpTokenVerifier(audience=audience)
    elif provider == "static":
        tokens = dict(auth_config.static.tokens)
        if not tokens:
            raise ValueError("Static auth config requires at least one token")
        return StaticTokenVerifier(tokens)
    raise ValueError(f"Unknown auth provider: {provider}")
