# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wire this cluster's finelog to a shared global finelog.

The log-forwarding mechanism itself lives in finelog (:class:`finelog.forwarder.LogForwarder`).
This module is the thin Iris glue: it resolves the delegation credential this cluster
presents to the global store and constructs a forwarder from the controller's local
finelog client to that store. No forwarding, namespacing, or watermark logic lives
here — only credential + wiring.
"""

import logging
import secrets
import threading
import time
from pathlib import Path

from finelog.client.log_client import LogClient
from finelog.forwarder import LogForwarder
from rigging.auth import BearerTokenInjector, StaticTokenProvider

from iris.cluster.config import ClusterFinelogConfig
from iris.cluster.controller.auth import JwtTokenManager

logger = logging.getLogger(__name__)

# Delegation-JWT lifetime and how early to re-mint before expiry. Short-lived so a
# leaked token's blast radius is TTL-bounded (the global store checks only signature +
# exp; it cannot reach a controller's revocation table).
_DELEGATION_TTL_SECONDS = 3600
_DELEGATION_REFRESH_MARGIN_SECONDS = 300
_RELAY_ROLE = "finelog-relay"

_STATE_FILENAME = "finelog_forwarder_state.json"


class _DelegationTokenProvider:
    """Mints short-lived HS256 delegation JWTs, re-minting each before it expires.

    Signs with the cluster's delegation key, which must be dedicated (not the
    control-plane signing key): a token the global store can verify then never grants
    the power to mint control-plane tokens.
    """

    def __init__(
        self,
        *,
        subject: str,
        signing_key: str,
        ttl_seconds: int = _DELEGATION_TTL_SECONDS,
        refresh_margin_seconds: int = _DELEGATION_REFRESH_MARGIN_SECONDS,
    ):
        self._jwt = JwtTokenManager(signing_key)
        self._subject = subject
        self._ttl = ttl_seconds
        self._margin = refresh_margin_seconds
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expiry = 0.0

    def get_token(self) -> str | None:
        with self._lock:
            now = time.time()
            if self._token is None or now >= self._expiry - self._margin:
                jti = secrets.token_hex(8)
                self._token = self._jwt.create_token(self._subject, _RELAY_ROLE, jti, ttl_seconds=self._ttl)
                self._expiry = now + self._ttl
            return self._token


def finelog_relay_interceptors(config: ClusterFinelogConfig, subject: str) -> tuple:
    """Resolve the client interceptors the forwarder presents to the global finelog.

    ``delegation_key`` (preferred) mints short-lived JWTs — under ``subject`` (this
    cluster's name) — that the store verifies against its ``jwt`` layer; ``static_token``
    is a pre-minted bearer; neither means the store must admit this controller by a
    ``cidr`` layer (a same-VPC/loopback store).
    """
    if config.delegation_key:
        provider = _DelegationTokenProvider(subject=subject, signing_key=config.delegation_key)
        return (BearerTokenInjector(provider, "authorization"),)
    if config.static_token:
        return (BearerTokenInjector(StaticTokenProvider(config.static_token), "authorization"),)
    return ()


def build_log_forwarder(
    *,
    config: ClusterFinelogConfig,
    cluster_id: str,
    source_client: LogClient,
    state_dir: Path,
) -> LogForwarder:
    """Construct a forwarder from ``source_client`` to ``config.relay_address``.

    The target client is authenticated with this cluster's delegation credential (minted
    under ``cluster_id``) and owned by the forwarder (closed on its ``stop``);
    ``source_client`` is the controller's local finelog client and is not.
    """
    target = LogClient.connect(config.relay_address, interceptors=finelog_relay_interceptors(config, cluster_id))
    return LogForwarder(
        source=source_client,
        target=target,
        target_label=config.relay_address,
        state_path=state_dir / _STATE_FILENAME,
    )
