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
from rigging.auth import BearerTokenInjector

from iris.cluster.config import ClusterFinelogConfig
from iris.cluster.controller.auth import JwtTokenManager

logger = logging.getLogger(__name__)

# Delegation-JWT lifetime and how early to re-mint before expiry. Short-lived so a
# leaked token's blast radius is TTL-bounded: the global store checks only
# signature + exp, and iris tokens are never revocable.
_DELEGATION_TTL_SECONDS = 3600
_DELEGATION_REFRESH_MARGIN_SECONDS = 300

_STATE_FILENAME = "finelog_forwarder_state.json"


class _DelegationTokenProvider:
    """Mints short-lived EdDSA delegation JWTs, re-minting each before it expires.

    Delegates minting to the controller's :class:`JwtTokenManager`
    (:meth:`~JwtTokenManager.create_delegation_token`): an ``aud="finelog"`` token
    the federated finelog verifies against this controller's public key.
    """

    def __init__(
        self,
        *,
        subject: str,
        jwt_manager: JwtTokenManager,
        ttl_seconds: int = _DELEGATION_TTL_SECONDS,
        refresh_margin_seconds: int = _DELEGATION_REFRESH_MARGIN_SECONDS,
    ):
        self._jwt = jwt_manager
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
                self._token = self._jwt.create_delegation_token(self._subject, jti, ttl_seconds=self._ttl)
                self._expiry = now + self._ttl
            return self._token


def finelog_relay_interceptors(
    subject: str,
    jwt_manager: JwtTokenManager | None,
) -> tuple:
    """Resolve the client interceptors the forwarder presents to the global finelog.

    When this controller has a signer (``jwt_manager``), the relay injects a
    bearer minting a short-lived ``aud="finelog"`` delegation JWT under ``subject``
    (this cluster's name), which the store verifies against this controller's
    public key. With no signer (null-auth), no bearer is sent and the store must
    admit this controller by a ``cidr`` layer (a same-VPC/loopback store).

    A shared finelog must carry THIS controller's public key (served on its JWKS /
    printed by ``iris cluster init-keys``) in its ``jwt`` auth layer to verify the
    delegation token end-to-end; the finelog deploy configs carry a commented
    placeholder for that layer. Generating it across services -- deriving the public
    key and populating the shared finelog's config -- is a top-level marin
    admin-script concern (issue #6961), deliberately not built into iris: iris must
    not render finelog's deploy config.
    """
    if jwt_manager is not None:
        provider = _DelegationTokenProvider(subject=subject, jwt_manager=jwt_manager)
        return (BearerTokenInjector(provider, "authorization"),)
    return ()


def build_log_forwarder(
    *,
    config: ClusterFinelogConfig,
    cluster_id: str,
    source_client: LogClient,
    state_dir: Path,
    jwt_manager: JwtTokenManager | None,
) -> LogForwarder:
    """Construct a forwarder from ``source_client`` to ``config.relay_address``.

    The target client is authenticated with this cluster's delegation credential
    (an ``aud="finelog"`` JWT minted under ``cluster_id`` via ``jwt_manager``) and
    owned by the forwarder (closed on its ``stop``); ``source_client`` is the
    controller's local finelog client and is not.
    """
    target = LogClient.connect(
        config.relay_address,
        interceptors=finelog_relay_interceptors(cluster_id, jwt_manager),
    )
    return LogForwarder(
        source=source_client,
        target=target,
        target_label=config.relay_address,
        state_path=state_dir / _STATE_FILENAME,
        cluster=cluster_id,
    )
