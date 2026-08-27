# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open a :class:`LogClient` against a named deployment.

A config with ``client_url`` reaches the server through the Iris IAP proxy;
without one, through an SSH or Kubernetes tunnel to its port.

:func:`rigging.credentials.iap_provider_for` supplies the IAP token: the
desktop token cached by ``iris --cluster <name> login``, or one minted from
ambient service-account credentials when no login is cached.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

from rigging.connect import IapAuth, connect, disconnect
from rigging.credentials import iap_provider_for
from rigging.tunnel import open_tunnel

from finelog.client.log_client import LogClient
from finelog.deploy.config import FinelogConfig, tunnel_target_for

logger = logging.getLogger(__name__)

# Longer than the server's own 10s query deadline, so a slow query fails with
# the server's reason instead of a bare client-side timeout.
DEFAULT_REQUEST_TIMEOUT = 15.0
DEFAULT_TUNNEL_TIMEOUT = 60.0


@contextmanager
def open_client(
    cfg: FinelogConfig,
    name: str,
    *,
    tunnel_timeout: float = DEFAULT_TUNNEL_TIMEOUT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Generator[LogClient, None, None]:
    """Yield a LogClient for ``cfg``, closing it and its transport on exit."""
    timeout_ms = int(request_timeout * 1000)
    if not cfg.client_url:
        target = tunnel_target_for(cfg)
        logger.debug("Opening a tunnel to finelog %r", cfg.name)
        with open_tunnel(target, timeout=tunnel_timeout) as url:
            client = LogClient.connect(url, timeout_ms=timeout_ms)
            try:
                yield client
            finally:
                client.close()
        return

    client = connect(
        cfg.client_url,
        lambda ep: LogClient.connect(ep.url, interceptors=ep.interceptors, timeout_ms=timeout_ms),
        auth=IapAuth(iap_provider_for(name)),
        connect_timeout=tunnel_timeout,
    )
    try:
        yield client
    finally:
        client.close()
        disconnect(client)
