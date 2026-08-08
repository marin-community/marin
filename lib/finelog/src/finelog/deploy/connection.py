# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open a client against a deployed finelog, whichever way it is reachable.

A deployment is reached through the controller's IAP proxy when its config sets
``client_url``, and otherwise through an SSH (GCP) or ``kubectl port-forward``
(k8s) tunnel. Both arrive on the server's loopback, which its default-deny auth
stack admits, so the same call works for every backend.
"""

from collections.abc import Generator
from contextlib import contextmanager

from rigging.auth import IapLoginRequired
from rigging.connect import IapAuth, connect, disconnect
from rigging.credentials import iap_edge_provider
from rigging.tunnel import open_tunnel

from finelog.client.log_client import LogClient
from finelog.deploy.config import FinelogConfig, tunnel_target_for

# Sits just past the server's own 10s query deadline so a long query is ended by
# the server, which reports why, rather than by a client-side timeout that
# reports only that time ran out.
DEFAULT_REQUEST_TIMEOUT = 15.0


@contextmanager
def open_log_client(
    cfg: FinelogConfig,
    name: str,
    tunnel_timeout: float,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Generator[LogClient, None, None]:
    """Yield a LogClient for the deployment ``cfg`` describes."""
    timeout_ms = int(request_timeout * 1000)
    if cfg.client_url:
        provider = iap_edge_provider(name)
        if provider is None:
            raise IapLoginRequired(f"no cached IAP credentials for {name!r}; log in to {name!r} to refresh them")
        client = connect(
            cfg.client_url,
            lambda ep: LogClient.connect(ep.url, interceptors=ep.interceptors, timeout_ms=timeout_ms),
            auth=IapAuth(provider),
            connect_timeout=tunnel_timeout,
        )
        try:
            yield client
        finally:
            client.close()
            disconnect(client)
    else:
        target = tunnel_target_for(cfg)
        with open_tunnel(target, timeout=tunnel_timeout) as url:
            client = LogClient.connect(url, timeout_ms=timeout_ms)
            try:
                yield client
            finally:
                client.close()
