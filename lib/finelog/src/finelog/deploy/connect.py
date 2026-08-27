# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open a :class:`LogClient` against a named deployment.

Resolves the deployment's config and picks the transport it declares: the
Iris IAP proxy when the config sets ``client_url``, otherwise an SSH or
Kubernetes tunnel to the server's port. Callers name a deployment
(``marin``, ``cw-us-east-08a``) rather than assembling a URL.

Two IAP identities are supported. A ``client_url`` carrying an ``audience``
query parameter self-provisions the token from service-account credentials,
which is the unattended path used by CI. Without one, the token comes from
the desktop OAuth refresh token cached by ``iris --cluster <name> login``.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from rigging.auth import IapLoginRequired
from rigging.connect import Auth, IapAuth, NoAuth, connect, disconnect
from rigging.credentials import iap_edge_provider
from rigging.tunnel import open_tunnel

from finelog.client.log_client import LogClient
from finelog.deploy.config import FinelogConfig, load_finelog_config, tunnel_target_for

logger = logging.getLogger(__name__)

# Sits just past the server's own 10s query deadline so a long query is ended by
# the server, which reports why, rather than by a client-side timeout that
# reports only that time ran out.
DEFAULT_REQUEST_TIMEOUT = 15.0
DEFAULT_TUNNEL_TIMEOUT = 60.0


def _with_audience(client_url: str, audience: str | None) -> str:
    """Attach ``audience`` to an IAP transport URL that does not already carry one.

    An ``audience`` switches the scheme from "the caller supplies a desktop
    token" to "mint a service-account token for this IAP client id".
    """
    if not audience:
        return client_url
    parts = urlsplit(client_url)
    query = dict(parse_qsl(parts.query))
    if "audience" in query:
        return client_url
    query["audience"] = audience
    return urlunsplit(parts._replace(query=urlencode(query)))


def _has_audience(client_url: str) -> bool:
    """True when the transport URL self-provisions its own IAP token."""
    return "audience" in dict(parse_qsl(urlsplit(client_url).query))


@contextmanager
def open_client(
    cfg: FinelogConfig,
    name: str,
    *,
    tunnel_timeout: float = DEFAULT_TUNNEL_TIMEOUT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    iap_audience: str | None = None,
) -> Generator[LogClient, None, None]:
    """Yield a LogClient for ``cfg``, closing it and its transport on exit.

    ``iap_audience`` names the IAP client id to mint a service-account token
    for, which is how an unattended caller (CI, a cron) authenticates without
    the desktop OAuth refresh token.

    Raises:
        IapLoginRequired: the deployment is IAP-fronted, expects a desktop
            token, and no cached credentials exist for ``name``.
    """
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

    # An audience makes the scheme mint its own service-account token; the
    # desktop path has to supply the cached refresh token as edge auth.
    client_url = _with_audience(cfg.client_url, iap_audience)
    auth: Auth = NoAuth()
    if not _has_audience(client_url):
        provider = iap_edge_provider(name)
        if provider is None:
            raise IapLoginRequired(f"no cached IAP credentials for {name!r}; log in to {name!r} to refresh them")
        auth = IapAuth(provider)
    client = connect(
        client_url,
        lambda ep: LogClient.connect(ep.url, interceptors=ep.interceptors, timeout_ms=timeout_ms),
        auth=auth,
        connect_timeout=tunnel_timeout,
    )
    try:
        yield client
    finally:
        client.close()
        disconnect(client)


@contextmanager
def open_named_client(
    name: str,
    *,
    tunnel_timeout: float = DEFAULT_TUNNEL_TIMEOUT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    iap_audience: str | None = None,
) -> Generator[LogClient, None, None]:
    """Load the config for deployment ``name`` and yield a client for it."""
    with open_client(
        load_finelog_config(name),
        name,
        tunnel_timeout=tunnel_timeout,
        request_timeout=request_timeout,
        iap_audience=iap_audience,
    ) as client:
        yield client
