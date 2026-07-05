# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the client credentials for talking to a Marin cluster.

This is the consumer convention behind "one login": given a cluster's auth
shape, assemble the bearer material a client must attach. It never runs an
interactive flow — acquiring the IAP edge refresh token (the browser OAuth flow)
is the job of the login orchestration layer above; this module only locates
whatever that produced and hands back ready-to-attach interceptors.

The controller mints no user token, so the ``Authorization`` bearer carries
nothing for ordinary user / CLI traffic. A client may set ``$MARIN_CLUSTER_TOKEN``
to inject an explicit bearer (e.g. a worker JWT) for CI / headless runs.

IAP edge-token resolution (the ``Proxy-Authorization`` bearer, IAP clusters only)
is the sole per-request auth: the cached desktop-OAuth refresh token (the human
path) is preferred; failing that, an ambient service-account ID token (the
in-cluster / CI path) minted for a dedicated programmatic audience if one is
configured, else for the desktop client id (see :func:`_edge_provider`). The
desktop client that re-mints from a refresh token is the app's public identity
(:data:`~rigging.auth.MARIN_DESKTOP_OAUTH_CLIENT`), overridable per cluster.
"""

import os
from dataclasses import dataclass

from rigging.auth import (
    MARIN_DESKTOP_OAUTH_CLIENT,
    BearerTokenInjector,
    IapRefreshTokenProvider,
    IapServiceAccountTokenProvider,
    OAuthClient,
    StaticTokenProvider,
    TokenProvider,
)
from rigging.cluster_manifest import AuthProvider, ClusterAuth
from rigging.credential_store import load_credentials

MARIN_CLUSTER_TOKEN_ENV = "MARIN_CLUSTER_TOKEN"


@dataclass(frozen=True)
class ClientCredentials:
    """Bearer material for outgoing RPCs to a Marin service.

    Bundles the app-auth provider (attached on ``Authorization``) and, for an
    IAP-fronted cluster, the IAP edge provider (``Proxy-Authorization``). Passing
    both as one value keeps a call site from attaching one and forgetting the
    other — the failure where a command works on a tunneled cluster but is
    rejected by IAP because it never sent the edge token.
    """

    token_provider: TokenProvider | None = None
    iap_provider: TokenProvider | None = None

    def interceptors(self) -> tuple:
        """The client-side interceptor chain for these credentials.

        The app token rides in ``Authorization``; the IAP edge token in
        ``Proxy-Authorization`` so the app header stays free for the service's own
        JWT. Either may be absent (loopback trust sends neither).
        """
        chain: tuple = ()
        if self.token_provider is not None:
            chain += (BearerTokenInjector(self.token_provider, "authorization"),)
        if self.iap_provider is not None:
            chain += (BearerTokenInjector(self.iap_provider, "proxy-authorization"),)
        return chain


def _login_hint(cluster: str) -> str:
    """The canonical 'log in again' remedy for ``cluster``."""
    return f"log in to cluster {cluster!r} to authenticate"


def iap_edge_provider(
    cluster: str,
    *,
    desktop_client: OAuthClient = MARIN_DESKTOP_OAUTH_CLIENT,
) -> IapRefreshTokenProvider | None:
    """Build the IAP edge provider from ``cluster``'s cached desktop-OAuth login.

    Pairs the refresh token cached by the cluster login with ``cluster``'s
    desktop client to silently re-mint the OIDC ID token IAP requires. Returns
    None when the user has not logged in (so a pre-login command degrades to a
    clear UNAUTHENTICATED error rather than crashing on a missing credential).
    """
    record = load_credentials(cluster)
    if record is None or record.edge_refresh_token is None:
        return None
    return IapRefreshTokenProvider(
        desktop_client.client_id,
        desktop_client.client_secret,
        record.edge_refresh_token,
        login_hint=_login_hint(cluster),
    )


def _desktop_client(auth: ClusterAuth) -> OAuthClient:
    """The cluster's desktop OAuth client, falling back to the Marin app default."""
    iap = auth.iap
    if iap is not None and iap.desktop_oauth_client_id and iap.desktop_oauth_client_secret:
        return OAuthClient(iap.desktop_oauth_client_id, iap.desktop_oauth_client_secret)
    return MARIN_DESKTOP_OAUTH_CLIENT


def _edge_provider(cluster: str, auth: ClusterAuth) -> TokenProvider | None:
    """Resolve the IAP edge provider: cached human login, then ambient service account."""
    if auth.provider is not AuthProvider.IAP or auth.iap is None:
        return None
    human = iap_edge_provider(cluster, desktop_client=_desktop_client(auth))
    if human is not None:
        return human
    # No cached login (the CI / in-cluster path): mint an ambient service-account
    # ID token for the edge. Prefer a dedicated programmatic audience when the
    # cluster configures one; otherwise use the desktop client id, which IAP
    # registers as a programmatic client and admits for service-account tokens
    # too -- the same aud the human login path presents. The audience only clears
    # IAP's authentication step; the caller's identity is still checked against
    # the backend allowlist for authorization.
    audiences = auth.iap.programmatic_audiences
    audience = audiences[0] if audiences else _desktop_client(auth).client_id
    return IapServiceAccountTokenProvider(audience)


def credentials_for(
    cluster: str,
    auth: ClusterAuth,
    *,
    token_env: str = MARIN_CLUSTER_TOKEN_ENV,
) -> ClientCredentials:
    """Assemble the :class:`ClientCredentials` for ``cluster`` from the standard sources.

    ``auth`` is the cluster's resolved auth shape (provider + IAP params). The IAP
    edge provider is the sole per-request auth (``Proxy-Authorization``); the
    ``Authorization`` bearer is empty unless ``$MARIN_CLUSTER_TOKEN`` injects one.
    """
    override = os.environ.get(token_env)
    return ClientCredentials(
        token_provider=StaticTokenProvider(override) if override else None,
        iap_provider=_edge_provider(cluster, auth),
    )
